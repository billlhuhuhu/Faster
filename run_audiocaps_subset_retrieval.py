import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def sanitize_name(name):
    return str(name).replace("\\", "-").replace("/", "-").replace(" ", "_")


def load_cache(feature_dir):
    feature_dir = Path(feature_dir)
    audio = torch.load(feature_dir / "img_features_selection.pt", map_location="cpu").float()
    text = torch.load(feature_dir / "txt_features_selection.pt", map_location="cpu").float()
    with open(feature_dir / "sample_meta.json", "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    return audio, text, meta


def load_selected_indices(path, num_samples=None):
    if path is None or str(path).strip() == "":
        if num_samples is None:
            raise ValueError("num_samples is required when selected_indices_path is empty.")
        return list(range(int(num_samples)))
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("selected_indices", payload.get("indices"))
    return [int(item) for item in payload]


class DualProjection(nn.Module):
    def __init__(self, audio_dim, text_dim, embed_dim):
        super().__init__()
        self.audio_proj = nn.Linear(int(audio_dim), int(embed_dim))
        self.text_proj = nn.Linear(int(text_dim), int(embed_dim))

    def encode_audio(self, audio):
        return nn.functional.normalize(self.audio_proj(audio), dim=-1)

    def encode_text(self, text):
        return nn.functional.normalize(self.text_proj(text), dim=-1)

    def forward(self, audio, text):
        return self.encode_audio(audio), self.encode_text(text)


def contrastive_loss(audio_emb, text_emb, temperature=0.07):
    logits = audio_emb @ text_emb.t() / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (nn.functional.cross_entropy(logits, labels) + nn.functional.cross_entropy(logits.t(), labels))


def build_relevance(meta):
    groups = {}
    for idx, item in enumerate(meta):
        key = str(item.get("audio_id") or item.get("audio") or idx)
        groups.setdefault(key, []).append(idx)
    relevant = []
    for item in meta:
        key = str(item.get("audio_id") or item.get("audio"))
        relevant.append(set(groups.get(key, [])))
    return relevant


def recall_at_k(similarity, relevant, ks=(1, 5, 10)):
    sim = np.asarray(similarity)
    order = np.argsort(-sim, axis=1)
    out = {}
    for k in ks:
        hits = 0
        kk = min(int(k), order.shape[1])
        for row_idx in range(order.shape[0]):
            if relevant[row_idx].intersection(order[row_idx, :kk].tolist()):
                hits += 1
        out[f"r{k}"] = 100.0 * hits / max(1, order.shape[0])
    return out


@torch.no_grad()
def encode_all(model, audio, text, batch_size, device):
    model.eval()
    audio_out = []
    text_out = []
    for start in range(0, audio.shape[0], int(batch_size)):
        a = audio[start : start + int(batch_size)].to(device)
        t = text[start : start + int(batch_size)].to(device)
        audio_out.append(model.encode_audio(a).cpu())
        text_out.append(model.encode_text(t).cpu())
    return torch.cat(audio_out, dim=0), torch.cat(text_out, dim=0)


def train_projection(args, train_audio, train_text, selected_indices):
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    selected = torch.tensor(selected_indices, dtype=torch.long)
    audio = train_audio[selected]
    text = train_text[selected]
    dataset = TensorDataset(audio, text)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, drop_last=False)
    model = DualProjection(train_audio.shape[1], train_text.shape[1], args.embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    history = []
    for epoch in range(int(args.epochs)):
        model.train()
        losses = []
        for batch_audio, batch_text in loader:
            if batch_audio.shape[0] < 2:
                continue
            batch_audio = batch_audio.to(device)
            batch_text = batch_text.to(device)
            audio_emb, text_emb = model(batch_audio, batch_text)
            loss = contrastive_loss(audio_emb, text_emb, temperature=args.temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses)) if losses else 0.0
        history.append({"epoch": epoch + 1, "train_loss": mean_loss})
        print(f"[AudioCaps retrieval] epoch={epoch + 1} train_loss={mean_loss:.6f}", flush=True)
    return model, history


def run(args):
    train_audio, train_text, train_meta = load_cache(args.train_feature_dir)
    eval_audio, eval_text, eval_meta = load_cache(args.eval_feature_dir)
    selected_indices = load_selected_indices(args.selected_indices_path, num_samples=train_audio.shape[0])
    model, history = train_projection(args, train_audio, train_text, selected_indices)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    audio_emb, text_emb = encode_all(model, eval_audio, eval_text, args.eval_batch_size, device)
    sim = (audio_emb @ text_emb.t()).numpy()
    relevant = build_relevance(eval_meta)
    a2t = recall_at_k(sim, relevant, ks=(1, 5, 10))
    t2a = recall_at_k(sim.T, relevant, ks=(1, 5, 10))
    metrics = {
        "dataset": "audiocaps",
        "selected_indices_path": args.selected_indices_path,
        "subset_mode": args.subset_mode,
        "train_feature_dir": args.train_feature_dir,
        "eval_feature_dir": args.eval_feature_dir,
        "subset_size": int(len(selected_indices)),
        "num_train_samples": int(train_audio.shape[0]),
        "num_eval_samples": int(eval_audio.shape[0]),
        "a2t_r1": a2t["r1"],
        "a2t_r5": a2t["r5"],
        "a2t_r10": a2t["r10"],
        "t2a_r1": t2a["r1"],
        "t2a_r5": t2a["r5"],
        "t2a_r10": t2a["r10"],
        "i2t_r1": a2t["r1"],
        "i2t_r5": a2t["r5"],
        "i2t_r10": a2t["r10"],
        "mean_recall": float(np.mean([a2t["r1"], a2t["r5"], a2t["r10"], t2a["r1"], t2a["r5"], t2a["r10"]])),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "history": history,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    torch.save(model.state_dict(), output_dir / "dual_projection.pt")
    print("AudioCaps retrieval finished:")
    print(f"  output_dir: {output_dir}")
    print(f"  mean_recall: {metrics['mean_recall']:.4f}")
    return metrics


def build_parser():
    parser = argparse.ArgumentParser(description="Train/evaluate lightweight AudioCaps audio-text retrieval on selected subsets.")
    parser.add_argument("--train_feature_dir", type=str, required=True)
    parser.add_argument("--eval_feature_dir", type=str, required=True)
    parser.add_argument("--selected_indices_path", type=str, default=None)
    parser.add_argument("--subset_mode", type=str, default="ours", choices=["ours", "random", "full"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
