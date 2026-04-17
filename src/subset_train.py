import copy
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import create_dataset
from data.subset_dataset import PairSubsetDataset, load_selected_indices
from src.epoch import epoch, epoch_test, itm_eval
from src.networks import CLIPModel_full


def sanitize_name(name):
    return str(name).replace("\\", "-").replace("/", "-").replace(" ", "_")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset_args(args):
    dataset_args = copy.deepcopy(args)
    dataset_args.return_sample_idx = True
    return dataset_args


def infer_subset_tag(selected_indices_path):
    selected_indices_path = Path(selected_indices_path)
    return sanitize_name(selected_indices_path.parent.name)


def build_budget_tag(subset_size, subset_ratio=None):
    if subset_ratio is not None:
        return f"ratio_{int(round(float(subset_ratio) * 100)):02d}"
    return f"size_{int(subset_size):04d}"


def build_output_dir(args, subset_size):
    model_tag = f"{sanitize_name(args.image_encoder)}_{sanitize_name(args.text_encoder)}"
    ratio_tag = build_budget_tag(subset_size, subset_ratio=getattr(args, "subset_ratio", None))
    subset_tag = sanitize_name(getattr(args, "subset_tag", None) or infer_subset_tag(args.selected_indices_path))
    seed_tag = f"seed_{int(args.seed)}"
    return Path(args.output_root) / args.dataset / model_tag / ratio_tag / subset_tag / seed_tag


def build_dataloaders(args):
    dataset_args = build_dataset_args(args)
    train_dataset, val_dataset, test_dataset = create_dataset(dataset_args)

    selected_indices = load_selected_indices(args.selected_indices_path)
    subset_dataset = PairSubsetDataset(
        train_dataset,
        selected_indices,
        return_sample_idx=True,
    )

    train_loader = DataLoader(
        subset_dataset,
        batch_size=args.batch_size_train,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size_test,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size_test,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_dataset, subset_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, selected_indices


@torch.no_grad()
def encode_texts(model, texts, device, batch_size=1024):
    model.eval()
    encoded = []
    for start in range(0, len(texts), int(batch_size)):
        batch_texts = texts[start:start + int(batch_size)]
        batch_embed = model.text_encoder(batch_texts, device=device)
        encoded.append(batch_embed.detach().cpu())
    return torch.cat(encoded, dim=0)


def convert_eval_result(raw_metrics):
    return {
        "i2t_r1": float(raw_metrics["txt_r1"]),
        "i2t_r5": float(raw_metrics["txt_r5"]),
        "i2t_r10": float(raw_metrics["txt_r10"]),
        "t2i_r1": float(raw_metrics["img_r1"]),
        "t2i_r5": float(raw_metrics["img_r5"]),
        "t2i_r10": float(raw_metrics["img_r10"]),
        "mean_recall": float(raw_metrics["r_mean"]),
    }


@torch.no_grad()
def evaluate_retrieval(model, dataloader, device, text_batch_size=1024):
    text_embed = encode_texts(model, dataloader.dataset.text, device=device, batch_size=text_batch_size)
    scores_i2t, scores_t2i = epoch_test(dataloader, model, device, text_embed)
    raw_metrics = itm_eval(scores_i2t, scores_t2i, dataloader.dataset.txt2img, dataloader.dataset.img2txt)
    return raw_metrics, convert_eval_result(raw_metrics)


def create_optimizer(model, args):
    return torch.optim.SGD(
        [
            {"params": model.image_encoder.parameters(), "lr": args.lr_teacher_img},
            {"params": model.text_projection.parameters(), "lr": args.lr_teacher_txt},
        ],
        lr=0,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def create_scheduler(optimizer, args):
    milestone = max(1, int(args.epochs) // 2 + 1)
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestone], gamma=args.lr_decay_gamma)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def append_log(path, message):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def build_metrics_payload(args, subset_size, best_epoch, best_val_metrics, test_metrics):
    payload = {
        "dataset": args.dataset,
        "backbone": args.image_encoder,
        "text_encoder": args.text_encoder,
        "diagnostic_experiment_id": None if getattr(args, "diagnostic_experiment_id", None) is None else int(getattr(args, "diagnostic_experiment_id")),
        "enable_stage2_correction": bool(getattr(args, "enable_stage2_correction", True)),
        "enable_stage3_fusion": bool(getattr(args, "enable_stage3_fusion", True)),
        "enable_stage4_lsrc": bool(getattr(args, "enable_stage4_lsrc", True)),
        "subset_ratio": float(args.subset_ratio) if getattr(args, "subset_ratio", None) is not None else None,
        "subset_size": int(subset_size),
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "i2t_r1": float(test_metrics["i2t_r1"]),
        "i2t_r5": float(test_metrics["i2t_r5"]),
        "i2t_r10": float(test_metrics["i2t_r10"]),
        "t2i_r1": float(test_metrics["t2i_r1"]),
        "t2i_r5": float(test_metrics["t2i_r5"]),
        "t2i_r10": float(test_metrics["t2i_r10"]),
        "mean_recall": float(test_metrics["mean_recall"]),
        "val_mean_recall": float(best_val_metrics["mean_recall"]),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return payload


def train_and_evaluate_subset(args):
    set_seed(args.seed)
    args.device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, subset_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, selected_indices = build_dataloaders(args)
    if getattr(args, "subset_size", None) is None:
        args.subset_size = int(len(subset_dataset))
    output_dir = build_output_dir(args, subset_size=len(subset_dataset))
    output_dir.mkdir(parents=True, exist_ok=True)

    model = CLIPModel_full(args).to(args.device)
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args)

    log_path = output_dir / "train.log"
    history_path = output_dir / "history.json"
    checkpoint_path = output_dir / "best_checkpoint.pt"
    metrics_path = output_dir / "metrics.json"

    history = []
    best_val_raw = None
    best_val_metrics = None
    best_state = None
    best_epoch = -1

    for ep in range(int(args.epochs)):
        loss_train, acc_train = epoch(ep, train_loader, model, optimizer, args)
        log_line = f"[Train] epoch={ep + 1} loss={loss_train:.6f} acc={acc_train:.4f}"
        print(log_line)
        append_log(log_path, log_line)

        should_eval = ((ep + 1) % int(args.eval_interval) == 0) or (ep + 1 == int(args.epochs))
        if should_eval:
            val_raw, val_metrics = evaluate_retrieval(
                model,
                val_loader,
                device=args.device,
                text_batch_size=args.text_batch_size,
            )
            history_item = {
                "epoch": int(ep + 1),
                "train_loss": float(loss_train),
                "train_acc": float(acc_train),
                "val_metrics": val_metrics,
            }
            history.append(history_item)

            eval_line = (
                "[Eval] epoch={epoch} "
                "i2t_r1={i2t_r1:.2f} i2t_r5={i2t_r5:.2f} i2t_r10={i2t_r10:.2f} "
                "t2i_r1={t2i_r1:.2f} t2i_r5={t2i_r5:.2f} t2i_r10={t2i_r10:.2f} "
                "mean_recall={mean_recall:.2f}"
            ).format(epoch=ep + 1, **val_metrics)
            print(eval_line)
            append_log(log_path, eval_line)

            if best_val_metrics is None or val_metrics["mean_recall"] > best_val_metrics["mean_recall"]:
                best_val_raw = val_raw
                best_val_metrics = val_metrics
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = ep + 1
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "epoch": int(best_epoch),
                        "dataset": args.dataset,
                        "image_encoder": args.image_encoder,
                        "text_encoder": args.text_encoder,
                        "subset_ratio": float(args.subset_ratio) if getattr(args, "subset_ratio", None) is not None else None,
                        "subset_size": int(len(subset_dataset)),
                        "selected_indices_path": str(args.selected_indices_path),
                        "selected_indices": selected_indices,
                        "val_metrics_raw": best_val_raw,
                        "val_metrics": best_val_metrics,
                    },
                    checkpoint_path,
                )

        scheduler.step()

    if best_state is None:
        raise RuntimeError("Training finished without any evaluation result.")

    model.load_state_dict(best_state)
    test_raw, test_metrics = evaluate_retrieval(
        model,
        test_loader,
        device=args.device,
        text_batch_size=args.text_batch_size,
    )
    metrics_payload = build_metrics_payload(
        args,
        subset_size=len(subset_dataset),
        best_epoch=best_epoch,
        best_val_metrics=best_val_metrics,
        test_metrics=test_metrics,
    )
    metrics_payload["val_metrics"] = best_val_metrics
    metrics_payload["test_metrics"] = test_metrics
    save_json(metrics_path, metrics_payload)
    save_json(history_path, history)

    final_line = (
        "[Test] best_epoch={best_epoch} "
        "i2t_r1={i2t_r1:.2f} i2t_r5={i2t_r5:.2f} i2t_r10={i2t_r10:.2f} "
        "t2i_r1={t2i_r1:.2f} t2i_r5={t2i_r5:.2f} t2i_r10={t2i_r10:.2f} "
        "mean_recall={mean_recall:.2f}"
    ).format(best_epoch=best_epoch, **test_metrics)
    print(final_line)
    append_log(log_path, final_line)

    return {
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "history_path": str(history_path),
        "log_path": str(log_path),
        "subset_size": len(subset_dataset),
        "metrics": metrics_payload,
    }
