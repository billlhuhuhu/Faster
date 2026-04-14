from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SurrogateConfig:
    epochs: int = 5
    batch_size: int = 256
    proj_dim: int = 128
    lr: float = 1e-2
    temperature: float = 0.07
    seed: int = 0
    device: str = "cpu"


class PairProjectionModel(torch.nn.Module):
    def __init__(self, dim_img: int, dim_txt: int, proj_dim: int):
        super().__init__()
        self.img_proj = torch.nn.Linear(dim_img, proj_dim, bias=False)
        self.txt_proj = torch.nn.Linear(dim_txt, proj_dim, bias=False)

    def forward(self, img_x: torch.Tensor, txt_x: torch.Tensor):
        img_z = F.normalize(self.img_proj(img_x), dim=1)
        txt_z = F.normalize(self.txt_proj(txt_x), dim=1)
        return img_z, txt_z


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_surrogate_training(
    image_features: np.ndarray,
    text_features: np.ndarray,
    cfg: SurrogateConfig,
) -> Dict[str, object]:
    _seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    img = torch.tensor(image_features, dtype=torch.float32, device=device)
    txt = torch.tensor(text_features, dtype=torch.float32, device=device)
    n = img.shape[0]

    model = PairProjectionModel(img.shape[1], txt.shape[1], cfg.proj_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history_loss: List[np.ndarray] = []
    history_conf: List[np.ndarray] = []
    history_el2n: List[np.ndarray] = []
    history_grand: List[np.ndarray] = []

    for _ in range(int(cfg.epochs)):
        indices = torch.randperm(n, device=device)
        loss_epoch = torch.zeros(n, device=device)
        conf_epoch = torch.zeros(n, device=device)
        el2n_epoch = torch.zeros(n, device=device)
        grand_epoch = torch.zeros(n, device=device)

        for start in range(0, n, int(cfg.batch_size)):
            batch_ids = indices[start:start + int(cfg.batch_size)]
            img_b = img[batch_ids]
            txt_b = txt[batch_ids]
            bsz = img_b.shape[0]
            target = torch.arange(bsz, device=device)

            img_z, txt_z = model(img_b, txt_b)
            logits = (img_z @ txt_z.t()) / float(cfg.temperature)
            p_row = torch.softmax(logits, dim=1)
            p_col = torch.softmax(logits.t(), dim=1)

            loss_row = F.cross_entropy(logits, target, reduction="none")
            loss_col = F.cross_entropy(logits.t(), target, reduction="none")
            loss = (loss_row.mean() + loss_col.mean()) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            one_hot = F.one_hot(target, num_classes=bsz).float()
            el2n_row = torch.norm(p_row - one_hot, dim=1)
            el2n_col = torch.norm(p_col - one_hot, dim=1)
            conf = (p_row[torch.arange(bsz), target] + p_col[torch.arange(bsz), target]) / 2.0
            grand = torch.norm(p_row - one_hot, dim=1) + torch.norm(p_col - one_hot, dim=1)

            per_sample_loss = (loss_row + loss_col) / 2.0
            loss_epoch[batch_ids] = per_sample_loss.detach()
            conf_epoch[batch_ids] = conf.detach()
            el2n_epoch[batch_ids] = ((el2n_row + el2n_col) / 2.0).detach()
            grand_epoch[batch_ids] = grand.detach()

        history_loss.append(loss_epoch.detach().cpu().numpy().astype(np.float32))
        history_conf.append(conf_epoch.detach().cpu().numpy().astype(np.float32))
        history_el2n.append(el2n_epoch.detach().cpu().numpy().astype(np.float32))
        history_grand.append(grand_epoch.detach().cpu().numpy().astype(np.float32))

    model.eval()
    with torch.no_grad():
        img_z, txt_z = model(img, txt)

    return {
        "model": model,
        "img_embed": img_z.detach().cpu().numpy().astype(np.float32),
        "txt_embed": txt_z.detach().cpu().numpy().astype(np.float32),
        "history": {
            "loss": history_loss,
            "confidence": history_conf,
            "el2n": history_el2n,
            "grand": history_grand,
        },
    }


def compute_forgetting_counts(confidence_history: List[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    conf = np.stack(confidence_history, axis=0)
    learned = conf > float(threshold)
    forget = np.zeros(conf.shape[1], dtype=np.float32)
    ever_learned = np.zeros(conf.shape[1], dtype=bool)

    for epoch in range(conf.shape[0]):
        ever_learned = np.logical_or(ever_learned, learned[epoch])
        if epoch == 0:
            continue
        forgot_now = np.logical_and(ever_learned, np.logical_not(learned[epoch]))
        recovered_before = learned[epoch - 1]
        forget += np.logical_and(forgot_now, recovered_before).astype(np.float32)
    return forget


def build_sample_gradients(img_embed: np.ndarray, txt_embed: np.ndarray) -> np.ndarray:
    # Practical gradient surrogate on pair alignment objective.
    return np.concatenate([img_embed - txt_embed, txt_embed - img_embed], axis=1).astype(np.float32)


def split_train_val(n: int, val_ratio: float = 0.1, seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    val_size = max(1, int(round(n * float(val_ratio))))
    val_idx = np.sort(order[:val_size])
    train_idx = np.sort(order[val_size:])
    return {"train": train_idx, "val": val_idx}


def cosine_similarity_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_norm = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    y_norm = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-8)
    return x_norm @ y_norm.T

