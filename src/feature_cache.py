import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import create_dataset
from data.subset_dataset import PairSubsetDataset
from src.networks import ImageEncoder, TextEncoder


IMAGE_ENCODER_ALIASES = {
    "nfnet": "nfnet",
    "nfnet_l0": "nfnet",
    "resnet50": "resnet50",
    "resnet-50": "resnet50",
    "vit": "vit",
    "vit_b16": "vit_base_patch16_224",
    "vit-b16": "vit_base_patch16_224",
    "vit_base_patch16_224": "vit_base_patch16_224",
    "vit-b/16": "vit_base_patch16_224",
}


def sanitize_name(name):
    return name.replace("\\", "-").replace("/", "-").replace(" ", "_")


def resolve_image_encoder_name(image_encoder):
    key = image_encoder.lower()
    if key not in IMAGE_ENCODER_ALIASES:
        raise ValueError(f"Unsupported image encoder alias: {image_encoder}")
    return IMAGE_ENCODER_ALIASES[key]


def make_dataset_args(args):
    return SimpleNamespace(
        dataset=args.dataset,
        image_root=args.image_root,
        ann_root=args.ann_root,
        image_size=args.image_size,
        no_aug=True,
        return_sample_idx=True,
    )


def load_train_dataset(args):
    dataset_args = make_dataset_args(args)
    train_dataset, _, _ = create_dataset(dataset_args)
    if args.max_samples is not None:
        selected_indices = list(range(min(args.max_samples, len(train_dataset))))
        train_dataset = PairSubsetDataset(train_dataset, selected_indices, return_sample_idx=True)
    return train_dataset


def make_model_args(args):
    return SimpleNamespace(
        image_encoder=resolve_image_encoder_name(args.image_encoder),
        image_pretrained=True,
        image_trainable=False,
        text_encoder=args.text_encoder,
        text_pretrained=True,
        text_trainable=False,
    )


def build_cache_dir(args):
    model_tag = f"{sanitize_name(args.image_encoder)}_{sanitize_name(args.text_encoder)}"
    return Path(args.cache_root) / args.dataset / args.split / model_tag


def collate_meta(dataset, sample_indices):
    return [dataset.get_pair_metadata(sample_idx) for sample_idx in sample_indices]


def move_images_to_device(images, device):
    if device.startswith("cuda"):
        return images.to(device, non_blocking=True)
    return images.to(device)


@torch.no_grad()
def extract_feature_cache(args):
    dataset = load_train_dataset(args)
    model_args = make_model_args(args)

    image_encoder = ImageEncoder(model_args).to(args.device)
    text_encoder = TextEncoder(model_args).to(args.device)
    image_encoder.eval()
    text_encoder.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )

    total_samples = len(dataset)
    img_features = None
    txt_features = None
    sample_meta = []
    write_offset = 0

    expected_sample_idx = 0
    for batch in tqdm(dataloader, desc="Extracting features"):
        images, captions, sample_indices, img_ids = batch
        captions = list(captions)
        sample_indices_list = [int(x) for x in sample_indices.tolist()]

        if args.enforce_sequential_sample_idx:
            expected = list(range(expected_sample_idx, expected_sample_idx + len(sample_indices_list)))
            if sample_indices_list != expected:
                raise AssertionError(
                    f"sample_idx order mismatch: expected {expected[:3]}..., got {sample_indices_list[:3]}..."
                )
            expected_sample_idx += len(sample_indices_list)

        image_batch = move_images_to_device(images, args.device)
        image_batch_features = image_encoder(image_batch).detach().cpu().float()
        text_batch_features = text_encoder(captions, device=args.device).detach().cpu().float()

        batch_size = image_batch_features.shape[0]
        if img_features is None:
            img_features = torch.empty((total_samples, image_batch_features.shape[1]), dtype=torch.float32)
        if txt_features is None:
            txt_features = torch.empty((total_samples, text_batch_features.shape[1]), dtype=torch.float32)

        img_features[write_offset:write_offset + batch_size] = image_batch_features
        txt_features[write_offset:write_offset + batch_size] = text_batch_features
        write_offset += batch_size

        batch_meta = collate_meta(dataset, sample_indices_list)
        for meta, img_id, caption in zip(batch_meta, img_ids.tolist(), captions):
            sample_meta.append(
                {
                    "sample_idx": int(meta["sample_idx"]),
                    "img_id": int(img_id),
                    "dataset": args.dataset,
                    "split": args.split,
                    "caption": caption,
                    "image": meta["image"],
                    "raw_image_id": meta.get("raw_image_id"),
                }
            )

    if img_features is None or txt_features is None:
        raise RuntimeError("No features were extracted from the dataset.")
    if write_offset != total_samples:
        raise RuntimeError(f"Feature write size mismatch: wrote {write_offset}, expected {total_samples}.")

    return {
        "img_features": img_features,
        "txt_features": txt_features,
        "sample_meta": sample_meta,
        "num_samples": len(sample_meta),
    }


def save_feature_cache(cache, cache_dir, args):
    cache_dir.mkdir(parents=True, exist_ok=True)

    img_path = cache_dir / "img_features.pt"
    txt_path = cache_dir / "txt_features.pt"
    meta_path = cache_dir / "sample_meta.json"
    info_path = cache_dir / "feature_info.json"

    torch.save(cache["img_features"], img_path)
    torch.save(cache["txt_features"], txt_path)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(cache["sample_meta"], handle, ensure_ascii=False, indent=2)

    info = {
        "dataset": args.dataset,
        "split": args.split,
        "image_encoder": args.image_encoder,
        "resolved_image_encoder": resolve_image_encoder_name(args.image_encoder),
        "text_encoder": args.text_encoder,
        "num_samples": int(cache["num_samples"]),
        "img_feature_shape": list(cache["img_features"].shape),
        "txt_feature_shape": list(cache["txt_features"].shape),
        "img_feature_dtype": str(cache["img_features"].dtype),
        "txt_feature_dtype": str(cache["txt_features"].dtype),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(info_path, "w", encoding="utf-8") as handle:
        json.dump(info, handle, ensure_ascii=False, indent=2)

    return {
        "cache_dir": str(cache_dir),
        "img_features": str(img_path),
        "txt_features": str(txt_path),
        "sample_meta": str(meta_path),
        "feature_info": str(info_path),
    }


def cache_exists(cache_dir):
    required = [
        cache_dir / "img_features.pt",
        cache_dir / "txt_features.pt",
        cache_dir / "sample_meta.json",
        cache_dir / "feature_info.json",
    ]
    return all(path.exists() for path in required)


def run_feature_cache(args):
    if args.split != "train":
        raise ValueError("First version only supports split='train'.")

    cache_dir = build_cache_dir(args)
    if cache_exists(cache_dir) and not args.overwrite:
        print(f"Cache already exists at {cache_dir}. Use --overwrite to regenerate.")
        return {
            "cache_dir": str(cache_dir),
            "img_features": str(cache_dir / "img_features.pt"),
            "txt_features": str(cache_dir / "txt_features.pt"),
            "sample_meta": str(cache_dir / "sample_meta.json"),
            "feature_info": str(cache_dir / "feature_info.json"),
        }

    cache = extract_feature_cache(args)
    return save_feature_cache(cache, cache_dir, args)
