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
from src.fixed_image_features import extract_fixed_image_features
from src.fixed_text_features import extract_fixed_text_features
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


def build_selection_sample_meta(dataset, args):
    sample_meta = []
    for dataset_index in range(len(dataset)):
        meta = dataset.get_pair_metadata(dataset_index)
        sample_meta.append(
            {
                "sample_idx": int(meta["sample_idx"]),
                "img_id": int(meta["img_id"]),
                "dataset": args.dataset,
                "split": args.split,
                "caption": meta["caption"],
                "image": meta["image"],
                "raw_image_id": meta.get("raw_image_id"),
            }
        )
    return sample_meta


def build_selection_feature_cache(dataset, args):
    sample_meta = build_selection_sample_meta(dataset, args)
    image_paths = [os.path.join(args.image_root, item["image"]) for item in sample_meta]
    texts = [item["caption"] for item in sample_meta]

    img_features_selection, image_info = extract_fixed_image_features(
        image_paths,
        method=args.selection_image_repr_method,
        image_size=args.selection_image_size,
        hog_orientations=args.hog_orientations,
        hog_pixels_per_cell=args.hog_pixels_per_cell,
        hog_cells_per_block=args.hog_cells_per_block,
        color_space=args.color_space,
        color_hist_bins=args.color_hist_bins,
        raw_resize_size=args.selection_raw_resize_size,
        raw_pca_dim=args.selection_raw_pca_dim,
        batch_size=args.selection_image_batch_size,
        random_state=args.selection_random_state,
    )
    txt_features_selection = extract_fixed_text_features(
        texts,
        text_repr_method=args.selection_text_repr_method,
        batch_size=args.selection_text_batch_size,
        device=args.device,
        tfidf_ngram_max=args.tfidf_ngram_max,
        tfidf_stop_words=args.tfidf_stop_words,
        tfidf_max_features=args.tfidf_max_features,
        tfidf_min_df=args.tfidf_min_df,
        tfidf_svd_dim=args.tfidf_svd_dim,
        tfidf_random_state=args.selection_random_state,
    )

    img_features_selection = torch.tensor(img_features_selection, dtype=torch.float32)
    txt_features_selection = torch.tensor(txt_features_selection, dtype=torch.float32)
    return {
        "img_features_selection": img_features_selection,
        "txt_features_selection": txt_features_selection,
        "sample_meta": sample_meta,
        "num_samples": len(sample_meta),
        "image_info": image_info,
    }


def load_selection_features(feature_dir):
    feature_dir = Path(feature_dir)
    img_path = feature_dir / "img_features_selection.pt"
    txt_path = feature_dir / "txt_features_selection.pt"
    if not img_path.exists():
        img_path = feature_dir / "img_features.pt"
    if not txt_path.exists():
        txt_path = feature_dir / "txt_features.pt"

    meta_path = feature_dir / "sample_meta.json"
    info_path = feature_dir / "feature_info.json"

    img_features = torch.load(img_path, map_location="cpu")
    txt_features = torch.load(txt_path, map_location="cpu")
    with open(meta_path, "r", encoding="utf-8") as handle:
        sample_meta = json.load(handle)
    with open(info_path, "r", encoding="utf-8") as handle:
        feature_info = json.load(handle)
    return img_features, txt_features, sample_meta, feature_info


def move_images_to_device(images, device):
    if device.startswith("cuda"):
        return images.to(device, non_blocking=True)
    return images.to(device)


@torch.no_grad()
def extract_model_feature_cache(args):
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


@torch.no_grad()
def extract_feature_cache(args):
    dataset = load_train_dataset(args)
    if getattr(args, "selection_only_fixed_repr", True):
        return build_selection_feature_cache(dataset, args)
    return extract_model_feature_cache(args)


def save_feature_cache(cache, cache_dir, args):
    cache_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "selection_only_fixed_repr", True):
        img_tensor = cache["img_features_selection"]
        txt_tensor = cache["txt_features_selection"]
        img_path = cache_dir / "img_features_selection.pt"
        txt_path = cache_dir / "txt_features_selection.pt"
    else:
        img_tensor = cache["img_features"]
        txt_tensor = cache["txt_features"]
        img_path = cache_dir / "img_features.pt"
        txt_path = cache_dir / "txt_features.pt"
    meta_path = cache_dir / "sample_meta.json"
    info_path = cache_dir / "feature_info.json"

    torch.save(img_tensor, img_path)
    torch.save(txt_tensor, txt_path)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(cache["sample_meta"], handle, ensure_ascii=False, indent=2)

    info = {
        "dataset": args.dataset,
        "split": args.split,
        "selection_only_fixed_repr": bool(getattr(args, "selection_only_fixed_repr", True)),
        "image_encoder": args.image_encoder,
        "text_encoder": args.text_encoder,
        "num_samples": int(cache["num_samples"]),
        "img_feature_shape": list(img_tensor.shape),
        "txt_feature_shape": list(txt_tensor.shape),
        "img_feature_dtype": str(img_tensor.dtype),
        "txt_feature_dtype": str(txt_tensor.dtype),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if getattr(args, "selection_only_fixed_repr", True):
        info.update(
            {
                "selection_image_repr_method": args.selection_image_repr_method,
                "selection_text_repr_method": args.selection_text_repr_method,
                "selection_image_size": int(args.selection_image_size),
                "selection_raw_resize_size": int(args.selection_raw_resize_size),
                "selection_raw_pca_dim": int(args.selection_raw_pca_dim),
                "selection_image_batch_size": int(args.selection_image_batch_size),
                "selection_text_batch_size": int(args.selection_text_batch_size),
            }
        )
        if args.selection_text_repr_method == "tfidf":
            info.update(
                {
                    "tfidf_ngram_max": int(args.tfidf_ngram_max),
                    "tfidf_stop_words": args.tfidf_stop_words,
                    "tfidf_max_features": int(args.tfidf_max_features),
                    "tfidf_min_df": int(args.tfidf_min_df),
                    "tfidf_svd_dim": int(args.tfidf_svd_dim),
                }
            )
        if "image_info" in cache:
            info["selection_image_repr_info"] = cache["image_info"]
    else:
        info["resolved_image_encoder"] = resolve_image_encoder_name(args.image_encoder)
    with open(info_path, "w", encoding="utf-8") as handle:
        json.dump(info, handle, ensure_ascii=False, indent=2)

    return {
        "cache_dir": str(cache_dir),
        "img_features": str(img_path),
        "txt_features": str(txt_path),
        "sample_meta": str(meta_path),
        "feature_info": str(info_path),
    }


def cache_exists(cache_dir, selection_only_fixed_repr=True):
    required_fixed = [
        cache_dir / "img_features_selection.pt",
        cache_dir / "txt_features_selection.pt",
        cache_dir / "sample_meta.json",
        cache_dir / "feature_info.json",
    ]
    required_legacy = [
        cache_dir / "img_features.pt",
        cache_dir / "txt_features.pt",
        cache_dir / "sample_meta.json",
        cache_dir / "feature_info.json",
    ]
    if selection_only_fixed_repr:
        return all(path.exists() for path in required_fixed)
    return all(path.exists() for path in required_legacy)


def cache_output_paths(cache_dir):
    cache_dir = Path(cache_dir)
    img_selection = cache_dir / "img_features_selection.pt"
    txt_selection = cache_dir / "txt_features_selection.pt"
    if img_selection.exists() and txt_selection.exists():
        return {
            "cache_dir": str(cache_dir),
            "img_features_selection": str(img_selection),
            "txt_features_selection": str(txt_selection),
            "sample_meta": str(cache_dir / "sample_meta.json"),
            "feature_info": str(cache_dir / "feature_info.json"),
        }
    return {
        "cache_dir": str(cache_dir),
        "img_features": str(cache_dir / "img_features.pt"),
        "txt_features": str(cache_dir / "txt_features.pt"),
        "sample_meta": str(cache_dir / "sample_meta.json"),
        "feature_info": str(cache_dir / "feature_info.json"),
    }


def run_feature_cache(args):
    if args.split != "train":
        raise ValueError("First version only supports split='train'.")

    cache_dir = build_cache_dir(args)
    if cache_exists(cache_dir, selection_only_fixed_repr=getattr(args, "selection_only_fixed_repr", True)) and not args.overwrite:
        print(f"Cache already exists at {cache_dir}. Use --overwrite to regenerate.")
        return cache_output_paths(cache_dir)

    cache = extract_feature_cache(args)
    return save_feature_cache(cache, cache_dir, args)
