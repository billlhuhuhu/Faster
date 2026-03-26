import os
import argparse

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import torch

from src.feature_cache import run_feature_cache


def build_parser():
    parser = argparse.ArgumentParser(description="Extract and cache selection-stage fixed image/text features for subset selection.")
    parser.add_argument("--dataset", type=str, required=True, choices=["flickr", "coco"])
    parser.add_argument("--split", type=str, default="train", choices=["train"])
    parser.add_argument("--image_encoder", type=str, required=True, choices=["nfnet", "resnet50", "resnet-50", "vit_b16", "vit-b16", "vit-b/16", "vit_base_patch16_224"])
    parser.add_argument("--text_encoder", type=str, default="bert", choices=["bert"])
    parser.add_argument("--selection_image_repr_method", type=str, default="hog_color", choices=["hog_color", "raw_pca"])
    parser.add_argument("--selection_text_repr_method", type=str, default="bert", choices=["bert"])
    parser.add_argument("--selection_image_size", type=int, default=128)
    parser.add_argument("--selection_raw_resize_size", type=int, default=32)
    parser.add_argument("--selection_raw_pca_dim", type=int, default=256)
    parser.add_argument("--selection_image_batch_size", type=int, default=512)
    parser.add_argument("--selection_text_batch_size", type=int, default=256)
    parser.add_argument("--selection_random_state", type=int, default=0)
    parser.add_argument("--hog_orientations", type=int, default=9)
    parser.add_argument("--hog_pixels_per_cell", type=int, default=8)
    parser.add_argument("--hog_cells_per_block", type=int, default=2)
    parser.add_argument("--color_hist_bins", type=int, default=16)
    parser.add_argument("--color_space", type=str, default="rgb", choices=["rgb", "hsv"])
    parser.add_argument("--disable_selection_only_fixed_repr", action="store_true")
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--ann_root", type=str, default="data/Flickr30k_ann")
    parser.add_argument("--cache_root", type=str, default="artifacts/feature_cache")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional local smoke-test cap.")
    parser.add_argument("--disable_sequential_sample_idx_check", action="store_true")
    return parser


def fill_default_paths(args):
    if args.image_root is None:
        args.image_root = {
            "flickr": "data/Flickr30k",
            "coco": "data/COCO",
        }[args.dataset]
    return args


def main():
    parser = build_parser()
    args = fill_default_paths(parser.parse_args())
    args.enforce_sequential_sample_idx = not args.disable_sequential_sample_idx_check
    args.selection_only_fixed_repr = not args.disable_selection_only_fixed_repr
    outputs = run_feature_cache(args)
    print("Feature cache saved:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
