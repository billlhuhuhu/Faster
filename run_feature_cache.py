import argparse

import torch

from src.feature_cache import run_feature_cache


def build_parser():
    parser = argparse.ArgumentParser(description="Extract and cache full-train image/text features for subset selection.")
    parser.add_argument("--dataset", type=str, required=True, choices=["flickr", "coco"])
    parser.add_argument("--split", type=str, default="train", choices=["train"])
    parser.add_argument("--image_encoder", type=str, required=True, choices=["nfnet", "resnet50", "resnet-50", "vit_b16", "vit-b16", "vit-b/16", "vit_base_patch16_224"])
    parser.add_argument("--text_encoder", type=str, default="bert", choices=["bert"])
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
    outputs = run_feature_cache(args)
    print("Feature cache saved:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
