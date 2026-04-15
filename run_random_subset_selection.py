import argparse
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

from src.random_subset_selection import run_random_subset_selection


def build_parser():
    parser = argparse.ArgumentParser(description="Generate a random real subset from the train split.")
    parser.add_argument("--dataset", type=str, required=True, choices=["flickr", "coco"])
    parser.add_argument("--split", type=str, default="train", choices=["train"])
    parser.add_argument("--image_encoder", type=str, required=True)
    parser.add_argument("--text_encoder", type=str, default="bert")
    parser.add_argument("--feature_cache_root", type=str, default="artifacts/feature_cache")
    parser.add_argument("--output_root", type=str, default="artifacts/subset_selection_random")
    budget_group = parser.add_mutually_exclusive_group(required=True)
    budget_group.add_argument("--budget_ratio", type=float, default=None)
    budget_group.add_argument("--budget_size", type=int, default=None)
    parser.add_argument("--selection_method", type=str, default="random")
    parser.add_argument("--random_state", type=int, default=0)
    return parser


def main():
    args = build_parser().parse_args()
    outputs = run_random_subset_selection(args)
    print("Random subset selection finished:")
    print(f"  output_dir: {outputs['output_dir']}")
    print(f"  selected_indices_path: {outputs['selected_indices_path']}")
    print(f"  summary_path: {outputs['summary_path']}")
    print(f"  subset_size: {outputs['subset_size']}")


if __name__ == "__main__":
    main()
