import argparse
import os

import baselines.methods  # noqa: F401
from baselines.common.io import ensure_dir
from baselines.common.result_aggregation import export_all_tables
from baselines.registry import list_methods


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export unified baseline result tables.")
    parser.add_argument("--root", type=str, default="artifacts/baselines")
    parser.add_argument("--output_dir", type=str, default="artifacts/baselines")
    parser.add_argument("--budgets", nargs="*", type=int, default=[100, 200, 500])
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--dataset", type=str, default="flickr")
    parser.add_argument("--image_encoder", type=str, default="nfnet")
    parser.add_argument("--text_encoder", type=str, default="bert")
    parser.add_argument(
        "--method_mapping",
        type=str,
        default=os.path.join("baselines", "docs", "method_mapping.md"),
    )
    return parser


def main():
    args = build_parser().parse_args()
    ensure_dir(args.output_dir)
    methods = args.methods if args.methods else list_methods()
    outputs = export_all_tables(
        root=args.root,
        output_dir=args.output_dir,
        budgets=[int(x) for x in args.budgets] if args.budgets else None,
        methods=[str(x).lower() for x in methods],
        mapping_doc_path=args.method_mapping,
        expected_dataset=args.dataset,
        expected_image_encoder=args.image_encoder,
        expected_text_encoder=args.text_encoder,
    )
    print("Unified baseline tables exported:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()

