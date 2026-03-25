import argparse

from src.cross_modal_topology import run_cross_modal_topology


def build_parser():
    parser = argparse.ArgumentParser(description="Correct collapsed modality topology and reconstruct a unified cross-modal topology.")
    parser.add_argument("--dataset", type=str, required=True, choices=["flickr", "coco"])
    parser.add_argument("--split", type=str, default="train", choices=["train"])
    parser.add_argument("--image_encoder", type=str, required=True)
    parser.add_argument("--text_encoder", type=str, default="bert")
    parser.add_argument("--topology_root", type=str, default="artifacts/topology_graph")
    parser.add_argument("--output_root", type=str, default="artifacts/cross_modal_topology")
    parser.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "cosine"])
    parser.add_argument("--image_metric", type=str, default=None, choices=["euclidean", "cosine"])
    parser.add_argument("--text_metric", type=str, default=None, choices=["euclidean", "cosine"])
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--multi_scale_ks", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=1.0, help="Strength of healthy-modality correction.")
    parser.add_argument("--fusion_mode", type=str, default="intersection", choices=["intersection"])
    parser.add_argument("--prefer_healthy_modality", type=str, default=None, choices=["image", "text"])
    parser.add_argument("--num_eigs", type=int, default=64)
    parser.add_argument("--spectral_embedding_dim", type=int, default=32)
    parser.add_argument("--spectrum_solver_mode", type=str, default="normalized_adjacency_largest", choices=["normalized_adjacency_largest", "laplacian_smallest"])
    parser.add_argument("--save_eigenvectors", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    outputs = run_cross_modal_topology(args)
    print("Cross-modal topology saved:")
    print(f"  output_dir: {outputs['output_dir']}")
    print(f"  summary_path: {outputs['summary_path']}")
    summary = outputs["summary"]
    print(f"  healthy_modality: {summary['healthy_modality']}")
    print(f"  corrected_summary: {summary['corrected_summary']}")
    print(f"  unified_summary: {summary['unified_summary']}")


if __name__ == "__main__":
    main()
