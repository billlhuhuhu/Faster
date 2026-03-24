import argparse

from src.subset_match import run_subset_selection


def build_parser():
    parser = argparse.ArgumentParser(description="Select a real subset from unified representation/topology.")
    parser.add_argument("--dataset", type=str, required=True, choices=["flickr", "coco"])
    parser.add_argument("--split", type=str, default="train", choices=["train"])
    parser.add_argument("--image_encoder", type=str, required=True)
    parser.add_argument("--text_encoder", type=str, default="bert")
    parser.add_argument("--feature_cache_root", type=str, default="artifacts/feature_cache")
    parser.add_argument("--cross_modal_root", type=str, default="artifacts/cross_modal_topology")
    parser.add_argument("--output_root", type=str, default="artifacts/subset_selection")
    parser.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "cosine"])
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--budget_ratio", type=float, required=True, choices=[0.05, 0.1, 0.2])
    parser.add_argument("--representation_mode", type=str, default="concat", choices=["concat"])
    parser.add_argument("--selection_method", type=str, default="proxy_opt", choices=["baseline", "proxy_opt"])
    parser.add_argument("--cluster_method", type=str, default="kmeans", choices=["kmeans", "minibatch_kmeans"])
    parser.add_argument("--degree_weight", type=float, default=0.1, help="Tie-breaking weight from unified graph degree.")
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--minibatch_size", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--proxy_projection_dim", type=int, default=128)
    parser.add_argument("--proxy_init_method", type=str, default="kmeans", choices=["kmeans", "minibatch_kmeans", "sample"])
    parser.add_argument("--proxy_num_frequencies", type=int, default=64)
    parser.add_argument("--proxy_frequency_scale", type=float, default=1.0)
    parser.add_argument("--proxy_lr", type=float, default=0.05)
    parser.add_argument("--proxy_num_steps", type=int, default=200)
    parser.add_argument("--proxy_reg_weight", type=float, default=0.01)
    parser.add_argument("--proxy_target_batch_size", type=int, default=4096)
    parser.add_argument("--proxy_batch_size", type=int, default=4096)
    parser.add_argument("--matching_top_k", type=int, default=64)
    parser.add_argument("--matching_candidate_batch_size", type=int, default=128)
    parser.add_argument("--topology_weight", type=float, default=0.5)
    parser.add_argument("--topology_hop_weight", type=float, default=0.5)
    return parser


def main():
    args = build_parser().parse_args()
    outputs = run_subset_selection(args)
    print("Subset selection finished:")
    print(f"  output_dir: {outputs['output_dir']}")
    print(f"  subset_size: {outputs['subset_size']}")
    print(f"  selected_indices_path: {outputs['saved']['selected_indices']}")
    print(f"  selected_meta_path: {outputs['saved']['selected_meta']}")
    print(f"  summary_path: {outputs['saved']['summary']}")
    if "proxy_points" in outputs["saved"]:
        print(f"  proxy_points_path: {outputs['saved']['proxy_points']}")
    if "matching_cost" in outputs["saved"]:
        print(f"  matching_cost_path: {outputs['saved']['matching_cost']}")
    print(f"  first_selected_indices: {outputs['selected_indices'][:10]}")


if __name__ == "__main__":
    main()
