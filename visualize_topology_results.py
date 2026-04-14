import argparse

from src.topology_visualization import add_topology_visualization_args, visualize_cross_modal_topology_results


def build_parser():
    parser = argparse.ArgumentParser(description="Visualize saved cross-modal topology artifacts without re-running graph construction.")
    parser.add_argument("--result_dir", type=str, required=True, help="Cross-modal topology result directory containing summary.json and saved graph artifacts.")
    add_topology_visualization_args(parser)
    return parser


def main():
    args = build_parser().parse_args()
    outputs = visualize_cross_modal_topology_results(
        args.result_dir,
        visualization_output_dir=args.visualization_output_dir,
        visualization_topk_edges=args.visualization_topk_edges,
        visualization_node_order_mode=args.visualization_node_order_mode,
        visualization_layout_mode=args.visualization_layout_mode,
        visualization_num_local_cases=args.visualization_num_local_cases,
        visualization_local_case_topk=args.visualization_local_case_topk,
        visualization_max_heatmap_nodes=args.visualization_max_heatmap_nodes,
        visualization_show_labels_if_available=args.visualization_show_labels_if_available,
    )
    print("Topology visualization saved:")
    print(f"  output_dir: {outputs['output_dir']}")
    print(f"  summary_path: {outputs['summary_path']}")
    print(f"  local_case_dir: {outputs['local_case_dir']}")


if __name__ == "__main__":
    main()
