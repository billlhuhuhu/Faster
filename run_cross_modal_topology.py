import os
import argparse

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

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
    parser.add_argument("--alpha", type=float, default=1.0, help="Compatibility coefficient kept for cross-modal correction stage.")
    parser.add_argument("--correction_mode", type=str, default="bidirectional", choices=["directional", "bidirectional"])
    parser.add_argument("--tau_g", type=float, default=0.5, help="Temperature for turning modality collapse scores into global confidences.")
    parser.add_argument("--correction_eps", type=float, default=1e-8, help="Numerical stability epsilon used by bidirectional correction coefficients.")
    parser.add_argument("--enable_directional_correction_gate", action="store_true", default=True)
    parser.add_argument("--disable_directional_correction_gate", action="store_false", dest="enable_directional_correction_gate")
    parser.add_argument("--correction_gate_tau_high", type=float, default=0.6)
    parser.add_argument("--correction_gate_tau_low", type=float, default=0.3)
    parser.add_argument("--correction_gate_tau_gap", type=float, default=0.15)
    parser.add_argument("--enable_local_node_confidence", action="store_true")
    parser.add_argument("--local_node_confidence_mode", type=str, default="multi_view", choices=["none", "entropy", "multi_view"])
    parser.add_argument("--tau_l", type=float, default=0.25)
    parser.add_argument("--kappa_min", type=float, default=0.05)
    parser.add_argument("--local_conf_eps", type=float, default=1e-8)
    parser.add_argument("--local_conf_weight_entropy", type=float, default=1.0)
    parser.add_argument("--local_conf_weight_agreement", type=float, default=1.0)
    parser.add_argument("--local_conf_weight_diffusion", type=float, default=1.0)
    parser.add_argument("--local_conf_agreement_topk", type=int, default=15)
    parser.add_argument("--local_conf_agreement_type", type=str, default="jaccard", choices=["jaccard"])
    parser.add_argument("--local_conf_diffusion_hops", type=int, default=2)
    parser.add_argument("--local_conf_diffusion_type", type=str, default="p_vs_p2_cosine", choices=["p_vs_p2_cosine"])
    parser.add_argument("--fusion_mode", type=str, default="confidence_aware", choices=["intersection", "confidence_aware"])
    parser.add_argument("--correction_fusion_mode", type=str, default="thresholded_autonomy", choices=["legacy", "thresholded_autonomy"])
    parser.add_argument("--lambda_f", type=float, default=1.0, help="Exponent for confidence-aware weighted fusion.")
    parser.add_argument("--mu_f", type=float, default=1.0, help="Exponent for soft consistency gating in unified fusion.")
    parser.add_argument("--fusion_eps", type=float, default=1e-8, help="Numerical stability epsilon used by confidence-aware fusion.")
    parser.add_argument("--prefer_healthy_modality", type=str, default=None, choices=["image", "text"])
    parser.add_argument("--num_eigs", type=int, default=64)
    parser.add_argument("--spectral_embedding_dim", type=int, default=32)
    parser.add_argument("--spectrum_solver_mode", type=str, default="normalized_adjacency_largest", choices=["normalized_adjacency_largest", "laplacian_smallest"])
    parser.add_argument("--embedding_type", type=str, default="diffusion", choices=["laplacian", "diffusion"])
    parser.add_argument("--diffusion_dim", type=int, default=None)
    parser.add_argument("--diffusion_time", type=float, default=1.0)
    parser.add_argument("--diffusion_eig_solver", type=str, default="auto", choices=["auto", "dense", "sparse"])
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
