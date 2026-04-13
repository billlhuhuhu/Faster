import os
import argparse

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

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
    budget_group = parser.add_mutually_exclusive_group(required=True)
    budget_group.add_argument("--budget_ratio", type=float, default=None)
    budget_group.add_argument("--budget_size", type=int, default=None)
    parser.add_argument("--representation_mode", type=str, default="concat", choices=["concat"])
    parser.add_argument("--reference_embedding_mode", type=str, default="hybrid", choices=["concat", "spectral", "hybrid"])
    parser.add_argument("--spectral_weight", type=float, default=1.0)
    parser.add_argument("--selection_method", type=str, default="proxy_opt", choices=["baseline", "proxy_opt"])
    parser.add_argument("--cluster_method", type=str, default="kmeans", choices=["kmeans", "minibatch_kmeans"])
    parser.add_argument("--degree_weight", type=float, default=0.1, help="Tie-breaking weight from unified graph degree.")
    parser.add_argument("--geometry_weight", type=float, default=1.0)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--minibatch_size", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--proxy_projection_dim", type=int, default=128)
    parser.add_argument("--proxy_init_method", type=str, default="kmeans", choices=["kmeans", "minibatch_kmeans", "sample"])
    parser.add_argument(
        "--proxy_loss_type",
        type=str,
        default=None,
        choices=["wavelet_main", "pdcfd", "diffusion_mmd", "diffusion_swd", "diffusion_ms_swd", "legacy_pdcfd", "legacy_cfd", "legacy_diffusion_mmd", "legacy_diffusion_swd", "legacy_diffusion_ms_swd"],
        help="Primary proxy distribution matching loss. If omitted, defaults to wavelet_main unless a deprecated legacy objective flag is explicitly set.",
    )
    parser.add_argument("--proxy_objective_mode", type=str, default=None, choices=["cfd", "pd_cfd"], help="Deprecated legacy objective switch kept for backward compatibility.")
    parser.add_argument("--use_pdcfd", action="store_true")
    parser.add_argument("--proxy_num_frequencies", type=int, default=64, help="Deprecated: only used by legacy CFD / PD-CFD paths.")
    parser.add_argument("--proxy_frequency_scale", type=float, default=1.0, help="Deprecated: only used by legacy CFD / PD-CFD paths.")
    parser.add_argument("--proxy_lr", type=float, default=0.05)
    parser.add_argument("--proxy_num_steps", type=int, default=200)
    parser.add_argument("--proxy_reg_weight", type=float, default=0.01, help="Legacy init regularization weight. Retained for backward compatibility and folded into the grouped regularization block.")
    parser.add_argument("--proxy_target_batch_size", type=int, default=4096)
    parser.add_argument("--proxy_batch_size", type=int, default=4096)
    parser.add_argument("--mmd_kernel", type=str, default="rbf", choices=["rbf"])
    parser.add_argument("--mmd_bandwidth", type=float, default=None)
    parser.add_argument("--mmd_use_median_heuristic", action="store_true", default=True)
    parser.add_argument("--disable_mmd_median_heuristic", action="store_false", dest="mmd_use_median_heuristic")
    parser.add_argument("--swd_num_projections", type=int, default=64)
    parser.add_argument("--swd_p", type=float, default=2.0)
    parser.add_argument("--swd_projection_seed", type=int, default=None)
    parser.add_argument("--swd_use_fixed_projections", action="store_true")
    parser.add_argument("--use_wavelet_multiscale", action="store_true")
    parser.add_argument("--wavelet_scales", type=str, default="1,2,4")
    parser.add_argument("--wavelet_loss_weight", type=float, default=0.1, help="Legacy multiscale weight. Retained for backward compatibility; if --lambda_ms is unset it is reused as the multiscale block weight.")
    parser.add_argument("--wavelet_distance_type", type=str, default="swd", choices=["mmd", "swd"])
    parser.add_argument("--wavelet_swd_num_projections", type=int, default=None)
    parser.add_argument("--wavelet_swd_p", type=float, default=None)
    parser.add_argument("--wavelet_schedule", type=str, default="coarse_to_fine", choices=["coarse_to_fine", "all"])
    parser.add_argument("--lambda_main", type=float, default=1.0)
    parser.add_argument("--wavelet_main_scales", type=str, default="1,2,4")
    parser.add_argument("--wavelet_main_scale_weights", type=str, default=None)
    parser.add_argument("--wavelet_main_swd_num_projections", type=int, default=None)
    parser.add_argument("--wavelet_cov_weight", type=float, default=0.5)
    parser.add_argument("--wavelet_edge_weight", type=float, default=0.25)
    parser.add_argument("--wavelet_curriculum_schedule", type=str, default="coarse_to_fine", choices=["coarse_to_fine", "all"])
    parser.add_argument("--use_pdas", action="store_true", help="Deprecated: only used by legacy CFD / PD-CFD paths.")
    parser.add_argument("--pdas_num_stages", type=int, default=4, help="Deprecated: only used by legacy CFD / PD-CFD paths.")
    parser.add_argument("--pdas_schedule_mode", type=str, default="low_to_high", choices=["low_to_high", "uniform"], help="Deprecated: only used by legacy CFD / PD-CFD paths.")
    parser.add_argument("--num_freq_pool", type=int, default=256, help="Deprecated: only used by legacy CFD / PD-CFD paths.")
    parser.add_argument("--tau_min", type=float, default=0.1, help="Deprecated: only used by legacy CFD / PD-CFD paths.")
    parser.add_argument("--tau_max", type=float, default=1.0, help="Deprecated: only used by legacy CFD / PD-CFD paths.")
    parser.add_argument("--use_dpp", action="store_true")
    parser.add_argument("--lambda_div", type=float, default=0.01, help="Legacy diversity coefficient kept for backward compatibility and folded into the grouped regularization block.")
    parser.add_argument("--lambda_match", type=float, default=0.05, help="Legacy topology-matching coefficient kept for backward compatibility and folded into the grouped regularization block.")
    parser.add_argument("--lambda_graph", type=float, default=0.05, help="Legacy topology regularization coefficient kept for backward compatibility and folded into the grouped regularization block.")
    parser.add_argument("--lambda_phase", type=float, default=0.1, help="Deprecated: only used by legacy PD-CFD objective.")
    parser.add_argument("--diversity_sigma", type=float, default=1.0)
    parser.add_argument("--phase_weight_mode", type=str, default="uniform", choices=["uniform", "linear"], help="Deprecated: only used by legacy PD-CFD objective.")
    parser.add_argument("--lambda_diff", type=float, default=1.0, help="Weight for the global alignment block L_diff.")
    parser.add_argument("--lambda_ms", type=float, default=None, help="Deprecated for the new diffusion_ms_swd default path. Still used by compatibility modes where L_ms remains a separate top-level term.")
    parser.add_argument("--lambda_lsrc", type=float, default=None, help="Weight for the grouped LSRC block. If omitted, legacy lambda_lsrc_cov / lambda_lsrc_rel weighting is preserved.")
    parser.add_argument("--lsrc_mu", type=float, default=1.0, help="Relative weight mu inside L_cov_LSRC + mu * L_rel_LSRC.")
    parser.add_argument("--lambda_reg", type=float, default=1.0, help="Weight for the grouped regularization block L_reg.")
    parser.add_argument("--reg_alpha_div", type=float, default=1.0, help="Alpha in L_reg = alpha * L_div + beta * L_topo + gamma * L_init.")
    parser.add_argument("--reg_beta_topo", type=float, default=1.0, help="Beta in L_reg = alpha * L_div + beta * L_topo + gamma * L_init.")
    parser.add_argument("--reg_gamma_init", type=float, default=1.0, help="Gamma in L_reg = alpha * L_div + beta * L_topo + gamma * L_init.")
    parser.add_argument("--enable_lsrc", action="store_true")
    parser.add_argument("--keep_lsrc", action="store_true", default=True)
    parser.add_argument("--disable_lsrc", action="store_false", dest="keep_lsrc")
    parser.add_argument("--lsrc_k", type=int, default=32)
    parser.add_argument("--lsrc_tau_r", type=float, default=1.0)
    parser.add_argument("--lsrc_tau_c", type=float, default=1.0)
    parser.add_argument("--lsrc_eta", type=float, default=0.5)
    parser.add_argument("--lsrc_beta", type=float, default=0.5)
    parser.add_argument("--lambda_lsrc_cov", type=float, default=0.0, help="Deprecated legacy LSRC coverage weight. Kept for backward compatibility when --lambda_lsrc is unset.")
    parser.add_argument("--lambda_lsrc_rel", type=float, default=0.0, help="Deprecated legacy LSRC relation weight. Kept for backward compatibility when --lambda_lsrc is unset.")
    parser.add_argument("--lsrc_eps", type=float, default=1e-8)
    parser.add_argument("--lsrc_batch_size", type=int, default=4096)
    parser.add_argument("--lsrc_use_global_confidence", action="store_true")
    parser.add_argument("--lsrc_coverage_mode", type=str, default="mean", choices=["sum", "mean"])
    parser.add_argument("--lsrc_rel_loss_mode", type=str, default="weight_mean", choices=["edge_mean", "weight_mean"])
    parser.add_argument("--matching_top_k", type=int, default=64)
    parser.add_argument("--matching_candidate_batch_size", type=int, default=128)
    parser.add_argument("--matching_cost_mode", type=str, default="candidate_topk", choices=["candidate_topk", "degree_aware_global"])
    parser.add_argument("--topology_weight", type=float, default=0.5)
    parser.add_argument("--topology_hop_weight", type=float, default=0.5)
    parser.add_argument("--cost_alpha_diff", type=float, default=0.25)
    parser.add_argument("--cost_beta_wavelet", type=float, default=1.0)
    parser.add_argument("--matching_wavelet_weight", type=float, default=1.0)
    parser.add_argument("--cost_gamma_topo", type=float, default=0.1)
    parser.add_argument("--cost_eta_lsrc", type=float, default=0.1)
    return parser


def main():
    args = build_parser().parse_args()
    if args.use_pdcfd:
        args.proxy_loss_type = "pdcfd"
        args.proxy_objective_mode = "pd_cfd"
    elif args.proxy_loss_type is None:
        if args.proxy_objective_mode == "pd_cfd":
            args.proxy_loss_type = "pdcfd"
        elif args.proxy_objective_mode == "cfd":
            args.proxy_loss_type = "cfd"
        else:
            args.proxy_loss_type = "wavelet_main"
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
