import csv
import json
import platform
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from src.cross_modal_topology import run_cross_modal_topology
from src.energy_meter import EnergyMeter
from src.runtime_meter import RuntimeMeter
from src.subset_match import run_subset_selection


try:
    import resource
except ImportError:  # pragma: no cover
    resource = None


def benchmark_log(message):
    print(f"[selection-benchmark][{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def sanitize_name(name):
    return str(name).replace("\\", "-").replace("/", "-").replace(" ", "_")


def build_budget_tag(args):
    if args.budget_size is not None:
        return f"size_{int(args.budget_size):04d}"
    return f"ratio_{int(round(float(args.budget_ratio) * 100)):02d}"


def infer_variant_name(args):
    if args.variant_name:
        return args.variant_name
    parts = [args.selection_method]
    if getattr(args, "enable_local_node_confidence", False):
        parts.append("localconf")
    if getattr(args, "correction_mode", None) == "bidirectional":
        parts.append("bidir")
    if getattr(args, "fusion_mode", None):
        parts.append(str(args.fusion_mode))
    if getattr(args, "enable_lsrc", False):
        parts.append("lsrc")
    return "_".join(parts)


def read_json_if_exists(path):
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_process_peak_memory_mb():
    if resource is None:
        return None
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None
    if peak <= 0:
        return None
    if platform.system().lower() == "darwin":
        return float(peak) / (1024.0 * 1024.0)
    return float(peak) / 1024.0


class PhaseBenchmark:
    def __init__(self, phase_name="selection", enable_benchmark=False, energy_backend="auto", poll_interval_ms=200):
        self.phase_name = phase_name
        self.enable_benchmark = bool(enable_benchmark)
        self.runtime_meter = RuntimeMeter()
        self.energy_meter = EnergyMeter(backend=energy_backend, poll_interval_ms=poll_interval_ms)
        self.time_s = None
        self.cpu_energy_wh = None
        self.gpu_energy_wh = None
        self.total_energy_wh = None
        self.peak_memory_mb = None
        self.peak_gpu_memory_mb = None

    def __enter__(self):
        if self.enable_benchmark:
            if torch.cuda.is_available():
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
            self.runtime_meter.start()
            self.energy_meter.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enable_benchmark:
            self.energy_meter.stop()
            self.runtime_meter.stop()
            self.time_s = float(self.runtime_meter.elapsed_seconds or 0.0)
            self.cpu_energy_wh = self.energy_meter.cpu_energy_wh
            self.gpu_energy_wh = self.energy_meter.gpu_energy_wh
            self.total_energy_wh = float(self.energy_meter.total_energy_wh)
            self.peak_memory_mb = get_process_peak_memory_mb()
            if torch.cuda.is_available():
                try:
                    self.peak_gpu_memory_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
                except Exception:
                    self.peak_gpu_memory_mb = None
        return False

    def to_dict(self):
        payload = {
            "phase_name": self.phase_name,
            "time_s": None if self.time_s is None else float(self.time_s),
            "cpu_energy_wh": None if self.cpu_energy_wh is None else float(self.cpu_energy_wh),
            "gpu_energy_wh": None if self.gpu_energy_wh is None else float(self.gpu_energy_wh),
            "total_energy_wh": None if self.total_energy_wh is None else float(self.total_energy_wh),
            "peak_memory_mb": None if self.peak_memory_mb is None else float(self.peak_memory_mb),
            "peak_gpu_memory_mb": None if self.peak_gpu_memory_mb is None else float(self.peak_gpu_memory_mb),
        }
        payload.update(self.energy_meter.to_dict())
        return payload


def build_cross_args(args):
    return SimpleNamespace(
        dataset=args.dataset,
        split=args.split,
        image_encoder=args.image_encoder,
        text_encoder=args.text_encoder,
        topology_root=args.topology_root,
        output_root=args.cross_output_root,
        metric=args.metric,
        image_metric=args.image_metric,
        text_metric=args.text_metric,
        k=args.k,
        multi_scale_ks=args.multi_scale_ks,
        alpha=args.alpha,
        correction_mode=args.correction_mode,
        tau_g=args.tau_g,
        correction_eps=args.correction_eps,
        enable_local_node_confidence=args.enable_local_node_confidence,
        tau_l=args.tau_l,
        kappa_min=args.kappa_min,
        local_conf_eps=args.local_conf_eps,
        fusion_mode=args.fusion_mode,
        lambda_f=args.lambda_f,
        mu_f=args.mu_f,
        fusion_eps=args.fusion_eps,
        prefer_healthy_modality=args.prefer_healthy_modality,
        num_eigs=args.num_eigs,
        spectral_embedding_dim=args.spectral_embedding_dim,
        spectrum_solver_mode=args.spectrum_solver_mode,
        save_eigenvectors=args.save_eigenvectors,
    )


def build_selection_args(args):
    return SimpleNamespace(
        dataset=args.dataset,
        split=args.split,
        image_encoder=args.image_encoder,
        text_encoder=args.text_encoder,
        feature_cache_root=args.feature_cache_root,
        cross_modal_root=args.cross_output_root,
        output_root=args.selection_output_root,
        metric=args.metric,
        k=args.k,
        alpha=args.alpha,
        budget_ratio=args.budget_ratio,
        budget_size=args.budget_size,
        representation_mode=args.representation_mode,
        reference_embedding_mode=args.reference_embedding_mode,
        spectral_weight=args.spectral_weight,
        selection_method=args.selection_method,
        cluster_method=args.cluster_method,
        degree_weight=args.degree_weight,
        geometry_weight=args.geometry_weight,
        random_state=args.random_state,
        minibatch_size=args.minibatch_size,
        device=args.device,
        proxy_projection_dim=args.proxy_projection_dim,
        proxy_init_method=args.proxy_init_method,
        proxy_loss_type=args.proxy_loss_type,
        proxy_objective_mode=args.proxy_objective_mode,
        use_pdcfd=args.use_pdcfd,
        proxy_num_frequencies=args.proxy_num_frequencies,
        proxy_frequency_scale=args.proxy_frequency_scale,
        proxy_lr=args.proxy_lr,
        proxy_num_steps=args.proxy_num_steps,
        proxy_reg_weight=args.proxy_reg_weight,
        proxy_target_batch_size=args.proxy_target_batch_size,
        proxy_batch_size=args.proxy_batch_size,
        mmd_kernel=args.mmd_kernel,
        mmd_bandwidth=args.mmd_bandwidth,
        mmd_use_median_heuristic=args.mmd_use_median_heuristic,
        swd_num_projections=args.swd_num_projections,
        swd_p=args.swd_p,
        swd_projection_seed=args.swd_projection_seed,
        swd_use_fixed_projections=args.swd_use_fixed_projections,
        use_wavelet_multiscale=args.use_wavelet_multiscale,
        wavelet_scales=args.wavelet_scales,
        wavelet_loss_weight=args.wavelet_loss_weight,
        wavelet_distance_type=args.wavelet_distance_type,
        wavelet_schedule=args.wavelet_schedule,
        wavelet_swd_num_projections=args.wavelet_swd_num_projections,
        wavelet_swd_p=args.wavelet_swd_p,
        lambda_diff=args.lambda_diff,
        lambda_ms=args.lambda_ms,
        lambda_lsrc=args.lambda_lsrc,
        lsrc_mu=args.lsrc_mu,
        lambda_reg=args.lambda_reg,
        reg_alpha_div=args.reg_alpha_div,
        reg_beta_topo=args.reg_beta_topo,
        reg_gamma_init=args.reg_gamma_init,
        use_pdas=args.use_pdas,
        pdas_num_stages=args.pdas_num_stages,
        pdas_schedule_mode=args.pdas_schedule_mode,
        num_freq_pool=args.num_freq_pool,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        use_dpp=args.use_dpp,
        lambda_div=args.lambda_div,
        lambda_match=args.lambda_match,
        lambda_graph=args.lambda_graph,
        lambda_phase=args.lambda_phase,
        diversity_sigma=args.diversity_sigma,
        phase_weight_mode=args.phase_weight_mode,
        enable_lsrc=args.enable_lsrc,
        lsrc_k=args.lsrc_k,
        lsrc_tau_r=args.lsrc_tau_r,
        lsrc_tau_c=args.lsrc_tau_c,
        lsrc_eta=args.lsrc_eta,
        lsrc_beta=args.lsrc_beta,
        lambda_lsrc_cov=args.lambda_lsrc_cov,
        lambda_lsrc_rel=args.lambda_lsrc_rel,
        lsrc_eps=args.lsrc_eps,
        lsrc_batch_size=args.lsrc_batch_size,
        lsrc_use_global_confidence=args.lsrc_use_global_confidence,
        lsrc_coverage_mode=args.lsrc_coverage_mode,
        lsrc_rel_loss_mode=args.lsrc_rel_loss_mode,
        matching_top_k=args.matching_top_k,
        matching_candidate_batch_size=args.matching_candidate_batch_size,
        matching_cost_mode=args.matching_cost_mode,
        topology_weight=args.topology_weight,
        topology_hop_weight=args.topology_hop_weight,
        cost_alpha_diff=args.cost_alpha_diff,
        cost_beta_wavelet=args.cost_beta_wavelet,
        cost_gamma_topo=args.cost_gamma_topo,
        cost_eta_lsrc=args.cost_eta_lsrc,
    )


def infer_retrieval_metrics_path(args, variant_name):
    if args.retrieval_metrics_path:
        path = Path(args.retrieval_metrics_path)
        return path if path.exists() else None
    if not args.subset_train_root:
        return None
    model_tag = f"{sanitize_name(args.image_encoder)}_{sanitize_name(args.text_encoder)}"
    budget_tag = build_budget_tag(args)
    seed_tag = f"seed_{int(args.random_state)}"
    candidate = Path(args.subset_train_root) / args.dataset / model_tag / budget_tag / variant_name / seed_tag / "metrics.json"
    if candidate.exists():
        return candidate
    return None


def load_retrieval_metrics(args, variant_name):
    metrics_path = infer_retrieval_metrics_path(args, variant_name)
    if metrics_path is None:
        return None, None
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    payload = {
        "i2t_r1": metrics.get("i2t_r1"),
        "i2t_r5": metrics.get("i2t_r5"),
        "i2t_r10": metrics.get("i2t_r10"),
        "t2i_r1": metrics.get("t2i_r1"),
        "t2i_r5": metrics.get("t2i_r5"),
        "t2i_r10": metrics.get("t2i_r10"),
        "mean_recall": metrics.get("mean_recall"),
    }
    return payload, str(metrics_path)


def compute_derived_metrics(summary, baseline_summary=None, eps=1e-8):
    mean_recall = summary.get("mean_recall")
    selection_time_s = summary.get("selection_time_s")
    selection_total_energy_wh = summary.get("selection_total_energy_wh")

    if mean_recall is not None and selection_total_energy_wh is not None:
        summary["mean_recall_per_wh"] = float(mean_recall) / max(float(selection_total_energy_wh), float(eps))
    else:
        summary["mean_recall_per_wh"] = None

    if mean_recall is not None and selection_time_s is not None:
        summary["mean_recall_per_second"] = float(mean_recall) / max(float(selection_time_s), float(eps))
    else:
        summary["mean_recall_per_second"] = None

    if baseline_summary is not None:
        baseline_time = baseline_summary.get("selection_time_s")
        baseline_energy = baseline_summary.get("selection_total_energy_wh")
        summary["baseline_selection_time_s"] = baseline_time
        summary["baseline_selection_total_energy_wh"] = baseline_energy
        if baseline_time is not None and selection_time_s not in (None, 0):
            summary["speedup_vs_baseline"] = float(baseline_time) / max(float(selection_time_s), float(eps))
        else:
            summary["speedup_vs_baseline"] = None
        if baseline_energy is not None and selection_total_energy_wh is not None:
            summary["energy_reduction_vs_baseline"] = 1.0 - float(selection_total_energy_wh) / max(float(baseline_energy), float(eps))
        else:
            summary["energy_reduction_vs_baseline"] = None
    else:
        summary["baseline_selection_time_s"] = None
        summary["baseline_selection_total_energy_wh"] = None
        summary["speedup_vs_baseline"] = None
        summary["energy_reduction_vs_baseline"] = None
    return summary


def build_summary(args, phase_metrics, cross_outputs, selection_outputs, retrieval_metrics=None, retrieval_metrics_path=None):
    cross_summary = cross_outputs["summary"]
    selection_summary = read_json_if_exists(selection_outputs["saved"]["summary"]) or {}
    variant_name = infer_variant_name(args)

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "budget_ratio": None if args.budget_ratio is None else float(args.budget_ratio),
        "budget_size": None if args.budget_size is None else int(args.budget_size),
        "variant_name": variant_name,
        "correction_mode": args.correction_mode,
        "fusion_mode": args.fusion_mode,
        "enable_lsrc": bool(args.enable_lsrc),
        "enable_local_node_confidence": bool(args.enable_local_node_confidence),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hostname": socket.gethostname(),
        "device": args.device,
        "benchmark_enabled": bool(args.enable_selection_efficiency_benchmark),
        "selection_time_s": phase_metrics.get("time_s"),
        "selection_cpu_energy_wh": phase_metrics.get("cpu_energy_wh"),
        "selection_gpu_energy_wh": phase_metrics.get("gpu_energy_wh"),
        "selection_total_energy_wh": phase_metrics.get("total_energy_wh"),
        "selection_peak_memory_mb": phase_metrics.get("peak_memory_mb"),
        "selection_peak_gpu_memory_mb": phase_metrics.get("peak_gpu_memory_mb"),
        "cross_output_dir": cross_outputs["output_dir"],
        "cross_summary_path": cross_outputs["summary_path"],
        "selection_output_dir": selection_outputs["output_dir"],
        "selection_summary_path": selection_outputs["saved"]["summary"],
        "selected_indices_path": selection_outputs["saved"]["selected_indices"],
        "retrieval_metrics_path": retrieval_metrics_path,
        "energy_backend": phase_metrics.get("backend"),
        "cpu_energy_backend": phase_metrics.get("cpu_backend"),
        "gpu_energy_backend": phase_metrics.get("gpu_backend"),
        "healthy_modality": cross_summary.get("healthy_modality"),
        "selection_method": selection_summary.get("selection_method", args.selection_method),
        "selection_seed": int(args.random_state),
    }

    if retrieval_metrics is not None:
        summary.update(retrieval_metrics)
    else:
        summary.update(
            {
                "i2t_r1": None,
                "i2t_r5": None,
                "i2t_r10": None,
                "t2i_r1": None,
                "t2i_r5": None,
                "t2i_r10": None,
                "mean_recall": None,
            }
        )
    return summary


def save_benchmark_outputs(output_dir, summary, phase_metrics):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = output_dir / "selection_efficiency_summary.json"
    summary_csv_path = output_dir / "selection_efficiency_summary.csv"
    phase_json_path = output_dir / "selection_phase_metrics.json"

    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(phase_json_path, "w", encoding="utf-8") as handle:
        json.dump(phase_metrics, handle, ensure_ascii=False, indent=2)
    with open(summary_csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    return {
        "summary_json": str(summary_json_path),
        "summary_csv": str(summary_csv_path),
        "phase_json": str(phase_json_path),
    }


def run_selection_efficiency_benchmark(args):
    benchmark_log(
        f"start selection-only benchmark: dataset={args.dataset}, budget_size={args.budget_size}, "
        f"budget_ratio={args.budget_ratio}, correction={args.correction_mode}, fusion={args.fusion_mode}"
    )
    cross_args = build_cross_args(args)
    selection_args = build_selection_args(args)
    phase = PhaseBenchmark(
        phase_name="selection",
        enable_benchmark=bool(args.enable_selection_efficiency_benchmark),
        energy_backend=args.energy_backend,
        poll_interval_ms=args.poll_interval_ms,
    )

    with phase:
        cross_outputs = run_cross_modal_topology(cross_args)
        selection_outputs = run_subset_selection(selection_args)

    phase_metrics = phase.to_dict()
    variant_name = infer_variant_name(args)
    retrieval_metrics, retrieval_metrics_path = load_retrieval_metrics(args, variant_name)
    summary = build_summary(
        args,
        phase_metrics,
        cross_outputs,
        selection_outputs,
        retrieval_metrics=retrieval_metrics,
        retrieval_metrics_path=retrieval_metrics_path,
    )
    baseline_summary = read_json_if_exists(args.baseline_summary)
    summary = compute_derived_metrics(summary, baseline_summary=baseline_summary, eps=args.benchmark_eps)

    benchmark_output_dir = Path(args.benchmark_output_dir or selection_outputs["output_dir"])
    saved_paths = save_benchmark_outputs(benchmark_output_dir, summary, phase_metrics)
    benchmark_log(f"selection-only benchmark completed: {saved_paths['summary_json']}")
    return {
        "cross_outputs": cross_outputs,
        "selection_outputs": selection_outputs,
        "phase_metrics": phase_metrics,
        "summary": summary,
        "saved_paths": saved_paths,
    }
