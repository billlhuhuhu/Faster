import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.cross_modal_topology import run_cross_modal_topology


def make_sym_graph(edges, num_nodes):
    rows = np.array([src for src, _, _ in edges], dtype=np.int64)
    cols = np.array([dst for _, dst, _ in edges], dtype=np.int64)
    vals = np.array([weight for _, _, weight in edges], dtype=np.float32)
    directed = sparse.csr_matrix((vals, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32)
    directed = directed + directed.transpose()
    directed.data = np.clip(directed.data, 0.0, 1.0)
    directed.eliminate_zeros()
    directed = directed.maximum(directed.transpose()).tocsr()
    return directed


def row_normalize(graph):
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    degree = np.maximum(degree, 1e-12)
    inv_degree = sparse.diags(1.0 / degree.astype(np.float32))
    return (inv_degree @ graph).tocsr()


def write_graph_bundle(root_dir, modality, graph, collapse_score, spectral_entropy, sample_meta):
    graph_dir = root_dir / "topology_graph" / "flickr" / "train" / "nfnet_bert" / modality / "k4_euclidean"
    graph_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": "flickr",
        "split": "train",
        "modality": modality,
        "image_encoder": "nfnet",
        "text_encoder": "bert",
        "metric": "euclidean",
        "k": 4,
        "num_nodes": int(graph.shape[0]),
        "num_edges": int(graph.nnz),
        "avg_degree": float(graph.nnz / graph.shape[0]),
        "first_eigenvalues": [0.0, 0.2, 0.5],
        "spectral_entropy": spectral_entropy,
        "collapse_score": collapse_score,
        "num_eigs_used": 3,
    }

    sparse.save_npz(graph_dir / "symmetric_graph.npz", graph)
    sparse.save_npz(graph_dir / "transition_graph.npz", row_normalize(graph))
    with open(graph_dir / "sample_meta.json", "w", encoding="utf-8") as handle:
        json.dump(sample_meta, handle, ensure_ascii=False, indent=2)
    with open(graph_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def main():
    root_dir = Path("artifacts/tmp_verify_cross_modal")
    if root_dir.exists():
        shutil.rmtree(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    sample_meta = [
        {
            "sample_idx": idx,
            "img_id": idx // 2,
            "dataset": "flickr",
            "split": "train",
            "caption": f"caption {idx}",
            "image": f"flickr30k-images/{idx:06d}.jpg",
        }
        for idx in range(8)
    ]

    image_graph = make_sym_graph(
        [
            (0, 1, 0.9), (1, 2, 0.8), (2, 3, 0.85),
            (4, 5, 0.88), (5, 6, 0.83), (6, 7, 0.86),
            (1, 3, 0.52), (4, 7, 0.48),
        ],
        num_nodes=8,
    )
    text_graph = make_sym_graph(
        [
            (0, 1, 0.9), (0, 4, 0.7), (1, 5, 0.72),
            (2, 6, 0.68), (3, 7, 0.71), (4, 5, 0.85),
            (2, 3, 0.8), (6, 7, 0.82),
        ],
        num_nodes=8,
    )

    write_graph_bundle(root_dir, "image", image_graph, collapse_score=0.12, spectral_entropy=0.88, sample_meta=sample_meta)
    write_graph_bundle(root_dir, "text", text_graph, collapse_score=0.35, spectral_entropy=0.65, sample_meta=sample_meta)

    args = SimpleNamespace(
        dataset="flickr",
        split="train",
        image_encoder="nfnet",
        text_encoder="bert",
        topology_root=str(root_dir / "topology_graph"),
        output_root=str(root_dir / "cross_modal_topology"),
        metric="euclidean",
        k=4,
        alpha=1.0,
        fusion_mode="intersection",
        prefer_healthy_modality=None,
    )
    outputs = run_cross_modal_topology(args)
    summary = outputs["summary"]

    assert summary["healthy_modality"] == "image"
    assert summary["corrected_summary"]["num_edges"] > 0
    assert summary["unified_summary"]["num_edges"] > 0
    assert summary["unified_summary"]["num_edges"] <= summary["corrected_summary"]["num_edges"]

    output_dir = Path(outputs["output_dir"])
    assert (output_dir / "corrected_graph_symmetric.npz").exists()
    assert (output_dir / "unified_graph.npz").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "modality_selection.json").exists()

    print("Cross-modal topology verification passed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
