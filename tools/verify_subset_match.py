import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy import sparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.subset_dataset import load_selected_indices
from src.subset_match import run_subset_selection


def write_feature_cache(root_dir):
    feature_dir = root_dir / "feature_cache" / "flickr" / "train" / "nfnet_bert"
    feature_dir.mkdir(parents=True, exist_ok=True)

    img_features = []
    txt_features = []
    sample_meta = []

    centers = [
        (np.array([0.0, 0.0], dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32)),
        (np.array([3.0, 3.0], dtype=np.float32), np.array([3.0, 3.0], dtype=np.float32)),
        (np.array([-3.0, 3.0], dtype=np.float32), np.array([-3.0, 3.0], dtype=np.float32)),
        (np.array([3.0, -3.0], dtype=np.float32), np.array([3.0, -3.0], dtype=np.float32)),
    ]

    idx = 0
    for cluster_id, (img_center, txt_center) in enumerate(centers):
        for _ in range(5):
            img_features.append(img_center + np.random.randn(2).astype(np.float32) * 0.05)
            txt_features.append(txt_center + np.random.randn(2).astype(np.float32) * 0.05)
            sample_meta.append(
                {
                    "sample_idx": idx,
                    "img_id": cluster_id,
                    "dataset": "flickr",
                    "split": "train",
                    "caption": f"caption {idx}",
                    "image": f"flickr30k-images/{idx:06d}.jpg",
                }
            )
            idx += 1

    torch.save(torch.tensor(np.stack(img_features), dtype=torch.float32), feature_dir / "img_features.pt")
    torch.save(torch.tensor(np.stack(txt_features), dtype=torch.float32), feature_dir / "txt_features.pt")
    with open(feature_dir / "sample_meta.json", "w", encoding="utf-8") as handle:
        json.dump(sample_meta, handle, ensure_ascii=False, indent=2)


def write_cross_modal_topology(root_dir):
    topology_dir = root_dir / "cross_modal_topology" / "flickr" / "train" / "nfnet_bert" / "k15_euclidean_a1.0"
    topology_dir.mkdir(parents=True, exist_ok=True)

    num_nodes = 20
    rows = []
    cols = []
    vals = []
    for cluster_start in range(0, num_nodes, 5):
        for i in range(cluster_start, cluster_start + 5):
            for j in range(cluster_start, cluster_start + 5):
                if i != j:
                    rows.append(i)
                    cols.append(j)
                    vals.append(0.9)

    graph = sparse.csr_matrix((np.array(vals, dtype=np.float32), (np.array(rows), np.array(cols))), shape=(num_nodes, num_nodes))
    graph.eliminate_zeros()
    sparse.save_npz(topology_dir / "unified_graph.npz", graph)
    summary = {
        "healthy_modality": "image",
        "unified_summary": {
            "num_nodes": num_nodes,
            "num_edges": int(graph.nnz),
        },
    }
    with open(topology_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def main():
    root_dir = Path("artifacts/tmp_verify_subset")
    if root_dir.exists():
        shutil.rmtree(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    write_feature_cache(root_dir)
    write_cross_modal_topology(root_dir)

    args = SimpleNamespace(
        dataset="flickr",
        split="train",
        image_encoder="nfnet",
        text_encoder="bert",
        feature_cache_root=str(root_dir / "feature_cache"),
        cross_modal_root=str(root_dir / "cross_modal_topology"),
        output_root=str(root_dir / "subset_selection"),
        metric="euclidean",
        k=15,
        alpha=1.0,
        budget_ratio=0.1,
        representation_mode="concat",
        selection_method="proxy_opt",
        cluster_method="kmeans",
        degree_weight=0.1,
        random_state=0,
        minibatch_size=32,
        device="cpu",
        proxy_projection_dim=4,
        proxy_init_method="kmeans",
        proxy_num_frequencies=16,
        proxy_frequency_scale=1.0,
        proxy_lr=0.05,
        proxy_num_steps=30,
        proxy_reg_weight=0.01,
        proxy_target_batch_size=64,
        proxy_batch_size=64,
        matching_top_k=8,
        matching_candidate_batch_size=8,
        topology_weight=0.5,
        topology_hop_weight=0.5,
    )
    outputs = run_subset_selection(args)

    selected_indices = outputs["selected_indices"]
    assert len(selected_indices) == 2
    assert len(set(selected_indices)) == len(selected_indices)

    output_dir = Path(outputs["output_dir"])
    assert (output_dir / "selected_indices.json").exists()
    assert (output_dir / "selected_meta.json").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "proxy_points.pt").exists()
    assert (output_dir / "proxy_init.pt").exists()
    assert (output_dir / "proxy_debug.json").exists()
    assert (output_dir / "matching_cost.pt").exists()
    assert (output_dir / "matching_debug.json").exists()

    loaded_indices = load_selected_indices(output_dir / "selected_indices.json")
    assert loaded_indices == selected_indices

    selected_meta = json.load(open(output_dir / "selected_meta.json", "r", encoding="utf-8"))
    assert len(selected_meta) == len(selected_indices)
    assert [item["sample_idx"] for item in selected_meta] == selected_indices
    assert outputs["summary"]["selection_method"] == "proxy_opt"

    print("Subset match verification passed.")
    print(json.dumps(outputs["summary"], indent=2))


if __name__ == "__main__":
    main()
