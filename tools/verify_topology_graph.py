import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.topology_graph import run_topology_graph


def build_synthetic_feature_cache(root_dir: Path):
    feature_dir = root_dir / "feature_cache" / "flickr" / "train" / "nfnet_bert"
    feature_dir.mkdir(parents=True, exist_ok=True)

    img_cluster_a = torch.randn(8, 4) * 0.05 + 0.0
    img_cluster_b = torch.randn(8, 4) * 0.05 + 2.0
    txt_cluster_a = torch.randn(8, 3) * 0.05 + 0.0
    txt_cluster_b = torch.randn(8, 3) * 0.05 + 2.0

    img_features = torch.cat([img_cluster_a, img_cluster_b], dim=0).float()
    txt_features = torch.cat([txt_cluster_a, txt_cluster_b], dim=0).float()
    sample_meta = [
        {
            "sample_idx": idx,
            "img_id": idx // 2,
            "dataset": "flickr",
            "split": "train",
            "caption": f"caption {idx}",
            "image": f"flickr30k-images/{idx:06d}.jpg",
        }
        for idx in range(img_features.shape[0])
    ]

    torch.save(img_features, feature_dir / "img_features.pt")
    torch.save(txt_features, feature_dir / "txt_features.pt")
    with open(feature_dir / "sample_meta.json", "w", encoding="utf-8") as handle:
        json.dump(sample_meta, handle, ensure_ascii=False, indent=2)

    return feature_dir


def main():
    root_dir = Path("artifacts/tmp_verify_topology")
    if root_dir.exists():
        shutil.rmtree(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    build_synthetic_feature_cache(root_dir)

    args = SimpleNamespace(
        dataset="flickr",
        split="train",
        image_encoder="nfnet",
        text_encoder="bert",
        modality="image",
        feature_cache_root=str(root_dir / "feature_cache"),
        output_root=str(root_dir / "topology_graph"),
        metric="euclidean",
        k=4,
        num_eigs=6,
        n_jobs=None,
        local_connectivity=1.0,
        bandwidth=None,
        sigma_search_steps=32,
        save_eigenvectors=False,
        max_samples=None,
    )
    outputs = run_topology_graph(args)
    summary = outputs["summary"]

    assert summary["num_nodes"] == 16
    assert summary["num_edges"] > 0
    assert len(summary["first_eigenvalues"]) > 0
    assert 0.0 <= summary["collapse_score"] <= 1.0

    output_dir = Path(outputs["output_dir"])
    assert (output_dir / "knn_indices.pt").exists()
    assert (output_dir / "knn_distances.pt").exists()
    assert (output_dir / "local_scale.pt").exists()
    assert (output_dir / "symmetric_graph.npz").exists()
    assert (output_dir / "laplacian_normalized.npz").exists()
    assert (output_dir / "summary.json").exists()

    print("Topology graph verification passed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
