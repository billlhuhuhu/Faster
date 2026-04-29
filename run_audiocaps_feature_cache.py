import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import torch

from src.audiocaps_dataset import load_audiocaps_records
from src.fixed_audio_features import extract_logmel_stats_features
from src.fixed_text_features import extract_fixed_text_features
from src.topology_graph import sanitize_name


def build_cache_dir(args):
    model_tag = f"{sanitize_name(args.audio_encoder)}_{sanitize_name(args.text_encoder)}"
    return Path(args.cache_root) / "audiocaps" / args.split / model_tag


def cache_exists(cache_dir):
    required = ["img_features_selection.pt", "txt_features_selection.pt", "sample_meta.json", "feature_info.json"]
    return all((Path(cache_dir) / name).exists() for name in required)


def save_cache(cache_dir, audio_features, text_features, sample_meta, feature_info):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(audio_features, dtype=torch.float32), cache_dir / "img_features_selection.pt")
    torch.save(torch.tensor(text_features, dtype=torch.float32), cache_dir / "txt_features_selection.pt")
    with open(cache_dir / "sample_meta.json", "w", encoding="utf-8") as handle:
        json.dump(sample_meta, handle, ensure_ascii=False, indent=2)
    with open(cache_dir / "feature_info.json", "w", encoding="utf-8") as handle:
        json.dump(feature_info, handle, ensure_ascii=False, indent=2)


def run_audiocaps_feature_cache(args):
    cache_dir = build_cache_dir(args)
    if cache_exists(cache_dir) and not args.overwrite:
        print(f"AudioCaps cache already exists at {cache_dir}. Use --overwrite to regenerate.")
        return {"cache_dir": str(cache_dir)}

    records, annotation_path = load_audiocaps_records(
        args.data_root,
        split=args.split,
        annotation_path=args.annotation_path,
        max_samples=args.max_samples,
    )
    audio_paths = [item["audio"] for item in records]
    texts = [item["caption"] for item in records]

    if args.audio_feature_mode != "logmel_stats":
        raise ValueError(f"Unsupported AudioCaps audio_feature_mode: {args.audio_feature_mode}")
    audio_features, audio_info = extract_logmel_stats_features(
        audio_paths,
        sample_rate=args.audio_sample_rate,
        n_mels=args.audio_n_mels,
        n_fft=args.audio_n_fft,
        hop_length=args.audio_hop_length,
        max_duration_sec=args.audio_max_duration_sec,
    )
    text_features = extract_fixed_text_features(
        texts,
        text_repr_method=args.text_feature_mode,
        batch_size=args.text_batch_size,
        device=args.device,
        tfidf_ngram_max=args.tfidf_ngram_max,
        tfidf_stop_words=args.tfidf_stop_words,
        tfidf_max_features=args.tfidf_max_features,
        tfidf_min_df=args.tfidf_min_df,
        tfidf_svd_dim=args.tfidf_svd_dim,
        tfidf_random_state=args.random_state,
    )

    feature_info = {
        "dataset": "audiocaps",
        "split": args.split,
        "annotation_path": annotation_path,
        "audio_encoder": args.audio_encoder,
        "image_encoder": args.audio_encoder,
        "text_encoder": args.text_encoder,
        "selection_image_repr_method": args.audio_feature_mode,
        "selection_text_repr_method": args.text_feature_mode,
        "num_samples": int(len(records)),
        "img_feature_shape": list(audio_features.shape),
        "txt_feature_shape": list(text_features.shape),
        "audio_info": audio_info,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_cache(cache_dir, audio_features, text_features, records, feature_info)
    return {"cache_dir": str(cache_dir)}


def build_parser():
    parser = argparse.ArgumentParser(description="Extract AudioCaps audio/text fixed features for topology selection.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--annotation_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--cache_root", type=str, default="artifacts_audiocaps/feature_cache")
    parser.add_argument("--audio_encoder", type=str, default="logmel_stats")
    parser.add_argument("--text_encoder", type=str, default="bert")
    parser.add_argument("--audio_feature_mode", type=str, default="logmel_stats", choices=["logmel_stats"])
    parser.add_argument("--text_feature_mode", type=str, default="bert", choices=["bert", "tfidf"])
    parser.add_argument("--audio_sample_rate", type=int, default=16000)
    parser.add_argument("--audio_n_mels", type=int, default=64)
    parser.add_argument("--audio_n_fft", type=int, default=1024)
    parser.add_argument("--audio_hop_length", type=int, default=320)
    parser.add_argument("--audio_max_duration_sec", type=float, default=10.0)
    parser.add_argument("--text_batch_size", type=int, default=256)
    parser.add_argument("--tfidf_ngram_max", type=int, default=2)
    parser.add_argument("--tfidf_stop_words", type=str, default="english")
    parser.add_argument("--tfidf_max_features", type=int, default=20000)
    parser.add_argument("--tfidf_min_df", type=int, default=1)
    parser.add_argument("--tfidf_svd_dim", type=int, default=256)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    outputs = run_audiocaps_feature_cache(args)
    print("AudioCaps feature cache saved:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
