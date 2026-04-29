import argparse
import json
from pathlib import Path
from typing import Dict, List


def sanitize_name(value: str) -> str:
    safe = str(value).replace("\\", "-").replace("/", "-").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "-" for ch in safe)


def benchmark_names(raw_value: str) -> List[str]:
    value = str(raw_value or "").strip()
    if not value:
        return ["GQA", "ScienceQA-IMG", "MMBench", "TextVQA", "POPE"]
    return [item.strip() for item in value.replace(",", " ").split() if item.strip()]


def build_eval_dataset_mapping(benchmarks: List[str]) -> Dict[str, str]:
    default_mapping = {
        "GQA": "GQA",
        "ScienceQA-IMG": "ScienceQA_VAL",
        "MMBench": "MMBench_DEV_EN_V11",
        "TextVQA": "TextVQA_VAL",
        "POPE": "POPE",
    }
    return {name: default_mapping.get(name, name) for name in benchmarks}


def build_vlmeval_dataset_config(dataset_name: str) -> Dict[str, str]:
    lower = str(dataset_name).lower()
    if "mmbench" in lower or "scienceqa" in lower:
        dataset_class = "ImageMCQDataset"
    elif "pope" in lower:
        dataset_class = "ImageYORNDataset"
    else:
        dataset_class = "ImageVQADataset"
    return {"class": dataset_class, "dataset": dataset_name}


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a VLMEvalKit eval plan for the unfine-tuned base VLM.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="artifacts/vlm_finetune/qwen2vl_llava_subset")
    parser.add_argument("--dataset_name", type=str, default="llava_instruct_150k")
    parser.add_argument("--subset_mode", type=str, default="base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_benchmarks", type=str, default="GQA,ScienceQA-IMG,MMBench,TextVQA,POPE")
    parser.add_argument("--vlmeval_model_key", type=str, default="qwen2vl_base")
    parser.add_argument("--vlmeval_model_class", type=str, default="Qwen2VLChat")
    parser.add_argument("--vlmeval_use_flash_attn", action="store_true", default=False)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    args = parser.parse_args()

    model_tag = sanitize_name(args.model_name_or_path)
    output_dir = Path(args.output_root) / model_tag / args.dataset_name / args.subset_mode / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = benchmark_names(args.eval_benchmarks)
    dataset_mapping = build_eval_dataset_mapping(benchmarks)
    model_key = sanitize_name(args.vlmeval_model_key)
    model_path = str(Path(args.model_name_or_path))
    config_path = output_dir / "vlmeval_config.json"
    vlmeval_output_dir = output_dir / "vlmevalkit_outputs"
    lmms_output_dir = output_dir / "lmms_eval_outputs"

    config = {
        "model": {
            model_key: {
                "class": args.vlmeval_model_class,
                "model_path": model_path,
                "model": model_path,
                "pretrained": model_path,
                "trust_remote_code": bool(args.trust_remote_code),
                "use_flash_attn": bool(args.vlmeval_use_flash_attn),
                "attn_implementation": "flash_attention_2" if bool(args.vlmeval_use_flash_attn) else "sdpa",
            }
        },
        "data": {
            dataset_mapping.get(name, name): build_vlmeval_dataset_config(dataset_mapping.get(name, name))
            for name in benchmarks
        },
    }
    write_json(config_path, config)

    vlmeval_python_command = (
        f"python $VLMEVALKIT_ROOT/run.py --config {config_path} "
        f"--work-dir {vlmeval_output_dir} --mode all --verbose"
    )
    vlmeval_torchrun_command = (
        f"torchrun --nproc-per-node=$VLMEVAL_NPROC $VLMEVALKIT_ROOT/run.py "
        f"--config {config_path} --work-dir {vlmeval_output_dir} --mode all --verbose"
    )
    payload = {
        "status": "ready",
        "eval_backend": "vlmevalkit",
        "model_name_or_path": model_path,
        "base_model_path": model_path,
        "adapter_path": "",
        "merged_model_path": "",
        "merge_lora_for_eval": False,
        "merge_status": "base_model_no_merge",
        "recommended_model_source": "base_model",
        "recommended_model_path": model_path,
        "best_checkpoint": "",
        "last_checkpoint": "",
        "subset_mode": args.subset_mode,
        "subset_ratio": None,
        "seed": int(args.seed),
        "benchmarks": benchmarks,
        "vlmevalkit_dataset_mapping": dataset_mapping,
        "vlmevalkit_model_key": model_key,
        "vlmevalkit_model_class": args.vlmeval_model_class,
        "vlmevalkit_config_path": str(config_path),
        "output_dir": str(output_dir),
        "vlmevalkit_output_dir": str(vlmeval_output_dir),
        "lmms_eval_output_dir": str(lmms_output_dir),
        "recommended_commands": {
            "vlmevalkit_config_python": vlmeval_python_command,
            "vlmevalkit_config_torchrun": vlmeval_torchrun_command,
        },
    }
    write_json(output_dir / "benchmark_eval_plan.json", payload)
    print(f"base eval plan: {output_dir / 'benchmark_eval_plan.json'}")
    print(f"vlmeval config: {config_path}")


if __name__ == "__main__":
    main()
