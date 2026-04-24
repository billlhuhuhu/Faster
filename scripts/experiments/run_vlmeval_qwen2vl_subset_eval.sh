#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PLAN_ROOT="${VLM_EVAL_PLAN_ROOT:-artifacts/vlm_finetune/qwen2vl_llava_subset}"
PLAN_GLOB="${VLM_EVAL_PLAN_GLOB:-benchmark_eval_plan.json}"
COMMANDS_PATH="${VLM_EVAL_COMMANDS_PATH:-${PLAN_ROOT}/run_vlmevalkit_all.sh}"
EXECUTE="${VLM_EVAL_EXECUTE:-0}"
VLMEVALKIT_ROOT="${VLMEVALKIT_ROOT:-}"
VLMEVAL_NPROC="${VLMEVAL_NPROC:-1}"
USE_TORCHRUN="${VLM_EVAL_USE_TORCHRUN:-0}"
USE_FLASH_ATTN="${VLM_EVAL_USE_FLASH_ATTN:-0}"
DATASETS_OVERRIDE="${VLM_EVAL_VLMEVAL_DATASETS:-}"

python - "${PLAN_ROOT}" "${PLAN_GLOB}" "${COMMANDS_PATH}" "${USE_TORCHRUN}" "${USE_FLASH_ATTN}" "${VLMEVALKIT_ROOT}" "${DATASETS_OVERRIDE}" <<'PY'
import json
import os
import sys
from pathlib import Path

plan_root = Path(sys.argv[1])
plan_glob = sys.argv[2]
commands_path = Path(sys.argv[3])
use_torchrun = str(sys.argv[4]) == "1"
use_flash_attn = str(sys.argv[5]) == "1"
vlmevalkit_root = Path(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] else None
datasets_override = sys.argv[7] if len(sys.argv) > 7 else ""


def split_names(raw):
    return [item.strip() for item in str(raw or "").replace(",", " ").split() if item.strip()]


def collect_supported_dataset_names(root):
    if root is None or not root.exists():
        return set()
    import importlib

    sys.path.insert(0, str(root))
    names = set()
    try:
        dataset_module = importlib.import_module("vlmeval.dataset")
    except Exception as exc:
        print(f"[VLMEvalKit] warning: failed to inspect supported datasets: {exc}")
        return names
    for attr in dir(dataset_module):
        value = getattr(dataset_module, attr, None)
        if isinstance(value, dict):
            names.update(str(key) for key in value.keys())
    return names


def resolve_dataset_name(name, supported_names):
    if not supported_names or name in supported_names:
        return name
    aliases = {
        "GQA_TestDev_Balanced": ["GQA", "GQA_TESTDEV", "GQA_TestDev", "GQA_VAL"],
        "GQA": ["GQA", "GQA_TESTDEV", "GQA_TestDev", "GQA_VAL"],
        "ScienceQA_VAL": ["ScienceQA_TEST", "ScienceQA_VAL", "ScienceQA", "ScienceQA_IMG"],
        "ScienceQA-IMG": ["ScienceQA_TEST", "ScienceQA_VAL", "ScienceQA", "ScienceQA_IMG"],
        "MMBench_DEV_EN": ["MMBench_DEV_EN", "MMBench_DEV_EN_V11", "MMBench_DEV_EN_V12", "MMBench_DEV"],
        "MMBench": ["MMBench_DEV_EN", "MMBench_DEV_EN_V11", "MMBench_DEV_EN_V12", "MMBench_DEV"],
        "TextVQA_VAL": ["TextVQA_VAL", "TextVQA"],
        "TextVQA": ["TextVQA_VAL", "TextVQA"],
        "POPE": ["POPE", "POPE_COCO", "POPE_Random", "POPE_RANDOM", "POPE_POPULAR", "POPE_ADVERSARIAL"],
    }
    for candidate in aliases.get(name, []):
        if candidate in supported_names:
            return candidate
    lower = name.lower()
    for candidate in sorted(supported_names):
        if candidate.lower() == lower:
            return candidate
    compact = lower.replace("_", "").replace("-", "")
    for candidate in sorted(supported_names):
        cand_compact = candidate.lower().replace("_", "").replace("-", "")
        if compact and (compact in cand_compact or cand_compact in compact):
            return candidate
    return name


def build_dataset_config(name):
    lower = str(name).lower()
    if "mmbench" in lower or "scienceqa" in lower:
        dataset_class = "ImageMCQDataset"
    elif "pope" in lower:
        dataset_class = "ImageYORNDataset"
    else:
        dataset_class = "ImageVQADataset"
    return {
        "class": dataset_class,
        "dataset": name,
    }


supported_dataset_names = collect_supported_dataset_names(vlmevalkit_root)
override_names = split_names(datasets_override)
if supported_dataset_names:
    interesting = [
        name for name in sorted(supported_dataset_names)
        if any(token in name.lower() for token in ["gqa", "science", "mmbench", "textvqa", "pope"])
    ]
    print(f"[VLMEvalKit] matched dataset candidates: {interesting[:80]}")

plans = sorted(plan_root.rglob(plan_glob))
commands_path.parent.mkdir(parents=True, exist_ok=True)
lines = [
    "#!/usr/bin/env bash",
    "set -euo pipefail",
    ': "${VLMEVALKIT_ROOT:?Set VLMEVALKIT_ROOT to your VLMEvalKit checkout}"',
    'VLMEVAL_NPROC="${VLMEVAL_NPROC:-1}"',
    "",
]

for plan_path in plans:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    config_path = plan.get("vlmevalkit_config_path")
    if config_path:
        cfg_path = Path(config_path)
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            for model_cfg in cfg.get("model", {}).values():
                if isinstance(model_cfg, dict):
                    model_cfg["use_flash_attn"] = bool(use_flash_attn)
                    model_cfg["attn_implementation"] = "flash_attention_2" if use_flash_attn else "sdpa"
            original_data = cfg.get("data", {})
            original_data_names = [
                value.get("dataset", key) if isinstance(value, dict) and value.get("dataset") else key
                for key, value in original_data.items()
            ]
            if override_names:
                resolved_data_names = override_names
            else:
                resolved_data_names = [
                    resolve_dataset_name(name, supported_dataset_names)
                    for name in original_data_names
                ]
            cfg["data"] = {name: build_dataset_config(name) for name in resolved_data_names}
            if original_data_names != resolved_data_names:
                print(f"[VLMEvalKit] remapped datasets: {original_data_names} -> {resolved_data_names}")
            else:
                print(f"[VLMEvalKit] using datasets: {resolved_data_names}")
            cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    commands = plan.get("recommended_commands", {})
    command = commands.get("vlmevalkit_config_torchrun" if use_torchrun else "vlmevalkit_config_python")
    if not command:
        continue
    lines.append(f'echo "[VLMEvalKit] {plan_path}"')
    lines.append(command)
    lines.append("")

commands_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"found plans: {len(plans)}")
print(f"saved commands: {commands_path}")
PY

chmod +x "${COMMANDS_PATH}"

if [[ "${EXECUTE}" == "1" ]]; then
  if [[ -z "${VLMEVALKIT_ROOT}" ]]; then
    echo "VLMEVALKIT_ROOT is required when VLM_EVAL_EXECUTE=1" >&2
    exit 1
  fi
  export VLMEVALKIT_ROOT
  export VLMEVAL_NPROC
  export VLM_EVAL_USE_FLASH_ATTN="${USE_FLASH_ATTN}"
  if [[ "${USE_FLASH_ATTN}" != "1" ]]; then
    python - <<'PY'
import os
import shutil
from pathlib import Path

root = Path(os.environ["VLMEVALKIT_ROOT"])
target = root / "vlmeval" / "vlm" / "qwen2_vl" / "model.py"
if not target.exists():
    print(f"[VLMEvalKit patch] skip: {target} not found")
else:
    text = target.read_text(encoding="utf-8")
    patched = text.replace("flash_attention_2", "sdpa")
    if patched != text:
        backup = target.with_suffix(target.suffix + ".flash_attn.bak")
        if not backup.exists():
            shutil.copy2(target, backup)
        target.write_text(patched, encoding="utf-8")
        print(f"[VLMEvalKit patch] disabled hard-coded flash_attention_2 in {target}")
    else:
        print(f"[VLMEvalKit patch] no hard-coded flash_attention_2 found in {target}")
PY
  fi
  bash "${COMMANDS_PATH}"
else
  echo "Dry run only. Review and execute:"
  echo "  VLMEVALKIT_ROOT=/path/to/VLMEvalKit VLMEVAL_NPROC=${VLMEVAL_NPROC} bash ${COMMANDS_PATH}"
  echo "Or run through this wrapper:"
  echo "  VLM_EVAL_EXECUTE=1 VLMEVALKIT_ROOT=/path/to/VLMEvalKit bash scripts/experiments/run_vlmeval_qwen2vl_subset_eval.sh"
fi

python "${PROJECT_ROOT}/tools/collect_vlmeval_results.py" \
  --plan_root "${PLAN_ROOT}" \
  --output_csv "${PLAN_ROOT}/reports/vlmevalkit_results_summary.csv" \
  --output_json "${PLAN_ROOT}/reports/vlmevalkit_results_summary.json" || true
