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

python - "${PLAN_ROOT}" "${PLAN_GLOB}" "${COMMANDS_PATH}" "${USE_TORCHRUN}" <<'PY'
import json
import sys
from pathlib import Path

plan_root = Path(sys.argv[1])
plan_glob = sys.argv[2]
commands_path = Path(sys.argv[3])
use_torchrun = str(sys.argv[4]) == "1"

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
                    model_cfg["use_flash_attn"] = False
                    model_cfg["attn_implementation"] = "sdpa"
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
