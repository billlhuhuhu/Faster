#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VLM_MODEL_NAME_OR_PATH="${VLM_MODEL_NAME_OR_PATH:-/home/hzx/Faster/distill_utils/checkpoints/Qwen2-VL-2B-Instruct}"
VLM_BASE_EVAL_OUTPUT_ROOT="${VLM_BASE_EVAL_OUTPUT_ROOT:-artifacts/vlm_finetune/qwen2vl_base_eval}"
VLM_BASE_EVAL_DATASET_NAME="${VLM_BASE_EVAL_DATASET_NAME:-llava_instruct_150k}"
VLM_BASE_EVAL_BENCHMARKS="${VLM_BASE_EVAL_BENCHMARKS:-GQA,ScienceQA-IMG,MMBench,TextVQA,POPE}"
VLM_BASE_EVAL_SEED="${VLM_BASE_EVAL_SEED:-0}"
VLMEVALKIT_ROOT="${VLMEVALKIT_ROOT:-/home/hzx/Faster/VLMEvalKit}"
VLM_EVAL_EXECUTE="${VLM_EVAL_EXECUTE:-1}"
VLMEVAL_NPROC="${VLMEVAL_NPROC:-1}"
VLM_EVAL_USE_FLASH_ATTN="${VLM_EVAL_USE_FLASH_ATTN:-0}"

cd "${PROJECT_ROOT}"

python "${PROJECT_ROOT}/run_vlm_base_eval_plan.py" \
  --model_name_or_path "${VLM_MODEL_NAME_OR_PATH}" \
  --output_root "${VLM_BASE_EVAL_OUTPUT_ROOT}" \
  --dataset_name "${VLM_BASE_EVAL_DATASET_NAME}" \
  --subset_mode base \
  --seed "${VLM_BASE_EVAL_SEED}" \
  --eval_benchmarks "${VLM_BASE_EVAL_BENCHMARKS}" \
  --vlmeval_model_key qwen2vl_base \
  --vlmeval_model_class Qwen2VLChat

VLM_EVAL_PLAN_ROOT="${VLM_BASE_EVAL_OUTPUT_ROOT}" \
VLM_EVAL_PLAN_GLOB="benchmark_eval_plan.json" \
VLM_EVAL_EXECUTE="${VLM_EVAL_EXECUTE}" \
VLMEVALKIT_ROOT="${VLMEVALKIT_ROOT}" \
VLMEVAL_NPROC="${VLMEVAL_NPROC}" \
VLM_EVAL_USE_FLASH_ATTN="${VLM_EVAL_USE_FLASH_ATTN}" \
bash "${SCRIPT_DIR}/run_vlmeval_qwen2vl_subset_eval.sh"
