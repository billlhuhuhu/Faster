#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Official VLM downstream run:
# dense_sift_bovw+BERT Ours sampling -> Qwen2-VL LoRA finetuning -> VLMEvalKit evaluation.
# Override any variable from the command line for ablations.
export VLM_RUN_DENSE_SIFT_BOVW_SELECTION="${VLM_RUN_DENSE_SIFT_BOVW_SELECTION:-1}"
export VLM_RUN_FULL="${VLM_RUN_FULL:-1}"
export VLM_RUN_RANDOM="${VLM_RUN_RANDOM:-1}"
export VLM_RUN_OURS="${VLM_RUN_OURS:-1}"
export VLM_SUBSET_RATIOS="${VLM_SUBSET_RATIOS:-1 5 10}"

export VLM_FINETUNE_MODE="${VLM_FINETUNE_MODE:-lora}"
export VLM_NUM_TRAIN_EPOCHS="${VLM_NUM_TRAIN_EPOCHS:-2}"
export VLM_LEARNING_RATE="${VLM_LEARNING_RATE:-1e-4}"
export VLM_LORA_R="${VLM_LORA_R:-16}"
export VLM_LORA_ALPHA="${VLM_LORA_ALPHA:-32}"
export VLM_LORA_DROPOUT="${VLM_LORA_DROPOUT:-0.05}"
export VLM_PER_DEVICE_TRAIN_BATCH_SIZE="${VLM_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export VLM_PER_DEVICE_EVAL_BATCH_SIZE="${VLM_PER_DEVICE_EVAL_BATCH_SIZE:-1}"
export VLM_GRADIENT_ACCUMULATION_STEPS="${VLM_GRADIENT_ACCUMULATION_STEPS:-8}"
export VLM_MAX_LENGTH="${VLM_MAX_LENGTH:-2048}"
export VLM_VAL_RATIO="${VLM_VAL_RATIO:-0.02}"
export VLM_SAVE_TOTAL_LIMIT="${VLM_SAVE_TOTAL_LIMIT:-2}"
export VLM_MERGE_LORA_FOR_EVAL="${VLM_MERGE_LORA_FOR_EVAL:-1}"

export VLM_FINETUNE_USE_TORCHRUN="${VLM_FINETUNE_USE_TORCHRUN:-1}"
export VLM_FINETUNE_CUDA_VISIBLE_DEVICES="${VLM_FINETUNE_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
if [[ -z "${VLM_FINETUNE_NPROC_PER_NODE:-}" ]]; then
  export VLM_FINETUNE_NPROC_PER_NODE="$(python - <<PY
devices = "${VLM_FINETUNE_CUDA_VISIBLE_DEVICES}".strip()
print(max(len([item for item in devices.split(",") if item.strip()]), 1))
PY
)"
fi

export VLM_EVAL_EXECUTE="${VLM_EVAL_EXECUTE:-1}"
export VLM_EVAL_USE_FLASH_ATTN="${VLM_EVAL_USE_FLASH_ATTN:-0}"
export VLM_EVAL_USE_TORCHRUN="${VLM_EVAL_USE_TORCHRUN:-0}"
export VLMEVAL_NPROC="${VLMEVAL_NPROC:-1}"
export VLMEVALKIT_ROOT="${VLMEVALKIT_ROOT:-/home/hzx/Faster/VLMEvalKit}"

bash "${SCRIPT_DIR}/run_qwen2vl_llava_subset_full_pipeline.sh"
