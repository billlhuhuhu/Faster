#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Paths from the current server setup. Override from the command line when needed.
export VLM_MODEL_NAME_OR_PATH="${VLM_MODEL_NAME_OR_PATH:-/home/hzx/Faster/distill_utils/checkpoints/Qwen2-VL-2B-Instruct}"
export LLAVA_ANNOTATION_PATH="${LLAVA_ANNOTATION_PATH:-/home/hzx/Faster/data/llava/llava_instruct_150k.json}"
export LLAVA_IMAGE_ROOT="${LLAVA_IMAGE_ROOT:-/home/hzx/Faster/data/coco/train2014}"

# Optional: point this to your Ours subset indices. Supports {ratio} and {ratio_tag}.
# Example:
# export VLM_OURS_SELECTED_INDICES_TEMPLATE="/home/hzx/Faster/artifacts/llava_subset/{ratio_tag}/proxy_opt/seed_0/selected_indices.json"
export VLM_OURS_SELECTED_INDICES_TEMPLATE="${VLM_OURS_SELECTED_INDICES_TEMPLATE:-}"

export VLM_FINETUNE_OUTPUT_ROOT="${VLM_FINETUNE_OUTPUT_ROOT:-artifacts/vlm_finetune/qwen2vl_llava_subset}"
export VLM_DATASET_NAME="${VLM_DATASET_NAME:-llava_instruct_150k}"
export VLM_SUBSET_RATIOS="${VLM_SUBSET_RATIOS:-1 5 10}"
export VLM_SEED="${VLM_SEED:-0}"

export VLM_RUN_FULL="${VLM_RUN_FULL:-1}"
export VLM_RUN_RANDOM="${VLM_RUN_RANDOM:-1}"
export VLM_RUN_OURS="${VLM_RUN_OURS:-1}"

# Stage 0: build Ours subsets on the LLaVA mother set with dense_sift_bovw
# image features. If VLM_OURS_SELECTED_INDICES_TEMPLATE is already provided,
# this stage is skipped by default and the provided template is used.
export VLM_RUN_DENSE_SIFT_BOVW_SELECTION="${VLM_RUN_DENSE_SIFT_BOVW_SELECTION:-1}"
export VLM_DENSE_SIFT_BOVW_SELECTION_ROOT="${VLM_DENSE_SIFT_BOVW_SELECTION_ROOT:-artifacts/vlm_subset_selection/llava_dense_sift_bovw}"
export VLM_DENSE_SIFT_BOVW_CACHE_DIR="${VLM_DENSE_SIFT_BOVW_CACHE_DIR:-artifacts/vlm_feature_cache/llava_dense_sift_bovw}"
export VLM_DENSE_SIFT_BOVW_FEATURE_CACHE_ROOT="${VLM_DENSE_SIFT_BOVW_FEATURE_CACHE_ROOT:-artifacts/vlm_feature_cache_llava_dense_sift_bovw_full_pipeline}"
export VLM_DENSE_SIFT_BOVW_TOPOLOGY_ROOT="${VLM_DENSE_SIFT_BOVW_TOPOLOGY_ROOT:-artifacts/vlm_topology_graph_dense_sift_bovw}"
export VLM_DENSE_SIFT_BOVW_CROSS_MODAL_ROOT="${VLM_DENSE_SIFT_BOVW_CROSS_MODAL_ROOT:-artifacts/vlm_cross_modal_topology_dense_sift_bovw}"
export VLM_DENSE_SIFT_BOVW_PIPELINE_SELECTION_ROOT="${VLM_DENSE_SIFT_BOVW_PIPELINE_SELECTION_ROOT:-artifacts/vlm_subset_selection_dense_sift_bovw_full_pipeline}"
export VLM_SELECTION_DEVICE="${VLM_SELECTION_DEVICE:-cuda}"
export VLM_SELECTION_TEXT_REPR_METHOD="${VLM_SELECTION_TEXT_REPR_METHOD:-bert}"
export VLM_SELECTION_TEXT_BATCH_SIZE="${VLM_SELECTION_TEXT_BATCH_SIZE:-256}"
export VLM_SELECTION_IMAGE_METRIC="${VLM_SELECTION_IMAGE_METRIC:-euclidean}"
export VLM_SELECTION_TEXT_METRIC="${VLM_SELECTION_TEXT_METRIC:-cosine}"
export VLM_SELECTION_WAVELET_FUSION_WEIGHT_MODE="${VLM_SELECTION_WAVELET_FUSION_WEIGHT_MODE:-fixed_per_scale}"
export VLM_SELECTION_PROXY_NUM_STEPS="${VLM_SELECTION_PROXY_NUM_STEPS:-200}"
export VLM_SELECTION_PROXY_BATCH_SIZE="${VLM_SELECTION_PROXY_BATCH_SIZE:-2048}"
export VLM_SELECTION_PROXY_TARGET_BATCH_SIZE="${VLM_SELECTION_PROXY_TARGET_BATCH_SIZE:-2048}"
export VLM_SELECTION_LSRC_BATCH_SIZE="${VLM_SELECTION_LSRC_BATCH_SIZE:-2048}"

# Cost-effective default. Use qlora if LoRA still exceeds memory.
export VLM_FINETUNE_MODE="${VLM_FINETUNE_MODE:-lora}"
export VLM_NUM_TRAIN_EPOCHS="${VLM_NUM_TRAIN_EPOCHS:-1}"
export VLM_LEARNING_RATE="${VLM_LEARNING_RATE:-2e-4}"
export VLM_LORA_R="${VLM_LORA_R:-16}"
export VLM_LORA_ALPHA="${VLM_LORA_ALPHA:-32}"
export VLM_LORA_DROPOUT="${VLM_LORA_DROPOUT:-0.05}"
export VLM_PER_DEVICE_TRAIN_BATCH_SIZE="${VLM_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export VLM_PER_DEVICE_EVAL_BATCH_SIZE="${VLM_PER_DEVICE_EVAL_BATCH_SIZE:-1}"
export VLM_GRADIENT_ACCUMULATION_STEPS="${VLM_GRADIENT_ACCUMULATION_STEPS:-16}"
export VLM_MAX_LENGTH="${VLM_MAX_LENGTH:-2048}"
export VLM_NUM_WORKERS="${VLM_NUM_WORKERS:-2}"
export VLM_LOGGING_STEPS="${VLM_LOGGING_STEPS:-10}"
export VLM_EVAL_STEPS="${VLM_EVAL_STEPS:-100}"
export VLM_SAVE_STEPS="${VLM_SAVE_STEPS:-500}"
export VLM_SAVE_TOTAL_LIMIT="${VLM_SAVE_TOTAL_LIMIT:-2}"
export VLM_VAL_RATIO="${VLM_VAL_RATIO:-0.02}"
export VLM_REPORT_TO="${VLM_REPORT_TO:-none}"
export VLM_BF16="${VLM_BF16:-1}"

# Export merged LoRA model for VLMEvalKit when possible. If this fails, the eval
# plan still records the adapter path and the merge failure reason.
export VLM_MERGE_LORA_FOR_EVAL="${VLM_MERGE_LORA_FOR_EVAL:-1}"

# Benchmark plan. Dataset names may need adjustment for your installed VLMEvalKit.
export VLM_EVAL_BENCHMARKS="${VLM_EVAL_BENCHMARKS:-GQA,ScienceQA-IMG,MMBench,TextVQA,POPE}"
export VLM_VLMEVAL_MODEL_KEY="${VLM_VLMEVAL_MODEL_KEY:-qwen2vl_subset}"
export VLM_VLMEVAL_MODEL_CLASS="${VLM_VLMEVAL_MODEL_CLASS:-Qwen2VLChat}"

# Evaluation wrapper controls. By default we only generate VLMEvalKit commands and
# collect existing results. Set VLM_EVAL_EXECUTE=1 and VLMEVALKIT_ROOT to run eval.
export VLM_EVAL_PLAN_ROOT="${VLM_EVAL_PLAN_ROOT:-${VLM_FINETUNE_OUTPUT_ROOT}}"
export VLM_EVAL_EXECUTE="${VLM_EVAL_EXECUTE:-0}"
export VLM_EVAL_USE_TORCHRUN="${VLM_EVAL_USE_TORCHRUN:-0}"
export VLMEVAL_NPROC="${VLMEVAL_NPROC:-1}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

echo "[$(timestamp)] Qwen2-VL LLaVA subset full pipeline start"
echo "  model: ${VLM_MODEL_NAME_OR_PATH}"
echo "  annotation: ${LLAVA_ANNOTATION_PATH}"
echo "  image root: ${LLAVA_IMAGE_ROOT}"
echo "  output root: ${VLM_FINETUNE_OUTPUT_ROOT}"
echo "  ratios: ${VLM_SUBSET_RATIOS}"
echo "  dense_sift_bovw selection: ${VLM_RUN_DENSE_SIFT_BOVW_SELECTION}"
echo "  selection text feature: ${VLM_SELECTION_TEXT_REPR_METHOD}"
echo "  selection topology root: ${VLM_DENSE_SIFT_BOVW_TOPOLOGY_ROOT}"
echo "  selection cross-modal root: ${VLM_DENSE_SIFT_BOVW_CROSS_MODAL_ROOT}"
echo "  ours template: ${VLM_OURS_SELECTED_INDICES_TEMPLATE:-<auto>}"
echo "  finetune mode: ${VLM_FINETUNE_MODE}"
echo "  merge for eval: ${VLM_MERGE_LORA_FOR_EVAL}"
echo "  eval execute: ${VLM_EVAL_EXECUTE}"

if [[ ! -e "${VLM_MODEL_NAME_OR_PATH}" ]]; then
  echo "Model path not found: ${VLM_MODEL_NAME_OR_PATH}" >&2
  exit 1
fi
if [[ ! -f "${LLAVA_ANNOTATION_PATH}" ]]; then
  echo "LLaVA annotation file not found: ${LLAVA_ANNOTATION_PATH}" >&2
  exit 1
fi
if [[ ! -d "${LLAVA_IMAGE_ROOT}" ]]; then
  echo "LLaVA image root not found: ${LLAVA_IMAGE_ROOT}" >&2
  exit 1
fi

if [[ "${VLM_RUN_OURS}" == "1" && -z "${VLM_OURS_SELECTED_INDICES_TEMPLATE}" ]]; then
  export VLM_OURS_SELECTED_INDICES_TEMPLATE="${VLM_DENSE_SIFT_BOVW_SELECTION_ROOT}/{ratio_tag}/proxy_opt_lsrc/seed_${VLM_SEED}/selected_indices.json"
  if [[ "${VLM_RUN_DENSE_SIFT_BOVW_SELECTION}" == "1" ]]; then
    echo "[$(timestamp)] Stage 0/4: sample LLaVA Ours subsets with dense_sift_bovw"
    python -u "${PROJECT_ROOT}/run_llava_dense_sift_bovw_selection.py" \
      --annotation_path "${LLAVA_ANNOTATION_PATH}" \
      --image_root "${LLAVA_IMAGE_ROOT}" \
      --output_root "${VLM_DENSE_SIFT_BOVW_SELECTION_ROOT}" \
      --cache_dir "${VLM_DENSE_SIFT_BOVW_CACHE_DIR}" \
      --feature_cache_root "${VLM_DENSE_SIFT_BOVW_FEATURE_CACHE_ROOT}" \
      --topology_root "${VLM_DENSE_SIFT_BOVW_TOPOLOGY_ROOT}" \
      --cross_modal_root "${VLM_DENSE_SIFT_BOVW_CROSS_MODAL_ROOT}" \
      --pipeline_selection_output_root "${VLM_DENSE_SIFT_BOVW_PIPELINE_SELECTION_ROOT}" \
      --ratios "${VLM_SUBSET_RATIOS}" \
      --seed "${VLM_SEED}" \
      --device "${VLM_SELECTION_DEVICE}" \
      --text_repr_method "${VLM_SELECTION_TEXT_REPR_METHOD}" \
      --selection_text_batch_size "${VLM_SELECTION_TEXT_BATCH_SIZE}" \
      --image_metric "${VLM_SELECTION_IMAGE_METRIC}" \
      --text_metric "${VLM_SELECTION_TEXT_METRIC}" \
      --wavelet_fusion_weight_mode "${VLM_SELECTION_WAVELET_FUSION_WEIGHT_MODE}" \
      --proxy_num_steps "${VLM_SELECTION_PROXY_NUM_STEPS}" \
      --proxy_batch_size "${VLM_SELECTION_PROXY_BATCH_SIZE}" \
      --proxy_target_batch_size "${VLM_SELECTION_PROXY_TARGET_BATCH_SIZE}" \
      --lsrc_batch_size "${VLM_SELECTION_LSRC_BATCH_SIZE}"
  else
    echo "[$(timestamp)] Stage 0/4 skipped: VLM_RUN_DENSE_SIFT_BOVW_SELECTION=0"
    echo "  expecting Ours indices at: ${VLM_OURS_SELECTED_INDICES_TEMPLATE}"
  fi
elif [[ "${VLM_RUN_OURS}" == "1" ]]; then
  echo "[$(timestamp)] Stage 0/4 skipped: using provided Ours template"
  echo "  ${VLM_OURS_SELECTED_INDICES_TEMPLATE}"
fi

echo "[$(timestamp)] Stage 1/4: train Qwen2-VL on full/random/ours subsets"
bash "${SCRIPT_DIR}/run_qwen2vl_llava_subset_finetune.sh"

echo "[$(timestamp)] Stage 2/4: generate or execute VLMEvalKit evaluation commands"
bash "${SCRIPT_DIR}/run_vlmeval_qwen2vl_subset_eval.sh"

echo "[$(timestamp)] Stage 3/4: collect VLMEvalKit/lmms-eval results"
python "${PROJECT_ROOT}/tools/collect_vlmeval_results.py" \
  --plan_root "${VLM_EVAL_PLAN_ROOT}" \
  --output_csv "${VLM_EVAL_PLAN_ROOT}/reports/vlmevalkit_results_summary.csv" \
  --output_json "${VLM_EVAL_PLAN_ROOT}/reports/vlmevalkit_results_summary.json"

echo "[$(timestamp)] Qwen2-VL LLaVA subset full pipeline done"
echo "  train/eval root: ${VLM_FINETUNE_OUTPUT_ROOT}"
echo "  finetune table: ${VLM_FINETUNE_OUTPUT_ROOT}/reports/"
echo "  benchmark table: ${VLM_EVAL_PLAN_ROOT}/reports/vlmevalkit_results_summary.csv"
