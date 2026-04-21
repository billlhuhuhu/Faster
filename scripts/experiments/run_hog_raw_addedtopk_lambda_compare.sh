#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Diagnostic control: keep the original HOG+color image graph feature, and only
# test the stage-2 correction-strength / added-edge-limit change.
export WAVELET_MAIN_LATEST_DATASET="${HOG_RAW_ADDEDTOPK_DATASET:-flickr}"
export WAVELET_MAIN_LATEST_BACKBONE="${HOG_RAW_ADDEDTOPK_BACKBONE:-nfnet}"
export WAVELET_MAIN_LATEST_TEXT_ENCODER="${HOG_RAW_ADDEDTOPK_TEXT_ENCODER:-bert}"
export WAVELET_MAIN_LATEST_VARIANT="${HOG_RAW_ADDEDTOPK_VARIANT:-wavelet_main_hog_color_raw_addedtopk5_lambda03}"
export WAVELET_MAIN_LATEST_BUDGETS="${HOG_RAW_ADDEDTOPK_BUDGETS:-100 200 500}"
export WAVELET_MAIN_LATEST_RATIOS="${HOG_RAW_ADDEDTOPK_RATIOS-0.01 0.02 0.03}"
export WAVELET_MAIN_LATEST_SEEDS="${HOG_RAW_ADDEDTOPK_SEEDS:-0}"

export FEATURE_CACHE_ROOT="${HOG_RAW_ADDEDTOPK_FEATURE_CACHE_ROOT:-artifacts/feature_cache_hog_color_raw_addedtopk5_lambda03}"
export TOPOLOGY_ROOT="${HOG_RAW_ADDEDTOPK_TOPOLOGY_ROOT:-artifacts/topology_graph_hog_color_raw_addedtopk5_lambda03}"
export WAVELET_MAIN_LATEST_CROSS_OUTPUT_ROOT="${HOG_RAW_ADDEDTOPK_CROSS_OUTPUT_ROOT:-artifacts/cross_modal_topology_hog_color_raw_addedtopk5_lambda03}"
export WAVELET_MAIN_LATEST_SELECTION_OUTPUT_ROOT="${HOG_RAW_ADDEDTOPK_SELECTION_OUTPUT_ROOT:-artifacts/subset_selection_hog_color_raw_addedtopk5_lambda03}"
export WAVELET_MAIN_LATEST_TRAIN_OUTPUT_ROOT="${HOG_RAW_ADDEDTOPK_TRAIN_OUTPUT_ROOT:-artifacts/subset_train_hog_color_raw_addedtopk5_lambda03}"
export WAVELET_MAIN_LATEST_REPORT_NAME="${HOG_RAW_ADDEDTOPK_REPORT_NAME:-hog_color_raw_addedtopk5_lambda03}"

# Use raw HOG+color, not the newer Hellinger-modified hog_color mode.
export SELECTION_IMAGE_REPR_METHOD="${HOG_RAW_ADDEDTOPK_IMAGE_REPR_METHOD:-hog_color_raw}"

# Keep current text-side default unless explicitly overridden.
export SELECTION_TEXT_REPR_METHOD="${HOG_RAW_ADDEDTOPK_TEXT_REPR_METHOD:-${SELECTION_TEXT_REPR_METHOD:-bert}}"

# The two stage-2 changes under test.
export ASYMMETRIC_CORRECTION_LAMBDA="${HOG_RAW_ADDEDTOPK_ASYMMETRIC_CORRECTION_LAMBDA:-0.3}"
export CORRECTED_IMAGE_ADDED_TOPK="${HOG_RAW_ADDEDTOPK_CORRECTED_IMAGE_ADDED_TOPK:-5}"

exec bash "${SCRIPT_DIR}/run_wavelet_main_latest_combo.sh"
