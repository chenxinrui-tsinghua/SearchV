#!/usr/bin/env bash
# Evaluate a LLaVA-NeXT-13B (llava-v1.6) checkpoint with an optional SearchV skip config.
# See scripts/eval_llava_1.5_7b.sh for mode details.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="${MODEL:-/path_to_model/llava-v1.6-vicuna-13b}"
SKIP_CONFIG="${SKIP_CONFIG:-}"
LAYER_IDX="${LAYER_IDX:-}"
NUM_LAYERS="${NUM_LAYERS:-40}"
CONV_TEMPLATE="${CONV_TEMPLATE:-v1}"
EVAL_TASKS="${EVAL_TASKS:-mme,mmbench_en,mmmu_val,mmstar,gqa,seedbench_image,flickr30k_test_cider}"

if [[ -n "${LAYER_IDX}" && -n "${SKIP_CONFIG}" ]]; then
  echo "[Error] LAYER_IDX and SKIP_CONFIG are mutually exclusive." >&2
  exit 1
fi

if [[ -n "${SKIP_CONFIG}" && ! -f "$SKIP_CONFIG" ]]; then
  echo "[Error] skip config file not found: $SKIP_CONFIG" >&2
  exit 1
fi

echo "[Info] MODEL=$MODEL"
if [[ -n "${LAYER_IDX}" ]]; then
  echo "[Info] mode=layer_idx LAYER_IDX=$LAYER_IDX NUM_LAYERS=$NUM_LAYERS"
else
  echo "[Info] mode=skip-config SKIP_CONFIG=${SKIP_CONFIG:-<empty, dense model>}"
fi
echo "[Info] CONV_TEMPLATE=$CONV_TEMPLATE EVAL_TASKS=$EVAL_TASKS"

PY_ARGS=(
  --pretrained "$MODEL"
  --conv-template "$CONV_TEMPLATE"
  --tasks "$EVAL_TASKS"
)
if [[ -n "${LAYER_IDX}" ]]; then
  PY_ARGS+=(--layer_idx "$LAYER_IDX" --num_layers "$NUM_LAYERS")
elif [[ -n "${SKIP_CONFIG}" ]]; then
  PY_ARGS+=(--skip-config "$SKIP_CONFIG")
fi

python -m src.load_state_and_eval "${PY_ARGS[@]}"
