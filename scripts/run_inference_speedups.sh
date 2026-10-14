#!/usr/bin/env bash
# Run inference_speedup: --verify-only (geometry) then timed benchmark.
# Run from repo root: python -m src.inference_speedup ...
#
# Override examples:
#   SKIP_CONFIG=/path/to/layer_drop_config.txt MODEL=/path/to/llava-v1.6-vicuna-13b bash scripts/run_inference_speedups.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="${MODEL:-/path_to_model/llava-v1.6-vicuna-13b}"
SKIP_CONFIG="${SKIP_CONFIG:-}"
CONV_TEMPLATE="${CONV_TEMPLATE:-v1}"
TEXT_TOKENS="${TEXT_TOKENS:-64}"
RUNS="${RUNS:-20}"
WARMUP="${WARMUP:-3}"
DTYPE="${DTYPE:-float16}"

if [[ -z "$SKIP_CONFIG" ]]; then
  echo "[Error] Set SKIP_CONFIG to your layer_drop_config_*.txt path, e.g.:" >&2
  echo "  SKIP_CONFIG=\$REPO_ROOT/outputs/layer_drop_config_....txt $0" >&2
  exit 1
fi
if [[ ! -f "$SKIP_CONFIG" ]]; then
  echo "[Error] skip config not found: $SKIP_CONFIG" >&2
  exit 1
fi

echo "[Info] MODEL=$MODEL"
echo "[Info] SKIP_CONFIG=$SKIP_CONFIG"
echo "[Info] TEXT_TOKENS=$TEXT_TOKENS RUNS=$RUNS WARMUP=$WARMUP DTYPE=$DTYPE CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

PY=(python -m src.inference_speedup
  --pretrained "$MODEL"
  --skip-config "$SKIP_CONFIG"
  --conv-template "$CONV_TEMPLATE"
  --text-tokens "$TEXT_TOKENS"
  --runs "$RUNS"
  --warmup "$WARMUP"
  --dtype "$DTYPE"
)

echo "========== verify (geometry) =========="
"${PY[@]}" --verify-only

echo "========== benchmark =========="
"${PY[@]}"
