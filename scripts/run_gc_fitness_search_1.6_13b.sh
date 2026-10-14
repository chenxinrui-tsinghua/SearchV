#!/usr/bin/env bash
# GC fitness search, LLaVA-1.6 Vicuna-13B (layout matches 1.6_7b).
# Arch sizes: see HF config under your checkpoint (e.g. 40 layers / 5120 hidden / 13824 FFN).
# Run from repo root: bash scripts/run_gc_fitness_search_1.6_13b.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

MODEL="/path_to_model/llava-v1.6-vicuna-13b"

# Vicuna-13B: 40 layers / 5120 hidden / 13824 FFN (HF config)
ARCH_NUM_LAYERS=40
ARCH_HIDDEN_SIZE=5120
ARCH_INTERMEDIATE_SIZE=13824
ARCH_TEXT_TOKENS=64
# 1.6 anyres: vision token count v is FLOPs approx; tune ARCH_VISION_TOKENS for your resolution
ARCH_VISION_TOKENS=2880

NUM_SKIP=28

CALI_SET="mmmu_val,gqa,flickr30k_test_cider,seedbench_image,mme"
NUM_CALI_SET=5

GC_LIMIT_PER_TASK=64
GC_CONV_TEMPLATE="v1"

SAMPLES_STAGE2=$((GC_LIMIT_PER_TASK * NUM_CALI_SET))
INITIAL_SAMPLES=$SAMPLES_STAGE2
SAMPLES_STAGE1=$((SAMPLES_STAGE2 / 2))

IFS=',' read -r -a _CALI_TASKS_ARR <<< "$CALI_SET"
if [[ "${#_CALI_TASKS_ARR[@]}" -ne "$NUM_CALI_SET" ]]; then
  echo "[Warn] NUM_CALI_SET=$NUM_CALI_SET != parsed task count ${#_CALI_TASKS_ARR[@]}; check CALI_SET" >&2
fi

GENERATIONS=200
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/outputs}"
NAME="n=28_gc_fitness_1.6_13b_0410"

MODEL_BASENAME="${MODEL##*/}"
EVAL_SKIP_CONFIG="${SAVE_DIR}/layer_drop_config_${MODEL_BASENAME}_nskip${NUM_SKIP}_${NAME}.txt"
EVAL_TASKS="mme,mmbench_en_dev,mmbench_en,mmmu_val,mmstar,gqa,seedbench_image,flickr30k_test_cider"

SEARCHV_RANKING="39 32 29 36 27 30 37 23 25 31 26 2 28 22 33 35 34 24 38 21 20 18 1 17 5 4 16 19 6 15 13 14 3 7 8 10 11 9 12 0"

POP_SIZE=3
OFFSPRING=32
TOURNAMENT_K=3
MUTATION_WINDOW=4

echo "Running GC fitness_search NUM_SKIP=${NUM_SKIP} POP=$POP_SIZE GEN=$GENERATIONS model=${MODEL_BASENAME}"

if [[ -n "${SEARCHV_RANKING:-}" ]]; then
  SEARCHV_ARGS=(--searchv_importance_ranking ${SEARCHV_RANKING})
else
  SEARCHV_ARGS=()
fi

python -m src.searchv_evolution \
    --model_name_or_path "$MODEL" \
    --num_skip_layers "$NUM_SKIP" \
    --arch_num_layers "$ARCH_NUM_LAYERS" \
    --arch_hidden_size "$ARCH_HIDDEN_SIZE" \
    --arch_intermediate_size "$ARCH_INTERMEDIATE_SIZE" \
    --arch_text_tokens "$ARCH_TEXT_TOKENS" \
    --arch_vision_tokens "$ARCH_VISION_TOKENS" \
    --eval_every 1 \
    --eval_datasets wikitext2 \
    --population_size "$POP_SIZE" \
    --generations "$GENERATIONS" \
    --offspring "$OFFSPRING" \
    --initially_generated 16 \
    --initial_samples "$INITIAL_SAMPLES" \
    --survivors_per_selection 8 "$POP_SIZE" \
    --samples_per_selection "$SAMPLES_STAGE1" "$SAMPLES_STAGE2" \
    --early_stop 3 \
    --no_eval \
    --protect_initial 0 \
    --protect_final 0 \
    --fitness_fn gc \
    --gc_calib_tasks "$CALI_SET" \
    --gc_calib_limit_per_task "$GC_LIMIT_PER_TASK" \
    --gc_conv_template "$GC_CONV_TEMPLATE" \
    --tournament_k "$TOURNAMENT_K" \
    --mutation_window "$MUTATION_WINDOW" \
    --use_fast_tokenizer \
    --dtype bfloat16 \
    ${LOG_WANDB:+--log_wandb} \
    --save_dir "$SAVE_DIR" \
    --task_name "$NAME" \
    "${SEARCHV_ARGS[@]}"


if [[ -f "$EVAL_SKIP_CONFIG" ]]; then
  python -m src.load_state_and_eval \
    --pretrained "$MODEL" \
    --skip-config "$EVAL_SKIP_CONFIG" \
    --conv-template "$GC_CONV_TEMPLATE" \
    --tasks "$EVAL_TASKS"
else
  echo "[Warn] Skip config not found, skipping eval: $EVAL_SKIP_CONFIG"
fi
