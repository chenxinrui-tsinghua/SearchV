# [NeurIPS 2026] SearchV: Evolutionary Fine-Grained Visual-Token Skipping for Efficient Vision-Language Models

Official PyTorch implementation of **SearchV**.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)

This repository contains the official code for the NeurIPS 2026 paper **SearchV: Evolutionary Fine-Grained Visual-Token Skipping for Efficient Vision-Language Models**.

---

## TL;DR

**SearchV** is a **training-free** framework that accelerates vision–language models by searching **fine-grained visual-token skipping policies**.
It independently decides whether visual tokens participate in **MHA** and **FFN** at each decoder layer, guided by a **Global Contribution (GC)** fitness objective and budget-preserving evolutionary mutation:

- **Higher multi-benchmark retention** than FastV / VTW / ShortV at matched FLOPs
- **Hardware-friendly** contiguous K/V layout (compatible with standard KV cache / FlashAttention-style kernels)
- **Plug-and-play** on frozen LLaVA-family checkpoints (extendable to other VLMs)

---

## Contents

- [Preparations](#preparations)
- [Usage](#usage)
- [Results](#results)
- [Repository Layout](#repository-layout)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

---

## Preparations

### Installation

```bash
conda create -n searchv python=3.10 -y
conda activate searchv

cd /path/to/SearchV_release
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
cd lmms-eval && pip install -e . && cd ..
```

Optional logging:

```bash
pip install wandb   # then run search with LOG_WANDB=1
```

Use the **repository root** as the working directory for all commands below.

### Model Preparation

Download (or point to) Hugging Face checkpoints used in the paper:

| Model | Example Hugging Face ID / path |
| -------------------------------- | ------------------------------------------------------------ |
| LLaVA-1.5-7B | [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) |
| LLaVA-1.5-13B | [liuhaotian/llava-v1.5-13b](https://huggingface.co/liuhaotian/llava-v1.5-13b) |
| LLaVA-NeXT-7B | [liuhaotian/llava-v1.6-vicuna-7b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b) |
| LLaVA-NeXT-13B | [liuhaotian/llava-v1.6-vicuna-13b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-13b) |

Edit `MODEL=/path_to_model/...` inside `scripts/*.sh`, or export `MODEL=...` when launching.

### Data / Benchmarks

Evaluation and GC calibration use **[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)** task loaders (auto-download from Hugging Face on first run). Default calibration tasks:

| Task (lmms-eval name) | Role |
| --------------------- | ---- |
| `mmmu_val` | GC calibration / eval |
| `gqa` | GC calibration / eval |
| `flickr30k_test_cider` | GC calibration / eval |
| `seedbench_image` | GC calibration / eval |
| `mme_perception` | GC calibration / eval |
| `mmbench_en`, `mmstar`, … | Final reporting |

---

## Usage

### 1. Evolutionary search (GC fitness)

A typical run on **LLaVA-1.5-7B** (~44% FLOPs target via `NUM_SKIP`):

```bash
# Edit MODEL=... inside the script first
CUDA_VISIBLE_DEVICES=0 bash scripts/run_gc_fitness_search_1.5_7b.sh

# Optional W&B
LOG_WANDB=1 CUDA_VISIBLE_DEVICES=0 bash scripts/run_gc_fitness_search_1.5_7b.sh
```

Other wrappers:

```bash
bash scripts/run_gc_fitness_search_1.5_13b.sh
bash scripts/run_gc_fitness_search_1.6_7b.sh    # LLaVA-NeXT-7B
bash scripts/run_gc_fitness_search_1.6_13b.sh
```

Equivalent Python entrypoint:

```bash
python -m src.searchv_evolution \
  --model_name_or_path /path_to_model/llava-v1.5-7b \
  --num_skip_layers 19 \
  --arch_num_layers 32 \
  --arch_hidden_size 4096 \
  --arch_intermediate_size 11008 \
  --arch_text_tokens 64 \
  --arch_vision_tokens 576 \
  --fitness_fn gc \
  --gc_calib_tasks mmmu_val,gqa,flickr30k_test_cider,seedbench_image,mme_perception \
  --gc_calib_limit_per_task 64 \
  --population_size 3 \
  --offspring 32 \
  --generations 20 \
  --tournament_k 3 \
  --mutation_window 4 \
  --dtype float16 \
  --save_dir ./outputs \
  --task_name searchv_llava15_7b
```

Searched policies are written to `./outputs/layer_drop_config_*.txt` (two tokens per layer: MHA / FFN mode, e.g. `vision vision` or `None None`).

### 2. Evaluate a saved policy

```bash
SKIP_CONFIG=./outputs/layer_drop_config_....txt \
MODEL=/path_to_model/llava-v1.5-7b \
EVAL_TASKS=mme,mmbench_en,mmmu_val,mmstar,gqa,seedbench_image,flickr30k_test_cider \
bash scripts/eval_llava_1.5_7b.sh
```

Or directly:

```bash
python -m src.load_state_and_eval \
  --pretrained /path_to_model/llava-v1.5-7b \
  --skip-config ./outputs/layer_drop_config_....txt \
  --conv-template v1 \
  --tasks mme,mmbench_en,mmmu_val,mmstar,gqa,seedbench_image,flickr30k_test_cider
```

### 3. Inference micro-benchmark

```bash
SKIP_CONFIG=./outputs/layer_drop_config_....txt \
MODEL=/path_to_model/llava-v1.5-7b \
bash scripts/run_inference_speedups.sh
```

---

## Results

*(Numbers from the paper. **Per.** = average performance retention vs. Vanilla. Best efficiency method in each group is **bold**.)*

### LLaVA-1.5-7B (44% FLOPs)

| Method | TFLOPs | Ratio | MME | MMB | MMMU | MMStar | SEED | GQA | Flickr | Avg. | Per. |
| ---------------------- | ------ | ----- | ------ | ---- | ---- | ------ | ---- | ---- | ------ | ---- | ------- |
| Vanilla | 8.5 | 100% | 1510.7 | 64.1 | 36.3 | 33.7 | 66.1 | 61.9 | 74.9 | 58.9 | 100% |
| FastV (K=2, R=65%) | 3.8 | 45% | 1394.1 | 59.8 | 31.8 | **34.1** | 62.6 | 57.4 | **73.2** | 55.5 | 94.3% |
| VTW (K=12) | 3.7 | 44% | 1473.6 | **67.6** | **36.8** | 32.8 | 64.9 | 51.4 | 20.3 | 49.6 | 84.3% |
| ShortV (N=24) | 3.7 | 44% | 1341.7 | 60.7 | 34.1 | 33.4 | 62.5 | 58.3 | 64.2 | 54.3 | 92.2% |
| **SearchV** | 3.7 | 44% | **1502.7** | 64.7 | 36.3 | 33.7 | **65.6** | **59.1** | 64.6 | **57.0** | **96.8%** |

### LLaVA-1.5-13B (40% FLOPs)

| Method | TFLOPs | Ratio | MME | MMB | MMMU | MMStar | SEED | GQA | Flickr | Avg. | Per. |
| ---------------------- | ------ | ----- | ------ | ---- | ---- | ------ | ---- | ---- | ------ | ---- | ------- |
| Vanilla | 16.6 | 100% | 1531.3 | 68.9 | 35.4 | 36.1 | 68.2 | 63.3 | 79.6 | 61.2 | 100% |
| FastV (K=2, R=70%) | 6.6 | 40% | 1364.3 | 53.7 | 30.3 | 34.7 | 63.3 | 58.8 | **76.9** | 55.1 | 90.1% |
| VTW (K=14) | 6.9 | 42% | **1538.5** | 68.1 | 34.4 | 35.8 | **67.6** | 53.8 | 36.4 | 53.3 | 87.1% |
| ShortV (N=32) | 6.6 | 40% | 1298.8 | 64.5 | 33.6 | 36.0 | 63.2 | **59.3** | 68.4 | 55.7 | 91.0% |
| **SearchV** | 6.6 | 40% | 1480.1 | **68.0** | **36.0** | **36.7** | 67.5 | 61.3 | 72.4 | **59.4** | **97.1%** |

### LLaVA-NeXT-7B (37% FLOPs)

| Method | TFLOPs | Ratio | Avg. | Per. |
| ---------------------- | ------ | ----- | ---- | ------- |
| Vanilla | 42.7 | 100% | 60.1 | 100% |
| FastV | 16.2 | 38% | 55.1 | 91.6% |
| VTW | 16.5 | 39% | 48.7 | 81.0% |
| ShortV | 16.0 | 37% | 54.0 | 89.8% |
| **SearchV** | 16.0 | 37% | **58.5** | **97.3%** |

Additional backbones in the paper: **Qwen2.5-VL-7B** (96.3% retention @ 49%), **Gemma-3-4B-it** (96.9% @ 48%), **InternVL3-8B** (94.0% @ 50%).

---

## Repository Layout

```
SearchV_release/
├── src/                      # SearchV core
│   ├── searchv_evolution.py  # GC-guided evolutionary search
│   ├── load_state_and_eval.py
│   ├── mutate.py             # Window Shuffle / Submodule Swap
│   ├── vision_skip_flops.py  # FLOPs accounting
│   ├── metrics.py            # GC (last-prompt-token KL)
│   └── inference_speedup.py
├── llava/                    # Minimal LLaVA model (with MHA/FFN vision skip)
├── lmms-eval/                # Vendored lmms-eval (+ SearchV eval hooks)
├── scripts/                  # Example search / eval / speedup wrappers
├── requirements.txt
└── pyproject.toml
```

Skip-config format (one layer per line):

```
None None
vision vision
vision None
None vision
...
```

---

## Acknowledgments

This release builds on **[LLaVA](https://github.com/haotian-liu/LLaVA)** and **[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)**. Third-party code retains its original licenses where applicable.

---

## Citation

If you find SearchV helpful, please cite:

```bibtex
@inproceedings{searchv2026,
  title     = {SearchV: Evolutionary Fine-Grained Visual-Token Skipping for Efficient Vision-Language Models},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
