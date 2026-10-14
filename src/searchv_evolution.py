import argparse
import json
import random
import os
import sys
import hashlib
import numpy as np
from tqdm import trange
from typing import List, Optional
from tqdm import tqdm

import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention
# from transformers import AutoProcessor, LlavaForConditionalGeneration, TextStreamer
from llava.model.builder import load_pretrained_llava

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False

from src.model_utils import get_layers
from src.metrics import compute_gc_div
from src.load_state_and_eval import (
    simple_eval,
    extract_mme_cognition_score,
    extract_mme_perception_score,
    MME_COGNITION_TASK,
    MME_PERCEPTION_TASK,
    weighted_multitask_score,
    TASK_PRIMARY_METRIC_KEY,
    normalize_fitness_fn_string,
    is_multitask_fitness_fn_string,
    parse_multitask_task_list,
    build_gc_searchv_calibration_batches,
)
from lmms_eval.models.llava import Llava
import src.vision_skip_flops as _gs
from src.vision_skip_flops import (
    generate_parent_configurations,
    get_flops_saved_matrix,
    set_architecture,
    sparsity_from_num_skip_vision_vision,
)
from src.mutate import generate_offspring, TabuCache, config_flops_saved
import time



def _config_hash(config):
    """Hash layer config for fitness cache keys.

    Keys include num_samples and (for lmms) subset_shuffle_seed. Coarse-to-fine schedules
    use different num_samples so survivors are re-scored; hits mean same config, budget,
    and shuffle under deterministic eval. Use --no_fitness_cache if eval is non-deterministic.
    """
    raw = "|".join(f"{m},{f}" for m, f in config)
    return hashlib.md5(raw.encode()).hexdigest()


# Single-task lmms fitness names (see load_state_and_eval.TASK_PRIMARY_METRIC_KEY)
LMMS_SINGLE_EVAL_FNS = frozenset(
    {"gqa", "flickr30k_test_cider", "mme_cognition", "mme_perception"}
)


def is_lmms_eval_fitness(fitness_fn: str) -> bool:
    """True if fitness uses Llava + lmms-eval (single task, comma multitask, or alias)."""
    if fitness_fn == "gc":
        return False
    if is_multitask_fitness_fn_string(fitness_fn):
        return True
    return normalize_fitness_fn_string(fitness_fn) in LMMS_SINGLE_EVAL_FNS


def _extract_flickr_cider_from_results(results: dict) -> float:
    """Extract flickr_CIDEr from simple_evaluate results for flickr30k_test_cider."""
    sub = results["results"]["flickr30k_test_cider"]
    for k, v in sub.items():
        if "_stderr" in k or k in ("alias", "samples"):
            continue
        if "CIDEr" in k and isinstance(v, (int, float)):
            return float(v)
    raise KeyError(f"flickr_CIDEr metric not found; keys: {list(sub.keys())}")


def _compute_flops_stats(layer_drop_config, target_percent: float):
    """Compute target/actual FLOPs saved stats for the final config."""
    _, base_tflops, _ = get_flops_saved_matrix(
        _gs.L, _gs.h, _gs.m, _gs.t, _gs.v
    )
    target_saved = base_tflops * target_percent
    actual_saved = config_flops_saved(layer_drop_config)
    delta_saved = actual_saved - target_saved
    return {
        "base_tflops": base_tflops,
        "target_saved_tflops": target_saved,
        "actual_saved_tflops": actual_saved,
        "delta_saved_tflops": delta_saved,
        "target_saved_percent": target_percent * 100.0,
        "actual_saved_percent": (actual_saved / base_tflops) * 100.0,
        "delta_saved_percent": (delta_saved / base_tflops) * 100.0,
    }


def _layer_drop_config_file_prefix(
    model_name: str, task_name: str, sparsity: float, initial_num_skip: Optional[int]
) -> str:
    """Filename prefix aligned with run_gc_fitness_search*.sh EVAL_SKIP_CONFIG (nskip avoids floats)."""
    if initial_num_skip is not None:
        return f"layer_drop_config_{model_name}_nskip{initial_num_skip}_{task_name}"
    return f"layer_drop_config_{model_name}_sparsity{sparsity}_{task_name}"


def _move_forward_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move Llava forward batch dict to device; ``images`` may be Tensor or list (like process_images)."""
    out = {}
    for key, value in batch.items():
        if value is None:
            out[key] = None
        elif isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        elif isinstance(value, list):
            out[key] = [
                x.to(device) if isinstance(x, torch.Tensor) else x for x in value
            ]
        else:
            out[key] = value
    return out


def compute_fitness(
    model,
    data,
    fitness_fn,
    invert_fitness,
    target_logits: Optional[torch.Tensor] = None,
    eval_wrapper=None,
    num_samples=None,
    multitask_weights: Optional[List[float]] = None,
    subset_shuffle_seed: Optional[int] = 42,
) -> float:
    sign = 1
    if invert_fitness:
        sign = -1

    fn_norm = normalize_fitness_fn_string(fitness_fn)

    if is_multitask_fitness_fn_string(fitness_fn):
        assert eval_wrapper is not None, "Multitask fitness requires eval_wrapper (Llava instance)"
        tasks = parse_multitask_task_list(fitness_fn)
        assert multitask_weights is not None and len(multitask_weights) == len(tasks), (
            f"multitask_weights length must match tasks={tasks}"
        )
        results = simple_eval(
            eval_model=eval_wrapper,
            task_list=tasks,
            num_samples=num_samples,
            verbose=False,
            hf_model=model,
            subset_shuffle_seed=subset_shuffle_seed,
        )
        score = weighted_multitask_score(results, tasks, multitask_weights)
        return sign * score

    if fn_norm == "gqa":
        assert eval_wrapper is not None, "GQA fitness requires eval_wrapper (Llava instance)"

        results = simple_eval(
            eval_model=eval_wrapper,
            task_list=["gqa"],
            num_samples=num_samples,
            verbose=False,
            hf_model=model,
            subset_shuffle_seed=subset_shuffle_seed,
        )

        gqa_accuracy = results["results"]["gqa"]["exact_match,none"]
        return sign * gqa_accuracy
    if fn_norm == "flickr30k_test_cider":
        assert eval_wrapper is not None, "flickr30k_test_cider fitness requires eval_wrapper (Llava instance)"

        results = simple_eval(
            eval_model=eval_wrapper,
            task_list=["flickr30k_test_cider"],
            num_samples=num_samples,
            verbose=False,
            hf_model=model,
            subset_shuffle_seed=subset_shuffle_seed,
        )
        cider = _extract_flickr_cider_from_results(results)
        return sign * cider
    if fn_norm == "mme_cognition":
        assert eval_wrapper is not None, "mme_cognition fitness requires eval_wrapper (Llava instance)"
        results = simple_eval(
            eval_model=eval_wrapper,
            task_list=[MME_COGNITION_TASK],
            num_samples=num_samples,
            verbose=False,
            hf_model=model,
            subset_shuffle_seed=subset_shuffle_seed,
        )
        score = extract_mme_cognition_score(results)
        return sign * score
    if fn_norm == "mme_perception":
        assert eval_wrapper is not None, "mme_perception fitness requires eval_wrapper (Llava instance)"
        results = simple_eval(
            eval_model=eval_wrapper,
            task_list=[MME_PERCEPTION_TASK],
            num_samples=num_samples,
            verbose=False,
            hf_model=model,
            subset_shuffle_seed=subset_shuffle_seed,
        )
        score = extract_mme_perception_score(results)
        return sign * score
    if fn_norm == "gc":
        return sign * compute_gc_div(model, data, target_logits)

    raise ValueError(f"Unknown fitness_fn: {fitness_fn!r} (normalized {fn_norm!r})")


def selection(
    model,
    candidates,
    num_survive: int,
    calibration_data,
    num_samples: int,
    invert_fitness: bool,
    fitness_fn: str = "gc",
    target_logits: Optional[List[torch.Tensor]] = None,
    eval_wrapper=None,
    fitness_cache=None,
    multitask_weights: Optional[List[float]] = None,
    subset_shuffle_seed: Optional[int] = 42,
):
    calibration_minibatch = []
    minibatch_ids = []
    target_logits_minibatch = []
    samples_used = 0

    if not is_lmms_eval_fitness(fitness_fn):
        n_calib = len(calibration_data)
        cap = min(int(num_samples), n_calib) if n_calib > 0 else 0
        while samples_used < cap:
            minibatch_id = random.randint(0, len(calibration_data) - 1)
            if minibatch_id in minibatch_ids:
                continue
            calibration_minibatch.append(calibration_data[minibatch_id])
            target_logits_minibatch.append(target_logits[minibatch_id])
            minibatch_ids.append(minibatch_id)
            samples_used += 1

    fitnesses = []
    desc = f"Evaluating {len(candidates)} candidates"
    fn_norm = normalize_fitness_fn_string(fitness_fn)
    is_multi = is_multitask_fitness_fn_string(fitness_fn)

    pbar = tqdm(enumerate(candidates), total=len(candidates), desc=desc, leave=False, file=sys.stdout)
    cache_hits = 0
    cache_misses = 0
    for idx, candidate in pbar:
        # Must match compute_fitness(..., num_samples=...); stage changes must change cache key
        cache_key = (_config_hash(candidate), fitness_fn, num_samples)
        if multitask_weights is not None:
            cache_key = cache_key + (tuple(round(w, 9) for w in multitask_weights),)
        if is_lmms_eval_fitness(fitness_fn):
            cache_key = cache_key + (subset_shuffle_seed,)

        if fitness_cache is not None and cache_key in fitness_cache:
            fitness = fitness_cache[cache_key]
            cache_hits += 1
        else:
            model.model.update_skip_configs(candidate)
            fitness = compute_fitness(
                model,
                calibration_minibatch,
                fitness_fn,
                invert_fitness,
                target_logits_minibatch,
                eval_wrapper=eval_wrapper,
                num_samples=num_samples,
                multitask_weights=multitask_weights,
                subset_shuffle_seed=subset_shuffle_seed,
            )
            if fitness_cache is not None:
                fitness_cache[cache_key] = fitness
            cache_misses += 1
        fitnesses.append(fitness)

        # Multitask strings may contain "gqa"; use is_multi + fn_norm to avoid mis-labeling
        if not is_multi and fn_norm == "gqa":
            acc = -fitness if invert_fitness else fitness
            best_acc = max(-f if invert_fitness else f for f in fitnesses)
            pbar.set_postfix({'Acc': f'{acc*100:.2f}%', 'Best': f'{best_acc*100:.2f}%', 'Hit': cache_hits, 'Miss': cache_misses})
        elif not is_multi and fn_norm == "flickr30k_test_cider":
            # flickr fitness is CIDEr (not GQA acc); invert_fitness flips sign
            cur = -fitness if invert_fitness else fitness
            best_c = max(-f if invert_fitness else f for f in fitnesses)
            pbar.set_postfix(
                {"CIDEr": f"{cur:.4f}", "BestCIDEr": f"{best_c:.4f}", "Hit": cache_hits, "Miss": cache_misses}
            )
        elif not is_multi and fn_norm == "mme_cognition":
            cur = -fitness if invert_fitness else fitness
            best_s = max(-f if invert_fitness else f for f in fitnesses)
            pbar.set_postfix(
                {"MMEcg": f"{cur:.2f}", "BestMMEcg": f"{best_s:.2f}", "Hit": cache_hits, "Miss": cache_misses}
            )
        elif not is_multi and fn_norm == "mme_perception":
            cur = -fitness if invert_fitness else fitness
            best_s = max(-f if invert_fitness else f for f in fitnesses)
            pbar.set_postfix(
                {"MMEpx": f"{cur:.2f}", "BestMMEpx": f"{best_s:.2f}", "Hit": cache_hits, "Miss": cache_misses}
            )
        elif is_multi:
            cur = -fitness if invert_fitness else fitness
            best_s = max(-f if invert_fitness else f for f in fitnesses)
            pbar.set_postfix(
                {"Wmulti": f"{cur:.4f}", "BestWm": f"{best_s:.4f}", "Hit": cache_hits, "Miss": cache_misses}
            )
        else:
            pbar.set_postfix({'Fitness': f'{fitness:.3f}', 'Best': f'{min(fitnesses):.3f}', 'Hit': cache_hits, 'Miss': cache_misses})

    pbar.close()

    best_ids = np.argsort(fitnesses)[:num_survive]
    return [candidates[i] for i in best_ids], [fitnesses[i] for i in best_ids]


def parse_args():
    def samples_type(value):
        """Parse sample count: int or 'all'."""
        if value.lower() == 'all':
            return 'all'
        return int(value)
    
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path to the model being pruned",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="The name or path to the tokenizer. By default use model tokenizer.",
    )
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        type=str,
        default=["fineweb_edu", "wikitext2", "c4"],
        help="Datasets used for evaluation",
    )
    parser.add_argument("--no_eval", action="store_true", help="Whether to skip evaluation")
    parser.add_argument("--eval_every", default=1, type=int, help="Eval every # generations.")
    # Sparsification: either target fraction of BASE_TFLOPs saved, or N (Vision,Vision) layers -> converted
    sp = parser.add_mutually_exclusive_group(required=True)
    sp.add_argument(
        "--sparsity",
        type=float,
        default=None,
        help="Target fraction (0..1) of BASE_TFLOPs to save; passed as target_percent to generate_parent_configurations.",
    )
    sp.add_argument(
        "--num_skip_layers",
        type=int,
        default=None,
        help="N layers with (Vision,Vision) in initial parents (SearchV ranking path); converted to --sparsity via FLOPs matrix.",
    )
    # Arch knobs for FLOPs (defaults match LLaVA-7B in vision_skip_flops; override in shell for other models)
    parser.add_argument(
        "--arch_num_layers",
        type=int,
        default=None,
        help="Transformer depth L (default 32).",
    )
    parser.add_argument(
        "--arch_hidden_size",
        type=int,
        default=None,
        help="Hidden size h (default 4096).",
    )
    parser.add_argument(
        "--arch_intermediate_size",
        type=int,
        default=None,
        help="FFN intermediate size m (default 11008).",
    )
    parser.add_argument(
        "--arch_text_tokens",
        type=int,
        default=None,
        help="Text token count t for FLOPs model (default 64).",
    )
    parser.add_argument(
        "--arch_vision_tokens",
        type=int,
        default=None,
        help="Vision token count v for FLOPs (default 576; raise for 1.6 anyres approx).",
    )
    # Logging params
    parser.add_argument("--log_wandb", default=False, action="store_true", help="Whether to log to W&B")
    # Evolutionary Search params
    parser.add_argument("--task_name", required=True, type=str, help="Task name for saving config")
    parser.add_argument("--protect_initial", required=True, type=int, help="Number of protected initial layers")
    parser.add_argument("--protect_final", required=True, type=int, help="Number of protected final layers")
    parser.add_argument(
        "--fitness_fn",
        type=str,
        default="gc",
        help="Single: gqa | flickr30k_test_cider | mme_cognition | mme_perception | gc. "
        "gc=Global Contribution (SearchV / LC calibration): last-token KL KLDivLoss(log_softmax(full), softmax(skip)); "
        "full=no-skip ref (--gc_calib_*). Multitask: comma list, e.g. mmstar,mmmu_val,gqa "
        "(order matches --fitness_multi_weights); alias weighted_mmstar_mmmu_gqa.",
    )
    parser.add_argument(
        "--gc_calib_limit_per_task",
        type=int,
        default=20,
        help="For gc: docs per task from lmms-eval (LC calibration often uses --limit 20).",
    )
    parser.add_argument(
        "--gc_calib_tasks",
        type=str,
        default="gqa,flickr30k_test",
        help="For gc: comma-separated lmms-eval task names for build_gc_searchv_calibration_batches, "
        "e.g. mmmu_val,gqa,flickr30k_test_cider.",
    )
    parser.add_argument(
        "--gc_conv_template",
        type=str,
        default="v1",
        help="Llava conv_template key for GC calib batches (e.g. v1).",
    )
    parser.add_argument(
        "--fitness_multi_weights",
        type=float,
        nargs="*",
        default=None,
        help="Comma multitask only: per-task weights (normalized internally); omit for uniform.",
    )
    parser.add_argument("--generations", required=True, type=int, help="Number of generations in evolutionary search")
    parser.add_argument("--early_stop", required=True, type=int, help="Number of generations to early stop")
    parser.add_argument("--offspring", type=int, required=True, help="Number of offspring generated in each generation")
    parser.add_argument("--population_size", type=int, default=1, help="Population size in evolutionary search")
    parser.add_argument(
        "--initially_generated",
        type=int,
        required=True,
        help="Number of search points generated in the beginning; fittest are selected for the initial population",
    )
    parser.add_argument(
        "--initial_samples",
        type=int,
        required=True,
        help="Number of calibration samples used for the initial generation",
    )
    parser.add_argument(
        "--survivors_per_selection",
        type=int,
        nargs="+",
        required=True,
        help="Number of survivors after each stage of selection",
    )
    parser.add_argument(
        "--samples_per_selection",
        type=samples_type,
        nargs="+",
        required=True,
        help="Number of calibration samples at each stage of selection (can use 'all' for all samples)",
    )
    # Evolutionary Search ablation params
    parser.add_argument(
        "--invert_fitness", action="store_true", help="Whether to invert the fitness function (search for worst)"
    )
    parser.add_argument("--max_mutations", type=int, default=3, help="Maximum number of mutations in offspring")
    parser.add_argument("--tournament_k", type=int, default=3, help="Tournament size for parent selection")
    parser.add_argument("--mutation_window", type=int, default=4, help="Number of adjacent layers to shuffle in mutate (n)")
    parser.add_argument(
        "--legal_to_drop_path",
        type=str,
        default=None,
        help="Path to legal_to_drop file. A block can only be dropped if it is dropped in legal_to_drop configuration.",
    )
    parser.add_argument("--drop_entire_block", action="store_true", help="Whether to drop entire block (attn+mlp).")
    parser.add_argument(
        "--drop_two_consecutive",
        action="store_true",
        help="Only drop pairs of consecutive blocks (first and second, third and fourth,...). Can only be set when entire blocks are dropped.",
    )
    # Misc params
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="dtype to load the model.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation: eager, sdpa, or flash_attention_2",
    )
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Whether to use fast tokenizer.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument(
        "--fitness_subset_seed",
        type=int,
        default=42,
        help="Seed for deterministic shuffle before lmms subsample (independent of weights).",
    )
    parser.add_argument(
        "--no_fitness_subset_shuffle",
        action="store_true",
        help="Disable shuffle; use lmms default order (first limit docs).",
    )
    # Save params
    parser.add_argument("--save_dir", type=str, help="Where to save configs.")
    parser.add_argument(
        "--searchv_importance_ranking",
        type=int,
        nargs="+",
        default=None,
        help="Layer importance ranking for warm start (least important first). E.g. 31 29 30 28 ...",
    )
    parser.add_argument(
        "--no_fitness_cache",
        action="store_true",
        help="Disable fitness cache (slower; use if eval is non-deterministic or for debug).",
    )
    parser.add_argument(
        "--no_tabu",
        action="store_true",
        help="Disable TabuCache (allow duplicate configs; default Tabu on).",
    )

    args = parser.parse_args()
    return args


def apply_fitness_fn_cli_args(args):
    """Validate fitness_fn; normalize multitask weights."""
    fn = (args.fitness_fn or "").strip()
    args.fitness_fn = fn

    if is_multitask_fitness_fn_string(fn):
        tasks = parse_multitask_task_list(fn)
        if len(tasks) < 2:
            raise ValueError(
                "Multitask --fitness_fn needs >=2 comma-separated tasks (e.g. mmstar,mmmu_val,gqa); "
                "single-task names must not contain commas."
            )
        for t in tasks:
            if t not in TASK_PRIMARY_METRIC_KEY:
                raise ValueError(
                    f"Unknown multitask {t!r}; supported: {sorted(TASK_PRIMARY_METRIC_KEY)}"
                )
        ws = args.fitness_multi_weights
        if ws is None or len(ws) == 0:
            ws = [1.0] * len(tasks)
        else:
            ws = list(ws)
            if len(ws) != len(tasks):
                raise ValueError(
                    f"--fitness_multi_weights length must be {len(tasks)} for tasks {tasks}"
                )
        s = float(sum(ws))
        if s <= 0:
            raise ValueError("fitness_multi_weights must sum to a positive value")
        args.fitness_multi_weights = [w / s for w in ws]
        # Quiet by default; set MODSKIP_VERBOSE_FITNESS=1 to print multitask weights
        if os.environ.get("MODSKIP_VERBOSE_FITNESS", "").strip().lower() in ("1", "true", "yes"):
            print(f"[Info] multitask fitness tasks={tasks}, weights={args.fitness_multi_weights}")
        return

    if args.fitness_multi_weights is not None and len(args.fitness_multi_weights) > 0:
        print("[Warn] --fitness_multi_weights ignored unless comma multitask fitness_fn.")
    args.fitness_multi_weights = None

    n = normalize_fitness_fn_string(fn)
    if n == "gc":
        return
    if n not in LMMS_SINGLE_EVAL_FNS:
        raise ValueError(
            f"Unknown single-task fitness_fn: {fn!r}; choose {sorted(LMMS_SINGLE_EVAL_FNS)} or gc"
        )


def main():
    
    start_time_seconds = time.time()
    
    args = parse_args()
    set_architecture(
        num_layers=args.arch_num_layers,
        hidden_size=args.arch_hidden_size,
        intermediate_size=args.arch_intermediate_size,
        text_tokens=args.arch_text_tokens,
        vision_tokens=args.arch_vision_tokens,
    )
    initial_num_skip = args.num_skip_layers
    if args.num_skip_layers is not None:
        args.sparsity = sparsity_from_num_skip_vision_vision(args.num_skip_layers)
        print(
            f"[Info] num_skip_layers={args.num_skip_layers} -> target sparsity "
            f"(FLOPs saved / BASE_TFLOPs)={args.sparsity:.8f}"
        )
    apply_fitness_fn_cli_args(args)
    if args.no_fitness_subset_shuffle:
        args.fitness_subset_seed = None
    # Sanity checks
    assert len(args.survivors_per_selection) == len(
        args.samples_per_selection
    ), "Lists for selection survivors and tokens must have same length"
    assert args.survivors_per_selection[-1] == args.population_size, "Last stage should have population_size survivor"
    if args.drop_two_consecutive:
        assert args.drop_entire_block, "Can't drop two consecutive without dropping entire block"
        assert args.legal_to_drop_path == None, "Not implemented"
    if args.fitness_fn == "flickr30k_test_cider" and not args.invert_fitness:
        print(
            "[Warn] flickr_CIDEr is higher-better but EA minimizes fitness; add --invert_fitness."
        )
    if args.fitness_fn == "mme_cognition" and not args.invert_fitness:
        print(
            "[Warn] mme_cognition_score is higher-better but EA minimizes fitness; add --invert_fitness."
        )
    if args.fitness_fn == "mme_perception" and not args.invert_fitness:
        print(
            "[Warn] mme_perception_score is higher-better but EA minimizes fitness; add --invert_fitness."
        )
    if is_multitask_fitness_fn_string(args.fitness_fn) and not args.invert_fitness:
        print(
            "[Warn] Weighted multitask score is higher-better but EA minimizes fitness; add --invert_fitness."
        )

    # Get device and dtype
    assert torch.cuda.is_available()
    device = f"cuda"
    dtype = getattr(torch, args.dtype)
    # Fix seed
    # fix_seed(args.seed)
    # Init W&B logger
    if args.log_wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        wandb.init(config=args)
        
    # Inject SearchV importance ranking into vision_skip_flops if provided
    if args.searchv_importance_ranking is not None:
        _gs.SEARCHV_IMPORTANCE_RANKING = args.searchv_importance_ranking
        print(f"Using SearchV importance ranking from args: {args.searchv_importance_ranking}")

    # Generate initial population candidates
    initial_population_candidates = generate_parent_configurations(
        target_percent=args.sparsity, 
        num_parents=args.initially_generated,
        num_initial_protected_layers=args.protect_initial,
        num_final_protected_layers=args.protect_final,
        max_tolerance_percent=0.025,
        min_percent_threshold=0.025
    )
        
    # Load model
    eval_wrapper = None
    if is_lmms_eval_fitness(args.fitness_fn):
        eval_wrapper = Llava(
            pretrained=args.model_name_or_path,
            conv_template="v1",
            batch_size=1,
        )
        model = eval_wrapper.model          # underlying HF model
        tokenizer = eval_wrapper.tokenizer
        print(f"[Info] Llava wrapper sanity: has _model={hasattr(eval_wrapper, '_model')}")
    else:
        tokenizer, model, image_processor, max_length = \
            load_pretrained_llava(model_path=args.model_name_or_path, model_name="llava")

    model.config.use_cache = False
    num_layers = len(get_layers(model))

    calibration_data: List = []
    if args.fitness_fn == "gc":
        gc_tasks = tuple(
            t.strip() for t in args.gc_calib_tasks.split(",") if t.strip()
        )
        if not gc_tasks:
            raise ValueError("--gc_calib_tasks parsed empty; pass comma-separated lmms-eval task names")
        calibration_data = build_gc_searchv_calibration_batches(
            hf_model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            conv_template=args.gc_conv_template,
            task_names=gc_tasks,
            limit_per_task=args.gc_calib_limit_per_task,
        )
        print(
            f"[Info] GC calib (lmms-eval): tasks {gc_tasks}, "
            f"{args.gc_calib_limit_per_task} docs/task, total_batches={len(calibration_data)}"
        )

    target_logits = []
    if args.fitness_fn == "gc":
        print("Resetting model to baseline (no skip) for GC reference logits (full, no-skip branch)...")
        baseline_config = [('None', 'None')] * num_layers
        model.model.update_skip_configs(baseline_config)

        print("Computing reference logits with baseline model (no skip)...")
        for i in trange(0, len(calibration_data), desc="Computing GC reference logits (calib)", leave=False):
            with torch.inference_mode():
                batch_gpu = _move_forward_batch_to_device(
                    calibration_data[i], model.device
                )
                raw_logits = model(**batch_gpu).logits
                # compute_gc_div only uses last position; avoid full [B,S,V] on CPU / huge H2D copy
                logits = raw_logits[:, -1:, :].contiguous().cpu()
                del raw_logits
                target_logits.append(logits)
                del batch_gpu
                torch.cuda.empty_cache()

        print(
            f"Reference logits computed (last-position only). Sample shape: {target_logits[0].shape}"
        )
        print(
            "[Info] GC: SearchV LC-style KLDivLoss on last token; ref=no-skip, cand=current skip; mean over calib."
        )

    tabu = None if args.no_tabu else TabuCache()
    if tabu is not None:
        tabu.add_batch(initial_population_candidates)
        print(
            f"[Info] TabuCache on: seeded {len(initial_population_candidates)} initial configs; "
            f"duplicates skipped (see generate_offspring tabu_skipped)."
        )
    else:
        print("[Info] TabuCache off (--no_tabu); offspring may repeat configs.")
    fitness_cache = {} if not args.no_fitness_cache else None
    if args.no_fitness_cache:
        print("[Info] Fitness cache disabled (--no_fitness_cache)")

    population, train_fitnesses = selection(
        model=model,
        candidates=initial_population_candidates,
        calibration_data=calibration_data,
        invert_fitness=args.invert_fitness,
        num_samples=args.initial_samples,
        fitness_fn=args.fitness_fn,
        target_logits=target_logits,
        eval_wrapper=eval_wrapper,
        fitness_cache=fitness_cache,
        num_survive=args.population_size,
        multitask_weights=args.fitness_multi_weights,
        subset_shuffle_seed=args.fitness_subset_seed,
    )

    log_dict = {}
    early_stop_counter = 0
    global_best_fitness = float('inf')

    for gen_id in range(args.generations):
        print(f"---Generation {gen_id + 1}/{args.generations}---")
    
        current_best_fitness = train_fitnesses[0]
        
        # Format progress by fitness_fn
        if args.fitness_fn == "gqa":
            # GQA: accuracy percent
            current_acc = -current_best_fitness if args.invert_fitness else current_best_fitness
            print(f"Current Best Accuracy: {current_acc*100:.2f}%")
        elif args.fitness_fn == "flickr30k_test_cider":
            cur = -current_best_fitness if args.invert_fitness else current_best_fitness
            print(f"Current Best flickr_CIDEr: {cur:.4f}")
        elif args.fitness_fn == "mme_cognition":
            cur = -current_best_fitness if args.invert_fitness else current_best_fitness
            print(f"Current Best MME cognition score: {cur:.2f}")
        elif args.fitness_fn == "mme_perception":
            cur = -current_best_fitness if args.invert_fitness else current_best_fitness
            print(f"Current Best MME perception score: {cur:.2f}")
        elif is_multitask_fitness_fn_string(args.fitness_fn):
            cur = -current_best_fitness if args.invert_fitness else current_best_fitness
            print(f"Current Best weighted multi-task: {cur:.4f}")
        elif args.fitness_fn == "gc":
            print(
                f"Current Best GC (SearchV last-token KL, mean over calib): "
                f"{current_best_fitness:.6f}"
            )
        else:
            print(f"Current Best Fitness: {current_best_fitness:.2e}")

        if current_best_fitness < global_best_fitness:
            global_best_fitness = current_best_fitness
            early_stop_counter = 0
            
            if args.fitness_fn == "gqa":
                best_acc = -global_best_fitness if args.invert_fitness else global_best_fitness
                print(f"--- New Global Best Found! Accuracy: {best_acc*100:.2f}% ---")
            elif args.fitness_fn == "flickr30k_test_cider":
                best_c = -global_best_fitness if args.invert_fitness else global_best_fitness
                print(f"--- New Global Best Found! flickr_CIDEr: {best_c:.4f} ---")
            elif args.fitness_fn == "mme_cognition":
                best_s = -global_best_fitness if args.invert_fitness else global_best_fitness
                print(f"--- New Global Best Found! MME cognition: {best_s:.2f} ---")
            elif args.fitness_fn == "mme_perception":
                best_s = -global_best_fitness if args.invert_fitness else global_best_fitness
                print(f"--- New Global Best Found! MME perception: {best_s:.2f} ---")
            elif is_multitask_fitness_fn_string(args.fitness_fn):
                best_s = -global_best_fitness if args.invert_fitness else global_best_fitness
                print(f"--- New Global Best Found! weighted multi-task: {best_s:.4f} ---")
            elif args.fitness_fn == "gc":
                print(
                    f"--- New Global Best Found! GC: {global_best_fitness:.6f} ---"
                )
            else:
                print(f"--- New Global Best Found! Fitness: {global_best_fitness:.2e} ---")
        else:
            early_stop_counter += 1
            print(f"--- No improvement. Early stop counter: {early_stop_counter}/{args.early_stop} ---")

        if early_stop_counter >= args.early_stop:
            print(f"--- Early stopping triggered at generation {gen_id + 1} ---")
            break
        

        if gen_id % args.eval_every == 0 and not args.no_eval:
            model.model.update_skip_configs(population[0])
            log_dict["train_fitness"] = train_fitnesses[0]

            if normalize_fitness_fn_string(args.fitness_fn) == "gqa" and eval_wrapper is not None:
                print("\n--- Evaluating best candidate on MMStar (generalization check) ---")
                results = simple_eval(eval_wrapper, task_list=['mmstar'], verbose=True, hf_model=model)
                log_dict["mmstar_results"] = results

        if args.log_wandb:
            wandb.log(log_dict)


        offspring_list = generate_offspring(
            parent_configs=population,
            parent_fitnesses=train_fitnesses,
            num_offspring=args.offspring,
            num_layers=num_layers,
            protect_initial=args.protect_initial,
            protect_final=args.protect_final,
            tournament_k=args.tournament_k,
            tabu=tabu,
            mutation_window=args.mutation_window,
        )


        # Selection in multiple steps
        for num_survive, num_samples in zip(args.survivors_per_selection, args.samples_per_selection):
            # 'all' -> full lmms split or full GC calib list
            if num_samples == 'all':
                if is_lmms_eval_fitness(args.fitness_fn):
                    # lmms: None means full eval set
                    num_samples = None
                else:
                    num_samples = len(calibration_data)
            
            if num_survive == args.survivors_per_selection[-1]:
                for i in range(
                    len(population)
                ):  # Elitist EA: Add search points in current generation to final selection step
                    if population[i] not in offspring_list:
                        offspring_list.append(population[i])

            offspring_list, train_fitnesses = selection(
                model=model,
                candidates=offspring_list,
                num_survive=num_survive,
                calibration_data=calibration_data,
                invert_fitness=args.invert_fitness,
                num_samples=num_samples,
                fitness_fn=args.fitness_fn,
                target_logits=target_logits,
                eval_wrapper=eval_wrapper,
                fitness_cache=fitness_cache,
                multitask_weights=args.fitness_multi_weights,
                subset_shuffle_seed=args.fitness_subset_seed,
            )

        population = offspring_list

        layer_drop_config = population[0]
        
        
            
    end_time_seconds = time.time()

    duration_seconds = end_time_seconds - start_time_seconds
    duration_minutes = duration_seconds / 60

    print("-" * 60)
    print(f"✅ Search completed! Time cost: {duration_minutes:.2f} min")
    print("-" * 60)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        model_name = args.model_name_or_path.split('/')[-1]
        flops_stats = _compute_flops_stats(layer_drop_config, args.sparsity)
        
        # Output filename prefix (nskip{N}_ when --num_skip_layers set)
        file_prefix = _layer_drop_config_file_prefix(
            model_name, args.task_name, args.sparsity, initial_num_skip
        )
        
        # Save layer drop config
        config_file = os.path.join(args.save_dir, f"{file_prefix}.txt")
        with open(config_file, "w") as file:
            for line in layer_drop_config:
                file.write(f"{line[0]} {line[1]}" + "\n")
        print(f"\n✅ Saved layer drop config to: {config_file}")
        
        # Save run metadata and best metrics
        info_file = os.path.join(args.save_dir, f"{file_prefix}_info.txt")
        with open(info_file, "w") as file:
            file.write("="*60 + "\n")
            file.write("EVOLUTIONARY SEARCH RESULTS\n")
            file.write("="*60 + "\n\n")
            
            # Final metrics
            file.write("--- Final Results ---\n")
            if args.fitness_fn == "gqa":
                final_acc = -train_fitnesses[0] if args.invert_fitness else train_fitnesses[0]
                file.write(f"Best GQA Accuracy: {final_acc*100:.2f}%\n")
            elif args.fitness_fn == "flickr30k_test_cider":
                final_c = -train_fitnesses[0] if args.invert_fitness else train_fitnesses[0]
                file.write(f"Best flickr_CIDEr: {final_c:.6f}\n")
            elif args.fitness_fn == "mme_cognition":
                final_s = -train_fitnesses[0] if args.invert_fitness else train_fitnesses[0]
                file.write(f"Best MME cognition score: {final_s:.6f}\n")
            elif args.fitness_fn == "mme_perception":
                final_s = -train_fitnesses[0] if args.invert_fitness else train_fitnesses[0]
                file.write(f"Best MME perception score: {final_s:.6f}\n")
            elif is_multitask_fitness_fn_string(args.fitness_fn):
                final_w = -train_fitnesses[0] if args.invert_fitness else train_fitnesses[0]
                file.write(f"Best weighted multi-task score: {final_w:.6f}\n")
                file.write(
                    f"Tasks: {parse_multitask_task_list(args.fitness_fn)}\n"
                )
                file.write(
                    f"Normalized weights: {args.fitness_multi_weights}\n"
                )
            elif args.fitness_fn == "gc":
                file.write(
                    f"Best GC (SearchV last-token KL, mean over calib): {train_fitnesses[0]:.8f}\n"
                )
            else:
                file.write(f"Best Fitness: {train_fitnesses[0]:.6f}\n")
            file.write("\n--- FLOPs Saved (Target vs Actual) ---\n")
            file.write(f"Base TFLOPs: {flops_stats['base_tflops']:.6f}\n")
            file.write(f"Target Saved: {flops_stats['target_saved_tflops']:.6f} TFLOPs ({flops_stats['target_saved_percent']:.4f}%)\n")
            file.write(f"Actual Saved: {flops_stats['actual_saved_tflops']:.6f} TFLOPs ({flops_stats['actual_saved_percent']:.4f}%)\n")
            file.write(f"Delta (Actual-Target): {flops_stats['delta_saved_tflops']:.6f} TFLOPs ({flops_stats['delta_saved_percent']:.4f}%)\n")
            file.write(f"Total Generations: {gen_id + 1}\n")
            file.write(f"Time Cost: {duration_minutes:.2f} min\n\n")
            
            # Run parameters
            file.write("--- Run Parameters ---\n")
            file.write(f"Model: {args.model_name_or_path}\n")
            if initial_num_skip is not None:
                file.write(f"num_skip_layers (init): {initial_num_skip}\n")
            file.write(f"Sparsity (target FLOPs frac): {args.sparsity}\n")
            file.write(f"Fitness Function: {args.fitness_fn}\n")
            if args.fitness_fn == "gc":
                file.write(
                    f"GC calib tasks: {args.gc_calib_tasks}, "
                    f"{args.gc_calib_limit_per_task} docs/task; conv={args.gc_conv_template}; "
                    f"total_batches={len(calibration_data)}\n"
                )
            if is_multitask_fitness_fn_string(args.fitness_fn) and args.fitness_multi_weights:
                file.write(
                    f"Fitness Multi Weights: {args.fitness_multi_weights}\n"
                )
            file.write(f"Invert Fitness: {args.invert_fitness}\n")
            file.write(f"Population Size: {args.population_size}\n")
            file.write(f"Generations: {args.generations}\n")
            file.write(f"Offspring: {args.offspring}\n")
            file.write(f"Initially Generated: {args.initially_generated}\n")
            file.write(f"Initial Samples: {args.initial_samples}\n")
            file.write(f"Survivors per Selection: {args.survivors_per_selection}\n")
            file.write(f"Samples per Selection: {args.samples_per_selection}\n")
            file.write(
                f"Fitness subset shuffle seed: {args.fitness_subset_seed} "
                f"(None = first N docs in lmms order; independent of model weights)\n"
            )
            file.write(f"Early Stop: {args.early_stop}\n")
            file.write(f"Protect Initial: {args.protect_initial}\n")
            file.write(f"Protect Final: {args.protect_final}\n")
            file.write(f"Eval Every: {args.eval_every}\n\n")
            
            # Best layer config listing
            file.write("--- Best Layer Drop Configuration ---\n")
            for i, line in enumerate(layer_drop_config):
                file.write(f"Layer {i}: {line[0]} {line[1]}\n")
        
        print(f"✅ Saved run info to: {info_file}")

        # FLOPs summary for scripting / diff
        flops_file = os.path.join(args.save_dir, f"{file_prefix}_flops.txt")
        with open(flops_file, "w") as file:
            file.write(f"base_tflops={flops_stats['base_tflops']:.10f}\n")
            file.write(f"target_saved_tflops={flops_stats['target_saved_tflops']:.10f}\n")
            file.write(f"actual_saved_tflops={flops_stats['actual_saved_tflops']:.10f}\n")
            file.write(f"delta_saved_tflops={flops_stats['delta_saved_tflops']:.10f}\n")
            file.write(f"target_saved_percent={flops_stats['target_saved_percent']:.10f}\n")
            file.write(f"actual_saved_percent={flops_stats['actual_saved_percent']:.10f}\n")
            file.write(f"delta_saved_percent={flops_stats['delta_saved_percent']:.10f}\n")
        print(f"✅ Saved FLOPs stats to: {flops_file}")

    print("\n" + "="*60)
    print("Final configuration:")
    print("="*60)
    for i, line in enumerate(layer_drop_config):
        print(f"Layer {i}: {line}")

 
    if args.log_wandb:
        wandb.log(log_dict)



if __name__ == "__main__":
    main()
