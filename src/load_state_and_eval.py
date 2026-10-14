import copy
import hashlib
from datetime import datetime
from pathlib import Path
import sys

# lmms-eval does `import modskip_eval_hooks`; ensure src/ is on sys.path before that import runs.
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch
import logging
import argparse
from tqdm import tqdm
import os

try:
    from datasets.utils import disable_progress_bars

    disable_progress_bars()
except ImportError:
    pass

import numpy as np
from lmms_eval import evaluator, utils
from lmms_eval.evaluator_utils import get_sample_size, get_task_list
from lmms_eval.tasks import TaskManager, get_task_dict
from lmms_eval.api.model import lmms
from lmms_eval.models.llava import Llava
from lmms_eval.api.instance import Instance
from typing import List, Tuple, Any, Optional, Sequence, Dict
from PIL import Image
from .searchv_llava_patch import load_skip_config

from accelerate import Accelerator
accelerator = Accelerator()
current_device = f"cuda:{accelerator.local_process_index}"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_layer_idx(s: str) -> List[int]:
    """Parse comma/space-separated layer indices (0-based), e.g. "0,1,2"."""
    s = (s or "").strip()
    if not s:
        return []
    normalized = s.replace(",", " ")
    return [int(x) for x in normalized.split() if x]


def _eval_kv_cache_enabled() -> bool:
    """KV cache on by default for faster generate_until; set MODSKIP_EVAL_NO_KV_CACHE=1 to disable."""
    return os.environ.get("MODSKIP_EVAL_NO_KV_CACHE", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    )


def _apply_eval_kv_cache(eval_model: Any, enabled: bool) -> None:
    """Mirror use_cache on Llava wrapper and underlying HF config."""
    eval_model._model.config.use_cache = enabled
    eval_model.use_cache = enabled


# lmms-eval MME task names (see lmms-eval tasks/mme/*.yaml)
MME_COGNITION_TASK = "mme_cognition"
MME_PERCEPTION_TASK = "mme_perception"

# MME aggregate expects exactly 2 docs per question_id; subsample by pairs
MME_PAIRWISE_TASK_NAMES = frozenset({"mme", "mme_perception", "mme_cognition"})

# Primary scalar keys under results["results"][task]; None -> first non-_stderr float
TASK_PRIMARY_METRIC_KEY = {
    "gqa": "exact_match,none",
    "mmstar": "average,none",
    "mmmu_val": "mmmu_acc,none",
    "mme_cognition": "mme_cognition_score,none",
    "mme_perception": "mme_perception_score,none",
    "flickr30k_test_cider": None,
}

# Legacy alias: same as "mmstar,mmmu_val,gqa"
WEIGHTED_MULTITASK_ORDER: Tuple[str, ...] = ("mmstar", "mmmu_val", "gqa")

FITNESS_FN_ALIAS = {
    "weighted_mmstar_mmmu_gqa": "mmstar,mmmu_val,gqa",
}


def normalize_fitness_fn_string(s: str) -> str:
    """Single task or comma multitask string; expands alias weighted_mmstar_mmmu_gqa."""
    key = (s or "").strip()
    return FITNESS_FN_ALIAS.get(key, key)


def is_multitask_fitness_fn_string(s: str) -> bool:
    """True if comma-separated or alias expands to multiple tasks."""
    return "," in normalize_fitness_fn_string(s)


def parse_multitask_task_list(s: str) -> List[str]:
    """Split normalized fitness_fn into task names."""
    norm = normalize_fitness_fn_string(s)
    return [p.strip() for p in norm.split(",") if p.strip()]


def extract_mme_cognition_score(results: dict) -> float:
    """MME cognition aggregate from simple_evaluate results."""
    sub = results["results"][MME_COGNITION_TASK]
    for k, v in sub.items():
        if "_stderr" in k or k in ("alias", "samples"):
            continue
        if "mme_cognition_score" in k and isinstance(v, (int, float)):
            return float(v)
    raise KeyError(f"mme_cognition_score not found; keys: {list(sub.keys())}")


def extract_mme_perception_score(results: dict) -> float:
    """MME perception aggregate from simple_evaluate results."""
    sub = results["results"][MME_PERCEPTION_TASK]
    for k, v in sub.items():
        if "_stderr" in k or k in ("alias", "samples"):
            continue
        if "mme_perception_score" in k and isinstance(v, (int, float)):
            return float(v)
    raise KeyError(f"mme_perception_score not found; keys: {list(sub.keys())}")


def extract_task_scalar_metric(results: dict, task_name: str) -> float:
    """Primary scalar metric for task_name (higher is better)."""
    if "results" not in results or task_name not in results["results"]:
        raise KeyError(
            f"results missing task {task_name}; have: {list(results.get('results', {}).keys())}"
        )
    sub = results["results"][task_name]
    preferred = TASK_PRIMARY_METRIC_KEY.get(task_name)
    if preferred and preferred in sub:
        v = sub[preferred]
        if isinstance(v, (int, float)):
            return float(v)
    for k, v in sub.items():
        if "_stderr" in k or k in ("alias", "samples"):
            continue
        if isinstance(v, (int, float)):
            return float(v)
    raise KeyError(f"No numeric metric for task {task_name}; keys: {list(sub.keys())}")


def _flatten_visuals_for_llava(visual: Optional[Sequence[Any]]) -> List[Any]:
    """Flatten doc_to_visual output like lmms-eval Llava.flatten."""
    if not visual:
        return []
    if any(v is None for v in visual):
        return []
    out: List[Any] = []
    for item in visual:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            for x in item:
                if x is not None:
                    out.append(x)
        else:
            out.append(item)
    return out


def build_gc_searchv_calibration_batches(
    hf_model: Any,
    tokenizer: Any,
    image_processor: Any,
    conv_template: str = "v1",
    task_names: Sequence[str] = ("gqa", "flickr30k_test"),
    limit_per_task: int = 20,
    dtype: Optional[torch.dtype] = None,
    tensor_device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """
    Build GC calibration batches (SearchV / LC-style) from lmms-eval tasks.

    Takes up to ``limit_per_task`` eval docs per task, encodes Llava forward dicts
    (input_ids, attention_mask, images, image_sizes) for ``compute_gc_div``.

    Default ``tensor_device=None`` stores tensors on CPU during build to avoid OOM;
    move to model device before forward.
    """
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    if dtype is None:
        dtype = next(hf_model.parameters()).dtype

    if tensor_device is None:
        tensor_device = torch.device("cpu")
    elif isinstance(tensor_device, str):
        tensor_device = torch.device(tensor_device)

    tm = TaskManager("ERROR", model_name="llava")
    task_dict = get_task_dict(list(task_names), tm)
    batches: List[Dict[str, Any]] = []

    for to in get_task_list(task_dict):
        t = to.task
        if isinstance(t, tuple):
            _, t = t
        if t is None:
            continue
        docs = t.eval_docs
        n_docs = len(docs)
        n_take = min(int(limit_per_task), n_docs)
        for i in range(n_take):
            doc = docs[i]
            visuals = _flatten_visuals_for_llava(t.doc_to_visual(doc))
            context = t.doc_to_text(doc)

            if visuals and DEFAULT_IMAGE_TOKEN not in context:
                image_tokens = " ".join([DEFAULT_IMAGE_TOKEN] * len(visuals))
                question = image_tokens + "\n" + context
            else:
                question = context

            if "llama_3" in conv_template:
                conv = copy.deepcopy(conv_templates[conv_template])
            else:
                conv = conv_templates[conv_template].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .to(device=tensor_device, dtype=torch.long)
            )
            attention_mask = torch.ones_like(
                input_ids, device=tensor_device, dtype=torch.long
            )

            if visuals:
                image_tensor = process_images(visuals, image_processor, hf_model.config)
                if isinstance(image_tensor, list):
                    image_tensor = [
                        _img.to(device=tensor_device, dtype=dtype)
                        for _img in image_tensor
                    ]
                else:
                    image_tensor = image_tensor.to(device=tensor_device, dtype=dtype)
                image_sizes = [im.size for im in visuals]
            else:
                image_tensor = None
                image_sizes = None

            item: Dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if image_tensor is not None:
                item["images"] = image_tensor
            if image_sizes is not None:
                item["image_sizes"] = image_sizes
            batches.append(item)

    return batches


def _mme_pairwise_subset_indices(
    eval_docs: Sequence[Any],
    k: int,
    rng: np.random.RandomState,
) -> List[int]:
    """
    MME needs 2 docs per question_id. Uniform random indices often break pairing;
    group by question_id, shuffle pairs, take min(num_pairs, floor(k/2)) pairs.
    """
    from collections import defaultdict

    qid_to_idxs: Dict[Any, List[int]] = defaultdict(list)
    for i, doc in enumerate(eval_docs):
        qid = doc.get("question_id")
        if qid is None:
            continue
        qid_to_idxs[qid].append(i)
    pairs: List[List[int]] = []
    for qid, idxs in qid_to_idxs.items():
        if len(idxs) == 2:
            pairs.append(sorted(idxs))
    if not pairs:
        logger.warning(
            "MME: no question_id with exactly 2 docs; falling back to random indices (aggregate may fail)"
        )
        return rng.permutation(len(eval_docs))[: max(k, 0)].tolist()

    max_pairs = min(len(pairs), k // 2)
    if max_pairs <= 0:
        return []
    order = rng.permutation(len(pairs))
    selected: List[int] = []
    for j in order[:max_pairs]:
        selected.extend(pairs[int(j)])
    return selected


def _prepare_modskip_shuffle_subset(
    eval_model,
    task_list: Sequence[str],
    num_samples: Optional[int],
    subset_shuffle_seed: Optional[int],
) -> None:
    """
    Register deterministic shuffled doc indices before simple_evaluate (independent of weights).
    If subset_shuffle_seed is None, skip (lmms default order, first limit docs).
    """
    from . import modskip_eval_hooks as mseh

    mseh.clear_shuffle_subset()
    if num_samples is None or subset_shuffle_seed is None:
        return

    task_manager = TaskManager("ERROR", model_name=eval_model)
    task_dict = get_task_dict(list(task_list), task_manager)
    for to in get_task_list(task_dict):
        t = to.task
        if isinstance(t, tuple):
            _, t = t
        if t is None:
            continue
        tname = t.config.task
        limit_i = get_sample_size(t, num_samples)
        if limit_i is None:
            continue
        n_docs = len(t.eval_docs)
        k = min(int(limit_i), n_docs)
        if k <= 0:
            continue
        seed_mix = int(
            hashlib.md5(f"{subset_shuffle_seed}:{tname}".encode()).hexdigest()[:8],
            16,
        )
        rng = np.random.RandomState(seed_mix)
        if tname in MME_PAIRWISE_TASK_NAMES:
            idxs = _mme_pairwise_subset_indices(t.eval_docs, k, rng)
        else:
            idxs = rng.permutation(n_docs)[:k].tolist()
        if not idxs:
            continue
        mseh.register_shuffle_subset(tname, idxs, t)


def weighted_multitask_score(
    results: dict,
    tasks: Sequence[str],
    weights: Sequence[float],
) -> float:
    """Weighted sum sum_i w_i * metric_i (caller may normalize weights)."""
    if len(tasks) != len(weights):
        raise ValueError("tasks and weights must have the same length")
    return float(
        sum(w * extract_task_scalar_metric(results, t) for t, w in zip(tasks, weights))
    )


def simple_eval(
    eval_model,
    task_list=None,
    num_samples=None,
    verbose=True,
    hf_model=None,
    save_path=None,
    subset_shuffle_seed: Optional[int] = 42,
):
    """
    Run lmms-eval via a Llava wrapper.

    Args:
        eval_model: lmms Llava instance.
        task_list: task names; default [MME_COGNITION_TASK] if None.
        num_samples: cap per task (None = all).
        subset_shuffle_seed: deterministic shuffle before subsample; None = lmms default order.
        verbose: print progress and table.
        hf_model: optional HF backbone (overrides eval_model._model).
        save_path: if set, append utils.make_table(results) to {stem}_evaluation_summary.txt.
    """
    if task_list is None:
        # Default single-task: MME cognition only
        task_list = [MME_COGNITION_TASK]

    class MockArgs:
        output_path = "./"
        process_with_media = False
        output_base_path = "results"

    # Honor caller hf_model so update_skip_configs is evaluated
    if hf_model is not None:
        eval_model._model = hf_model
    if not hasattr(eval_model, "_model"):
        raise AttributeError("eval_model has no attribute `_model`; please pass hf_model to simple_eval.")

    _kv = _eval_kv_cache_enabled()
    _apply_eval_kv_cache(eval_model, _kv)
    if verbose:
        if _kv:
            print(
                "[Info] KV cache on for generation; export MODSKIP_EVAL_NO_KV_CACHE=1 to disable."
            )
        else:
            print("[Info] KV cache off (MODSKIP_EVAL_NO_KV_CACHE).")
        if num_samples is None:
            print(f"Start evaluate: {task_list} (all samples)")
        else:
            _shuf = f", shuffle_seed={subset_shuffle_seed}" if subset_shuffle_seed is not None else ", shuffle=off"
            print(f"Start evaluate: {task_list} ({num_samples} samples{_shuf})")

    _prepare_modskip_shuffle_subset(eval_model, task_list, num_samples, subset_shuffle_seed)

    # PTBTokenizer spawns Java and may spam stderr; silence child stderr for flickr caption tasks
    _subprocess_popen_orig = None
    if not verbose:
        import sys
        import subprocess as _subprocess
        # Some tasks print mid-aggregation to stdout; hide when quiet
        _dn = open(os.devnull, "w")
        _old_out, _old_err = sys.stdout, sys.stderr
        sys.stdout = _dn
        sys.stderr = _dn
        if any("flickr" in str(t) for t in task_list):
            _subprocess_popen_orig = _subprocess.Popen

            def _popen_silence_stderr(*args, **kwargs):
                if "stderr" not in kwargs:
                    kwargs["stderr"] = _subprocess.DEVNULL
                return _subprocess_popen_orig(*args, **kwargs)

            _subprocess.Popen = _popen_silence_stderr

    try:
        results = evaluator.simple_evaluate(
            model=eval_model,
            tasks=task_list,
            batch_size=1,
            limit=num_samples,
            cli_args=MockArgs(),
        )
    finally:
        try:
            from . import modskip_eval_hooks as mseh

            mseh.clear_shuffle_subset()
        except ImportError:
            pass
        if not verbose:
            if _subprocess_popen_orig is not None:
                import subprocess as _subprocess

                _subprocess.Popen = _subprocess_popen_orig
            sys.stdout = _old_out
            sys.stderr = _old_err
            _dn.close()

    table_text = utils.make_table(results)
    if verbose:
        print(table_text)

    if save_path is not None:
        base_dir = os.path.dirname(save_path)
        base_name = os.path.splitext(os.path.basename(save_path))[0]
        if not base_name:
            base_name = "evaluation"
        summary_path = os.path.join(base_dir, f"{base_name}_evaluation_summary.txt")
        try:
            append_after = (
                os.path.isfile(summary_path) and os.path.getsize(summary_path) > 0
            )
            ts = datetime.now().isoformat(timespec="seconds")
            with open(summary_path, "a", encoding="utf-8") as f:
                if append_after:
                    f.write("\n\n")
                f.write("=" * 72 + "\n")
                f.write(f"# evaluation {ts}\n")
                f.write("=" * 72 + "\n\n")
                f.write(table_text)
                if not table_text.endswith("\n"):
                    f.write("\n")
            if verbose:
                verb = "Appended" if append_after else "Wrote"
                print(f"[Info] {verb} evaluation summary (make_table) to {summary_path}")
        except Exception as e:
            logger.warning("Failed to save evaluation summary: %s", e)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Load LLaVA with optional layer skip config and run lmms-eval tasks."
    )
    parser.add_argument(
        "--pretrained",
        "--model",
        dest="pretrained",
        type=str,
        default="/path_to_model/llava-v1.5-7b",
        help="HF or local LLaVA checkpoint (lmms-eval Llava pretrained)",
    )
    parser.add_argument(
        "--skip-config",
        "--config",
        dest="skip_config",
        type=str,
        default="",
        help="Layer skip file (mha ffn per line); mutually exclusive with --layer_idx; empty = base model",
    )
    parser.add_argument(
        "--layer_idx",
        type=str,
        default="",
        help=(
            "Mutually exclusive with --skip-config. 0-based layer indices for VV (Vision Vision); "
            "others NN; length --num_layers (default 32)."
        ),
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=None,
        help="Total layers when using --layer_idx (default 32).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="mme_cognition",
        help="Comma-separated lmms-eval task names, e.g. mme_cognition,seedbench_image,mme_perception",
    )
    parser.add_argument(
        "--conv-template",
        type=str,
        default="v1",
        help="Llava conv_template (default v1; same as searchv_evolution --gc_conv_template)",
    )
    parser.add_argument(
        "--num-samples",
        "--num_samples",
        type=int,
        default=None,
        help="Random subset size per task; omit for full eval",
    )
    parser.add_argument(
        "--subset-shuffle-seed",
        "--subset_shuffle_seed",
        type=int,
        default=None,
        help="RNG seed paired with --num-samples shuffle",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Append summary to {stem}_evaluation_summary.txt; default follows --skip-config",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress table and verbose prints",
    )
    args, _ = parser.parse_known_args()

    model_id = args.pretrained
    file_path = (args.skip_config or "").strip()
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not task_list:
        raise SystemExit("Error: --tasks parsed empty; pass at least one task name")

    save_path = args.save_path
    if save_path is None and file_path:
        save_path = file_path

    layer_idx_list = _parse_layer_idx(args.layer_idx)
    layer_idx_set = set(layer_idx_list)
    if len(layer_idx_list) != len(layer_idx_set):
        print("[Warn] --layer_idx had duplicate indices; deduplicated", file=sys.stderr)

    if file_path and layer_idx_set:
        raise SystemExit(
            "Error: --skip-config and --layer_idx are mutually exclusive; use one mode only."
        )

    # NOTE:
    # - `skip_config` only tells each layer's strategy.
    # - In llava_llama.py, whether a layer uses SearchV skipping is decided
    #   at model construction time by `REPLACED_LAYERS`.
    # So we must prepare REPLACED_LAYERS before loading model.
    skip_config = None
    if file_path:
        if os.path.isfile(file_path):
            skip_config = load_skip_config(file_path)
        else:
            raise SystemExit(f"Error: --skip-config path missing or not a file: {file_path!r}")
    elif layer_idx_set:
        vv_pair = ("Vision", "Vision")
        n = args.num_layers if args.num_layers is not None else 32
        skip_config = [("None", "None")] * n
        for i in layer_idx_set:
            if i < 0 or i >= n:
                raise SystemExit(
                    f"Error: --layer_idx out of range {i} for --num_layers={n} (valid 0..{n-1})"
                )
            skip_config[i] = vv_pair
        print(
            f"[Info] Built skip from --layer_idx: VV layers {sorted(layer_idx_set)}, "
            f"{n - len(layer_idx_set)} NN layers; num_layers={n}"
        )
    elif (args.layer_idx or "").strip():
        raise SystemExit(
            "Error: --layer_idx set but no valid indices; use comma/space-separated non-negative ints."
        )

    if skip_config is not None:
        replaced_idxs = [
            idx for idx, (mha, ffn) in enumerate(skip_config)
            if not (mha.lower() == "none" and ffn.lower() == "none")
        ]
        replaced_layers = ",".join(str(i) for i in replaced_idxs)
        os.environ["REPLACED_LAYERS"] = replaced_layers
        print(f"[Info] REPLACED_LAYERS derived from config: {replaced_layers}")
    else:
        print(f"[Info] Evaluate base model")


    # CLI-aligned model path:
    # equivalent to `--model llava --model_args pretrained=...,conv_template=v1`
    eval_model = Llava(
        pretrained=model_id,
        conv_template=args.conv_template,
        batch_size=1,
    )
    if skip_config is not None:
        eval_model.model.model.update_skip_configs(skip_config)
    cfg = eval_model.model.model.get_skip_configs()
    
    n_vision = sum(1 for mha, ffn in cfg if mha == "vision" and ffn == "vision")
    print(f"[Info] Applied skip config: vision/vision={n_vision}, total_layers={len(cfg)}")
    n_vision = sum(1 for mha, ffn in cfg if mha == "vision" and ffn == "none")
    print(f"[Info] Applied skip config: vision/None={n_vision}, total_layers={len(cfg)}")
    n_vision = sum(1 for mha, ffn in cfg if mha == "none" and ffn == "vision")
    print(f"[Info] Applied skip config: None/vision={n_vision}, total_layers={len(cfg)}")

    _apply_eval_kv_cache(eval_model, _eval_kv_cache_enabled())

    # cfg = model.model.get_skip_configs()
    # print(cfg)
    # # logits_1 = model(**calibration_data[0]).logits.cpu()
    
    # output = model.generate(**calibration_data[0], min_new_tokens=300, max_new_tokens=500,do_sample=False, use_cache=False, return_dict_in_generate=True)
    # generated_sequence_tensor = output.sequences[0]
    # decoded_text_1 = tokenizer.decode(
    #     generated_sequence_tensor, 
    #     skip_special_tokens=True
    # )
    
    # print(decoded_text_1)
    
    
    print(f"[Info] pretrained={model_id!r} conv_template={args.conv_template!r} tasks={task_list}")
    _eval_kw: Dict[str, Any] = {
        "eval_model": eval_model,
        "task_list": task_list,
        "num_samples": args.num_samples,
        "verbose": not args.quiet,
        "hf_model": eval_model._model,
        "save_path": save_path,
    }
    if args.subset_shuffle_seed is not None:
        _eval_kw["subset_shuffle_seed"] = args.subset_shuffle_seed
    simple_eval(**_eval_kw)

if __name__ == "__main__":
    main()