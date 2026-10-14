#!/usr/bin/env python3
"""
Micro-benchmark: one fixed multimodal forward (like GC / eval stack), comparing
a skip-config txt vs all-``None`` baseline. Uses the same layer-replacement and
``update_skip_configs`` path as ``python -m src.load_state_and_eval`` /
``scripts/eval_llava_*.sh``.  Reports per-phase
peak CUDA allocated memory (GiB) during warmup + timed forwards.

**``--text-tokens``** (default 64): exact **decoder text-region** length — tokens in the
Text slice after prefix + vision in SearchV (``VISUAL_START_IDX`` + ``n_visual``).  Under
single-image LLaVA, embed length ``H = len(input_ids) - 1 + n_visual`` and
``text_len = H - VISUAL_START_IDX - n_visual = len(input_ids) - (VISUAL_START_IDX + 1)``,
so we set **total tokenizer length** ``len(input_ids) = text_tokens + VISUAL_START_IDX + 1``.

**``--target-input-len``**: optional override for that **total** ``input_ids`` length (before
vision-token expansion).  If set, it **does not** re-derive text length; use only if you
know what you are doing.

Run from repository root::

    python -m src.inference_speedup \\
        --pretrained /path/to/llava-v1.5-7b \\
        --skip-config outputs/layer_drop_config_....txt
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from .searchv_llava_patch import VISUAL_START_IDX, load_skip_config
from lmms_eval.models.llava import Llava

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token


def input_ids_len_for_text_tokens(text_tokens: int) -> int:
    """Total ``input_ids`` length (before vision expand) for exactly ``text_tokens`` in decoder Text region.

    With one image placeholder: embed seq ``H = len(input_ids) - 1 + n_visual``,
    ``text_len = H - VISUAL_START_IDX - n_visual = len(input_ids) - VISUAL_START_IDX - 1``.
    """
    return int(text_tokens) + VISUAL_START_IDX + 1


def _move_forward_batch_to_device(batch: dict, device: torch.device) -> dict:
    out: Dict[str, Any] = {}
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


def _set_replaced_layers_env(skip_config: Optional[List[Tuple[str, str]]]) -> None:
    os.environ.pop("REPLACED_LAYERS", None)
    if skip_config is None:
        return
    replaced_idxs = [
        idx
        for idx, (mha, ffn) in enumerate(skip_config)
        if not (mha.lower() == "none" and ffn.lower() == "none")
    ]
    if replaced_idxs:
        os.environ["REPLACED_LAYERS"] = ",".join(str(i) for i in replaced_idxs)


def build_benchmark_batch(
    tokenizer: Any,
    image_processor: Any,
    hf_model: torch.nn.Module,
    conv_template: str,
    target_input_ids_len: int,
    tensor_device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    """
    Encode dummy text, **truncate** to ``target_input_ids_len`` (total ``input_ids`` before
    vision expand).  When that length comes from ``input_ids_len_for_text_tokens(T)``, the
    decoder Text region has exactly ``T`` tokens (single-image LLaVA + SearchV layout).
    Do not set length so small that the image placeholder is truncated away.
    """
    pil = Image.new("RGB", (336, 336), color=(120, 120, 120))

    def _encode_ids(tail: str) -> torch.Tensor:
        question = DEFAULT_IMAGE_TOKEN + "\n" + tail
        if "llama_3" in conv_template:
            conv = copy.deepcopy(conv_templates[conv_template])
        else:
            conv = conv_templates[conv_template].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0)

    # Long enough filler; extend once if tokenizer is unusually sparse.
    tail = " benchmark" * max(target_input_ids_len * 8, 128)
    input_ids = _encode_ids(tail)
    if input_ids.shape[1] < target_input_ids_len:
        tail = tail + (" x" * (target_input_ids_len * 16))
        input_ids = _encode_ids(tail)

    if input_ids.shape[1] < target_input_ids_len:
        raise RuntimeError(
            f"Encoded length {input_ids.shape[1]} < target {target_input_ids_len}; "
            "increase filler or set a smaller --target-input-len."
        )

    input_ids = input_ids[:, :target_input_ids_len].contiguous().to(
        device=tensor_device, dtype=torch.long
    )
    attention_mask = torch.ones_like(input_ids, device=tensor_device, dtype=torch.long)

    image_tensor = process_images([pil], image_processor, hf_model.config)
    if isinstance(image_tensor, list):
        image_tensor = [img.to(device=tensor_device, dtype=dtype) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(device=tensor_device, dtype=dtype)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "images": image_tensor,
        "image_sizes": [pil.size],
    }


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cuda_dev_index(model: torch.nn.Module) -> int:
    dev = next(model.parameters()).device
    if dev.type != "cuda":
        raise RuntimeError("CUDA device required for peak memory stats")
    return int(dev.index) if dev.index is not None else torch.cuda.current_device()


def benchmark_forward(
    hf_model: torch.nn.Module,
    batch_gpu: Dict[str, Any],
    n_runs: int,
    warmup: int,
) -> Tuple[float, float]:
    """Average forward time (s) and peak CUDA allocated memory (GiB) during warmup + timed runs.

    Peak uses ``torch.cuda.max_memory_allocated`` after ``reset_peak_memory_stats`` with
    model + batch already on GPU (captures activation spikes on top of weights).
    """
    hf_model.eval()
    dev_i = _cuda_dev_index(hf_model)
    torch.cuda.reset_peak_memory_stats(dev_i)
    _sync()

    for _ in range(warmup):
        with torch.inference_mode():
            hf_model(**batch_gpu)
    _sync()

    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.inference_mode():
            hf_model(**batch_gpu)
    _sync()
    t1 = time.perf_counter()

    peak_bytes = torch.cuda.max_memory_allocated(dev_i)
    peak_gib = peak_bytes / (1024.0**3)
    return (t1 - t0) / float(n_runs), peak_gib


def _verify_geometry(
    hf_model: torch.nn.Module,
    batch_cpu: Dict[str, Any],
    device: torch.device,
    *,
    expected_text_tokens: Optional[int] = None,
) -> None:
    """One forward: print total ``input_ids`` len, ``n_visual``, embed len, and text-region len."""
    b = _move_forward_batch_to_device(batch_cpu, device)
    with torch.inference_mode():
        hf_model(**b)
    inner = getattr(hf_model, "model", None)
    nv = getattr(inner, "n_visual", None) if inner is not None else None
    n_tok = int(batch_cpu["input_ids"].shape[1])
    if nv is not None:
        h = n_tok - 1 + int(nv)
        text_span = h - VISUAL_START_IDX - int(nv)
        print(
            f"[verify] input_ids_total_len={n_tok} (tokenizer, before vision expand) | "
            f"n_visual={nv} embed_seq_len={h} | text_region_len={text_span}"
        )
        if expected_text_tokens is not None and text_span != expected_text_tokens:
            raise RuntimeError(
                f"text_region_len={text_span} != --text-tokens={expected_text_tokens}; "
                "layout may differ from single-image SearchV (anyres / template)."
            )
    else:
        print("[verify] could not read model.model.n_visual after forward")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", type=str, required=True, help="HF / local LLaVA path")
    p.add_argument(
        "--skip-config",
        type=str,
        required=True,
        help="Layer skip txt (two tokens per line: MHA FFN)",
    )
    p.add_argument("--conv-template", type=str, default="v1")
    p.add_argument(
        "--text-tokens",
        type=int,
        default=64,
        help="Decoder Text-region length in tokens (exact, single-image layout); "
        "total input_ids len = text-tokens + 36 (35+1+text-tokens)",
    )
    p.add_argument(
        "--target-input-len",
        type=int,
        default=None,
        help="Override **total** input_ids length before vision expand (not text-only). "
        "If set, ignores --text-tokens for sizing; verify may fail --text-tokens check.",
    )
    p.add_argument("--runs", type=int, default=20, help="Timed forward passes per config")
    p.add_argument("--warmup", type=int, default=3, help="Warmup forwards (not timed)")
    p.add_argument("--dtype", type=str, default="float16", choices=("float16", "bfloat16", "float32"))
    p.add_argument("--verify-only", action="store_true", help="Print geometry and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.skip_config):
        raise SystemExit(f"skip-config not found: {args.skip_config}")

    skip_cfg = load_skip_config(args.skip_config)
    if not skip_cfg:
        raise SystemExit("skip-config parsed empty")

    if args.target_input_len is not None:
        target_input_len = args.target_input_len
    else:
        target_input_len = input_ids_len_for_text_tokens(args.text_tokens)

    dtype = getattr(torch, args.dtype)
    assert torch.cuda.is_available(), "CUDA required"

    def run_phase(
        env_cfg: Optional[List[Tuple[str, str]]],
        apply_cfg: Optional[List[Tuple[str, str]]],
    ) -> Tuple[float, float, Llava]:
        _set_replaced_layers_env(env_cfg)
        eval_model = Llava(
            pretrained=args.pretrained,
            conv_template=args.conv_template,
            batch_size=1,
        )
        hf = eval_model.model
        hf.config.use_cache = False
        if apply_cfg is not None:
            hf.model.update_skip_configs(apply_cfg)

        device = next(hf.parameters()).device
        batch_cpu = build_benchmark_batch(
            eval_model.tokenizer,
            eval_model._image_processor,
            hf,
            args.conv_template,
            target_input_len,
            torch.device("cpu"),
            dtype,
        )
        batch_gpu = _move_forward_batch_to_device(batch_cpu, device)
        avg_s, peak_gib = benchmark_forward(hf, batch_gpu, args.runs, args.warmup)
        return avg_s, peak_gib, eval_model

    print(
        f"[Info] text_tokens={args.text_tokens} -> input_ids_total_len={target_input_len} "
        f"(use --target-input-len to override total); runs={args.runs} warmup={args.warmup}"
    )

    if args.verify_only:
        _set_replaced_layers_env(skip_cfg)
        eval_model = Llava(
            pretrained=args.pretrained,
            conv_template=args.conv_template,
            batch_size=1,
        )
        hf = eval_model.model
        hf.config.use_cache = False
        hf.model.update_skip_configs(skip_cfg)
        device = next(hf.parameters()).device
        batch = build_benchmark_batch(
            eval_model.tokenizer,
            eval_model._image_processor,
            hf,
            args.conv_template,
            target_input_len,
            torch.device("cpu"),
            dtype,
        )
        _verify_geometry(
            hf,
            batch,
            device,
            expected_text_tokens=args.text_tokens
            if args.target_input_len is None
            else None,
        )
        return

    t_skip, peak_skip, m_skip = run_phase(skip_cfg, skip_cfg)
    del m_skip
    torch.cuda.empty_cache()
    os.environ.pop("REPLACED_LAYERS", None)

    t_base, peak_base, m_base = run_phase(None, None)
    del m_base
    torch.cuda.empty_cache()

    speedup = t_base / t_skip if t_skip > 0 else float("inf")

    print(f"[Result] skip_config avg forward: {t_skip * 1000:.3f} ms  ({args.skip_config})")
    print(
        f"[Result] skip_config peak GPU memory (warmup+runs): {peak_skip:.3f} GiB  "
        f"(torch.cuda.max_memory_allocated)"
    )
    print(f"[Result] baseline (all None) avg: {t_base * 1000:.3f} ms")
    print(
        f"[Result] baseline peak GPU memory (warmup+runs): {peak_base:.3f} GiB  "
        f"(torch.cuda.max_memory_allocated)"
    )
    print(f"[Result] speedup (baseline / skip): {speedup:.4f}x")


if __name__ == "__main__":
    main()
