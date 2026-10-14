"""FLOPs estimates and parent configs for vision-centric layer skip (SearchV-style)."""
import numpy as np
import random
from typing import List, Optional, Tuple

# Defaults aligned with LLaVA-7B; override via searchv_evolution --arch_* or set_architecture.
L = 32  # num layers
h = 4096  # hidden size
m = 11008  # FFN intermediate (~2.7 * h)
t = 64  # text tokens (FLOPs model)
v = 576  # vision tokens (1.5 single image; 1.6 anyres approximate -- tune --arch_vision_tokens)
p = 0  # prefix tokens (fixed; not exposed in CLI)
NUM_LAYERS = 32


def set_architecture(
    num_layers: Optional[int] = None,
    hidden_size: Optional[int] = None,
    intermediate_size: Optional[int] = None,
    text_tokens: Optional[int] = None,
    vision_tokens: Optional[int] = None,
) -> None:
    """Override global L/h/m/t/v before first FLOPs / parent generation (called from searchv_evolution)."""
    global L, NUM_LAYERS, h, m, t, v
    if num_layers is not None:
        L = NUM_LAYERS = int(num_layers)
    if hidden_size is not None:
        h = int(hidden_size)
    if intermediate_size is not None:
        m = int(intermediate_size)
    if text_tokens is not None:
        t = int(text_tokens)
    if vision_tokens is not None:
        v = int(vision_tokens)
    # Invalidate mutate FLOPs cache after arch change (avoids stale config_flops_saved / sparsity).
    try:
        from . import mutate as _mutate  # noqa: PLC0415 — break circular import at runtime

        _mutate.clear_flops_cache()
    except (ImportError, AttributeError):
        pass


# Baseline palette: two region-skip modes only (vision / none)
CONFIG_NAMES = [("Vision", "Vision"), ("None", "None")]

# SearchV layer importance (least important first); set by --searchv_importance_ranking; None disables.
SEARCHV_IMPORTANCE_RANKING = None

# Index of (None, None); saved FLOPs must be zero.
NONE_NONE_IDX = CONFIG_NAMES.index(("None", "None"))


def sparsity_from_num_skip_vision_vision(num_skip: int) -> float:
    """
    Map count of (Vision, Vision) layers to target_percent of BASE_TFLOPs (same as ranking branch
    in generate_parent_configurations).
    """
    if num_skip < 0:
        raise ValueError("num_skip must be >= 0")
    if num_skip > NUM_LAYERS:
        raise ValueError(f"num_skip={num_skip} exceeds NUM_LAYERS={NUM_LAYERS}")
    FLOPS_SAVED_MATRIX, BASE_TFLOPS, _ = get_flops_saved_matrix(L, h, m, t, v)
    vision_vision_idx = CONFIG_NAMES.index(("Vision", "Vision"))
    saved_per_layer = float(FLOPS_SAVED_MATRIX[0, vision_vision_idx])
    return (num_skip * saved_per_layer) / BASE_TFLOPS


def calculate_layer_flops(
    t: int,
    v: int,
    h: int,
    m: int,
    mha_strategy: str,
    ffn_strategy: str,
    p: int = p,
) -> float:
    """Per-layer TFLOPs for one (mha, ffn) skip pair."""
    SCALE_FACTOR = 1e-12
    S_total = p + v + t  # sequence length incl. prefix

    # FFN: prefix always computed (LlavaLlama-style)
    if ffn_strategy == "None":
        S_ffn = S_total
    elif ffn_strategy == "Vision":
        S_ffn = p + t
    elif ffn_strategy == "Text":
        S_ffn = p + v
    elif ffn_strategy == "All":
        S_ffn = p
    else:
        S_ffn = S_total

    flops_ffn = 2 * S_ffn * (3 * m * h)

    # MHA projections: K,V full sequence
    flops_kv = 4 * S_total * (h * h)

    if mha_strategy == "None":
        S_qo = S_total
    elif mha_strategy == "Vision":
        S_qo = p + t
    elif mha_strategy == "Text":
        S_qo = p + v
    elif mha_strategy == "All":
        S_qo = p
    else:
        S_qo = S_total

    flops_qo = 4 * S_qo * (h * h)
    flops_mha_proj = flops_kv + flops_qo

    # Attention scores + context: 4 * S_Q * S_K * h
    flops_mha_matrix = 4 * S_qo * S_total * h

    total_flops = flops_ffn + flops_mha_proj + flops_mha_matrix
    return total_flops * SCALE_FACTOR


def get_flops_saved_matrix(
    L: int, h: int, m: int, t: int, v: int
) -> Tuple[np.ndarray, float]:
    """
    Absolute TFLOPs saved per layer per config vs (None, None), and baseline model TFLOPs.

    Returns:
        FLOPS_SAVED_MATRIX: (L, len(CONFIG_NAMES)) absolute savings per layer (broadcast).
        BASE_TFLOPS: full model baseline TFLOPs.
        FLOPS_SAVED_RATE_MATRIX: savings / BASE_TFLOPS per entry.
    """
    flops_base_layer = calculate_layer_flops(t, v, h, m, "None", "None")
    BASE_TFLOPS = L * flops_base_layer

    FLOPS_SAVED_MATRIX = np.zeros((L, len(CONFIG_NAMES)), dtype=np.float64)
    FLOPS_SAVED_RATE_MATRIX = np.zeros((L, len(CONFIG_NAMES)), dtype=np.float64)

    for config_idx, (mha_strategy, ffn_strategy) in enumerate(CONFIG_NAMES):
        flops_config_layer = calculate_layer_flops(t, v, h, m, mha_strategy, ffn_strategy)
        layer_saved_flops = flops_base_layer - flops_config_layer
        FLOPS_SAVED_MATRIX[:, config_idx] = layer_saved_flops
        FLOPS_SAVED_RATE_MATRIX[:, config_idx] = layer_saved_flops / BASE_TFLOPS

    FLOPS_SAVED_MATRIX[:, NONE_NONE_IDX] = 0.0
    FLOPS_SAVED_RATE_MATRIX[:, NONE_NONE_IDX] = 0.0

    return FLOPS_SAVED_MATRIX, BASE_TFLOPS, FLOPS_SAVED_RATE_MATRIX


def generate_parent_configurations(
    target_percent: float,
    num_parents: int,
    num_initial_protected_layers: int,
    num_final_protected_layers: int,
    max_tolerance_percent: float = 0.1,
    min_percent_threshold: float = 0.1,
) -> List[List[Tuple[str, str]]]:
    """Sample parent layer-skip configs near target FLOPs savings with protected head/tail layers."""
    FLOPS_SAVED_MATRIX, BASE_TFLOPS, _ = get_flops_saved_matrix(L, h, m, t, v)

    if (
        num_initial_protected_layers == 0
        and num_final_protected_layers == 0
        and SEARCHV_IMPORTANCE_RANKING is not None
    ):
        print("Using SearchV importance ranking for deterministic initialization...")

        target_flops_saved = BASE_TFLOPS * target_percent
        vision_vision_idx = CONFIG_NAMES.index(("Vision", "Vision"))
        saved_per_layer = FLOPS_SAVED_MATRIX[0, vision_vision_idx]
        num_layers_to_skip = int(round(target_flops_saved / saved_per_layer))

        max_available_layers = len(SEARCHV_IMPORTANCE_RANKING)
        if num_layers_to_skip > max_available_layers:
            print(
                f"Warning: need to skip {num_layers_to_skip} layers but ranking has only "
                f"{max_available_layers}; filling remainder randomly."
            )

            base_config = [("None", "None")] * NUM_LAYERS
            skip_layers = SEARCHV_IMPORTANCE_RANKING.copy()
            remaining_layers = [i for i in range(NUM_LAYERS) if i not in skip_layers]
            additional_needed = num_layers_to_skip - max_available_layers
            additional_skip = random.sample(
                remaining_layers, min(additional_needed, len(remaining_layers))
            )
            skip_layers.extend(additional_skip)
        else:
            base_config = [("None", "None")] * NUM_LAYERS
            skip_layers = SEARCHV_IMPORTANCE_RANKING[:num_layers_to_skip]

        for layer_idx in skip_layers:
            base_config[layer_idx] = ("Vision", "Vision")

        actual_saved = sum(
            FLOPS_SAVED_MATRIX[i, vision_vision_idx]
            for i, (m, f) in enumerate(base_config)
            if (m, f) == ("Vision", "Vision")
        )

        print(f"Skipping {num_layers_to_skip} layers: {skip_layers}")
        print(
            f"Actual saved: {actual_saved / BASE_TFLOPS * 100:.2f}% TFLOPs "
            f"(target {target_percent * 100:.2f}%)"
        )

        parent_configs = []
        for p in range(num_parents):
            if p == 0:
                parent_configs.append(base_config.copy())
            else:
                varied_config = base_config.copy()
                skip_indices = [i for i, (m, f) in enumerate(base_config) if (m, f) == ("Vision", "Vision")]
                non_skip_indices = [i for i, (m, f) in enumerate(base_config) if (m, f) == ("None", "None")]

                if len(skip_indices) > 0 and len(non_skip_indices) > 0:
                    swap_out = random.choice(skip_indices)
                    swap_in = random.choice(non_skip_indices)
                    varied_config[swap_out] = ("None", "None")
                    varied_config[swap_in] = ("Vision", "Vision")

                parent_configs.append(varied_config)

        return parent_configs

    target_flops_saved = BASE_TFLOPS * target_percent
    min_flops_saved = BASE_TFLOPS * (target_percent - min_percent_threshold)
    max_flops_saved = BASE_TFLOPS * (target_percent + max_tolerance_percent)

    protected_indices = set(range(num_initial_protected_layers))
    protected_indices.update(range(NUM_LAYERS - num_final_protected_layers, NUM_LAYERS))

    searchable_layers = [i for i in range(NUM_LAYERS) if i not in protected_indices]

    parent_configs = []
    p = 0
    while p < num_parents:
        current_config = [None] * NUM_LAYERS
        current_flops_saved = 0.0

        if SEARCHV_IMPORTANCE_RANKING is not None:
            searchable_layers_sorted = [i for i in SEARCHV_IMPORTANCE_RANKING if i in searchable_layers]
            remaining = [i for i in searchable_layers if i not in searchable_layers_sorted]
            random.shuffle(remaining)
            searchable_layers = searchable_layers_sorted + remaining
        else:
            middle_start = NUM_LAYERS // 4
            middle_end = NUM_LAYERS * 3 // 4
            middle_layers = [i for i in searchable_layers if middle_start <= i < middle_end]
            non_middle_layers = [i for i in searchable_layers if i < middle_start or i >= middle_end]
            random.shuffle(middle_layers)
            random.shuffle(non_middle_layers)
            searchable_layers = middle_layers + non_middle_layers

        for idx in protected_indices:
            current_config[idx] = CONFIG_NAMES[NONE_NONE_IDX]

        for layer_idx in searchable_layers:
            remaining_max_flops = max_flops_saved - current_flops_saved
            feasible_configs: List[Tuple[int, float, float]] = []

            for config_idx, _config_name in enumerate(CONFIG_NAMES):
                flops = FLOPS_SAVED_MATRIX[layer_idx, config_idx]
                if 0 <= flops <= remaining_max_flops:
                    layers_left = len(searchable_layers) - searchable_layers.index(layer_idx)
                    avg_needed = (target_flops_saved - current_flops_saved) / layers_left
                    dist = abs(flops - avg_needed)
                    weight = 1.0 / (dist + 1e-6)
                    feasible_configs.append((config_idx, flops, weight))

            if not feasible_configs:
                selected_idx = NONE_NONE_IDX
            else:
                indices = [c[0] for c in feasible_configs]
                weights = [c[2] for c in feasible_configs]
                selected_idx = random.choices(indices, weights=weights, k=1)[0]
            selected_flops = FLOPS_SAVED_MATRIX[layer_idx, selected_idx]
            current_flops_saved += selected_flops
            current_config[layer_idx] = CONFIG_NAMES[selected_idx]

        if min_flops_saved <= current_flops_saved <= max_flops_saved:
            parent_configs.append(current_config)
            print(
                f"Parent {p + 1}/{num_parents} ok. "
                f"Saved {current_flops_saved / BASE_TFLOPS * 100:.2f}% TFLOPs."
            )
            p += 1

    return parent_configs
