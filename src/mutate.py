from typing import Tuple, List, Set, Optional
import hashlib
import os
import random

from . import vision_skip_flops as _gs
from .vision_skip_flops import CONFIG_NAMES, calculate_layer_flops, get_flops_saved_matrix


# ---------------------------------------------------------------------------
# FLOPs helpers
# ---------------------------------------------------------------------------

_FLOPS_CACHE = {}


def clear_flops_cache() -> None:
    """Call after vision_skip_flops.set_architecture so FLOPs cache matches L/h/m/t/v."""
    _FLOPS_CACHE.clear()


def _get_flops_info():
    """Lazy-init and cache FLOPs saved matrix and base TFLOPs."""
    if "matrix" not in _FLOPS_CACHE:
        mat, base, rate = get_flops_saved_matrix(
            _gs.L, _gs.h, _gs.m, _gs.t, _gs.v
        )
        _FLOPS_CACHE["matrix"] = mat
        _FLOPS_CACHE["base"] = base
        _FLOPS_CACHE["rate"] = rate
    return _FLOPS_CACHE["matrix"], _FLOPS_CACHE["base"]


def _layer_save(layer_idx: int, mha: str, ffn: str) -> float:
    """Single-layer FLOPs saved (vs None,None)."""
    mat, _ = _get_flops_info()
    base_layer_flops = calculate_layer_flops(
        _gs.t, _gs.v, _gs.h, _gs.m, "None", "None"
    )
    if (mha, ffn) in CONFIG_NAMES:
        return mat[layer_idx, CONFIG_NAMES.index((mha, ffn))]
    layer_flops = calculate_layer_flops(_gs.t, _gs.v, _gs.h, _gs.m, mha, ffn)
    return base_layer_flops - layer_flops


def config_flops_saved(config: List[Tuple[str, str]]) -> float:
    return sum(_layer_save(i, mha, ffn) for i, (mha, ffn) in enumerate(config))


def _strict_flops_enabled() -> bool:
    return os.environ.get("MODSKIP_STRICT_FLOPS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _flops_saved_close(a: float, b: float) -> bool:
    """Whether two total-saved-TFLOPs values are equal within tolerance."""
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= max(1e-12, 1e-9 * scale)


def _configs_equal(
    a: List[Tuple[str, str]], b: List[Tuple[str, str]]
) -> bool:
    return len(a) == len(b) and all(x == y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Tabu cache  –  hash configs to avoid duplicate fitness evaluations
# ---------------------------------------------------------------------------

class TabuCache:
    def __init__(self, maxsize: int = 50000):
        self._seen: set = set()
        self._maxsize = maxsize

    @staticmethod
    def _hash(config: List[Tuple[str, str]]) -> str:
        raw = "|".join(f"{m},{f}" for m, f in config)
        return hashlib.md5(raw.encode()).hexdigest()

    def contains(self, config: List[Tuple[str, str]]) -> bool:
        return self._hash(config) in self._seen

    def add(self, config: List[Tuple[str, str]]):
        if len(self._seen) >= self._maxsize:
            self._seen.pop()
        self._seen.add(self._hash(config))

    def add_batch(self, configs: List[List[Tuple[str, str]]]):
        for c in configs:
            self.add(c)

    def __len__(self):
        return len(self._seen)


# ---------------------------------------------------------------------------
# Core mutation  –  shuffle configs among n adjacent layers (preserves FLOPs)
# ---------------------------------------------------------------------------

def mutate(
    parent_config: List[Tuple[str, str]],
    protected_indices: Set[int],
    num_layers: int,
    n: int = 4,
    max_attempts: int = 40,
) -> List[Tuple[str, str]]:
    """
    Pick n adjacent mutable layers, then with 50% probability:
    - Shuffle their whole-layer configs among the window.
    - Else: swap MHA-only or FFN-only between two layers in the window
      (e.g. (None,None)+(Vision,Vision) -> (Vision,None)+(None,Vision)).

    If ``MODSKIP_STRICT_FLOPS=1``, assert child ``config_flops_saved`` matches parent before accept.
    Retry if identical to parent; after ``max_attempts`` return parent unchanged.

    Shuffling whole-layer tuples within CONFIG_NAMES preserves total saved FLOPs under the
    homogeneous-layer ``get_flops_saved_matrix`` broadcast; MHA/FFN swaps are checked in strict mode.
    """
    valid_starts = [
        s for s in range(num_layers - n + 1)
        if all(s + i not in protected_indices for i in range(n))
    ]
    if not valid_starts:
        return list(parent_config)

    strict = _strict_flops_enabled()
    parent_saved = config_flops_saved(parent_config) if strict else 0.0

    for _ in range(max_attempts):
        child = list(parent_config)
        start = random.choice(valid_starts)
        indices = list(range(start, start + n))
        configs = [child[i] for i in indices]

        if random.random() < 0.5:
            # Branch 1: shuffle whole-layer tuples in the window
            if len(set(configs)) < 2:
                continue
            shuffled = configs.copy()
            random.shuffle(shuffled)
            for k, idx in enumerate(indices):
                child[idx] = shuffled[k]
        else:
            # Branch 2: swap MHA or FFN between two layers in the window
            if n < 2:
                continue
            i, j = random.sample(range(n), 2)
            idx_i, idx_j = indices[i], indices[j]
            mha_i, ffn_i = child[idx_i]
            mha_j, ffn_j = child[idx_j]
            dim = random.choice(["MHA", "FFN"])
            if dim == "MHA":
                if mha_i == mha_j:
                    continue
                child[idx_i] = (mha_j, ffn_i)
                child[idx_j] = (mha_i, ffn_j)
            else:
                if ffn_i == ffn_j:
                    continue
                child[idx_i] = (mha_i, ffn_j)
                child[idx_j] = (mha_j, ffn_i)

        if _configs_equal(child, parent_config):
            continue

        if strict:
            child_saved = config_flops_saved(child)
            assert _flops_saved_close(
                parent_saved, child_saved
            ), (
                "MODSKIP_STRICT_FLOPS: parent/child config_flops_saved mismatch: "
                f"parent={parent_saved:.12f} child={child_saved:.12f}"
            )

        return child

    return list(parent_config)


# ---------------------------------------------------------------------------
# Tournament selection
# ---------------------------------------------------------------------------

def tournament_select(
    population: List[List[Tuple[str, str]]],
    fitnesses: List[float],
    k: int = 3,
) -> List[Tuple[str, str]]:
    """
    Tournament selection: randomly sample *k* individuals, return the one with
    the best (lowest) fitness.
    """
    indices = random.sample(range(len(population)), min(k, len(population)))
    best = min(indices, key=lambda i: fitnesses[i])
    return population[best]


# ---------------------------------------------------------------------------
# Offspring generation (main entry point for searchv_evolution)
# ---------------------------------------------------------------------------

def generate_offspring(
    parent_configs: List[List[Tuple[str, str]]],
    parent_fitnesses: List[float],
    num_offspring: int,
    num_layers: int,
    protect_initial: int,
    protect_final: int,
    tournament_k: int = 3,
    tabu: Optional[TabuCache] = None,
    mutation_window: int = 4,
) -> List[List[Tuple[str, str]]]:
    """
    Generate *num_offspring* children via tournament selection + mutation.

    Mutation: pick n adjacent layers (not all same config), shuffle their configs.
    Preserves FLOPs; offspring similar to parent.

    Args:
        tournament_k:    tournament size for parent selection.
        tabu:            optional TabuCache to skip duplicates.
        mutation_window: n adjacent layers to shuffle in mutate.
    """
    protected = set(range(protect_initial))
    protected.update(range(num_layers - protect_final, num_layers))

    if not parent_configs:
        return []

    offspring = []
    tabu_skipped = 0
    max_tabu_retries = 3

    while len(offspring) < num_offspring:
        parent = tournament_select(parent_configs, parent_fitnesses, tournament_k)
        child = mutate(
            parent,
            protected,
            num_layers,
            n=mutation_window,
        )

        if tabu is not None and tabu.contains(child):
            tabu_skipped += 1
            if tabu_skipped > max_tabu_retries * num_offspring:
                offspring.append(child)
            continue
        offspring.append(child)
        if tabu is not None:
            tabu.add(child)

    print(
        f"[Offspring] generated={len(offspring)}, "
        f"window={mutation_window}, "
        f"tabu_skipped={tabu_skipped}, "
        f"tabu_size={len(tabu) if tabu else 'N/A'}"
    )
    return offspring


# ---------------------------------------------------------------------------
# Backward-compatible wrapper (old interface, kept for safety)
# ---------------------------------------------------------------------------

def generate_offspring_by_mutation(
    parent_configs: List[List[Tuple[str, str]]],
    num_offspring: int,
    num_layers: int,
    protect_initial: int,
    protect_final: int,
    **kwargs,
) -> List[List[Tuple[str, str]]]:
    """Legacy wrapper – delegates to generate_offspring with uniform fitnesses."""
    dummy_fitnesses = [0.0] * len(parent_configs)
    return generate_offspring(
        parent_configs=parent_configs,
        parent_fitnesses=dummy_fitnesses,
        num_offspring=num_offspring,
        num_layers=num_layers,
        protect_initial=protect_initial,
        protect_final=protect_final,
        **kwargs,
    )
