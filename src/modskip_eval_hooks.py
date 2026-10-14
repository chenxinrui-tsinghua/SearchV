"""
Deterministic lmms-eval doc subsampling: load_state_and_eval registers indices before
simple_evaluate; SearchV/lmms-eval utils.create_iterator consumes them when the stack
matches. Independent of model weights so different runs share the same fitness subset.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

_indices_by_task: Dict[str, List[int]] = {}
_tasks_by_name: Dict[str, Any] = {}
_stack: List[str] = []


def register_shuffle_subset(task_name: str, indices: List[int], task_obj: Any) -> None:
    """Register shuffled doc indices keyed by task.config.task (matches lmms-eval stack)."""
    _indices_by_task[task_name] = list(indices)
    _tasks_by_name[task_name] = task_obj


def clear_shuffle_subset() -> None:
    _indices_by_task.clear()
    _tasks_by_name.clear()
    _stack.clear()


def push(task_name: str) -> None:
    _stack.append(task_name)


def pop() -> None:
    if _stack:
        _stack.pop()


def top() -> Optional[str]:
    return _stack[-1] if _stack else None


def registered_subset_len(task_name: str) -> Optional[int]:
    if task_name in _indices_by_task:
        return len(_indices_by_task[task_name])
    return None


def maybe_subset_iterator(
    raw_iterator: Any,
    rank: int,
    world_size: int,
    limit: Optional[int],
) -> Optional[Iterator[Tuple[int, Any]]]:
    """
    If the stack-top task has registered indices, yield (doc_id, doc) in that order.
    Return None to use default lmms-eval islice behavior.
    """
    if rank != 0 or world_size != 1 or limit is None:
        return None
    tname = top()
    if not tname or tname not in _indices_by_task:
        return None
    task = _tasks_by_name.get(tname)
    if task is None:
        return None
    idxs = _indices_by_task[tname]
    edocs = task.eval_docs_no_media

    def _gen() -> Iterator[Tuple[int, Any]]:
        for i in idxs:
            yield (i, edocs[i])

    return _gen()
