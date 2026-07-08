# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from collections.abc import Iterable

import paddle


INDEXCACHE_STATE_KIND_NONE = "none"
INDEXCACHE_STATE_KIND_TOPK_ONLY = "topk_only"
INDEXCACHE_STATE_KIND_DISTILL = "distill"
INDEXCACHE_STATE_KIND_INVALID = "invalid"

INDEXCACHE_TOPK_ONLY_STATE_LEN = 3
INDEXCACHE_DISTILL_STATE_LEN = 8
INDEXCACHE_RECOMPUTE_STATE_MAX_LEN = INDEXCACHE_DISTILL_STATE_LEN

INDEXCACHE_STATE_TOPK_IDXS = 0
INDEXCACHE_TOPK_ONLY_STATE_PRODUCER_LAYER = 1
INDEXCACHE_TOPK_ONLY_STATE_SERVED_COUNT = 2

INDEXCACHE_DISTILL_STATE_Q = 1
INDEXCACHE_DISTILL_STATE_WEIGHTS = 2
INDEXCACHE_DISTILL_STATE_K = 3
INDEXCACHE_DISTILL_STATE_TOPK_INDICES = 4
INDEXCACHE_DISTILL_STATE_TOPK_PROBS = 5
INDEXCACHE_DISTILL_STATE_PRODUCER_LAYER = 6
INDEXCACHE_DISTILL_STATE_SERVED_COUNT = 7

INDEXCACHE_DISTILL_GRAD_INDICES = (
    INDEXCACHE_DISTILL_STATE_Q,
    INDEXCACHE_DISTILL_STATE_WEIGHTS,
    INDEXCACHE_DISTILL_STATE_K,
)


def state_kind(indexcache_state: tuple | list | None) -> str:
    if not indexcache_state:
        return INDEXCACHE_STATE_KIND_NONE
    state_len = len(indexcache_state)
    if state_len == INDEXCACHE_TOPK_ONLY_STATE_LEN:
        return INDEXCACHE_STATE_KIND_TOPK_ONLY
    if state_len == INDEXCACHE_DISTILL_STATE_LEN:
        return INDEXCACHE_STATE_KIND_DISTILL
    return INDEXCACHE_STATE_KIND_INVALID


def is_valid_state(indexcache_state: tuple | list | None) -> bool:
    return state_kind(indexcache_state) in (
        INDEXCACHE_STATE_KIND_NONE,
        INDEXCACHE_STATE_KIND_TOPK_ONLY,
        INDEXCACHE_STATE_KIND_DISTILL,
    )


def _as_state_tuple(indexcache_state: tuple | list | None) -> tuple | None:
    if indexcache_state is None:
        return None
    return tuple(indexcache_state)


def apply_stop_gradient_mask(
    indexcache_state: tuple | list | None,
) -> tuple | None:
    indexcache_state = _as_state_tuple(indexcache_state)
    kind = state_kind(indexcache_state)
    if kind == INDEXCACHE_STATE_KIND_NONE:
        return None
    if kind == INDEXCACHE_STATE_KIND_INVALID:
        raise ValueError(
            "IndexCache state must be either topk-only "
            f"({INDEXCACHE_TOPK_ONLY_STATE_LEN} tensors) or distill "
            f"({INDEXCACHE_DISTILL_STATE_LEN} tensors), got "
            f"len={len(indexcache_state)}."
        )

    for idx, tensor in enumerate(indexcache_state):
        if not isinstance(tensor, paddle.Tensor):
            raise TypeError(
                "IndexCache state entries must be Paddle tensors, got "
                f"type={type(tensor).__name__} at index={idx}."
            )
        tensor.stop_gradient = not (
            kind == INDEXCACHE_STATE_KIND_DISTILL
            and idx in INDEXCACHE_DISTILL_GRAD_INDICES
        )
    return indexcache_state


def detach_stop_gradient_tensor(value: paddle.Tensor) -> paddle.Tensor:
    value = value.detach()
    value.stop_gradient = True
    return value


def clone_state_outputs(indexcache_state: tuple | list | None) -> tuple | None:
    masked = apply_stop_gradient_mask(indexcache_state)
    if masked is None:
        return None
    cloned = tuple(tensor.clone() for tensor in masked)
    return apply_stop_gradient_mask(cloned)


def flatten_state(indexcache_state: tuple | list | None) -> tuple:
    masked = apply_stop_gradient_mask(indexcache_state)
    if masked is None:
        return ()
    return masked


def state_to_slots(indexcache_state: tuple | list | None) -> tuple:
    slots = list(flatten_state(indexcache_state))
    if len(slots) > INDEXCACHE_RECOMPUTE_STATE_MAX_LEN:
        raise ValueError(
            "IndexCache recompute state has too many slots: "
            f"{len(slots)} > {INDEXCACHE_RECOMPUTE_STATE_MAX_LEN}."
        )
    slots.extend([None] * (INDEXCACHE_RECOMPUTE_STATE_MAX_LEN - len(slots)))
    return tuple(slots)


def state_from_slots(slots: Iterable) -> tuple | None:
    state = list(slots)
    while state and state[-1] is None:
        state.pop()
    if not state:
        return None
    return apply_stop_gradient_mask(tuple(state))
