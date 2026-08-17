# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hardware (E) and model-structure (M) constraints, plus dim candidates.

The topology validator only covers the C1..C4 communication-group rules.
Shrinking EP / PP additionally requires:

* E-family -- hardware / allocation prerequisites (:func:`check_hardware`)
* M-family -- model-structure feasibility (:func:`check_ep_shrink`,
  :func:`check_pp_shrink`)

Hard rule shared by every candidate generator: **a parallel degree that is
greater than 1 in the source config is never collapsed to 1.** Dropping a
dimension entirely (EP 8 -> 1) removes the communication group under test,
so the smallest legal shrink target is :data:`MIN_PARALLEL_DEGREE`.

All functions here are pure: no I/O, no mutation of the arguments.
"""

from __future__ import annotations

from .topology import TopologyValidator

DEFAULT_MIN_HIDDEN_LAYERS = 4

#: A source degree > 1 may shrink no further than this.
MIN_PARALLEL_DEGREE = 2


def divisors(n):
    """Ascending list of positive divisors of ``n``."""
    if n <= 0:
        return []
    out = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            out.append(i)
            if i != n // i:
                out.append(n // i)
        i += 1
    return sorted(out)


def shrink_floor(degree):
    """Smallest degree ``degree`` is allowed to shrink to."""
    return 1 if degree <= 1 else MIN_PARALLEL_DEGREE


def floor_dims(tp, pp, ep, cp, sep):
    """Dims after shrinking EP / PP as far as the no-collapse rule allows."""
    return tp, shrink_floor(pp), shrink_floor(ep), cp, sep


def ep_candidates(ep_orig, tp, sep=1):
    """EP-shrink candidates, largest first.

    Set: ``({powers of 2} | {multiples of 8})`` restricted to
    ``[MIN_PARALLEL_DEGREE, ep_orig)`` and filtered by
    ``ep_new % (tp * sep) == 0`` (C3: dense_sharding = EP / (TP * SEP)).
    Real deployments only ever pick EP from those two families, which keeps
    the search space small and avoids exotic values like EP=3.
    """
    if ep_orig <= MIN_PARALLEL_DEGREE:
        return []
    powers_of_2 = {1 << k for k in range(ep_orig.bit_length())}
    multiples_of_8 = {8 * k for k in range(1, ep_orig // 8 + 1)}
    pool = (powers_of_2 | multiples_of_8) & set(
        range(MIN_PARALLEL_DEGREE, ep_orig)
    )
    dense_factor = tp * sep
    return sorted(
        (c for c in pool if dense_factor == 1 or c % dense_factor == 0),
        reverse=True,
    )


def pp_candidates(pp_orig):
    """PP-shrink candidates, largest first.

    Divisors of ``pp_orig`` in ``[MIN_PARALLEL_DEGREE, pp_orig)``.  Requiring
    divisibility keeps ``num_hidden_layers * pp_new // pp_orig`` an integer;
    the residual VPP / tail padding is handled by :func:`align_layers`.
    """
    if pp_orig <= MIN_PARALLEL_DEGREE:
        return []
    return sorted(
        (d for d in divisors(pp_orig) if MIN_PARALLEL_DEGREE <= d < pp_orig),
        reverse=True,
    )


def align_layers(layers_new, head, tail_orig, pp_new, vpp_orig):
    """Align pipeline layers to ``pp_new * vpp_new`` with empty tail layers.

    Returns ``(vpp_new, tail_new)``, or ``(None, None)`` when no divisor of
    ``vpp_orig`` can be aligned.  ``pp_new`` is always >= 2 here because a
    source PP > 1 never collapses to 1.
    """
    if pp_new < MIN_PARALLEL_DEGREE:
        return None, None
    if vpp_orig is None or vpp_orig < 1:
        vpp_orig = 1
    for vpp_new in sorted(divisors(vpp_orig), reverse=True):
        divisor = pp_new * vpp_new
        pad = (-(layers_new + head + tail_orig)) % divisor
        if pad < divisor:
            return vpp_new, tail_orig + pad
    return None, None


def min_shrink_cards(tp, pp, ep, cp, sep, cards_per_node):
    """Smallest GPU count reachable without collapsing any degree to 1."""
    validator = TopologyValidator(cards_per_node, cards_per_node)
    return validator.suggest_valid_cards(*floor_dims(tp, pp, ep, cp, sep))[0]


def check_hardware(target_cards, cards_per_node, tp, pp, ep, cp, sep):
    """E1/E2/E3 pre-checks, run once per plan. Returns ``(ok, reason)``.

    A failure aborts the whole plan (rather than skipping one candidate):
    these depend only on the fixed inputs.
    """
    for name, value in (
        ("tp", tp),
        ("pp", pp),
        ("ep", ep),
        ("cp", cp),
        ("sep", sep),
    ):
        if not isinstance(value, int) or value < 1:
            return False, f"E3 不满足：{name}={value!r} 必须是 >=1 的整数"

    # E1: Fleet's rank mapping assumes symmetric per-node allocation, so a
    # multi-machine job must take whole nodes.
    if target_cards > cards_per_node and target_cards % cards_per_node != 0:
        return False, (
            f"E1 不满足：target_cards={target_cards} 超过单机 "
            f"{cards_per_node} 卡但不是其整数倍；跨机必须整节点分配"
        )

    # E2: TP*SEP must fit into one node -- neither is ever shrunk, because a
    # smaller TP raises per-card memory and risks OOM.
    max_intra = min(cards_per_node, target_cards)
    if tp * sep > max_intra:
        min_cards = min_shrink_cards(tp, pp, ep, cp, sep, cards_per_node)
        return False, (
            f"E2 不满足：TP={tp} × SEP={sep} 需要单机内 {tp * sep} 张卡，"
            f"但目标只有 {target_cards} 张卡。\n"
            f"  说明：适配不会修改 TP/SEP（减小 TP 会增大单卡显存，"
            f"有 OOM 风险）。\n"
            f"  建议：至少使用 {min_cards} 张卡 "
            f"（--target-nodes {max(min_cards // cards_per_node, 1)}）"
        )

    return True, ""


def check_ep_shrink(ep_orig, ep_new, num_experts, num_experts_per_tok):
    """M1 + M2 for one ``ep_new``.

    Returns ``(ok, reason, experts_new)``.  ``experts_new`` is meaningful even
    on failure so callers can log it.
    """
    if ep_new >= ep_orig:
        return True, "", num_experts

    experts_new = num_experts * ep_new // ep_orig

    # M1: every EP rank must hold the same number of experts.
    if experts_new % ep_new != 0:
        return (
            False,
            f"M1 不满足：experts_new={experts_new} 不能被 ep_new={ep_new} 整除",
            experts_new,
        )

    # M2: the routing width must remain satisfiable.
    if experts_new < num_experts_per_tok:
        return (
            False,
            f"M2 不满足：experts_new={experts_new} < "
            f"num_experts_per_tok={num_experts_per_tok}",
            experts_new,
        )

    return True, "", experts_new


def check_pp_shrink(
    pp_orig,
    pp_new,
    num_hidden_layers,
    head,
    tail,
    vpp,
    first_k_dense_replace=0,
    min_hidden_layers=DEFAULT_MIN_HIDDEN_LAYERS,
):
    """M3 + M4 + M5 for one ``pp_new``.

    Returns ``(ok, reason, meta)`` where ``meta`` carries ``layers_new``,
    ``vpp_new``, ``tail_new`` and an optional soft ``warning``.  M4 is a
    warning rather than a rejection; the hard floor is one hidden layer.
    """
    if pp_new >= pp_orig:
        return (
            True,
            "",
            {
                "layers_new": num_hidden_layers,
                "vpp_new": vpp,
                "tail_new": tail,
                "warning": None,
            },
        )

    layers_new = num_hidden_layers * pp_new // pp_orig
    meta = {
        "layers_new": layers_new,
        "vpp_new": None,
        "tail_new": None,
        "warning": None,
    }

    if layers_new < 1:
        return False, f"layers_new={layers_new}，模型至少需要 1 层", meta

    if layers_new < min_hidden_layers:
        meta["warning"] = (
            f"num_hidden_layers 缩减至 {layers_new}"
            f"（推荐 >= {min_hidden_layers}），仅适合调试，训练效果不保证"
        )

    # M5: the dense-only prefix must still fit in the shrunk stack.
    if layers_new < first_k_dense_replace:
        return (
            False,
            f"M5 不满足：layers_new={layers_new} < "
            f"first_k_dense_replace={first_k_dense_replace}",
            meta,
        )

    # M3: layer alignment feasibility.
    vpp_new, tail_new = align_layers(layers_new, head, tail, pp_new, vpp)
    if vpp_new is None:
        return (
            False,
            f"M3 不满足：(layers_new={layers_new} + head={head} + "
            f"tail>={tail}) 无法对齐到 pp_new={pp_new} × vpp_new",
            meta,
        )

    meta["vpp_new"] = vpp_new
    meta["tail_new"] = tail_new
    return True, "", meta
