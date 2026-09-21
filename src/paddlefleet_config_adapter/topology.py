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

"""Communication-group constraints for a given GPU count (C1..C5)."""

from __future__ import annotations

from .utils import multi_lcm


def min_valid_cards(tp, pp, ep, cp, sep=1, cards_per_node=8):
    """Smallest legal GPU count for these parallel degrees.

    Every legal card count is an integer multiple of this minimum, so it
    doubles as the unit for deriving a target scale (see
    ``ConfigAdapter._resolve_target_cards``).

    Returns ``None`` when the dims violate a card-count-independent rule
    (C3 or C5): no card count can rescue those.
    """
    if ep > 1 and ep % (tp * sep) != 0:
        return None
    if sep > 1 and cp > 1:
        return None
    factors = [tp * sep * pp, cards_per_node]
    if ep > 1:
        factors.append(pp * ep)
    if cp > 1:
        factors.append(tp * sep * pp * cp)
    return multi_lcm(*factors)


class TopologyValidator:
    """Validates Fleet communication-group constraints.

    Constraints (derived from PaddlePaddle's topology.py plus the trainer's
    own assertions):
      C1: N % (TP * SEP * PP) == 0 -> sharding is a positive integer
      C2: N % (PP * EP) == 0       -> moe_sharding is a positive integer
      C3: EP % (TP * SEP) == 0     -> dense_sharding is a positive integer,
          since dense_sharding = sharding / moe_sharding = EP / (TP * SEP)
      C4: sharding % CP == 0       -> cp_sharding is a positive integer
      C5: not (SEP > 1 and CP > 1) -> "sep parallel and context parallel
          cannot be used together" (PaddleFormers training_args.py)

    C2 and C4 constrain the *same* block of sharding ranks, read two different
    ways -- ``Topology`` itself has neither a cp nor an ep axis, only
    ``["dp", "pp", "sharding", "mp", "sep"]``.  The MoE view factors it as
    ``sharding = moe_sharding * dense_sharding`` with
    ``dense_sharding = EP / (TP * SEP)``, the data view as
    ``sharding = dataset_world_size * CP``.  Both factorizations have to come
    out integral, so the two conditions cannot be merged into one: they are
    different divisors of the same number.
    """

    def __init__(self, target_cards, cards_per_node=8):
        self.target_cards = target_cards
        self.cards_per_node = cards_per_node

    def validate(self, tp, pp, ep, cp, sep=1):
        """Check C1..C5. Returns ``(is_valid, message, details)``."""
        n = self.target_cards
        errors = []

        if n % (tp * sep * pp) != 0:
            errors.append(
                f"C1 不满足：{n} % (TP={tp} × SEP={sep} × PP={pp}) = "
                f"{n % (tp * sep * pp)} != 0，sharding 不是整数"
            )

        if ep > 1 and n % (pp * ep) != 0:
            errors.append(
                f"C2 不满足：{n} % (PP={pp} × EP={ep}) = {n % (pp * ep)} "
                f"!= 0，moe_sharding 不是整数"
            )

        if ep > 1 and ep % (tp * sep) != 0:
            errors.append(
                f"C3 不满足：EP={ep} % (TP={tp} × SEP={sep}) != 0，"
                f"dense_sharding = EP/(TP×SEP) 不是正整数"
            )

        if n % (tp * sep * pp) == 0:
            sharding = n // (tp * sep * pp)
            if sharding % cp != 0:
                errors.append(
                    f"C4 不满足：sharding={sharding} % CP={cp} = "
                    f"{sharding % cp} != 0，cp_sharding 不是整数"
                )

        if sep > 1 and cp > 1:
            errors.append(
                f"C5 不满足：SEP={sep} 与 CP={cp} 不能同时大于 1"
                f"（框架禁止 sep parallel 与 context parallel 同时使用）"
            )

        if errors:
            msg = "目标卡数 {} 不满足通信组约束：\n  {}".format(
                n, "\n  ".join(errors)
            )
            cards = self.suggest_valid_cards(tp, pp, ep, cp, sep)
            if cards:
                nodes = [c // self.cards_per_node for c in cards]
                msg += (
                    f"\n  16 节点内的合法节点数：{nodes}"
                    f"（每节点 {self.cards_per_node} 卡）"
                )
            else:
                msg += (
                    "\n  没有任何卡数能满足：上面的 C3/C5 只取决于并行度本身，"
                    "与卡数无关，必须先修改 TP/SEP/EP/CP 的组合"
                )
            return False, msg, {}

        sharding = n // (tp * sep * pp)
        moe_sharding = n // (pp * ep) if ep > 1 else 1
        details = {
            "sharding": sharding,
            "moe_sharding": moe_sharding,
            "dense_sharding": (
                sharding // moe_sharding if moe_sharding > 0 else sharding
            ),
            "cp_sharding": sharding // cp,
        }
        return True, "通信组约束校验通过", details

    def suggest_valid_cards(self, tp, pp, ep, cp, sep=1, max_nodes=16):
        """List every valid GPU count within ``max_nodes``.

        Returns ``[]`` when the dims violate a card-count-independent rule
        (C3 or C5): no target scale can rescue those, so offering "legal"
        node counts would be misleading.
        """
        min_unit = min_valid_cards(
            tp, pp, ep, cp, sep, cards_per_node=self.cards_per_node
        )
        if min_unit is None:
            return []

        max_cards = max_nodes * self.cards_per_node
        suggestions = [
            min_unit * k for k in range(1, max_cards // min_unit + 1)
        ]
        return suggestions or [min_unit]
