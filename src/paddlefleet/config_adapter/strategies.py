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

"""How batch settings follow the GPU count. One strategy per test profile.

* ``--test-performance`` -> :func:`scale_batch`: shrink ``global_batch_size``
  proportionally and keep ``gradient_accumulation_steps``, so per-step work
  per card is unchanged and the measured step time stays comparable.
* ``--test-accuracy`` -> :func:`scale_accumulation`: keep
  ``global_batch_size`` and raise ``gradient_accumulation_steps`` by the same
  factor, so the effective batch (and therefore the loss curve) matches the
  full-scale run.

Neither strategy touches parallelism dimensions.  Both return
``(config_map, reason, error)``; a non-empty ``error`` aborts adaptation
rather than silently rounding.
"""

from __future__ import annotations


def scale_batch(gbs, grad_accum, orig_cards, target_cards):
    """Shrink GBS proportionally to the card count, keep grad_accum."""
    if gbs is None:
        return (
            {"gradient_accumulation_steps": grad_accum},
            f"源配置未写 global_batch_size，acc 保持 {grad_accum} 不变，"
            f"由框架反推 GBS",
            None,
        )

    if gbs * target_cards % orig_cards != 0:
        return (
            None,
            "",
            f"GBS 无法整除：{gbs} × {target_cards} / {orig_cards} = "
            f"{gbs * target_cards / orig_cards}，"
            f"请换一个能整除的目标机器规模",
        )

    new_gbs = gbs * target_cards // orig_cards
    if new_gbs <= 0:
        return None, "", f"缩放后 GBS={new_gbs} <= 0，目标机器规模太小"

    return (
        {
            "global_batch_size": new_gbs,
            "gradient_accumulation_steps": grad_accum,
        },
        f"GBS 按卡数等比缩放 {gbs} × {target_cards} / "
        f"{orig_cards} = {new_gbs}，acc 保持 {grad_accum} 不变",
        None,
    )


def scale_accumulation(gbs, grad_accum, orig_cards, target_cards):
    """Keep GBS, raise grad_accum so the effective batch is unchanged."""
    if gbs is None:
        return (
            {"gradient_accumulation_steps": grad_accum},
            f"源配置未写 global_batch_size，acc 保持 {grad_accum} 不变",
            None,
        )

    if grad_accum * orig_cards % target_cards != 0:
        return (
            None,
            "",
            f"acc 无法整除：{grad_accum} × {orig_cards} / {target_cards} = "
            f"{grad_accum * orig_cards / target_cards}，"
            f"请换一个能整除的目标机器规模",
        )

    new_grad_accum = grad_accum * orig_cards // target_cards
    if new_grad_accum <= 0:
        return None, "", f"缩放后 acc={new_grad_accum} <= 0"

    return (
        {
            "global_batch_size": gbs,
            "gradient_accumulation_steps": new_grad_accum,
        },
        f"保持等效 batch：GBS 保持 {gbs} 不变，acc 放大为 {grad_accum} × "
        f"{orig_cards} / {target_cards} = {new_grad_accum}",
        None,
    )


BATCH_STRATEGIES = {
    "scale_batch": scale_batch,
    "scale_accumulation": scale_accumulation,
}
