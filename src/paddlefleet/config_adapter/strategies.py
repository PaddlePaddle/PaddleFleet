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

"""How batch settings follow the data-parallel width, one per test profile.

* ``--test-performance`` -> :func:`scale_batch`: shrink ``global_batch_size``
  proportionally and keep ``gradient_accumulation_steps``, so per-step work
  per card is unchanged and the measured step time stays comparable.
* ``--test-accuracy`` -> :func:`scale_accumulation`: keep
  ``global_batch_size`` and raise ``gradient_accumulation_steps`` by the same
  factor, so the effective batch (and therefore the loss curve) matches the
  full-scale run.

Scaling must follow ``dataset_world_size`` (= cards / (TP*SEP*PP*CP)), not
the raw card count: the trainer asserts ``GBS == micro_bs * acc *
dataset_world_size``, and when the adapter also shrinks EP/PP/CP the two
ratios differ.  The caller passes the before/after data-parallel widths as
``orig_units`` / ``target_units`` (falling back to card counts, with a
matching ``unit`` label, only when a width cannot be derived).

Neither strategy touches parallelism dimensions.  Both return
``(config_map, reason, error)``; a non-empty ``error`` aborts adaptation
rather than silently rounding.
"""

from __future__ import annotations

DEFAULT_UNIT = "数据并行路数"


def scale_batch(gbs, grad_accum, orig_units, target_units, unit=DEFAULT_UNIT):
    """Shrink GBS proportionally to the DP width, keep grad_accum."""
    if gbs is None:
        return (
            {"gradient_accumulation_steps": grad_accum},
            f"源配置未写 global_batch_size，acc 保持 {grad_accum} 不变，"
            f"由框架反推 GBS",
            None,
        )

    if gbs * target_units % orig_units != 0:
        return (
            None,
            "",
            f"GBS 无法整除：{gbs} × {target_units} / {orig_units} = "
            f"{gbs * target_units / orig_units}，"
            f"请换一个能整除的目标机器规模",
        )

    new_gbs = gbs * target_units // orig_units
    if new_gbs <= 0:
        return None, "", f"缩放后 GBS={new_gbs} <= 0，目标机器规模太小"

    return (
        {
            "global_batch_size": new_gbs,
            "gradient_accumulation_steps": grad_accum,
        },
        f"GBS 按{unit}等比缩放 {gbs} × {target_units} / "
        f"{orig_units} = {new_gbs}，acc 保持 {grad_accum} 不变",
        None,
    )


def scale_accumulation(
    gbs, grad_accum, orig_units, target_units, unit=DEFAULT_UNIT
):
    """Keep GBS, raise grad_accum so the effective batch is unchanged."""
    if gbs is None:
        return (
            {"gradient_accumulation_steps": grad_accum},
            f"源配置未写 global_batch_size，acc 保持 {grad_accum} 不变",
            None,
        )

    if grad_accum * orig_units % target_units != 0:
        return (
            None,
            "",
            f"acc 无法整除：{grad_accum} × {orig_units} / {target_units} = "
            f"{grad_accum * orig_units / target_units}，"
            f"请换一个能整除的目标机器规模",
        )

    new_grad_accum = grad_accum * orig_units // target_units
    if new_grad_accum <= 0:
        return None, "", f"缩放后 acc={new_grad_accum} <= 0"

    return (
        {
            "global_batch_size": gbs,
            "gradient_accumulation_steps": new_grad_accum,
        },
        f"保持等效 batch：GBS 保持 {gbs} 不变，acc 按{unit}放大为 "
        f"{grad_accum} × {orig_units} / {target_units} = {new_grad_accum}",
        None,
    )


BATCH_STRATEGIES = {
    "scale_batch": scale_batch,
    "scale_accumulation": scale_accumulation,
}
