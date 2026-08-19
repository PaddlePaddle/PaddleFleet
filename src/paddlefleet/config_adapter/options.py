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

"""What the caller asked for: two orthogonal, optional test dimensions.

``--test-performance`` and ``--test-accuracy`` are independent switches, not
alternatives, and neither is required::

    (none)                              缩到目标规模即可：允许缩小 EP/PP
    --test-performance                  冻结 TP/PP/EP/CP/SEP 与 acc
    --test-accuracy                     注入避免 aadiff 的开关，保持等效 batch
    --test-performance --test-accuracy  冻结并行策略 + 注入精度开关

Each switch owns one concern:

* ``--test-performance`` freezes the parallelism and
  ``gradient_accumulation_steps`` so a step time stays comparable with the
  full-scale job -- only ``sharding`` and ``global_batch_size`` move.
* ``--test-accuracy`` pins the determinism switches in
  :mod:`paddlefleet.config_adapter.precision`, and (unless the performance
  switch already froze ``acc``) keeps the effective batch by raising ``acc``.
"""

from __future__ import annotations


class AdaptOptions:
    """Derived behaviour of the two test switches."""

    def __init__(self, test_performance=False, test_accuracy=False):
        self.test_performance = bool(test_performance)
        self.test_accuracy = bool(test_accuracy)

    @property
    def freeze_parallel(self):
        """True when no parallel dimension (nor ``acc``) may be rewritten."""
        return self.test_performance

    @property
    def inject_precision(self):
        """True when the determinism switches must be pinned."""
        return self.test_accuracy

    @property
    def batch_strategy(self):
        """Batch strategy name from ``strategies.BATCH_STRATEGIES``.

        The performance switch wins: it freezes ``acc``, so the only way to
        follow the card count is to scale ``global_batch_size``.
        """
        if self.test_accuracy and not self.test_performance:
            return "scale_accumulation"
        return "scale_batch"

    @property
    def needs_model_config(self):
        """True when planning may have to read ``model_config.json``."""
        return not self.freeze_parallel or self.inject_precision

    @property
    def label(self):
        """Short name used in the report and the generated file header."""
        if self.test_performance and self.test_accuracy:
            return "performance+accuracy"
        if self.test_performance:
            return "performance"
        if self.test_accuracy:
            return "accuracy"
        return "default"

    @property
    def flags(self):
        """The switches as typed on the command line."""
        given = []
        if self.test_performance:
            given.append("--test-performance")
        if self.test_accuracy:
            given.append("--test-accuracy")
        return " ".join(given) if given else "未指定测试维度"
