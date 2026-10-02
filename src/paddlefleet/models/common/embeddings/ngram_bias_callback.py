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

"""Aux-loss-free sub-table balancer of the N-gram MoE embedding.

This is the N-gram embedding counterpart of the ``noaux_tc`` bias callback used
by the main MoE.  The mechanism is DeepSeek-V3's: instead of adding a
gradient-producing auxiliary loss, a per-sub-table bias is added to the
*selection* score only, and after every optimizer step it is nudged by a fixed
amount against the observed load::

    bias += sign(mean_load - load_i) * update_rate

The load counter is accumulated inside ``NgramMoeEmbedding.forward`` and only
sees this rank's tokens, so it has to be all-reduced over the parallel groups
that split the batch before the sign is taken -- otherwise each rank would drive
the bias with its own local imbalance and the copies would diverge.  The counter
is zeroed here, which makes the update strictly per-step.

Required whenever ``ngram_moe_balance_type="noaux_bias"`` (the default): the
embedding allocates the bias but never moves it, so without this callback the
configuration silently degrades to ``"none"``.

Usage (PaddleFormers TrainerCallback interface):
    from paddlefleet.models.common.embeddings.ngram_bias_callback import (
        NgramBiasAdjustCallback,
    )
    callback = NgramBiasAdjustCallback(
        update_rate=config.ngram_moe_bias_update_rate,
    )
    trainer.add_callback(callback)

Unlike ``MoECorrectionBiasAdjustCallback``, which is handed ``use_mp=args.sequence_parallel``, ``use_mp`` stays False here: the N-gram signal is fused before the sequence-parallel scatter, so every model-parallel rank has already counted the whole sequence.

Or standalone after each optimizer step:
    callback.on_train_begin(model=model)
    callback.on_optimizer_end()
"""

from __future__ import annotations

import logging

import paddle
import paddle.distributed as dist

__all__ = ["NgramBiasAdjustCallback", "find_ngram_balance_statics"]

logger = logging.getLogger(__name__)


def find_ngram_balance_statics(model):
    """Locate the sub-layer holding ``table_bias`` / ``table_usage``.

    Returns ``None`` when the model has no N-gram embedding, or has one that is not running the ``noaux_bias`` style; in both cases the sub-layer is never constructed and this callback becomes a no-op.
    """
    if model is None:
        return None
    candidates = [model]
    if hasattr(model, "named_sublayers"):
        candidates += [layer for _, layer in model.named_sublayers()]
    for layer in candidates:
        bias = getattr(layer, "table_bias", None)
        usage = getattr(layer, "table_usage", None)
        if bias is not None and usage is not None:
            return layer
    return None


class NgramBiasAdjustCallback:
    """Updates the N-gram sub-table selection bias once per optimizer step."""

    def __init__(self, update_rate: float = 1e-4, use_mp: bool = False):
        self.update_rate = float(update_rate)
        self.use_mp = bool(use_mp)
        self.statics = None
        self._warned = False

    def on_train_begin(
        self, args=None, state=None, control=None, model=None, **kwargs
    ):
        """Bind to the balancer sub-layer; a no-op when there is none."""
        self.statics = find_ngram_balance_statics(model)
        if self.statics is None:
            logger.warning(
                "[ngram_bias] no layer carrying table_bias was found on this "
                "rank; if ngram_moe_balance_type=noaux_bias, the sub-table load "
                "is NOT being balanced."
            )
            return
        logger.info(
            "[ngram_bias] aux-loss-free sub-table balancing active, "
            "update_rate=%s, bias shape=%s.",
            self.update_rate,
            list(self.statics.table_bias.shape),
        )

    def on_optimizer_end(self, args=None, state=None, control=None, **kwargs):
        """Called after optimizer.step() -- the bias update entry point."""
        if self.statics is None:
            # Late binding: the model is passed on every callback invocation.
            self.statics = find_ngram_balance_statics(kwargs.get("model"))
            if self.statics is None:
                return
        if getattr(args, "freeze_training", False):
            if not self._warned:
                logger.warning(
                    "[ngram_bias] freeze_training is enabled; the sub-table "
                    "bias will NOT be updated."
                )
                self._warned = True
            return

        # fp32 for the reduction and the mean: the counts stay far inside fp32's
        # exact-integer range, and an int64 mean would truncate the threshold
        # that sign() is taken against.
        usages = self.statics.table_usage.astype("float32")
        for group in _try_get_comm_groups(self.use_mp):
            dist.all_reduce(usages, group=group)

        with paddle.no_grad():
            # The sub-tables of one order compete only with each other.
            usages_mean = usages.mean(axis=-1, keepdim=True)
            update = paddle.sign(usages_mean - usages) * self.update_rate
            self.statics.table_bias.add_(update)
            self.statics.table_usage.zero_()


def _try_get_comm_groups(use_mp: bool):
    """Groups to all-reduce the per-sub-table usage counts over.

    These are the groups whose ranks hold a different slice of the batch, so their counts must be summed to recover global-batch load.
    ``None`` means the default (world) group; an empty list means there is nothing to reduce.

    Model parallel is offered for symmetry with the MoE callback but should stay off: this embedding runs before the sequence-parallel scatter, so its counters already cover the whole sequence on every model-parallel rank even when sequence parallel is on.
    Summing over them would scale every count by the same factor, which ``sign()`` ignores but which makes the numbers meaningless to log.
    """
    from paddle.distributed import fleet

    if not dist.is_initialized():
        return []
    if not hasattr(fleet, "_hcg"):
        return [None]

    hcg = fleet.get_hybrid_communicate_group()
    groups = [
        hcg.get_data_parallel_group(),
        hcg.get_sharding_parallel_group(),
    ]
    if use_mp:
        groups.append(hcg.get_model_parallel_group())
    return [g for g in groups if g is not None and g.nranks > 1]
