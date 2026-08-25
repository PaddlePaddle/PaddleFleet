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

"""Step-wide store of document-mask metadata, built before the forward starts.

Holds two kinds: ``CSADocMaskMetadata`` for the DSv4-hybrid (HCA/CSA) layers and
``MQADocMeta`` for the latent-MQA (``-2``) ones. Enabled by the single switch
``config.csa_share_docmask_meta`` -- the field name predates the MQA half and is
kept because it is already in the training configs.

What it replaces
----------------
Without it, every layer builds its own metadata on every forward pass. On the
layer43 config that is 37 HCA layers x (forward + recompute replay) = 74 builds
plus 6 MQA layers x 2 = 12, per micro-batch, all of them bit-identical, and each
one paying a ``.item()`` device sync inside ``build``. Those syncs sit on the
pipeline's critical path, so they show up as bubbles.

How it works
------------
The trainer already holds every micro-batch of a step before it calls
``forward_backward_pipeline`` (it buffers them in ``_pp_data_buffer``), so the
whole step's metadata is built there, once per ``(micro-batch, kind, mask
group)``, and the layers only look it up.

Identifying the micro-batch inside a layer
------------------------------------------
Layer identity cannot key the store (that would defeat sharing) and neither can
the mask object: ``TransformerLayer`` hands a fresh ``clone()`` to each
recomputed segment, the legacy dataflow re-slices the mask per layer, and each
pipeline stage receives its own tensor. So each consumer owns a forward counter,
the same scheme ``ernie-core``'s ``MagicInstance`` uses:

* one counter per consumer, keyed by ``(layer_number, is_mtp_layer)``;
* advanced exactly once per micro-batch, from ``TransformerLayer.forward`` --
  which always runs outside the recompute wrapper, unlike ``_forward_impl``;
* ``mb_idx = counter % accumulate_steps`` picks the slot;
* counters are reset and audited at the step boundary by the trainer.

Virtual-pipeline interleaving is fine: chunks are interleaved with each other,
but each chunk still sees micro-batches in increasing order.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from paddle.base import framework

if TYPE_CHECKING:
    import paddle

    from paddlefleet.transformer.csa_attention import CSADocMaskMetadata

logger = logging.getLogger(__name__)


class DocMaskMetaRegistry:
    """One document-mask metadata object per slot of a step.

    A slot is ``(micro-batch, kind, mask_group)``, where ``kind`` is the compress
    ratio for the DSv4-hybrid layers' ``CSADocMaskMetadata`` and the literal
    ``"mqa"`` for the ``-2`` layers' ``MQADocMeta``. The two kinds are separate
    objects on purpose -- see ``MQADocMeta``'s docstring for why the MQA layers
    cannot read a ``CSADocMaskMetadata``.

    ``mask_group`` says which logical document mask a consumer reads: main layers
    share ``("main",)``, while every MTP depth gets its own group because MTP
    layers are fed a slice of ``mtp_startend_row_indices_all`` -- same shape as
    the main mask, different contents. Grouping is deliberately conservative:
    over-separating only costs an extra build, under-separating would serve a
    metadata built from another mask.
    """

    def __init__(self) -> None:
        self._cnt: dict[Any, int] = {}
        self._store: dict[Any, CSADocMaskMetadata] = {}
        self._acc = 1
        self.n_prebuild = 0
        self.n_hit = 0

    # ------------------------------------------------------------------
    # consumers register themselves at build time
    # ------------------------------------------------------------------
    def register(self, key: Any) -> None:
        """Declare a consumer so the step-boundary audit knows about it."""
        self._cnt.setdefault(key, -1)

    # ------------------------------------------------------------------
    # producer side: the trainer, once per step, before the forward
    # ------------------------------------------------------------------
    def begin_step(self, accumulate_steps: int) -> dict[str, int]:
        """Drop the previous step's metadata and reset every counter.

        ``accumulate_steps`` is the only value that cannot come from the model
        config -- it lives on the trainer arguments, and
        ``ProgreesiveBatchingCallback`` may change it mid-training, so it is
        re-read every step. Returns the previous step's stats for logging.
        """
        stats = {"prebuild": self.n_prebuild, "hit": self.n_hit}
        self._acc = max(1, int(accumulate_steps))
        self._store.clear()
        for key in self._cnt:
            self._cnt[key] = -1
        self.n_prebuild = 0
        self.n_hit = 0
        return stats

    def preload(
        self,
        mb_idx: int,
        ratio: int,
        batch_size: int,
        seqlen: int,
        mask: paddle.Tensor,
        dense_mode: bool,
        mask_group: Any,
        window_size: int,
    ) -> None:
        """Build one slot ahead of the forward, index tables included.

        ``ratio`` is the raw ``csa_compress_ratios`` entry: the slot itself is
        keyed by ``max(1, ratio)`` because that is what ``build`` normalises to,
        but which index tables the layers will read depends on the raw value.
        """
        from paddlefleet.transformer.csa_attention import CSADocMaskMetadata

        key = (int(mb_idx), max(1, int(ratio)), mask_group)
        meta = self._store.get(key)
        if meta is None:
            meta = CSADocMaskMetadata.build(
                max(1, int(ratio)),
                int(batch_size),
                int(seqlen),
                mask,
                dense_mode=dense_mode,
            )
            self._store[key] = meta
            self.n_prebuild += 1
        if meta is not None:
            self._warm(meta, int(ratio), int(seqlen), int(window_size))

    @staticmethod
    def _warm(
        meta: CSADocMaskMetadata, ratio: int, seqlen: int, window_size: int
    ) -> None:
        """Force the lazy index tables so the layers only read them.

        ``build`` leaves the index tables lazy, so without this the first layer of
        the step would still compute them inside the forward -- on the pipeline's
        critical path, which is what this whole switch exists to avoid. The
        getters cache on their own argument, so warming is idempotent.

        Only the tables whose size is linear in ``seqlen`` are warmed. The
        ``O(seqlen^2)`` full-causal table the ``ratio == -1`` layers read is left
        lazy on purpose: at seqlen 65536 it is 16 GiB plus a 32 GiB int64
        intermediate, which measured as +54 GB peak memory for no throughput
        gain. Sharing still dedups it across layers, it just is not hoisted out
        of the forward.
        """
        from paddlefleet.transformer.csa_attention import CSA_MQA_RATIO

        if ratio == CSA_MQA_RATIO:
            # Full-causal MQA reads only that O(seqlen^2) table, so there is
            # nothing cheap left to warm here.
            return
        meta.get_window_topk_idxs(window_size)
        if meta.ratio > 1:
            # Compressed indices follow the vanilla KV inside ``kv_full``, so the
            # offset is sq (no CP) or sq_global (CP) -- both are the seqlen the
            # slot was built with.
            meta.get_compress_topk_idxs(seqlen)

    def preload_mqa(
        self,
        mb_idx: int,
        batch_size: int,
        seqlen: int,
        mask: paddle.Tensor,
        mask_group: Any,
        window_size: int,
    ) -> None:
        """Build the ``-2`` (absorbed-MQA) layers' slot ahead of the forward.

        A separate slot from the CSA ones, and not derivable from them: under
        ``csa_dense_mode`` a ``CSADocMaskMetadata`` carries no ``is_valid`` /
        ``doc_lens`` at all, and its position tables come from a different
        boundary rule (see ``MQADocMeta``'s docstring). The slot is tagged
        ``"mqa"`` where a CSA slot carries its ratio, so the two never collide.
        """
        from paddlefleet.transformer.mqa_latent_attention import MQADocMeta

        key = (int(mb_idx), "mqa", mask_group)
        meta = self._store.get(key)
        if meta is None:
            meta = MQADocMeta.build(mask, int(batch_size), int(seqlen))
            self._store[key] = meta
            self.n_prebuild += 1
        meta.warm(int(window_size))

    # ------------------------------------------------------------------
    # consumer side: the layers
    # ------------------------------------------------------------------
    def advance(self, key: Any, training: bool) -> int:
        """Advance ``key``'s forward counter and return the micro-batch slot.

        MUST be called from outside any recompute wrapper -- i.e. from
        ``TransformerLayer.forward``, never from ``_forward_impl``. Being nested
        inside one is rejected rather than worked around: paddle's reentrant
        recompute runs the *original* forward under ``no_grad``
        (``RecomputeFunction.forward``) and only restores ``_has_grad`` for the
        replay in backward. A counter that skipped the no-grad pass would hand
        that pass -- the forward whose output actually flows onward -- the
        previous micro-batch's slot, and only the replay would see the right
        one. That is silent corruption, so it fails loudly instead.

        ``training`` separates the two no-grad cases: evaluation has no backward
        at all and must advance normally.
        """
        if training and not framework._dygraph_tracer()._has_grad:
            raise RuntimeError(
                f"doc-mask metadata: advance({key!r}) ran in a no-grad "
                "training forward, i.e. nested inside a recompute segment. It "
                "must be called from TransformerLayer.forward, outside the "
                "recompute wrapper -- otherwise the forward reads the previous "
                "micro-batch's slot."
            )
        cnt = self._cnt.get(key, -1) + 1
        self._cnt[key] = cnt
        return cnt % self._acc

    def get(
        self,
        mb_idx: int,
        ratio: int,
        batch_size: int,
        seqlen: int,
        mask_group: Any,
    ) -> CSADocMaskMetadata | None:
        """Look up the slot the trainer prepared. Pure lookup, never builds.

        Idempotent, so it is safe to call from inside a recompute segment: the
        replay of a layer reads the same slot the forward pass did, because the
        slot index arrives as a plain int argument rather than being recomputed.

        Returns ``None`` when the slot is absent, for mask groups the trainer
        does not prebuild (the MTP depths' ``("mtp", ...)`` groups: their mask
        is a slice of ``mtp_startend_row_indices_all``, absent by design), so
        the caller falls back to building its own metadata -- the same contract
        ``get_mqa`` gives. The main group is always preloaded, so a miss there
        raises ``KeyError``: that means a prebuild or forward-counter bug.
        """
        ratio = max(1, int(ratio))
        key = (int(mb_idx), ratio, mask_group)
        meta = self._store.get(key)
        if meta is None:
            if mask_group == ("main",):
                raise KeyError(
                    f"csa_share_docmask_meta: no metadata preloaded for {key}. "
                    f"Known slots: {sorted(self._store.keys())}. The trainer "
                    "preloads one slot per (micro-batch, ratio, mask group) "
                    "before the forward; a missing slot means the consumer's "
                    "ratio or mask group was not preloaded, or its forward "
                    "counter drifted."
                )
            return None
        # Explicit exception, not an ``assert``: training may run ``python -O``
        # (paddle release images do), which strips asserts -- a layout mismatch
        # must fail loudly in every spell, not silently feed the wrong metadata.
        if (meta.ratio, meta.batch_size, meta.seqlen) != (
            ratio,
            int(batch_size),
            int(seqlen),
        ):
            raise ValueError(
                f"csa_share_docmask_meta: slot {key} holds "
                f"{(meta.ratio, meta.batch_size, meta.seqlen)} but the layer "
                f"asked for {(ratio, int(batch_size), int(seqlen))}; the "
                "preloaded mask does not have the layout the layer sees."
            )
        self.n_hit += 1
        return meta

    def get_mqa(
        self,
        mb_idx: int,
        batch_size: int,
        seqlen: int,
        mask_group: Any,
    ) -> Any:
        """Look up the ``-2`` layers' slot. Pure lookup, never builds.

        Returns ``None`` when the slot is absent, so the caller can fall back to
        building its own -- unlike the CSA lookup, which raises. The MQA layers
        reach this from mask groups the trainer does not prebuild (an MTP layer's
        own mask, for one), and falling back there is correct, just unshared.
        """
        meta = self._store.get((int(mb_idx), "mqa", mask_group))
        if meta is None:
            return None
        # Explicit exception, not an ``assert`` -- same ``python -O`` argument
        # as ``get`` above: a mis-preloaded MQA slot must fail loudly.
        if (meta.batch_size, meta.seqlen) != (int(batch_size), int(seqlen)):
            raise ValueError(
                f"csa_share_docmask_meta: MQA slot {(int(mb_idx), 'mqa', mask_group)} "
                f"holds {(meta.batch_size, meta.seqlen)} but the layer asked for "
                f"{(int(batch_size), int(seqlen))}; the preloaded mask does not have "
                "the layout the layer sees."
            )
        self.n_hit += 1
        return meta

    # ------------------------------------------------------------------
    # audit
    # ------------------------------------------------------------------
    def check(self) -> None:
        """Assert every consumer ran exactly ``accumulate_steps`` times.

        Called by the trainer right after ``forward_backward_pipeline``. A
        counter that is still ``-1`` means the consumer never ran this step
        (a layer skipped by an elastic or MTP-disabled path), which is allowed;
        any other value means it ran a wrong number of times, so its slot index
        was unreliable.
        """
        # Explicit exception, not an ``assert``: a wrong run count means the
        # step's slot indices were unreliable -- it must crash the step even
        # under ``python -O``.
        for key, cnt in self._cnt.items():
            if cnt not in (-1, self._acc - 1):
                raise RuntimeError(
                    f"csa_share_docmask_meta: consumer {key} advanced to {cnt}, "
                    f"expected {self._acc - 1} (or -1 if never executed); its "
                    "micro-batch slot index cannot be trusted."
                )


doc_mask_meta_registry = DocMaskMetaRegistry()
