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

"""Latent MQA core attention for hybrid MLA layers, with DSA.

``hybrid_mla_attention`` selects which core attention the
``csa_compress_ratios == -2`` (MLA) layers of a ``dsv4_hybrid`` model run, and
within the DSA mode ``dsa_indexer_use_sparse_loss`` selects the training phase.
``MQALatentAttention._phase()`` is the single place that is decided, and each
phase has its own ``_forward_*`` with no loss code shared between them:

* ``"mha"`` -- unchanged dense MLA (MHA); this module is not used.
* ``"mqa_full_causal"`` -> ``_forward_full_causal``. Latent MQA with the indexer
  dropped, attending to the full per-document causal set. Mathematically
  identical to the dense MHA phase, so it isolates the absorption from the
  sparsity for equivalence experiments.
* ``"mqa_dsa"`` + ``dsa_indexer_use_sparse_loss=False`` -> ``_forward_warmup``
  (phase 2, DSA warmup, paired with ``train_indexer_only``). The backbone is
  frozen and the indexer is random, so **neither side uses top-k**: attention
  runs the same full per-document causal set phase 1 did, and the indexer KL
  spans every causal column on both sides, via one ``csa_indexer_topk_fwd``
  call in its documented "full-candidate selection" mode.
* ``"mqa_dsa"`` + ``dsa_indexer_use_sparse_loss=True`` -> ``_forward_sparse``
  (phase 3). A forced local window plus Lightning-indexer top-k, i.e. DeepSeek
  Sparse Attention on the KV latent, with the KL restricted to that same
  selected set. This is the only phase that reads ``index_topk``.

The indexer reuses the model-wide ``index_n_heads`` / ``index_head_dim``.

The two full-causal phases attend over the whole per-document causal span, which
is exactly what the caller's ``attn_mask_startend_row_indices`` already says, so
they run as dense FA4 flashmask (``_dense_attn``), context parallel included.
FA4 is their only backend: ``_assert_dense_fa4`` refuses to start rather than
substitute one that would need an ``O(s^2)`` column table (see
``_dense_attn`` for what that costs). Only phase 3 genuinely selects columns,
and only it builds a table.

Note the ``mqa_*`` modes here are *latent* MQA (this module). A
``csa_compress_ratios`` entry of ``-1`` is *CSA full-causal MQA*, a different
layer kind handled by ``csa_attention.py``.

``MLASelfAttention`` performs the activation-level absorption (see its
``mqa_latent`` flag), so this module receives

    query [b, s, h, kv_lora_rank + qk_rope_head_dim]
    key   [b, s, 1, kv_lora_rank + qk_rope_head_dim]

and de-absorbs the value side with ``v_b_proj_weight``
(``[kv_lora_rank, h, v_head_dim]``). Every parameter stays byte-identical to
the MHA layout, so an MHA checkpoint loads into an MQA run unchanged.

``add_full_attention_sink_bias`` (or ``softmax_type``) adds one learnable
per-head sink logit as ``softmax_offset``, built by the same
``build_softmax_offset`` helper ``DotProductAttention`` uses, so the parameter
name matches the dense phase. It is fed to the block-sparse kernel as its
``attn_sink``, which then enables the finite-sink LSE correction and the
analytic sink gradient. The indexer KL target does not see it: that target
normalises over the indexer's own candidate set, and the sink -- like the forced
window -- is outside that set by construction, not by cancellation. See
``_attn_target`` for the measured size of the alternative definition.

Multi-document equivalence: RoPE/YaRN scores depend only on ``pos_q - pos_k``,
the YaRN ``mscale`` is a constant and the Hadamard ``rotate_activation`` is
orthogonal, so no per-document position reset is needed. Equivalence to running
every document on its own therefore reduces to index correctness, which is what
``_derive_csa_doc_boundaries`` plus the index builders below guarantee.

Context parallel (``contiguous_allgather`` only, as the HCA layers of the same
model require at ``dsv4_hybrid_attention.py:607-611``) follows the same
"Miles pattern" as ``CSASelfAttention._forward_cp``: the query slice stays
sharded, only the low-dimensional KV latent is all-gathered to the global
length, and the output comes back sharded with no reduce-scatter. The index
tables are built over the *global* sequence and then row-sliced to this rank,
which is exactly right because ``_derive_csa_doc_boundaries`` and the builders
are ``seqlen``-agnostic and their column values are global token ids -- the same
space the all-gathered KV is indexed in. ``attn_mask_startend_row_indices``
already arrives at the global length under CP
(``dsv4_hybrid_attention.py:634``), while ``query`` / ``key`` / ``x`` / ``qr``
arrive local but RoPE'd with global positions
(``multi_latent_attention.py:1485-1545``). The dense full-causal backend needs
one extra step for the same reason -- its mask is a *row* bound, not a column id
-- which ``_cp_row_bounds`` does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

from paddlefleet.context_parallel_utils import (
    ContextParallelGatherOp,
    preprocess_index,
)
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.transformer.cp_utils import (
    all_gather_cp,
    dualchunk_chunk_ids,
    dualchunk_swap,
)
from paddlefleet.transformer.csa_attention import (
    TileLangCSAIndexerLossAutoScaler,
    _build_window_topk_idxs_from_doc_bounds,
    _derive_csa_doc_boundaries,
    _validate_csa_docmask_shape,
)
from paddlefleet.transformer.dot_product_attention import build_softmax_offset
from paddlefleet.transformer.dsa_attention import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from paddlefleet.transformer.layer import FleetLayer

if TYPE_CHECKING:
    from paddlefleet.transformer.enums import AttnMaskType

# Working set of the phase-3 KL-target gather, as a ``rows x slots`` budget:
# 256 rows x 512 slots x 576 dims is ~150MB of gathered bf16 keys, transient and
# freed every iteration. Measured at s=8192/h=64/topk=512/dk=576 on one B30Z:
# 15.9ms at 128 rows, 13.3ms at 256, 12.4ms at 512, 12.4ms at 1024 -- past 256
# the curve is flat, so larger chunks only buy peak memory.
_TARGET_ROW_SLOTS = 256 * 512
# ``lse_indexer`` widths ``flash_mla_sparse_fwd`` implements; anything else is
# rejected outright (FlashMLA ``sparse_fwd.h:160``). Narrower budgets keep the
# Python target path.
_LSE_INDEXER_TOPKS = (512, 1024, 2048)
# Narrowest query-head count the score-recompute kernel's MMA ``M`` tile
# handles; see ``_attn_target_cudnn``.
_TARGET_QHEAD_MIN = 16
_NEG_INF = -1e30
_EPS = 1e-10
# Widest full-candidate KL the tilelang indexer can launch. Its two bitonic
# buffers are sized ``2 * topk``, so one block asks for ``16 * topk + 25344`` B
# against the SM100 opt-in limit of 232448 B: ``topk <= 12944``, i.e. 8192 as a
# power of two. Measured: width 16384 fails with ``Failed to set the allowed
# dynamic shared memory size to 287488``, 8192 launches. Not a config knob --
# it is a property of the device -- so the warmup KL rejects wider spans and
# points at ``csa_indexer_backend="cudnn"`` instead.
_TILELANG_KL_MAX_WIDTH = 8192
# Row-chunk budgets, in tensor elements, for the two loops of the dense
# (full-candidate) warmup KL. The LSE loop materialises ``[c, h, k_end]`` twice
# (bf16 matmul output plus its fp32 mask/where copy), so 16Mi elements is ~128MB
# transient; the KL reduction materialises ``[c, width]`` a few times over, and
# 64Mi elements keeps it in the same ballpark.
_LSE_CHUNK_ELEMS = 1 << 24
_KL_CHUNK_ELEMS = 1 << 26

logger = logging.getLogger(__name__)


def _dense_pylayer_inputs(
    query: Tensor, key: Tensor, value: Tensor, sink: Tensor | None
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """Make the flashmask ``PyLayer``'s differentiable inputs agree on freezing.

    ``flashmask_attention`` is a ``PyLayer``, and Paddle requires its backward to
    return ``None`` at exactly the positions whose forward input had
    ``stop_gradient=True``. The flash kernel returns a dense gradient at every
    position instead, so any *partially* frozen call aborts with
    ``FlashMaskFunc's backward function should return None at N position``
    (``py_layer_node.cc:213``). Frozen inputs are not exotic here: the warmup
    phase runs under ``train_indexer_only``, which freezes the backbone, and the
    attention sink is a parameter of its own, so which subset arrives frozen
    depends on the configuration rather than on this layer.

    Two regimes, and this returns the right thing for both:

    * nothing wants a gradient -- pass everything through, so no grad node is
      built at all and the contract never comes up;
    * something wants one -- hand the kernel a detached ``stop_gradient=False``
      view of each frozen input. The unwanted gradient lands on that throwaway
      view and is dropped, the real tensor stays frozen and receives nothing,
      and the forward is bit-identical (measured). Neither ``detach()`` alone
      nor ``* 1.0`` works: both inherit ``stop_gradient``.

    The sparse backend needs none of this -- it records
    ``attn_sink_needs_grad`` itself (``csa_sparse_attn.py:178-182``).
    """
    tensors = (query, key, value, sink)
    if all(t is None or t.stop_gradient for t in tensors):
        return tensors

    def proxy(t):
        if t is None or not t.stop_gradient:
            return t
        out = t.detach()
        out.stop_gradient = False
        return out

    return tuple(proxy(t) for t in tensors)


def _doc_segment_lens(doc_starts: Tensor, s_global: int) -> list[int]:
    """Segment lengths that tile ``[0, s_global)``, padding slots included.

    ``_derive_csa_doc_boundaries`` returns document *content* lengths, which stop
    short of the slot when a document is padded. Feeding those to
    ``dense_kl_cu_seqlens`` would leave a hole between segments and let the next
    document's tokens into a query's causal prefix, so the borders are taken from
    the starts instead: consecutive differences, with the tail reaching
    ``s_global``. Padding rows keep their preceding document's segment, which is
    what the row gating already assumes.
    """
    starts = [int(v) for v in doc_starts.tolist()]
    ends = [*starts[1:], int(s_global)]
    return [b - a for a, b in zip(starts, ends)]


class _HashableTensor(paddle.Tensor):
    """``paddle.Tensor`` with hashable ``shape`` / ``stride()``.

    The cuDNN score-recompute wrapper keys its kernel cache on
    ``(dtype, shape, stride(), ...)``, and Paddle returns those as lists, which
    are unhashable.
    """

    @property
    def shape(self):
        return tuple(super().shape)

    def stride(self, dim=None):
        if dim is None:
            return tuple(super().stride())
        return super().stride(dim)


@dataclass
class DenseWarmupKLPlan:
    """Everything the dense warmup KL backward needs that is not a grad input.

    Passed to :class:`DenseWarmupIndexerLossAutoScaler` as a single opaque
    object on purpose: Paddle's ``PyLayer`` counts *tensor* arguments to decide
    how many gradients ``backward`` must return, so bundling the THD descriptors
    in here keeps that count at the four differentiable tensors instead of
    making it depend on which of ``q_causal_offsets`` / ``row_active`` happens to
    be present.

    Attributes:
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        q_causal_offsets: THD descriptors from ``dense_kl_cu_seqlens``, shared
            by the indexer score, the attention score and the backward -- all
            three must see the same segmentation and the same ``max_seqlen_k``,
            since the backward's shape check ties the two score matrices
            together.
        loss_coeff: already carries the ``total_q / num_rows`` compensation for
            the backward kernel's built-in ``1 / total_q``; see
            :meth:`MQALatentAttention._warmup_kl_dense_cudnn`.
        softmax_scale: the attention scale, needed to recompute the target.
        block_I: backward tile width.
    """

    cu_seqlens_q: Tensor
    cu_seqlens_k: Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    q_causal_offsets: Tensor | None
    loss_coeff: float
    softmax_scale: float
    block_I: int = 128


class DenseWarmupIndexerLossAutoScaler(paddle.autograd.PyLayer):
    """Attach the dense full-candidate warmup KL gradient to the main output.

    The dense counterpart of ``TileLangCSAIndexerLossAutoScaler``, and separate
    from it on purpose: that class carries a ``[b, s, width]`` ``target`` and
    ``topk_probs`` pair from forward into backward, which is exactly the
    ``O(s_local x s_global)`` residency this path exists to remove. Here nothing
    width-proportional is saved -- the backward recomputes both score matrices
    from the same inputs the forward used, which is what the cuDNN dense triple
    is designed for.

    Saved tensors are the recompute inputs (``index_q`` / ``weights`` /
    ``index_k`` / ``query`` / ``kv``) plus the tiny ``[s_local, h]`` attention
    LSE. ``attn_lse`` is *not* recomputable cheaply, and it is also where row
    gating lives: an inactive row carries ``+inf``, which makes its attention
    score row zero and, together with the ``+inf`` this backward writes into
    ``index_lse``, drives the row's gradient to an exact zero.
    """

    @staticmethod
    def forward(
        ctx,
        output: Tensor,
        index_q: Tensor,
        weights: Tensor,
        index_k: Tensor,
        query: Tensor,
        kv: Tensor,
        attn_lse: Tensor,
        row_active: Tensor,
        plan: DenseWarmupKLPlan,
    ) -> Tensor:
        ctx.save_for_backward(
            index_q.detach(),
            weights.detach(),
            index_k.detach(),
            query.detach(),
            kv.detach(),
            attn_lse.detach(),
            row_active,
        )
        ctx.plan = plan
        # Same contract as ``TileLangCSAIndexerLossAutoScaler.forward``: with the
        # backbone frozen, ``output`` is a leaf with ``stop_gradient=True`` and
        # returning it unchanged would look like an inplace alias.
        ctx.output_needs_grad = not output.stop_gradient
        return output if ctx.output_needs_grad else output.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        from paddlefleet.cudnn_ops import (
            dense_attn_kl_scores,
            dense_indexer_kl_bwd,
            dense_indexer_kl_scores,
        )

        (
            index_q,
            weights,
            index_k,
            query,
            kv,
            attn_lse,
            row_active,
        ) = ctx.saved_tensor()
        plan = ctx.plan
        thd = {
            "cu_seqlens_q": plan.cu_seqlens_q,
            "cu_seqlens_k": plan.cu_seqlens_k,
            "max_seqlen_q": plan.max_seqlen_q,
            "max_seqlen_k": plan.max_seqlen_k,
            "q_causal_offsets": plan.q_causal_offsets,
        }

        index_score, index_lse = dense_indexer_kl_scores(
            index_q, index_k, weights, **thd
        )
        # Row gating, and the reason it sits on ``index_lse`` rather than on the
        # attention side: ``+inf`` here zeroes both factors of the kernel's score
        # gradient (``predict = exp(score - lse) -> 0`` and
        # ``log_clip_mask -> 0``), so the row contributes exactly nothing.
        # Zeroing ``attn_l1norm`` instead would leave the target clamp
        # ``max(target, exp(-100))`` behind as a residue.
        index_lse = paddle.where(
            row_active, index_lse, paddle.full_like(index_lse, float("inf"))
        )
        attn_score, attn_l1norm = dense_attn_kl_scores(
            query,
            kv,
            attn_lse,
            plan.softmax_scale,
            **thd,
        )
        # Masked columns come back as ``-inf``; the kernel would survive them
        # (``log_clip_mask`` is 0 there) but the clamp on a ``-inf`` target is
        # not worth relying on, and this costs one pass over a matrix that is
        # about to be consumed in place anyway.
        attn_score = attn_score.clip_(min=0.0)

        scale = DSAIndexerLossAutoScaler._main_loss_backward_scale
        grad_loss = 1.0 if scale is None else scale

        grad_q, grad_weights, grad_k = dense_indexer_kl_bwd(
            index_q,
            weights,
            index_k,
            attn_score,
            attn_l1norm,
            index_score,
            index_lse,
            loss_coeff=plan.loss_coeff,
            grad_loss=grad_loss,
            block_I=plan.block_I,
            **thd,
        )
        if grad_q.dtype != index_q.dtype:
            grad_q = grad_q.cast(index_q.dtype)
        if grad_weights.dtype != weights.dtype:
            grad_weights = grad_weights.cast(weights.dtype)
        if grad_k.dtype != index_k.dtype:
            grad_k = grad_k.cast(index_k.dtype)

        grad_main = grad_output if ctx.output_needs_grad else None
        # Tensor inputs in signature order: output, index_q, weights, index_k,
        # query, kv, attn_lse, row_active. The last four are recompute inputs
        # only -- they arrive detached, so ``None`` is the required answer.
        return (
            grad_main,
            grad_q,
            grad_weights,
            grad_k,
            None,
            None,
            None,
            None,
        )


@dataclass
class MQALatentAttentionSublayersSpec:
    """Sublayers spec for :class:`MQALatentAttention`.

    Args:
        indexer: ``DSAIndexer`` spec. ``hybrid_mla_attention="mqa_dsa"`` provides
            one; ``"mqa_full_causal"`` leaves it ``None`` and the layer attends
            to the full per-document causal set (dense MHA equivalent). Also
            ``None`` in the absorption equivalence unit tests.
    """

    indexer: LayerSpec | type = None


@dataclass
class MQADocMeta:
    """Document-mask derivations for the ``-2`` (absorbed-MQA) layers.

    A pure function of ``(attn_mask_startend_row_indices, seqlen)``: no
    randomness, no collectives, every output an index/bound table with
    ``stop_gradient``. ``seqlen`` is always the *global* sequence length
    (``s_local * cp_size``) and every table is built over it, so column values
    stay global token ids -- which is what the kernels expect, the KV being
    all-gathered. Rows are sliced to the local CP rank on the way out.

    ``CSADocMaskMetadata`` deliberately does **not** stand in for this. Two
    thirds of it is compression semantics, its ``build`` pays a ``.item()`` for a
    compression bound, and under ``csa_dense_mode`` -- which the layer43 configs
    set -- its ``is_valid`` / ``doc_lens`` / ``doc_starts`` are all ``None``
    (``csa_attention.py:512-521``), which are exactly the fields needed here.
    Its ``doc_start_per_pos`` also comes from a different boundary rule in that
    mode (``document_mask_triton`` rather than ``_derive_csa_doc_boundaries``).

    The derived tables are cached per distinct argument. That only pays off when
    one instance is shared by the ``-2`` layers of a micro-batch, which is what
    ``csa_share_docmask_meta`` arranges (see ``doc_mask_meta_registry``); with the
    switch off every forward builds its own instance, no cache ever hits and the
    work per layer is what it was before this class existed.
    """

    batch_size: int
    seqlen: int
    doc_start_per_pos: Tensor
    doc_len_per_pos: Tensor
    is_valid: Tensor
    doc_lens: Tensor
    doc_starts: Tensor
    _window_topk_idxs: Tensor | None = None
    _window_size: int | None = None
    _valid_range: dict[int, tuple[Tensor, Tensor]] | None = None
    _cu_seqlens_arg: tuple[list[int] | None] | None = None
    _segment_lens: list[int] | None = None

    @classmethod
    def build(
        cls, row_end: Tensor | None, batch_size: int, seqlen: int
    ) -> MQADocMeta:
        """Derive the boundary tables from the document mask.

        ``row_end is None`` means "one document covering everything", the same
        fallback the layer used inline.
        """
        with paddle.no_grad():
            if row_end is None:
                row_end = paddle.full(
                    [batch_size, 1, seqlen, 1], seqlen, dtype="int32"
                )
            _validate_csa_docmask_shape(row_end, batch_size, seqlen)
            doc_start, doc_len, is_valid, doc_lens, doc_starts = (
                _derive_csa_doc_boundaries(row_end, seqlen)
            )
        return cls(
            batch_size=int(batch_size),
            seqlen=int(seqlen),
            doc_start_per_pos=doc_start,
            doc_len_per_pos=doc_len,
            is_valid=is_valid,
            doc_lens=doc_lens,
            doc_starts=doc_starts,
        )

    # ------------------------------------------------------------------
    # Derived tables. Each comes in two halves on purpose: a ``_global_*``
    # builder that is a pure function of the document mask (no CP rank enters
    # it, every rank derives the same table, and that is what gets cached), and
    # a public accessor that takes this rank's row range out of it. The prebuild
    # only ever calls the ``_global_*`` half, so it needs no rank information;
    # the layers call the public half with their real ``(position_offset,
    # s_local)``.
    # ------------------------------------------------------------------
    def _global_window_topk_idxs(self, window_size: int) -> Tensor:
        """Cached ``[b, seqlen, window_size]`` int32 sliding-window column ids."""
        window_size = int(window_size)
        if self._window_topk_idxs is None or self._window_size != window_size:
            with paddle.no_grad():
                self._window_topk_idxs = (
                    _build_window_topk_idxs_from_doc_bounds(
                        self.batch_size,
                        self.seqlen,
                        window_size,
                        self.doc_start_per_pos,
                        self.is_valid,
                    ).cast("int32")
                )
            self._window_size = window_size
        return self._window_topk_idxs

    def window_topk_idxs(
        self, window_size: int, position_offset: int, s_local: int
    ) -> Tensor:
        """``[b, s_local, window_size]`` int32, this rank's rows."""
        idxs = self._global_window_topk_idxs(window_size)
        if int(s_local) != self.seqlen:
            idxs = idxs[:, position_offset : position_offset + int(s_local)]
        return idxs

    def _global_valid_range(self, window: int) -> tuple[Tensor, Tensor]:
        """Cached ``([seqlen, 2] int32 range, [seqlen] available count)``.

        ``window`` is how many trailing causal tokens to exclude: the sparse
        phase passes the forced local window it adds separately, the warmup phase
        passes ``0`` because its candidate set is the whole causal span. Cached
        per ``window`` since a layer only ever asks for one of the two.
        """
        window = int(window)
        if self._valid_range is None:
            self._valid_range = {}
        cached = self._valid_range.get(window)
        if cached is None:
            with paddle.no_grad():
                positions = paddle.arange(self.seqlen, dtype="int64")
                causal_avail = paddle.minimum(
                    positions - self.doc_start_per_pos + 1, self.doc_len_per_pos
                )
                n_avail = paddle.clip(causal_avail - window, min=0)
                n_avail = paddle.where(
                    self.is_valid, n_avail, paddle.zeros_like(n_avail)
                )
                valid_range = paddle.stack(
                    [self.doc_start_per_pos, self.doc_start_per_pos + n_avail],
                    axis=-1,
                ).cast("int32")
            cached = (valid_range, n_avail)
            self._valid_range[window] = cached
        return cached

    def indexer_valid_range(
        self, window: int, position_offset: int, s_local: int
    ) -> tuple[Tensor, Tensor]:
        """``(valid_range [1, s_local, 2] int32, row_empty [1, s_local, 1])``."""
        valid_range, n_avail = self._global_valid_range(window)
        if int(s_local) != self.seqlen:
            valid_range = valid_range[
                position_offset : position_offset + int(s_local)
            ]
            n_avail = n_avail[position_offset : position_offset + int(s_local)]
        rows = int(valid_range.shape[0])
        return valid_range.unsqueeze(0), (n_avail == 0).reshape([1, rows, 1])

    def cu_seqlens_arg(self) -> list[int] | None:
        """Host-side ``cu_seqlens`` for the cuDNN indexer's THD fast path.

        ``None`` when the documents do not exactly tile the sequence, in which
        case the caller falls back to the dense path. Costs two D2H syncs, which
        is why the result is cached -- wrapped in a 1-tuple because ``None`` is
        itself a valid answer and cannot double as "not computed yet".
        """
        if self._cu_seqlens_arg is None:
            self._cu_seqlens_arg = (
                self.doc_lens.tolist()
                if int(self.doc_lens.sum()) == self.seqlen
                else None,
            )
        return self._cu_seqlens_arg[0]

    def doc_segment_lens(self) -> list[int]:
        """Host-side segment lengths that tile ``[0, seqlen)``.

        What the dense warmup KL needs instead of ``cu_seqlens_arg``: see
        :func:`_doc_segment_lens` for why the borders come from the starts and
        not from ``doc_lens``. Cached for the same reason -- it is a D2H sync.
        """
        if self._segment_lens is None:
            self._segment_lens = _doc_segment_lens(self.doc_starts, self.seqlen)
        return self._segment_lens

    def warm(self, window_size: int) -> None:
        """Build the cheap tables now so no layer builds them inside the forward.

        Called by the prebuild, i.e. before ``forward_backward_pipeline``, so the
        builds and the ``cu_seqlens`` D2H syncs stay off the steady-state pipeline
        schedule where they cost bubble time.

        Only the ``_global_*`` halves are called, so no CP rank information is
        needed here: what gets cached is the rank-independent table and each layer
        slices its own rows out of it later. Both window widths are warmed (the
        sparse phase asks for ``csa_window_size``, the warmup phase for ``0``)
        because the phase is not known at this level; together they cost
        ``seqlen * (window_size + 4)`` bytes.

        Everything this class holds is ``O(seqlen)`` or ``O(seqlen *
        window_size)``, so warming all of it is cheap. The one ``O(seqlen^2)``
        table latent MQA used to need -- the per-document causal column ids -- is
        gone: both non-sparse phases now run as dense FA4 flashmask off
        ``attn_mask_startend_row_indices`` (``_dense_attn``) and build no table at
        all.
        """
        self._global_window_topk_idxs(window_size)
        self._global_valid_range(window_size)
        self._global_valid_range(0)
        self.cu_seqlens_arg()
        self.doc_segment_lens()


class MQALatentAttention(FleetLayer):
    """Sparse attention on the absorbed MLA KV latent (``core_attention``).

    Consumes the pre-absorbed ``query`` / ``key`` produced by
    ``MLASelfAttention`` and returns ``[b, s, h * v_head_dim]``, so the MLA
    output tail (gate, ``o_proj``) is unchanged.
    """

    def __init__(
        self,
        config,
        sublayers_spec: MQALatentAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        softmax_scale: float | None = None,
        k_channels: int | None = None,
        v_channels: int | None = None,
        is_mtp_layer: bool = False,
        is_swa: bool = False,
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        super().__init__(config=config)

        DSAIndexerLossLoggingHelper.register_total_num_layers(config)
        self.layer_number = layer_number
        # Which logical document mask this layer reads, i.e. the mask group half
        # of the shared-metadata slot key. Derived here rather than assumed at
        # the lookup: the trainer only prebuilds ``("main",)``, so an MTP layer
        # asking for its own group misses and falls back to building privately --
        # which is correct. Hardcoding ``("main",)`` at the lookup would instead
        # hand an MTP layer metadata built from the *decoder* mask, silently.
        self.docmask_mask_group = (
            ("mtp", int(layer_number)) if is_mtp_layer else ("main",)
        )
        self.attn_mask_type = attn_mask_type
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        # CP state, same derivation as csa_attention.py:2079-2090.
        cp_pg = pg_collection.cp if pg_collection is not None else None
        if cp_pg is not None and getattr(cp_pg, "nranks", 1) > 1:
            self.cp_group = cp_pg
            self.cp_size = cp_pg.nranks
            self.cp_rank = cp_pg.rank
            self.cp_enabled = True
            # Deliberately the *same* constraint the HCA layers of this model
            # assert (dsv4_hybrid_attention.py:607-611), not a weaker one: the
            # contiguous layout is what makes "build the index table over the
            # global sequence, then row-slice this rank's queries" correct, and
            # it is what makes the all-gathered KV land in natural global order.
            if (
                getattr(config, "cp_balance_mode", None)
                != "contiguous_allgather"
            ):
                raise NotImplementedError(
                    "latent MQA under context parallel requires "
                    "cp_balance_mode='contiguous_allgather' (the same mode the "
                    "hybrid model's HCA layers require), got "
                    f"{getattr(config, 'cp_balance_mode', None)!r}."
                )
        else:
            self.cp_group = None
            self.cp_size = 1
            self.cp_rank = 0
            self.cp_enabled = False

        # ``k_channels`` is the MHA q_head_dim (qk_nope + qk_rope), NOT the 576
        # latent width: absorption is exactly score-preserving, so the MHA
        # softmax scale must be kept.
        if softmax_scale is None:
            k_ch = k_channels if k_channels is not None else config.head_dim
            self.softmax_scale = float(k_ch**-0.5)
        else:
            self.softmax_scale = float(softmax_scale)

        self.window_size = int(config.csa_window_size)
        self.indexer = (
            build_spec_layer(
                sublayers_spec.indexer,
                config=config,
                layer_number=layer_number,
                pg_collection=pg_collection,
            )
            if sublayers_spec.indexer is not None
            else None
        )
        self.indexer_loss_coeff = float(
            getattr(config, "dsa_indexer_loss_coeff", 0.0) or 0.0
        )
        # Phase selector, read by ``_phase()`` -- see its docstring. Stored raw
        # rather than as a derived phase so that flipping it on a live module
        # (tests do) cannot leave a stale phase behind.
        # ``transformer_config.__post_init__`` pins it against
        # ``train_indexer_only`` so the two cannot disagree.
        self.indexer_use_sparse_loss = bool(
            getattr(config, "dsa_indexer_use_sparse_loss", False)
        )
        # ``dsa_indexer_use_sparse_loss=False`` is the phase-2 (warmup) mode:
        # the indexer is still being learned, so attention must not consume its
        # ranking yet -- it attends to the full per-document causal set, exactly
        # like ``hybrid_mla_attention="mqa_full_causal"``, while the indexer is
        # trained on the widest candidate table the kernel allows. Consuming a
        # freshly initialised indexer's top-k here would feed the (frozen)
        # backbone an essentially random sparse pattern for the whole phase.
        # ``True`` is the phase-3/4 mode: attention consumes window + top-k and
        # the KL is restricted to that same set.
        # ``transformer_config.__post_init__`` pins this to ``train_indexer_only``
        # so the two cannot disagree. Deliberately *not* cached as a second
        # attribute: ``_forward_dsa`` reads ``self.indexer_use_sparse_loss``
        # directly at both use sites, so a test (or anything else) flipping it on
        # a live module cannot desynchronise the loss width from the attention
        # set.
        # Backward kernel for the indexer loss scaler, same switch the CSA layers
        # read (csa_attention.py:2054). cuDNN's ``d_index_k`` scatters each query
        # row's gradient to arbitrary key positions with atomics and is not
        # run-to-run reproducible; "tilelang" has an atomic-free path gated on
        # ``FLAGS_cudnn_deterministic``. Both apply the same
        # ``sm_scale = dim**-0.5``, so switching does not change the forward or
        # the scale. Default matches csa_attention.py's default ("tilelang").
        #
        # ``"unfused"`` -- the third value the config accepts -- is served by
        # tilelang here, and must be: the scaler
        # (``TileLangCSAIndexerLossAutoScaler.backward``) implements only
        # "cudnn" and "tilelang" and raises ``NotImplementedError`` otherwise,
        # so forwarding the name verbatim would turn a configuration the
        # validation accepts into a crash at the *first sparse backward*, long
        # after construction. This is the same substitution the warmup KL
        # already makes for the same reason (see ``_forward_warmup``:
        # ``unfused`` has no full-candidate implementation to offer either), so
        # one config value now means one kernel across both phases of the layer.
        backend = str(getattr(config, "csa_indexer_backend", "tilelang"))
        self.indexer_backend = "tilelang" if backend == "unfused" else backend
        # Backward kernel for the sparse MQA attention (dkv). cuDNN accumulates
        # dkv with atomics and is not run-to-run reproducible (bounded by
        # ``test_block_sparse_dsa_gradcheck.py::TestDeterminism``); "tilelang"
        # (mqa_latent_sparse_bwd) is bitwise stable for identical inputs -- by
        # construction, not via ``FLAGS_cudnn_deterministic`` -- but ~14x slower
        # on SM100. The forward is always FlashMLA regardless of this switch
        # (the tilelang forward cannot accept d_qk=576, which is not a power of
        # two). Default "cudnn" preserves the previous behaviour.
        self.sparse_attn_backward_backend = str(
            getattr(config, "mqa_sparse_attn_backward_backend", "cudnn")
        )
        # Learnable per-head attention-sink logit, from the model-wide
        # ``add_full_attention_sink_bias`` / ``softmax_type``. Built by the same
        # helper ``DotProductAttention`` uses, so the state_dict name
        # (``core_attention.softmax_offset``) and the switch are shared with the
        # dense MHA phase. ``None`` keeps the kernel on its sinkless ``-1e30``
        # path, bit-for-bit unchanged.
        self.softmax_offset = build_softmax_offset(
            self,
            config,
            num_attention_heads
            if num_attention_heads is not None
            else config.num_attention_heads,
            is_swa,
        )
        # Shares ``mqa_split_kv_b_proj`` with the query-side
        # absorption: when set, ``MLASelfAttention`` passes its standalone
        # ``v_b_proj`` parameter instead of a view of ``kv_b_proj.weight``, laid
        # out as ``[h, v_head_dim, kv_lora_rank]`` for ``fused_grouped_matmul``
        # rather than the ``[kv_lora_rank, h, v_head_dim]`` the einsum wants.
        self.split_kv_b = bool(getattr(config, "mqa_split_kv_b_proj", False))
        # Latent width, i.e. the value width the sparse kernel sees and the
        # contraction dim of the de-absorption weight. This layer only exists on
        # the ``dsv4_hybrid`` path, so the hybrid field is the authoritative one.
        kv_lora_rank = getattr(config, "hybrid_mla_kv_lora_rank", None)
        if kv_lora_rank is None:
            kv_lora_rank = config.kv_lora_rank
        self.kv_lora_rank = int(kv_lora_rank)
        # Fused Triton epilogue for the analytic sink gradient. Only reachable
        # when ``softmax_offset`` exists; a sinkless layer has no sink gradient.
        self.sink_grad_fusion = getattr(config, "dsa_sink_grad_fusion", False)
        # Row layout the indexer forward runs on; see ``mqa_indexer_cp_mode`` in
        # ``transformer_config`` and ``_indexer_topk_dualchunk``. Gated on a real
        # CP group: with ``cp_size == 1`` there is nothing to rebalance.
        self.indexer_dualchunk = (
            getattr(config, "mqa_indexer_cp_mode", None) == "dualchunk_p2p"
            and self.cp_enabled
            and self.cp_size > 1
        )

    def _chunk_valid_range(
        self, meta, s_global, doc_start, doc_len, is_valid, offset, length
    ):
        """``valid_range [1, length, 2]`` for the rows at ``[offset, offset+length)``.

        Both the cached ``MQADocMeta`` and the eager fallback expose the same
        ``(offset, length)`` slice of one global table, which is what lets the
        dual-chunk layout ask for its two segments by chunk offset instead of
        needing a second table.
        """
        if meta is not None:
            return meta.indexer_valid_range(self.window_size, offset, length)[0]
        return self._indexer_valid_range(
            s_global, doc_start, doc_len, is_valid, offset, length
        )[0]

    def _dualchunk_valid_range(
        self, meta, s_global, doc_start, doc_len, is_valid, s
    ):
        """``valid_range [1, s, 2]`` for this rank's rows in dual-chunk order.

        The same global table the contiguous path slices once, read instead as
        the two chunks ``dualchunk_chunk_ids`` assigns, so it lines up
        row-for-row with what ``dualchunk_swap`` produces and each kernel call
        can be given its own ``seq_offset``.

        ``row_empty`` deliberately has no counterpart here: it is applied to the
        results after they come back, i.e. in contiguous order.
        """
        lo, hi = dualchunk_chunk_ids(self.cp_rank, self.cp_size)
        m = s // 2
        return paddle.concat(
            [
                self._chunk_valid_range(
                    meta, s_global, doc_start, doc_len, is_valid, c * m, m
                )
                for c in (lo, hi)
            ],
            axis=1,
        )

    def _indexer_topk_dualchunk(
        self, q_idx, w_idx, k_idx, topk, doc_lens_arg, vr_zz, need_loss
    ):
        """Indexer top-k on the balanced dual-chunk row layout.

        Rank ``r`` scores global chunks ``(2r, 2*cp_size-1-2r)`` out of
        ``2*cp_size`` instead of its contiguous ``(2r, 2r+1)``. The ids sum to
        ``2*cp_size-1`` on every rank, and a causal row's candidate count grows
        linearly with its global position, so the two halves' work sums to a
        constant: measured 31x between cp0 and cp15 at 256k/cp16 collapses to 1x.

        **Two calls, not one.** The kernel learns where its rows sit globally
        from ``q_causal_offsets``, one scalar per batch
        (``csa_indexer_fwd_cudnn.py``), so a single affine map cannot describe two
        disjoint segments. Each chunk is internally contiguous, so each call
        carries its own ``seq_offset``. This is the same shape as the query tiling
        the dense path already does (it passes ``seq_offset + start`` per tile);
        only the second segment's start jumps.

        ``vr_zz`` must be the ``valid_range`` of exactly these rows, in the same
        order. A ``seq_offset`` that disagrees with it is silently wrong in one
        direction: too large only wastes work (the extra columns are masked out of
        the top-k anyway), too small writes ``-inf`` over legal candidates so they
        can never be selected.

        The two results are concatenated and swapped back to the contiguous
        layout before returning, so ``row_empty``, ``window_idxs``, attention, the
        KL and ``TileLangCSAIndexerLossAutoScaler`` all keep seeing one unpermuted
        ``[b, s_local, topk]`` tensor. Applying the loss PyLayer per chunk instead
        would halve ``target.shape[1]``, which is where its backward reads the
        row-count denominator, and double the gradient.

        The swaps are not differentiable and do not need to be: this whole path
        runs under ``paddle.no_grad()`` on detached inputs, and the indexer
        gradient reaches the weights through the loss scaler applied to the
        *unpermuted* tensors.
        """
        from paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk_fwd,
        )

        m = int(q_idx.shape[1]) // 2
        lo, hi = dualchunk_chunk_ids(self.cp_rank, self.cp_size)

        q_zz = dualchunk_swap(q_idx.detach(), self.cp_group, axis=1)
        w_zz = dualchunk_swap(w_idx.detach(), self.cp_group, axis=1)

        def _chunk(sl, chunk_id):
            return cudnn_indexer_topk_fwd(
                q_zz[:, sl].contiguous(),
                k_idx.detach(),
                w_zz[:, sl].contiguous(),
                ratio=1,
                topk_effective=topk,
                valid_range=vr_zz[:, sl],
                doc_lens=doc_lens_arg,
                seq_offset=chunk_id * m,
                return_topk_scores=need_loss,
            )

        r_lo = _chunk(slice(0, m), lo)
        r_hi = _chunk(slice(m, 2 * m), hi)

        selected = dualchunk_swap(
            paddle.concat([r_lo[0], r_hi[0]], axis=1), self.cp_group, axis=1
        )
        scores_out = []
        if need_loss:
            scores_out = [
                dualchunk_swap(
                    paddle.concat([r_lo[2], r_hi[2]], axis=1),
                    self.cp_group,
                    axis=1,
                )
            ]
        return selected, scores_out

    def _needs_indexer_loss(self) -> bool:
        """Whether this forward should build and attach the indexer loss.

        Both DSA phases share this predicate. ``paddle.is_grad_enabled()`` is
        what makes the loss count exactly once under recompute: the first
        (no-grad) forward only materialises the attention columns, the second one
        attaches the loss.
        """
        return (
            self.training
            and paddle.is_grad_enabled()
            and self.indexer_loss_coeff > 0
        )

    def _phase(self) -> str:
        """Which of the three training phases this layer runs.

        The single place the phase is decided, so the attention candidate set
        and the indexer-loss shape cannot disagree:

        * ``"full_causal"`` -- no indexer at all
          (``hybrid_mla_attention="mqa_full_causal"``, and the
          absorption-equivalence unit tests). Per-document full causal
          attention, no indexer loss.
        * ``"warmup"`` -- phase 2 (DSA warmup). The indexer exists but is still
          being learned, so attention must not consume its ranking: full causal
          attention, and the KL runs over the *full* causal set on both sides.
          No top-k anywhere.
        * ``"sparse"`` -- phase 3. Attention consumes window + top-k and the
          KL is restricted to that same selected set.

        Read live rather than cached in ``__init__``: a test flipping
        ``indexer_use_sparse_loss`` on a live module must not be able to
        desynchronise the two.
        """
        if self.indexer is None:
            return "full_causal"
        return "sparse" if self.indexer_use_sparse_loss else "warmup"

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        attn_mask_startend_row_indices: Tensor | None = None,
        attn_mask_type: AttnMaskType | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params=None,
        use_rr_flash_attention: bool = False,
        past_key_values=None,
        layer_idx=None,
        use_cache: bool = False,
        x: Tensor | None = None,
        qr: Tensor | None = None,
        kv_compressed: Tensor | None = None,
        k_pos_emb: Tensor | None = None,
        q_absorbed: Tensor | None = None,
        v_b_proj_weight: Tensor | None = None,
        input_ids: Tensor | None = None,
        docmask_mb_idx: int = -1,
    ) -> Tensor:
        """Absorbed-MQA forward.

        Args:
            query: ``[b, s, h, kv_lora_rank + qk_rope_head_dim]`` absorbed query.
            key:   ``[b, s, 1, kv_lora_rank + qk_rope_head_dim]`` shared latent.
            value: unused (``None``); the V side lives inside ``key``.
            attn_mask_startend_row_indices: ``[b, 1, s, 1]`` exclusive per-token
                document end rows, at the **global** sequence length under CP.
                ``None`` means a single document.
            x / qr: hidden states / q latent, inputs of the DSA indexer.
            v_b_proj_weight: de-absorption weight. ``[kv_lora_rank, h,
                v_head_dim]`` (the V slice of ``kv_b_proj``) by default, or the
                standalone ``v_b_proj`` parameter as ``[h, v_head_dim,
                kv_lora_rank]`` under
                ``mqa_split_kv_b_proj``.
            input_ids: ``[b, s]`` token ids, only used to build the indexer-loss
                row mask (``!= pad_token_id``). ``None`` falls back to the plain
                row mean, as CSA does at ``csa_attention.py:1306``.

        Returns:
            ``[b, s, h * v_head_dim]`` (this rank's query slice under CP)
        """
        if packed_seq_params is not None:
            raise NotImplementedError(
                "latent MQA does not support packed_seq_params; "
                "document masking is driven by "
                "attn_mask_startend_row_indices."
            )
        if v_b_proj_weight is None:
            raise ValueError(
                "MQALatentAttention requires v_b_proj_weight; it is only valid "
                "as the core_attention of an absorbed MLA layer."
            )

        b, s = int(query.shape[0]), int(query.shape[1])
        if b != 1:
            raise NotImplementedError(
                "latent MQA requires micro batch size 1 (documents "
                f"are packed along the sequence), got b={b}."
            )
        # Under CP this rank owns global query positions
        # [position_offset, position_offset + s). Everything index-related is
        # derived at s_global and row-sliced; the KV is all-gathered so that the
        # global column ids in those tables address it directly.
        s_global = s * self.cp_size
        position_offset = self.cp_rank * s
        kv = key.squeeze(2).contiguous()  # [b, s, kv_lora + qk_rope]
        kv = all_gather_cp(
            kv, dim=1, group=self.cp_group
        )  # -> [b, s_global, .]
        # Not derivable from ``v_b_proj_weight.shape[0]``: that only holds for the
        # einsum layout ``[l, h, v]``. The grouped-matmul layout is
        # ``[h, v, l]``, so the rank has to come from the config.
        kv_lora_rank = self.kv_lora_rank
        # The de-absorption weight's contraction dim sits last in the
        # grouped-matmul layout and first in the einsum one. Checking it here
        # turns a silently-wrong ``[G, R, D]`` / ``[l, h, v]`` mix-up (both
        # reshape fine) into an error at the first forward.
        # The rank is checked before the contraction dim: a folded 2-D
        # parameter that was never reshaped back would otherwise pass the
        # contraction test and fail deeper down, on an unpacking or reshape
        # whose message says nothing about the layout.
        if len(v_b_proj_weight.shape) != 3:
            expected = (
                "[h, v_head_dim, kv_lora_rank]"
                if self.split_kv_b
                else "[kv_lora_rank, h, v_head_dim]"
            )
            raise ValueError(
                f"v_b_proj_weight must be 3-D {expected}, got shape "
                f"{v_b_proj_weight.shape} with "
                f"mqa_split_kv_b_proj={self.split_kv_b}."
            )
        contraction = v_b_proj_weight.shape[-1 if self.split_kv_b else 0]
        if int(contraction) != kv_lora_rank:
            raise ValueError(
                "v_b_proj_weight layout mismatch: expected contraction dim "
                f"kv_lora_rank={kv_lora_rank}, got shape "
                f"{v_b_proj_weight.shape} with "
                f"mqa_split_kv_b_proj={self.split_kv_b}."
            )

        with paddle.no_grad():
            row_end = attn_mask_startend_row_indices
            if row_end is None:
                row_end = paddle.full(
                    [b, 1, s_global, 1], s_global, dtype="int32"
                )
            _validate_csa_docmask_shape(row_end, b, s_global)

        # Document-mask derivations. ``docmask_mb_idx >= 0`` plus the switch means
        # the trainer prebuilt this micro-batch's slot before the forward and the
        # ``-2`` layers of this mask group share it. A miss returns ``None`` -- an
        # MTP layer's group is never prebuilt -- and so does the switch being off
        # or inference; in every one of those cases everything below runs the
        # pre-existing code untouched.
        #
        # ``row_end`` itself stays outside: the dense-FA4 phases pass it to the
        # kernel as the mask, so it is needed either way, and normalising it is
        # ``O(1)`` -- the part worth sharing is ``_derive_csa_doc_boundaries``.
        meta = None
        doc_start = doc_len = is_valid = doc_lens = doc_starts = None
        if docmask_mb_idx >= 0 and getattr(
            self.config, "mqa_share_docmask_meta", False
        ):
            from paddlefleet.transformer.doc_mask_meta_registry import (
                doc_mask_meta_registry,
            )

            meta = doc_mask_meta_registry.get_mqa(
                docmask_mb_idx, b, s_global, self.docmask_mask_group
            )
        if meta is None:
            with paddle.no_grad():
                doc_start, doc_len, is_valid, doc_lens, doc_starts = (
                    _derive_csa_doc_boundaries(row_end, s_global)
                )

        phase = self._phase()
        if phase == "full_causal":
            return self._forward_full_causal(
                query,
                kv,
                v_b_proj_weight,
                kv_lora_rank,
                row_end,
            )

        if phase == "warmup":
            return self._forward_warmup(
                query,
                kv,
                x,
                qr,
                v_b_proj_weight,
                doc_start,
                doc_len,
                is_valid,
                doc_starts,
                kv_lora_rank,
                input_ids,
                position_offset,
                s,
                s_global,
                row_end,
                meta=meta,
            )

        return self._forward_sparse(
            query,
            kv,
            x,
            qr,
            v_b_proj_weight,
            doc_start,
            doc_len,
            is_valid,
            doc_lens,
            kv_lora_rank,
            input_ids,
            position_offset,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # full_causal
    # ------------------------------------------------------------------
    def _forward_full_causal(
        self,
        query: Tensor,
        kv: Tensor,
        v_b_proj_weight: Tensor,
        kv_lora_rank: int,
        row_end: Tensor,
    ) -> Tensor:
        """Per-document full-causal attention on the absorbed latent.

        No indexer is involved: the column set is decided by the document
        boundaries alone, so this output is mathematically identical to the
        dense MHA phase and is bit-identical across repeated calls (nothing
        here depends on a top-k tie-break).

        Used by two phases -- ``hybrid_mla_attention="mqa_full_causal"``, and
        the attention half of the phase-2 warmup, which must not consume the
        indexer's ranking while the indexer is still being learned.

        One backend: dense FA4 flashmask (``_dense_attn``). "The whole causal
        span" is already what ``attn_mask_startend_row_indices`` says, so the
        mask stays ``O(s)``; ``_assert_dense_fa4`` rejects an environment where
        FA4 would not serve it.
        """
        q_dim = int(query.shape[-1])
        self._assert_dense_fa4(q_dim, kv_lora_rank, row_end)
        core_out = self._dense_attn(query, kv, row_end, kv_lora_rank)
        return self._deabsorb(core_out, v_b_proj_weight, self.split_kv_b)

    # ------------------------------------------------------------------
    # warmup (phase 2)
    # ------------------------------------------------------------------
    def _forward_warmup(
        self,
        query: Tensor,
        kv: Tensor,
        x: Tensor,
        qr: Tensor,
        v_b_proj_weight: Tensor,
        doc_start: Tensor,
        doc_len: Tensor,
        is_valid: Tensor,
        doc_starts: Tensor,
        kv_lora_rank: int,
        input_ids: Tensor | None,
        position_offset: int,
        s_local: int,
        s_global: int,
        row_end: Tensor,
        meta: MQADocMeta | None = None,
    ) -> Tensor:
        """Phase 2: frozen backbone, full-causal attention, full-causal KL.

        No top-k on either side. The two halves:

        * **attention** is the same deterministic full-causal set phase 1 uses,
          so the frozen backbone sees exactly the activations it was pretrained
          with while the indexer is still random -- literally the same
          ``_forward_full_causal`` call.
        * **the indexer** is supervised over the *whole* per-document causal
          span, so it cannot reinforce its own initial ranking.

        The indexer half is one tilelang call with ``topk_effective = s_global``,
        which is the "full-candidate selection" mode ``csa_indexer_topk_fwd``
        documents for exactly this phase -- the CSA layers use it the same way
        with ``topk_effective = n_compressed``. The kernel returns the softmax
        probabilities over every candidate column plus the column ids, and the
        head dimension never leaves the kernel; the backward is upstream's
        ``csa_indexer_bwd`` via ``TileLangCSAIndexerLossAutoScaler``, whose
        tilelang branch computes exactly ``(P - Q) * coeff / valid_rows``.

        Recompute: the attention mask is the caller's own row-end vector, so the
        two forwards of a recompute segment are bit-identical and there is no
        top-k tie-break to worry about. The loss is
        attached on the grad-enabled forward only, so it is counted once; the
        no-grad forward skips the indexer entirely rather than computing and
        discarding it.

        CP: ``index_k`` is all-gathered to ``s_global`` inside
        ``forward_before_topk`` and ``kv`` by the caller, ``valid_range`` is built
        over the global sequence and row-sliced, and ``valid_rows`` is the global
        valid-row count -- so the per-rank losses sum to the single-rank one and
        no ``/cp_size`` correction is needed.

        Two KL backends, chosen by ``config.csa_indexer_backend``:

        * ``"tilelang"`` (default) -- the single ``csa_indexer_topk_fwd`` call
          described above. Its two bitonic buffers make the shared-memory
          request proportional to the candidate width, so it stops at
          ``_TILELANG_KL_MAX_WIDTH``;
        * ``"cudnn"`` -- the dense cuDNN triple, which has no top-k stage and
          therefore neither the shared-memory wall nor an ``[s, width]`` tensor
          that has to survive into the backward. See
          :meth:`_warmup_kl_dense_cudnn`.

        Both compute the same objective; the switch is a memory/width trade, not
        a change of loss. The field's third value, ``"unfused"``, has no
        full-candidate implementation of its own and is served by tilelang --
        explicitly, and with tilelang's width and head-count limits still
        enforced by name.
        """
        output = self._forward_full_causal(
            query,
            kv,
            v_b_proj_weight,
            kv_lora_rank,
            row_end,
        )
        if not self._needs_indexer_loss():
            return output

        backend = self.config.csa_indexer_backend
        if backend == "cudnn":
            return self._warmup_kl_dense_cudnn(
                output,
                query,
                kv,
                x,
                qr,
                doc_start,
                is_valid,
                doc_starts,
                input_ids,
                position_offset,
                s_local,
                s_global,
                meta=meta,
            )
        # ``__post_init__`` narrowed the field to {unfused, tilelang, cudnn}, so
        # everything left here is ``tilelang`` or ``unfused``, and both are
        # served by tilelang: this phase's KL never had a pure-paddle
        # implementation to offer as ``unfused`` (the reference
        # ``FusedDSAIndexerLoss`` is a top-k loss, not a full-candidate one), and
        # several existing hybrid-MLA suites configure ``unfused`` while running
        # this path. Tilelang's own two limits are not silent under that name:
        # the check below runs before the import and raises naming the width or
        # head bound and pointing at ``"cudnn"``.
        self._check_tilelang_indexer_support(s_global)

        from paddlefleet.tilelang_ops import csa_indexer_topk_fwd

        b = int(query.shape[0])
        index_q, index_k, weights = self._indexer_projections(
            x, qr, position_offset, grad_enabled=True
        )
        # ``DSAIndexer`` pre-bakes ``head_dim**-0.5`` into the weights, and the
        # tilelang indexer kernels apply ``dim**-0.5`` themselves -- same
        # convention as the cuDNN pair the sparse phase uses, and the opposite of
        # the pure-paddle ``FusedDSAIndexerLoss`` reference, which applies none.
        # Undo the pre-bake once, before both the forward call and the weights
        # handed to the backward, so the two agree.
        # Measured (validation_reports/precision_audit_20260809_022929/ops_edge):
        # against a plain-paddle reference the un-baked weights match to
        # max_abs 3.0e-8 / cosine 1-1.5e-13, while passing them through unscaled
        # gives max_abs 7.5e-1 / cosine 0.62.
        weights = weights * (float(self.indexer.head_dim) ** 0.5)
        # ``topk_effective`` is the causal span itself. The wrapper rounds it up
        # to a power-of-two multiple of its block internally and crops the result
        # back to the requested width (``csa_indexer_fwd.py:430-462`` /
        # ``csa_indexer_bwd.py:617-638``), so there is nothing to round here and
        # no surplus ``-1`` slot to carry. Measured at
        # s = 1/2/4/8/16/32/300/384/512/8192: the returned width equals
        # ``s_global`` exactly, the per-row valid-slot count equals the causal
        # length, rows sum to 1 within 9.6e-7 and the backward is finite.
        with paddle.no_grad():
            # No forced window in this phase, so the candidate range is the
            # whole per-document causal span.
            if meta is not None:
                valid_range, row_empty = meta.indexer_valid_range(
                    0, position_offset, s_local
                )
            else:
                valid_range, row_empty = self._indexer_valid_range(
                    s_global,
                    doc_start,
                    doc_len,
                    is_valid,
                    position_offset,
                    s_local,
                    window=0,
                )
            columns, probs = csa_indexer_topk_fwd(
                index_q.detach(),
                index_k.detach(),
                weights.detach(),
                ratio=1,
                topk_effective=s_global,
                seq_offset=position_offset,
                valid_range=valid_range,
            )
            columns = paddle.where(
                row_empty, paddle.full_like(columns, -1), columns
            )
            probs = paddle.where(columns >= 0, probs, paddle.zeros_like(probs))
            target = self._attn_target(query.detach(), kv, columns)
            loss_mask, valid_rows = self._indexer_loss_mask(
                input_ids, b, s_local
            )
            # Same reduction as ``_forward_sparse`` -- see the long comment there
            # for why the unmasked branch's ``/cp_size`` has to sit in
            # ``loss_coeff`` (and therefore reach the backward) rather than only
            # in the logged scalar the way ``csa_attention`` places it.
            loss_coeff = (
                self.indexer_loss_coeff
                if loss_mask is not None
                else self.indexer_loss_coeff / self.cp_size
            )
            kl = (
                target * (paddle.log(target + _EPS) - paddle.log(probs + _EPS))
            ).sum(axis=-1)
            if loss_mask is None:
                loss = kl.mean() * loss_coeff
            else:
                loss = (kl * loss_mask).sum() / valid_rows * loss_coeff

        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss=loss,
            layer_number=self.layer_number,
            num_layers=DSAIndexerLossLoggingHelper.get_total_num_layers(
                self.config
            ),
        )
        return TileLangCSAIndexerLossAutoScaler.apply(
            output,
            target,
            index_q,
            weights,
            index_k,
            columns,
            probs,
            loss_coeff,
            "tilelang",
            valid_rows,
            loss_mask,
        )

    # ------------------------------------------------------------------
    # warmup KL, dense (full-candidate cuDNN) backend
    # ------------------------------------------------------------------
    def _warmup_kl_dense_cudnn(
        self,
        output: Tensor,
        query: Tensor,
        kv: Tensor,
        x: Tensor,
        qr: Tensor,
        doc_start: Tensor,
        is_valid: Tensor,
        doc_starts: Tensor,
        input_ids: Tensor | None,
        position_offset: int,
        s_local: int,
        s_global: int,
        meta: MQADocMeta | None = None,
    ) -> Tensor:
        """Phase-2 KL over every causal candidate, on the dense cuDNN triple.

        Same objective as the tilelang branch, and deliberately the same
        arithmetic: identical KL expression, identical ``_EPS``, identical
        denominator convention. What changes is where the ``[s_local, width]``
        matrices live -- transient inside this forward and *recomputed* in the
        backward, rather than carried across by the autoscaler. At 64k/cp=8 that
        is two temporaries against three resident 2 GiB tensors per layer, and it
        is what removes the tilelang shared-memory ceiling entirely.

        Row gating collapses into a single ``row_active [s_local]``: a row is in
        the loss when its document is real (``is_valid``) *and* its token is not
        padding (``loss_mask``). With ``window=0`` those are exactly the rows the
        tilelang branch keeps, since ``row_empty == ~is_valid`` there. Setting
        ``attn_lse = +inf`` on an inactive row zeroes its whole attention-score
        row, hence ``target == 0`` and a zero KL term -- no separate mask
        multiply. The backward additionally forces ``index_lse = +inf`` there,
        which zeroes that row's gradient exactly.

        THD, not BSHD, even though the batch is always 1: the dense op's BSHD
        mode takes one ``q_causal_offsets`` entry per *batch* element
        (``_interface_sm100.py:748``), so it can only express a single causal
        span for the whole packed sequence. ``cu_seqlens`` is what makes
        per-document masking -- and the CP left border -- expressible at all, and
        it also compacts ``max_seqlen_k`` to the longest visible document. The
        layout change itself is free: ``[1, s, ...] -> [s, ...]`` is a view.
        """
        from paddlefleet.cudnn_ops import (
            dense_attn_kl_scores,
            dense_indexer_kl_scores,
            dense_kl_cu_seqlens,
        )

        self._check_cudnn_dense_indexer_support()
        # Prebuilt shared metadata, when the trainer made one: the tables it
        # holds are the very same global ones the inline path derives, so only
        # the source changes here -- and the segment-length D2H has already been
        # paid off the pipeline schedule.
        if meta is not None:
            doc_start, is_valid = meta.doc_start_per_pos, meta.is_valid
            segment_lens = meta.doc_segment_lens()
        else:
            segment_lens = _doc_segment_lens(doc_starts, s_global)
        # Left pre-baked, unlike the tilelang branch: the dense score is called
        # with ``sm_scale=1.0``, which skips the un-bake/re-bake bf16 round trip
        # (measured max_rel 8e-7 against an fp32 reference, versus 4.2e-4 when
        # un-baked). ``d_weights`` therefore comes back in pre-baked space too.
        index_q, index_k, weights = self._indexer_projections(
            x, qr, position_offset, grad_enabled=True
        )
        # Grad-carrying THD views: slicing off the batch axis is differentiable,
        # so the gradients this returns still reach the indexer parameters, and
        # the ``PyLayer`` gets tensors whose shape already matches what the dense
        # ops (and hence their gradients) use.
        iq, ik, iw = index_q[0], index_k[0], weights[0]
        query_thd, kv_thd = query[0].detach(), kv[0].detach()
        with paddle.no_grad():
            loss_mask, valid_rows = self._indexer_loss_mask(
                input_ids, 1, s_local
            )
            row_active = is_valid[position_offset : position_offset + s_local]
            if loss_mask is None:
                # ``kl.mean()`` on the tilelang side divides by *every* row,
                # doc-invalid ones included (they contribute 0). Same here.
                num_rows = float(s_local)
                coeff = self.indexer_loss_coeff / self.cp_size
            else:
                row_active = row_active & (loss_mask.reshape([s_local]) > 0)
                num_rows = float(valid_rows)
                coeff = self.indexer_loss_coeff

            cu_q, cu_k, max_q, max_k, q_off = dense_kl_cu_seqlens(
                segment_lens,
                position_offset,
                s_local,
            )
            thd = {
                "cu_seqlens_q": cu_q,
                "cu_seqlens_k": cu_k,
                "max_seqlen_q": max_q,
                "max_seqlen_k": max_k,
                "q_causal_offsets": q_off,
            }
            attn_lse = self._dense_kl_attn_lse(
                query_thd, kv_thd, doc_start, position_offset, row_active
            )
            index_score, index_lse = dense_indexer_kl_scores(
                iq.detach(), ik.detach(), iw.detach(), **thd
            )
            attn_score, attn_l1norm = dense_attn_kl_scores(
                query_thd, kv_thd, attn_lse, self.softmax_scale, **thd
            )
            loss = (
                self._dense_kl_reduce(
                    index_score,
                    index_lse,
                    # Masked columns come back as ``-inf``; clip before they
                    # reach the division by ``attn_l1norm``.
                    attn_score.clip_(min=0.0),
                    attn_l1norm,
                    num_rows,
                )
                * coeff
            )
            del index_score, index_lse, attn_score, attn_l1norm

        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss=loss,
            layer_number=self.layer_number,
            num_layers=DSAIndexerLossLoggingHelper.get_total_num_layers(
                self.config
            ),
        )
        plan = DenseWarmupKLPlan(
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            q_causal_offsets=q_off,
            # ``dense_indexer_kl_bwd`` hardcodes ``/ total_q`` (= ``s_local``)
            # inside the kernel, so pre-multiply to land on the ``/ num_rows``
            # this forward actually used. Same compensation as
            # ``csa_attention.py:1338-1349``.
            loss_coeff=coeff * s_local / num_rows,
            softmax_scale=self.softmax_scale,
        )
        return DenseWarmupIndexerLossAutoScaler.apply(
            output,
            iq,
            iw,
            ik,
            query_thd,
            kv_thd,
            attn_lse,
            row_active,
            plan,
        )

    def _dense_kl_attn_lse(
        self, query, kv, doc_start, position_offset, row_active
    ) -> Tensor:
        """``[s_local, h]`` fp32 per-head LSE over the KL candidate set.

        ``dense_attn_kl_scores`` needs the true per-head log-sum-exp *over the
        same candidate set* to produce the uniform head mixture the DSA KL
        targets; any other per-head constant silently yields a mass-weighted
        mixture instead. It cannot be taken from the attention forward: the
        flashmask facade flattens its LSE to ``[b, s]``
        (``flash_mask_facade.py:205``), which cannot hold a per-head one. And the
        candidate set is narrower than the attention's anyway -- no forced window,
        no sink -- so even a per-head FA4 LSE would be the wrong normaliser
        (measured gap 4.4e-3 at the production sink init; see
        :meth:`_attn_target`).

        So it is computed here, chunked over query rows. The logits are a bf16
        matmul rounded to bf16 on output while the cuDNN kernel scores in fp32;
        that shows up as a sub-percent per-head mass error, inside the band bf16
        reduction reordering already occupies on this path.
        """
        h, s_global = int(query.shape[1]), int(kv.shape[0])
        s_local = int(query.shape[0])
        seg_start = doc_start[position_offset : position_offset + s_local]
        rows = paddle.arange(
            position_offset, position_offset + s_local, dtype="int64"
        )
        chunk = max(1, _LSE_CHUNK_ELEMS // max(h * s_global, 1))
        parts = []
        for beg in range(0, s_local, chunk):
            end = min(beg + chunk, s_local)
            # Columns past the chunk's last row can never be in range, so the
            # matmul only covers ``[0, k_end)``. Under CP this is what keeps the
            # first rank from scoring the whole gathered KV.
            k_end = position_offset + end
            logits = paddle.matmul(
                query[beg:end], kv[:k_end], transpose_y=True
            ).astype("float32")
            cols = paddle.arange(k_end, dtype="int64").unsqueeze(0)
            keep = (cols >= seg_start[beg:end].unsqueeze(1)) & (
                cols <= rows[beg:end].unsqueeze(1)
            )
            # Additive rather than ``where``: broadcasting an ``[c, 1, k]`` bias
            # over ``[c, h, k]`` needs no shape gymnastics, and ``_NEG_INF``
            # rather than ``-inf`` keeps the sum away from ``nan``.
            bias = paddle.where(
                keep,
                paddle.zeros_like(keep, dtype="float32"),
                paddle.full(keep.shape, _NEG_INF, dtype="float32"),
            ).unsqueeze(1)
            parts.append(
                paddle.logsumexp(logits * self.softmax_scale + bias, axis=-1)
            )
        lse = paddle.concat(parts, axis=0)
        # ``+inf`` on an inactive row: ``exp(score - inf) == 0`` for every column,
        # so the score row is all zeros and the row leaves the loss exactly.
        return paddle.where(
            row_active.unsqueeze(-1), lse, paddle.full_like(lse, float("inf"))
        )

    @staticmethod
    def _dense_kl_reduce(
        index_score, index_lse, attn_score, attn_l1norm, num_rows
    ) -> Tensor:
        """``sum(KL) / num_rows`` over the two dense score matrices, chunked.

        Character-identical to the tilelang branch's reduction --
        ``target * (log(target + _EPS) - log(probs + _EPS))`` summed over columns
        -- so the two backends are comparable to the last epsilon. Masked columns
        arrive as ``probs == 0`` (``exp(-inf - lse)``) and ``target == 0`` (the
        caller clipped ``-inf`` away), whose term is an exact 0.
        """
        rows, width = (int(dim) for dim in index_score.shape)
        chunk = max(1, _KL_CHUNK_ELEMS // max(width, 1))
        total = paddle.zeros([], dtype="float32")
        for beg in range(0, rows, chunk):
            end = min(beg + chunk, rows)
            probs = paddle.exp(
                index_score[beg:end] - index_lse[beg:end].unsqueeze(-1)
            )
            target = attn_score[beg:end] / attn_l1norm[beg:end].unsqueeze(
                -1
            ).clip(min=_EPS)
            log_ratio = paddle.log(target + _EPS) - paddle.log(probs + _EPS)
            total = total + (target * log_ratio).sum()
        return total / num_rows

    def _check_cudnn_dense_indexer_support(self) -> None:
        """The one dense-cuDNN constraint that is not implied by the layer.

        ``DenseIndexerBackward.check_support`` rejects fewer than 64 index heads
        outright (``dense_indexer_backward_sm100.py``), and unlike the forward it
        does so from inside the backward, i.e. from a recompute replay. Reject it
        here instead. The batch size needs no check -- ``forward`` already
        requires 1.

        Not a reason to point at tilelang: that path pins ``index_n_heads`` to
        exactly 64 as well (see :meth:`_check_tilelang_indexer_support`), so a
        narrower indexer has no warmup KL backend at all.
        """
        heads = int(self.indexer.n_heads)
        if heads < 64:
            raise ValueError(
                "the dense cuDNN indexer backward requires index_n_heads >= 64, "
                f"got {heads}. Raise index_n_heads to 64; the tilelang backend "
                "requires exactly 64 too, so it is not a way around this."
            )

    def _check_tilelang_indexer_support(self, width: int) -> None:
        """Fail loudly on the two tilelang indexer constraints we cannot absorb.

        Neither is a config-time check, on purpose: both depend on the geometry
        this layer is actually handed, and hoisting them would make every
        small-geometry unit fixture unrepresentable.

        **Width.** Any causal span is served *functionally* -- the wrappers round
        ``topk_effective`` up to a power-of-two multiple of their block and crop
        the result back (``csa_indexer_fwd.py:430-462``,
        ``csa_indexer_bwd.py:617-638``), measured at
        s = 1/2/4/8/16/32/300/384/512/8192. Shared memory is the real limit: the
        two bitonic buffers are sized ``2 * topk``, so one block requests
        ``16 * width + 25344`` B against the SM100 opt-in limit of 232448 B.
        Past ``_TILELANG_KL_MAX_WIDTH`` the launch dies with a bare ``Failed to
        set the allowed dynamic shared memory size`` raised from wherever the
        segment is *replayed* (``recompute.py:389``), not from here, so reject it
        at the call site and name the way out.

        **Head count.** ``index_n_heads`` other than 64 trips the kernel's warp
        tiling with a bare ``Check failed: (m_warp * n_warp == num_warps)`` from
        inside tilelang (measured with 8).
        """
        if int(width) > _TILELANG_KL_MAX_WIDTH:
            raise ValueError(
                "the tilelang full-candidate indexer cannot cover a candidate "
                f"width of {int(width)}: its shared-memory request "
                f"({16 * int(width) + 25344} B) exceeds the device limit past "
                f"{_TILELANG_KL_MAX_WIDTH} columns. Set "
                'csa_indexer_backend="cudnn" to run the warmup KL on the dense '
                "cuDNN indexer instead, which has no top-k stage and so no "
                "width-proportional shared memory."
            )
        heads = int(self.indexer.n_heads)
        if heads != 64:
            raise ValueError(
                "the tilelang indexer's warp tiling requires index_n_heads == 64 "
                f"(measured: 8 fails inside the kernel), got {heads}."
            )

    # ------------------------------------------------------------------
    # index construction / kernel plumbing
    # ------------------------------------------------------------------
    def _indexer_projections(self, x, qr, position_offset, grad_enabled):
        """``(index_q, index_k, weights)`` from the DSA indexer.

        ``x`` / ``qr`` are always detached first: the indexer loss must never
        flow back into the backbone, independently of whether the backbone
        parameters are frozen. ``index_k`` comes back all-gathered to
        ``s_global`` when CP is on (the indexer gathers the 128-wide key rather
        than the hidden states, which is ~32x less traffic).

        ``weights`` is returned exactly as ``DSAIndexer.forward_before_topk``
        produced it, i.e. carrying ``n_heads**-0.5 * head_dim**-0.5``. Every
        kernel-backed caller must undo the ``head_dim`` half itself, because both
        the cuDNN and the tilelang indexer kernels re-apply ``dim**-0.5``
        internally; only a pure-paddle evaluation of the score (as in
        ``dsa_attention.FusedDSAIndexerLoss``) uses it unscaled.
        """
        x_det, qr_det = x.detach(), qr.detach()
        if grad_enabled:
            x_det.stop_gradient = False
            qr_det.stop_gradient = False
            return self.indexer.forward_before_topk(
                x_det, qr_det, position_offset, self.cp_group
            )
        with paddle.no_grad():
            return self.indexer.forward_before_topk(
                x_det, qr_det, position_offset, self.cp_group
            )

    def _indexer_valid_range(
        self,
        s_global,
        doc_start,
        doc_len,
        is_valid,
        position_offset=0,
        s_local=None,
        window=None,
    ):
        """Candidate range per query, in **global token** space.

        ``window`` is how many trailing causal tokens to exclude, i.e. the
        forced local window the sparse phase adds separately: clamping the right
        edge to ``doc_start + causal_len - window`` removes every duplicate while
        leaving the full top-k budget for distant tokens. Because the clamped end
        never exceeds the kernel's own causal limit, no masked ``-inf`` column can
        enter the top-k. Defaults to ``self.csa_window_size``; the warmup phase
        passes ``0`` because it has no forced window -- its candidate set is the
        whole per-document causal span.

        Built over the global sequence and row-sliced to this CP rank; the two
        columns stay global token ids, which is what the kernel's
        ``seq_offset``-aware causal bound expects.

        Returns:
            ``(valid_range [1, s_local, 2] int32, row_empty [1, s_local, 1])``.
        """
        if window is None:
            window = self.window_size
        positions = paddle.arange(s_global, dtype="int64")
        causal_avail = paddle.minimum(positions - doc_start + 1, doc_len)
        n_avail = paddle.clip(causal_avail - window, min=0)
        n_avail = paddle.where(is_valid, n_avail, paddle.zeros_like(n_avail))
        valid_range = paddle.stack(
            [doc_start, doc_start + n_avail], axis=-1
        ).cast("int32")
        if s_local is not None and s_local != s_global:
            valid_range = valid_range[
                position_offset : position_offset + s_local
            ]
            n_avail = n_avail[position_offset : position_offset + s_local]
        rows = int(valid_range.shape[0])
        return valid_range.unsqueeze(0), (n_avail == 0).reshape([1, rows, 1])

    def _sparse_attn(
        self, query, kv, token_indices, sm_scale, d_v, indexer_topk=0
    ):
        """Sparse MQA over the absorbed latent, via the shared cudnn backend.

        Same FlashMLA sparse forward + cuDNN DSA backward pair that the CSA/HCA
        layers use; the absorbed layout only differs in ``d_v`` (512 value dims
        out of a 576-wide query/key) and in the sink being optional --
        ``softmax_offset`` is ``None`` when ``add_full_attention_sink_bias`` is
        off, which the backend turns into a sinkless softmax. Query-head padding
        to the DSA-fixed ``h_q == 64`` is the backend's job.

        The forward is always FlashMLA sparse; the backward kernel is selected
        by ``mqa_sparse_attn_backward_backend`` (see ``__init__``).

        ``indexer_topk > 0`` additionally returns the LSE over the first
        ``indexer_topk`` columns, which is the indexer-loss target's normalizer.
        """
        from paddlefleet.fusions.mqa_sparse_attn import mqa_sparse_attn

        return mqa_sparse_attn(
            query,
            kv,
            token_indices,
            sm_scale,
            d_v,
            attn_sink=self.softmax_offset,
            indexer_topk=indexer_topk,
            sink_grad_fusion=self.sink_grad_fusion,
            backward_backend=self.sparse_attn_backward_backend,
        )

    @staticmethod
    def _assert_dense_fa4(q_dim: int, v_dim: int, row_end: Tensor) -> None:
        """Refuse to run the full-causal phases on anything but FA4.

        Applies to both phases that attend over the whole per-document causal
        span -- ``mqa_full_causal`` and the attention half of the warmup. FA4 is
        the only backend they support: query/key are
        ``kv_lora_rank + qk_rope_head_dim`` wide while the value is only
        ``kv_lora_rank``, a pair FA2/FA3 serve by padding the value out to the
        query width, and the alternative of expanding the document mask into a
        ``[b, s, s]`` column table for the sparse kernel costs ``O(s^2)`` memory
        (100 GiB at s=65536 on the production geometry -- see ``_dense_attn``)
        and is not addressable in int32 past s=46336. Neither is worth having as
        a silent substitution, so fail here instead.

        ``get_fa_version`` answers from the head-dim whitelist *and* two
        process-global flags, so all three inputs go into the message -- the
        head-dim pair alone rarely explains the answer. In particular
        ``FLAGS_flash_attn_version`` must be 4, which production sets from the
        compute capability (``TrainingArguments.__post_init__``, SM100 -> 4), and
        ``FLAGS_cudnn_deterministic`` must be off, since FA4's big-head-dim
        backward has no ordered-accumulation variant
        (``flash_mask/cute/interface.py:1238,1249``).

        ``get_fa_version`` answers from flags alone and never checks that an FA4
        backend exists, so the extension's availability is a separate condition
        here. Without it the facade binds ``_flashmask_attention`` to Paddle's
        native implementation (``flash_mask_facade.py``, the ``else`` branch of
        the ``is_flash_mask_available()`` import guard), which serves neither
        this head-dim pair nor the sink, and the failure would surface from
        inside the kernel call as ``Invalid flash attention version: 4`` -- true
        but not actionable.

        Checked every forward rather than in ``__init__``: the flags are
        settable at any point and the check is a whitelist lookup.
        """
        from paddlefleet_ops import is_flash_mask_available
        from paddlefleet_ops.flash_mask_facade import get_fa_version

        fa_version = get_fa_version(q_dim, v_dim, row_end)
        flash_mask_ok = is_flash_mask_available()
        if fa_version != 4 or not flash_mask_ok:
            flags = paddle.get_flags(
                ["FLAGS_flash_attn_version", "FLAGS_cudnn_deterministic"]
            )
            raise RuntimeError(
                "latent MQA full-causal attention requires FA4 dense "
                f"flashmask, but head dims ({q_dim}, {v_dim}) resolve to "
                f"FA{fa_version} and flash_mask_available="
                f"{flash_mask_ok}. FLAGS_flash_attn_version="
                f"{flags['FLAGS_flash_attn_version']}, "
                "FLAGS_cudnn_deterministic="
                f"{flags['FLAGS_cudnn_deterministic']}. Run on a device whose "
                "compute capability selects FA4 (SM100+, which is also what the "
                "sparse phase-3 kernels require) with the flash_mask (cute) "
                "extension built into paddlefleet_ops, leave "
                "FLAGS_flash_attn_version at the value the trainer derives, and "
                "keep FLAGS_cudnn_deterministic off -- FA4 has no deterministic "
                "backward for this head-dim pair."
            )

    def _dense_attn(
        self, query: Tensor, kv: Tensor, row_end: Tensor, kv_lora_rank: int
    ) -> Tensor:
        """Per-document dense causal MQA on the absorbed latent, via FA4.

        Mathematically the same softmax over the same column set the phase-3
        sparse kernel would compute from an explicit ``[b, s, s]`` table, but the
        causal/document structure stays in ``startend_row_indices`` -- the
        caller's own ``attn_mask_startend_row_indices``, already in flashmask's
        per-column "masked from this row on" convention -- so the mask never gets
        expanded.

        That expansion is what would make the full-causal phases quadratic.
        Measured peaks at s = 8192 / 32768 / 65536, production geometry (h=64,
        d_v=256), net allocated over the module's own steady state:

        * building the table alone: 1.6 / 25.0 / 100.0 GiB -- 6.25x the table's
          own ``s^2`` int32 (0.25 / 4.0 / 16.0 GiB), the rest being int64
          intermediates in ``_derive_csa_doc_boundaries``;
        * dense forward + backward: 3.3 / 13.0 / 26.0 GiB, i.e. linear.

        Past s=46336 that table is not merely expensive but unusable: the sparse
        kernel flattens it with int32 ids
        (``csa_sparse_attn_utils.py::_local_to_global_flat``, ``row * topk + col``
        after padding ``topk`` up to 64), so the row base wraps -- measured
        s=46337 returns finite but wrong numbers for its last 55 rows (cosine
        0.7296 against this backend) and only s=46341 crashes.

        The key is broadcast MQA (one head against ``h`` query heads) and the
        value is its first ``kv_lora_rank`` channels, so nothing is materialised
        per head. Output is flattened to the ``[b, s, h * kv_lora_rank]`` layout
        ``_deabsorb`` consumes.

        Under CP the row bounds are localised first and the kernel's own causal
        mode is turned off; see ``_cp_row_bounds``.
        """
        from paddlefleet_ops.flash_mask_facade import flashmask_attention

        if self.cp_size > 1:
            row_end = self._cp_row_bounds(row_end, int(query.shape[1]))

        key = kv.unsqueeze(2)
        query, key, value, sink = _dense_pylayer_inputs(
            query, key, key[..., :kv_lora_rank], self.softmax_offset
        )
        core_out = flashmask_attention(
            query,
            key,
            value,
            startend_row_indices=row_end,
            causal=self.cp_size == 1,
            learnable_sink=sink,
            softmax_scale=self.softmax_scale,
        )
        b, s = core_out.shape[:2]
        return core_out.reshape([b, s, -1])

    def _cp_row_bounds(self, row_end: Tensor, s_local: int) -> Tensor:
        """Global ``[LTS]`` row bounds -> this rank's local ``[LTS, UTE]`` pair.

        ``flashmask_attention``'s own ``causal=True`` bottom-right-aligns the
        diagonal: it masks column ``j`` above row ``j + seqlen_k - seqlen_q``
        (``flash_mask/cute/flashmask_utils.py:650``, ``:411``, and the comment at
        ``flash_bwd_sm100_bigd.py:2009``). Under CP this rank holds query rows
        ``[cp_rank * s_local, (cp_rank + 1) * s_local)`` against an all-gathered
        ``s_global`` key, so that implied offset -- ``s_global - s_local`` -- is
        only the right one for the last rank. Handing the kernel the global
        ``row_end`` with ``causal=True`` therefore returns *silently* wrong
        numbers on every other rank (measured cosine 2.7e-1 at cp_size=2, no
        exception), and the document bounds are wrong on every rank including the
        last, since their values are global row ids compared against local ones.

        The fix is the contract the rest of the model's CP attention already
        uses: express the diagonal as an explicit second flashmask bound instead
        of asking the kernel for it, then shift both bounds into this rank's row
        space. ``DotProductAttention`` does exactly this at
        ``dot_product_attention.py:614-617`` (``is_causal = False`` plus a
        two-column mask), the HySparse MLA scorer at
        ``multi_latent_attention.py:1233-1245``, and ``FlashMaskContextParallel``
        rejects ``causal=True`` outright for the same reason
        (``context_parallel_utils.py:1116-1119``).

        With ``causal=False`` and two bounds the kernel reads them as
        ``[LTS, UTE]`` (``flashmask_utils.py:261-270``) and masks
        ``row >= LTS or row < UTE`` (``mask.py:513-518``), i.e. column ``j`` is
        visible on rows ``[UTE_j, LTS_j)``. So ``UTE_j = j`` reproduces the
        causal diagonal in *global* rows, ``LTS_j`` stays the caller's document
        end, and ``preprocess_index`` -- the same helper
        ``cp_flashmask_allgatherkv_balance_forward`` uses for this mode
        (``context_parallel_utils.py:716-721``) -- shifts both by
        ``cp_rank * s_local`` and clips to ``[0, s_local]``. Only
        ``contiguous_allgather`` is localised this way, which is the only mode
        this class accepts (``__init__``).

        Worked example, ``s_global=8``, two documents ``0-3`` / ``4-7``,
        ``cp_size=2``. Global bounds ``LTS=[4,4,4,4,8,8,8,8]``,
        ``UTE=[0,1,2,...,7]``. On rank 1 (rows 4-7) they become
        ``LTS'=[0,0,0,0,4,4,4,4]`` and ``UTE'=[0,0,0,0,0,1,2,3]``: document A's
        columns are masked on every local row (``LTS'=0``), and column 5 is
        visible on local rows 1-3, i.e. global rows 5-7. Both correct, where
        ``causal=True`` on the unshifted bounds would have left document A fully
        visible.
        """
        s_global = int(row_end.shape[2])
        causal_end = paddle.arange(s_global, dtype=row_end.dtype).reshape(
            [1, 1, s_global, 1]
        )
        bounds = paddle.concat(
            [row_end, paddle.expand_as(causal_end, row_end)], axis=-1
        )
        return preprocess_index(
            bounds,
            chunk_id=self.cp_rank,
            seq_blocksize=s_local,
            max_seqlen_q=s_local,
        )

    @staticmethod
    def _deabsorb(core_out, v_b_proj_weight, split_kv_b=False) -> Tensor:
        """``[b, s, h * kv_lora_rank]`` -> ``[b, s, h * v_head_dim]``."""
        b, s, _ = core_out.shape
        if split_kv_b:
            # ``v_b_proj``: [h, v_head_dim, kv_lora_rank], the grouped-matmul
            # ``[G, R, D]`` contract -- one Triton GEMM, no transpose.
            from paddlefleet.triton_ops import fused_grouped_matmul

            h, v_head_dim, kv_lora_rank = v_b_proj_weight.shape
            out = fused_grouped_matmul(
                core_out.reshape([b, s, h, kv_lora_rank]), v_b_proj_weight
            )
        else:
            kv_lora_rank, h, v_head_dim = v_b_proj_weight.shape
            out = core_out.reshape([b, s, h, kv_lora_rank])
            out = paddle.einsum("bshl,lhv->bshv", out, v_b_proj_weight)
        out = out.reshape([b, s, h * v_head_dim])
        return out

    # ------------------------------------------------------------------
    # sparse (phase 3)
    # ------------------------------------------------------------------
    def _forward_sparse(
        self,
        query,
        kv,
        x,
        qr,
        v_b_proj_weight,
        doc_start,
        doc_len,
        is_valid,
        doc_lens,
        kv_lora_rank,
        input_ids=None,
        position_offset=0,
        meta=None,
    ) -> Tensor:
        """Phase 3: attention consumes window + top-k, KL on that same set.

        The indexer is trained enough to steer attention, so both sides narrow to
        the selected set. Reached only when ``_phase() == "sparse"``, i.e.
        ``dsa_indexer_use_sparse_loss=True``; the warmup phase has its own branch
        and shares no loss code with this one.

        Recompute: the loss is attached on the grad-enabled forward only, so under
        full recompute it is counted once, on the second pass. Both passes see
        identical inputs and reselect the same columns. (The top-k tie-break is
        not bit-stable in every shape -- one document spanning the whole sequence
        drifts by ~2% of the emitted slots between identical calls -- but the
        selected *set* is, and any residual drift would only change which columns
        the backward differentiates.)
        """
        from paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk_fwd,
        )

        b, s = int(query.shape[0]), int(query.shape[1])
        s_global = s * self.cp_size
        need_loss = self._needs_indexer_loss()

        with paddle.no_grad():
            if meta is not None:
                window_idxs = meta.window_topk_idxs(
                    self.window_size, position_offset, s
                )
                valid_range, row_empty = meta.indexer_valid_range(
                    self.window_size, position_offset, s
                )
            else:
                window_idxs = _build_window_topk_idxs_from_doc_bounds(
                    b, s_global, self.window_size, doc_start, is_valid
                ).cast("int32")
                if self.cp_enabled:
                    window_idxs = window_idxs[
                        :, position_offset : position_offset + s
                    ]
                valid_range, row_empty = self._indexer_valid_range(
                    s_global, doc_start, doc_len, is_valid, position_offset, s
                )
            # ``row_empty`` stays contiguous -- it is applied after the results
            # come back. ``vr_zz`` is the same table read in dual-chunk row
            # order, for the two kernel calls: both sources slice a cached
            # global table by ``(offset, length)``, so asking twice with the
            # chunk offsets is exactly the row set the swap produces.
            vr_zz = (
                self._dualchunk_valid_range(
                    meta, s_global, doc_start, doc_len, is_valid, s
                )
                if self.indexer_dualchunk
                else None
            )

        q_idx, k_idx, w_idx = self._indexer_projections(
            x, qr, position_offset, grad_enabled=need_loss
        )
        # ``DSAIndexer`` pre-bakes ``head_dim**-0.5`` into the weights, but both
        # cuDNN indexer kernels apply ``dim**-0.5`` themselves (the backward one
        # hardcodes it). Undo the pre-bake once so forward and backward agree.
        w_idx = w_idx * (float(self.indexer.head_dim) ** 0.5)

        # The cuDNN top-k backward (``indexer_backward_sm100.__init__``) asserts
        # ``topk % block_I == 0`` with ``block_I = 128``, so keep the configured
        # budget instead of clamping it to the sequence length. Short rows come
        # back ``-1`` padded. One width, consumed by attention and by the KL --
        # that identity *is* this phase.
        topk = int(self.indexer.index_topk)
        # The THD/varlen fast path builds ``cu_seqlens_k`` from ``doc_lens``,
        # i.e. it assumes a document-compacted K buffer. At ratio 1 the K buffer
        # is the raw token sequence, so the two only coincide when the documents
        # exactly tile the sequence; otherwise fall back to the dense path.
        # ``doc_lens`` stays global: ``_make_cu_seqlens`` is CP-aware and uses
        # ``seq_offset`` to pick out the documents this rank queries
        # (csa_indexer_fwd_cudnn.py:407-466).
        doc_lens_arg = (
            meta.cu_seqlens_arg()
            if meta is not None
            else (
                doc_lens.tolist() if int(doc_lens.sum()) == s_global else None
            )
        )

        with paddle.no_grad():
            if self.indexer_dualchunk:
                selected, scores_out = self._indexer_topk_dualchunk(
                    q_idx, w_idx, k_idx, topk, doc_lens_arg, vr_zz, need_loss
                )
            else:
                selected, _, *scores_out = cudnn_indexer_topk_fwd(
                    q_idx.detach(),
                    k_idx.detach(),
                    w_idx.detach(),
                    ratio=1,
                    topk_effective=topk,
                    valid_range=valid_range,
                    doc_lens=doc_lens_arg,
                    seq_offset=position_offset,
                    return_topk_scores=need_loss,
                )
            topk_indices = paddle.where(
                row_empty, paddle.full_like(selected, -1), selected
            )
            # ``[indexer topk, window]``, not the other way round: the kernel's
            # ``lse_indexer`` covers the *first* ``indexer_topk`` columns
            # (``flash_mla_sparse_fwd``), and that restricted per-head LSE is
            # exactly the normalizer the loss target needs -- with it every head
            # contributes mass 1 over the selected set, matching the per-head
            # softmax of ``_attn_target_python``. Feeding the *attention* LSE
            # instead (window + sink included) would turn the target into a
            # head-mass-weighted mixture: a different objective, invisible in the
            # forward output.
            #
            # Attention itself is a softmax over a set, so the order changes only
            # the accumulation order (hence the last bits), not the value. The
            # order is unconditional on purpose: gating it on ``need_loss`` would
            # make the two forwards of a full-recompute step disagree on the
            # table layout and on ``topk_length``.
            token_indices = paddle.concat(
                [topk_indices, window_idxs], axis=-1
            ).contiguous()
        token_indices.stop_gradient = True

        # The kernel only implements a handful of ``lse_indexer`` widths; other
        # budgets fall back to the Python target path.
        if topk in _LSE_INDEXER_TOPKS:
            core_out, lse_indexer = self._sparse_attn(
                query,
                kv,
                token_indices,
                self.softmax_scale,
                kv_lora_rank,
                indexer_topk=topk,
            )
        else:
            lse_indexer = None
            core_out = self._sparse_attn(
                query, kv, token_indices, self.softmax_scale, kv_lora_rank
            )
        output = self._deabsorb(core_out, v_b_proj_weight, self.split_kv_b)
        if not need_loss:
            return output

        with paddle.no_grad():
            (topk_scores,) = scores_out
            valid = topk_indices >= 0
            scores = paddle.where(
                valid,
                topk_scores.cast("float32"),
                paddle.full(topk_indices.shape, _NEG_INF, dtype="float32"),
            )
            topk_probs = F.softmax(scores, axis=-1)
            topk_probs = paddle.where(
                valid, topk_probs, paddle.zeros_like(topk_probs)
            )
            # The window is force-selected, so it is outside the indexer's
            # decision space and outside the KL: only the top-k columns.
            target = self._attn_target(
                query.detach(), kv, topk_indices, lse_indexer
            )
            kl = target * (
                paddle.log(target + _EPS) - paddle.log(topk_probs + _EPS)
            )
            # Masked reduction, same as csa_attention.py:1302-1306, over the same
            # row mask csa_attention.py:2411-2443 builds. It has to come from
            # ``input_ids``, not from the document metadata: a packed sequence's
            # trailing padding is folded into the last document's row range, so
            # ``attn_mask_startend_row_indices`` -- and therefore ``is_valid`` --
            # still marks those rows valid. ``input_ids != pad_token_id`` is the
            # backstop that catches them, keeping the padding out of both the
            # logged loss and the gradient denominator.
            loss_mask, valid_rows = self._indexer_loss_mask(input_ids, b, s)
            # Reduction shape follows ``csa_attention._compute_fused_indexer_target``
            # (:2325-2330) with one deliberate difference: the ``/cp_size`` of the
            # unmasked branch is folded into ``loss_coeff``, i.e. it also reaches
            # the **backward**, instead of being applied to the logged scalar
            # alone the way CSA does it at :2868-2869.
            #
            # That difference is load-bearing, not cosmetic. CP sums parameter
            # gradients across the group, so every rank must normalise by the
            # *global* row count. The masked branch gets that for free
            # (``valid_rows`` is global, and it is what ``num_rows_override``
            # hands the backward); the unmasked branch has no mask, so the
            # backward falls back to the kernel's own ``1/(B*Sq)`` -- a *local*
            # denominator -- and the only place left to correct it is the
            # coefficient. Copying CSA's placement here was measured: CP=2
            # ``test_4_indexer_loss_cp_normalisation`` fails on
            # ``indexer.wq_b.linear.weight`` with relative error ``1.000`` against
            # CP=1 (exactly a factor of ``cp_size``), in both phases, while the
            # logged scalar still matched. CSA has the same latent gap on its
            # unmasked CP path; production always supplies ``input_ids``, so it is
            # the masked branch that runs.
            loss_coeff = (
                self.indexer_loss_coeff
                if loss_mask is not None
                else self.indexer_loss_coeff / self.cp_size
            )
            kl_per_pos = kl.sum(axis=-1)
            if loss_mask is None:
                loss = kl_per_pos.mean() * loss_coeff
            else:
                loss = (kl_per_pos * loss_mask).sum() / valid_rows * loss_coeff

        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss=loss,
            layer_number=self.layer_number,
            num_layers=DSAIndexerLossLoggingHelper.get_total_num_layers(
                self.config
            ),
        )
        # Argument order follows ``TileLangCSAIndexerLossAutoScaler.forward``
        # (csa_attention.py): ``output, target, index_q, weights, index_k_comp,
        # topk_indices, topk_probs, loss_coeff, indexer_backend,
        # num_rows_override, loss_mask``. ``target`` sits second, right after
        # ``output`` -- the CSA call sites spell that as
        # ``apply(output, target, *state)``.
        return TileLangCSAIndexerLossAutoScaler.apply(
            output,
            target,
            q_idx,
            w_idx,
            k_idx,
            topk_indices,
            topk_probs,
            loss_coeff,
            self.indexer_backend,
            # ``num_rows_override`` + ``loss_mask``: the backward zeroes the pad
            # rows and rescales the cuDNN kernel's built-in ``1/(B*Sq)`` into
            # ``1/valid_rows``, matching the forward reduction above. Both
            # ``None`` (no ``input_ids``) leaves the kernel's own mean in place.
            valid_rows,
            loss_mask,
        )

    def _indexer_loss_mask(self, input_ids, b, s):
        """``([b, s] float32 row mask, its row count)`` from ``input_ids``.

        ``(None, None)`` when no ``input_ids`` reached this layer (inference and
        the direct-construction unit tests), which keeps the plain row mean.

        Under CP the mask is this rank's row slice but the denominator is the
        **global** valid-row count, so summing the per-rank losses reproduces the
        single-rank reduction. ``input_ids`` arrives sharded unless
        ``experimental_dataflow``, exactly as at ``csa_attention.py:2419-2428``.
        """
        if input_ids is None:
            return None, None
        pad_token_id = getattr(self.config, "pad_token_id", 0)
        # Explicit raise, not ``assert``: this validates a user-supplied config
        # field and must survive ``python -O`` (which strips asserts). A None
        # ``pad_token_id`` would silently compare every token against None and
        # mark the whole batch valid.
        if pad_token_id is None:
            raise ValueError(
                "pad_token_id must be set in config when input_ids is provided"
            )
        if self.cp_enabled:
            if not getattr(self.config, "experimental_dataflow", False):
                input_ids = ContextParallelGatherOp.apply(
                    input_ids, axis=1, mode=self.config.cp_balance_mode
                )
            loss_mask_global = (
                input_ids.reshape([b, self.cp_size * s]) != pad_token_id
            ).astype(paddle.float32)
            valid_rows = max(float(loss_mask_global.sum()), 1.0)
            offset = self.cp_rank * s
            return loss_mask_global[:, offset : offset + s], valid_rows
        loss_mask = (input_ids.reshape([b, s]) != pad_token_id).astype(
            paddle.float32
        )
        return loss_mask, max(float(loss_mask.sum()), 1.0)

    def _attn_target(self, query, kv, kl_columns, lse_indexer=None) -> Tensor:
        """KL target: head-summed attention probs over the indexer's own columns.

        The denominator is **the KL's candidate set and nothing else** -- not the
        forced window, not the sink. That is the definition of this loss, not an
        oversight: the indexer is only ever asked to rank the columns it can
        choose, and ``_indexer_valid_range`` explicitly clamps the window out of
        its candidate range, so neither the window nor the sink is in its decision
        space. Upstream does the same for the CSA layers --
        ``dsa_attention._compute_dsa_indexer_loss`` adds ``index_mask`` to
        ``attention_scores`` *before* the softmax, i.e. it normalises over the
        selected set.

        Worth recording because it was measured and is easy to misread as a bug:
        normalising over the full attention row instead (window + top-k + sink)
        gives a *different* target -- a head mixture weighted by each head's mass
        on the candidate set, ``Σ_h c_h·softmax_K(l^h) / Σ_h c_h``, rather than
        the uniform ``(1/H)·Σ_h softmax_K(l^h)`` here. The gap is 4.4e-3 at the
        production sink initialisation, 1.7e-2 at ``sink=+3``, and 1.9e-2 for the
        window term alone. Those are two objectives, not right and wrong; this one
        is the intended one.

        Args:
            query: ``[1, s, h, dk]`` detached absorbed query (local rows).
            kv: ``[1, s_global, dk]`` latent keys (all-gathered under CP).
            kl_columns: ``[1, s, w]`` int32 global column ids the KL scores,
                ``-1`` for empty slots. Column *order* is irrelevant, so the
                warmup phase passes the indexer's score-ordered table directly.
            lse_indexer: ``[1, s, 64]`` float32 per-head LSE over exactly
                ``kl_columns`` (``mqa_sparse_attn(indexer_topk=...)``). When
                present the cuDNN score-recompute kernel does the whole thing in
                one launch. ``None`` in the warmup phase -- its candidate set is
                not the attention set, so no matching LSE exists -- and when the
                budget is not one of ``_LSE_INDEXER_TOPKS``, so the Python path
                stays as the reference and the fallback.

        Returns:
            ``[1, s, w]`` float32 rows summing to 1 (0 for empty rows).
        """
        if lse_indexer is not None:
            return self._attn_target_cudnn(query, kv, kl_columns, lse_indexer)
        return self._attn_target_python(query, kv, kl_columns)

    def _attn_target_cudnn(self, query, kv, kl_columns, lse_indexer) -> Tensor:
        """``_attn_target`` via the cuDNN DSA score-recompute kernel.

        The kernel computes ``sum_h exp(Q_h·K_i*scale - LSE_h)`` L1-normalised
        over the selected columns. With ``LSE_h`` restricted to those same
        columns each head contributes mass 1, which is what the per-head softmax
        of :meth:`_attn_target_python` produces.

        Two different head paddings meet here, and they pull in opposite
        directions:

        * The *attention* kernel pads the query to its fixed ``h_q == 64``, and
          those pad heads must stay out of the head sum. Hence the LSE is sliced
          down to the layer's real ``h`` -- the query itself is already the
          unpadded one.
        * The *target* kernel uses the query-head count as its MMA ``M`` tile
          (``_dispatch_sparse_attn_tile_params``: ``m = qhead_per_kv_head``), and
          only powers of two from 16 up work: 24/40/48/80 raise, and ``h == 8``
          -- the unit fixture's width -- silently returns an all-zero target,
          which would make the KL target uniform-after-renormalisation with no
          error anywhere. So pad back up to ``_TARGET_QHEAD_MIN`` when the layer
          is narrower.

        The pad heads carry an **infinite** LSE, so ``exp(score - inf) == 0``
        keeps them out of the head sum exactly rather than approximately (the
        query pad rows are zeros, whose score is a finite 0). Measured against
        :meth:`_attn_target_python` at ``h=8 -> 16``: 7.5e-4 max abs error, i.e.
        bf16 matmul noise.
        """
        from paddlefleet_ops.cudnn.deepseek_sparse_attention import (
            sparse_attn_score_recompute_wrapper,
        )

        b, s, h, dk = (int(dim) for dim in query.shape)
        idx = kl_columns.cast("int32").contiguous()
        lse = lse_indexer[:, :, :h].cast("float32")
        h_padded = max(_TARGET_QHEAD_MIN, 1 << (h - 1).bit_length())
        if h_padded != h:
            pad = h_padded - h
            query = paddle.concat(
                [query, paddle.zeros([b, s, pad, dk], dtype=query.dtype)],
                axis=2,
            )
            lse = paddle.concat(
                [lse, paddle.full([b, s, pad], float("inf"), dtype="float32")],
                axis=2,
            )
        target = sparse_attn_score_recompute_wrapper(
            _HashableTensor(query.contiguous()),
            _HashableTensor(kv.contiguous()),
            _HashableTensor(lse.contiguous()),
            _HashableTensor(idx),
            self.softmax_scale,
        )["target"]

        # Empty slots / all-empty rows: the kernel is not contracted to return
        # zeros there, and a row of zeros must stay a row of zeros (the KL
        # reduction divides by the valid-row count, not by the row sum).
        valid = idx >= 0
        target = paddle.where(valid, target, paddle.zeros_like(target))
        return target / target.sum(axis=-1, keepdim=True).clip(min=_EPS)

    def _attn_target_python(self, query, kv, kl_columns) -> Tensor:
        """Reference ``_attn_target``: per-head softmax over gathered keys.

        The tilelang ``csa_attn_target_reducesum`` kernel is not usable here (it
        requires a power-of-two head dim; the latent is 576) and the dense
        ``_compute_attn_target_on_selected_set`` materialises ``[b, h, s, s]``, so
        gather the selected keys in query-row chunks instead: ``s*h*w*dk`` MACs,
        no ``s*s`` tensor. The matmul runs in the input dtype (bf16) with fp32
        accumulation, as the tilelang kernel does internally for the CSA layers;
        the softmax and the L1 normalisation are fp32.
        """
        s, width = int(query.shape[1]), int(kl_columns.shape[-1])
        dk = int(query.shape[-1])
        chunk = max(1, _TARGET_ROW_SLOTS // width)
        q0, kv0, idx0 = query[0], kv[0], kl_columns[0]
        parts = []
        for start in range(0, s, chunk):
            end = min(start + chunk, s)
            idx_c = idx0[start:end].cast("int64")
            valid = idx_c >= 0
            safe = paddle.where(valid, idx_c, paddle.zeros_like(idx_c))
            k_sel = paddle.gather(kv0, safe.flatten(), axis=0).reshape(
                [end - start, width, dk]
            )
            scores = (
                paddle.matmul(q0[start:end], k_sel, transpose_y=True).cast(
                    "float32"
                )
                * self.softmax_scale
            )
            scores = paddle.where(
                valid.unsqueeze(1), scores, paddle.full_like(scores, _NEG_INF)
            )
            probs = F.softmax(scores, axis=-1).sum(axis=1)  # head-sum [c, w]
            parts.append(paddle.where(valid, probs, paddle.zeros_like(probs)))
        target = paddle.concat(parts, axis=0)
        target = target / target.sum(axis=-1, keepdim=True).clip(min=_EPS)
        return target.unsqueeze(0)
