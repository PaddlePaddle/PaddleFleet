# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Routed (MoE-style) N-gram Embedding -- learned high bits, hashed low bits.

This is a sibling of :class:`NgramEmbedding`, selected by the single config flag ``ngram_moe_enabled``.
When that flag is off nothing here is imported and the original N-gram path is bit-for-bit unchanged.

Addressing
----------
The baseline addresses a fixed sub-table with a pure hash, ``row = H(g) mod V_i``.
Here the high bits of the address are learned and the low bits stay hashed::

    addr(t, r) = offset[s] + H_s(g_t) mod V_s,    s = S_r(t)

``S_r(t)`` is the r-th sub-table picked by a router that reads a wide window of preceding tokens, and ``H_s`` is the same polynomial rolling hash taken modulo that sub-table's own modulus.
The router decides where to look, the hash decides where inside.

Sizing
------
The only quantity derived from the vocabulary is the row count of one sub-table, expressed as a multiple of it exactly like the baseline's ``ngram_vocab_size_ratio``::

    tables         = ngram_moe_tables_per_order * (N - 1)
    rows           = int(ngram_moe_table_rows_ratio * vocab_size)
    rows(table i)  = rows + 2*i + 1
    total_rows     = tables * rows + tables**2
    params         = total_rows * ngram_moe_table_dim
                     + (N - 1) * ngram_moe_table_dim * hidden_size
    lookups/token  = ngram_moe_active_tables * (N - 1)

Design notes
------------
* Only the addressing is conditional; the read-out projection is shared per N-gram order.
  So there is no grouped matmul, no All2All and no capacity factor: the ``active_tables`` reads of a token are exchangeable and one gather covers them all.
* Per-table moduli are kept pairwise distinct (``rows + 2*i + 1``), as the baseline keeps its K sub-tables distinct.
  Without that, all selected sub-tables would hash a key to the same offset and the multi-read redundancy would collapse.
* The ``active_tables`` reads are gate-combined in the d-dimensional table space before the projection, so the number of projections drops from ``K(N-1)`` to ``N-1``.
* The router must see strictly more context than the N-gram itself, otherwise it can only rediscover a hash of the same N-gram.
  Hence the depthwise Conv1D of width ``ngram_moe_router_width`` (default 32) >> N, reading only tokens up to the current position.
* The flat table is stored as several slices along the feature dimension, because one parameter may not exceed int32 numel (see ``_MAX_PARAM_NUMEL``).

Load balance is not enforced by default.  Two styles are available, selected by ``ngram_moe_balance_type``:

* ``"noaux_bias"`` -- the aux-loss-free bias of DeepSeek-V3, and the default here.
  No loss term: a per-sub-table bias shifts the selection score only and is nudged by ``sign(mean_load - load)`` after every optimizer step, so the combine weights stay those of the unbiased router.
  The nudge lives in ``ngram_bias_callback.NgramBiasAdjustCallback``, which has to be registered on the trainer; without it the bias never moves.
* ``"aux_loss"`` -- the classic Switch/GShard auxiliary loss, per order, gated by ``ngram_moe_balance_loss_enabled`` + ``ngram_moe_balance_loss_coef``.
* ``"none"`` -- no balancing.

Neither a dense warm-up nor a late router freeze is implemented.
Per-order independent routers can be enabled via ``ngram_moe_router_per_order``.

Parallelism
-----------
The tables, the projections and the router are not tensor-parallel sharded, so this is written for TP = 1; under TP > 1 every TP rank would hold a replica and recompute the same signal.
Data parallel and sharding are supported: the usage counts of the aux-loss-free bias are all-reduced over exactly the groups whose ranks hold different tokens, as in ``moe.qb_callback``.

Hierarchical gating
-------------------
``ngram_moe_order_gate_enabled`` adds a second gate one level up, over the N-gram orders, so the embedding becomes a two-level mixture::

    w(o, s | t) = beta_o(t) * g_{o,s}(t)
                  ^^^^^^^^   ^^^^^^^^^^^
                  gram level  sub-table level (top-A softmax, unchanged)

``beta`` is a dense softmax over the ``N - 1`` orders: every order is computed regardless, so unlike the sub-table level there is no capacity, no dropped token and nothing to load-balance, and this level carries no auxiliary loss.
With N = 3 a top-k gate over the two orders would either discard half the signal (k = 1) or be a no-op (k = 2), so dense is the only meaningful form here.

Two details matter:

* The gate multiplies the projected signal, not ``u``.  Each order owns its ``post_projs[oi]``, so the ``u`` of different orders live in unrelated table spaces and only become comparable after the projection into hidden space.
* With ``ngram_moe_order_gate_scale`` (default) the gate is multiplied by ``N - 1``, so the weights sum to ``N - 1`` rather than 1 -- the same total the plain sum it replaces had.
  This keeps the injected signal at the magnitude that ``normalizer = 1 + (N - 1)`` in :class:`LanguageModelEmbedding` was chosen for, and makes a freshly initialised gate reproduce the ungated forward pass.

Everything is off by default: with ``ngram_moe_order_gate_enabled=False`` no parameter and no operation is added.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import paddle
import paddle.nn.functional as F
from paddle import Tensor, nn

if TYPE_CHECKING:
    from paddlefleet.transformer.transformer_config import TransformerConfig


# Paddle's sharding-init broadcast splits with int32 sections, so a parameter
# with numel >= 2**31 aborts startup.  The flat table is sliced to stay below.
_MAX_PARAM_NUMEL = 2**31 - 1


def _num_table_shards(rows: int, emb_dim: int) -> int:
    """Smallest divisor of ``emb_dim`` that keeps every slice under int32."""
    for shards in range(1, emb_dim + 1):
        if emb_dim % shards:
            continue
        if rows * (emb_dim // shards) <= _MAX_PARAM_NUMEL:
            return shards
    raise ValueError(
        f"cannot store a [{rows}, {emb_dim}] table with at most "
        f"{_MAX_PARAM_NUMEL} elements per parameter; lower "
        "ngram_moe_table_rows_ratio or ngram_moe_table_dim"
    )


class NgramTableRouter(nn.Layer):
    """Wide-context scorer over ``num_tables`` choices, reading only past tokens.

    ``width - 1`` left padding plus a depthwise Conv1D gives every position a receptive field of ``width`` tokens ending at itself, so nothing after the position can leak in.
    Depthwise keeps the cost at ``router_dim * width`` MACs per token.

    Used at both levels: ``num_tables=tables_per_order`` scores the sub-tables of one order, ``num_tables=num_orders`` scores the N-gram orders themselves.
    """

    def __init__(
        self, hidden_size: int, num_tables: int, router_dim: int, width: int
    ):
        super().__init__()
        self.width = width
        self.down = nn.Linear(hidden_size, router_dim, bias_attr=False)
        self.conv = nn.Conv1D(
            router_dim, router_dim, width, groups=router_dim, bias_attr=False
        )
        self.score = nn.Linear(router_dim, num_tables, bias_attr=False)

    def forward(self, word_emb: Tensor) -> Tensor:
        """[B, S, H] -> [B, S, tables_per_order] routing logits."""
        z = self.down(word_emb).transpose([0, 2, 1])
        z = F.pad(z, [self.width - 1, 0], data_format="NCL")
        c = self.conv(z).transpose([0, 2, 1])
        return self.score(F.silu(c))


class NgramBalanceStatics(nn.Layer):
    """State of the aux-loss-free sub-table balancer.

    Mirrors the main MoE's ``moe_statics``: a persistable fp32 bias that the selection uses, and a non-persistable integer usage counter that the trainer callback drains after every optimizer step.

    It is a separate sub-layer because ``_cast_to_low_precision = False`` has to apply to the bias, which must stay fp32, and must not apply to the parent's embedding tables, which are cast to bf16 under O2.

    The bias is persistable so a warm restart keeps the balance the run already found; losing it only costs a transient load spike while it re-converges.

    Shapes are [num_orders, tables_per_order]: an order's sub-tables compete only against each other, like the auxiliary loss they replace.
    """

    def __init__(self, num_orders: int, tables_per_order: int):
        super().__init__()
        self.register_buffer(
            "table_bias",
            paddle.zeros([num_orders, tables_per_order], dtype="float32"),
            persistable=True,
        )
        self._cast_to_low_precision = False
        # Not a buffer: reset every step, never checkpointed, never dtype-cast.
        self.table_usage = paddle.zeros(
            [num_orders, tables_per_order], dtype="int64"
        )
        self.table_usage.stop_gradient = True


class NgramMoeEmbedding(nn.Layer):
    """N-gram embedding whose sub-table is chosen by a context router.

    Same contract as :class:`NgramEmbedding`, except that ``forward`` also takes the word embedding, which is the router's input.

    Args:
        config: TransformerConfig.  Shares ``ngram_emb_neighbor_num`` (N) and ``ngram_pad_token_id`` with the baseline; everything else comes from the routed variant's own fields:
            - ngram_moe_tables_per_order (int): sub-tables per N-gram order
            - ngram_moe_active_tables (int): sub-tables read per token per order
            - ngram_moe_table_rows_ratio (float): base row count of one sub-table, as a multiple of ``vocab_size``
            - ngram_moe_table_dim (int): embedding width of one sub-table
            - ngram_moe_router_dim (int): router bottleneck width
            - ngram_moe_router_width (int): router receptive field
            - ngram_moe_order_gate_enabled (bool): add the dense gram-level gate on top of the sub-table routers
            - ngram_moe_order_gate_scale (bool): keep the gate's weights summing to N - 1 instead of 1
        vocab_size: base vocabulary size, used as the radix of the polynomial hash and as the unit of the row-count ratio.
    """

    def __init__(self, config: TransformerConfig, vocab_size: int):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.vocab_size = vocab_size
        self.n = config.ngram_emb_neighbor_num
        self.pad_token_id = getattr(config, "ngram_pad_token_id", 0)
        assert self.n >= 2, f"ngram_emb_neighbor_num must be >= 2, got {self.n}"

        self.num_orders = self.n - 1
        # One "embedder" per order: the reads are combined before the projection.
        self.num_embedders = self.num_orders

        self.tables_per_order = int(config.ngram_moe_tables_per_order)
        self.active_tables = int(config.ngram_moe_active_tables)
        assert 1 <= self.active_tables <= self.tables_per_order, (
            f"ngram_moe_active_tables ({self.active_tables}) must be in "
            f"[1, ngram_moe_tables_per_order={self.tables_per_order}]"
        )
        self.num_tables = self.tables_per_order * self.num_orders

        self.table_rows_ratio = float(config.ngram_moe_table_rows_ratio)
        self.table_rows = int(self.table_rows_ratio * vocab_size)
        assert self.table_rows > 0, (
            "ngram_moe_table_rows_ratio * vocab_size must be a positive row "
            f"count, got {self.table_rows_ratio} * {vocab_size}"
        )
        self.emb_dim = int(config.ngram_moe_table_dim)
        assert self.emb_dim > 0, "ngram_moe_table_dim must be positive"

        self.normalizer = 1 + self.num_embedders

        # Sub-table moduli, order-major, so that the flat index
        # ``(order - 2) * tables_per_order + s`` identifies a sub-table globally.
        table_sizes, offsets, total = [], [], 0
        for idx in range(self.num_tables):
            size = self.table_rows + idx * 2 + 1
            table_sizes.append(size)
            offsets.append(total)
            total += size
        self.total_rows = total

        for oi in range(self.num_orders):
            lo = oi * self.tables_per_order
            mods = table_sizes[lo : lo + self.tables_per_order]
            self.register_buffer(
                f"table_mods_{oi}",
                paddle.to_tensor(mods, dtype="int64"),
                persistable=False,
            )
            self.register_buffer(
                f"table_offsets_{oi}",
                paddle.to_tensor(
                    offsets[lo : lo + self.tables_per_order], dtype="int64"
                ),
                persistable=False,
            )
            # pow_mod[s, j] = V^(j+1) mod mods[s], j < order - 1.
            pow_mod = []
            for mod in mods:
                row, p = [], 1
                for _ in range(oi + 1):
                    p = (p * self.vocab_size) % mod
                    row.append(p)
                pow_mod.append(row)
            self.register_buffer(
                f"table_pow_{oi}",
                paddle.to_tensor(pow_mod, dtype="int64"),
                persistable=False,
            )

        # One flat table over all (order, sub-table) pairs, so the routed address
        # is already the row index.  Sliced along the feature dim to stay within
        # _MAX_PARAM_NUMEL.
        self.num_shards = _num_table_shards(total, self.emb_dim)
        self.shard_dim = self.emb_dim // self.num_shards
        # fan_in / fan_out pinned to the unsharded shape, so the init does not
        # depend on the number of slices.
        self.table_shards = nn.LayerList(
            [
                nn.Embedding(
                    total,
                    self.shard_dim,
                    weight_attr=paddle.ParamAttr(
                        initializer=nn.initializer.XavierNormal(
                            fan_in=total, fan_out=self.emb_dim
                        )
                    ),
                )
                for _ in range(self.num_shards)
            ]
        )
        self.post_projs = nn.LayerList(
            [
                nn.Linear(self.emb_dim, self.hidden_size, bias_attr=False)
                for _ in range(self.num_orders)
            ]
        )

        # Router construction: either one shared router or one per order.
        self.router_per_order = bool(
            getattr(config, "ngram_moe_router_per_order", False)
        )
        _router_dim = int(getattr(config, "ngram_moe_router_dim", 64))
        _router_width = int(getattr(config, "ngram_moe_router_width", 32))
        if self.router_per_order:
            self.routers = nn.LayerList(
                [
                    NgramTableRouter(
                        hidden_size=self.hidden_size,
                        num_tables=self.tables_per_order,
                        router_dim=_router_dim,
                        width=_router_width,
                    )
                    for _ in range(self.num_orders)
                ]
            )
        else:
            self.router = NgramTableRouter(
                hidden_size=self.hidden_size,
                num_tables=self.tables_per_order,
                router_dim=_router_dim,
                width=_router_width,
            )

        # Sub-table load balancing.  Exactly one style is active.
        self.balance_type = str(
            getattr(config, "ngram_moe_balance_type", "noaux_bias")
        ).lower()
        assert self.balance_type in ("aux_loss", "noaux_bias", "none"), (
            f"unknown ngram_moe_balance_type {self.balance_type!r}; expected "
            "aux_loss, noaux_bias or none"
        )
        self.balance_loss_enabled = self.balance_type == "aux_loss" and bool(
            getattr(config, "ngram_moe_balance_loss_enabled", False)
        )
        self.balance_loss_coef = float(
            getattr(config, "ngram_moe_balance_loss_coef", 0.0)
        )
        # Aux-loss-free: bias the selection, never the combine weights.  The
        # update rate lives on the trainer callback that owns the update.
        self.balance_bias_enabled = self.balance_type == "noaux_bias"
        self.balance_statics = (
            NgramBalanceStatics(self.num_orders, self.tables_per_order)
            if self.balance_bias_enabled
            else None
        )

        # Gram-level gate, dense over orders: no capacity, nothing to balance.
        # Built only when enabled, so the disabled path adds no parameter.
        self.order_gate_enabled = (
            bool(getattr(config, "ngram_moe_order_gate_enabled", False))
            and self.num_orders > 1
        )
        self.order_gate_scale = (
            float(self.num_orders)
            if bool(getattr(config, "ngram_moe_order_gate_scale", True))
            else 1.0
        )
        if self.order_gate_enabled:
            self.order_router = NgramTableRouter(
                hidden_size=self.hidden_size,
                num_tables=self.num_orders,
                router_dim=int(
                    getattr(config, "ngram_moe_order_gate_dim", 0)
                    or _router_dim
                ),
                width=int(
                    getattr(config, "ngram_moe_order_gate_width", 0)
                    or _router_width
                ),
            )

        self._valid_mask_cache = {}

    def _valid_mask(self, seq_len: int, ngram: int) -> Tensor:
        """[1, S, 1] mask, False where the N-gram window is padding-filled."""
        key = (seq_len, ngram)
        if key not in self._valid_mask_cache:
            pos = paddle.arange(seq_len, dtype="int64").reshape([1, seq_len, 1])
            self._valid_mask_cache[key] = pos >= (ngram - 1)
        return self._valid_mask_cache[key]

    def _compute_shifted_ids(self, input_ids: Tensor) -> dict:
        """Right-shifted copies of input_ids, one per offset (same as baseline)."""
        shifted_ids = {}
        B, S = input_ids.shape
        for k in range(1, self.n):
            shifted = paddle.full(
                [B, k], self.pad_token_id, dtype=input_ids.dtype
            )
            shifted = paddle.concat([shifted, input_ids[:, : S - k]], axis=1)
            shifted_ids[k + 1] = shifted
        return shifted_ids

    def _route(self, word_emb: Tensor, order_idx: int = 0):
        """Top-k sub-table selection and gates for one N-gram order.

        Same arrangement as ``moe_router.MoERouter._topk_noaux_tc``: the bias enters the selection, the gate comes from the unbiased score.

        With ``router_per_order=True`` each order has its own router; otherwise the shared ``self.router`` is used.

        With ``balance_type="noaux_bias"`` the selection runs on ``softmax(logits) + bias`` while the returned gate is the top-A renormalisation of the unbiased ``softmax(logits)``.
        The bias is added to probabilities rather than logits so that ``bias_update_rate`` keeps one fixed meaning instead of drifting with the router's logit scale.
        Selecting on the biased score and combining on the unbiased one is what makes this loss-free: balancing moves which sub-tables a token reads, never the weights it reads them with.

        Returns:
            sel  : [B, S, active_tables] int64 sub-table indices
            gate : [B, S, active_tables] float32 softmax weights
            logits: [B, S, tables_per_order] float32 full router logits, for the balance loss; ``None`` otherwise, to avoid retaining it.
        """
        router = (
            self.routers[order_idx] if self.router_per_order else self.router
        )
        logits = router(word_emb).astype("float32")
        if self.balance_bias_enabled:
            probs = F.softmax(logits, axis=-1)
            bias = self.balance_statics.table_bias[order_idx].detach()
            sel = paddle.topk(probs + bias, self.active_tables, axis=-1)[1]
            top_scores = paddle.take_along_axis(probs, sel, axis=-1)
            # Renormalising the selected probabilities equals a softmax over the
            # selected logits, which is what the unbiased path does.
            gate = top_scores / top_scores.sum(axis=-1, keepdim=True).clip(
                min=1e-12
            )
        else:
            top_scores, sel = paddle.topk(logits, self.active_tables, axis=-1)
            gate = F.softmax(top_scores, axis=-1)
        keep_logits = self.balance_loss_enabled and self.balance_loss_coef > 0.0
        return sel.astype("int64"), gate, (logits if keep_logits else None)

    def _accumulate_usage(
        self, sel: Tensor, valid_mask: Tensor, order_idx: int
    ):
        """Add this micro-batch's per-sub-table selection counts to the counter.

        Masked-out positions are redirected to a sentinel bin that is then dropped, which keeps the whole update to one ``bincount`` and avoids materialising a [B, S, A, M] one-hot.
        """
        m = self.tables_per_order
        with paddle.no_grad():
            masked = paddle.where(valid_mask, sel, paddle.full_like(sel, m))
            counts = paddle.bincount(masked.reshape([-1]), minlength=m + 1)[:m]
            self.balance_statics.table_usage[order_idx] += counts.astype(
                "int64"
            )

    @staticmethod
    def _balance_loss(
        logits: Tensor, sel: Tensor, valid_mask: Tensor
    ) -> Tensor:
        """Classic Switch/GShard per-order auxiliary load-balancing loss.

        L = M * sum_i( p_mean_i * m_mean_i )

        where M = tables_per_order, p_mean_i = mean softmax prob over valid positions for sub-table i, and m_mean_i = mean of the binary selection indicator for sub-table i.

        Same form and same normalisation as ``moe_router.MoERouter._cal_aux_loss``, computed per N-gram order and restricted to valid positions.

        ``valid_mask``: [B, S, 1] bool, True where the position is not padding and the N-gram window has not run off the left edge.
        """
        # probs: [B, S, M]
        probs = F.softmax(logits, axis=-1)
        tables_per_order = logits.shape[-1]
        # sel: [B, S, A]  ->  one_hot: [B, S, A, M]  ->  sum over A: [B, S, M]
        sel_oh = (
            F.one_hot(sel, num_classes=tables_per_order)
            .astype("float32")
            .sum(-2)
        )

        mask_f = valid_mask.astype("float32")  # [B, S, 1]
        denom = mask_f.sum().clip(min=1.0)
        # p_mean_i and m_mean_i each have shape [M]
        p_mean = (probs * mask_f).sum(axis=[0, 1]) / denom
        m_mean = (sel_oh * mask_f).sum(axis=[0, 1]) / denom
        return paddle.sum(p_mean * m_mean) * float(tables_per_order)

    def _order_gate(self, word_emb: Tensor) -> Tensor:
        """Dense gram-level gate: softmax over the N-gram orders.

        One decision across orders, so it is computed once and the sub-table routers stay as ``ngram_moe_router_per_order`` configures them.

        Returns:
            [B, S, num_orders] float32 probabilities, unscaled; ``forward`` applies ``order_gate_scale``.
        """
        return F.softmax(self.order_router(word_emb).astype("float32"), axis=-1)

    def _addresses(
        self, input_ids: Tensor, shifted_ids: dict, sel: Tensor, order: int
    ) -> Tensor:
        """Routed row indices [B, S, A] into the flat table.

        ``offset[s] + (id[t] + sum_j id[t-j] * V^j mod V_s) mod V_s``, with the moduli gathered per selected sub-table.
        """
        oi = order - 2
        flat = sel.reshape([-1])
        shape = sel.shape
        mods = paddle.gather(getattr(self, f"table_mods_{oi}"), flat).reshape(
            shape
        )
        offs = paddle.gather(
            getattr(self, f"table_offsets_{oi}"), flat
        ).reshape(shape)
        pows = paddle.gather(getattr(self, f"table_pow_{oi}"), flat).reshape(
            [*shape, oi + 1]
        )
        key = input_ids.cast("int64").unsqueeze(-1)
        for j in range(order - 1):
            key = (
                key
                + shifted_ids[j + 2].cast("int64").unsqueeze(-1) * pows[..., j]
            )
        return key % mods + offs

    def forward(self, input_ids: Tensor, word_emb: Tensor) -> Tensor:
        """Compute the routed N-gram signal.

        Args:
            input_ids: [B, S] token ids.
            word_emb: [B, S, H] word embeddings; the router's context input.

        Returns:
            [B, S, H] signal, to be added to the word embedding and divided by ``normalizer`` together.
        """
        from paddlefleet.transformer.moe.moe_utils import AddAuxiliaryLoss

        shifted_ids = self._compute_shifted_ids(input_ids)

        # One dense weight per order per token, or None when the gate is off, in
        # which case the loop below is the original plain sum.
        order_probs = (
            self._order_gate(word_emb) if self.order_gate_enabled else None
        )
        beta = (
            None if order_probs is None else order_probs * self.order_gate_scale
        )

        ngram_output = None
        total_balance_loss = None
        pad_mask = (input_ids != self.pad_token_id).unsqueeze(-1)  # [B, S, 1]
        for order in range(2, self.n + 1):
            oi = order - 2
            # valid = not padding AND position has enough left context.
            valid_mask = pad_mask & self._valid_mask(input_ids.shape[1], order)
            sel, gate, logits = self._route(word_emb, order_idx=oi)
            if self.balance_bias_enabled:
                self._accumulate_usage(sel, valid_mask, oi)
            addr = self._addresses(input_ids, shifted_ids, sel, order)
            u = paddle.concat(
                [shard(addr) for shard in self.table_shards], axis=-1
            )  # [B, S, A, d]
            u = (u * gate.unsqueeze(-1).astype(u.dtype)).sum(-2)
            x_proj = self.post_projs[oi](u)
            if beta is not None:
                x_proj = x_proj * beta[..., oi : oi + 1].astype(x_proj.dtype)
            ngram_output = (
                x_proj if ngram_output is None else ngram_output + x_proj
            )

            # Per-order balance loss.
            if (
                self.balance_loss_enabled
                and logits is not None
                and self.balance_loss_coef > 0.0
            ):
                bloss = (
                    self._balance_loss(logits, sel, valid_mask)
                    * self.balance_loss_coef
                )
                total_balance_loss = (
                    bloss
                    if total_balance_loss is None
                    else total_balance_loss + bloss
                )

        # AddAuxiliaryLoss reaches the backward graph without changing the output.
        if total_balance_loss is not None:
            ngram_output = AddAuxiliaryLoss.apply(
                ngram_output, total_balance_loss
            )

        return ngram_output
