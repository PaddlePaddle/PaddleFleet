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

This is a *sibling* of :class:`NgramEmbedding`, selected by the single config
flag ``ngram_moe_enabled``.  When that flag is off nothing here is imported and
the original N-gram path is bit-for-bit unchanged.

Addressing
----------
The baseline addresses a fixed sub-table with a pure hash, ``row = H(g) mod V_i``.
Here the *high* bits of the address are learned and the *low* bits stay hashed::

    addr(t, r) = offset[s] + H_s(g_t) mod V_s,    s = S_r(t)

where ``S_r(t)`` is the r-th sub-table picked by a router that reads a wide causal
context window, and ``H_s`` is the same polynomial rolling hash taken modulo that
sub-table's own modulus.  The router decides *where* to look, the hash decides
*where inside*.

Sizing
------
The only quantity derived from the vocabulary is the row count of one sub-table,
expressed as a multiple of it exactly like the baseline's
``ngram_vocab_size_ratio``; everything else is an absolute number and nothing is
derived from the baseline's K-split fields::

    tables         = ngram_moe_tables_per_order * (N - 1)
    rows           = int(ngram_moe_table_rows_ratio * vocab_size)
    rows(table i)  = rows + 2*i + 1
    total_rows     = tables * rows + tables**2
    params         = total_rows * ngram_moe_table_dim
                     + (N - 1) * ngram_moe_table_dim * hidden_size
    lookups/token  = ngram_moe_active_tables * (N - 1)

Design notes
------------
* Only the **addressing** is conditional; the read-out projection is shared per
  N-gram order.  So there is no grouped matmul, no All2All and no capacity
  factor: the ``active_tables`` reads of a token are exchangeable and one gather
  covers them all.
* Per-table moduli are kept pairwise distinct (``rows + 2*i + 1``), exactly as
  the baseline keeps its K sub-tables distinct.  Without that, all selected
  sub-tables would hash a key to the same offset and the multi-read redundancy
  would collapse.  This is why ``ngram_moe_table_rows_ratio * vocab_size`` is the
  base row count rather than the exact one.
* The ``active_tables`` reads are gate-combined in the d-dimensional table space
  *before* the projection, so the number of projections drops from ``K(N-1)`` to
  ``N-1``.
* The router must see strictly more context than the N-gram itself, otherwise it
  can only rediscover a hash of the same N-gram.  Hence the depthwise causal
  Conv1D of width ``ngram_moe_router_width`` (default 32) >> N.
* The flat table is stored as several slices along the feature dimension, because
  one parameter may not exceed int32 numel (see ``_MAX_PARAM_NUMEL``).

This is the basic version on purpose: no dense warm-up, no late router freeze.
Load balance is *observed* (per-table lookup counts, entropy and dispersion
from the usage monitor) and can optionally be *enforced* via a Switch-Transformer
style auxiliary loss (``ngram_moe_load_balance_coef``).

Per-order routing
------------------
By default (``ngram_moe_shared_router = True``) one router serves all N-gram
orders, so 2-gram and 3-gram share the same sub-table selection.  Set
``ngram_moe_shared_router = False`` to give each order its own router with
independent weights, allowing 2-gram and 3-gram to learn different routing
strategies.

Load-balancing loss
--------------------
When ``ngram_moe_load_balance_coef > 0``, a Switch-Transformer style auxiliary
loss is computed::

    L_aux = α · N · Σ_o Σ_i (f_{o,i} · P_{o,i})

where *f* is the fraction of tokens routed to expert *i* (hard count) and *P*
is the mean routing probability (soft).  When the router is shared, the sum
over *o* collapses to one term.  The loss is returned as the second element of
the ``forward`` tuple; the caller is responsible for adding it to the total
training loss.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor

if TYPE_CHECKING:
    from paddlefleet.transformer.transformer_config import TransformerConfig


# A single parameter must stay below int32 in element count.  Paddle's
# sharding-init parameter broadcast (``sync_params_buffers`` in
# ``paddle/distributed/parallel.py``) coalesces parameters into one buffer and
# then splits them apart with the legacy ``split`` op, whose ``sections``
# attribute is a ``vector<int>``; a parameter with numel >= 2**31 aborts startup
# with ``OutOfRangeError``.  The baseline never hit this because its K sub-tables
# are separate parameters.  The flat routed table is therefore stored as several
# equal slices along the feature dimension -- purely a storage layout, the
# arithmetic is identical to one [total_rows, emb_dim] table.
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
    """Strictly causal, wide-context scorer over one order's sub-tables.

    ``width - 1`` left padding plus a depthwise Conv1D gives every position a
    receptive field of ``width`` tokens ending at itself -- no future leakage.
    Depthwise keeps the cost at ``router_dim * width`` MACs per token.
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


class NgramMoeEmbedding(nn.Layer):
    """N-gram embedding whose sub-table is chosen by a context router.

    Same external contract as :class:`NgramEmbedding` (``.monitor``,
    ``.num_embedders``, returns the [B, S, H] signal to be fused with the word
    embedding), except that ``forward`` also takes the word embedding, which is
    the router's input.

    Args:
        config: TransformerConfig.  Shares ``ngram_emb_neighbor_num`` (N) and
            ``ngram_pad_token_id`` with the baseline; everything else comes from
            the routed variant's own, fully explicit fields:
            - ngram_moe_tables_per_order (int): sub-tables per N-gram order
            - ngram_moe_active_tables (int): sub-tables read per token per order
            - ngram_moe_table_rows_ratio (float): base row count of one
              sub-table, as a multiple of ``vocab_size``
            - ngram_moe_table_dim (int): embedding width of one sub-table
            - ngram_moe_router_dim (int): router bottleneck width
            - ngram_moe_router_width (int): router causal receptive field
        vocab_size: base vocabulary size of the model, used as the radix of the
            polynomial hash and as the unit of the row-count ratio.
    """

    def __init__(self, config: TransformerConfig, vocab_size: int):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.vocab_size = vocab_size
        self.n = config.ngram_emb_neighbor_num
        self.pad_token_id = getattr(config, "ngram_pad_token_id", 0)

        self.num_orders = self.n - 1
        # One "embedder" per order: the active reads are gate-combined before the
        # projection, so an order contributes exactly one summand to the signal
        # and the injection normalizer is 1 + (N - 1).
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

        # Sub-table moduli, laid out order-major so that the flat index
        # ``(order - 2) * tables_per_order + s`` is also the monitor's sub-table
        # index and the row offsets below are the monitor's offsets.
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

        # One flat table over all (order, sub-table) pairs: the routed address is
        # then already the embedding row index *and* the monitor's global index.
        # Materialized as ``num_shards`` slices along the feature dim so that no
        # single parameter exceeds int32 numel (see _MAX_PARAM_NUMEL).
        self.num_shards = _num_table_shards(total, self.emb_dim)
        self.shard_dim = self.emb_dim // self.num_shards
        # fan_in / fan_out pinned to the unsharded shape, so the initial
        # distribution does not depend on how many slices are used.
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
        self.router = NgramTableRouter(
            hidden_size=self.hidden_size,
            num_tables=self.tables_per_order,
            router_dim=int(getattr(config, "ngram_moe_router_dim", 64)),
            width=int(getattr(config, "ngram_moe_router_width", 32)),
        )
        # Per-order independent routers (when shared_router is False, each
        # N-gram order gets its own router with independent weights).
        self.shared_router = getattr(config, "ngram_moe_shared_router", True)
        if not self.shared_router:
            self.routers = nn.LayerList([
                NgramTableRouter(
                    hidden_size=self.hidden_size,
                    num_tables=self.tables_per_order,
                    router_dim=int(getattr(config, "ngram_moe_router_dim", 64)),
                    width=int(getattr(config, "ngram_moe_router_width", 32)),
                )
                for _ in range(self.num_orders)
            ])

        # Load-balancing loss (Switch-Transformer style, optional).
        self.load_balance_coef = float(
            getattr(config, "ngram_moe_load_balance_coef", 0.0)
        )

        self.monitor = None
        if getattr(config, "ngram_monitor_enabled", False):
            from paddlefleet.models.common.embeddings.ngram_monitor import (
                NgramUsageMonitor,
            )

            self.monitor = NgramUsageMonitor(
                config=config,
                table_sizes=table_sizes,
                table_orders=[
                    2 + idx // self.tables_per_order
                    for idx in range(len(table_sizes))
                ],
                vocab_size=vocab_size,
            )
            # Collision analysis pairs a dense per-sub-table bucket tensor with
            # the exact key.  Under routing the modulus varies per position, so
            # that comparison is not defined; the usage block (per-sub-table
            # lookups, entropy, dispersion) carries the load-balance signal.
            self.monitor.collision_enabled = False
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
            shifted = paddle.full([B, k], self.pad_token_id, dtype=input_ids.dtype)
            shifted = paddle.concat([shifted, input_ids[:, : S - k]], axis=1)
            shifted_ids[k + 1] = shifted
        return shifted_ids

    def _route(self, word_emb: Tensor, order_idx: int = 0):
        """Top-k sub-table selection and gates.

        Args:
            word_emb: [B, S, H] router context input.
            order_idx: which order's router to use (ignored when shared).

        Returns:
            sel: [B, S, A] int64 selected sub-table indices.
            gate: [B, S, A] float32 softmax weights over the winners.
            probs: [B, S, tables_per_order] float32 full softmax probabilities
                (needed by the load-balancing loss).
        """
        router = self.router if self.shared_router else self.routers[order_idx]
        scores = router(word_emb).astype("float32")
        top_scores, sel = paddle.topk(scores, self.active_tables, axis=-1)
        gate = F.softmax(top_scores, axis=-1)
        probs = F.softmax(scores, axis=-1)
        return sel.astype("int64"), gate, probs

    def _addresses(
        self, input_ids: Tensor, shifted_ids: dict, sel: Tensor, order: int
    ) -> Tensor:
        """Routed row indices [B, S, A] into the flat table.

        ``offset[s] + (id[t] + sum_j id[t-j] * V^j mod V_s) mod V_s`` with the
        moduli gathered per selected sub-table.
        """
        oi = order - 2
        flat = sel.reshape([-1])
        shape = sel.shape
        mods = paddle.gather(getattr(self, f"table_mods_{oi}"), flat).reshape(shape)
        offs = paddle.gather(getattr(self, f"table_offsets_{oi}"), flat).reshape(
            shape
        )
        pows = paddle.gather(getattr(self, f"table_pow_{oi}"), flat).reshape(
            shape + [oi + 1]
        )
        key = input_ids.cast("int64").unsqueeze(-1)
        for j in range(order - 1):
            key = key + shifted_ids[j + 2].cast("int64").unsqueeze(-1) * pows[..., j]
        return key % mods + offs

    def _load_balance_loss(self, sel_list, probs_list):
        """Switch-Transformer L_aux = α · N · Σ_o Σ_i (f_{o,i} · P_{o,i}).

        Args:
            sel_list: list of [B, S, A] int64 selections, one per router.
            probs_list: list of [B, S, T] float32 full softmax, one per router.

        Returns:
            scalar tensor, or None when coef is zero.
        """
        if self.load_balance_coef <= 0:
            return None
        T = self.tables_per_order
        total_loss = paddle.to_tensor(0.0, dtype="float32")
        for sel, probs in zip(sel_list, probs_list):
            # f_i: fraction of tokens that selected expert i (hard count).
            one_hot = F.one_hot(sel, num_classes=T)  # [B, S, A, T]
            f = one_hot.sum(axis=2).astype("float32")  # [B, S, T]
            f = f.mean(axis=[0, 1])  # [T]
            # P_i: mean routing probability for expert i (soft).
            P = probs.mean(axis=[0, 1])  # [T]
            total_loss = total_loss + self.load_balance_coef * T * (f * P).sum()
        return total_loss

    def forward(self, input_ids: Tensor, word_emb: Tensor):
        """Compute the routed N-gram signal.

        Args:
            input_ids: [B, S] token ids.
            word_emb: [B, S, H] word embeddings; the router's context input.

        Returns:
            (ngram_output, aux_loss) where ngram_output is [B, S, H] and
            aux_loss is a scalar tensor (or None when load balancing is off).
        """
        shifted_ids = self._compute_shifted_ids(input_ids)

        # Route: once when shared, once per order when independent.
        sel_list, probs_list, routes = [], [], []
        if self.shared_router:
            sel, gate, probs = self._route(word_emb)
            routes = [(sel, gate)] * self.num_orders
            sel_list.append(sel)
            probs_list.append(probs)
        else:
            for oi in range(self.num_orders):
                sel_o, gate_o, probs_o = self._route(word_emb, oi)
                routes.append((sel_o, gate_o))
                sel_list.append(sel_o)
                probs_list.append(probs_o)

        ngram_output = None
        for order in range(2, self.n + 1):
            oi = order - 2
            sel_o, gate_o = routes[oi]

            addr = self._addresses(input_ids, shifted_ids, sel_o, order)
            if self.monitor is not None:
                self.monitor.observe_flat(
                    addr, self._valid_mask(input_ids.shape[1], order)
                )
            u = paddle.concat(
                [shard(addr) for shard in self.table_shards], axis=-1
            )  # [B, S, A, d]
            u = (u * gate_o.unsqueeze(-1).astype(u.dtype)).sum(-2)
            x_proj = self.post_projs[oi](u)
            ngram_output = x_proj if ngram_output is None else ngram_output + x_proj

        if self.monitor is not None:
            self.monitor.commit()

        aux_loss = self._load_balance_loss(sel_list, probs_list)
        return ngram_output, aux_loss
