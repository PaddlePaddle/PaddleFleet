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

"""HySparse-specific model configuration.

This module intentionally lives **outside** ``transformer_config.py`` so that
HySparse feature configuration can be extended by contributors without touching
the core, heavily-guarded ``TransformerConfig`` class.  Any new HySparse
knobs should be added here rather than in ``transformer_config.py``.

Usage::

    from paddlefleet.transformer import HySparseConfig

    cfg = HySparseConfig(
        num_hidden_layers=24,
        enable_hy_sparse_attention=True,
        hy_sparse_block_size=64,
        hy_sparse_topk=16,
    )
"""

from __future__ import annotations

from dataclasses import dataclass

from .transformer_config import TransformerConfig


@dataclass
class HySparseConfig(TransformerConfig):
    """TransformerConfig extended with HySparse attention knobs.

    All fields added here are **opt-in** (default values keep the model
    identical to a plain ``TransformerConfig``).  Downstream code that must
    support both ``TransformerConfig`` and ``HySparseConfig`` instances should
    use ``getattr(config, 'enable_hy_sparse_attention', False)`` instead of
    direct attribute access.
    """

    # ------------------------------------------------------------------
    # HySparse core switches
    # ------------------------------------------------------------------

    enable_hy_sparse_attention: bool = False
    """Enable the HySparse Attention variant.

    HySparse has the following features: (1) adding a Block Sparse Attention in
    SWA layers. (2) KV sharing between full attention and Block Sparse Attention.
    (3) using MQA instead of MLA.
    """

    hy_sparse_block_size: int = 64
    """HySparse key block size (``block_B``) used by the TileLang block-score /
    block-sparse attention operators. Key columns are grouped into contiguous
    blocks of this size (document-relative) for scoring and sparse selection.

    Default 64 follows the HySparse paper (arXiv:2602.03560, Table 1: "Sparse
    Attn Block Size = 64" for all 7B/80B configurations)."""

    hy_sparse_topk: int = 16
    """Number of key *blocks* selected per query token in the HySparse block-sparse
    branch (the ``topk`` fed to :func:`select_topk_blocks`). The full attention
    layer scores all blocks and the top-``hy_sparse_topk`` (shared across the
    query group by group-wise max) are attended by the SWA layers' block-sparse
    branch.

    Default 16 follows the HySparse paper (arXiv:2602.03560): the paper reports
    selection in *tokens* (k = 1024, "Sparse Attn TopK Tokens = 1024"), which maps
    to k / block_size = 1024 / 64 = 16 blocks. This field counts blocks, so 16 is
    the block-space equivalent of the paper's 1024-token budget."""

    # ------------------------------------------------------------------
    # Backend selectors (for development / debugging)
    # ------------------------------------------------------------------

    hy_sparse_full_attn_use_tilelang: bool = False
    """Route the HySparse **full-attention block-score** branch through the
    independent TileLang operator (``block_score_mha_attn_fwd``) instead of the
    production FA4 fused block-score kernel (``block_score_fa4_attn_fwd``).

    Independent from :attr:`hy_sparse_block_sparse_use_tilelang`: the full-score
    and block-sparse-gather branches each pick their backend separately, so you
    can mix (e.g. TileLang scorer + production DSA gather) to isolate which
    branch an anomaly comes from.

    Set from the training YAML as a top-level key::

        enable_hy_sparse_attention: true
        hy_sparse_full_attn_use_tilelang: true      # default false -> FA4

    The TileLang op is numerically cross-checked against FA4 (bf16-level fwd+bwd
    agreement, exact block_logit and TopK-index bridge). Leave ``False`` for
    production runs (FA4 is faster)."""

    hy_sparse_block_sparse_use_tilelang: bool = False
    """Route the HySparse **block-sparse gather** branch through the independent
    TileLang operator (``block_sparse_mqa_attention_tl``) instead of the
    production cuDNN-DSA gather kernel (``block_sparse_mqa_attention_dsa``).

    Independent from :attr:`hy_sparse_full_attn_use_tilelang` (see there).

    Set from the training YAML as a top-level key::

        enable_hy_sparse_attention: true
        hy_sparse_block_sparse_use_tilelang: true   # default false -> DSA

    The TileLang op is numerically cross-checked against DSA (bf16-level fwd+bwd
    agreement) and needs no head padding / handles any ``kv_lora_rank`` natively.
    Leave ``False`` for production runs (DSA is faster)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self._validate_hysparse()

    def _validate_hysparse(self) -> None:
        """Validate HySparse-specific config constraints."""
        if not self.enable_hy_sparse_attention:
            return

        # HySparse: MTP layers must be FULL attention layers, never SWA.
        # An SWA layer consumes shared_kv (compressed KV latent + block
        # indices) produced by an upstream full layer. The MTP boundary
        # (MultiTokenPredictionLayer._proj_and_transformer_layer) rebuilds a
        # fresh input_dict and does NOT forward shared_key/shared_block_indices
        # from the backbone, so an SWA MTP layer would receive shared_kv=[None,
        # None] and crash at shared_key.squeeze(2) in the block-sparse branch.
        # Fail fast here with a clear message instead.
        if (
            self.window_attn_skip_freq is not None
            and self.num_nextn_predict_layers is not None
            and self.num_nextn_predict_layers > 0
        ):
            mtp_window_flags = self.window_attn_skip_freq[
                self.num_hidden_layers :
            ]
            if any(flag != 0 for flag in mtp_window_flags):
                raise ValueError(
                    "When enable_hy_sparse_attention is True, the MTP portion "
                    f"of window_attn_skip_freq (indices "
                    f"[{self.num_hidden_layers}:], i.e. {mtp_window_flags}) "
                    "must be all 0 (full attention layers). MTP layers cannot "
                    "be sliding-window (SWA) layers because they do not receive "
                    "the shared KV latent from the backbone across the MTP "
                    "boundary. Set the MTP entries to 0."
                )

        # HySparse: the block-score (FA4) and block-sparse (DSA) backends only
        # support hy_sparse_block_size == 64. The FA4 block-score op requires
        # 128 % block_B == 0 and the SM100 DSA block-sparse gather requires
        # block_B == 64 (one block == one TopK tile chunk). Other values either
        # silently mis-bucket keys or fail deep in the CUDA kernels, so reject
        # them up front.
        if self.hy_sparse_block_size != 64:
            raise ValueError(
                "hy_sparse_block_size must be 64 when enable_hy_sparse_attention "
                f"is True (got {self.hy_sparse_block_size}). The FA4 block-score "
                "op requires 128 % block_B == 0 and the SM100 DSA block-sparse "
                "gather requires block_B == 64 (TopK tile alignment)."
            )
