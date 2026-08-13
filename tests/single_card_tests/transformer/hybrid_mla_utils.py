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

"""Shared fixtures for the hybrid-MLA (``csa_compress_ratios == -2``) tests.

Not a test module (no ``test_`` prefix, so pytest does not collect it). Two
groups of fixtures:

- **Direct construction** -- geometry constants, stub sublayers, the config
  factory, the module builder, the document-boundary helpers and the fp32 dense
  reference. Used by ``test_mqa_latent_attention.py``,
  ``test_hybrid_mla_doc_equivalence.py`` and ``test_hybrid_mla_grad_health.py``.
- **Production config chain** -- the on-disk config names, the ``-2`` layer
  index set, the CUDA probe, the process-group stubs and the
  ``ErnieFleetModelConfig -> Ernie5V2Provider -> build_spec_layer`` loader that
  mirrors ``ernie5/pretrain.py``. Used by
  ``test_hybrid_mla_config_pipeline.py`` and
  ``test_hybrid_mla_recompute_mtp_ckpt.py``. All parent-repo imports are local
  to the functions, so importing this module never requires the parent repo.
"""

import contextlib
import sys
import unittest
from pathlib import Path

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddlefleet.transformer.csa_attention import (
    _build_mqa_causal_topk_idxs_from_doc_bounds,
    _derive_csa_doc_boundaries,
)
from paddlefleet.transformer.dsa_attention import (
    DSAIndexer,
    DSAIndexerSublayersSpec,
)
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.mqa_latent_attention import (
    MQALatentAttention,
    MQALatentAttentionSublayersSpec,
)
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import init_method_normal, scaled_init_method_normal

# An installed ``paddlefleet`` wheel shadows this checkout whenever the source
# tree misses from ``sys.path`` -- a mistyped ``PYTHONPATH`` is enough, and the
# symptom is not an ImportError but a stale class: every test that touches a
# method added after the wheel was built dies with a bare
# ``AttributeError: 'RecordingMQA' object has no attribute '_dense_attn'`` out of
# ``Layer.__getattr__``, which says nothing about where the class came from.
# Name the resolved file instead. ``_dense_attn`` is only a marker; any method
# this file's fixtures rely on would do.
if not hasattr(MQALatentAttention, "_dense_attn"):
    raise ImportError(
        "MQALatentAttention was imported from "
        f"{sys.modules[MQALatentAttention.__module__].__file__}, which predates "
        "the code these fixtures test (no _dense_attn). Put the PaddleFleet "
        "source tree ahead of site-packages on PYTHONPATH -- e.g. "
        "PYTHONPATH=<checkout>:<checkout>/src."
    )

# ---------------------------------------------------------------------------
# Geometry. DK/DV are hard requirements of the FlashMLA sparse kernel
# (d_qk in {512, 576}, d_v == 512). K_CHANNELS is the *MHA* q_head_dim
# (qk_nope 192 + qk_rope 64): absorption preserves scores exactly, so the MHA
# softmax scale must be kept instead of the 576-wide latent one.
# INDEX_TOPK must stay a multiple of 128 (``indexer_backward_sm100`` asserts
# ``topk % block_I == 0``) and INDEX_HEADS >= 64 (``assert heads >= 64``).
# ---------------------------------------------------------------------------
H = 8
DK = 576
DV = 512
V_HEAD_DIM = 64
K_CHANNELS = 256
WINDOW = 128
INDEX_TOPK = 128
INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
HIDDEN = 256
Q_LORA = 128


# ---------------------------------------------------------------------------
# Stub layers (same pattern as test_dsa_attention.py)
# ---------------------------------------------------------------------------
class BiasedLinear(paddle.nn.Layer):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.linear = paddle.nn.Linear(in_features, out_features)

    def forward(self, x):
        if x.dtype != self.linear.weight.dtype:
            x = x.cast(self.linear.weight.dtype)
        return self.linear(x), self.linear.bias


class LayerNormStub(paddle.nn.Layer):
    """LayerNorm stub accepting either ``hidden_size``/``eps`` naming."""

    def __init__(
        self,
        hidden_size=None,
        eps=None,
        normalized_shape=None,
        epsilon=None,
        **kwargs,
    ):
        super().__init__()
        size = hidden_size if hidden_size is not None else normalized_shape
        self.eps = (
            eps
            if eps is not None
            else (epsilon if epsilon is not None else 1e-5)
        )
        self.weight = paddle.nn.Parameter(paddle.ones([size]))
        self.bias = paddle.nn.Parameter(paddle.zeros([size]))

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = x.var(axis=-1, keepdim=True, unbiased=False)
        x = (x - mean) / paddle.sqrt(var + self.eps)
        return x * self.weight + self.bias


def _dsa_kernels_available():
    if not paddle.is_compiled_with_cuda():
        return False
    try:
        from paddlefleet.cudnn_ops.block_sparse_mqa_dsa import is_dsa_available

        return bool(is_dsa_available())
    except Exception:
        return False


_GPU = unittest.skipUnless(
    _dsa_kernels_available(),
    "requires SM100+ FlashMLA sparse fwd + cuDNN DSA bwd kernels",
)


# Fixture label -> production ``TransformerConfig.hybrid_mla_attention`` value.
# The label ``"mqa"`` predates the enum and means "latent MQA, no indexer",
# which the enum spells ``"mqa_full_causal"``.
_HYBRID_MLA_ATTENTION = {
    "mha": "mha",
    "mqa": "mqa_full_causal",
    "mqa_dsa": "mqa_dsa",
}


def _create_mqa_config(mode="mqa", loss_coeff=0.0, num_hidden_layers=2):
    """dsv4_hybrid config for a ``csa_compress_ratios == -2`` layer.

    ``mode`` is a test-only fixture label mapped onto the production
    ``hybrid_mla_attention`` enum by ``_HYBRID_MLA_ATTENTION``: ``"mha"`` keeps
    the dense per-head path, ``"mqa"`` selects ``"mqa_full_causal"`` (latent MQA
    over the full per-document causal set, no indexer) and ``"mqa_dsa"`` selects
    ``"mqa_dsa"`` (latent MQA + DSA indexer). Whether an indexer actually exists
    is expressed by ``_build_module`` attaching one to the sublayers spec,
    mirroring the production source which reads the layer path from the spec,
    not from a config string.

    Attributes are assigned after construction so that ``__post_init__``
    validation (exercised by the production model config, not by this unit) is
    bypassed -- same convention as ``test_dsa_attention.py``.
    """
    config = TransformerConfig(
        num_hidden_layers=num_hidden_layers,
        hidden_size=HIDDEN,
        num_attention_heads=H,
    )
    config.num_key_value_heads = H
    config.head_dim = K_CHANNELS
    config.experimental_attention_variant = "dsv4_hybrid"
    config.hybrid_mla_attention = _HYBRID_MLA_ATTENTION[mode]
    # Test-only markers read by ``_build_module``: production builds the indexer
    # exactly for ``hybrid_mla_attention="mqa_dsa"``, so the indexer-less latent
    # MQA path is reachable only by constructing the layer directly with
    # ``MQALatentAttentionSublayersSpec(indexer=None)``.
    config.test_attn_mode = mode
    config._build_dsa_indexer = mode == "mqa_dsa"
    config.hybrid_mla_q_lora_rank = Q_LORA
    config.hybrid_mla_kv_lora_rank = DV
    config.hybrid_mla_qk_nope_head_dim = 192
    config.hybrid_mla_qk_rope_head_dim = 64
    config.hybrid_mla_v_head_dim = V_HEAD_DIM
    config.hybrid_mla_num_attention_heads = H
    config.hybrid_mla_num_key_value_heads = H
    # The indexer dims are model-wide (HF json aliases index_n_heads /
    # index_head_dim / index_topk), shared with the CSA layers.
    config.dsa_index_n_heads = INDEX_HEADS
    config.dsa_index_head_dim = INDEX_HEAD_DIM
    config.dsa_index_topk = INDEX_TOPK
    config.csa_window_size = WINDOW
    config.dsa_indexer_loss_coeff = loss_coeff
    config.dsa_indexer_use_sparse_loss = True
    config.dsa_indexer_rotary_interleaved = False
    # The -2 layers are uncompressed, hence plain RoPE (base 10000); YaRN only
    # applies to the compressed HCA layers.
    config.rope_type = "rope"
    config.rope_theta = 10000.0
    config.rotary_interleaved = False
    config.rotary_percent = 1.0
    config.apply_rope_fusion = False
    config.num_nextn_predict_layers = 0
    config.mtp_num_layers = 0
    config.init_method = init_method_normal(0.02)
    config.output_layer_init_method = scaled_init_method_normal(0.02, 1, 2.0)
    config.rms_norm_eps = 1e-5
    config.context_parallel_size = 1
    config.sequence_parallel = False
    return config


_CAPTURED = []


class RecordingMQA(MQALatentAttention):
    """Captures, as a column table, what each backend was asked to attend over.

    ``_sparse_attn`` (phase 3) receives that table literally. The full-causal
    phases hand FA4 an ``O(s)`` flashmask row bound instead and never materialise
    anything, so ``_dense_attn`` records the column set that bound *implies*,
    derived from the very tensor the kernel is about to get. Both land in
    ``_CAPTURED`` in call order, so a test can ask "what did attention see"
    without caring which backend served it, and a recompute test still counts two
    entries per recomputed step.

    Skipped when ``cp_size > 1``: the bound is localised inside ``_dense_attn``
    and its values are global row ids, so no per-rank ``[b, s, s]`` table is the
    honest reconstruction. The CP suites assert on outputs and gradients instead.
    """

    def _sparse_attn(
        self, query, kv, token_indices, sm_scale, d_v, indexer_topk=0
    ):
        _CAPTURED.append(token_indices.numpy().copy())
        return super()._sparse_attn(
            query, kv, token_indices, sm_scale, d_v, indexer_topk
        )

    def _dense_attn(self, query, kv, row_end, kv_lora_rank):
        if self.cp_size == 1:
            _CAPTURED.append(_row_end_column_table(row_end).numpy())
        return super()._dense_attn(query, kv, row_end, kv_lora_rank)


def _row_end_column_table(row_end):
    """``[b, s, s]`` int32 column set implied by a flashmask row bound.

    The dense backend's ``O(s)`` mask, written out. Same builder the CP suites
    use (``_full_causal_indices``), fed from the document boundaries the
    production deriver reads out of the bound itself, so this is a decoding of
    what the kernel was told rather than a second opinion about it.
    """
    seqlen = int(row_end.shape[-2])
    with paddle.no_grad():
        doc_start, _, is_valid, _, _ = _derive_csa_doc_boundaries(
            row_end, seqlen
        )
    return _full_causal_indices(
        int(row_end.shape[0]), seqlen, doc_start, is_valid
    )


def _build_module(
    config,
    layer_number=1,
    bf16=False,
    sink=None,
    is_mtp=False,
    pg_collection=None,
):
    """Build a ``RecordingMQA``.

    ``pg_collection`` must be passed explicitly by the context-parallel tests
    (``tests/multi_card_tests/transformer/test_mqa_dsa_cp.py``): left ``None``
    the layer falls back to ``ProcessGroupCollection.use_mpu_process_groups()``,
    which inside a ``fleet.init``-ed process would hand the CP=1 reference the
    real CP group.
    """
    indexer = None
    if getattr(config, "_build_dsa_indexer", False):
        indexer = LayerSpec(
            layer=DSAIndexer,
            sublayers_spec=DSAIndexerSublayersSpec(
                linear_wq_b=BiasedLinear,
                linear_wk=BiasedLinear,
                k_norm=LayerNormStub,
                linear_weights_proj=BiasedLinear,
            ),
            extra_kwargs={"is_hybrid_mla_indexer": True},
        )
    module = RecordingMQA(
        config=config,
        sublayers_spec=MQALatentAttentionSublayersSpec(indexer=indexer),
        layer_number=layer_number,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        k_channels=K_CHANNELS,
        is_mtp_layer=is_mtp,
        pg_collection=pg_collection,
    )
    if bf16:
        # ``rotate_activation`` asserts bf16 inputs, so the indexer projections
        # must hold bf16 weights.
        module.to(dtype="bfloat16")
    if sink is not None:
        # In production ``MQALatentAttention.__init__`` builds this parameter
        # via the shared ``build_softmax_offset`` helper (name
        # ``core_attention.softmax_offset``, identical to the dense
        # ``DotProductAttention`` phase, so an MHA checkpoint stays loadable).
        # This unit uses a default config with no sink configured, so the
        # helper returns ``None``; inject the sink here instead. Created *after*
        # ``to(dtype=...)`` and in the module dtype: production uses
        # ``params_dtype`` (bf16), which is what the FA4 cute kernel of the
        # dense path requires, and the DSA path returns the sink gradient in
        # the parameter's own dtype.
        module.softmax_offset = module.create_parameter(
            shape=[H],
            dtype="bfloat16" if bf16 else "float32",
            default_initializer=paddle.nn.initializer.Assign(
                np.asarray(sink, dtype="float32")
            ),
        )
    return module


def _row_end(doc_lens, seqlen):
    """``[1, 1, s, 1]`` int32 exclusive per-token document end row.

    The trailing gap (``sum(doc_lens) < seqlen``) becomes one final document
    ending at ``seqlen``, so every row is valid.
    """
    out = np.empty([seqlen], dtype="int32")
    pos = 0
    for length in doc_lens:
        end = pos + length
        out[pos : min(end, seqlen)] = end
        pos = end
        if pos >= seqlen:
            break
    if pos < seqlen:
        out[pos:] = seqlen
    return paddle.to_tensor(out).reshape([1, 1, seqlen, 1])


def _pad_row_end(doc_lens, seqlen):
    """``[1, 1, s, 1]`` int32 ``row_end`` that produces real pad rows.

    ``_row_end`` fills the trailing gap with ``seqlen``, which
    ``_derive_csa_doc_boundaries`` reads as one more document, so every row comes
    back ``is_valid``. Repeating the *last document's* end instead leaves
    ``doc_len_per_pos`` short of ``pos_in_doc`` for the tail, which is the
    ``is_valid == False`` state a packed batch's padding actually takes.
    """
    out = np.empty([seqlen], dtype="int32")
    pos, end = 0, 0
    for length in doc_lens:
        end = pos + length
        out[pos : min(end, seqlen)] = end
        pos = end
        if pos >= seqlen:
            break
    if pos < seqlen:
        out[pos:] = end
    return paddle.to_tensor(out).reshape([1, 1, seqlen, 1])


def _doc_meta(row_end, seqlen):
    doc_start, _, is_valid, _, _ = _derive_csa_doc_boundaries(row_end, seqlen)
    return doc_start.numpy(), is_valid.numpy()


def _full_causal_indices(
    b, s_global, doc_start, is_valid, position_offset=0, s_local=None
):
    """Per-document full-causal ``[b, s_local, s_global]`` int32 (``-1`` pad).

    The column set the two full-causal phases attend over, written out as an
    explicit table. Production never materialises it -- ``_dense_attn`` hands the
    same set to FA4 as an ``O(s)`` row bound -- so this lives on the test side,
    with two uses: as an independent derivation of what the dense mask must be,
    and as the input for feeding the sparse kernel directly (``_sparse_attn``)
    when a test wants the two backends compared.

    Built over the global sequence and row-sliced afterwards, so the column
    values stay global token ids -- which is what the all-gathered KV of a CP
    rank is indexed in.
    """
    with paddle.no_grad():
        indices, _ = _build_mqa_causal_topk_idxs_from_doc_bounds(
            b, s_global, doc_start, is_valid
        )
        if s_local is not None and s_local != s_global:
            indices = indices[:, position_offset : position_offset + s_local]
        indices = indices.contiguous()
    indices.stop_gradient = True
    return indices


def _make_inputs(seqlen, seed=0, with_hidden=False):
    """``(query, key, w_v)``, plus ``(x, q_lora)`` when ``with_hidden``."""
    paddle.seed(seed)
    query = (paddle.randn([1, seqlen, H, DK]) * 0.5).cast("bfloat16")
    key = (paddle.randn([1, seqlen, 1, DK]) * 0.5).cast("bfloat16")
    w_v = (paddle.randn([DV, H, V_HEAD_DIM]) * 0.05).cast("bfloat16")
    if not with_hidden:
        return query, key, w_v
    x = (paddle.randn([1, seqlen, HIDDEN]) * 0.5).cast("bfloat16")
    qr = (paddle.randn([1, seqlen, Q_LORA]) * 0.5).cast("bfloat16")
    return query, key, w_v, x, qr


def _rel(actual, expected):
    a = actual.cast("float32")
    e = expected.cast("float32")
    return float((a - e).norm() / e.norm().clip(min=1e-12))


# One bf16 ULP, relative: bf16 carries 8 significant bits, so the smallest
# representable step at magnitude ``m`` is ``m * 2**-8``.
_BF16_ULP = 2.0**-8

# How many of those steps a re-reduction is allowed to move the result by. The
# measured worst case over every layout these suites use is 1.63 ULPs of the
# compared slice's own maximum (layout ``[127]`` at ``s=256``, second document,
# with a sink: 3.906e-03 at slice scale 0.613); 4 leaves ~2.5x headroom without
# coming near the ~240 ULPs that losing document isolation costs on the same
# tensors.
_ULP_BUDGET = 4


def _assert_agrees_to_bf16_ulps(test, actual, expected, msg):
    """Two runs of the same mathematical softmax agree to a few bf16 ULPs.

    Bit equality is the wrong bar whenever the two sides do not reduce in the
    same order, and on these paths there are two ways that happens:

    * *repacking* -- FA4 derives its tile scheduling, hence its accumulation
      order, from the flashmask row bounds, so a packed batch and a
      single-document run of the same rows are not bitwise equal even though
      they sum over the same columns;
    * *changing backend* -- the phase-3 sparse kernel reduces each query row over
      an explicit column list in a fixed order, dense FA4 does not, so a claim
      that the two agree on a layout where their column sets coincide is also
      only an ULP claim.

    A bit-equality assertion written against the sparse backend therefore does
    not transfer. The tolerance is ``_ULP_BUDGET`` ULPs of the compared slice's
    own scale; see that constant for the measurement behind it. Losing document
    isolation -- the property most callers are actually testing -- costs the
    output scale itself, some 240 ULPs, so the assertion still fails on a real
    leak. ``_assert_isolation_is_observable`` pins that separation down where the
    non-isolated variant is cheap to build.

    Both arguments are fp32 numpy (the bf16 widening is exact).
    """
    worst = float(np.abs(actual - expected).max())
    bound = _ulp_bound(actual)
    test.assertLessEqual(
        worst,
        bound,
        f"{msg}: maxabs {worst:.3e} > {_ULP_BUDGET} bf16 ULP {bound:.3e}",
    )
    return worst


def _ulp_bound(reference):
    return _ULP_BUDGET * _BF16_ULP * max(float(np.abs(reference).max()), 1e-6)


def _assert_isolation_is_observable(test, worst, leaked, packed):
    """The ULP bound of ``_assert_agrees_to_bf16_ulps`` is not vacuous.

    ``leaked`` is the same comparison run against a deliberately non-isolated
    forward (one document spanning the whole sequence). It has to miss by at
    least an order of magnitude more than the tolerance the isolated case was
    allowed, or the tolerance would be hiding cross-document attention.

    Only meaningful for a document that does not start at row 0: the *first*
    document's causal set is the same with and without isolation, so there the
    control is identically zero and proves nothing either way.
    """
    bound = _ulp_bound(packed)
    test.assertGreater(
        leaked,
        10.0 * bound,
        f"a non-isolated forward only differs by {leaked:.3e}, so the "
        f"{bound:.3e} tolerance on {worst:.3e} proves nothing",
    )


def _dense_reference(query, key, w_v, row_end, scale, sink=None):
    """Per-document full-causal attention on the latent, computed in fp32.

    This *is* the mathematical ``mha`` path (dense MLA on the absorbed latent):
    absorption is exactly score preserving, so ``mha``/``mqa``/``mqa_dsa`` must
    all reduce to this reference. ``sink`` is a ``[H]`` per-head logit appended
    as one extra value-less softmax column (drains probability mass only), i.e.
    the exact semantics of ``attn_sink`` in the block-sparse kernel.
    """
    seqlen = int(query.shape[1])
    doc_start, is_valid = _doc_meta(row_end, seqlen)
    pos = np.arange(seqlen)
    allowed = (
        (pos[None, :] <= pos[:, None])
        & (pos[None, :] >= doc_start[:, None])
        & is_valid[:, None]
    )
    q = query[0].cast("float32")
    k = key.squeeze(2)[0].cast("float32")
    scores = paddle.einsum("shd,td->sht", q, k) * scale
    keep = paddle.to_tensor(allowed).unsqueeze(1)
    scores = paddle.where(keep, scores, paddle.full_like(scores, -1e30))
    if sink is None:
        probs = F.softmax(scores, axis=-1)
    else:
        sink_col = paddle.to_tensor(np.asarray(sink, dtype="float32")).reshape(
            [1, H, 1]
        )
        sink_col = paddle.expand(sink_col, [seqlen, H, 1])
        probs = F.softmax(paddle.concat([scores, sink_col], axis=-1), axis=-1)
        probs = probs[:, :, :seqlen]
    ctx = paddle.einsum("sht,tl->shl", probs, k[:, :DV])
    out = paddle.einsum("shl,lhv->shv", ctx, w_v.cast("float32"))
    row_ok = paddle.to_tensor(is_valid).cast("float32").reshape([seqlen, 1, 1])
    return (out * row_ok).reshape([1, seqlen, H * V_HEAD_DIM])


def _check_index_invariants(test, indices, row_end, seqlen, expect_full=False):
    """Assert the per-row selected column set is sound.

    Invariants: no duplicate column (a duplicate would double-count in the
    softmax), every column in range, causal and inside the query's own
    document, the forced ``WINDOW`` columns always present and clipped at
    ``doc_start``, and pad rows select nothing.
    """
    doc_start, is_valid = _doc_meta(row_end, seqlen)
    for q in range(seqlen):
        cols = indices[0, q]
        cols = cols[cols >= 0].tolist()
        test.assertEqual(
            len(cols), len(set(cols)), f"row {q}: duplicate column"
        )
        test.assertTrue(
            all(0 <= c < seqlen for c in cols), f"row {q}: out-of-range column"
        )
        if not is_valid[q]:
            test.assertEqual(cols, [], f"pad row {q} must select nothing")
            continue
        start = int(doc_start[q])
        test.assertTrue(
            all(start <= c <= q for c in cols),
            f"row {q}: non-causal or cross-document column",
        )
        window = set(range(max(start, q - WINDOW + 1), q + 1))
        test.assertEqual(
            window - set(cols), set(), f"row {q}: lost forced-window columns"
        )
        if expect_full:
            test.assertEqual(
                set(cols),
                set(range(start, q + 1)),
                f"row {q}: not the full causal set",
            )


# ---------------------------------------------------------------------------
# Production config chain, mirroring ``ernie5/pretrain.py``
# (ErnieFleetModelConfig -> Ernie5V2Provider -> apply_ernie_config_overrides
# -> LayerSpec dispatch -> build_spec_layer). Every parent-repo import below
# is function-local on purpose: importing this module must never require the
# parent repo, since the direct-construction fixtures above do not need it.
# ---------------------------------------------------------------------------
_CONFIG_SUBDIR = Path("model_config_separated") / "conf" / "fleet_align"


def _find_repo_root():
    """Walk up to the erniebot checkout that owns the production configs.

    A fixed ancestor index would raise ``IndexError`` at import time wherever
    PaddleFleet is checked out standalone (upstream CI does exactly that), which
    kills collection for every module importing this one. On a miss, return a
    path that does not exist so the config-driven suites skip instead.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / _CONFIG_SUBDIR).is_dir():
            return parent
    return here.parent / "_no_erniebot_parent_repo"


_REPO_ROOT = _find_repo_root()
_CONFIG_DIR = _REPO_ROOT / _CONFIG_SUBDIR
_PARENT_REPO_AVAILABLE = _CONFIG_DIR.is_dir()

# ---------------------------------------------------------------------------
# The five production ``layer43`` configs. All are phase variants of the same
# 44-layer skeleton and ``_MHA_CFG`` (phase 1, the live reference run) is the
# baseline the others are diffed against.
#
# NOTE on naming: the enum rename (2026-08-07) fixed the vocabulary. An MLA
# layer (``csa_compress_ratios == -2``) running ``MQALatentAttention`` is
# **latent MQA**; a CSA-family layer with ``csa_compress_ratios == -1`` is
# **CSA full-causal MQA** and is a different class entirely
# (``DSv4HybridSelfAttention`` + ``CompressedSparseAttention``). ``_CSA_MQA_CFG``
# is the latter and therefore carries NO ``hybrid_mla_*`` field at all -- it must
# never be used as a stand-in for a latent-MQA config.
# ---------------------------------------------------------------------------
# Phase 1: dense per-head MHA on the -2 layers (``hybrid_mla_attention`` unset
# -> defaults to ``"mha"``). Carries ``add_full_attention_sink_bias``.
_MHA_CFG = "ernielite_layer43_mla_hca"
# ``hybrid_mla_attention="mqa_full_causal"``: latent MQA with no indexer
# (equivalence experiment, not a production phase). The on-disk directory still
# carries the pre-rename ``non_absorbed`` name; only the config *field* was
# renamed, not the parent repo's config directories.
_FULL_CAUSAL_CFG = "ernielite_layer43_non_absorbed_mqa_dense"
# Phase 2 (DSA warmup): ``hybrid_mla_attention="mqa_dsa"`` +
# ``dsa_indexer_use_sparse_loss=false`` + YAML ``train_indexer_only=true``.
_DSA_CFG = "ernielite_layer43_non_absorbed_mqa_hca_dsa"
# Phase 3: ``hybrid_mla_attention="mqa_dsa"`` +
# ``dsa_indexer_use_sparse_loss=true``, backbone unfrozen.
_DSA_SPARSE_LOSS_CFG = "ernielite_layer43_non_absorbed_mqa_hca_dsa_sparse_loss"
# CSA full-causal MQA (``csa_compress_ratios == -1``). NOT a hybrid-MLA config:
# no ``-2`` layer, no ``hybrid_mla_*`` / ``rope_type`` / ``use_vha_attention`` /
# ``add_full_attention_sink_bias`` key.
_CSA_MQA_CFG = "ernielite_layer43_mqa_hca"

# The two production configs that build a latent MQA + DSA indexer. Tests that
# used to pair "mqa" with "dsa" must iterate this instead: since the enum
# rename there is no separate indexer-less ``mqa`` config, the second
# ``"mqa_dsa"`` config is the sparse-loss phase.
_MQA_DSA_CFGS = (_DSA_CFG, _DSA_SPARSE_LOSS_CFG)
# Every config that has ``-2`` layers, i.e. the hybrid-MLA phase chain whose
# parameter sets must stay checkpoint compatible.
_HYBRID_MLA_CFGS = (_MHA_CFG, _FULL_CAUSAL_CFG, *_MQA_DSA_CFGS)
# All five, for the config-drift sentinels.
_LAYER43_CFGS = (_CSA_MQA_CFG, *_HYBRID_MLA_CFGS)

# csa_compress_ratios == -2 marks a hybrid-MLA layer; 43 is the MTP layer.
# This is the parent repo's online layout (7 MLA layers: 6 in the backbone plus
# the MTP one). All five layer43 configs share the *indices*, so a phase-1
# checkpoint loads into the later phases; they diverged for a while and were
# realigned. ``_CSA_MQA_CFG`` puts ``-1`` at exactly these indices.
_MINUS2_LAYERS = (7, 14, 21, 28, 35, 42, 43)
_NUM_HIDDEN = 43

# The training YAMLs are named after the model_config directory with a
# ``pretrain_`` infix. Only phase 1 is a live online run and stays in
# ``conf/online``; the MQA phases are experiments and moved to
# ``conf/experiment/ernielite_layer43_mqa`` (2026-08-08).
_ONLINE_YAML_DIR = _REPO_ROOT / "conf" / "online"
_EXPERIMENT_YAML_DIR = (
    _REPO_ROOT / "conf" / "experiment" / "ernielite_layer43_mqa"
)


def _yaml_path(name):
    """Training YAML for a ``model_config_separated`` directory.

    Raises instead of returning a non-existent path: a silently missing YAML
    turns every config-drift assertion into a skip, which is exactly how an
    earlier rename went unnoticed.
    """
    filename = f"{name.replace('layer43_', 'layer43_pretrain_')}.yaml"
    for directory in (_ONLINE_YAML_DIR, _EXPERIMENT_YAML_DIR):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no training YAML named {filename} under {_ONLINE_YAML_DIR} or "
        f"{_EXPERIMENT_YAML_DIR}; if it moved again, add the new directory "
        "here rather than letting the config tests skip"
    )


def _load_yaml(name):
    """Parsed production training YAML (key -> value, order insensitive)."""
    import yaml

    with open(_yaml_path(name)) as f:
        return yaml.safe_load(f)


def _add_repo_root_to_sys_path():
    """The pytest invocation only puts ``PaddleFleet/src`` and
    ``PaddleFormers`` on PYTHONPATH; add the repo root (for ``fleet_model``)
    and ``ernie5`` (for ``src.ernie_core_compat``).
    """
    for p in (str(_REPO_ROOT), str(_REPO_ROOT / "ernie5")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _try_use_cuda_device():
    if not paddle.is_compiled_with_cuda():
        return False
    try:
        paddle.set_device("gpu:0")
        place = str(paddle.empty([1]).place).lower()
    except Exception:
        return False
    return paddle.get_device().startswith("gpu") and (
        "gpu" in place or "cuda" in place
    )


def _stub_device_capability():
    """Let the pure-config (non-kernel) tests import paddlefleet and build the
    provider on a box without a usable GPU.
    """
    paddle.cuda.get_device_capability = lambda device=None: (0, 0)
    paddle.device.cuda.get_device_capability = lambda device=None: (0, 0)


class _FakeGroup:
    def __init__(self, nranks=1):
        self.nranks = nranks
        self.world_size = nranks
        self.ranks = list(range(nranks))
        self.rank = 0


class _FakePGCollection:
    def __init__(self):
        self.tp = _FakeGroup(1)
        self.cp = _FakeGroup(1)


def _load_json(name):
    """Raw on-disk ``model_config.json`` (no provider normalisation)."""
    import json

    with open(_CONFIG_DIR / name / "model_config.json") as f:
        return json.load(f)


@contextlib.contextmanager
def _flash_attn_version(value):
    """Temporarily pin the process-global ``FLAGS_flash_attn_version``.

    Production derives it from the compute capability in
    ``TrainingArguments.__post_init__`` (PaddleFormers
    ``trainer/training_args.py:1764-1780``): SM100 -> 4, SM90 -> 3, else 2. None
    of the five layer43 YAMLs pins ``fa_version``, so on the SM100 boxes these
    runs target the effective value is 4.

    A bare pytest process never constructs ``TrainingArguments``, so the flag
    keeps the image default 2, and the dense-MHA sink guard
    (``multi_latent_attention.py:561-581``) then refuses to build the layer.
    Reproducing the production flag value is the faithful fix; weakening the
    guard would not be.
    """
    key = "FLAGS_flash_attn_version"
    previous = paddle.get_flags([key])[key]
    paddle.set_flags({key: value})
    try:
        yield
    finally:
        paddle.set_flags({key: previous})


def _fa4_can_serve():
    """Whether pinning ``FLAGS_flash_attn_version=4`` can actually be served.

    ``flash_mask_facade.get_fa_version`` answers from the flag and the head dims
    alone -- it never checks that a FA4 backend exists. With no cute kernel the
    facade's ``_flashmask_attention`` is ``paddle.nn.functional``'s, which knows
    only 2 and 3 and raises ``ValueError: Invalid flash attention version: 4``
    (``paddle/nn/functional/flash_attention.py:2179``) for *every* head-dim pair
    the facade whitelists, ``(256, 256)`` plain MHA included. The upstream CI
    images are cc 9.0 without the kernel, so pinning 4 unconditionally would
    break the ``mha`` cases, which are not ``_GPU``-gated and used to run there
    on the image default 2.
    """
    try:
        from paddlefleet_ops import is_flash_mask_available
    except ImportError:  # pragma: no cover - packaged build always has it
        return False
    return bool(is_flash_mask_available()) and _production_fa_version() == 4


@contextlib.contextmanager
def _fa4_pin():
    """``_flash_attn_version(4)`` where FA4 can serve, a no-op where it cannot.

    The full-causal phases have exactly one backend, and
    ``MQALatentAttention._assert_dense_fa4`` raises when the process flags do not
    resolve to FA4, so a module running such a forward has to reproduce the
    production flag value. Those cases are all ``_GPU``-gated to the SM100 boxes,
    where ``_fa4_can_serve()`` is True -- and the same modules also hold ``mha``
    cases that do run without FA4, which is why the pin has to stand down rather
    than force a value the box cannot honour.
    """
    if not _fa4_can_serve():
        yield
        return
    with _flash_attn_version(4):
        yield


def _fa4_module_hooks():
    """``(setUpModule, tearDownModule)`` wrapping the module in ``_fa4_pin()``.

    Doing it once per module beats repeating the pin at every call site, and the
    restore keeps it from leaking into modules that assert on the refusal.
    """
    pin = _fa4_pin()

    def setUpModule():
        pin.__enter__()

    def tearDownModule():
        pin.__exit__(None, None, None)

    return setUpModule, tearDownModule


def _production_fa_version():
    """The ``fa_version`` the production trainer would pick on this box."""
    major, minor = paddle.device.cuda.get_device_capability()
    if major == 10:
        return 4
    if (major, minor) == (9, 0):
        return 3
    return 2


@contextlib.contextmanager
def _cudnn_deterministic(value):
    """Temporarily pin the process-global ``FLAGS_cudnn_deterministic``.

    Set by the accuracy-diff harnesses (``script/test_aadiff/check_aadiff.sh``,
    ``script/run_compare_fleet_ec.sh``), and read by
    ``paddlefleet_ops.flash_mask_facade.get_fa_version``, so it takes part in
    backend selection rather than only in kernel behaviour.
    """
    key = "FLAGS_cudnn_deterministic"
    previous = paddle.get_flags([key])[key]
    paddle.set_flags({key: value})
    try:
        yield
    finally:
        paddle.set_flags({key: previous})


def _load_provider(name):
    """``(cfg, provider)`` from the on-disk ``model_config.json``.

    NOTE: only the JSON is read. Provider fields that live in the training YAML
    (``csa_sparse_attn_backend``, ``csa_indexer_backend``,
    ``indexer_init_from_scratch``, ``train_indexer_only``, ...) are absent, so a
    test that needs one has to set it explicitly -- see ``_load_yaml``.
    """
    _add_repo_root_to_sys_path()
    from fleet_model.ernie5_v2.modeling import (
        Ernie5V2Provider,
        apply_ernie_config_overrides,
    )
    from src.ernie_core_compat.configuration import ErnieFleetModelConfig

    cfg = ErnieFleetModelConfig.from_pretrained(
        str(_CONFIG_DIR / name), _configuration_file="model_config.json"
    )
    provider = Ernie5V2Provider.from_config(cfg)
    apply_ernie_config_overrides(provider, cfg)
    return cfg, provider


def _build_real_attn(provider, layer_number, seed=42):
    """Instantiate the real self-attention module for one layer index."""
    from paddle.distributed.fleet.meta_parallel import build_spec_layer

    from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from paddlefleet.tensor_parallel.random import (
        model_parallel_cuda_manual_seed,
    )

    model_parallel_cuda_manual_seed(seed)
    spec = get_gpt_layer_local_spec(
        config=provider,
        normalization=provider.normalization,
        layer_number=layer_number,
    ).sublayers_spec.self_attn
    return build_spec_layer(
        spec,
        config=provider,
        layer_number=layer_number,
        pg_collection=_FakePGCollection(),
    )
