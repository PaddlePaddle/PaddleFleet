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

import sys
import unittest
from pathlib import Path

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddlefleet.transformer.csa_attention import _derive_csa_doc_boundaries
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


def _create_mqa_config(mode="mqa", loss_coeff=0.0, num_hidden_layers=2):
    """dsv4_hybrid config for a ``csa_compress_ratios == -2`` layer.

    ``mode`` is a test-only fixture label: ``"mha"`` leaves
    ``non_absorbed_mqa=False`` (dense path), while both ``"mqa"`` (dense,
    indexer-less) and ``"mqa_dsa"`` (DSA) set ``non_absorbed_mqa=True``. The
    dense/sparse distinction is expressed by whether ``_build_module`` attaches
    an indexer to the sublayers spec, mirroring the production source which
    reads the layer path from the spec, not from a config string.

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
    config.non_absorbed_mqa = mode != "mha"
    # Test-only markers read by ``_build_module``: production always builds the
    # indexer when ``non_absorbed_mqa`` is set, so the indexer-less dense path
    # is reachable only by constructing the layer directly with
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
    """Captures the ``token_indices`` handed to the sparse kernel."""

    def _sparse_attn(self, query, kv, token_indices, sm_scale, d_v):
        _CAPTURED.append(token_indices.numpy().copy())
        return super()._sparse_attn(query, kv, token_indices, sm_scale, d_v)


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


def _doc_meta(row_end, seqlen):
    doc_start, _, is_valid, _, _ = _derive_csa_doc_boundaries(row_end, seqlen)
    return doc_start.numpy(), is_valid.numpy()


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

_MHA_CFG = "ernielite_layer43_mla_hca"
_MQA_CFG = "ernielite_layer43_mla_mqa_hca"
_DSA_CFG = "ernielite_layer43_mla_dsa_hca"
_DENSE_CFG = "ernielite_layer43_non_absorbed_mqa_dense"

# csa_compress_ratios == -2 marks a hybrid-MLA layer; 43 is the MTP layer.
_MINUS2_LAYERS = (8, 17, 26, 34, 42, 43)
_NUM_HIDDEN = 43


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


def _load_provider(name):
    """``(cfg, provider)`` from the on-disk ``model_config.json``."""
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
