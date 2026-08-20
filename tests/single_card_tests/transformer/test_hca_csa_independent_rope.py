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

"""Unit tests for per-attention-type (HCA / CSA) RoPE variant configuration.

Covers the incremental behavior that lets HCA layers
(``csa_compress_ratios == 128``) and CSA layers (``2 <= ratio < 128``)
independently choose the RoPE variant (plain RoPE vs YaRN) through
``hca_rope_type`` / ``csa_rope_type``, while staying fully backward compatible
when the fields are unset. The positional-embedding width is not part of this
feature: it always remains the global ``qk_pos_emb_head_dim``.

``TestLegacyPlainRopeEquivalence`` /
``TestLegacyPlainRopeForwardEquivalence`` pin the central guarantee: setting
``hca_rope_type="rope"`` (resp. ``csa_rope_type="rope"``) reproduces bit for
bit what the pre-feature code produced when its local ``use_compressed_yarn``
flag was forced to ``False`` -- both for the constructed rotary module and for
a real forward/backward pass.
"""

import unittest
from unittest import mock

import numpy as np
import paddle
from paddle.distributed.fleet.meta_parallel import build_spec_layer

from paddlefleet.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)
from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding,
)
from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
)
from paddlefleet.transformer import dsv4_hybrid_attention
from paddlefleet.transformer.transformer_config import TransformerConfig

_SEED = 42
_ROTARY_PERCENT = 1.0
# csa_compress_rotary_base used by _make_config: compressed layers (HCA/CSA)
# override the global rotary_base with this value.
_COMPRESSED_ROPE_BASE = 160000.0
_GLOBAL_POS_DIM = 16


def _try_use_cuda_device():
    """Return True only when a real CUDA device is usable for kernels."""
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


_REQUIRES_CUDA = unittest.skipUnless(
    _try_use_cuda_device(),
    "requires a usable CUDA device to run the DSv4 hybrid attention forward",
)


class _FakeGroup:
    def __init__(self, nranks=1):
        self.nranks = nranks
        self.world_size = nranks
        self.ranks = list(range(nranks))
        self.rank = 0


class _FakePGCollection:
    def __init__(self, tp_nranks=1, cp_nranks=1):
        self.tp = _FakeGroup(tp_nranks)
        self.cp = _FakeGroup(cp_nranks)


def _make_config(
    csa_compress_ratios,
    qk_pos_emb_head_dim=_GLOBAL_POS_DIM,
    hca_rope_type=None,
    csa_rope_type=None,
    num_layers=None,
):
    """Build a dsv4_hybrid TransformerConfig forwarding the new per-type fields.

    Mirrors the construction-friendly setup used by test_dsv4_hybrid_attention
    (CPU-buildable dimensions), plus the two incremental fields under test.
    """
    if num_layers is None:
        num_layers = len(csa_compress_ratios)
    v_head_dim = 32
    return TransformerConfig(
        num_hidden_layers=num_layers,
        num_nextn_predict_layers=0,
        hidden_size=256,
        num_attention_heads=8,
        params_dtype=paddle.bfloat16,
        bf16=True,
        use_bias=False,
        multi_latent_attention=True,
        experimental_attention_variant="dsv4_hybrid",
        q_lora_rank=64,
        kv_lora_rank=v_head_dim - qk_pos_emb_head_dim,
        qk_nope_head_dim=v_head_dim - qk_pos_emb_head_dim,
        qk_rope_head_dim=qk_pos_emb_head_dim,
        qk_pos_emb_head_dim=qk_pos_emb_head_dim,
        v_head_dim=v_head_dim,
        hybrid_mla_q_lora_rank=1536,
        hybrid_mla_kv_lora_rank=512,
        hybrid_mla_qk_nope_head_dim=192,
        hybrid_mla_qk_rope_head_dim=64,
        hybrid_mla_v_head_dim=256,
        hybrid_mla_num_attention_heads=64,
        hybrid_mla_num_key_value_heads=64,
        o_groups=4,
        o_lora_rank=32,
        rope_type="rope",
        rotary_base=10000.0,
        rotary_percent=_ROTARY_PERCENT,
        normalization="RMSNorm",
        use_qk_norm=True,
        csa_compress_ratios=csa_compress_ratios,
        csa_window_size=16,
        csa_compress_rotary_base=str(_COMPRESSED_ROPE_BASE),
        dsa_index_n_heads=4,
        dsa_index_head_dim=32,
        dsa_index_topk=8,
        dsa_indexer_loss_coeff=1.0,
        dsa_indexer_use_sparse_loss=False,
        dsa_indexer_rotary_interleaved=False,
        apply_rope_fusion=False,
        attention_dropout=0.0,
        attention_softmax_in_fp32=True,
        masked_softmax_fusion=False,
        softmax_type="vanilla",
        csa_indexer_backend="unfused",
        csa_sparse_attn_backend="unfused",
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        csa_dense_mode=False,
        hca_rope_type=hca_rope_type,
        csa_rope_type=csa_rope_type,
    )


def _build_dsv4(config, layer_number):
    """Construct a DSv4HybridSelfAttention layer (CPU friendly)."""
    spec = get_gpt_layer_local_spec(
        config=config,
        normalization=config.normalization,
        layer_number=layer_number,
    ).sublayers_spec.self_attn
    return build_spec_layer(
        spec,
        config=config,
        layer_number=layer_number,
        pg_collection=_FakePGCollection(),
    )


def _rope_dim(module):
    """Return the effective RoPE head dim for either embedding variant.

    YarnRotaryEmbedding stores ``dim`` directly; plain RotaryEmbedding does
    not, so derive it from ``inv_freq`` (length == dim // 2).
    """
    if isinstance(module, YarnRotaryEmbedding):
        return module.dim
    return int(module.inv_freq.shape[0]) * 2


def _legacy_plain_rope_patch():
    """Emulate the pre-feature code path with ``use_compressed_yarn = False``.

    Before this feature DSv4HybridAttention picked the RoPE variant with a
    local flag::

        use_compressed_yarn = compress_ratio > 1
        if not use_compressed_yarn:
            self.rotary_pos_emb = RotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_percent=getattr(config, "rotary_percent", 1.0),
                rotary_base=rope_base,
            )
        else:
            self.rotary_pos_emb = YarnRotaryEmbedding(...)

    Forcing that flag to ``False`` on a compressed layer therefore means
    building a plain ``RotaryEmbedding`` with the very same dim and (compressed)
    base. Patching the YaRN symbol with a shim that does exactly that gives the
    reference behavior without editing the shipped source.
    """

    def _shim(dim, *, rotary_base, **_yarn_only_kwargs):
        return RotaryEmbedding(
            dim,
            rotary_percent=_ROTARY_PERCENT,
            rotary_base=rotary_base,
        )

    return mock.patch.object(
        dsv4_hybrid_attention, "YarnRotaryEmbedding", _shim
    )


# Layer layout used by every test below (csa_compress_ratios):
# index 0 window(0), 1 CSA(4), 2 HCA(128), 3 full-causal MQA(-1).
_RATIOS = [0, 4, 128, -1]
_WINDOW_LAYER, _CSA_LAYER, _HCA_LAYER, _MQA_LAYER = 0, 1, 2, 3


class TestConfigDefaultsAndValidation(unittest.TestCase):
    """Field declaration, transform_rules exposure and __post_init__ checks."""

    def test_new_fields_default_to_none(self):
        config = _make_config(csa_compress_ratios=_RATIOS)
        self.assertIsNone(config.hca_rope_type)
        self.assertIsNone(config.csa_rope_type)

    def test_transform_rules_expose_new_fields(self):
        rules = TransformerConfig.transform_rules
        for name in ("hca_rope_type", "csa_rope_type"):
            self.assertIn(name, rules)
            self.assertEqual(rules[name], name)

    def test_valid_rope_types_accepted(self):
        for hca, csa in (
            ("rope", "yarn"),
            ("yarn", "rope"),
            ("rope", "rope"),
            ("yarn", "yarn"),
        ):
            config = _make_config(
                csa_compress_ratios=_RATIOS,
                hca_rope_type=hca,
                csa_rope_type=csa,
            )
            self.assertEqual(config.hca_rope_type, hca)
            self.assertEqual(config.csa_rope_type, csa)

    def test_invalid_hca_rope_type_raises(self):
        with self.assertRaisesRegex(ValueError, "hca_rope_type"):
            _make_config(csa_compress_ratios=_RATIOS, hca_rope_type="linear")

    def test_invalid_csa_rope_type_raises(self):
        with self.assertRaisesRegex(ValueError, "csa_rope_type"):
            _make_config(csa_compress_ratios=_RATIOS, csa_rope_type="ROPE")


class TestPerTypeRopeResolution(unittest.TestCase):
    """Per-attention-type RoPE variant resolution in DSv4HybridAttention."""

    def _build_all(self, **kwargs):
        model_parallel_cuda_manual_seed(_SEED)
        config = _make_config(csa_compress_ratios=_RATIOS, **kwargs)
        return tuple(
            _build_dsv4(config, layer_number=i)
            for i in (_WINDOW_LAYER, _CSA_LAYER, _HCA_LAYER, _MQA_LAYER)
        )

    def test_default_backward_compatible(self):
        window, csa, hca, mqa = self._build_all()
        # Compressed layers (CSA/HCA) historically default to YaRN.
        self.assertIsInstance(csa.rotary_pos_emb, YarnRotaryEmbedding)
        self.assertIsInstance(hca.rotary_pos_emb, YarnRotaryEmbedding)
        # Window / MQA layers default to plain RoPE.
        self.assertIsInstance(window.rotary_pos_emb, RotaryEmbedding)
        self.assertIsInstance(mqa.rotary_pos_emb, RotaryEmbedding)

    def test_hca_rope_override_keeps_csa_yarn(self):
        _, csa, hca, _ = self._build_all(hca_rope_type="rope")
        self.assertIsInstance(hca.rotary_pos_emb, RotaryEmbedding)
        self.assertIsInstance(csa.rotary_pos_emb, YarnRotaryEmbedding)

    def test_csa_rope_override_keeps_hca_yarn(self):
        _, csa, hca, _ = self._build_all(csa_rope_type="rope")
        self.assertIsInstance(csa.rotary_pos_emb, RotaryEmbedding)
        self.assertIsInstance(hca.rotary_pos_emb, YarnRotaryEmbedding)

    def test_explicit_yarn_and_rope_mix(self):
        _, csa, hca, _ = self._build_all(
            hca_rope_type="yarn", csa_rope_type="rope"
        )
        self.assertIsInstance(hca.rotary_pos_emb, YarnRotaryEmbedding)
        self.assertIsInstance(csa.rotary_pos_emb, RotaryEmbedding)

    def test_window_and_mqa_layers_ignore_overrides(self):
        # Neither field applies to window(0) / MQA(-1) layers: they keep plain
        # RoPE even when both compressed types are pinned to YaRN.
        window, _, _, mqa = self._build_all(
            hca_rope_type="yarn", csa_rope_type="yarn"
        )
        self.assertIsInstance(window.rotary_pos_emb, RotaryEmbedding)
        self.assertIsInstance(mqa.rotary_pos_emb, RotaryEmbedding)

    def test_override_keeps_global_pos_dim_and_compressed_base(self):
        # The feature only switches the variant: the RoPE width stays the
        # global qk_pos_emb_head_dim and the compressed rotary base is still
        # csa_compress_rotary_base.
        _, csa, hca, _ = self._build_all(
            hca_rope_type="rope", csa_rope_type="rope"
        )
        reference = RotaryEmbedding(
            _GLOBAL_POS_DIM,
            rotary_percent=_ROTARY_PERCENT,
            rotary_base=_COMPRESSED_ROPE_BASE,
        )
        for layer in (csa, hca):
            self.assertEqual(layer.qk_pos_emb_head_dim, _GLOBAL_POS_DIM)
            self.assertEqual(_rope_dim(layer.rotary_pos_emb), _GLOBAL_POS_DIM)
            self.assertTrue(
                paddle.equal_all(
                    layer.rotary_pos_emb.inv_freq, reference.inv_freq
                )
            )


def _build_layer(layer_number, **kwargs):
    model_parallel_cuda_manual_seed(_SEED)
    config = _make_config(csa_compress_ratios=_RATIOS, **kwargs)
    return _build_dsv4(config, layer_number=layer_number)


def _build_legacy_plain_rope_layer(layer_number):
    """Reference layer: pre-feature code with ``use_compressed_yarn = False``."""
    with _legacy_plain_rope_patch():
        return _build_layer(layer_number)


class TestLegacyPlainRopeEquivalence(unittest.TestCase):
    """rope_type="rope" builds the same module as legacy use_compressed_yarn=False."""

    def _assert_same_rotary(self, layer_number, **kwargs):
        legacy = _build_legacy_plain_rope_layer(layer_number)
        new = _build_layer(layer_number, **kwargs)
        self.assertIsInstance(legacy.rotary_pos_emb, RotaryEmbedding)
        self.assertIsInstance(new.rotary_pos_emb, RotaryEmbedding)
        self.assertEqual(
            _rope_dim(new.rotary_pos_emb), _rope_dim(legacy.rotary_pos_emb)
        )
        self.assertTrue(
            paddle.equal_all(
                new.rotary_pos_emb.inv_freq, legacy.rotary_pos_emb.inv_freq
            )
        )

    def test_hca_rotary_matches_legacy(self):
        self._assert_same_rotary(_HCA_LAYER, hca_rope_type="rope")

    def test_csa_rotary_matches_legacy(self):
        self._assert_same_rotary(_CSA_LAYER, csa_rope_type="rope")

    def test_default_yarn_differs_from_legacy_plain_rope(self):
        # Guards the assertions above against being vacuous: with the fields
        # unset the compressed layer really is a different (YaRN) module.
        legacy = _build_legacy_plain_rope_layer(_HCA_LAYER)
        default = _build_layer(_HCA_LAYER)
        self.assertIsInstance(legacy.rotary_pos_emb, RotaryEmbedding)
        self.assertIsInstance(default.rotary_pos_emb, YarnRotaryEmbedding)


@_REQUIRES_CUDA
class TestLegacyPlainRopeForwardEquivalence(unittest.TestCase):
    """End-to-end training-step equivalence with legacy use_compressed_yarn=False.

    Runs a real forward + backward on both layers with identical weights and
    identical inputs, then requires bit-wise identical outputs and input
    gradients. This is the guarantee an actual training run depends on.
    """

    SEQ_LEN = 16
    HIDDEN_SIZE = 256

    def setUp(self):
        rng = np.random.default_rng(_SEED)
        shape = [1, self.SEQ_LEN, self.HIDDEN_SIZE]
        self.hidden_np = rng.standard_normal(shape, dtype=np.float32)
        self.out_grad_np = rng.standard_normal(shape, dtype=np.float32)

    def _forward_backward(self, layer):
        layer.train()
        hidden = paddle.to_tensor(self.hidden_np, dtype="bfloat16")
        hidden.stop_gradient = False
        out, _ = layer(hidden_states=hidden, attention_mask=None)
        self.assertEqual(out.shape, [1, self.SEQ_LEN, self.HIDDEN_SIZE])
        out.backward(paddle.to_tensor(self.out_grad_np, dtype=out.dtype))
        self.assertIsNotNone(hidden.grad)
        return (
            out.astype("float32").numpy(),
            hidden.grad.astype("float32").numpy(),
        )

    def _assert_forward_equivalent(self, layer_number, **kwargs):
        legacy = _build_legacy_plain_rope_layer(layer_number)
        new = _build_layer(layer_number, **kwargs)
        # Neutralize any init nondeterminism: the comparison must isolate the
        # RoPE path, not the parameter initialization.
        new.set_state_dict(legacy.state_dict())
        legacy_out, legacy_grad = self._forward_backward(legacy)
        new_out, new_grad = self._forward_backward(new)
        np.testing.assert_array_equal(new_out, legacy_out)
        np.testing.assert_array_equal(new_grad, legacy_grad)
        return legacy_out

    def test_hca_forward_matches_legacy(self):
        self._assert_forward_equivalent(_HCA_LAYER, hca_rope_type="rope")

    def test_csa_forward_matches_legacy(self):
        self._assert_forward_equivalent(_CSA_LAYER, csa_rope_type="rope")

    def test_default_yarn_forward_differs_from_legacy(self):
        # Vacuity guard: the equality above must come from the rope_type
        # override, not from YaRN and plain RoPE happening to agree.
        legacy = _build_legacy_plain_rope_layer(_HCA_LAYER)
        default = _build_layer(_HCA_LAYER)
        default.set_state_dict(legacy.state_dict())
        legacy_out, _ = self._forward_backward(legacy)
        default_out, _ = self._forward_backward(default)
        self.assertFalse(np.array_equal(default_out, legacy_out))


if __name__ == "__main__":
    unittest.main()
