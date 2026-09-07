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

"""Unit tests for DSv4HybridAttention selective recompute (full_attn / gated_attn)."""

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


import types as _types
import unittest
from unittest.mock import patch

import paddle

from paddlefleet.transformer.dsv4_hybrid_attention import (
    DSv4HybridAttention,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dsv4_instance(
    recompute_granularity="selective",
    recompute_modules=None,
    gated_attention=True,
    gated_attn_use_q_lora=False,
):
    """Create a SimpleNamespace that behaves like DSv4HybridAttention for forward().

    Uses SimpleNamespace to avoid paddle.nn.Layer.__setattr__ restrictions.
    Binds the real forward / _full_attn_forward / _gate methods from DSv4HybridAttention.
    """
    import types

    config = _types.SimpleNamespace(
        recompute_granularity=recompute_granularity,
        recompute_modules=recompute_modules,
        gated_attention=gated_attention,
        gated_attn_use_q_lora=gated_attn_use_q_lora,
        q_lora_rank=32,
        hidden_size=64,
        num_attention_heads=4,
        v_head_dim=16,
        qk_pos_emb_head_dim=8,
        o_groups=2,
        o_lora_rank=8,
        use_bias=False,
        sigmoid_gate_fusion=False,
        apply_rope_fusion=False,
        high_precision_rope=False,
        csa_dense_mode=True,
        cp_balance_mode="contiguous_allgather",
        max_sequence_length=128,
        fp8=None,
        full_fp8_computation=False,
        fp8_wgrad=False,
    )

    inst = _types.SimpleNamespace()
    inst.config = config
    inst.num_attention_heads = config.num_attention_heads
    inst.v_head_dim = config.v_head_dim
    inst.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim
    inst.o_local_groups = config.o_groups
    inst.training = True
    # inspect_tensor probes in forward() / _full_attn_forward() read
    # self.layer_number; the real module sets it in Attention.__init__.
    inst.layer_number = 0

    # Recompute flags (same logic as __init__)
    inst.recompute_gated_attn = (
        recompute_granularity == "selective"
        and recompute_modules is not None
        and "gated_attn" in recompute_modules
    )
    inst.recompute_full_attn = (
        recompute_granularity == "selective"
        and recompute_modules is not None
        and "full_attn" in recompute_modules
    )
    inst.recompute_qkv = (
        recompute_granularity == "selective"
        and recompute_modules is not None
        and "dsv4_hybrid_attn_qkv" in recompute_modules
    )
    inst.recompute_post_core = False
    inst._full_attn_recompute = None
    inst._qkv_recompute = None
    inst._post_core_recompute = None
    inst._gate_recompute = None

    # VHA postmix disabled for these recompute-path tests (forward() reads
    # these flags; the real module sets them in __init__).
    inst.use_vha_postmix = False
    inst.recompute_vha_postmix = False

    # Gated attention
    inst.gated_attention = gated_attention
    inst.gated_attn_use_q_lora = gated_attn_use_q_lora

    # Fake projections
    hidden_size = config.hidden_size
    o_lora_rank = config.o_lora_rank
    o_groups = config.o_groups
    np_ = config.num_attention_heads
    v_dim = config.v_head_dim

    # linear_o_group_proj: [o_groups * o_lora_rank, np*v_dim // o_groups]
    group_proj_in = np_ * v_dim // o_groups
    inst.linear_o_group_proj = paddle.randn(
        [o_groups * o_lora_rank, group_proj_in]
    )

    # o_proj mock
    def _fake_o_proj(x):
        b, s, _ = x.shape
        return paddle.randn([b, s, hidden_size]), None

    inst.o_proj = _fake_o_proj

    # gate_proj mock
    if gated_attention:

        def _fake_gate_proj(x):
            b, s, _ = x.shape
            return paddle.randn([b, s, o_groups * o_lora_rank]), None

        inst.gate_proj = _fake_gate_proj

    # Rotary embedding mock
    class _FakeRotaryEmb:
        def __call__(self, max_seq_len, offset=0, **kwargs):
            dim = config.qk_pos_emb_head_dim
            return (
                paddle.zeros([1, max_seq_len + offset, 1, dim]),
                1.0,
            )

    inst.rotary_pos_emb = _FakeRotaryEmb()

    # pg_collection mock (cp disabled)
    inst.pg_collection = _types.SimpleNamespace(cp=None, tp=None)

    # core_attention mock
    def _fake_core_attention(q, k, v, mask, **kwargs):
        b, s = q.shape[0], q.shape[1]
        return paddle.randn([b, s, np_ * v_dim])

    inst.core_attention = _fake_core_attention
    inst.core_attention.compress_ratio = 0

    # get_query_key_value_tensors mock
    def _fake_get_qkv(
        hidden_states, position_offset=0, docmask_meta=None, **kw
    ):
        b, s, _ = hidden_states.shape
        q = paddle.randn([b, s, np_, v_dim])
        k = paddle.randn([b, s, 1, v_dim])
        v_ = k
        q_compressed = paddle.randn([b, s, config.q_lora_rank])
        kv_compressed = paddle.randn([b, s, v_dim])
        return q, k, v_, q_compressed, kv_compressed

    inst.get_query_key_value_tensors = _fake_get_qkv

    # Bind the real methods from DSv4HybridAttention
    inst.forward = types.MethodType(DSv4HybridAttention.forward, inst)
    inst._full_attn_forward = types.MethodType(
        DSv4HybridAttention._full_attn_forward, inst
    )
    inst._qkv_forward = types.MethodType(DSv4HybridAttention._qkv_forward, inst)
    inst._post_core_forward = types.MethodType(
        DSv4HybridAttention._post_core_forward, inst
    )
    inst._inv_rope_postmix_forward = types.MethodType(
        DSv4HybridAttention._inv_rope_postmix_forward, inst
    )
    inst._o_group_proj_forward = types.MethodType(
        DSv4HybridAttention._o_group_proj_forward, inst
    )
    inst._gate = types.MethodType(DSv4HybridAttention._gate, inst)
    # _full_attn_forward consults this before the inverse-RoPE block. Bind the
    # real gate rather than a stub so a change to its conditions surfaces here;
    # with use_vha_postmix False it returns False, keeping these tests on the
    # unfused path they were written for.
    inst._can_fuse_inv_rope_postmix = types.MethodType(
        DSv4HybridAttention._can_fuse_inv_rope_postmix, inst
    )
    inst.vha_postmix_grouped = False

    return inst


# ---------------------------------------------------------------------------
# Test: Recompute flag initialization
# ---------------------------------------------------------------------------


class TestDSv4RecomputeFlags(unittest.TestCase):
    """Tests for recompute_full_attn / recompute_gated_attn flag initialization."""

    def test_full_attn_enabled(self):
        inst = _make_dsv4_instance(recompute_modules=["full_attn"])
        self.assertTrue(inst.recompute_full_attn)
        self.assertFalse(inst.recompute_gated_attn)

    def test_gated_attn_enabled(self):
        inst = _make_dsv4_instance(recompute_modules=["gated_attn"])
        self.assertFalse(inst.recompute_full_attn)
        self.assertTrue(inst.recompute_gated_attn)

    def test_both_enabled(self):
        inst = _make_dsv4_instance(
            recompute_modules=["full_attn", "gated_attn"]
        )
        self.assertTrue(inst.recompute_full_attn)
        self.assertTrue(inst.recompute_gated_attn)

    def test_none_modules(self):
        inst = _make_dsv4_instance(recompute_modules=None)
        self.assertFalse(inst.recompute_full_attn)
        self.assertFalse(inst.recompute_gated_attn)

    def test_non_selective_granularity(self):
        inst = _make_dsv4_instance(
            recompute_granularity="full",
            recompute_modules=["full_attn", "gated_attn"],
        )
        self.assertFalse(inst.recompute_full_attn)
        self.assertFalse(inst.recompute_gated_attn)


# ---------------------------------------------------------------------------
# Test: forward() with full_attn recompute
# ---------------------------------------------------------------------------


class TestDSv4FullAttnRecompute(unittest.TestCase):
    """Tests for the full_attn recompute branch in forward()."""

    def setUp(self):
        paddle.seed(42)

    def test_full_attn_recompute_uses_RecomputeWithoutOutput(self):
        """Verify full_attn branch instantiates RecomputeWithoutOutput."""
        inst = _make_dsv4_instance(recompute_modules=["full_attn"])

        from paddlefleet.transformer import dsv4_hybrid_attention as mod

        call_count = [0]

        class _MockRecompute:
            def __init__(self):
                call_count[0] += 1

            def recompute(
                self,
                fn,
                *args,
                preserve_rng_state=True,
                share_grad_holder=False,
            ):
                return fn(*args)

            def discard_output_and_register_recompute(self, output):
                pass

        hidden = paddle.randn([1, 8, 64])
        with patch.object(mod, "RecomputeWithoutOutput", _MockRecompute):
            output, bias = inst.forward(hidden, attention_mask=None)

        self.assertEqual(call_count[0], 1)
        self.assertEqual(output.shape, [1, 8, 64])
        self.assertIsNone(bias)

    def test_full_attn_recompute_output_matches_else_branch(self):
        """full_attn path and else path should produce same shape output."""
        paddle.seed(123)
        inst_full = _make_dsv4_instance(recompute_modules=["full_attn"])
        hidden = paddle.randn([1, 8, 64])

        from paddlefleet.transformer import dsv4_hybrid_attention as mod

        class _PassthroughRecompute:
            def recompute(
                self,
                fn,
                *args,
                preserve_rng_state=True,
                share_grad_holder=False,
            ):
                return fn(*args)

            def discard_output_and_register_recompute(self, output):
                pass

        with patch.object(mod, "RecomputeWithoutOutput", _PassthroughRecompute):
            out_full, _ = inst_full.forward(hidden, attention_mask=None)

        paddle.seed(123)
        inst_else = _make_dsv4_instance(recompute_modules=None)
        out_else, _ = inst_else.forward(hidden, attention_mask=None)

        self.assertEqual(out_full.shape, out_else.shape)

    def test_full_attn_sets_and_clears_recompute_attr(self):
        """_full_attn_recompute is set during forward and cleared after."""
        inst = _make_dsv4_instance(recompute_modules=["full_attn"])

        from paddlefleet.transformer import dsv4_hybrid_attention as mod

        recompute_instances = []

        class _TrackingRecompute:
            def __init__(self):
                recompute_instances.append(self)

            def recompute(
                self,
                fn,
                *args,
                preserve_rng_state=True,
                share_grad_holder=False,
            ):
                return fn(*args)

            def discard_output_and_register_recompute(self, output):
                pass

        hidden = paddle.randn([1, 8, 64])
        with patch.object(mod, "RecomputeWithoutOutput", _TrackingRecompute):
            inst.forward(hidden, attention_mask=None)

        # After forward, _full_attn_recompute should be None
        self.assertIsNone(inst._full_attn_recompute)


# ---------------------------------------------------------------------------
# Test: forward() with gated_attn recompute (else branch)
# ---------------------------------------------------------------------------


class TestDSv4GatedAttnRecompute(unittest.TestCase):
    """Tests for the gated_attn recompute in the else branch of forward()."""

    def setUp(self):
        paddle.seed(42)

    def test_gated_attn_recompute_uses_RecomputeWithoutOutput(self):
        """Verify gated_attn branch instantiates RecomputeWithoutOutput."""
        inst = _make_dsv4_instance(recompute_modules=["gated_attn"])

        from paddlefleet.transformer import dsv4_hybrid_attention as mod

        call_count = [0]

        class _MockRecompute:
            def __init__(self):
                call_count[0] += 1

            def recompute(
                self,
                fn,
                *args,
                preserve_rng_state=True,
                share_grad_holder=False,
            ):
                return fn(*args)

            def discard_output_and_register_recompute(self, output):
                pass

        hidden = paddle.randn([1, 8, 64])
        with patch.object(mod, "RecomputeWithoutOutput", _MockRecompute):
            output, bias = inst.forward(hidden, attention_mask=None)

        # gated_attn recompute should have been called once
        self.assertEqual(call_count[0], 1)
        self.assertEqual(output.shape, [1, 8, 64])

    def test_gated_attn_not_triggered_when_full_attn_active(self):
        """When full_attn is active, gated_attn SR in else branch is skipped."""
        inst = _make_dsv4_instance(
            recompute_modules=["full_attn", "gated_attn"]
        )

        from paddlefleet.transformer import dsv4_hybrid_attention as mod

        gate_recompute_count = [0]
        full_recompute_count = [0]

        class _MockRecompute:
            def __init__(self):
                pass

            def recompute(
                self,
                fn,
                *args,
                preserve_rng_state=True,
                share_grad_holder=False,
            ):
                # Distinguish: _full_attn_forward has 6 args (with _in_full_recompute), _gate has 2
                if len(args) >= 5:
                    full_recompute_count[0] += 1
                else:
                    gate_recompute_count[0] += 1
                return fn(*args)

            def discard_output_and_register_recompute(self, output):
                pass

        hidden = paddle.randn([1, 8, 64])
        with patch.object(mod, "RecomputeWithoutOutput", _MockRecompute):
            inst.forward(hidden, attention_mask=None)

        # full_attn recompute used once, gated_attn NOT used (inside full_attn scope)
        self.assertEqual(full_recompute_count[0], 1)
        self.assertEqual(gate_recompute_count[0], 0)

    def test_no_gated_attn_when_gated_attention_disabled(self):
        """When gated_attention=False, no gate recompute happens."""
        inst = _make_dsv4_instance(
            recompute_modules=["gated_attn"],
            gated_attention=False,
        )

        from paddlefleet.transformer import dsv4_hybrid_attention as mod

        call_count = [0]

        class _MockRecompute:
            def __init__(self):
                call_count[0] += 1

            def recompute(self, fn, *args, **kwargs):
                return fn(*args)

            def discard_output_and_register_recompute(self, output):
                pass

        hidden = paddle.randn([1, 8, 64])
        with patch.object(mod, "RecomputeWithoutOutput", _MockRecompute):
            inst.forward(hidden, attention_mask=None)

        # No recompute at all
        self.assertEqual(call_count[0], 0)


# ---------------------------------------------------------------------------
# Test: forward() without any recompute (direct path)
# ---------------------------------------------------------------------------


class TestDSv4DirectPath(unittest.TestCase):
    """Tests for the else branch without any selective recompute."""

    def setUp(self):
        paddle.seed(42)

    def test_direct_path_no_gate(self):
        """Forward without recompute and without gated_attention."""
        inst = _make_dsv4_instance(
            recompute_modules=None, gated_attention=False
        )
        hidden = paddle.randn([1, 8, 64])
        output, bias = inst.forward(hidden, attention_mask=None)
        self.assertEqual(output.shape, [1, 8, 64])
        self.assertIsNone(bias)

    def test_direct_path_with_gate(self):
        """Forward without recompute but with gated_attention."""
        inst = _make_dsv4_instance(recompute_modules=None, gated_attention=True)
        hidden = paddle.randn([1, 8, 64])
        output, bias = inst.forward(hidden, attention_mask=None)
        self.assertEqual(output.shape, [1, 8, 64])
        self.assertIsNone(bias)


# ---------------------------------------------------------------------------
# Test: _full_attn_forward method
# ---------------------------------------------------------------------------


class TestDSv4FullAttnForwardMethod(unittest.TestCase):
    """Tests for the _full_attn_forward helper method."""

    def setUp(self):
        paddle.seed(42)

    def test_full_attn_forward_returns_correct_shape(self):
        """_full_attn_forward should return [b, sq, o_groups * o_lora_rank]."""
        inst = _make_dsv4_instance(
            recompute_modules=["full_attn"], gated_attention=True
        )
        hidden = paddle.randn([1, 8, 64])
        result = inst._full_attn_forward(hidden, None, 0, None, None)
        # o_groups=2, o_lora_rank=8 => 16
        self.assertEqual(result.shape, [1, 8, 16])

    def test_full_attn_forward_includes_gate(self):
        """_full_attn_forward should call _gate when gated_attention=True."""
        inst = _make_dsv4_instance(
            recompute_modules=["full_attn"], gated_attention=True
        )
        gate_called = [False]
        original_gate = inst._gate

        def _tracking_gate(gate_source, core_attn_out):
            gate_called[0] = True
            return original_gate(gate_source, core_attn_out)

        inst._gate = _tracking_gate

        hidden = paddle.randn([1, 8, 64])
        inst._full_attn_forward(hidden, None, 0, None, None)

        self.assertTrue(gate_called[0])

    def test_full_attn_forward_skips_gate_when_disabled(self):
        """_full_attn_forward should not call _gate when gated_attention=False."""
        inst = _make_dsv4_instance(
            recompute_modules=["full_attn"], gated_attention=False
        )
        gate_called = [False]

        def _tracking_gate(gate_source, core_attn_out):
            gate_called[0] = True
            return core_attn_out

        inst._gate = _tracking_gate

        hidden = paddle.randn([1, 8, 64])
        inst._full_attn_forward(hidden, None, 0, None, None)

        self.assertFalse(gate_called[0])


# ---------------------------------------------------------------------------
# Test: _gate method
# ---------------------------------------------------------------------------


class TestDSv4Gate(unittest.TestCase):
    """Tests for the _gate method."""

    def test_gate_output_shape(self):
        """_gate should return same shape as core_attn_out."""
        inst = _make_dsv4_instance(gated_attention=True)
        gate_source = paddle.randn([1, 8, 64])
        core_attn_out = paddle.randn([1, 8, 16])
        result = inst._gate(gate_source, core_attn_out)
        self.assertEqual(result.shape, [1, 8, 16])

    def test_gate_applies_sigmoid(self):
        """_gate should apply sigmoid gating (output = attn_out * sigmoid(gate))."""
        inst = _make_dsv4_instance(gated_attention=True)

        # Make gate_proj return all zeros => sigmoid(0)=0.5
        def _zero_gate(x):
            b, s, _ = x.shape
            return paddle.zeros([b, s, 16]), None

        inst.gate_proj = _zero_gate
        core_attn_out = paddle.ones([1, 8, 16])
        result = inst._gate(paddle.randn([1, 8, 64]), core_attn_out)
        # Should be approximately 0.5
        self.assertTrue(
            paddle.allclose(
                result,
                paddle.full([1, 8, 16], 0.5),
                atol=1e-6,
            ).item()
        )


# ---------------------------------------------------------------------------
# Test: Real backward/gradient regression for full_attn + gated_attn recompute
# ---------------------------------------------------------------------------

# Import real infrastructure for backward tests
from paddle.distributed.fleet.meta_parallel import build_spec_layer

from paddlefleet.models.gpt.gpt_layer_specs import get_attention_spec
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.transformer_config import TransformerConfig

_SEED = 42


class _FakeGroup:
    def __init__(self, nranks=1):
        self.nranks = nranks
        self.ranks = list(range(nranks))
        self.rank = 0


class _FakePGCollection:
    def __init__(self, tp_nranks=1, cp_nranks=1):
        self.tp = _FakeGroup(tp_nranks)
        self.cp = _FakeGroup(cp_nranks)


def _make_real_config(
    recompute_granularity=None,
    recompute_modules=None,
    gated_attention=False,
    gated_attn_use_q_lora=False,
):
    """Create a TransformerConfig suitable for DSv4 backward testing."""
    return TransformerConfig(
        num_hidden_layers=4,
        hidden_size=256,
        num_attention_heads=8,
        params_dtype=paddle.bfloat16,
        bf16=True,
        use_bias=False,
        multi_latent_attention=True,
        experimental_attention_variant="dsv4_hybrid",
        q_lora_rank=64,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
        o_groups=4,
        o_lora_rank=32,
        rope_type="rope",
        rotary_base=10000.0,
        rotary_percent=1.0,
        normalization="RMSNorm",
        use_qk_norm=True,
        csa_compress_ratios=[0, 4, 128, 4],
        csa_window_size=16,
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
        csa_dense_mode=True,
        gated_attention=gated_attention,
        gated_attn_use_q_lora=gated_attn_use_q_lora,
        recompute_granularity=recompute_granularity,
        recompute_modules=recompute_modules,
    )


def _build_real_attention(config, layer_number=1):
    """Build a real DSv4HybridSelfAttention instance."""
    spec = get_attention_spec(
        config=config,
        attention_layer_type="dsv4_hybrid_attention",
        attn_mask_type=AttnMaskType.causal,
    )
    return build_spec_layer(spec, config=config, layer_number=layer_number)


class TestDSv4RecomputeBackwardGradients(unittest.TestCase):
    """Real backward/gradient regression tests for full_attn + gated_attn recompute.

    These tests use real RecomputeWithoutOutput (not mocks) and call .backward()
    to verify that gradients are correct when recompute is enabled.
    """

    def _make_startend(self, batch_size, seq_len):
        return paddle.full([batch_size, 1, seq_len, 1], seq_len, dtype="int32")

    def test_full_attn_recompute_backward_gradient_flow(self):
        """full_attn recompute produces finite, non-None gradients on backward."""
        paddle.seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)
        config = _make_real_config(
            recompute_granularity="selective",
            recompute_modules=["full_attn"],
            gated_attention=False,
        )
        attn = _build_real_attention(config, layer_number=1)
        attn.train()

        batch_size, seq_len = 1, 64
        hidden = paddle.randn(
            [batch_size, seq_len, config.hidden_size], dtype=paddle.bfloat16
        )
        hidden.stop_gradient = False

        output, _ = attn(
            hidden_states=hidden,
            attention_mask=None,
            attn_mask_startend_row_indices=self._make_startend(
                batch_size, seq_len
            ),
        )
        loss = output.cast("float32").sum()
        loss.backward()

        # Input grad must exist and be finite
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(
            paddle.isfinite(hidden.grad.cast("float32")).all().item(),
            "Input gradient contains non-finite values with full_attn recompute",
        )

        # At least some parameters should have gradients
        params_with_grad = [
            name
            for name, p in attn.named_parameters()
            if not p.stop_gradient and p.grad is not None
        ]
        self.assertGreater(
            len(params_with_grad),
            0,
            "No parameter received gradient with full_attn recompute",
        )

    def test_gated_attn_recompute_backward_gradient_flow(self):
        """gated_attn recompute produces finite, non-None gradients on backward."""
        paddle.seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)
        config = _make_real_config(
            recompute_granularity="selective",
            recompute_modules=["gated_attn"],
            gated_attention=True,
            gated_attn_use_q_lora=False,
        )
        attn = _build_real_attention(config, layer_number=1)
        attn.train()

        batch_size, seq_len = 1, 64
        hidden = paddle.randn(
            [batch_size, seq_len, config.hidden_size], dtype=paddle.bfloat16
        )
        hidden.stop_gradient = False

        output, _ = attn(
            hidden_states=hidden,
            attention_mask=None,
            attn_mask_startend_row_indices=self._make_startend(
                batch_size, seq_len
            ),
        )
        loss = output.cast("float32").sum()
        loss.backward()

        self.assertIsNotNone(hidden.grad)
        self.assertTrue(
            paddle.isfinite(hidden.grad.cast("float32")).all().item(),
            "Input gradient contains non-finite values with gated_attn recompute",
        )

        # gate_proj should receive gradients
        gate_params = [
            name
            for name, p in attn.named_parameters()
            if "gate_proj" in name and p.grad is not None
        ]
        self.assertGreater(
            len(gate_params),
            0,
            "gate_proj parameters did not receive gradients",
        )

    def test_full_attn_plus_gated_attn_recompute_backward_gradient_flow(self):
        """full_attn + gated_attn recompute produces finite gradients on backward."""
        paddle.seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)
        config = _make_real_config(
            recompute_granularity="selective",
            recompute_modules=["full_attn", "gated_attn"],
            gated_attention=True,
            gated_attn_use_q_lora=False,
        )
        attn = _build_real_attention(config, layer_number=1)
        attn.train()

        batch_size, seq_len = 1, 64
        hidden = paddle.randn(
            [batch_size, seq_len, config.hidden_size], dtype=paddle.bfloat16
        )
        hidden.stop_gradient = False

        output, _ = attn(
            hidden_states=hidden,
            attention_mask=None,
            attn_mask_startend_row_indices=self._make_startend(
                batch_size, seq_len
            ),
        )
        loss = output.cast("float32").sum()
        loss.backward()

        self.assertIsNotNone(hidden.grad)
        self.assertTrue(
            paddle.isfinite(hidden.grad.cast("float32")).all().item(),
            "Input gradient contains non-finite values with full+gated recompute",
        )

        # Both gate_proj and o_proj should receive gradients
        for pattern in ["gate_proj", "o_proj", "linear_o_group_proj"]:
            matched = [
                name
                for name, p in attn.named_parameters()
                if pattern in name and p.grad is not None
            ]
            self.assertGreater(
                len(matched),
                0,
                f"'{pattern}' parameters did not receive gradients "
                f"with full_attn + gated_attn recompute",
            )

    def test_full_attn_gated_attn_recompute_gradient_matches_no_recompute(self):
        """Gradients with full_attn + gated_attn recompute match those without recompute.

        This is the key regression test: it builds two identical layers (same weights),
        runs forward + backward on both, and verifies that input gradients and parameter
        gradients match between the recompute and no-recompute paths.
        """
        paddle.seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        # --- Recompute path ---
        config_rc = _make_real_config(
            recompute_granularity="selective",
            recompute_modules=["full_attn", "gated_attn"],
            gated_attention=True,
            gated_attn_use_q_lora=False,
        )
        attn_rc = _build_real_attention(config_rc, layer_number=1)

        # --- No-recompute path ---
        config_no = _make_real_config(
            recompute_granularity=None,
            recompute_modules=None,
            gated_attention=True,
            gated_attn_use_q_lora=False,
        )
        attn_no = _build_real_attention(config_no, layer_number=1)

        # Copy weights from recompute layer to no-recompute layer
        attn_no.set_state_dict(attn_rc.state_dict())

        attn_rc.train()
        attn_no.train()

        batch_size, seq_len = 1, 64
        x_data = paddle.randn(
            [batch_size, seq_len, config_rc.hidden_size], dtype=paddle.bfloat16
        )

        x_rc = x_data.clone()
        x_rc.stop_gradient = False
        x_no = x_data.clone()
        x_no.stop_gradient = False

        startend = self._make_startend(batch_size, seq_len)

        # Forward
        out_rc, _ = attn_rc(
            hidden_states=x_rc,
            attention_mask=None,
            attn_mask_startend_row_indices=startend,
        )
        out_no, _ = attn_no(
            hidden_states=x_no,
            attention_mask=None,
            attn_mask_startend_row_indices=startend,
        )

        # Forward outputs should match
        self.assertTrue(
            paddle.allclose(
                out_rc.cast("float32"),
                out_no.cast("float32"),
                rtol=1e-4,
                atol=1e-4,
            ).item(),
            "Forward outputs differ between recompute and no-recompute",
        )

        # Backward
        loss_rc = out_rc.cast("float32").sum()
        loss_no = out_no.cast("float32").sum()
        loss_rc.backward()
        loss_no.backward()

        # Input gradients must match
        self.assertIsNotNone(x_rc.grad)
        self.assertIsNotNone(x_no.grad)
        self.assertTrue(
            paddle.allclose(
                x_rc.grad.cast("float32"),
                x_no.grad.cast("float32"),
                rtol=1e-3,
                atol=1e-3,
            ).item(),
            "Input gradients differ between recompute and no-recompute",
        )

        # Parameter gradients must match
        rc_params = dict(attn_rc.named_parameters())
        no_params = dict(attn_no.named_parameters())

        # Critical params that MUST have grads on both sides
        critical_patterns = [
            "gate_proj",
            "o_proj",
            "linear_o_group_proj",
            "linear_q_down_proj",
            "linear_q_up_proj",
            "linear_kv_proj",
        ]
        for pattern in critical_patterns:
            matched_rc = [n for n in rc_params if pattern in n]
            for name in matched_rc:
                rc_grad = rc_params[name].grad
                no_grad = no_params[name].grad
                # Both sides must have non-None gradients
                self.assertIsNotNone(
                    rc_grad,
                    f"Recompute side grad is None for {name}",
                )
                self.assertIsNotNone(
                    no_grad,
                    f"No-recompute side grad is None for {name}",
                )
                # Gradients must be close
                self.assertTrue(
                    paddle.allclose(
                        rc_grad.cast("float32"),
                        no_grad.cast("float32"),
                        rtol=1e-2,
                        atol=1e-2,
                    ).item(),
                    f"Grad mismatch for param {name}",
                )

    def test_full_attn_recompute_gradient_matches_no_recompute(self):
        """Gradients with full_attn recompute (no gated) match those without recompute."""
        paddle.seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        config_rc = _make_real_config(
            recompute_granularity="selective",
            recompute_modules=["full_attn"],
            gated_attention=False,
        )
        attn_rc = _build_real_attention(config_rc, layer_number=1)

        config_no = _make_real_config(
            recompute_granularity=None,
            recompute_modules=None,
            gated_attention=False,
        )
        attn_no = _build_real_attention(config_no, layer_number=1)
        attn_no.set_state_dict(attn_rc.state_dict())

        attn_rc.train()
        attn_no.train()

        batch_size, seq_len = 1, 64
        x_data = paddle.randn(
            [batch_size, seq_len, config_rc.hidden_size], dtype=paddle.bfloat16
        )

        x_rc = x_data.clone()
        x_rc.stop_gradient = False
        x_no = x_data.clone()
        x_no.stop_gradient = False

        startend = self._make_startend(batch_size, seq_len)

        out_rc, _ = attn_rc(
            hidden_states=x_rc,
            attention_mask=None,
            attn_mask_startend_row_indices=startend,
        )
        out_no, _ = attn_no(
            hidden_states=x_no,
            attention_mask=None,
            attn_mask_startend_row_indices=startend,
        )

        self.assertTrue(
            paddle.allclose(
                out_rc.cast("float32"),
                out_no.cast("float32"),
                rtol=1e-4,
                atol=1e-4,
            ).item(),
            "Forward outputs differ (full_attn only)",
        )

        loss_rc = out_rc.cast("float32").sum()
        loss_no = out_no.cast("float32").sum()
        loss_rc.backward()
        loss_no.backward()

        self.assertIsNotNone(x_rc.grad)
        self.assertIsNotNone(x_no.grad)
        self.assertTrue(
            paddle.allclose(
                x_rc.grad.cast("float32"),
                x_no.grad.cast("float32"),
                rtol=1e-3,
                atol=1e-3,
            ).item(),
            "Input gradients differ (full_attn recompute vs no recompute)",
        )

        # Verify all trainable params have matching grads
        for (name_rc, p_rc), (_, p_no) in zip(
            attn_rc.named_parameters(), attn_no.named_parameters()
        ):
            if p_rc.grad is not None and p_no.grad is not None:
                self.assertTrue(
                    paddle.allclose(
                        p_rc.grad.cast("float32"),
                        p_no.grad.cast("float32"),
                        rtol=1e-2,
                        atol=1e-2,
                    ).item(),
                    f"Grad mismatch for param {name_rc} (full_attn recompute)",
                )


if __name__ == "__main__":
    unittest.main()
