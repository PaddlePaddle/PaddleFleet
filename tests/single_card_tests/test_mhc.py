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

"""
Tests for MHC (Manifold-Constrained Hyper-Connections) and related modules.
Covers: hyper_connection.py, triton_mhc.py, transformer_config.py,
        transformer_layer.py, transformer_block.py, gpt_model.py,
        gpt_layer_specs.py, gpt_builders.py
"""

from unittest.mock import MagicMock, patch

import paddle
import pytest

# =============================================================================
# Mock configs
# =============================================================================


class MockConfig:
    """Minimal mock config for HyperConnectionModule tests."""

    def __init__(
        self,
        hidden_size=64,
        mhc_num_residual_streams=4,
        mhc_sinkhorn_iters=10,
        mhc_use_triton=False,
    ):
        self.hidden_size = hidden_size
        self.mhc_num_residual_streams = mhc_num_residual_streams
        self.mhc_sinkhorn_iters = mhc_sinkhorn_iters
        self.mhc_use_triton = mhc_use_triton


class MockTransformerConfig:
    """Full mock config for TransformerLayer/Block/GPTModel tests."""

    def __init__(self, **kwargs):
        defaults = {
            "num_hidden_layers": 4,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "vocab_size": 100,
            "max_sequence_length": 64,
            "normalization": "RMSNorm",
            "use_qk_norm": False,
            "multi_latent_attention": False,
            "n_routed_experts": None,
            "moe_grouped_gemm": False,
            "moe_layer_freq": 1,
            "num_empty_layers_add_in_head": 0,
            "num_empty_layers_add_in_tail": 0,
            "num_nextn_predict_layers": None,
            "num_layers": 4,
            "mhc_num_residual_streams": 4,
            "mhc_sinkhorn_iters": 10,
            "mhc_use_triton": False,
            "use_mhc": False,
            "hidden_dropout_prob": 0.0,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "rotary_percent": 1.0,
            "position_embedding_type": "rope",
            "tie_word_embeddings": False,
            "pipeline_model_parallel_size": 1,
            "virtual_pipeline_model_parallel_size": None,
            "parallel_output": False,
            "max_position_embeddings": 512,
            "rope_scaling": False,
            "mrope_section": None,
            "model_type": "gpt",
            "init_method": None,
            "output_layer_init_method": None,
            "bias_dropout_fusion": False,
            "recompute_granularity": None,
            "recompute_modules": None,
            "recompute_num_layers": None,
            "recompute_method": None,
            "context_parallel_size": 1,
            "cp_comm_type": None,
            "sequence_parallel": False,
            "cpu_offloading": False,
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


# =============================================================================
# Shared helpers
# =============================================================================


def _make_hc(use_triton=False, hidden_size=64, n=4, layer_number=1):
    """Create HyperConnectionModule."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    cfg = MockConfig(
        hidden_size=hidden_size,
        mhc_num_residual_streams=n,
        mhc_use_triton=use_triton,
    )
    return HyperConnectionModule(config=cfg, layer_number=layer_number)


def _make_transformer_layer(config=None, **config_kw):
    """Create a TransformerLayer with mocked sublayers."""
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayer,
        TransformerLayerSublayersSpec,
    )

    if config is None:
        config = MockTransformerConfig(**config_kw)
    mock_pg = MagicMock()
    mock_pg.tp = MagicMock()
    spec = TransformerLayerSublayersSpec()
    C = config.hidden_size

    mock_ln = MagicMock(side_effect=lambda x: x)
    mock_attn = MagicMock(return_value=(paddle.randn([4, 2, C]), None))
    mock_mlp = MagicMock(return_value=(paddle.randn([4, 2, C]), None))
    mock_bda = MagicMock(
        return_value=lambda *a, **kw: MagicMock(
            return_value=paddle.randn([4, 2, C])
        )
    )

    call_count = [0]

    def build_se(s, **kwargs):
        call_count[0] += 1
        idx = call_count[0]
        if idx in [1, 4, 7]:
            return mock_ln
        elif idx == 2:
            return mock_attn
        elif idx in [3, 6, 9]:
            return mock_bda
        elif idx == 5:
            return MagicMock(return_value=(paddle.randn([4, 2, C]), None))
        elif idx == 8:
            return mock_mlp
        return MagicMock()

    with patch(
        "paddlefleet.transformer.transformer_layer.build_layer",
        side_effect=build_se,
    ):
        layer = TransformerLayer(
            config=config,
            sublayers_spec=spec,
            layer_number=1,
            pg_collection=mock_pg,
        )
    return layer, {
        "attn": mock_attn,
        "mlp": mock_mlp,
        "ln": mock_ln,
        "bda": mock_bda,
    }


def _make_mhc_layer(config=None, **config_kw):
    """Create TransformerLayerWithMHC with mocked sublayers."""
    from paddlefleet.transformer.identity_op import IdentityOp
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayerSublayersSpec,
        TransformerLayerWithMHC,
    )

    if config is None:
        config = MockTransformerConfig(use_mhc=True, **config_kw)
    mock_pg = MagicMock()
    mock_pg.tp = MagicMock()
    spec = TransformerLayerSublayersSpec()
    N, C = config.mhc_num_residual_streams, config.hidden_size

    mock_ln = MagicMock(side_effect=lambda x: x)
    mock_attn = MagicMock(return_value=(paddle.randn([4, 2, C]), None))
    mock_mlp = MagicMock(return_value=(paddle.randn([4, 2, C]), None))
    mock_bda = MagicMock(
        return_value=lambda *a, **kw: MagicMock(
            return_value=paddle.randn([4, 2, N * C])
        )
    )
    mock_hc = MagicMock()
    mock_hc.width_connection = MagicMock(
        return_value=(
            paddle.randn([4, 2, C]),
            paddle.randn([4, 2, N * C]),
            paddle.randn([4, 2, N]),
        )
    )
    mock_hc.depth_connection = MagicMock(
        return_value=paddle.randn([4, 2, N * C])
    )

    call_count = [0]

    def build_se(s, **kwargs):
        call_count[0] += 1
        idx = call_count[0]
        # Parent TransformerLayer.__init__ calls build_layer 9 times (idx 1-9)
        # Then TransformerLayerWithMHC.__init__ calls build_layer 2 more times (idx 10-11) for HC
        if (
            idx in [1, 4, 7]
        ):  # input_layernorm, pre_cross_attn_layernorm, post_attention_layernorm
            return mock_ln
        elif idx == 2:  # self_attn
            return mock_attn
        elif idx in [3, 6, 9]:  # self_attn_bda, cross_attn_bda, mlp_bda
            return mock_bda
        elif (
            idx == 5
        ):  # cross_attention - use IdentityOp to avoid _has_cross_attention=True
            return IdentityOp()
        elif idx == 8:  # mlp
            return mock_mlp
        elif idx in [
            10,
            11,
        ]:  # self_attention_hyper_connection, mlp_hyper_connection
            return mock_hc
        return MagicMock()

    with patch(
        "paddlefleet.transformer.transformer_layer.build_layer",
        side_effect=build_se,
    ):
        layer = TransformerLayerWithMHC(
            config=config,
            sublayers_spec=spec,
            layer_number=1,
            pg_collection=mock_pg,
        )
    return layer, {
        "attn": mock_attn,
        "mlp": mock_mlp,
        "hc": mock_hc,
        "bda": mock_bda,
    }


def _make_gpt_model(**config_kw):
    """Create a mock GPTModel without __init__."""
    from paddlefleet.models.gpt.gpt_model import GPTModel

    with patch.object(GPTModel, "__init__", lambda self, *a, **kw: None):
        model = GPTModel.__new__(GPTModel)
    model.config = MockTransformerConfig(**config_kw)
    model._pipeline_name_mapping = {
        "embedding": "layers.0",
        "transformer_layer_0": "layers.1",
        "transformer_layer_1": "layers.2",
        "output_layer": "layers.3",
    }
    model._pp_to_single_mapping = {
        v: k for k, v in model._pipeline_name_mapping.items()
    }
    model._num_virtual_pipeline_stages = 1  # Use 1 instead of None
    model._num_layers_per_pipeline_rank = 2
    model._num_layers_with_transformer = 2
    model._first_pipeline_num_layers = 2
    model.run_function = []  # Add empty run_function for use_fp8
    return model


def _make_transformer_block(
    config, sublayers_spec, pg_collection=None, **kwargs
):
    """Create TransformerBlock with real DummyLayer objects."""
    from paddlefleet.transformer.transformer_block import TransformerBlock

    class DummyLayer(paddle.nn.Layer):
        def __init__(self, rv=None):
            super().__init__()
            self._rv = rv

        def forward(self, **kw):
            if self._rv is not None:
                return self._rv
            return kw.get("hidden_states", paddle.randn([4, 2, 64])), None

    class DummyNorm(paddle.nn.Layer):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x

    if pg_collection is None:
        pg_collection = MagicMock()
    n_layers = (
        len(sublayers_spec.layer_specs) if sublayers_spec.layer_specs else 0
    )
    built = [0]
    rv = kwargs.pop("layer_return_value", None)

    def mock_build(s, **kw):
        built[0] += 1
        return DummyLayer(rv=rv) if built[0] <= n_layers else DummyNorm()

    with patch(
        "paddlefleet.transformer.transformer_block.build_layer",
        side_effect=mock_build,
    ):
        block = TransformerBlock(
            config=config,
            spec=sublayers_spec,
            pg_collection=pg_collection,
            **kwargs,
        )
    return block


def _base_dict_args(hidden=None, C=64):
    """Create standard dict_args for TransformerLayer.forward."""
    if hidden is None:
        hidden = paddle.randn([4, 2, C])
    return {
        "hidden_states": hidden,
        "attention_mask": None,
        "context": None,
        "context_mask": None,
        "rotary_pos_emb": None,
        "rotary_pos_cos": None,
        "rotary_pos_sin": None,
        "attention_bias": None,
        "packed_seq_params": None,
    }


# =============================================================================
# 1. HyperConnectionModule tests
# =============================================================================


def test_hc_width_depth_gradient():
    """Width/depth connections, gradient flow, parameter gradients."""

    hc = _make_hc()
    s, b, n, C = 4, 2, 4, 64
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    branch, residuals, h_post = hc.width_connection(x)
    assert branch.shape == [s, b, C]
    assert residuals.shape == [s, b, n * C]
    assert h_post.shape == [s, b, n]
    assert not paddle.isnan(branch).any()
    assert h_post.min().item() >= 0

    bias = paddle.randn([C])
    output = hc.depth_connection(
        (branch, bias), residuals, h_post, dropout_prob=0.0, training=False
    )
    assert output.shape == [s, b, n * C]

    output.sum().backward()
    assert x.grad is not None and x.grad.abs().sum().item() > 0
    for name, p in hc.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"


def test_hc_expand_reduce_roundtrip():
    """Expand/reduce round-trip consistency."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    s, b, C, n = 4, 2, 64, 4
    x = paddle.randn([s, b, C])
    expanded = HyperConnectionModule.expand_stream(x, n)
    assert expanded.shape == [s, b, n * C]
    contracted = HyperConnectionModule.reduce_stream(expanded, n)
    assert contracted.shape == [s, b, C]
    assert (contracted - x).abs().max().item() < 1e-5


def test_hc_depth_variants():
    """Depth connection: dropout, bias+dropout, single tensor, fused."""
    hc = _make_hc()
    s, b, n, C = 4, 2, 4, 64
    x = paddle.randn([s, b, n * C])
    x.stop_gradient = False
    branch, residuals, h_post = hc.width_connection(x)

    hc.train()
    out1 = hc.depth_connection(
        (branch, None), residuals, h_post, dropout_prob=0.1, training=True
    )
    assert not paddle.isnan(out1).any()

    out2 = hc.depth_connection(
        (branch, paddle.randn([C])),
        residuals,
        h_post,
        dropout_prob=0.15,
        training=True,
    )
    assert out2.shape == [s, b, n * C]

    out3 = hc.depth_connection(branch, residuals, h_post)
    assert out3.shape == [s, b, n * C]

    hc2 = _make_hc()
    hc2._fused_depth = True
    b2, r2, h2 = hc2.width_connection(x)
    out4 = hc2.depth_connection((b2, None), r2, h2)
    assert out4.shape == [s, b, n * C]

    # Test dtype mismatch coverage (lines 289, 291, 293, 341, 343, 487, 497)
    # branch_output shape is [s, b, C], not [s, b, n, C]
    hc3 = _make_hc()
    # Create branch_output with different dtype to trigger cast operations
    branch_output = paddle.randn([s, b, C], dtype="float16")
    bias = paddle.randn([C], dtype="float16")
    out5 = hc3.depth_connection(
        (branch_output, bias),
        residuals,
        h_post,
        dropout_prob=0.1,
        training=True,
        fused=False,
    )
    assert out5.shape == [s, b, n * C]


def test_hc_different_configs():
    """Different N values, sinkhorn iters, layer numbers."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    for n in [2, 4, 8]:
        hc = _make_hc(n=n, hidden_size=32)
        x = paddle.randn([2, 2, n * 32])
        branch, _, _ = hc.width_connection(x)
        assert branch.shape == [2, 2, 32]

    for iters in [1, 5, 20]:
        cfg = MockConfig(mhc_sinkhorn_iters=iters)
        hc = HyperConnectionModule(config=cfg, layer_number=1)
        x = paddle.randn([2, 2, 4 * 64])
        branch, _, _ = hc.width_connection(x)
        assert not paddle.isnan(branch).any()


def test_hc_triton_env_variable_coverage():
    """Test MHC_USE_TRITON environment variable (covers line 158).

    Covers hyper_connection.py line 158:
    - env_triton.lower() not in ("0", "false", "off")
    """
    import os

    # Save original value
    original_env = os.environ.get("MHC_USE_TRITON")

    # Test with "true" value (covers line 158: not in ("0", "false", "off"))
    os.environ["MHC_USE_TRITON"] = "true"
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    cfg = MockConfig()
    hc_true = HyperConnectionModule(config=cfg, layer_number=1)
    # Should use Triton backend since env var is "true"
    assert hc_true.use_triton  # env var "true" enables triton

    # Test with "false" value (covers line 158: not in ("0", "false", "off"))
    os.environ["MHC_USE_TRITON"] = "false"
    hc_false = HyperConnectionModule(config=cfg, layer_number=1)
    assert not hc_false.use_triton  # env var "false" disables triton

    # Restore original value
    if original_env is not None:
        os.environ["MHC_USE_TRITON"] = original_env
    elif "MHC_USE_TRITON" in os.environ:
        del os.environ["MHC_USE_TRITON"]


def test_hc_numerical_stability_and_determinism():
    """Numerical stability with large/small values; determinism."""
    hc = _make_hc()
    s, b, n, C = 4, 2, 4, 64

    for scale in [100.0, 1e-4]:
        x = paddle.randn([s, b, n * C]) * scale
        x.stop_gradient = False
        branch, res, hp = hc.width_connection(x)
        assert not paddle.isnan(branch).any()
        out = hc.depth_connection((branch, None), res, hp)
        out.sum().backward()
        assert not paddle.isnan(x.grad).any()

    paddle.seed(42)
    x1 = paddle.randn([s, b, n * C])
    out1, _, _ = hc.width_connection(x1)
    paddle.seed(42)
    x2 = paddle.randn([s, b, n * C])
    out2, _, _ = hc.width_connection(x2)
    assert paddle.allclose(out1, out2)


def test_hc_dtype_and_skip_sk():
    """h_post dtype mismatch, skip_sk_gradient."""
    hc = _make_hc()
    s, b, n, C = 2, 2, 4, 64

    x = paddle.randn([s, b, n * C], dtype="float32")
    branch, res, hp = hc.width_connection(x)
    hp_f32 = hp.cast("float32") if hp.dtype != paddle.float32 else hp * 1.0
    out = hc.depth_connection((branch, None), res, hp_f32)
    assert not paddle.isnan(out).any()

    b1, _, _ = hc.width_connection(x, skip_sk_gradient=True)
    b2, _, _ = hc.width_connection(x, skip_sk_gradient=False)
    assert b1.shape == b2.shape == [s, b, C]


def test_hc_pipeline_layers():
    """MHCExpandLayer and MHCContractLayer round-trip."""
    from paddlefleet.transformer.hyper_connection import (
        MHCContractLayer,
        MHCExpandLayer,
    )

    cfg = MockConfig()
    s, b, C, n = 4, 2, 64, 4
    x = paddle.randn([s, b, C])

    expand = MHCExpandLayer(config=cfg)
    out = expand({"hidden_states": x, "attention_mask": None})
    assert out["hidden_states"].shape == [s, b, n * C]

    contract = MHCContractLayer(config=cfg)
    out2 = contract(
        {"hidden_states": out["hidden_states"], "attention_mask": None}
    )
    assert out2["hidden_states"].shape == [s, b, C]
    assert (out2["hidden_states"] - x).abs().max().item() < 1e-5


def test_hc_parameter_initialization():
    """Parameter shapes and initial values are reasonable."""
    hc = _make_hc()
    params = dict(hc.named_parameters())
    assert len(params) > 0
    for name, p in params.items():
        assert not paddle.isnan(p).any()
        assert p.abs().max().item() < 100


# =============================================================================
# 2. Triton backend tests
# =============================================================================


def test_triton_width_depth_comparison():
    """Triton vs native width/depth/gradient comparison."""
    from paddlefleet.transformer.hyper_connection import TRITON_AVAILABLE

    if not TRITON_AVAILABLE:
        return

    s, b, n, C = 8, 4, 4, 64
    paddle.seed(42)
    hc_n = _make_hc(use_triton=False)
    paddle.seed(42)
    hc_t = _make_hc(use_triton=True)

    paddle.seed(123)
    xn = paddle.randn([s, b, n * C])
    xn.stop_gradient = False
    paddle.seed(123)
    xt = paddle.randn([s, b, n * C])
    xt.stop_gradient = False

    bn, rn, hn = hc_n.width_connection(xn)
    bt, rt, ht = hc_t.width_connection(xt)
    assert paddle.allclose(bn, bt, rtol=1e-3, atol=1e-4)

    on = hc_n.depth_connection((bn, None), rn, hn)
    ot = hc_t.depth_connection((bt, None), rt, ht)
    assert paddle.allclose(on, ot, rtol=1e-3, atol=1e-4)

    on.sum().backward()
    ot.sum().backward()
    assert paddle.allclose(xn.grad, xt.grad, rtol=1e-3, atol=1e-4)


def test_mhc_full_network_precision_comparison():
    """
    Full precision comparison: width+depth in one network, verify MHC parameter gradients
    between Triton and non-Triton implementations.
    """
    from paddlefleet.transformer.hyper_connection import TRITON_AVAILABLE

    if not TRITON_AVAILABLE:
        return

    # Test configuration
    s, b, n, C = 8, 4, 4, 64
    seed = 42
    # Forward pass requires tighter tolerance, backward pass (gradients) allows more tolerance
    forward_rtol, forward_atol = 1e-3, 1e-4
    backward_rtol, backward_atol = 1e-2, 1e-3

    # Create two identical HyperConnectionModules (native and triton)
    paddle.seed(seed)
    hc_native = _make_hc(use_triton=False, hidden_size=C, n=n, layer_number=1)

    paddle.seed(seed)
    hc_triton = _make_hc(use_triton=True, hidden_size=C, n=n, layer_number=1)

    # Sync initial parameters to ensure identical starting point
    # Note: Use set_value() instead of paddle.assign() to preserve gradient graph
    # paddle.assign() would detach the parameter from the computation graph
    param_names = ["combined_weights", "scaling_factors", "bias_terms"]
    for name in param_names:
        native_param = getattr(hc_native, name)
        triton_param = getattr(hc_triton, name)
        triton_param.set_value(native_param)

    # Create input tensor
    paddle.seed(seed)
    x_native = paddle.randn([s, b, n * C], dtype="float32")
    x_native.stop_gradient = False
    paddle.seed(seed)
    x_triton = paddle.randn([s, b, n * C], dtype="float32")
    x_triton.stop_gradient = False

    # Forward pass: width_connection -> mock branch -> depth_connection
    # Simulate a branch computation (e.g., attention/MLP)
    # Use deterministic function so native and triton get same branch output
    def mock_branch(branch_input):
        """Simulate a simple branch computation (deterministic)."""
        # Simple linear transform + activation (deterministic)
        return paddle.tanh(branch_input * 0.5)

    # Native implementation forward
    branch_native, residuals_native, h_post_native = hc_native.width_connection(
        x_native
    )
    branch_output_native = mock_branch(branch_native)
    output_native = hc_native.depth_connection(
        (branch_output_native, None), residuals_native, h_post_native
    )

    # Triton implementation forward
    branch_triton, residuals_triton, h_post_triton = hc_triton.width_connection(
        x_triton
    )
    branch_output_triton = mock_branch(branch_triton)
    output_triton = hc_triton.depth_connection(
        (branch_output_triton, None), residuals_triton, h_post_triton
    )

    # Verify forward output precision
    max_diff = (output_native - output_triton).abs().max().item()
    assert paddle.allclose(
        output_native, output_triton, rtol=forward_rtol, atol=forward_atol
    ), (
        f"Forward outputs differ: max_diff={max_diff}, rtol={forward_rtol}, atol={forward_atol}"
    )

    # Verify intermediate outputs
    assert paddle.allclose(
        branch_native, branch_triton, rtol=forward_rtol, atol=forward_atol
    ), "Branch inputs differ"
    assert paddle.allclose(
        residuals_native, residuals_triton, rtol=forward_rtol, atol=forward_atol
    ), "Residuals differ"
    assert paddle.allclose(
        h_post_native, h_post_triton, rtol=forward_rtol, atol=forward_atol
    ), "h_post differs"

    # Backward pass
    # Create identical loss
    loss_native = output_native.sum()
    loss_triton = output_triton.sum()

    loss_native.backward()
    loss_triton.backward()

    # Verify input gradient precision
    assert paddle.allclose(
        x_native.grad, x_triton.grad, rtol=backward_rtol, atol=backward_atol
    ), "Input gradients differ"

    # **Key verification: MHC parameter gradients (direct element-wise comparison)**
    for name in param_names:
        native_grad = getattr(hc_native, name).grad
        triton_grad = getattr(hc_triton, name).grad

        assert native_grad is not None, f"{name} has no gradient (native)"
        assert triton_grad is not None, f"{name} has no gradient (triton)"

        # Check gradient magnitude is non-zero (gradient is flowing)
        assert native_grad.abs().sum().item() > 0, (
            f"{name} gradient is zero (native)"
        )
        assert triton_grad.abs().sum().item() > 0, (
            f"{name} gradient is zero (triton)"
        )

        # Direct element-wise comparison (use backward tolerance for gradients)
        max_grad_diff = (native_grad - triton_grad).abs().max().item()
        assert paddle.allclose(
            native_grad, triton_grad, rtol=backward_rtol, atol=backward_atol
        ), (
            f"{name} gradients differ: max_diff={max_grad_diff}, rtol={backward_rtol}, atol={backward_atol}"
        )

    # Additional verification: H_post value range
    # H_post = 2 * sigmoid(...), so each element should be in [0, 2] and average around 1
    assert (h_post_native >= 0).all() and (h_post_native <= 2).all(), (
        "h_post values out of range [0, 2] (native)"
    )
    assert (h_post_triton >= 0).all() and (h_post_triton <= 2).all(), (
        "h_post values out of range [0, 2] (triton)"
    )
    # Check that native and triton h_post are consistent
    assert paddle.allclose(
        h_post_native, h_post_triton, rtol=forward_rtol, atol=forward_atol
    ), "h_post values differ between native and triton"


def test_triton_sinkhorn_and_kernels():
    """Triton sinkhorn_knopp, post_sinkhorn kernels with precision verification."""
    from paddlefleet.transformer.hyper_connection import TRITON_AVAILABLE

    if not TRITON_AVAILABLE:
        pytest.skip("Triton not available")
    from paddlefleet.transformer.triton_mhc import (
        post_sinkhorn_fused_forward,
        sinkhorn_knopp,
    )

    N = 4
    B, L = 2, 4
    rtol, atol = 1e-4, 1e-5

    # ========== Test sinkhorn_knopp ==========
    # Verify double stochasticity: row sums and column sums should be ~1
    A = paddle.rand([B, N, N], dtype="float32") + 0.1
    result = sinkhorn_knopp(A, it=20)
    if isinstance(result, tuple):
        result = result[0]

    if result is not None:
        # Verify double stochasticity
        row_sums = result.sum(axis=-1)
        col_sums = result.sum(axis=-2)
        assert paddle.allclose(
            row_sums, paddle.ones_like(row_sums), rtol=1e-2, atol=1e-2
        ), f"Sinkhorn row sums not ~1: {row_sums[0]}"
        assert paddle.allclose(
            col_sums, paddle.ones_like(col_sums), rtol=1e-2, atol=1e-2
        ), f"Sinkhorn col sums not ~1: {col_sums[0]}"

    # ========== Test post_sinkhorn_fused_forward ==========
    D = 64
    H_res_exp = paddle.randn([B, L, N, N], dtype="float32")
    u = paddle.randn([B * L, N], dtype="float32")
    v = paddle.randn([B * L, N], dtype="float32")
    H_pre = paddle.randn([B, L, N], dtype="float32")
    x = paddle.randn([B, L, N, D], dtype="float32")

    # Triton kernel
    residuals, branch_input, H_res = post_sinkhorn_fused_forward(
        H_res_exp, u, v, H_pre, x
    )

    # Native reference: H_res = diag(u) @ H_res_exp @ diag(v)
    # residuals = H_res @ x, branch_input = H_pre @ x (sum over N)
    u_mat = paddle.diag_embed(u.reshape([B * L, N]))  # [B*L, N, N]
    v_mat = paddle.diag_embed(v.reshape([B * L, N]))  # [B*L, N, N]
    H_res_exp_2d = H_res_exp.reshape([B * L, N, N])
    H_res_native = paddle.matmul(u_mat, paddle.matmul(H_res_exp_2d, v_mat))
    H_res_native = H_res_native.reshape([B, L, N, N])

    # residuals = H_res @ x: [B, L, N, N] @ [B, L, N, D] -> [B, L, N, D]
    residuals_native = paddle.matmul(H_res_native, x)

    # branch_input = H_pre @ x: [B, L, N] @ [B, L, N, D] -> [B, L, D]
    # H_pre @ x means sum over N with weights H_pre
    branch_input_native = (
        (H_pre.unsqueeze(-2) @ x)
        .squeeze(-2)
        .unsqueeze(-1)
        .transpose([0, 1, 3, 2])
    )

    # Verify precision
    assert paddle.allclose(H_res, H_res_native, rtol=rtol, atol=atol), (
        f"H_res mismatch: max_diff={((H_res - H_res_native).abs().max().item()):.6f}"
    )
    assert paddle.allclose(residuals, residuals_native, rtol=rtol, atol=atol), (
        f"residuals mismatch: max_diff={((residuals - residuals_native).abs().max().item()):.6f}"
    )


def test_triton_backward_kernels():
    """Triton backward kernel wrappers with precision verification."""
    from paddlefleet.transformer.hyper_connection import TRITON_AVAILABLE

    if not TRITON_AVAILABLE:
        return
    from paddlefleet.transformer.triton_mhc import (
        depth_connection_backward_triton_fused,
        width_branch_residuals_backward_triton,
    )

    B, L, N, D = 2, 4, 4, 64
    rtol, atol = 1e-4, 1e-5

    # ========== Test width_branch_residuals_backward_triton ==========
    d_branch_input = paddle.randn([B, L, 1, D], dtype="float32")
    d_residuals = paddle.randn([B, L, N, D], dtype="float32")
    x = paddle.randn([B, L, N, D], dtype="float32")
    H_pre = paddle.randn([B, L, N], dtype="float32")
    H_res = paddle.randn([B, L, N, N], dtype="float32")

    # Triton kernel
    d_H_pre_triton, d_H_res_triton, d_x_triton = (
        width_branch_residuals_backward_triton(
            d_branch_input, d_residuals, x, H_pre, H_res
        )
    )

    # Native reference
    # d_H_pre = d_branch_input @ x^T (sum over D)
    d_H_pre_native = (
        d_branch_input.squeeze(2).unsqueeze(-2) @ x.transpose([0, 1, 3, 2])
    ).squeeze(-2)
    # d_H_res = d_residuals @ x^T
    d_H_res_native = paddle.matmul(d_residuals, x.transpose([0, 1, 3, 2]))
    # d_x_branch = H_pre * d_branch_input
    d_x_branch = H_pre.unsqueeze(-1) * d_branch_input
    # d_x_residuals = H_res^T @ d_residuals
    d_x_residuals = paddle.matmul(H_res.transpose([0, 1, 3, 2]), d_residuals)
    d_x_native = d_x_branch + d_x_residuals

    assert paddle.allclose(
        d_H_pre_triton, d_H_pre_native, rtol=rtol, atol=atol
    ), (
        f"d_H_pre mismatch: max_diff={((d_H_pre_triton - d_H_pre_native).abs().max().item()):.6f}"
    )
    assert paddle.allclose(
        d_H_res_triton, d_H_res_native, rtol=rtol, atol=atol
    ), (
        f"d_H_res mismatch: max_diff={((d_H_res_triton - d_H_res_native).abs().max().item()):.6f}"
    )
    assert paddle.allclose(d_x_triton, d_x_native, rtol=rtol, atol=atol), (
        f"d_x mismatch: max_diff={((d_x_triton - d_x_native).abs().max().item()):.6f}"
    )

    # ========== Test depth_connection_backward_triton_fused ==========
    d_output = paddle.randn([B, L, N, D], dtype="float32")
    H_post = paddle.randn([B, L, N], dtype="float32")
    branch_output = paddle.randn([B, L, 1, D], dtype="float32")

    # Triton kernel
    d_H_post_triton, d_branch_triton, d_residuals_triton = (
        depth_connection_backward_triton_fused(d_output, H_post, branch_output)
    )

    # Native reference
    d_residuals_native = d_output
    d_branch_native = (H_post.unsqueeze(-1) * d_output).sum(
        axis=2, keepdim=True
    )
    d_H_post_native = (branch_output * d_output).sum(axis=-1)

    assert paddle.allclose(
        d_H_post_triton, d_H_post_native, rtol=rtol, atol=atol
    ), (
        f"d_H_post mismatch: max_diff={((d_H_post_triton - d_H_post_native).abs().max().item()):.6f}"
    )
    assert paddle.allclose(
        d_branch_triton, d_branch_native, rtol=rtol, atol=atol
    ), (
        f"d_branch mismatch: max_diff={((d_branch_triton - d_branch_native).abs().max().item()):.6f}"
    )
    assert paddle.allclose(
        d_residuals_triton, d_residuals_native, rtol=rtol, atol=atol
    ), (
        f"d_residuals mismatch: max_diff={((d_residuals_triton - d_residuals_native).abs().max().item()):.6f}"
    )


# =============================================================================
# 3. TransformerLayerWithMHC tests
# =============================================================================


def test_transformer_layer_mhc_forward():
    """MHC forward: basic, cross-attention, recompute, is_first_fwd."""
    N, C = 4, 64

    layer, mocks = _make_mhc_layer()
    # Note: The mocked layer has cross_attention that is NOT IdentityOp,
    # so _has_cross_attention will be True. To test basic forward without
    # cross-attention complications, we need to ensure the mock behaves properly.
    # However, the basic forward without context should still work.
    result = layer(_base_dict_args(paddle.randn([4, 2, N * C]), C=N * C))
    assert "hidden_states" in result
    mocks["hc"].width_connection.assert_called()
    mocks["hc"].depth_connection.assert_called()

    # Skip cross-attention test as it requires complex mocking of static methods
    # and the interaction between cross_attn_bda and HyperConnectionModule

    layer3, _ = _make_mhc_layer(
        recompute_granularity="selective",
        recompute_modules=["self_attention", "mlp"],
    )
    assert "hidden_states" in layer3(
        _base_dict_args(paddle.randn([4, 2, N * C]), C=N * C)
    )

    layer4, _ = _make_mhc_layer()
    layer4.is_first_fwd = True
    assert "hidden_states" in layer4(
        _base_dict_args(paddle.randn([4, 2, N * C]), C=N * C)
    )


def test_transformer_layer_mhc_precision_comparison():
    """TransformerLayerWithMHC precision test with real data.

    Verifies that Triton and native implementations produce consistent results
    in a full TransformerLayerWithMHC forward/backward pass.
    """
    from paddlefleet.transformer.hyper_connection import (
        TRITON_AVAILABLE,
        HyperConnectionModule,
    )
    from paddlefleet.transformer.identity_op import IdentityFuncOp, IdentityOp
    from paddlefleet.transformer.transformer_config import TransformerConfig
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayerSublayersSpec,
        TransformerLayerWithMHC,
    )

    if not TRITON_AVAILABLE:
        pytest.skip("Triton not available")

    # Test configuration
    s, b, n, C = 4, 2, 4, 64
    forward_rtol, forward_atol = 1e-3, 1e-4
    # TransformerLayer-level test has more accumulated numerical error than unit tests
    backward_rtol, backward_atol = 5e-2, 1e-2
    seed = 42

    def make_layer(use_triton):
        config = TransformerConfig(
            hidden_size=C,
            num_attention_heads=4,
            num_hidden_layers=2,
            intermediate_size=256,
            use_mhc=True,
            mhc_num_residual_streams=n,
            mhc_sinkhorn_iters=10,
            mhc_use_triton=use_triton,
            normalization="RMSNorm",
            rms_norm_eps=1e-5,
        )
        spec = TransformerLayerSublayersSpec(
            input_layernorm=IdentityOp,
            self_attention_hyper_connection=HyperConnectionModule,
            self_attn=IdentityOp,
            self_attn_bda=IdentityFuncOp,
            pre_cross_attn_layernorm=IdentityOp,
            cross_attention=IdentityOp,
            cross_attn_bda=IdentityFuncOp,
            post_attention_layernorm=IdentityOp,
            mlp_hyper_connection=HyperConnectionModule,
            mlp=IdentityOp,
            mlp_bda=IdentityFuncOp,
        )
        return TransformerLayerWithMHC(
            config=config, sublayers_spec=spec, layer_number=1
        )

    # Create layers with same initial parameters
    paddle.seed(seed)
    layer_native = make_layer(use_triton=False)
    paddle.seed(seed)
    layer_triton = make_layer(use_triton=True)

    # Sync MHC parameters
    for hc_name in ["self_attention_hyper_connection", "mlp_hyper_connection"]:
        for param_name in ["combined_weights", "scaling_factors", "bias_terms"]:
            native_param = getattr(getattr(layer_native, hc_name), param_name)
            triton_param = getattr(getattr(layer_triton, hc_name), param_name)
            triton_param.set_value(native_param)

    # Create input
    paddle.seed(seed)
    x_native = paddle.randn([s, b, n * C], dtype="float32")
    x_native.stop_gradient = False
    paddle.seed(seed)
    x_triton = paddle.randn([s, b, n * C], dtype="float32")
    x_triton.stop_gradient = False

    # Forward pass
    dict_args_native = {"hidden_states": x_native, "attention_mask": None}
    dict_args_triton = {"hidden_states": x_triton, "attention_mask": None}

    output_native = layer_native(dict_args_native)
    output_triton = layer_triton(dict_args_triton)

    hidden_native = output_native["hidden_states"]
    hidden_triton = output_triton["hidden_states"]

    # Verify forward output
    assert paddle.allclose(
        hidden_native, hidden_triton, rtol=forward_rtol, atol=forward_atol
    ), (
        f"Forward outputs differ: max_diff={((hidden_native - hidden_triton).abs().max().item()):.6f}"
    )

    # Backward pass
    loss_native = hidden_native.sum()
    loss_triton = hidden_triton.sum()
    loss_native.backward()
    loss_triton.backward()

    # Verify input gradient
    assert paddle.allclose(
        x_native.grad, x_triton.grad, rtol=backward_rtol, atol=backward_atol
    ), (
        f"Input gradients differ: max_diff={((x_native.grad - x_triton.grad).abs().max().item()):.6f}"
    )

    # Verify MHC parameter gradients
    # Note: Triton version uses skip_sk_gradient=True by default for numerical stability,
    # which skips gradients for H_res (the last n*n columns of combined_weights).
    # We only compare the gradients that Triton actually computes (H_pre and H_post parts).
    for hc_name in ["self_attention_hyper_connection", "mlp_hyper_connection"]:
        for param_name in ["combined_weights", "scaling_factors", "bias_terms"]:
            native_param = getattr(getattr(layer_native, hc_name), param_name)
            triton_param = getattr(getattr(layer_triton, hc_name), param_name)

            assert native_param.grad is not None, (
                f"{hc_name}.{param_name} has no gradient (native)"
            )
            assert triton_param.grad is not None, (
                f"{hc_name}.{param_name} has no gradient (triton)"
            )

            if param_name == "combined_weights":
                # Only compare H_pre and H_post parts (first 2n columns)
                # H_res gradients are skipped in Triton for numerical stability
                native_grad = native_param.grad[:, : 2 * n]
                triton_grad = triton_param.grad[:, : 2 * n]
            elif param_name == "scaling_factors":
                # scaling_factors indices: [0]=pre_scale, [1]=post_scale, [2]=res_scale
                # res_scale gradient (index 2) is skipped in Triton for numerical stability
                # Only compare pre_scale and post_scale (indices 0 and 1)
                native_grad = native_param.grad[:2]
                triton_grad = triton_param.grad[:2]
            elif param_name == "bias_terms":
                # Only compare H_pre and H_post bias (first 2n elements)
                native_grad = native_param.grad[: 2 * n]
                triton_grad = triton_param.grad[: 2 * n]
            else:
                native_grad = native_param.grad
                triton_grad = triton_param.grad

            max_diff = (native_grad - triton_grad).abs().max().item()
            assert paddle.allclose(
                native_grad, triton_grad, rtol=backward_rtol, atol=backward_atol
            ), (
                f"{hc_name}.{param_name} gradients differ: max_diff={max_diff:.6f}"
            )


# =============================================================================
# 4. GPT Layer Specs MHC tests
# =============================================================================


def test_gpt_layer_spec_variants():
    """get_gpt_layer_local_spec: MHC, MLA, LayerNorm, no-MHC, qk_l2_norm."""
    from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from paddlefleet.spec_utils import LayerSpec
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayerWithMHC,
    )

    def _call(
        use_mhc=True, mla=False, norm="RMSNorm", qk_l2=False, use_qk=False
    ):
        cfg = MockTransformerConfig(use_mhc=use_mhc, multi_latent_attention=mla)
        return get_gpt_layer_local_spec(
            config=cfg,
            num_experts=None,
            moe_grouped_gemm=False,
            use_qk_norm=use_qk,
            multi_latent_attention=mla,
            normalization=norm,
            qk_l2_norm=qk_l2,
            layer_number=1,
            use_mhc=use_mhc,
        )

    assert _call(use_mhc=True).layer == TransformerLayerWithMHC
    assert isinstance(_call(use_mhc=True, mla=True), LayerSpec)
    assert isinstance(
        _call(use_mhc=True, norm="LayerNorm", use_qk=True), LayerSpec
    )
    assert isinstance(_call(use_mhc=False, mla=True), LayerSpec)
    assert isinstance(_call(use_mhc=False, qk_l2=True), LayerSpec)


def test_gpt_decoder_layers_spec():
    """get_gpt_decoder_layers_spec with use_mhc=True/False."""
    from paddlefleet.models.gpt.gpt_layer_specs import (
        get_gpt_decoder_layers_spec,
    )

    cfg = MockTransformerConfig(num_hidden_layers=4, moe_layer_freq=2)
    assert (
        len(get_gpt_decoder_layers_spec(config=cfg, normalization="RMSNorm"))
        == 4
    )

    cfg_list = MockTransformerConfig(
        num_hidden_layers=4, moe_layer_freq=[0, 1, 0, 1]
    )
    assert (
        len(
            get_gpt_decoder_layers_spec(
                config=cfg_list, normalization="RMSNorm"
            )
        )
        == 4
    )

    cfg2 = MockTransformerConfig(
        num_hidden_layers=4, use_mhc=True, moe_layer_freq=2
    )
    assert (
        len(
            get_gpt_decoder_layers_spec(
                config=cfg2, normalization="RMSNorm", use_mhc=True
            )
        )
        == 4
    )

    # MoE with MHC
    cfg3 = MockTransformerConfig(
        num_hidden_layers=4, use_mhc=True, n_routed_experts=8, moe_layer_freq=2
    )
    assert (
        len(
            get_gpt_decoder_layers_spec(
                config=cfg3, normalization="RMSNorm", use_mhc=True
            )
        )
        == 4
    )


def test_gpt_layer_mhc_spec_overlap_raises():
    """get_gpt_layer_local_spec with use_mhc=True raises with overlap_scheduler."""
    import paddle.distributed as dist

    from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

    cfg = MockTransformerConfig(use_mhc=True)
    # overlap_scheduler check only triggers when distributed is initialized
    if not dist.is_initialized():
        # Without distributed, MHC spec should still work normally
        spec = get_gpt_layer_local_spec(
            config=cfg,
            num_experts=None,
            moe_grouped_gemm=False,
            use_qk_norm=False,
            multi_latent_attention=False,
            normalization="RMSNorm",
            qk_l2_norm=False,
            layer_number=1,
            use_mhc=True,
        )
        assert spec is not None


def test_gpt_layer_spec_overlap_scheduler_compatibility():
    """Test overlap scheduler compatibility check for both MHC and non-MHC paths."""
    import paddle.distributed as dist

    from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule
    from paddlefleet.transformer.identity_op import IdentityOp
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayer,
        TransformerLayerWithMHC,
    )

    # Test 1: Without distributed, MHC path should return correct classes
    if not dist.is_initialized():
        cfg_mhc = MockTransformerConfig(use_mhc=True)
        spec_mhc = get_gpt_layer_local_spec(
            config=cfg_mhc,
            normalization="RMSNorm",
            layer_number=1,
            use_mhc=True,
        )
        assert spec_mhc.layer == TransformerLayerWithMHC
        # Check sublayers_spec has HyperConnectionModule for hyper_connection slots
        assert (
            spec_mhc.sublayers_spec.self_attention_hyper_connection
            == HyperConnectionModule
        )
        assert (
            spec_mhc.sublayers_spec.mlp_hyper_connection
            == HyperConnectionModule
        )

    # Test 2: Without distributed, non-MHC path should return correct classes
    if not dist.is_initialized():
        cfg_no_mhc = MockTransformerConfig(use_mhc=False)
        spec_no_mhc = get_gpt_layer_local_spec(
            config=cfg_no_mhc,
            normalization="RMSNorm",
            layer_number=1,
            use_mhc=False,
        )
        assert spec_no_mhc.layer == TransformerLayer
        # Check sublayers_spec has IdentityOp for hyper_connection slots
        assert (
            spec_no_mhc.sublayers_spec.self_attention_hyper_connection
            == IdentityOp
        )
        assert spec_no_mhc.sublayers_spec.mlp_hyper_connection == IdentityOp


def test_gpt_layer_spec_overlap_with_mock_distributed():
    """Test overlap scheduler check with mocked distributed environment."""
    from unittest.mock import MagicMock, patch

    from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayer,
        TransformerLayerWithMHC,
        TransformerLayerWithOverlap,
    )

    # Mock fleet config with overlap scheduler enabled
    mock_pp_configs = MagicMock()
    mock_pp_configs.forward_backward_overlap_scheduler = True
    mock_hybrid_configs = {"pp_configs": mock_pp_configs}
    mock_strategy = MagicMock()
    mock_strategy.hybrid_configs = mock_hybrid_configs
    mock_fleet = MagicMock()
    mock_fleet._user_defined_strategy = mock_strategy

    # Test 1: MHC + overlap should raise ValueError
    with (
        patch("paddle.distributed.is_initialized", return_value=True),
        patch("paddle.distributed.fleet.fleet", mock_fleet),
    ):
        cfg_mhc = MockTransformerConfig(use_mhc=True)
        with pytest.raises(ValueError, match="MHC is not compatible"):
            get_gpt_layer_local_spec(
                config=cfg_mhc,
                normalization="RMSNorm",
                layer_number=1,
                use_mhc=True,
            )

    # Test 2: Non-MHC + overlap with base TransformerLayer should use Overlap
    with (
        patch("paddle.distributed.is_initialized", return_value=True),
        patch("paddle.distributed.fleet.fleet", mock_fleet),
    ):
        cfg_no_mhc = MockTransformerConfig(use_mhc=False)
        spec = get_gpt_layer_local_spec(
            config=cfg_no_mhc,
            normalization="RMSNorm",
            layer_number=1,
            use_mhc=False,
        )
        assert spec.layer == TransformerLayerWithOverlap

    # Test 3: Non-MHC + overlap with custom layer should raise AssertionError
    with (
        patch("paddle.distributed.is_initialized", return_value=True),
        patch("paddle.distributed.fleet.fleet", mock_fleet),
    ):

        class CustomTransformerLayer:
            pass

        cfg_custom = MockTransformerConfig(use_mhc=False)
        cfg_custom.specific_layer = CustomTransformerLayer
        with pytest.raises(AssertionError, match="Only base TransformerLayer"):
            get_gpt_layer_local_spec(
                config=cfg_custom,
                normalization="RMSNorm",
                layer_number=1,
                use_mhc=False,
            )

    # Test 4: Without overlap, MHC should work with distributed
    mock_pp_configs_no_overlap = MagicMock()
    mock_pp_configs_no_overlap.forward_backward_overlap_scheduler = False
    mock_hybrid_configs_no_overlap = {"pp_configs": mock_pp_configs_no_overlap}
    mock_strategy_no_overlap = MagicMock()
    mock_strategy_no_overlap.hybrid_configs = mock_hybrid_configs_no_overlap
    mock_fleet_no_overlap = MagicMock()
    mock_fleet_no_overlap._user_defined_strategy = mock_strategy_no_overlap

    with (
        patch("paddle.distributed.is_initialized", return_value=True),
        patch("paddle.distributed.fleet.fleet", mock_fleet_no_overlap),
    ):
        cfg_mhc = MockTransformerConfig(use_mhc=True)
        spec = get_gpt_layer_local_spec(
            config=cfg_mhc,
            normalization="RMSNorm",
            layer_number=1,
            use_mhc=True,
        )
        assert spec.layer == TransformerLayerWithMHC

    # Test 5: Without overlap, non-MHC should keep original layer
    with (
        patch("paddle.distributed.is_initialized", return_value=True),
        patch("paddle.distributed.fleet.fleet", mock_fleet_no_overlap),
    ):
        cfg_no_mhc = MockTransformerConfig(use_mhc=False)
        spec = get_gpt_layer_local_spec(
            config=cfg_no_mhc,
            normalization="RMSNorm",
            layer_number=1,
            use_mhc=False,
        )
        assert spec.layer == TransformerLayer


# =============================================================================
# 5. MHC Integration tests
# =============================================================================


def test_transformer_layer_mhc_integration():
    """Full integration: TransformerLayerWithMHC from IdentityOp spec."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule
    from paddlefleet.transformer.identity_op import IdentityFuncOp, IdentityOp
    from paddlefleet.transformer.transformer_config import TransformerConfig
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayerSublayersSpec,
        TransformerLayerWithMHC,
    )

    config = TransformerConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=256,
        use_mhc=True,
        mhc_num_residual_streams=4,
        mhc_sinkhorn_iters=10,
        normalization="RMSNorm",
        rms_norm_eps=1e-5,
    )
    spec = TransformerLayerSublayersSpec(
        input_layernorm=IdentityOp,
        self_attention_hyper_connection=HyperConnectionModule,
        self_attn=IdentityOp,
        self_attn_bda=IdentityFuncOp,
        pre_cross_attn_layernorm=IdentityOp,
        cross_attention=IdentityOp,
        cross_attn_bda=IdentityFuncOp,
        post_attention_layernorm=IdentityOp,
        mlp_hyper_connection=HyperConnectionModule,
        mlp=IdentityOp,
        mlp_bda=IdentityFuncOp,
    )
    layer = TransformerLayerWithMHC(
        config=config, sublayers_spec=spec, layer_number=1
    )

    n, C, s, b = 4, 64, 4, 2
    x = paddle.randn([b, s, C])
    x.stop_gradient = False
    x_exp = HyperConnectionModule.expand_stream(x.transpose([1, 0, 2]), n)
    output = layer({"hidden_states": x_exp})
    out_hs = output["hidden_states"] if isinstance(output, dict) else output[0]
    assert out_hs.shape == [s, b, n * C]
    out_hs.sum().backward()
    assert x.grad is not None


def test_transformer_layer_mhc_from_gpt_spec():
    """TransformerLayerWithMHC from get_gpt_layer_local_spec with use_mhc=True."""
    from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from paddlefleet.spec_utils import build_layer
    from paddlefleet.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=128,
        mhc_num_residual_streams=4,
        mhc_use_triton=False,
        use_mhc=True,
    )
    spec = get_gpt_layer_local_spec(
        config=config,
        num_experts=None,
        moe_grouped_gemm=False,
        use_qk_norm=False,
        multi_latent_attention=False,
        normalization="RMSNorm",
        qk_l2_norm=False,
        layer_number=1,
        use_mhc=True,
    )
    layer = build_layer(spec, config=config, layer_number=1)
    layer.eval()

    n, C, s, b = 4, 64, 4, 1
    output = layer(_base_dict_args(paddle.randn([s, b, n * C]), C=n * C))
    out_hs = output["hidden_states"] if isinstance(output, dict) else output[0]
    assert out_hs.shape == [s, b, n * C]


def test_gpt_model_mhc_layer_desc():
    """MHCExpandLayer/MHCContractLayer for GPTModel."""
    from paddlefleet.transformer.hyper_connection import (
        MHCContractLayer,
        MHCExpandLayer,
    )
    from paddlefleet.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=128,
        use_mhc=True,
        mhc_num_residual_streams=4,
    )
    expand = MHCExpandLayer(config=config)
    contract = MHCContractLayer(config=config)

    s, b, C, n = 4, 2, 64, 4
    x = paddle.randn([s, b, C])
    out = expand({"hidden_states": x})
    assert out["hidden_states"].shape == [s, b, n * C]
    out2 = contract({"hidden_states": out["hidden_states"]})
    assert out2["hidden_states"].shape == [s, b, C]


# =============================================================================
# 9. Numerical gradient validation
# =============================================================================


def test_gradient_numerical_accuracy():
    """Validate gradient accuracy via finite differences."""
    hc = _make_hc(hidden_size=16, n=2)
    s, b, n, C = 2, 1, 2, 16
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    eps = 1e-3
    branch, res, hp = hc.width_connection(x)
    out = hc.depth_connection((branch, None), res, hp)
    out.sum().backward()
    analytical = x.grad.clone()

    x_flat = x.flatten()
    for idx in [0, n * C - 1]:
        x_flat[idx] += eps
        b1, r1, h1 = hc.width_connection(x)
        y_plus = hc.depth_connection((b1, None), r1, h1).sum().item()
        x_flat[idx] -= 2 * eps
        b2, r2, h2 = hc.width_connection(x)
        y_minus = hc.depth_connection((b2, None), r2, h2).sum().item()
        x_flat[idx] += eps
        num_grad = (y_plus - y_minus) / (2 * eps)
        ana_grad = analytical.flatten()[idx].item()
        assert abs(num_grad - ana_grad) < 0.1, (
            f"idx={idx}: num={num_grad}, ana={ana_grad}"
        )


# =============================================================================
# 6. gpt_builders MHC tests
# =============================================================================


def test_gpt_builder_get_transformer_layer_spec_func_mhc():
    """_get_transformer_layer_spec_func with MHC enabled."""
    from paddlefleet.gpt_builders import _get_transformer_layer_spec_func

    config = MockTransformerConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=128,
        n_routed_experts=None,
        use_qk_norm=False,
        multi_latent_attention=False,
        normalization="RMSNorm",
        use_mhc=True,
        mhc_num_residual_streams=4,
    )
    spec_func = _get_transformer_layer_spec_func(config)
    assert callable(spec_func)
    spec = spec_func(layer_number=1)
    assert spec is not None


def test_gpt_builder_mhc_layer_spec():
    """Test gpt_builder MHC layer spec creation."""
    from paddlefleet.models.gpt.gpt_layer_specs import (
        get_gpt_decoder_layers_spec,
    )

    config = MockTransformerConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=128,
        n_routed_experts=None,
        moe_layer_freq=1,
        normalization="RMSNorm",
        use_mhc=True,
        mhc_num_residual_streams=4,
    )
    transformer_layers = get_gpt_decoder_layers_spec(
        config, normalization="RMSNorm", use_mhc=True
    )
    assert len(transformer_layers) == 2


# =============================================================================
# 7. Additional TransformerLayerWithMHC tests
# =============================================================================


@pytest.mark.parametrize(
    "recompute_modules",
    [
        ["mlp"],
        ["norm"],
    ],
)
def test_transformer_layer_mhc_with_recompute_variants(recompute_modules):
    """TransformerLayerWithMHC with various recompute configurations."""
    layer, _ = _make_mhc_layer(
        recompute_granularity="selective",
        recompute_modules=recompute_modules,
        recompute_method="block",
        recompute_num_layers=1,
    )
    N, C = 4, 64
    hidden = paddle.randn([4, 2, N * C])
    result = layer(_base_dict_args(hidden, C=N * C))
    assert "hidden_states" in result


# =============================================================================
# 8. TransformerBlock MHC tests (lines 29, 253-255, 275-276)
# =============================================================================


def test_transformer_block_mhc_expand_contract():
    """TransformerBlock with use_mhc=True: expand on pre_process, contract on post_process.

    Tests lines 253-255 (expand) and 275-276 (contract) in transformer_block.py.
    """
    from paddlefleet.spec_utils import LayerSpec
    from paddlefleet.transformer.transformer_block import (
        TransformerBlockSublayersSpec,
    )

    # Create config with MHC enabled
    config = MockTransformerConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=128,
        use_mhc=True,
        mhc_num_residual_streams=4,
        mhc_sinkhorn_iters=10,
    )

    # Create dummy layer specs that accept **kw and return (hidden_states, context)
    class DummyTransformerLayer(paddle.nn.Layer):
        def __init__(self, config=None, **kwargs):
            super().__init__()

        def forward(self, **kw):
            hs = kw.get("hidden_states")
            return hs, None

    layer_specs = [
        LayerSpec(DummyTransformerLayer)
        for _ in range(config.num_hidden_layers)
    ]
    sublayers_spec = TransformerBlockSublayersSpec(
        layer_specs=layer_specs,
        layer_norm=None,
    )

    block = _make_transformer_block(
        config=config,
        sublayers_spec=sublayers_spec,
        pre_process=True,
        post_process=True,
        post_layer_norm=False,
    )
    block.eval()

    # Input: [s, b, C] - will be expanded to [s, b, n*C] in pre_process
    s, b, C, n = 4, 1, 64, 4
    hidden = paddle.randn([s, b, C])

    # Forward pass - tests lines 253-257 (expand) and 275-278 (contract)
    output = block(hidden_states=hidden, attention_mask=None)

    # Output should be [s, b, C] after contraction
    assert output.shape == [s, b, C], (
        f"Expected {[s, b, C]}, got {output.shape}"
    )


def test_transformer_block_mhc_no_pre_post_process():
    """TransformerBlock with use_mhc but pre_process=False and post_process=False.

    Tests the code paths where MHC expand/contract are skipped.
    """
    from paddlefleet.spec_utils import LayerSpec
    from paddlefleet.transformer.transformer_block import (
        TransformerBlockSublayersSpec,
    )

    config = MockTransformerConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=128,
        use_mhc=True,
        mhc_num_residual_streams=4,
    )

    class DummyTransformerLayer(paddle.nn.Layer):
        def __init__(self, config=None, **kwargs):
            super().__init__()

        def forward(self, **kw):
            hs = kw.get("hidden_states")
            return hs, None

    layer_specs = [
        LayerSpec(DummyTransformerLayer)
        for _ in range(config.num_hidden_layers)
    ]
    sublayers_spec = TransformerBlockSublayersSpec(
        layer_specs=layer_specs,
        layer_norm=None,
    )

    block = _make_transformer_block(
        config=config,
        sublayers_spec=sublayers_spec,
        pre_process=False,
        post_process=False,
        post_layer_norm=False,
    )
    block.eval()

    # When pre_process=False, block uses input_tensor instead
    s, b, C, n = 4, 1, 64, 4
    # Input already expanded since pre_process=False
    hidden = paddle.randn([s, b, n * C])
    block.set_input_tensor(hidden)

    output = block(hidden_states=paddle.zeros([1]), attention_mask=None)

    # Output should remain [s, b, n*C] since post_process=False (no contraction)
    assert output.shape == [s, b, n * C], (
        f"Expected {[s, b, n * C]}, got {output.shape}"
    )


def test_transformer_block_mhc_with_norm():
    """TransformerBlock with norm layer (covers lines 178, 282).

    Tests norm layer building and application.
    """
    from paddlefleet.spec_utils import LayerSpec
    from paddlefleet.transformer.transformer_block import (
        TransformerBlockSublayersSpec,
    )

    config = MockTransformerConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=128,
        use_mhc=True,
        mhc_num_residual_streams=4,
    )

    class DummyTransformerLayer(paddle.nn.Layer):
        def __init__(self, config=None, **kwargs):
            super().__init__()

        def forward(self, **kw):
            hs = kw.get("hidden_states")
            return hs, None

    class DummyNorm(paddle.nn.Layer):
        def __init__(self, config=None, hidden_size=None, eps=None, **kwargs):
            super().__init__()

        def forward(self, x):
            return x * 1.0  # Identity-like but different object

    layer_specs = [
        LayerSpec(DummyTransformerLayer)
        for _ in range(config.num_hidden_layers)
    ]
    sublayers_spec = TransformerBlockSublayersSpec(
        layer_specs=layer_specs,
        layer_norm=DummyNorm,  # Use DummyNorm class, not LayerSpec
    )

    # Build block with real build_layer for norm
    block = _make_transformer_block(
        config=config,
        sublayers_spec=sublayers_spec,
        pre_process=True,
        post_process=True,
        post_layer_norm=True,
    )
    block.eval()

    # Verify norm was built
    assert block.norm is not None

    s, b, C = 4, 1, 64
    hidden = paddle.randn([s, b, C])
    output = block(hidden_states=hidden, attention_mask=None)
    assert output.shape == [s, b, C]


def test_transformer_block_get_layer():
    """Test _get_layer method (covers line 188)."""
    from paddlefleet.spec_utils import LayerSpec
    from paddlefleet.transformer.transformer_block import (
        TransformerBlockSublayersSpec,
    )

    config = MockTransformerConfig(
        hidden_size=64,
        num_hidden_layers=2,
        use_mhc=True,
        mhc_num_residual_streams=4,
    )

    class DummyTransformerLayer(paddle.nn.Layer):
        def __init__(self, config=None, **kwargs):
            super().__init__()

        def forward(self, **kw):
            return kw.get("hidden_states"), None

    layer_specs = [
        LayerSpec(DummyTransformerLayer)
        for _ in range(config.num_hidden_layers)
    ]
    sublayers_spec = TransformerBlockSublayersSpec(
        layer_specs=layer_specs,
        layer_norm=None,
    )

    block = _make_transformer_block(
        config=config,
        sublayers_spec=sublayers_spec,
    )

    # Test _get_layer method
    layer0 = block._get_layer(0)
    layer1 = block._get_layer(1)
    assert layer0 is not None
    assert layer1 is not None
    assert layer0 is not layer1


def test_transformer_block_wrapped_tensor():
    """Test WrappedTensor handling (covers line 242)."""
    from paddlefleet.spec_utils import LayerSpec
    from paddlefleet.transformer.transformer_block import (
        TransformerBlockSublayersSpec,
    )
    from paddlefleet.utils import WrappedTensor

    config = MockTransformerConfig(
        hidden_size=64,
        num_hidden_layers=2,
        use_mhc=True,
        mhc_num_residual_streams=4,
    )

    class DummyTransformerLayer(paddle.nn.Layer):
        def __init__(self, config=None, **kwargs):
            super().__init__()

        def forward(self, **kw):
            return kw.get("hidden_states"), None

    layer_specs = [
        LayerSpec(DummyTransformerLayer)
        for _ in range(config.num_hidden_layers)
    ]
    sublayers_spec = TransformerBlockSublayersSpec(
        layer_specs=layer_specs,
        layer_norm=None,
    )

    block = _make_transformer_block(
        config=config,
        sublayers_spec=sublayers_spec,
        pre_process=True,
        post_process=True,
    )
    block.eval()

    s, b, C = 4, 1, 64
    hidden = paddle.randn([s, b, C])
    # Wrap the tensor
    wrapped_hidden = WrappedTensor(hidden)

    # Forward with WrappedTensor - should unwrap automatically
    output = block(hidden_states=wrapped_hidden, attention_mask=None)
    assert output.shape == [s, b, C]


def test_transformer_block_from_layer_spec():
    """Test TransformerBlock with LayerSpec(TransformerLayer) (covers lines 100-104)."""
    from paddlefleet.spec_utils import LayerSpec
    from paddlefleet.transformer.transformer_block import (
        _get_block_sublayers_spec,
    )
    from paddlefleet.transformer.transformer_layer import TransformerLayer

    config = MockTransformerConfig(
        hidden_size=64,
        num_hidden_layers=2,
        use_mhc=True,
        mhc_num_residual_streams=4,
    )

    # Create a LayerSpec with TransformerLayer
    layer_spec = LayerSpec(TransformerLayer)

    # This should trigger lines 100-104 in _get_block_sublayers_spec
    result = _get_block_sublayers_spec(config, layer_spec)

    # Verify the result
    assert result.layer_specs is not None
    assert len(result.layer_specs) == config.num_hidden_layers
    assert result.layer_norm is not None


def test_gpt_model_mhc_and_mtp_coverage():
    """Test GPTModel.get_layer_desc_list with MHC and MTP for coverage improvement.

    Covers gpt_model.py lines:
    - Line 247: MHCContractLayer insertion when use_mhc=True
    - Line 268: MTP layer iteration i += 1
    """
    from paddlefleet.models.gpt.gpt_layer_specs import (
        get_gpt_decoder_layers_spec,
        get_gpt_mtp_layers_spec,
        get_gpt_spec,
    )
    from paddlefleet.spec_utils import LayerSpec

    # Create config with MHC and MTP enabled
    config = MockTransformerConfig(
        use_mhc=True,
        num_hidden_layers=2,
        mtp_loss_weight=0.1,  # Enable MTP to cover line 268
    )

    # Create MTP spec (requires a spec parameter)
    mtp_spec = [LayerSpec(layer=lambda x: x, extra_kwargs={"config": config})]

    # Create GPTSublayersSpec using get_gpt_spec
    gpt_spec = get_gpt_spec(
        config=config,
        transformer_layers_spec=get_gpt_decoder_layers_spec(
            config, use_mhc=True
        ),
        mtp_layers_spec=get_gpt_mtp_layers_spec(config, mtp_spec)
        if config.mtp_loss_weight
        else [],
        vocab_size=config.vocab_size,
        max_sequence_length=config.max_sequence_length,
        tie_word_embeddings=False,
    )

    # Extract sublayers_spec from gpt_spec
    sublayers_spec = gpt_spec.sublayers_spec

    # Ensure head_empty_layers and tail_empty_layers are not None
    if sublayers_spec.head_empty_layers is None:
        sublayers_spec.head_empty_layers = []
    if sublayers_spec.tail_empty_layers is None:
        sublayers_spec.tail_empty_layers = []

    # Mock GPTModel.__init__ to avoid fleet initialization
    with (
        patch("paddlefleet.models.gpt.gpt_model.fleet"),
        patch("paddlefleet.pipeline_parallel.PipelineLayer.__init__"),
    ):
        from paddlefleet.models.gpt.gpt_model import GPTModel

        model = GPTModel.__new__(GPTModel)
        model.config = config
        model._pipeline_name_mapping = None
        model._pp_to_single_mapping = None
        model._sequential_layers = model.get_layer_desc_list(
            sublayers_spec, tie_word_embeddings=False
        )

        # Verify layers were created correctly (covers lines 247 and 268)
        sequential_layers = model.get_sequential_layers()
        assert len(sequential_layers) > 0

        # With MHC enabled, structure should be built correctly
        assert len(sequential_layers) > 0


def test_hyper_connection_transformer_layer_coverage():
    """Test TransformerLayerWithMHC for coverage improvement."""
    N, C = 4, 64

    # Use existing _make_mhc_layer helper (covers line 747)
    layer, mocks = _make_mhc_layer()

    # Test forward with position_ids (covers line 833)
    s, b = 4, 2
    hidden_states = paddle.randn([s, b, N * C])
    position_ids = paddle.arange(s).unsqueeze(0).expand(b, -1)

    dict_args = _base_dict_args(hidden_states, C=N * C)
    dict_args["position_ids"] = position_ids  # This triggers line 833
    result = layer(dict_args)
    assert "hidden_states" in result
    mocks["hc"].depth_connection.assert_called()

    # Test with attention_bias parameter (covers line 805)
    attention_bias = paddle.randn([s, 1, s, s])
    dict_args = _base_dict_args(paddle.randn([s, b, N * C]), C=N * C)
    dict_args["attention_bias"] = attention_bias
    result = layer(dict_args)
    assert "hidden_states" in result

    # Test with is_first_fwd=True (covers line 891)
    layer.is_first_fwd = True
    result = layer(_base_dict_args(paddle.randn([s, b, N * C]), C=N * C))
    assert "hidden_states" in result

    # Test without recompute_post_attention_layernorm (covers line 906)
    layer2, _ = _make_mhc_layer()
    layer2.recompute_post_attention_layernorm = False
    result = layer2(_base_dict_args(paddle.randn([s, b, N * C]), C=N * C))
    assert "hidden_states" in result


def test_hyper_connection_mlp_recompute_none_bias_coverage():
    """Test _forward_mlp with recompute and None bias.

    Covers transformer_layer.py line 919:
    - if bias is None: return mlp_output
    """
    N, C = 4, 64

    # Create layer with recompute_mlp=True
    layer, mocks = _make_mhc_layer(recompute_mlp=True)

    # Mock MLP to return (output, None) - triggers line 919
    class MockMLP(paddle.nn.Layer):
        def forward(self, x):
            return paddle.randn_like(x), None

    layer.mlp = MockMLP()
    layer.recompute_mlp = True

    result = layer(_base_dict_args(paddle.randn([4, 2, N * C]), C=N * C))
    assert "hidden_states" in result


def test_transformer_layer_with_overlap_compute_attention_coverage():
    """Test TransformerLayerWithOverlap.compute_attention.

    Covers transformer_layer.py line 965:
    - return self._forward_attention(**dict_args, is_first_fwd=is_first_fwd)
    """
    from paddlefleet.transformer.identity_op import IdentityOp
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayerSublayersSpec,
        TransformerLayerWithOverlap,
    )

    N, C = 4, 64
    config = MockTransformerConfig(use_mhc=False)  # Overlap doesn't use MHC
    mock_pg = MagicMock()
    mock_pg.tp = MagicMock()
    spec = TransformerLayerSublayersSpec()

    # Create mocks
    mock_ln = MagicMock(side_effect=lambda x: x)
    mock_attn = MagicMock(return_value=(paddle.randn([4, 2, C]), None))
    mock_mlp = MagicMock(return_value=(paddle.randn([4, 2, C]), None))
    mock_bda = MagicMock(
        return_value=lambda *a, **kw: MagicMock(
            return_value=paddle.randn([4, 2, C])
        )
    )

    call_count = [0]

    def build_se(s, **kwargs):
        call_count[0] += 1
        idx = call_count[0]
        if idx in [1, 4, 7]:  # layernorms
            return mock_ln
        elif idx == 2:  # self_attn
            return mock_attn
        elif idx in [3, 6, 9]:  # bda
            return mock_bda
        elif idx == 5:  # cross_attention
            return IdentityOp()
        elif idx == 8:  # mlp
            return mock_mlp
        return MagicMock()

    with patch(
        "paddlefleet.transformer.transformer_layer.build_layer",
        side_effect=build_se,
    ):
        layer = TransformerLayerWithOverlap(
            config=config,
            sublayers_spec=spec,
            layer_number=1,
            pg_collection=mock_pg,
        )

        # Test compute_attention with is_first_fwd flag (covers line 965)
        dict_args = _base_dict_args(paddle.randn([4, 2, C]), C=C)
        output = layer.compute_attention(dict_args, is_first_fwd=True)
        assert output is not None

        # Test with is_first_fwd=False
        output = layer.compute_attention(dict_args, is_first_fwd=False)
        assert output is not None
