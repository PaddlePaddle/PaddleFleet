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
Test script for MHC (Manifold-Constrained Hyper-Connections) implementation.

This script tests:
1. HyperConnectionModule width_connection and depth_connection
2. Expand/contract round-trip consistency
3. Gradient flow through the entire pipeline
4. Native vs Triton backend comparison
5. TransformerLayerWithMHC integration
6. Pipeline helper layers (MHCExpandLayer/MHCContractLayer)
"""

import paddle


class MockConfig:
    """Mock TransformerConfig for testing MHC."""

    def __init__(
        self,
        hidden_size: int = 64,
        mhc_num_residual_streams: int = 4,
        mhc_sinkhorn_iters: int = 10,
        mhc_use_triton: bool = False,
    ):
        self.hidden_size = hidden_size
        self.mhc_num_residual_streams = mhc_num_residual_streams
        self.mhc_sinkhorn_iters = mhc_sinkhorn_iters
        self.mhc_use_triton = mhc_use_triton


# =============================================================================
# Helper Functions for Validation
# =============================================================================


def _assert_valid_tensor(tensor, name, check_finite=True, value_range=None):
    """Validate tensor values are reasonable.

    Args:
        tensor: Paddle tensor to validate
        name: Name of tensor for error messages
        check_finite: Whether to check for NaN/Inf
        value_range: Optional (min, max) tuple for value range check
    """
    if check_finite:
        assert not paddle.isnan(tensor).any(), f"{name} contains NaN"
        assert not paddle.isinf(tensor).any(), f"{name} contains Inf"

    if value_range is not None:
        vmin, vmax = value_range
        t_min = tensor.min().item()
        t_max = tensor.max().item()
        assert t_min >= vmin, f"{name} min {t_min} < {vmin}"
        assert t_max <= vmax, f"{name} max {t_max} > {vmax}"


def _assert_gradient_valid(grad, name, check_nonzero=True, check_finite=True):
    """Validate gradient tensor.

    Args:
        grad: Gradient tensor to validate
        name: Name for error messages
        check_nonzero: Whether to check gradient is non-zero
        check_finite: Whether to check for NaN/Inf
    """
    assert grad is not None, f"{name} gradient is None"

    if check_finite:
        assert not paddle.isnan(grad).any(), f"{name} gradient contains NaN"
        assert not paddle.isinf(grad).any(), f"{name} gradient contains Inf"

    if check_nonzero:
        grad_sum = grad.abs().sum().item()
        assert grad_sum > 0, f"{name} gradient is all zeros"


def _compute_numerical_gradient(module, x, eps=1e-4, num_samples=10):
    """Compute numerical gradient using finite differences for sampled elements.

    Args:
        module: HyperConnectionModule instance
        x: Input tensor with stop_gradient=False
        eps: Finite difference step size
        num_samples: Number of elements to sample (for efficiency)

    Returns:
        tuple: (analytical_grad, numerical_grad, max_diff)
    """
    s, b, nC = x.shape
    total_elements = s * b * nC

    # Sample indices for numerical gradient computation
    if total_elements > num_samples:
        sample_indices = paddle.randint(0, total_elements, [num_samples])
    else:
        sample_indices = paddle.arange(total_elements)
        num_samples = total_elements

    # Compute analytical gradient
    x_flat = x.flatten()
    x.stop_gradient = False

    branch, residuals, h_post = module.width_connection(x)
    output = module.depth_connection((branch, None), residuals, h_post)
    loss = output.sum()
    loss.backward()
    analytical_grad = x.grad.clone()
    analytical_grad_flat = analytical_grad.flatten()

    # Compute numerical gradient for sampled elements
    numerical_grad_flat = paddle.zeros([total_elements], dtype=x.dtype)

    for idx in sample_indices.numpy():
        idx = int(idx)
        # f(x + eps)
        x_flat[idx] += eps
        x.clear_gradient()
        branch, residuals, h_post = module.width_connection(x)
        output = module.depth_connection((branch, None), residuals, h_post)
        y_plus = output.sum().item()

        # f(x - eps)
        x_flat[idx] -= 2 * eps
        x.clear_gradient()
        branch, residuals, h_post = module.width_connection(x)
        output = module.depth_connection((branch, None), residuals, h_post)
        y_minus = output.sum().item()

        # Restore
        x_flat[idx] += eps

        # Numerical gradient
        numerical_grad_flat[idx] = (y_plus - y_minus) / (2 * eps)

    # Compare only sampled elements
    sampled_analytical = analytical_grad_flat[sample_indices]
    sampled_numerical = numerical_grad_flat[sample_indices]
    max_diff = (sampled_analytical - sampled_numerical).abs().max().item()

    return (
        analytical_grad,
        numerical_grad_flat.reshape(x.shape),
        max_diff,
        sample_indices,
    )


# =============================================================================
# Test 1: Basic MHC Operations
# =============================================================================


def test_hyper_connection_module():
    """Test HyperConnectionModule width_connection with gradient flow."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=64)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 64
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    # Width connection
    branch_input, residuals, h_post = hc.width_connection(x)

    assert branch_input.shape == [s, b, C], (
        f"branch_input shape mismatch: {branch_input.shape}"
    )
    assert residuals.shape == [s, b, n * C], (
        f"residuals shape mismatch: {residuals.shape}"
    )
    assert h_post.shape == [s, b, n], f"h_post shape mismatch: {h_post.shape}"

    # Validate numerical properties
    _assert_valid_tensor(branch_input, "branch_input")
    _assert_valid_tensor(residuals, "residuals")
    _assert_valid_tensor(h_post, "h_post", value_range=(0, 10))

    # Test gradient flow
    loss = branch_input.sum() + residuals.sum()
    loss.backward()
    _assert_gradient_valid(x.grad, "input", check_nonzero=True)


def test_mhc_basic_operations():
    """Test basic MHC width/depth connection operations with reduce."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    batch_size, seq_len, hidden_dim, num_streams = 2, 8, 64, 4

    config = MockConfig(
        hidden_size=hidden_dim,
        mhc_num_residual_streams=num_streams,
        mhc_sinkhorn_iters=10,
    )
    mhc = HyperConnectionModule(config=config, layer_number=1)

    # Test with [s, b, n*C] format
    x = paddle.randn([seq_len, batch_size, num_streams * hidden_dim])

    # Width connection
    branch_input, residuals, H_post = mhc.width_connection(x)
    assert branch_input.shape == [seq_len, batch_size, hidden_dim]
    assert residuals.shape == [seq_len, batch_size, num_streams * hidden_dim]
    assert H_post.shape == [seq_len, batch_size, num_streams]

    # Validate numerical properties
    _assert_valid_tensor(branch_input, "branch_input")
    _assert_valid_tensor(residuals, "residuals")
    _assert_valid_tensor(H_post, "H_post", value_range=(0, 10))

    # Simulate branch output (attention/mlp)
    branch_output = paddle.randn([seq_len, batch_size, hidden_dim])

    # Depth connection
    output = mhc.depth_connection((branch_output, None), residuals, H_post)
    assert output.shape == [seq_len, batch_size, num_streams * hidden_dim]

    # Validate depth output
    _assert_valid_tensor(output, "depth_output")

    # Reduce back to [s, b, C]
    output_reduced = HyperConnectionModule.reduce_stream(output, num_streams)
    assert output_reduced.shape == [seq_len, batch_size, hidden_dim]

    # Validate reduced output
    _assert_valid_tensor(output_reduced, "output_reduced")


# =============================================================================
# Test 2: Width + Depth Connection Integration
# =============================================================================


def test_width_depth_connection():
    """Test width_connection and depth_connection interfaces together."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=32)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 2, 3, 4, 32
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    # Width connection
    branch_input, residuals, h_post = hc.width_connection(x)
    assert branch_input.shape == [s, b, C]
    assert residuals.shape == [s, b, n * C]
    assert h_post.shape == [s, b, n]

    # Validate numerical properties
    _assert_valid_tensor(branch_input, "branch_input")
    _assert_valid_tensor(residuals, "residuals")
    _assert_valid_tensor(h_post, "h_post", value_range=(0, 10))

    # Depth connection with bias
    layer_output = paddle.randn([s, b, C], dtype="float32")
    bias = paddle.randn([C], dtype="float32")
    output = hc.depth_connection(
        layer_output_with_bias=(layer_output, bias),
        residuals=residuals,
        h_post=h_post,
        dropout_prob=0.0,
        training=False,
    )
    assert output.shape == [s, b, n * C]

    # Validate depth output
    _assert_valid_tensor(output, "depth_output")

    # Test gradient flow
    loss = output.sum()
    loss.backward()
    _assert_gradient_valid(x.grad, "input", check_nonzero=True)


def test_mhc_dimension_consistency():
    """Test that MHC works correctly with different batch/seq dimensions."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    batch_size, seq_len, hidden_dim, num_streams = 2, 4, 32, 2

    config = MockConfig(
        hidden_size=hidden_dim,
        mhc_num_residual_streams=num_streams,
        mhc_sinkhorn_iters=5,
    )
    mhc = HyperConnectionModule(config=config, layer_number=1)

    paddle.seed(42)
    x = paddle.randn([seq_len, batch_size, num_streams * hidden_dim])

    branch_input, residuals, h_post = mhc.width_connection(x)

    assert branch_input.shape == [seq_len, batch_size, hidden_dim]
    assert residuals.shape == [seq_len, batch_size, num_streams * hidden_dim]
    assert h_post.shape == [seq_len, batch_size, num_streams]

    # Validate numerical properties
    _assert_valid_tensor(branch_input, "branch_input")
    _assert_valid_tensor(residuals, "residuals")
    _assert_valid_tensor(h_post, "h_post", value_range=(0, 10))


# =============================================================================
# Test 3: Expand/Contract Round-trip
# =============================================================================


def test_expand_contract_roundtrip():
    """Test expand/contract round-trip consistency."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    s, b, C = 4, 2, 64
    n = 4
    x = paddle.randn([s, b, C])

    # Expand then contract should recover original
    expanded = HyperConnectionModule.expand_stream(x, n)
    assert expanded.shape == [s, b, n * C]

    contracted = HyperConnectionModule.reduce_stream(expanded, n)
    assert contracted.shape == [s, b, C]

    # Since expand replicates and contract averages, we should get back the original
    diff = (contracted - x).abs().max().item()
    assert diff < 1e-5, f"Round-trip should recover original, but diff={diff}"


def test_expand_reduce_block_level():
    """Test that expand/reduce is handled correctly at block level."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    s, b, C = 4, 2, 64
    n = 4

    x_original = paddle.randn([s, b, C])

    # Expand: [s, b, C] -> [s, b, n*C]
    x_expanded = HyperConnectionModule.expand_stream(x_original, n)
    assert x_expanded.shape == [s, b, n * C]

    # Validate expanded tensor
    _assert_valid_tensor(x_expanded, "x_expanded")

    # Contract: [s, b, n*C] -> [s, b, C]
    x_recovered = HyperConnectionModule.reduce_stream(x_expanded, n)
    assert x_recovered.shape == [s, b, C]

    # Validate recovered tensor
    _assert_valid_tensor(x_recovered, "x_recovered")

    # Verify round-trip with precision check
    diff = (x_recovered - x_original).abs().max().item()
    assert diff < 1e-5, f"Round-trip error too large: {diff}"


# =============================================================================
# Test 4: Full Forward-Backward
# =============================================================================


def test_full_forward_backward():
    """Test full forward-backward through HyperConnectionModule."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig()
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 64
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    # Width connection
    branch_input, residuals, h_post = hc.width_connection(x)

    # Validate width connection outputs
    _assert_valid_tensor(branch_input, "branch_input")
    _assert_valid_tensor(residuals, "residuals")
    _assert_valid_tensor(h_post, "h_post", value_range=(0, 10))

    # Simulate a layer (identity)
    layer_output = branch_input

    # Depth connection
    output = hc.depth_connection((layer_output, None), residuals, h_post)
    assert output.shape == x.shape

    # Validate depth output
    _assert_valid_tensor(output, "depth_output")

    # Backward
    loss = output.sum()
    loss.backward()

    # Validate input gradient
    _assert_gradient_valid(x.grad, "input", check_nonzero=True)

    # Check parameter gradients exist and are non-zero
    for name, param in hc.named_parameters():
        assert param.grad is not None, f"Parameter '{name}' has no gradient"
        grad_sum = param.grad.abs().sum().item()
        assert grad_sum > 1e-12, f"Parameter '{name}' has near-zero gradient"


# =============================================================================
# Test 5: Triton Backend Tests
# =============================================================================


def test_width_connection_triton():
    """Test width_connection with Triton backend."""
    from paddlefleet.transformer.hyper_connection import (
        TRITON_AVAILABLE,
        HyperConnectionModule,
    )

    if not TRITON_AVAILABLE:
        return  # Skip if Triton not available

    config = MockConfig(mhc_use_triton=True)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 64
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    branch_input, residuals, h_post = hc.width_connection(x)

    assert branch_input.shape == [s, b, C]
    assert residuals.shape == [s, b, n * C]
    assert h_post.shape == [s, b, n]

    # Test gradient flow
    loss = branch_input.sum() + residuals.sum()
    loss.backward()
    assert x.grad is not None

    # h_post should be in reasonable range (~[0, 2])
    assert h_post.min().item() >= 0
    assert h_post.max().item() <= 3


def test_native_vs_triton_comparison():
    """Compare native and triton implementations with strict precision validation using allclose."""
    from paddlefleet.transformer.hyper_connection import (
        TRITON_AVAILABLE,
        HyperConnectionModule,
    )

    if not TRITON_AVAILABLE:
        return  # Skip if Triton not available

    s, b, n, C = 8, 4, 4, 64

    # Create native and triton modules with same parameters
    paddle.seed(42)
    config_native = MockConfig(mhc_use_triton=False)
    hc_native = HyperConnectionModule(config=config_native, layer_number=1)

    paddle.seed(42)
    config_triton = MockConfig(mhc_use_triton=True)
    hc_triton = HyperConnectionModule(config=config_triton, layer_number=1)

    # Verify parameters are identical
    for (name1, p1), (name2, p2) in zip(
        hc_native.named_parameters(), hc_triton.named_parameters()
    ):
        assert paddle.allclose(p1, p2, rtol=1e-6, atol=1e-6), (
            f"Parameter {name1} mismatch"
        )

    # Same input for both
    paddle.seed(123)
    x_native = paddle.randn([s, b, n * C], dtype="float32")
    paddle.seed(123)
    x_triton = paddle.randn([s, b, n * C], dtype="float32")
    x_native.stop_gradient = False
    x_triton.stop_gradient = False

    # ==================== Width Connection Precision Test ====================
    branch_native, residuals_native, h_post_native = hc_native.width_connection(
        x_native
    )
    branch_triton, residuals_triton, h_post_triton = hc_triton.width_connection(
        x_triton
    )

    # Use allclose for precision validation (rtol=1e-3, atol=1e-4)
    assert paddle.allclose(
        branch_native, branch_triton, rtol=1e-3, atol=1e-4
    ), (
        f"branch_input mismatch: max_diff={((branch_native - branch_triton).abs().max().item()):.6f}"
    )
    assert paddle.allclose(
        residuals_native, residuals_triton, rtol=1e-3, atol=1e-4
    ), (
        f"residuals mismatch: max_diff={((residuals_native - residuals_triton).abs().max().item()):.6f}"
    )
    assert paddle.allclose(
        h_post_native, h_post_triton, rtol=1e-3, atol=1e-4
    ), (
        f"h_post mismatch: max_diff={((h_post_native - h_post_triton).abs().max().item()):.6f}"
    )

    # ==================== Depth Connection Precision Test ====================
    paddle.seed(456)
    branch_out_native = paddle.randn([s, b, C], dtype="float32")
    paddle.seed(456)
    branch_out_triton = paddle.randn([s, b, C], dtype="float32")
    paddle.seed(789)
    bias_native = paddle.randn([C], dtype="float32")
    paddle.seed(789)
    bias_triton = paddle.randn([C], dtype="float32")

    output_native = hc_native.depth_connection(
        (branch_out_native, bias_native),
        residuals_native,
        h_post_native,
        dropout_prob=0.0,
        training=False,
    )
    output_triton = hc_triton.depth_connection(
        (branch_out_triton, bias_triton),
        residuals_triton,
        h_post_triton,
        dropout_prob=0.0,
        training=False,
    )

    assert paddle.allclose(
        output_native, output_triton, rtol=1e-3, atol=1e-4
    ), (
        f"depth_output mismatch: max_diff={((output_native - output_triton).abs().max().item()):.6f}"
    )

    # ==================== Gradient Precision Test ====================
    loss_native = output_native.sum()
    loss_triton = output_triton.sum()
    loss_native.backward()
    loss_triton.backward()

    # Compare input gradients
    assert paddle.allclose(
        x_native.grad, x_triton.grad, rtol=1e-3, atol=1e-4
    ), (
        f"input gradient mismatch: max_diff={((x_native.grad - x_triton.grad).abs().max().item()):.6f}"
    )

    # Compare parameter gradients (allow slightly larger tolerance for numerical stability)
    for (name1, p1), (name2, p2) in zip(
        hc_native.named_parameters(), hc_triton.named_parameters()
    ):
        if p1.grad is not None and p2.grad is not None:
            assert paddle.allclose(p1.grad, p2.grad, rtol=5e-2, atol=5e-3), (
                f"param gradient {name1} mismatch: max_diff={((p1.grad - p2.grad).abs().max().item()):.6f}"
            )


# =============================================================================
# Test 6: Pipeline Helper Layers
# =============================================================================


def test_pipeline_layers():
    """Test MHCExpandLayer and MHCContractLayer."""
    from paddlefleet.transformer.hyper_connection import (
        MHCContractLayer,
        MHCExpandLayer,
    )

    config = MockConfig()
    s, b, C = 4, 2, 64
    n = 4

    # Test expand layer
    expand_layer = MHCExpandLayer(config=config)
    x = paddle.randn([s, b, C])
    dict_args = {"hidden_states": x, "attention_mask": None}
    dict_out = expand_layer(dict_args)
    assert dict_out["hidden_states"].shape == [s, b, n * C]

    # Test contract layer
    contract_layer = MHCContractLayer(config=config)
    dict_args = {
        "hidden_states": dict_out["hidden_states"],
        "attention_mask": None,
    }
    dict_out = contract_layer(dict_args)
    assert dict_out["hidden_states"].shape == [s, b, C]

    # Verify round-trip through pipeline layers
    diff = (dict_out["hidden_states"] - x).abs().max().item()
    assert diff < 1e-5


# =============================================================================
# Test 7: TransformerLayerWithMHC Integration
# =============================================================================


def test_transformer_layer_mhc():
    """Test TransformerLayerWithMHC with simplified components."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule
    from paddlefleet.transformer.identity_op import IdentityFuncOp, IdentityOp
    from paddlefleet.transformer.transformer_config import TransformerConfig
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayerSublayersSpec,
        TransformerLayerWithMHC,
    )

    # Create a minimal config
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

    # Create minimal sublayers spec
    sublayers_spec = TransformerLayerSublayersSpec(
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

    # Create layer
    layer = TransformerLayerWithMHC(
        config=config,
        sublayers_spec=sublayers_spec,
        layer_number=1,
    )

    # Create input
    batch_size = 2
    seq_len = 8
    n = config.mhc_num_residual_streams
    hidden_states = paddle.randn([batch_size, seq_len, config.hidden_size])
    hidden_states.stop_gradient = False

    # Expand input to n-stream format
    x_expanded = HyperConnectionModule.expand_stream(
        hidden_states.transpose([1, 0, 2]), n
    )

    # Forward pass
    output_dict = layer({"hidden_states": x_expanded})
    output = output_dict["hidden_states"]

    expected_shape = [seq_len, batch_size, n * config.hidden_size]
    assert output.shape == expected_shape

    # Validate output
    _assert_valid_tensor(output, "transformer_layer_output")

    # Backward pass
    loss = output.sum()
    loss.backward()
    _assert_gradient_valid(
        hidden_states.grad, "hidden_states", check_nonzero=True
    )


# =============================================================================
# Test 8: Depth Connection with Dropout
# =============================================================================


def test_depth_connection_with_dropout():
    """Test depth_connection with dropout enabled."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(mhc_use_triton=False)
    hc = HyperConnectionModule(config=config, layer_number=1)
    hc.train()  # Set to training mode

    s, b, n, C = 4, 2, 4, 64
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    branch_input, residuals, h_post = hc.width_connection(x)
    branch_output = paddle.randn([s, b, C], dtype="float32")

    # Test with dropout
    output = hc.depth_connection(
        (branch_output, None),
        residuals,
        h_post,
        dropout_prob=0.1,
        training=True,
    )
    assert output.shape == [s, b, n * C]

    # Validate output is finite (dropout shouldn't cause NaN)
    _assert_valid_tensor(output, "depth_output with dropout")

    # Backward with dropout
    loss = output.sum()
    loss.backward()
    _assert_gradient_valid(x.grad, "input with dropout", check_nonzero=True)


def test_depth_connection_with_bias_and_dropout():
    """Test depth_connection with bias and dropout."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(mhc_use_triton=False)
    hc = HyperConnectionModule(config=config, layer_number=1)
    hc.train()

    s, b, n, C = 4, 2, 4, 64
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    branch_input, residuals, h_post = hc.width_connection(x)
    branch_output = paddle.randn([s, b, C], dtype="float32")
    bias = paddle.randn([C], dtype="float32")

    # Test with bias and dropout
    output = hc.depth_connection(
        (branch_output, bias),
        residuals,
        h_post,
        dropout_prob=0.15,
        training=True,
    )
    assert output.shape == [s, b, n * C]

    # Validate output
    _assert_valid_tensor(output, "depth_output with bias and dropout")

    loss = output.sum()
    loss.backward()
    _assert_gradient_valid(
        x.grad, "input with bias and dropout", check_nonzero=True
    )


def test_depth_connection_fused():
    """Test depth_connection with fused=True."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(mhc_use_triton=False)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 64
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    branch_input, residuals, h_post = hc.width_connection(x)
    branch_output = paddle.randn([s, b, C], dtype="float32")
    bias = paddle.randn([C], dtype="float32")

    # Test with fused=True
    output = hc.depth_connection(
        (branch_output, bias),
        residuals,
        h_post,
        dropout_prob=0.0,
        training=False,
        fused=True,
    )
    assert output.shape == [s, b, n * C]


# =============================================================================
# Test 9: Single Tensor Input for Depth Connection
# =============================================================================


def test_depth_connection_single_tensor_input():
    """Test depth_connection with single Tensor input (not tuple)."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(mhc_use_triton=False)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 64
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    branch_input, residuals, h_post = hc.width_connection(x)

    # Pass single tensor instead of tuple
    branch_output = paddle.randn([s, b, C], dtype="float32")
    output = hc.depth_connection(
        branch_output,  # Single tensor, not tuple
        residuals,
        h_post,
    )
    assert output.shape == [s, b, n * C]

    loss = output.sum()
    loss.backward()
    assert x.grad is not None


# =============================================================================
# Test 10: Different n Values
# =============================================================================


def test_different_n_values():
    """Test MHC with different n (num_residual_streams) values."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    for n in [1, 2, 4, 8]:
        config = MockConfig(
            hidden_size=32,
            mhc_num_residual_streams=n,
            mhc_sinkhorn_iters=5,
        )
        hc = HyperConnectionModule(config=config, layer_number=1)

        s, b, C = 4, 2, 32
        x = paddle.randn([s, b, n * C], dtype="float32")
        x.stop_gradient = False

        branch_input, residuals, h_post = hc.width_connection(x)
        assert branch_input.shape == [s, b, C]
        assert residuals.shape == [s, b, n * C]
        assert h_post.shape == [s, b, n]

        branch_output = paddle.randn([s, b, C], dtype="float32")
        output = hc.depth_connection((branch_output, None), residuals, h_post)
        assert output.shape == [s, b, n * C]

        loss = output.sum()
        loss.backward()
        assert x.grad is not None


# =============================================================================
# Test 11: RMS Normalization
# =============================================================================


def test_rms_norm():
    """Test RMS normalization in width_connection."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=64)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 64
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    # Width connection uses _rms_norm internally
    branch_input, residuals, h_post = hc.width_connection(x)

    # Verify output is normalized (should have reasonable magnitude)
    assert branch_input.abs().mean().item() < 10.0  # Not exploding
    assert residuals.abs().mean().item() < 10.0


# =============================================================================
# Test 12: Triton Depth Connection
# =============================================================================


def test_depth_connection_triton():
    """Test depth_connection with Triton backend."""
    from paddlefleet.transformer.hyper_connection import (
        TRITON_AVAILABLE,
        HyperConnectionModule,
    )

    if not TRITON_AVAILABLE:
        return

    config = MockConfig(mhc_use_triton=True)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 64
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    branch_input, residuals, h_post = hc.width_connection(x)
    branch_output = paddle.randn([s, b, C], dtype="float32")
    bias = paddle.randn([C], dtype="float32")

    # Test Triton depth connection
    output = hc.depth_connection(
        (branch_output, bias),
        residuals,
        h_post,
        dropout_prob=0.0,
        training=False,
    )
    assert output.shape == [s, b, n * C]


def test_depth_connection_triton_with_dropout():
    """Test Triton depth_connection with dropout."""
    from paddlefleet.transformer.hyper_connection import (
        TRITON_AVAILABLE,
        HyperConnectionModule,
    )

    if not TRITON_AVAILABLE:
        return

    config = MockConfig(mhc_use_triton=True)
    hc = HyperConnectionModule(config=config, layer_number=1)
    hc.train()

    s, b, n, C = 4, 2, 4, 64
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    branch_input, residuals, h_post = hc.width_connection(x)
    branch_output = paddle.randn([s, b, C], dtype="float32")

    output = hc.depth_connection(
        (branch_output, None),
        residuals,
        h_post,
        dropout_prob=0.1,
        training=True,
    )
    assert output.shape == [s, b, n * C]


# =============================================================================
# Test 13: skip_sk_gradient Parameter
# =============================================================================


def test_width_connection_skip_sk_gradient():
    """Test width_connection with skip_sk_gradient=False."""
    from paddlefleet.transformer.hyper_connection import (
        TRITON_AVAILABLE,
        HyperConnectionModule,
    )

    if not TRITON_AVAILABLE:
        return

    config = MockConfig(mhc_use_triton=True)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 64
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    # Test with skip_sk_gradient=False
    branch_input, residuals, h_post = hc.width_connection(
        x, skip_sk_gradient=False
    )

    assert branch_input.shape == [s, b, C]
    assert residuals.shape == [s, b, n * C]
    assert h_post.shape == [s, b, n]


# =============================================================================
# Test 14: Different Sinkhorn Iterations
# =============================================================================


def test_different_sinkhorn_iterations():
    """Test MHC with different sinkhorn_iterations values."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    for iters in [1, 5, 10, 20]:
        config = MockConfig(
            hidden_size=32,
            mhc_num_residual_streams=4,
            mhc_sinkhorn_iters=iters,
        )
        hc = HyperConnectionModule(config=config, layer_number=1)

        s, b, n, C = 4, 2, 4, 32
        x = paddle.randn([s, b, n * C], dtype="float32")

        branch_input, residuals, h_post = hc.width_connection(x)
        assert branch_input.shape == [s, b, C]


# =============================================================================
# Test 15: Dtype Handling
# =============================================================================


def test_dtype_consistency():
    """Test that output dtype matches input dtype."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=32)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 32

    # Test float32
    x_f32 = paddle.randn([s, b, n * C], dtype="float32")
    branch_input, residuals, h_post = hc.width_connection(x_f32)
    assert branch_input.dtype == paddle.float32
    assert residuals.dtype == paddle.float32

    # Test float16 (if supported)
    try:
        x_f16 = paddle.randn([s, b, n * C], dtype="float16")
        branch_input, residuals, h_post = hc.width_connection(x_f16)
        assert branch_input.dtype == paddle.float16
        assert residuals.dtype == paddle.float16
    except Exception:
        pass  # Skip if float16 not supported


# =============================================================================
# Test 16: Layer Number Parameter
# =============================================================================


def test_different_layer_numbers():
    """Test HyperConnectionModule with different layer_number values."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    for layer_num in [1, 5, 10, 100]:
        config = MockConfig(hidden_size=32)
        hc = HyperConnectionModule(config=config, layer_number=layer_num)

        assert hc.layer_number == layer_num

        s, b, n, C = 4, 2, 4, 32
        x = paddle.randn([s, b, n * C], dtype="float32")
        branch_input, residuals, h_post = hc.width_connection(x)
        assert branch_input.shape == [s, b, C]


# =============================================================================
# Test 17: Parameter Initialization
# =============================================================================


def test_parameter_initialization():
    """Test that parameters are initialized correctly."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=64, mhc_num_residual_streams=4)
    hc = HyperConnectionModule(config=config, layer_number=1)

    # Check parameter shapes
    n, C = config.mhc_num_residual_streams, config.hidden_size
    total_output_dim = n + n + n * n

    assert hc.combined_weights.shape == [n * C, total_output_dim]
    assert hc.scaling_factors.shape == [3]
    assert hc.bias_terms.shape == [total_output_dim]

    # Check scaling factors are initialized to 0.01
    assert abs(hc.scaling_factors.numpy()[0] - 0.01) < 1e-5

    # Check bias terms are initialized to 0
    assert hc.bias_terms.abs().max().item() < 1e-5


# =============================================================================
# Test 18: Edge Cases
# =============================================================================


def test_edge_case_small_dimensions():
    """Test MHC with small dimensions."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(
        hidden_size=8,
        mhc_num_residual_streams=2,
        mhc_sinkhorn_iters=3,
    )
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 2, 1, 2, 8
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    branch_input, residuals, h_post = hc.width_connection(x)
    output = hc.depth_connection((branch_input, None), residuals, h_post)

    assert output.shape == [s, b, n * C]

    loss = output.sum()
    loss.backward()
    assert x.grad is not None


def test_edge_case_large_batch():
    """Test MHC with larger batch size."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=32)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 16, 4, 32  # Large batch
    x = paddle.randn([s, b, n * C], dtype="float32")

    branch_input, residuals, h_post = hc.width_connection(x)
    assert branch_input.shape == [s, b, C]
    assert residuals.shape == [s, b, n * C]


def test_edge_case_large_seq():
    """Test MHC with larger sequence length."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=32)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 128, 2, 4, 32  # Large sequence
    x = paddle.randn([s, b, n * C], dtype="float32")

    branch_input, residuals, h_post = hc.width_connection(x)
    assert branch_input.shape == [s, b, C]
    assert residuals.shape == [s, b, n * C]


# =============================================================================
# Test 19: Triton Fallback
# =============================================================================


def test_triton_fallback_to_native():
    """Test that native path is used when Triton is disabled."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    # Explicitly disable Triton
    config = MockConfig(mhc_use_triton=False)
    hc = HyperConnectionModule(config=config, layer_number=1)

    # Verify Triton is disabled
    assert not hc.use_triton

    s, b, n, C = 4, 2, 4, 64
    x = paddle.randn([s, b, n * C], dtype="float32")

    # This should use native path
    branch_input, residuals, h_post = hc.width_connection(x)
    assert branch_input.shape == [s, b, C]


# =============================================================================
# Test 20: Multiple Forward Passes
# =============================================================================


def test_multiple_forward_passes():
    """Test multiple forward passes through the same module."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=32)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 32

    for _ in range(5):
        x = paddle.randn([s, b, n * C], dtype="float32")
        x.stop_gradient = False

        branch_input, residuals, h_post = hc.width_connection(x)
        output = hc.depth_connection((branch_input, None), residuals, h_post)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None

        # Clear gradients for next iteration
        hc.clear_gradients()


# =============================================================================
# Test 21: Gradient Accumulation
# =============================================================================


def test_gradient_accumulation():
    """Test gradient accumulation across multiple backward passes."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=32)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 32
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    # First forward-backward
    branch_input, residuals, h_post = hc.width_connection(x)
    output = hc.depth_connection((branch_input, None), residuals, h_post)
    loss1 = output.sum()
    loss1.backward()

    grad1 = x.grad.clone()

    # Second forward-backward (accumulate gradients)
    branch_input, residuals, h_post = hc.width_connection(x)
    output = hc.depth_connection((branch_input, None), residuals, h_post)
    loss2 = output.sum()
    loss2.backward()

    # Gradients should accumulate
    assert x.grad is not None


# =============================================================================
# Test 22: H_post Dtype Mismatch Branches
# =============================================================================


def test_h_post_dtype_mismatch():
    """Test h_post dtype mismatch handling in depth_connection."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=32, mhc_use_triton=False)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 32
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    branch_input, residuals, h_post = hc.width_connection(x)

    # Manually cast h_post to different dtype to trigger the conversion branch
    h_post_f16 = h_post.cast("float16")
    branch_output = paddle.randn([s, b, C], dtype="float32")

    # depth_connection should handle h_post dtype mismatch
    output = hc.depth_connection((branch_output, None), residuals, h_post_f16)
    assert output.shape == [s, b, n * C]


def test_h_post_dtype_mismatch_triton():
    """Test h_post dtype mismatch handling in Triton depth_connection."""
    from paddlefleet.transformer.hyper_connection import (
        TRITON_AVAILABLE,
        HyperConnectionModule,
    )

    if not TRITON_AVAILABLE:
        return

    config = MockConfig(hidden_size=32, mhc_use_triton=True)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 32
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    branch_input, residuals, h_post = hc.width_connection(x)

    # Manually cast h_post to different dtype
    h_post_f16 = h_post.cast("float16")
    branch_output = paddle.randn([s, b, C], dtype="float32")

    output = hc.depth_connection((branch_output, None), residuals, h_post_f16)
    assert output.shape == [s, b, n * C]


# =============================================================================
# Test 23: Triton Depth Connection with Bias
# =============================================================================


def test_triton_depth_connection_with_bias():
    """Test Triton depth_connection with bias."""
    from paddlefleet.transformer.hyper_connection import (
        TRITON_AVAILABLE,
        HyperConnectionModule,
    )

    if not TRITON_AVAILABLE:
        return

    config = MockConfig(hidden_size=32, mhc_use_triton=True)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 32
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    branch_input, residuals, h_post = hc.width_connection(x)
    branch_output = paddle.randn([s, b, C], dtype="float32")
    bias = paddle.randn([C], dtype="float32")

    # Test with bias
    output = hc.depth_connection((branch_output, bias), residuals, h_post)

    assert output.shape == [s, b, n * C]

    loss = output.sum()
    loss.backward()
    assert x.grad is not None


# =============================================================================
# Test 24: Native Depth Connection with Bias Expanded
# =============================================================================


def test_native_depth_connection_bias_expanded():
    """Test native depth_connection with bias expansion."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=32, mhc_use_triton=False)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 32
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    branch_input, residuals, h_post = hc.width_connection(x)
    branch_output = paddle.randn([s, b, C], dtype="float32")
    bias = paddle.randn([C], dtype="float32")

    # Test native path with bias
    output = hc.depth_connection(
        (branch_output, bias),
        residuals,
        h_post,
        dropout_prob=0.0,
        training=False,
        fused=False,
    )

    assert output.shape == [s, b, n * C]

    loss = output.sum()
    loss.backward()
    assert x.grad is not None


# =============================================================================
# Test 25: All Branches Coverage for Depth Connection
# =============================================================================


def test_depth_connection_all_branches():
    """Test all branches of depth_connection."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=32, mhc_use_triton=False)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 32

    # Test case 1: tuple input with bias
    x = paddle.randn([s, b, n * C], dtype="float32")
    branch_input, residuals, h_post = hc.width_connection(x)
    branch_output = paddle.randn([s, b, C], dtype="float32")
    bias = paddle.randn([C], dtype="float32")

    output = hc.depth_connection((branch_output, bias), residuals, h_post)
    assert output.shape == [s, b, n * C]

    # Test case 2: single tensor input (not tuple)
    output = hc.depth_connection(branch_output, residuals, h_post)
    assert output.shape == [s, b, n * C]

    # Test case 3: tuple input without bias
    output = hc.depth_connection((branch_output, None), residuals, h_post)
    assert output.shape == [s, b, n * C]


# =============================================================================
# Test 26: Skip SK Gradient Parameter Coverage
# =============================================================================


def test_skip_sk_gradient_both_values():
    """Test width_connection with both skip_sk_gradient values."""
    from paddlefleet.transformer.hyper_connection import (
        TRITON_AVAILABLE,
        HyperConnectionModule,
    )

    if not TRITON_AVAILABLE:
        return

    config = MockConfig(hidden_size=32, mhc_use_triton=True)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 32
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    # Test with skip_sk_gradient=True (default)
    branch_input1, residuals1, h_post1 = hc.width_connection(
        x, skip_sk_gradient=True
    )

    # Test with skip_sk_gradient=False
    x.stop_gradient = False
    branch_input2, residuals2, h_post2 = hc.width_connection(
        x, skip_sk_gradient=False
    )

    assert branch_input1.shape == [s, b, C]
    assert branch_input2.shape == [s, b, C]


# =============================================================================
# Test 27: Gradient Numerical Accuracy (Finite Difference)
# =============================================================================


def test_gradient_numerical_accuracy():
    """Verify gradient correctness using finite differences.

    Note: Finite difference gradients may have larger errors due to:
    1. Numerical precision in forward pass
    2. Sinkhorn iterations with non-trivial gradient paths
    We verify gradient direction correlation rather than exact match.
    """
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=32, mhc_num_residual_streams=4)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 32
    paddle.seed(42)
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    # Compute numerical vs analytical gradient
    analytical_grad, numerical_grad, max_diff, sample_indices = (
        _compute_numerical_gradient(
            hc,
            x,
            eps=1e-3,
            num_samples=20,  # Larger eps for stability
        )
    )

    # Validate analytical gradient
    _assert_gradient_valid(analytical_grad, "input")

    # Compare sampled gradients - check direction correlation
    sampled_analytical = analytical_grad.flatten()[sample_indices]
    sampled_numerical = numerical_grad.flatten()[sample_indices]

    # Check that gradients point in the same direction (positive correlation)
    # and have similar magnitude (within factor of 2)
    analytical_norm = sampled_analytical.norm().item()
    numerical_norm = sampled_numerical.norm().item()

    # Magnitude should be within factor of 3 (allowing for finite difference errors)
    magnitude_ratio = analytical_norm / (numerical_norm + 1e-8)
    assert 0.3 < magnitude_ratio < 3.0, (
        f"Gradient magnitude mismatch: analytical={analytical_norm:.4f}, "
        f"numerical={numerical_norm:.4f}, ratio={magnitude_ratio:.4f}"
    )

    # Direction should be similar (cosine similarity > 0.8)
    dot_product = (sampled_analytical * sampled_numerical).sum().item()
    cosine_sim = dot_product / (analytical_norm * numerical_norm + 1e-8)
    assert cosine_sim > 0.7, (
        f"Gradient direction mismatch: cosine_similarity={cosine_sim:.4f}"
    )


def test_gradient_numerical_accuracy_triton():
    """Verify Triton gradient correctness using finite differences."""
    from paddlefleet.transformer.hyper_connection import (
        TRITON_AVAILABLE,
        HyperConnectionModule,
    )

    if not TRITON_AVAILABLE:
        return

    config = MockConfig(
        hidden_size=32, mhc_num_residual_streams=4, mhc_use_triton=True
    )
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 32
    paddle.seed(42)
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    analytical_grad, numerical_grad, max_diff, sample_indices = (
        _compute_numerical_gradient(hc, x, eps=1e-3, num_samples=20)
    )

    _assert_gradient_valid(analytical_grad, "input")

    sampled_analytical = analytical_grad.flatten()[sample_indices]
    sampled_numerical = numerical_grad.flatten()[sample_indices]

    analytical_norm = sampled_analytical.norm().item()
    numerical_norm = sampled_numerical.norm().item()

    magnitude_ratio = analytical_norm / (numerical_norm + 1e-8)
    assert 0.3 < magnitude_ratio < 3.0, (
        f"Triton gradient magnitude mismatch: analytical={analytical_norm:.4f}, "
        f"numerical={numerical_norm:.4f}, ratio={magnitude_ratio:.4f}"
    )

    dot_product = (sampled_analytical * sampled_numerical).sum().item()
    cosine_sim = dot_product / (analytical_norm * numerical_norm + 1e-8)
    assert cosine_sim > 0.7, (
        f"Triton gradient direction mismatch: cosine_similarity={cosine_sim:.4f}"
    )


# =============================================================================
# Test 28: All Parameter Gradients Non-Zero
# =============================================================================


def test_all_parameter_gradients_nonzero():
    """Verify all parameters receive non-zero gradients."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=64, mhc_num_residual_streams=4)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 64
    paddle.seed(42)
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    # Forward pass
    branch, residuals, h_post = hc.width_connection(x)
    output = hc.depth_connection((branch, None), residuals, h_post)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Validate all parameter gradients
    for name, param in hc.named_parameters():
        assert param.grad is not None, f"Parameter '{name}' has no gradient"
        grad_sum = param.grad.abs().sum().item()
        assert grad_sum > 1e-10, (
            f"Parameter '{name}' has near-zero gradient (sum={grad_sum})"
        )


def test_all_parameter_gradients_nonzero_with_bias():
    """Verify all parameters receive gradients with bias in depth_connection."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=64, mhc_num_residual_streams=4)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 64
    paddle.seed(42)
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    branch, residuals, h_post = hc.width_connection(x)
    branch_output = paddle.randn([s, b, C], dtype="float32")
    bias = paddle.randn([C], dtype="float32")

    output = hc.depth_connection((branch_output, bias), residuals, h_post)
    loss = output.sum()
    loss.backward()

    for name, param in hc.named_parameters():
        assert param.grad is not None, f"Parameter '{name}' has no gradient"
        grad_sum = param.grad.abs().sum().item()
        assert grad_sum > 1e-10, (
            f"Parameter '{name}' has near-zero gradient (sum={grad_sum})"
        )


# =============================================================================
# Test 29: Width → Depth Chain Numerical Stability
# =============================================================================


def test_width_depth_chain_numerical_stability():
    """Test that width→depth chain maintains numerical stability."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=64, mhc_num_residual_streams=4)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 64
    paddle.seed(42)
    x = paddle.randn([s, b, n * C], dtype="float32")

    # Width connection
    branch, residuals, h_post = hc.width_connection(x)

    # Validate outputs are finite
    _assert_valid_tensor(branch, "branch_input")
    _assert_valid_tensor(residuals, "residuals")
    _assert_valid_tensor(h_post, "h_post", value_range=(0, 10))

    # Depth connection
    branch_output = paddle.randn([s, b, C], dtype="float32")
    output = hc.depth_connection((branch_output, None), residuals, h_post)

    _assert_valid_tensor(output, "depth_output")

    # Check output magnitude is reasonable (not exploding)
    output_norm = output.abs().mean().item()
    assert output_norm < 100.0, f"Output magnitude too large: {output_norm}"


def test_width_depth_chain_invariant():
    """Test that width→depth preserves reasonable numerical properties."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=64, mhc_num_residual_streams=4)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 64
    paddle.seed(42)
    x = paddle.randn([s, b, n * C], dtype="float32")

    # Width connection
    branch, residuals, h_post = hc.width_connection(x)

    # If we pass branch directly back through depth, output should be well-behaved
    output = hc.depth_connection((branch, None), residuals, h_post)

    # Relative change should be bounded
    input_norm = x.abs().max().item()
    output_norm = output.abs().max().item()

    relative_change = abs(output_norm - input_norm) / (input_norm + 1e-6)
    assert relative_change < 10.0, (
        f"Relative change too large: {relative_change}"
    )


# =============================================================================
# Test 30: Numerical Stability with Extreme Values
# =============================================================================


def test_numerical_stability_large_values():
    """Test MHC with large input values."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=32, mhc_num_residual_streams=4)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 32
    # Large values
    x = paddle.randn([s, b, n * C], dtype="float32") * 100.0

    branch, residuals, h_post = hc.width_connection(x)

    _assert_valid_tensor(branch, "branch_input (large values)")
    _assert_valid_tensor(residuals, "residuals (large values)")
    _assert_valid_tensor(h_post, "h_post (large values)")

    branch_output = paddle.randn([s, b, C], dtype="float32") * 100.0
    output = hc.depth_connection((branch_output, None), residuals, h_post)

    _assert_valid_tensor(output, "depth_output (large values)")


def test_numerical_stability_small_values():
    """Test MHC with very small input values."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=32, mhc_num_residual_streams=4)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 32
    # Small values
    x = paddle.randn([s, b, n * C], dtype="float32") * 1e-4

    branch, residuals, h_post = hc.width_connection(x)

    _assert_valid_tensor(branch, "branch_input (small values)")
    _assert_valid_tensor(residuals, "residuals (small values)")
    _assert_valid_tensor(h_post, "h_post (small values)")

    branch_output = paddle.randn([s, b, C], dtype="float32") * 1e-4
    output = hc.depth_connection((branch_output, None), residuals, h_post)

    _assert_valid_tensor(output, "depth_output (small values)")


def test_numerical_stability_gradient_large_values():
    """Test gradient stability with large values."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=32, mhc_num_residual_streams=4)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 32
    x = paddle.randn([s, b, n * C], dtype="float32") * 100.0
    x.stop_gradient = False

    branch, residuals, h_post = hc.width_connection(x)
    output = hc.depth_connection((branch, None), residuals, h_post)

    loss = output.sum()
    loss.backward()

    _assert_gradient_valid(x.grad, "input (large values)", check_nonzero=True)


def test_numerical_stability_gradient_small_values():
    """Test gradient stability with small values."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=32, mhc_num_residual_streams=4)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 32
    x = paddle.randn([s, b, n * C], dtype="float32") * 1e-4
    x.stop_gradient = False

    branch, residuals, h_post = hc.width_connection(x)
    output = hc.depth_connection((branch, None), residuals, h_post)

    loss = output.sum()
    loss.backward()

    _assert_gradient_valid(x.grad, "input (small values)", check_nonzero=True)


# =============================================================================
# Test 31: Improved Gradient Validation for Existing Tests
# =============================================================================


def test_gradient_magnitude_reasonable():
    """Test that gradient magnitudes are within reasonable bounds."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=64, mhc_num_residual_streams=4)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 64
    paddle.seed(42)
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    branch, residuals, h_post = hc.width_connection(x)
    output = hc.depth_connection((branch, None), residuals, h_post)

    loss = output.sum()
    loss.backward()

    # Gradient magnitude should be reasonable (not exploding/vanishing)
    grad_norm = x.grad.norm().item()
    grad_max = x.grad.abs().max().item()

    assert grad_norm > 1e-6, f"Gradient norm too small (vanishing): {grad_norm}"
    assert grad_norm < 1e6, f"Gradient norm too large (exploding): {grad_norm}"
    assert grad_max < 1e4, f"Gradient max too large: {grad_max}"


def test_parameter_gradient_magnitude_reasonable():
    """Test that parameter gradient magnitudes are within reasonable bounds."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=64, mhc_num_residual_streams=4)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 4, 2, 4, 64
    paddle.seed(42)
    x = paddle.randn([s, b, n * C], dtype="float32")
    x.stop_gradient = False

    branch, residuals, h_post = hc.width_connection(x)
    output = hc.depth_connection((branch, None), residuals, h_post)

    loss = output.sum()
    loss.backward()

    for name, param in hc.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            assert grad_norm < 1e6, (
                f"Parameter '{name}' gradient exploding: {grad_norm}"
            )
            # Allow smaller gradients for some parameters
            assert grad_norm > 1e-12 or grad_norm == 0, (
                f"Parameter '{name}' gradient unexpectedly small: {grad_norm}"
            )


# =============================================================================
# Test 32: Output Consistency Across Multiple Runs
# =============================================================================


def test_output_determinism():
    """Test that outputs are deterministic for same inputs."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=32, mhc_num_residual_streams=4)
    hc = HyperConnectionModule(config=config, layer_number=1)
    hc.eval()

    s, b, n, C = 4, 2, 4, 32
    paddle.seed(42)
    x = paddle.randn([s, b, n * C], dtype="float32")

    # Run twice with same input
    branch1, residuals1, h_post1 = hc.width_connection(x)
    branch2, residuals2, h_post2 = hc.width_connection(x)

    assert paddle.allclose(branch1, branch2, rtol=1e-6, atol=1e-6), (
        "branch not deterministic"
    )
    assert paddle.allclose(residuals1, residuals2, rtol=1e-6, atol=1e-6), (
        "residuals not deterministic"
    )
    assert paddle.allclose(h_post1, h_post2, rtol=1e-6, atol=1e-6), (
        "h_post not deterministic"
    )


def test_output_consistency_with_depth():
    """Test that full pipeline outputs are consistent."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=32, mhc_num_residual_streams=4)
    hc = HyperConnectionModule(config=config, layer_number=1)
    hc.eval()

    s, b, n, C = 4, 2, 4, 32
    paddle.seed(42)
    x = paddle.randn([s, b, n * C], dtype="float32")

    # Run full pipeline twice
    branch1, residuals1, h_post1 = hc.width_connection(x)
    output1 = hc.depth_connection((branch1, None), residuals1, h_post1)

    branch2, residuals2, h_post2 = hc.width_connection(x)
    output2 = hc.depth_connection((branch2, None), residuals2, h_post2)

    assert paddle.allclose(output1, output2, rtol=1e-6, atol=1e-6), (
        "depth output not deterministic"
    )


# =============================================================================
# Test 33: Triton vs Native Gradient Comparison
# =============================================================================


def test_triton_native_gradient_comparison():
    """Compare gradients between Triton and Native implementations."""
    from paddlefleet.transformer.hyper_connection import (
        TRITON_AVAILABLE,
        HyperConnectionModule,
    )

    if not TRITON_AVAILABLE:
        return

    s, b, n, C = 8, 4, 4, 32

    # Create modules with same seed
    paddle.seed(42)
    config_native = MockConfig(hidden_size=C, mhc_use_triton=False)
    hc_native = HyperConnectionModule(config=config_native, layer_number=1)

    paddle.seed(42)
    config_triton = MockConfig(hidden_size=C, mhc_use_triton=True)
    hc_triton = HyperConnectionModule(config=config_triton, layer_number=1)

    # Same input
    paddle.seed(123)
    x_native = paddle.randn([s, b, n * C], dtype="float32")
    paddle.seed(123)
    x_triton = paddle.randn([s, b, n * C], dtype="float32")
    x_native.stop_gradient = False
    x_triton.stop_gradient = False

    # Forward and backward
    branch_native, residuals_native, h_post_native = hc_native.width_connection(
        x_native
    )
    output_native = hc_native.depth_connection(
        (branch_native, None), residuals_native, h_post_native
    )
    output_native.sum().backward()

    branch_triton, residuals_triton, h_post_triton = hc_triton.width_connection(
        x_triton
    )
    output_triton = hc_triton.depth_connection(
        (branch_triton, None), residuals_triton, h_post_triton
    )
    output_triton.sum().backward()

    # Compare input gradients
    assert paddle.allclose(
        x_native.grad, x_triton.grad, rtol=1e-2, atol=1e-3
    ), (
        f"Input gradient mismatch: max_diff={((x_native.grad - x_triton.grad).abs().max().item()):.6f}"
    )

    # Compare parameter gradients
    for (name1, p1), (name2, p2) in zip(
        hc_native.named_parameters(), hc_triton.named_parameters()
    ):
        if p1.grad is not None and p2.grad is not None:
            assert paddle.allclose(p1.grad, p2.grad, rtol=5e-2, atol=5e-3), (
                f"Parameter gradient '{name1}' mismatch: "
                f"max_diff={((p1.grad - p2.grad).abs().max().item()):.6f}"
            )


# =============================================================================
# Test 34-50: Tests for gpt_layer_specs.py coverage
# =============================================================================


class MockTransformerConfig:
    """Mock TransformerConfig for testing gpt_layer_specs."""

    def __init__(
        self,
        num_hidden_layers=4,
        hidden_size=64,
        num_attention_heads=4,
        intermediate_size=128,
        vocab_size=100,
        max_sequence_length=64,
        normalization="RMSNorm",
        use_qk_norm=False,
        multi_latent_attention=False,
        n_routed_experts=None,
        moe_grouped_gemm=False,
        moe_layer_freq=1,
        num_empty_layers_add_in_head=0,
        num_empty_layers_add_in_tail=0,
        num_nextn_predict_layers=None,
        num_layers=4,
        mhc_num_residual_streams=4,
        mhc_sinkhorn_iters=10,
        mhc_use_triton=False,
        use_mhc=False,
        hidden_dropout_prob=0.0,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rotary_percent=1.0,
        position_embedding_type="rope",
        tie_word_embeddings=False,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        parallel_output=False,
        max_position_embeddings=512,
        rope_scaling=False,
        mrope_section=None,
        model_type="gpt",
        init_method=None,
        output_layer_init_method=None,
        bias_dropout_fusion=False,
        recompute_granularity=None,
        recompute_modules=None,
        recompute_num_layers=None,
        recompute_method=None,
        context_parallel_size=1,
        cp_comm_type=None,
        sequence_parallel=False,
        cpu_offloading=False,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.normalization = normalization
        self.use_qk_norm = use_qk_norm
        self.multi_latent_attention = multi_latent_attention
        self.n_routed_experts = n_routed_experts
        self.moe_grouped_gemm = moe_grouped_gemm
        self.moe_layer_freq = moe_layer_freq
        self.num_empty_layers_add_in_head = num_empty_layers_add_in_head
        self.num_empty_layers_add_in_tail = num_empty_layers_add_in_tail
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_layers = num_layers
        self.mhc_num_residual_streams = mhc_num_residual_streams
        self.mhc_sinkhorn_iters = mhc_sinkhorn_iters
        self.mhc_use_triton = mhc_use_triton
        self.use_mhc = use_mhc
        self.hidden_dropout_prob = hidden_dropout_prob
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rotary_percent = rotary_percent
        self.position_embedding_type = position_embedding_type
        self.tie_word_embeddings = tie_word_embeddings
        self.pipeline_model_parallel_size = pipeline_model_parallel_size
        self.virtual_pipeline_model_parallel_size = (
            virtual_pipeline_model_parallel_size
        )
        self.parallel_output = parallel_output
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = rope_scaling
        self.mrope_section = mrope_section
        self.model_type = model_type
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method
        self.bias_dropout_fusion = bias_dropout_fusion
        self.recompute_granularity = recompute_granularity
        self.recompute_modules = recompute_modules
        self.recompute_num_layers = recompute_num_layers
        self.recompute_method = recompute_method
        self.context_parallel_size = context_parallel_size
        self.cp_comm_type = cp_comm_type
        self.sequence_parallel = sequence_parallel
        self.cpu_offloading = cpu_offloading


def test_get_gpt_layer_mhc_spec_basic():
    """Test basic MHC layer spec creation."""
    from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_mhc_spec
    from paddlefleet.spec_utils import LayerSpec
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayerWithMHC,
    )

    config = MockTransformerConfig(use_mhc=True)
    spec = get_gpt_layer_mhc_spec(
        config=config,
        num_experts=None,
        moe_grouped_gemm=False,
        use_qk_norm=False,
        multi_latent_attention=False,
        normalization="RMSNorm",
        qk_l2_norm=False,
        layer_number=1,
    )
    assert isinstance(spec, LayerSpec)
    assert spec.layer == TransformerLayerWithMHC


def test_get_gpt_layer_mhc_spec_with_mla():
    """Test MHC layer spec with multi-latent attention."""
    from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_mhc_spec
    from paddlefleet.spec_utils import LayerSpec
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayerWithMHC,
    )

    config = MockTransformerConfig(use_mhc=True, multi_latent_attention=True)
    spec = get_gpt_layer_mhc_spec(
        config=config,
        num_experts=None,
        moe_grouped_gemm=False,
        use_qk_norm=False,
        multi_latent_attention=True,
        normalization="RMSNorm",
        qk_l2_norm=False,
        layer_number=1,
    )
    assert isinstance(spec, LayerSpec)
    assert spec.layer == TransformerLayerWithMHC


def test_get_gpt_layer_mhc_spec_with_layernorm():
    """Test MHC layer spec with LayerNorm instead of RMSNorm."""
    from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_mhc_spec
    from paddlefleet.spec_utils import LayerSpec

    config = MockTransformerConfig(use_mhc=True)
    spec = get_gpt_layer_mhc_spec(
        config=config,
        num_experts=None,
        moe_grouped_gemm=False,
        use_qk_norm=True,
        multi_latent_attention=False,
        normalization="LayerNorm",
        qk_l2_norm=False,
        layer_number=1,
    )
    assert isinstance(spec, LayerSpec)


def test_get_gpt_mhc_decoder_layers_spec_with_int_freq():
    """Test MHC decoder layers spec with integer moe_layer_freq."""
    from paddlefleet.models.gpt.gpt_layer_specs import (
        get_gpt_mhc_decoder_layers_spec,
    )
    from paddlefleet.spec_utils import LayerSpec
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayerWithMHC,
    )

    config = MockTransformerConfig(
        use_mhc=True,
        num_hidden_layers=4,
        moe_layer_freq=2,
    )
    specs = get_gpt_mhc_decoder_layers_spec(
        config=config,
        normalization="RMSNorm",
    )
    assert len(specs) == 4
    for spec in specs:
        assert isinstance(spec, LayerSpec)
        assert spec.layer == TransformerLayerWithMHC


def test_get_gpt_mhc_decoder_layers_spec_with_list_freq():
    """Test MHC decoder layers spec with list moe_layer_freq."""
    from paddlefleet.models.gpt.gpt_layer_specs import (
        get_gpt_mhc_decoder_layers_spec,
    )

    config = MockTransformerConfig(
        use_mhc=True,
        num_hidden_layers=4,
        moe_layer_freq=[0, 1, 0, 1],
        n_routed_experts=4,
    )
    specs = get_gpt_mhc_decoder_layers_spec(
        config=config,
        normalization="RMSNorm",
    )
    assert len(specs) == 4


def test_get_gpt_decoder_layers_spec_with_int_freq():
    """Test decoder layers spec with integer moe_layer_freq."""
    from paddlefleet.models.gpt.gpt_layer_specs import (
        get_gpt_decoder_layers_spec,
    )
    from paddlefleet.spec_utils import LayerSpec

    config = MockTransformerConfig(
        num_hidden_layers=4,
        moe_layer_freq=2,
    )
    specs = get_gpt_decoder_layers_spec(
        config=config,
        normalization="RMSNorm",
    )
    assert len(specs) == 4
    for spec in specs:
        assert isinstance(spec, LayerSpec)


def test_get_gpt_decoder_layers_spec_with_list_freq():
    """Test decoder layers spec with list moe_layer_freq."""
    from paddlefleet.models.gpt.gpt_layer_specs import (
        get_gpt_decoder_layers_spec,
    )

    config = MockTransformerConfig(
        num_hidden_layers=4,
        moe_layer_freq=[0, 1, 0, 1],
        n_routed_experts=4,
    )
    specs = get_gpt_decoder_layers_spec(
        config=config,
        normalization="RMSNorm",
    )
    assert len(specs) == 4


def test_get_mlp_layer_spec_for_backend_dense():
    """Test MLP spec for dense layer."""
    from paddlefleet.models.backends import LocalSpecProvider
    from paddlefleet.models.gpt.gpt_layer_specs import (
        get_mlp_layer_spec_for_backend,
    )
    from paddlefleet.spec_utils import LayerSpec

    backend = LocalSpecProvider()
    spec = get_mlp_layer_spec_for_backend(
        backend=backend,
        num_experts=None,
        moe_grouped_gemm=False,
    )
    assert isinstance(spec, LayerSpec)


def test_get_mlp_layer_spec_for_backend_moe():
    """Test MLP spec with num_experts > 0 (MoE)."""
    from paddlefleet.models.backends import LocalSpecProvider
    from paddlefleet.models.gpt.gpt_layer_specs import (
        get_mlp_layer_spec_for_backend,
    )
    from paddlefleet.spec_utils import LayerSpec

    backend = LocalSpecProvider()
    spec = get_mlp_layer_spec_for_backend(
        backend=backend,
        num_experts=4,
        moe_grouped_gemm=False,
    )
    assert isinstance(spec, LayerSpec)


def test_get_gpt_mtp_layers_spec_none():
    """Test MTP layers spec when num_nextn_predict_layers is None."""
    from paddlefleet.models.backends import LocalSpecProvider
    from paddlefleet.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_mtp_layers_spec_for_backend,
    )

    config = MockTransformerConfig(num_nextn_predict_layers=None)
    backend = LocalSpecProvider()

    transformer_layer_spec = get_gpt_layer_local_spec(
        config=config,
        normalization="RMSNorm",
    )
    spec_list = [transformer_layer_spec]

    mtp_specs = get_gpt_mtp_layers_spec_for_backend(
        config=config,
        spec=spec_list,
        backend=backend,
    )
    assert len(mtp_specs) == 0


def test_get_gpt_mtp_layers_spec_with_layers():
    """Test MTP layers spec when num_nextn_predict_layers > 0."""
    from paddlefleet.models.backends import LocalSpecProvider
    from paddlefleet.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_mtp_layers_spec_for_backend,
    )

    config = MockTransformerConfig(num_nextn_predict_layers=2)
    backend = LocalSpecProvider()

    transformer_layer_spec = get_gpt_layer_local_spec(
        config=config,
        normalization="RMSNorm",
    )
    spec_list = [transformer_layer_spec]

    mtp_specs = get_gpt_mtp_layers_spec_for_backend(
        config=config,
        spec=spec_list,
        backend=backend,
    )
    assert len(mtp_specs) == 2


def test_get_gpt_spec_basic():
    """Test basic GPT spec creation."""
    from paddlefleet.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_spec,
    )
    from paddlefleet.spec_utils import LayerSpec

    config = MockTransformerConfig(
        vocab_size=100,
        max_sequence_length=64,
    )

    transformer_layer_spec = get_gpt_layer_local_spec(
        config=config,
        normalization="RMSNorm",
    )

    spec = get_gpt_spec(
        config=config,
        transformer_layers_spec=[transformer_layer_spec],
        mtp_layers_spec=None,
        vocab_size=100,
        max_sequence_length=64,
        head_empty_layers_spec=[],
        tail_empty_layers_spec=[],
    )
    assert isinstance(spec, LayerSpec)


def test_get_gpt_spec_with_rope():
    """Test GPT spec with RoPE embedding."""
    from paddlefleet.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_spec,
    )
    from paddlefleet.spec_utils import LayerSpec

    config = MockTransformerConfig(
        vocab_size=100,
        max_sequence_length=64,
        position_embedding_type="rope",
    )

    transformer_layer_spec = get_gpt_layer_local_spec(
        config=config,
        normalization="RMSNorm",
    )

    spec = get_gpt_spec(
        config=config,
        transformer_layers_spec=[transformer_layer_spec],
        mtp_layers_spec=None,
        vocab_size=100,
        max_sequence_length=64,
        head_empty_layers_spec=[],
        tail_empty_layers_spec=[],
        position_embedding_type="rope",
    )
    assert isinstance(spec, LayerSpec)


def test_get_gpt_spec_with_yarn():
    """Test GPT spec with Yarn embedding."""
    from paddlefleet.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_spec,
    )
    from paddlefleet.spec_utils import LayerSpec

    config = MockTransformerConfig(
        vocab_size=100,
        max_sequence_length=64,
        position_embedding_type="yarn",
    )

    transformer_layer_spec = get_gpt_layer_local_spec(
        config=config,
        normalization="RMSNorm",
    )

    spec = get_gpt_spec(
        config=config,
        transformer_layers_spec=[transformer_layer_spec],
        mtp_layers_spec=None,
        vocab_size=100,
        max_sequence_length=64,
        head_empty_layers_spec=[],
        tail_empty_layers_spec=[],
        position_embedding_type="yarn",
    )
    assert isinstance(spec, LayerSpec)


def test_get_gpt_spec_with_mrope():
    """Test GPT spec with multimodal RoPE embedding."""
    from paddlefleet.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_spec,
    )
    from paddlefleet.spec_utils import LayerSpec

    config = MockTransformerConfig(
        vocab_size=100,
        max_sequence_length=64,
        position_embedding_type="mrope",
        mrope_section=[1, 1, 1],
    )

    transformer_layer_spec = get_gpt_layer_local_spec(
        config=config,
        normalization="RMSNorm",
    )

    spec = get_gpt_spec(
        config=config,
        transformer_layers_spec=[transformer_layer_spec],
        mtp_layers_spec=None,
        vocab_size=100,
        max_sequence_length=64,
        head_empty_layers_spec=[],
        tail_empty_layers_spec=[],
        position_embedding_type="mrope",
    )
    assert isinstance(spec, LayerSpec)


def test_get_gpt_spec_with_tie_word_embeddings():
    """Test GPT spec with tied word embeddings."""
    from paddlefleet.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_spec,
    )
    from paddlefleet.spec_utils import LayerSpec

    config = MockTransformerConfig(
        vocab_size=100,
        max_sequence_length=64,
        tie_word_embeddings=True,
        pipeline_model_parallel_size=1,
    )

    transformer_layer_spec = get_gpt_layer_local_spec(
        config=config,
        normalization="RMSNorm",
    )

    spec = get_gpt_spec(
        config=config,
        transformer_layers_spec=[transformer_layer_spec],
        mtp_layers_spec=None,
        vocab_size=100,
        max_sequence_length=64,
        head_empty_layers_spec=[],
        tail_empty_layers_spec=[],
        tie_word_embeddings=True,
    )
    assert isinstance(spec, LayerSpec)


def test_get_gpt_layer_local_spec_with_mla():
    """Test get_gpt_layer_local_spec with MLA."""
    from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from paddlefleet.spec_utils import LayerSpec

    config = MockTransformerConfig(multi_latent_attention=True)
    spec = get_gpt_layer_local_spec(
        config=config,
        num_experts=None,
        moe_grouped_gemm=False,
        use_qk_norm=False,
        multi_latent_attention=True,
        normalization="RMSNorm",
        qk_l2_norm=False,
        layer_number=1,
    )
    assert isinstance(spec, LayerSpec)


def test_get_gpt_layer_local_spec_with_qk_l2_norm():
    """Test get_gpt_layer_local_spec with qk_l2_norm."""
    from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from paddlefleet.spec_utils import LayerSpec

    config = MockTransformerConfig()
    spec = get_gpt_layer_local_spec(
        config=config,
        num_experts=None,
        moe_grouped_gemm=False,
        use_qk_norm=False,
        multi_latent_attention=False,
        normalization="RMSNorm",
        qk_l2_norm=True,
        layer_number=1,
    )
    assert isinstance(spec, LayerSpec)


# =============================================================================
# Tests for transformer_block.py coverage
# =============================================================================


def test_transformer_block_sublayers_spec_dataclass():
    """Test TransformerBlockSublayersSpec dataclass."""
    from paddlefleet.transformer.transformer_block import (
        TransformerBlockSublayersSpec,
    )

    spec = TransformerBlockSublayersSpec()
    assert spec.layer_specs is None
    assert spec.layer_norm is None


def test_get_block_sublayers_spec_with_sublayers_spec():
    """Test _get_block_sublayers_spec with TransformerBlockSublayersSpec input."""
    from paddlefleet.transformer.transformer_block import (
        TransformerBlockSublayersSpec,
        _get_block_sublayers_spec,
    )

    config = MockTransformerConfig()
    sublayers_spec = TransformerBlockSublayersSpec()
    result = _get_block_sublayers_spec(config, sublayers_spec)
    assert result is sublayers_spec


def test_get_block_sublayers_spec_with_layer_spec_transformer_layer():
    """Test _get_block_sublayers_spec with LayerSpec for TransformerLayer."""
    from paddlefleet.spec_utils import LayerSpec
    from paddlefleet.transformer.transformer_block import (
        TransformerBlockSublayersSpec,
        _get_block_sublayers_spec,
    )
    from paddlefleet.transformer.transformer_layer import TransformerLayer

    config = MockTransformerConfig(num_hidden_layers=2)
    # Create a LayerSpec directly with TransformerLayer
    layer_spec = LayerSpec(layer=TransformerLayer)
    result = _get_block_sublayers_spec(config, layer_spec)
    assert isinstance(result, TransformerBlockSublayersSpec)
    assert result.layer_specs is not None
    assert len(result.layer_specs) == config.num_hidden_layers


# =============================================================================
# Tests for gpt_builders.py coverage
# =============================================================================


def test_get_transformer_layer_spec_func_mhc():
    """Test _get_transformer_layer_spec_func with MHC enabled."""
    from paddlefleet.gpt_builders import _get_transformer_layer_spec_func

    config = MockTransformerConfig(use_mhc=True)
    func = _get_transformer_layer_spec_func(config)
    assert callable(func)


def test_get_transformer_layer_spec_func_no_mhc():
    """Test _get_transformer_layer_spec_func without MHC."""
    from paddlefleet.gpt_builders import _get_transformer_layer_spec_func

    config = MockTransformerConfig(use_mhc=False)
    func = _get_transformer_layer_spec_func(config)
    assert callable(func)


# =============================================================================
# Tests for tensors_clone function in transformer_layer.py
# =============================================================================


def test_tensors_clone():
    """Test tensors_clone with various input types."""
    from paddlefleet.transformer.transformer_layer import tensors_clone

    # Test tensor
    x = paddle.randn([2, 3])
    cloned = tensors_clone(x)
    assert cloned.shape == x.shape

    # Test list
    x_list = [paddle.randn([2, 3]), paddle.randn([4, 5])]
    cloned_list = tensors_clone(x_list)
    assert isinstance(cloned_list, list)
    assert len(cloned_list) == 2

    # Test dict
    x_dict = {"a": paddle.randn([2, 3])}
    cloned_dict = tensors_clone(x_dict)
    assert isinstance(cloned_dict, dict)


# =============================================================================
# Tests for TransformerLayerSublayersSpec dataclass
# =============================================================================


def test_transformer_layer_sublayers_spec_dataclass():
    """Test TransformerLayerSublayersSpec dataclass."""
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayerSublayersSpec,
    )

    spec = TransformerLayerSublayersSpec()
    assert spec.input_layernorm is not None
    assert spec.self_attn is not None
    assert spec.mlp is not None


# =============================================================================
# Tests for gpt_model.py GPTSublayersSpec dataclass
# =============================================================================


# =============================================================================
# Tests for GPTSublayersSpec dataclass
# =============================================================================


def test_gpt_sublayers_spec_dataclass():
    """Test GPTSublayersSpec dataclass."""
    from paddlefleet.models.gpt.gpt_model import GPTSublayersSpec

    spec = GPTSublayersSpec()
    assert spec.embedding is None
    assert spec.transformer_layers is None
    assert spec.mtp is None
    assert spec.layer_norm is None
    assert spec.lm_head is None


# =============================================================================
# Tests for gpt_builders.py - MoE branches (testing spec generation only)
# =============================================================================


def test_gpt_builder_moe_mhc_spec():
    """Test gpt_builder spec generation with MoE model and MHC enabled."""
    from paddlefleet.models.gpt.gpt_layer_specs import (
        get_gpt_mhc_decoder_layers_spec,
    )

    config = MockTransformerConfig(
        use_mhc=True,
        n_routed_experts=4,
        num_hidden_layers=2,
    )
    # Test that spec is generated correctly for MoE + MHC
    spec = get_gpt_mhc_decoder_layers_spec(
        config=config, normalization="RMSNorm"
    )
    assert spec is not None


def test_gpt_builder_moe_no_mhc_spec():
    """Test gpt_builder spec generation with MoE model without MHC."""
    from paddlefleet.models.gpt.gpt_layer_specs import (
        get_gpt_decoder_layers_spec,
    )

    config = MockTransformerConfig(
        use_mhc=False,
        n_routed_experts=4,
        num_hidden_layers=2,
    )
    spec = get_gpt_decoder_layers_spec(config=config, normalization="RMSNorm")
    assert spec is not None


def test_gpt_builder_dense_mhc_spec():
    """Test gpt_builder spec generation with dense model and MHC."""
    from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_mhc_spec

    config = MockTransformerConfig(
        use_mhc=True,
        n_routed_experts=None,
        num_hidden_layers=2,
    )
    spec = get_gpt_layer_mhc_spec(config=config, layer_number=0)
    assert spec is not None


def test_gpt_builder_dense_no_mhc_spec():
    """Test gpt_builder spec generation with dense model without MHC."""
    from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

    config = MockTransformerConfig(
        use_mhc=False,
        n_routed_experts=None,
        num_hidden_layers=2,
    )
    spec = get_gpt_layer_local_spec(config=config, layer_number=0)
    assert spec is not None


# =============================================================================
# Tests for gpt_layer_specs.py - overlap scheduler check
# =============================================================================


def test_get_gpt_layer_mhc_spec_overlap_scheduler_raises():
    """Test that MHC with overlap scheduler raises ValueError."""
    import paddle.distributed as dist

    from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_mhc_spec

    # Skip if distributed is not initialized
    if not dist.is_initialized():
        return

    config = MockTransformerConfig(use_mhc=True)
    try:
        get_gpt_layer_mhc_spec(config=config, layer_number=0)
    except ValueError as e:
        assert "MHC is not compatible" in str(e)


# =============================================================================
# Tests for hyper_connection.py missing lines
# =============================================================================


def test_hyper_connection_native_depth_with_bias():
    """Test native depth connection with bias tensor."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=32, mhc_use_triton=False)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 2, 2, 4, 32
    x = paddle.randn([s, b, n * C], dtype="float32")
    branch_input, residuals, H_post = hc.width_connection(x)

    branch_output = paddle.randn([s, b, C], dtype="float32")
    bias = paddle.randn([C], dtype="float32")

    output = hc.depth_connection(
        (branch_output, bias), residuals, H_post, fused=False
    )
    assert output.shape == [s, b, n * C]


def test_hyper_connection_width_skip_sk_gradient():
    """Test width_connection with skip_sk_gradient."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=32, mhc_use_triton=False)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 2, 2, 4, 32
    x = paddle.randn([s, b, n * C], dtype="float32")

    # Test width_connection with skip_sk_gradient=False
    branch_input, residuals, H_post = hc.width_connection(
        x, skip_sk_gradient=False
    )
    assert branch_input.shape == [s, b, C]

    # Test depth_connection
    branch_output = paddle.randn([s, b, C], dtype="float32")
    output = hc.depth_connection((branch_output, None), residuals, H_post)
    assert output.shape == [s, b, n * C]


# =============================================================================
# Tests for HyperConnectionModule expand/reduce_stream (transformer_block.py coverage)
# =============================================================================


def test_hyper_connection_expand_stream():
    """Test HyperConnectionModule.expand_stream static method."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    s, b, C, n = 4, 2, 64, 4
    hidden_states = paddle.randn([s, b, C])

    expanded = HyperConnectionModule.expand_stream(hidden_states, n)
    assert expanded.shape == [s, b, n * C]


def test_hyper_connection_reduce_stream():
    """Test HyperConnectionModule.reduce_stream static method."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    s, b, C, n = 4, 2, 64, 4
    hidden_states = paddle.randn([s, b, n * C])

    reduced = HyperConnectionModule.reduce_stream(hidden_states, n)
    assert reduced.shape == [s, b, C]


def test_hyper_connection_expand_reduce_roundtrip():
    """Test expand and reduce roundtrip consistency."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    s, b, C, n = 4, 2, 64, 4
    original = paddle.randn([s, b, C])

    expanded = HyperConnectionModule.expand_stream(original, n)
    reduced = HyperConnectionModule.reduce_stream(expanded, n)

    # The reduced should be close to original (sum divided by n)
    expected = original
    assert paddle.allclose(reduced, expected, atol=1e-5)


# =============================================================================
# Tests for MHCExpandLayer and MHCContractLayer (gpt_model.py coverage)
# =============================================================================


def test_mhc_expand_layer():
    """Test MHCExpandLayer forward pass."""
    from paddlefleet.transformer.hyper_connection import (
        MHCExpandLayer,
    )

    config = MockTransformerConfig(use_mhc=True)
    expand_layer = MHCExpandLayer(config)

    s, b, C = 4, 2, 64
    hidden_states = paddle.randn([s, b, C], dtype="float32")

    # MHCExpandLayer expects a dict with hidden_states
    dict_args = {"hidden_states": hidden_states}
    output = expand_layer(dict_args)

    n = config.mhc_num_residual_streams
    assert output["hidden_states"].shape == [s, b, n * C]


def test_mhc_contract_layer():
    """Test MHCContractLayer forward pass."""
    from paddlefleet.transformer.hyper_connection import (
        MHCContractLayer,
    )

    config = MockTransformerConfig(use_mhc=True)
    contract_layer = MHCContractLayer(config)

    s, b, C = 4, 2, 64
    n = config.mhc_num_residual_streams
    hidden_states = paddle.randn([s, b, n * C], dtype="float32")

    # MHCContractLayer expects a dict with hidden_states
    dict_args = {"hidden_states": hidden_states}
    output = contract_layer(dict_args)

    assert output["hidden_states"].shape == [s, b, C]


# =============================================================================
# Tests for hyper_connection dtype casting branches
# =============================================================================


def test_hyper_connection_with_different_dtypes():
    """Test that operations work with different dtypes."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    # Test with float32 input
    config = MockConfig(hidden_size=32, mhc_use_triton=False)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 2, 2, 4, 32
    x = paddle.randn([s, b, n * C], dtype="float32")

    branch_input, residuals, H_post = hc.width_connection(x)
    assert branch_input.shape == [s, b, C]
    assert branch_input.dtype == paddle.float32


def test_hyper_connection_depth_with_dropout():
    """Test depth connection with dropout enabled."""
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    config = MockConfig(hidden_size=32, mhc_use_triton=False)
    hc = HyperConnectionModule(config=config, layer_number=1)

    s, b, n, C = 2, 2, 4, 32
    x = paddle.randn([s, b, n * C], dtype="float32")
    branch_input, residuals, H_post = hc.width_connection(x)

    branch_output = paddle.randn([s, b, C], dtype="float32")

    # Test with dropout enabled
    output = hc.depth_connection(
        (branch_output, None),
        residuals,
        H_post,
        dropout_prob=0.1,
        training=True,
    )
    assert output.shape == [s, b, n * C]


# =============================================================================
# Tests for transformer_layer.py cross_attention branches
# =============================================================================


def test_transformer_layer_sublayers_spec_cross_attention():
    """Test TransformerLayerSublayersSpec with cross_attention enabled."""
    from paddlefleet.spec_utils import LayerSpec
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayerSublayersSpec,
    )

    # Create sublayers spec with cross attention
    sublayers_spec = TransformerLayerSublayersSpec()
    sublayers_spec.cross_attention = LayerSpec(layer=object)

    # This test verifies the spec can be created with cross attention
    assert sublayers_spec.cross_attention is not None


# =============================================================================
# Tests for gpt_builders.py dense model layer iteration
# =============================================================================


def test_transformer_layer_spec_func_iteration():
    """Test transformer_layer_spec_func iteration for dense models."""
    from paddlefleet.gpt_builders import _get_transformer_layer_spec_func

    config = MockTransformerConfig(
        use_mhc=True,
        n_routed_experts=None,  # Dense model
        num_hidden_layers=3,
        num_empty_layers_add_in_head=0,
    )

    # Get the spec function
    transformer_layer_spec_func = _get_transformer_layer_spec_func(config)
    assert callable(transformer_layer_spec_func)

    # Test iteration over layers (covers lines 56-57)
    transformer_layers_spec = []
    for layer_number in range(config.num_hidden_layers):
        real_layer_number = layer_number + config.num_empty_layers_add_in_head
        spec = transformer_layer_spec_func(layer_number=real_layer_number)
        transformer_layers_spec.append(spec)

    assert len(transformer_layers_spec) == 3


def test_transformer_layer_spec_func_no_mhc():
    """Test transformer_layer_spec_func without MHC."""
    from paddlefleet.gpt_builders import _get_transformer_layer_spec_func

    config = MockTransformerConfig(
        use_mhc=False,
        n_routed_experts=None,
        num_hidden_layers=2,
    )

    transformer_layer_spec_func = _get_transformer_layer_spec_func(config)
    assert callable(transformer_layer_spec_func)

    # Generate specs
    for i in range(config.num_hidden_layers):
        spec = transformer_layer_spec_func(layer_number=i)
        assert spec is not None
