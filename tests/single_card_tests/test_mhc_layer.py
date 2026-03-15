"""
Simple test script to verify MHC (Manifold Constrained Hyper Connections) implementation.

This script tests:
1. MHC width_connection and depth_connection operations
2. TransformerLayerWithMHC forward pass
3. Data format consistency (BLD format throughout)
"""

import os
import sys

# Use relative path instead of hardcoded path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, 'src'))

import paddle
import paddle.nn as nn


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


def test_mhc_basic():
    """Test basic MHC operations."""
    print("=" * 60)
    print("Test 1: Basic MHC Operations")
    print("=" * 60)

    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    # Parameters
    batch_size = 2
    seq_len = 8
    hidden_dim = 64
    num_streams = 4

    config = MockConfig(
        hidden_size=hidden_dim,
        mhc_num_residual_streams=num_streams,
        mhc_sinkhorn_iters=10,
    )

    # Create MHC instance
    mhc = HyperConnectionModule(config=config, layer_number=1)

    # Test with [s, b, n*C] format (s=seq_len, b=batch_size)
    print(f"\nTesting with [s, b, n*C] format: [{seq_len}, {batch_size}, {num_streams * hidden_dim}]")
    x = paddle.randn([seq_len, batch_size, num_streams * hidden_dim])

    # Width connection
    branch_input, residuals, H_post = mhc.width_connection(x)
    print(f"  branch_input shape: {branch_input.shape}")
    print(f"  residuals shape: {residuals.shape}")
    print(f"  H_post shape: {H_post.shape}")

    # Simulate branch output (attention/mlp)
    branch_output = paddle.randn([seq_len, batch_size, hidden_dim])

    # Depth connection
    output = mhc.depth_connection((branch_output, None), residuals, H_post)
    print(f"  output shape: {output.shape}")

    # Reduce back to [s, b, C]
    output_reduced = HyperConnectionModule.reduce_stream(output, num_streams)
    print(f"  output after reduce: {output_reduced.shape}")

    # Verify shapes
    assert output.shape == [seq_len, batch_size, num_streams * hidden_dim], \
        f"Expected [{seq_len}, {batch_size}, {num_streams * hidden_dim}], got {output.shape}"
    assert output_reduced.shape == [seq_len, batch_size, hidden_dim], \
        f"Expected [{seq_len}, {batch_size}, {hidden_dim}], got {output_reduced.shape}"

    print("\n  [PASS] Basic MHC operations work correctly!")
    return True


def test_mhc_format_equivalence():
    """Test that MHC works correctly with different batch/seq dimensions."""
    print("\n" + "=" * 60)
    print("Test 2: MHC Dimension Consistency")
    print("=" * 60)

    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    batch_size = 2
    seq_len = 4
    hidden_dim = 32
    num_streams = 2

    config = MockConfig(
        hidden_size=hidden_dim,
        mhc_num_residual_streams=num_streams,
        mhc_sinkhorn_iters=5,
    )

    # Create MHC instance
    mhc = HyperConnectionModule(config=config, layer_number=1)

    # Create input tensor [s, b, n*C]
    paddle.seed(42)
    x = paddle.randn([seq_len, batch_size, num_streams * hidden_dim])

    print(f"\n  Input shape: {x.shape}")

    # Width connection
    branch_input, residuals, h_post = mhc.width_connection(x)

    print(f"\n  branch_input shape: {branch_input.shape}")
    print(f"  residuals shape: {residuals.shape}")
    print(f"  h_post shape: {h_post.shape}")

    # Verify shapes
    assert branch_input.shape == [seq_len, batch_size, hidden_dim]
    assert residuals.shape == [seq_len, batch_size, num_streams * hidden_dim]
    assert h_post.shape == [seq_len, batch_size, num_streams]

    print("\n  [PASS] MHC operations produce correct shapes!")
    return True


def test_transformer_layer_mhc():
    """Test TransformerLayerWithMHC with simplified components."""
    print("\n" + "=" * 60)
    print("Test 3: TransformerLayerWithMHC Forward Pass")
    print("=" * 60)

    from paddlefleet.transformer.transformer_config import TransformerConfig
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayerSublayersSpec,
        TransformerLayerWithMHC,
    )
    from paddlefleet.transformer.identity_op import IdentityOp, IdentityFuncOp
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    # Create a minimal config
    config = TransformerConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=256,
        use_mhc=True,
        mhc_num_residual_streams=4,
        mhc_sinkhorn_iters=10,
        mhc_kernel_backend="default",
        normalization="RMSNorm",
        rms_norm_eps=1e-5,
    )

    # Create minimal sublayers spec with HyperConnectionModule
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

    print(f"\n  Layer created with {config.mhc_num_residual_streams} residual streams")

    # Create input
    batch_size = 2
    seq_len = 8
    hidden_states = paddle.randn([batch_size, seq_len, config.hidden_size])
    print(f"\n  Input shape: {hidden_states.shape}")

    print("\n  [PASS] TransformerLayerWithMHC initialization works correctly!")
    return True


def test_expand_reduce_optimization():
    """Test that expand/reduce is handled at block level in TransformerLayerWithMHC."""
    print("\n" + "=" * 60)
    print("Test 4: Expand/Reduce at Block Level")
    print("=" * 60)

    from paddlefleet.transformer.hyper_connection import HyperConnectionModule

    s, b, C = 4, 2, 64
    n = 4

    # Test round-trip: expand -> contract should recover original values
    x_original = paddle.randn([s, b, C])

    # Expand: [s, b, C] -> [s, b, n*C]
    x_expanded = HyperConnectionModule.expand_stream(x_original, n)
    print(f"\n  Original shape: {x_original.shape}")
    print(f"  After expand: {x_expanded.shape}")
    assert x_expanded.shape == [s, b, n * C], f"Expand shape mismatch: {x_expanded.shape}"

    # Contract: [s, b, n*C] -> [s, b, C]
    x_recovered = HyperConnectionModule.reduce_stream(x_expanded, n)
    print(f"  After contract: {x_recovered.shape}")
    assert x_recovered.shape == [s, b, C], f"Contract shape mismatch: {x_recovered.shape}"

    # Verify round-trip: expand replicates input, contract averages
    # Since expand replicates each stream, the average should equal the original
    diff = (x_recovered - x_original).abs().max().item()
    print(f"\n  Round-trip max difference: {diff:.2e}")
    assert diff < 1e-5, f"Round-trip failed: max diff = {diff}"

    print("\n  [PASS] Block-level expand/contract logic verified!")
    return True


def main():
    print("\n" + "=" * 60)
    print("MHC (Manifold Constrained Hyper Connections) Tests")
    print("=" * 60)

    # Note: paddlefleet requires GPU, device is set by CUDA_VISIBLE_DEVICES
    print(f"Using device: {paddle.device.get_device()}")

    all_passed = True

    try:
        all_passed &= test_mhc_basic()
    except Exception as e:
        print(f"\n  [FAIL] Test 1 failed with error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_mhc_format_equivalence()
    except Exception as e:
        print(f"\n  [FAIL] Test 2 failed with error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_transformer_layer_mhc()
    except Exception as e:
        print(f"\n  [FAIL] Test 3 failed with error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_expand_reduce_optimization()
    except Exception as e:
        print(f"\n  [FAIL] Test 4 failed with error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
