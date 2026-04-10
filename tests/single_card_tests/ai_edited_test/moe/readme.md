# AI-Edited MoE Unit Tests

This directory contains unit tests for the MoE (Mixture of Experts) and related transformer modules in PaddleFleet.

## Test Files

| Test File | Source File | Description |
|-----------|-----------|-------------|
| `test_ai_fp8_utils.py` | `src/paddlefleet/transformer/moe/fp8_utils.py` | FP8 utilities for MoE expert computation |
| `test_ai_moe_layer.py` | `src/paddlefleet/transformer/moe/moe_layer.py` | MoE layer orchestration |
| `test_ai_moe_router.py` | `src/paddlefleet/transformer/moe/moe_router.py` | MoE token routing logic |
| `test_ai_moe_utils.py` | `src/paddlefleet/transformer/moe/moe_utils.py` | MoE utility functions (permute, unpermute, etc.) |
| `test_ai_moe_expert.py` | `src/paddlefleet/transformer/moe/moe_expert.py` | Grouped and standard MLP expert implementations |
| `test_ai_token_dispatcher.py` | `src/paddlefleet/transformer/moe/token_dispatcher.py` | Token dispatch/combine for expert parallelism |
| `test_ai_fused_a2a.py` | `src/paddlefleet/transformer/moe/fused_a2a.py` | Fused all-to-all communication kernels |
| `test_ai_fusion_layer_utils.py` | `src/paddlefleet/transformer/moe/fusion_layer_utils.py` | Fusion layer utilities (zip/unzip nodes) |
| `test_ai_multi_token_prediction.py` | `src/paddlefleet/transformer/multi_token_prediction.py` | Multi-Token Prediction (MTP) layer |
| `test_ai_multi_latent_attention.py` | `src/paddlefleet/transformer/multi_latent_attention.py` | Multi-Latent Attention (MLA) layer |
| `test_ai_block_attn_res.py` | `src/paddlefleet/transformer/block_attn_res.py` | Block Attention Residuals |

## Running Tests

```bash
# Run all tests in this directory
python -m pytest tests/single_card_tests/ai_edited_test/moe/ -v

# Run a single test file
python -m pytest tests/single_card_tests/ai_edited_test/moe/test_ai_fp8_utils.py -v

# Run with unittest
python -m unittest tests.single_card_tests.ai_edited_test.moe.test_ai_fp8_utils
```

## Design Principles

- **unittest.TestCase style** with standard Python unittest framework
- **Distributed operations mocked** to allow single-card testing
- **No CUDA required** for most tests (distributed/CUDA-dependent paths are mocked)
- **English-only comments** in test files
- **PaddlePaddle Apache 2.0 License** header in each file
