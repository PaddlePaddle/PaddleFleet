# AI-Edited Fusion Tests

## Overview

Unit tests for the `paddlefleet.fusions` module, written to improve code coverage.
These tests complement the existing test files from `20260402` by targeting
previously uncovered code paths.

## Test Files

| File | Source Module | Coverage Target |
|------|--------------|-----------------|
| `test_ai_fused_bias_gelu_extra.py` | `fused_bias_gelu` | Autograd (forward/backward), float16, batched inputs, gradient correctness |
| `test_ai_fused_bias_geglu_extra.py` | `fused_bias_geglu` | `BiasGeGLUFunction`, `GeGLUFunction`, `WeightedQuickGeGLUFunction`, `WeightedBiasQuickGeGLUFunction` autograd, fp8 storage, correctness verification |
| `test_ai_fused_bias_swiglu_extra.py` | `fused_bias_swiglu` | `BiasSwiGLUFunction`, `SwiGLUFunction`, `WeightedSwiGLUFunction` autograd, CUDA/non-CUDA paths for `swiglu_back`, fp8 storage, cpu offload |
| `test_ai_fused_swiglu_scale_extra.py` | `fused_swiglu_scale` | CUDA forward/backward paths, 3D/4D inputs, dtype handling, CPU fallback correctness |
| `test_ai_fused_softmax_extra.py` | `fused_softmax` | `SoftmaxOne`, fp16/bf16 conversion, sliding window mask, softmax offset, assertion errors, various `AttnMaskType` values |
| `test_ai_fused_bias_dropout.py` | `fused_bias_dropout` | `_bias_dropout_add_func` (inplace/non-inplace, with/without bias), `bias_dropout_add_unfused`, `get_bias_dropout_add` |

## Running the Tests

```bash
# Run all fusion tests
python -m pytest tests/single_card_tests/ai_edited_test/fusions/ -v

# Run a single test file
python -m pytest tests/single_card_tests/ai_edited_test/fusions/test_ai_fused_bias_gelu_extra.py -v

# Run via unittest
python -m unittest tests.single_card_tests.ai_edited_test.fusions.test_ai_fused_bias_gelu_extra
```

## Notes

- CUDA-dependent tests are skipped automatically when CUDA is not available.
- CPU fallback paths are tested using `unittest.mock.patch` on `paddle.is_compiled_with_cuda`.
- All tests follow `unittest.TestCase` style with PaddlePaddle Apache 2.0 license headers.
