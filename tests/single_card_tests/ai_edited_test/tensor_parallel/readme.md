# Tensor Parallel Unit Tests (AI Edited)

## Overview

This directory contains unit tests for the `paddlefleet.tensor_parallel` module,
aimed at improving code coverage across all tensor parallel components.

## Test Files

| Test File | Source File | Description |
|---|---|---|
| `test_ai_data.py` | `tensor_parallel/data.py` | Tests for broadcast_data, _check_data_types, _build_key_size_numel_dictionaries |
| `test_ai_cross_entropy.py` | `tensor_parallel/cross_entropy.py` | Tests for VocabParallelCrossEntropy static methods, forward/backward, label smoothing |
| `test_ai_utils.py` | `tensor_parallel/utils.py` | Tests for split_tensor_along_last_dim, VocabUtility, split/gather 1D chunks |
| `test_ai_random.py` | `tensor_parallel/random.py` | Tests for CudaRNGStatesTracker, RNG initialization, CheckpointWithoutOutput |
| `test_ai_mappings.py` | `tensor_parallel/mappings.py` | Tests for all autograd communication primitives (copy, scatter, gather, reduce_scatter) |
| `test_ai_layers.py` | `tensor_parallel/layers.py` | Tests for Linear, ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding, weight init |

## Running Tests

```bash
# Run all tests in this directory
python -m pytest tests/single_card_tests/ai_edited_test/tensor_parallel/ -v

# Run a specific test file
python -m pytest tests/single_card_tests/ai_edited_test/tensor_parallel/test_ai_data.py -v

# Run a specific test class
python -m pytest tests/single_card_tests/ai_edited_test/tensor_parallel/test_ai_layers.py::TestLinearLayer -v

# Run with coverage
python -m pytest tests/single_card_tests/ai_edited_test/tensor_parallel/ -v --cov=paddlefleet.tensor_parallel
```

## Design Notes

- Tests use `unittest.TestCase` style
- Distributed initialization is mocked to enable single-card testing
- CUDA-dependent tests are guarded with `@unittest.skipIf(not paddle.is_compiled_with_cuda())`
- Imports are placed inside test methods to avoid import-time side effects
