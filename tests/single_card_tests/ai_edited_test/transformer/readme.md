# AI-Edited Tests for Transformer Module

This directory contains unit tests for the PaddleFleet transformer module,
automatically generated with focus on improving code coverage.

## Test Files

| Test File | Source File | Priority |
|-----------|-------------|----------|
| `test_ai_utils.py` | `utils.py` | High (53.8%) |
| `test_ai_paddle_norm.py` | `paddle_norm.py` | High (82.4%) |
| `test_ai_mlp.py` | `mlp.py` | High (77.8%) |
| `test_ai_attention.py` | `attention.py` | High (80.9%) |
| `test_ai_dot_product_attention.py` | `dot_product_attention.py` | Medium (87.1%) |
| `test_ai_transformer_block.py` | `transformer_block.py` | Medium (72.6%) |
| `test_ai_transformer_layer.py` | `transformer_layer.py` | Critical (47.7%) |
| `test_ai_transformer_encoder.py` | `transformer_encoder.py` | Critical (16.5%) |

## Running Tests

```bash
# Run all tests in this directory
python -m pytest tests/single_card_tests/ai_edited_test/transformer/ -v

# Run a specific test file
python -m pytest tests/single_card_tests/ai_edited_test/transformer/test_ai_utils.py -v

# Run with unittest
python -m unittest tests.single_card_tests.ai_edited_test.transformer.test_ai_utils
```

## Notes

- Tests use `unittest.TestCase` style
- CUDA-dependent tests are skipped on CPU-only environments
- Distributed calls (e.g., `paddle.distributed.is_initialized`) are mocked
- Only English comments are used in test files
