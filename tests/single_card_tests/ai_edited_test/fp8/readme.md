# FP8 Tests

Unit tests for `paddlefleet.fp8` module.

## Files

| Test File | Source File | Coverage Target |
|-----------|-------------|-----------------|
| `test_quantization.py` | `fp8/quantization.py` | `get_quant_func` (blockwise recipe, invalid recipes) |
| `test_linear.py` | `fp8/linear.py` | `FP8Linear`, `_FP8Gemm` initialization and forward |
| `test_utils.py` | `fp8/utils.py` | `is_fp8_tensor` (valid/invalid inputs, dtype checks) |

## Running Tests

```bash
# Run all FP8 tests
python -m pytest tests/single_card_tests/ai_edited_test/fp8/ -v

# Run individual test file
python -m pytest tests/single_card_tests/ai_edited_test/fp8/test_quantization.py -v
python -m pytest tests/single_card_tests/ai_edited_test/fp8/test_linear.py -v
python -m pytest tests/single_card_tests/ai_edited_test/fp8/test_utils.py -v
```

## Notes

- Tests requiring CUDA use `@unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")`.
- Imports are placed inside test methods to avoid import-time side effects.
