# Extensions Unit Tests

This directory contains unit tests for PaddleFleet's custom CUDA extensions and triton operators.

## Test Files

| File | Source Under Test | Description |
|------|------------------|-------------|
| `test_extensions_init.py` | `src/paddlefleet/_extensions/__init__.py` | Package import and error handling |
| `test_ops.py` | `src/paddlefleet/_extensions/ops.py` | All custom ops: filter_scores, swiglu, cumsum, zip/unzip, router, etc. |
| `test_rr_attn_estimate_triton_op.py` | `src/paddlefleet/_extensions/flashmask/rr_attn_estimate_triton_op.py` | Triton-based attention estimation: raw ptrs extraction, stride prep, input validation |
| `test_block_mask_utils.py` | `src/paddlefleet/_extensions/flashmask/block_mask_utils.py` | Triton block mask utils: top-p kernel, bitonic sort, flashmask apply |
| `test_index_utils.py` | `src/paddlefleet/_extensions/flashmask/index_utils.py` | Index preprocessing: max/min preparation for stride-level masking |

## Running Tests

```bash
cd /path/to/PaddleFleet
python -m pytest tests/single_card_tests/ai_edited_test/extensions/ -v
```

Or run a single test file:

```bash
python tests/single_card_tests/ai_edited_test/extensions/test_ops.py
```

## Notes

- All tests use `unittest.TestCase` style.
- Custom CUDA ops (`_C_ops._run_custom_op`) are mocked since they require CUDA kernels.
- Triton kernels are mocked since they require GPU execution.
- The `ops.py` file is auto-generated during build; tests target the installed version.
