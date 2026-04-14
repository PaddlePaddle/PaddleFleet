# Refined Recompute Tests

Unit tests for `paddlefleet.refined_recompute` module.

## Files

| Test File | Source File | Coverage Target |
|-----------|-------------|-----------------|
| `test_flash_attn.py` | `refined_recompute/flash_attn.py` | `_get_fa_version`, `flashattn_auto_cast`, `FlashAttnFunctor`, `FlashMaskAttnFunctor`, `FlashMaskAttnCpFunctor`, `RefinedRcomputeFlashAttention`, `RefinedRcomputeFlashMaskAttention`, `RefinedRcomputeFlashMaskCpAttention` |
| `test_queue_check.py` | `refined_recompute/queue_check.py` | `RefinedRcomputeQueue`, `global_rr_queue_log` |

## Running Tests

```bash
# Run all recompute tests
python -m pytest tests/single_card_tests/ai_edited_test/recompute/ -v

# Run individual test file
python -m pytest tests/single_card_tests/ai_edited_test/recompute/test_flash_attn.py -v
python -m pytest tests/single_card_tests/ai_edited_test/recompute/test_queue_check.py -v
```

## Notes

- CUDA kernel calls (`_C_ops.flash_attn`, `_C_ops.flash_attn_v3`, etc.) are mocked using `unittest.mock.patch`.
- The `framework._dygraph_tracer()._has_grad` flag is mocked to test both first-forward and second-forward paths.
- Version branching (v2, v3, v4, invalid) is thoroughly covered via `_get_fa_version` mocking.
