# PaddleFleet Ops Tests

This directory contains unit tests for PaddleFleet ops module and utilities.

## Test Files

| File | Source File | Coverage Target |
|------|-------------|-----------------|
| `test_ops_init.py` | `src/paddlefleet/ops/__init__.py` | Tests for module import, availability flags, _build_notice, _safe_load_ecosystem_lib |
| `test_ops_utils.py` | `src/paddlefleet/ops/utils.py` | Tests for ModuleContext, HardwareIncompatibleBlocker, import_custom_ops, namespace utilities |

## Running Tests

```bash
# Run all ops tests
python -m pytest tests/single_card_tests/ai_edited_test/ops/ -v

# Run a specific test file
python -m pytest tests/single_card_tests/ai_edited_test/ops/test_ops_utils.py -v

# Run with unittest
python -m unittest tests.single_card_tests.ai_edited_test.ops.test_ops_utils
```

## Notes

- Tests for `deep_ep`, `deep_gemm`, and `quack` submodules are excluded per project requirements
- Hardware-dependent tests are conditionally skipped based on CUDA availability
- Module import tests validate both success and failure paths
