# Common Tests

Unit tests for top-level `paddlefleet` package initialization and metadata.

## Files

| Test File | Source File | Coverage Target |
|-----------|-------------|-----------------|
| `test_paddlefleet_init.py` | `paddlefleet/__init__.py`, `paddlefleet/package_info.py` | Import checks, `__all__` exports, metadata fields, aliases |

## Running Tests

```bash
# Run all common tests
python -m pytest tests/single_card_tests/ai_edited_test/common/ -v
```

## Notes

- These tests verify that the top-level package is correctly structured and exports all expected symbols.
