# Accuracy-compatible regressions

This suite guards `use_accuracy_compatible` behavior: target selection, forward
values, gradient attachment and rounding, and preservation of the ordinary path.
It lives alongside `coverage_test` so its purpose and execution are explicit.
`ieee` is not a separate mode or environment gate; test names describe the
operation they exercise. Legacy environment independence is tested against real
production imports in `test_configuration.py`.

Run from the repository root with the project's Paddle environment:

```bash
python -m pytest -q tests/single_card_tests/accuracy_compatible_test
```

`ci/single_card_test.sh` already discovers this directory recursively. No separate
workflow or allowlist is needed. Tensor tests select `gpu:0` through the local
pytest fixture and restore the previous device after each test; a missing GPU is
an error, not a silent CPU fallback. Direct module execution is supported where
a `__main__` block exists, but pytest is the common suite entrypoint.

## Scope of the assertions

- Configuration/dispatch tests check enabled and disabled modes and actual group
  sizes, including mismatches with configured sizes. MLP dispatch and MTP loss
  attachment execute the complete production methods with their unrelated
  dependencies stubbed.
- Native tensor tests compare forward values and backward gradients. Assertions
  described as bitwise compare raw bytes (with matching dtype and shape), not
  approximate equality. HF reference tests also retain tolerance checks where
  they test mathematical properties rather than bitwise identity.
- Topology/layout tests isolate production fragments with explicit group adapters;
  they do not establish real distributed communication correctness.
- `test_glm52_weight_gradient_contracts.py` uses NumPy adapters to check orientation,
  empty-expert indexing, and cast/dispatch policy. It is not GPU numerical evidence.

These focused regressions do not replace multi-rank training, cross-framework
loss/checkpoint comparison, or end-to-end model acceptance. Add behavior regressions
here when they protect this numerical mode; keep general model structure and
coverage-only tests in their existing suites.
