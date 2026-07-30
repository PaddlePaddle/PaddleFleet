# PaddleFleet Review Checklist

Read only sections matching the changed paths or behavior.

## Configuration and Public Contracts

- Give new dataclass/config fields backward-compatible defaults; trace removals, renames, type changes, and semantic changes through every consumer.
- Keep `training/arguments.py` and `training/yaml_arguments.py` synchronized with public configuration.
- Check state-dict, serialization, hash/equality, and downstream model compatibility for public contract changes.
- Verify PR title/body claims exactly match implemented values, paths, backends, shapes, dtypes, and tests.

## Parallelism and Distributed State

- Verify process-group creation order, membership, teardown, and reinitialization across `parallel_state.py` and `process_groups_config.py`.
- For collectives, verify the intended group, tensor shape/dtype, autograd semantics, and matching inverse/paired collective.
- For TP/PP/EP/CP combinations, check rank-local shapes, sequence/position partitioning, RNG synchronization, shared weights, and pipeline boundaries.
- Require representative multi-card coverage for changed distributed behavior; inspect blacklist additions for hidden regressions.

## Transformer, MoE, and Models

- Keep model `layer_specs.py`, backends registration, package exports, and model construction synchronized.
- Verify attention masks, position/sequence metadata, prefill/training semantics, and mixed dense/sparse attention routing.
- For MoE, verify router top-k/normalization, dispatch/combine ordering, capacity/drop behavior, expert indices, shared experts, and gradient symmetry.
- Check MTP zero-layer/default paths and every changed feature flag combination.

## FP8, Fusion, and Accelerated Backends

- Verify scale granularity, layout, accumulation dtype, overflow/NaN handling, and forward/backward numerical parity.
- Keep fused and unfused paths behaviorally equivalent; require a fallback or explicit architecture/shape/dtype guard.
- Check Triton/TileLang/cuDNN launch geometry and support guards across target SMs and edge shapes.
- Flag hot-path host synchronization or copies such as `.cpu()`, `.numpy()`, unnecessary `to_tensor`, or `contiguous()`.

## Custom Operators and Build

- Register new C++/CUDA sources in the correct build backend/setup path and keep op names/signatures aligned with Python facades.
- Validate shape/dtype inference, device/stream use, integer width, bounds, allocation lifetime, and launch error handling.
- Keep extension imports and version requirements compatible; add focused tests under `tests/single_card_tests/custom_ops/`.
- Treat vendored third-party changes as exceptional and require upstream provenance plus synchronization intent.

## Recompute and CUDA Graph

- Verify recompute forward/backward equivalence, saved tensors, RNG state, and no missing gradients.
- Keep CUDA graph capture allocations and addresses stable; guard dynamic shapes and preserve a safe eager fallback.

## Tests and CI

- Reject tests that swallow assertions, patch the function under test, use `assert True`, or omit boundary/error cases.
- Prefer parameterization over duplicated test bodies and place tests in the closest single-card/multi-card subsystem.
- When changing `ci/rules`, update rule tests and snapshots. When changing CI scripts/workflows, verify conditions, artifacts, and path filters.
- Check new dependencies in `pyproject.toml` or the applicable build requirements and keep lockfile changes intentional.

## High-Signal Patterns

- Runtime validation implemented with Python `assert` instead of an explicit exception.
- Broad `except Exception` that suppresses failures without logging or recovery.
- User-controlled input concatenated into `subprocess`, shell, path, or deserialization operations.
- CUDA dims/index/offset/count narrowed to 32-bit without proved range bounds, or 64-bit index expressions with unsafe casts.
- Allocation without RAII/free, communication on the wrong group, or dispatch/combine asymmetry.
- New source files without the repository copyright header.
- `paddle.enable_static()` or `@to_static` where CI policy forbids it.
