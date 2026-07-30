---
name: paddlefleet-review
description: Review PaddleFleet pull requests and code changes using repository-specific architecture, distributed-training invariants, custom-op integration checks, and test mappings. Use for changes under src/paddlefleet, packages/paddlefleet_ops, tests, ci, auto_configurator, or build and dependency files in the PaddlePaddle/PaddleFleet repository.
---

# PaddleFleet Review

Review semantic correctness and cross-module impact. Use the invoking reviewer's severity, comment, and posting policy; this skill only supplies PaddleFleet-specific knowledge.

## Workflow

1. Classify changed files by subsystem and identify public API, configuration, distributed-state, kernel, build, or test impact.
2. Read [references/architecture.md](references/architecture.md) when a change crosses modules or modifies configuration, parallel state, models, or custom-op boundaries.
3. Read only the matching sections of [references/checklist.md](references/checklist.md). Do not load unrelated subsystem checks.
4. Trace each non-trivial change through definitions, callers, implementations, and tests with `rg` and focused file reads.
5. Verify candidate findings against the complete function and relevant tests. Report concrete defects; omit unsupported suspicions and generic style advice.

## Core Invariants

- Keep `TransformerConfig` and `ModelParallelConfig` changes compatible and synchronized with CLI/YAML entry points and consumers.
- Preserve process-group initialization order, rank membership, tensor shape, dtype, and collective symmetry across TP, PP, EP, and CP combinations.
- Preserve MoE dispatch/combine token ordering, expert indices, capacity handling, and forward/backward symmetry.
- Keep custom-op kernel registration, build sources, Python facade signatures, dtype/shape contracts, and tests synchronized.
- Require fused, FP8, Triton, TileLang, and cuDNN paths to retain a valid fallback or clearly enforced support boundary and numerical parity coverage.
- Match changed behavior with the closest single-card or multi-card regression tests; do not accept blacklist-only coverage without justification.

## Review Boundaries

- Treat `packages/paddlefleet_ops/third_party/` as vendored code. Require an explicit reason and upstream synchronization plan for modifications.
- Check PR claims against the actual implementation, especially configuration values, enabled backends, supported shapes/dtypes, and test coverage.
- For large PRs, prioritize public contracts, distributed state, custom kernels, hot paths, and build/CI behavior; state any material coverage gap.
