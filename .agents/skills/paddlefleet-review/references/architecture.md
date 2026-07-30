# PaddleFleet Architecture Reference

## Project Shape

PaddleFleet is a distributed-training core library. The main package is `src/paddlefleet`; compiled/custom operators are built from `packages/paddlefleet_ops`.

| Area | Primary paths | Main contracts |
| --- | --- | --- |
| Configuration | `model_parallel_config.py`, `transformer/transformer_config.py`, `training/{arguments,yaml_arguments}.py` | defaults, compatibility, CLI/YAML parity |
| Distributed state | `parallel_state.py`, `process_groups_config.py`, `distributed/` | group topology, initialization order, lifecycle |
| Tensor/Pipeline parallel | `tensor_parallel/`, `pipeline_parallel/` | communication semantics, partition shapes, RNG, scheduling |
| Context parallel | `context_parallel_utils.py`, `models/multimodal/context_parallel.py` | sequence partitioning, masks, positions |
| Transformer/MoE | `transformer/`, especially `transformer/moe/` | layer contracts, routing, dispatch/combine symmetry |
| Models | `models/` and model-specific `layer_specs.py` | registration, component wiring, state dict compatibility |
| Numerics/performance | `fp8/`, `fusions/`, `triton_ops/`, `tilelang_ops/`, `cudnn_ops/` | dtype/shape support, parity, fallback, hot-path behavior |
| Recompute/CUDA graph | `refined_recompute/`, `recompute_utils.py`, `cudagraph.py` | forward/backward parity, capture safety |
| Custom operators | `packages/paddlefleet_ops/src/paddlefleet_ops/_extensions/` and Python facade modules | build registration, op signature, version, tests |
| Auto configuration | `auto_configurator/` | search constraints, isolation from runtime internals |
| CI and policy | `ci/`, `.github/workflows/` | rule snapshots, test entry points, required checks |

## Cross-Module Checks

| Changed area | Also inspect |
| --- | --- |
| Config field add/remove/rename | CLI and YAML parsers, all consumers, defaults, serialization/hash tests, changelog if public |
| `parallel_state.py` or process groups | `process_groups_config.py`, initialization callers, TP/PP/EP/CP tests, teardown/reinit behavior |
| `tensor_parallel/` | model callers, collective/autograd semantics, tensor-parallel multi-card tests |
| `pipeline_parallel/` | model layer specs, PP utilities/VPP simulator, pipeline multi-card tests |
| `transformer/moe/` | GPT MoE layer specs, router, token dispatcher, fused A2A, custom kernels, MoE tests |
| Model or layer spec | `models/backends.py`, package exports, state-dict mapping, dense/MoE/vision tests as applicable |
| FP8/fusion backend | unfused reference path, MoE FP8 helpers, custom kernels, numerical parity tests |
| Triton/TileLang/cuDNN op | Python dispatch and fallback, architecture support, shape/dtype guards, custom-op tests |
| Custom C++/CUDA source | `packages/paddlefleet_ops/{setup,build_backend}.py`, Python facade, import tests, matching kernel tests |
| Recompute or CUDA graph | attention/model callers, gradient parity tests, capture/replay tests |
| `ci/rules/*.yml` | matching `ci/rule-tests/` case and snapshot |
| `third_party/` | upstream source/version, patch rationale, downstream build impact |

## Test Map

- Configuration and utilities: `tests/single_card_tests/test_transformer_config*.py` and `tests/single_card_tests/ai_edited_test/config_and_utils/`.
- TP/PP/EP/CP: `tests/multi_card_tests/{tensor_parallel,pipeline_parallel,moe,transformer}/` plus `test_parallel_states.py`.
- Models: `tests/single_card_tests/model/` and relevant multi-card model tests.
- Custom/performance ops: `tests/single_card_tests/custom_ops/`, `fp8/`, and `transformer/` parity tests.
- Recompute/CUDA graph: `test_need_recompute.py`, `test_recompute_without_output.py`, and `test_autocudagraph.py`.
