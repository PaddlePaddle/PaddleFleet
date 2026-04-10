# Pipeline Parallel Unit Tests (AI Edited)

This directory contains unit tests for the `paddlefleet.pipeline_parallel` module, generated to improve code coverage.

## Test Files

| Test File | Source File | Coverage Target |
|-----------|-------------|-----------------|
| `test_ai_utils.py` | `pipeline_parallel/utils.py` | 0% -> higher |
| `test_ai_vpp_simulator.py` | `pipeline_parallel/vpp_simulator.py` | 68% -> higher |
| `test_ai_pp_layers.py` | `pipeline_parallel/pp_layers.py` | 75% -> higher |
| `test_ai_pp_utils.py` | `pipeline_parallel/pp_utils/utils.py` | 75% -> higher |
| `test_ai_p2p_communication.py` | `pipeline_parallel/pp_utils/p2p_communication.py` | 82% -> higher |
| `test_ai_four_directions_p2p_communication.py` | `pipeline_parallel/pp_utils/four_directions_p2p_communication.py` | 0% -> higher |
| `test_ai_forward_backward_overlap_utils.py` | `pipeline_parallel/pp_utils/forward_backward_overlap_utils.py` | 17% -> higher |
| `test_ai_pipeline_parallel.py` | `pipeline_parallel/pipeline_parallel.py` | 74% -> higher |
| `test_ai_pipeline_parallel_withinterleave.py` | `pipeline_parallel/pipeline_parallel_withinterleave.py` | 70% -> higher |
| `test_ai_pipeline_parallel_withinterleave_fthenb.py` | `pipeline_parallel/pipeline_parallel_withinterleave_fthenb.py` | 80% -> higher |
| `test_ai_pipeline_hooks.py` | `pipeline_parallel/pipeline_hooks.py` | 85% -> higher |
| `test_ai_vpp_balanced_memory.py` | `pipeline_parallel/vpp_balanced_memory.py` | 85% -> higher |

## Key Priority Targets

- **`pipeline_parallel/utils.py`** (0% coverage) - Stream management, stage utilities, ScheduleNode, AbstractSchedulePlan
- **`pp_utils/four_directions_p2p_communication.py`** (0% coverage) - Four-direction P2P communication for XPU
- **`pp_utils/forward_backward_overlap_utils.py`** (17% coverage) - ScheduleNode, ScheduleChunk, FakeClone, recompute support

## Running Tests

```bash
# Run all pipeline parallel tests
python -m pytest tests/single_card_tests/ai_edited_test/pipeline_parallel/ -v

# Run a specific test file
python -m pytest tests/single_card_tests/ai_edited_test/pipeline_parallel/test_ai_utils.py -v

# Run with unittest
python -m unittest tests.single_card_tests.ai_edited_test.pipeline_parallel.test_ai_utils
```

## Testing Strategy

- Use `unittest.TestCase` style
- Extensive use of `unittest.mock` for distributed code
- All imports are placed inside test methods to handle PaddlePaddle availability
- CUDA-dependent tests use `@unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")`
- Only English comments
