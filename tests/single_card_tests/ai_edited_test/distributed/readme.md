# Distributed Unit Tests

This directory contains unit tests for PaddleFleet's distributed module and related parallelism utilities.

## Test Files

| File | Source Under Test | Description |
|------|------------------|-------------|
| `test_distributed_init.py` | `src/paddlefleet/distributed/__init__.py` | Package import tests |
| `test_model.py` | `src/paddlefleet/distributed/model.py` | `distributed_model()` function: pipeline parallel, AMP, strategy |
| `test_parallel_state.py` | `src/paddlefleet/parallel_state.py` | Parallel group management: tensor/pipeline/data/expert/context parallel |
| `test_process_groups_config.py` | `src/paddlefleet/process_groups_config.py` | `ProcessGroupCollection` dataclass and `use_mpu_process_groups()` |
| `test_model_parallel_config.py` | `src/paddlefleet/model_parallel_config.py` | `ModelParallelConfig` dataclass validation and defaults |
| `test_context_parallel_utils.py` | `src/paddlefleet/context_parallel_utils.py` | Context parallel ops: scatter/gather/reduce-scatter, flashmask CP |
| `test_recompute_utils.py` | `src/paddlefleet/recompute_utils.py` | Recompute layer selection: block, first_n, full |
| `test_packed_seq_params.py` | `src/paddlefleet/packed_seq_params.py` | `PackedSeqParams` dataclass for packed sequence format |

## Running Tests

```bash
cd /path/to/PaddleFleet
python -m pytest tests/single_card_tests/ai_edited_test/distributed/ -v
```

Or run a single test file:

```bash
python tests/single_card_tests/ai_edited_test/distributed/test_parallel_state.py
```

## Notes

- All tests use `unittest.TestCase` style.
- Distributed communication is heavily mocked since tests run in single-card mode.
- No CUDA requirement for these tests.
