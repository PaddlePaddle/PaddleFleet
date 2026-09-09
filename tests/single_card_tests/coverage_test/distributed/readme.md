# Distributed Tests / 分布式模块测试

Unit tests for PaddleFleet distributed module including parallel state, model parallel config, and context parallel utilities.
PaddleFleet 分布式模块的单元测试，包括并行状态、模型并行配置和上下文并行工具。

## Test Files

| File | Description / 描述 |
|------|-------------------|
| `test_coverage_context_parallel_utils.py` | Tests for mark_context_parallel_parameter_disable_scale_grad on layers / 测试层与参数的上下文并行缩放梯度禁用标记 |
| `test_coverage_context_parallel_utils_2.py` |  |
| `test_coverage_context_parallel_utils_3.py` |  |
| `test_coverage_context_parallel_utils_4.py` |  |
| `test_coverage_context_parallel_utils_5.py` |  |
| `test_coverage_distributed.py` |  |
| `test_coverage_distributed_2.py` |  |
| `test_coverage_flash_attn.py` |  |
| `test_coverage_model.py` | Tests for distributed_model function with PipelineLayer validation / 测试分布式模型函数与 PipelineLayer 校验 |
| `test_coverage_model_parallel_config.py` | Tests for ModelParallelConfig dataclass defaults and constraints / 测试模型并行配置数据类的默认值与约束 |
| `test_coverage_packed_seq_params.py` | Tests for PackedSeqParams dataclass default and custom values / 测试打包序列参数数据类 |
| `test_coverage_parallel_state.py` | Tests for parallel_state initialize_model_parallel and group setup / 测试并行状态初始化与通信组设置 |
| `test_coverage_parallel_state_2.py` |  |
| `test_coverage_parallel_state_3.py` |  |
| `test_coverage_process_groups_config.py` | Tests for ProcessGroupCollection initialization / 测试进程组集合初始化 |
| `test_coverage_recompute_utils.py` | Tests for need_recompute_in_block with various configurations / 测试不同配置下的块级重计算判断 |
| `test_coverage_utils.py` |  |
| `test_cp_balance_mode_dispatch.py` |  |
| `test_dualchunk_cp_utils.py` |  |