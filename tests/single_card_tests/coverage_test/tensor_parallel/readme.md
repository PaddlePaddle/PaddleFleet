# Tensor Parallel Tests / 张量并行模块测试

Unit tests for PaddleFleet tensor parallel module including mappings, layers, random states, cross entropy, and data utilities.
PaddleFleet 张量并行模块的单元测试，包括映射、层、随机状态、交叉熵和数据工具。

## Test Files

| File | Description / 描述 |
|------|-------------------|
| `test_coverage_cross_entropy.py` | Tests for VocabParallelCrossEntropy.calculate_logits_max / 测试词表并行交叉熵 logits 最大值计算 |
| `test_coverage_cross_entropy_2.py` |  |
| `test_coverage_cross_entropy_3.py` |  |
| `test_coverage_data.py` | Tests for _check_data_types / 测试数据类型检查函数 |
| `test_coverage_data_2.py` |  |
| `test_coverage_data_3.py` |  |
| `test_coverage_layers.py` | Tests for param_is_not_tensor_parallel_duplicate / 测试参数是否非张量并行副本 |
| `test_coverage_layers_2.py` |  |
| `test_coverage_layers_3.py` |  |
| `test_coverage_mappings.py` | Tests for _reduce helper function / 测试 reduce 辅助函数 |
| `test_coverage_mappings_2.py` |  |
| `test_coverage_mappings_3.py` |  |
| `test_coverage_mappings_4.py` |  |
| `test_coverage_mappings_5.py` |  |
| `test_coverage_mappings_6.py` |  |
| `test_coverage_mappings_7.py` |  |
| `test_coverage_random.py` | Tests for CudaRNGStatesTracker initialization / 测试 CUDA 随机状态追踪器初始化 |
| `test_coverage_random_2.py` |  |
| `test_coverage_random_3.py` |  |
| `test_coverage_random_4.py` |  |
| `test_coverage_random_5.py` |  |
| `test_coverage_random_6.py` |  |
| `test_coverage_utils.py` |  |
| `test_coverage_utils_2.py` |  |
