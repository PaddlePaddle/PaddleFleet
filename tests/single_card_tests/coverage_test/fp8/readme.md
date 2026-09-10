# FP8 Tests / FP8 模块测试

Unit tests for PaddleFleet FP8 quantization, linear layers, and related utilities.
PaddleFleet FP8 量化、线性层及相关工具的单元测试。

## Test Files

| File | Description / 描述 |
|------|-------------------|
| `test_coverage_fp8_module_structure.py` | Tests for fp8 module structure via direct source loading / 测试 FP8 模块结构 |
| `test_coverage_fp8_utils.py` | Tests for is_fp8_tensor function / 测试 FP8 张量判断函数 |
| `test_coverage_quantization.py` | Tests for get_quant_func with blockwise recipe / 测试 blockwise 量化函数获取 |
| `test_coverage_quantization_2.py` |  |
| `test_coverage_quantization_3.py` |  |
| `test_ue8m0.py` | Tests for use_ue8m0 code paths in fused_stack_quant and MoELayer fp8 / 测试 UE8M0 量化路径 |
