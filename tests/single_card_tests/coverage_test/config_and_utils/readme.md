# Config and Utils Tests / 配置与工具模块测试

Unit tests for PaddleFleet configuration, training arguments, timers, and utility functions.
PaddleFleet 配置、训练参数、计时器和工具函数的单元测试。

## Test Files

| File | Description / 描述 |
|------|-------------------|
| `test_coverage_arguments.py` | Tests for parse_args and core_transformer_config_from_args / 测试命令行参数解析与核心 Transformer 配置构建 |
| `test_coverage_config_logger.py` | Tests for config_logger module path and enabled state / 测试配置日志器模块路径与启用状态 |
| `test_coverage_config_logger_2.py` |  |
| `test_coverage_context_parallel_flashmask_modes.py` |  |
| `test_coverage_context_parallel_utils.py` |  |
| `test_coverage_global_vars.py` | Tests for global_vars get_args/set_args and get_timers/set_timers / 测试全局变量的 get/set 访问器 |
| `test_coverage_global_vars_2.py` |  |
| `test_coverage_gpt_builders.py` | Tests for gpt_builder and _get_transformer_layer_spec_func / 测试 GPT 构建器与 Transformer 层规格获取 |
| `test_coverage_initialize.py` |  |
| `test_coverage_jit.py` | Tests for jit module jit_fuser function / 测试 JIT 编译器的 fuser 函数 |
| `test_coverage_package_info.py` | Tests for package_info module metadata fields / 测试包信息模块元数据字段 |
| `test_coverage_parallel_state.py` |  |
| `test_coverage_separate_mtp_headloss.py` |  |
| `test_coverage_sonicmoe_configs.py` |  |
| `test_coverage_spec_utils.py` | Tests for LayerSpec, import_spec_layer, get_layer, and build_spec_layer / 测试层级规格定义、导入与构建 |
| `test_coverage_timers.py` | Tests for Timer, RuntimeTimer, and Timers classes / 测试计时器相关类 |
| `test_coverage_timers_2.py` |  |
| `test_coverage_transformer_config_hash.py` |  |
| `test_coverage_utils.py` | Tests for WrappedTensor, GlobalMemoryBuffer, ensure_divisibility, divide / 测试张量封装、全局内存缓冲区、整除检查等工具函数 |
| `test_coverage_utils_2.py` |  |
| `test_coverage_utils_3.py` |  |
| `test_coverage_utils_4.py` |  |
| `test_coverage_utils_5.py` |  |
| `test_coverage_yaml_arguments.py` | Tests for _flatten_configs function with various dict structures / 测试配置字典扁平化函数 |
| `test_coverage_yaml_arguments_2.py` |  |
| `test_coverage_yaml_arguments_3.py` |  |
| `test_coverage_yaml_arguments_4.py` |  |
| `test_coverage_yaml_arguments_5.py` |  |