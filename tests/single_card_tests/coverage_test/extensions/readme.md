# Extensions Tests / 扩展模块测试

Unit tests for PaddleFleet custom CUDA extensions, flashmask, triton operators, and index utilities.
PaddleFleet 自定义 CUDA 扩展、FlashMask、Triton 算子和索引工具的单元测试。

## Test Files

| File | Description / 描述 |
|------|-------------------|
| `test_coverage_block_mask_utils.py` | Tests for find_blocks_topp function and triton kernel existence / 测试 top-p block 查找函数 |
| `test_coverage_block_mask_utils_2.py` |  |
| `test_coverage_block_mask_utils_3.py` |  |
| `test_coverage_block_mask_utils_4.py` |  |
| `test_coverage_block_mask_utils_5.py` |  |
| `test_coverage_block_mask_utils_6.py` |  |
| `test_coverage_flashmask_ext_structure.py` | Tests for flashmask extensions module structure / 测试 FlashMask 扩展模块结构 |
| `test_coverage_index_utils.py` | Tests for prepare_maxmin with various chunk sizes and dtypes / 测试不同分块大小和数据类型的最大最小值准备 |
| `test_coverage_index_utils_2.py` |  |
| `test_coverage_index_utils_3.py` |  |
| `test_coverage_index_utils_4.py` |  |
| `test_coverage_ops.py` |  |
| `test_coverage_ops_2.py` |  |
| `test_coverage_rr_attn_estimate_triton_op.py` | Tests for _require, _extract_raw_ptrs, RawPtrs, StrideMaxMinPtrs, rr_attn_estimate / 测试旋转率注意力估算 Triton 算子 |
| `test_coverage_rr_attn_estimate_triton_op_2.py` |  |
| `test_coverage_rr_attn_estimate_triton_op_3.py` |  |
| `test_coverage_rr_attn_estimate_triton_op_4.py` |  |
