# Ops Tests / 算子模块测试

Unit tests for PaddleFleet ops module including cross entropy, MoE topk fusion, RMS norm, sigmoid gate, and triton utils.
PaddleFleet 算子模块的单元测试，包括交叉熵、MoE TopK 融合、RMS 归一化和 Triton 工具。

## Test Files

| File | Description / 描述 |
|------|-------------------|
| `test_coverage_cross_entropy.py` |  |
| `test_coverage_cross_entropy_2.py` |  |
| `test_coverage_dense_attn_kl_head_pad.py` |  |
| `test_coverage_dense_indexer_kl_grad_loss.py` |  |
| `test_coverage_fused_linear_cross_entropy.py` |  |
| `test_coverage_fused_linear_cross_entropy_2.py` |  |
| `test_coverage_moe_topk_fusion.py` | Tests for MoETopkFusion PyLayer class definition / 测试 MoE TopK 融合 PyLayer 定义 |
| `test_coverage_moe_topk_fusion_2.py` |  |
| `test_coverage_moe_topk_fusion_3.py` |  |
| `test_coverage_moe_topk_fusion_4.py` |  |
| `test_coverage_moe_topk_fusion_5.py` |  |
| `test_coverage_moe_topk_fusion_6.py` |  |
| `test_coverage_paddlefleet_ops_utils.py` |  |
| `test_coverage_rms_norm_fusion.py` | Tests for RMSNormFusionTriton PyLayer definition / 测试 RMS 归一化融合 Triton PyLayer |
| `test_coverage_rms_norm_fusion_2.py` |  |
| `test_coverage_rms_norm_fusion_3.py` |  |
| `test_coverage_score_target_head_pad.py` |  |
| `test_coverage_sigmoid_gate_fusion.py` | Tests for SigmoidGateFusionTriton PyLayer definition / 测试 Sigmoid 门控融合 Triton PyLayer |
| `test_coverage_sigmoid_gate_fusion_2.py` |  |
| `test_coverage_sigmoid_gate_fusion_3.py` |  |
| `test_coverage_triton_compat.py` | Tests for _is_package_installed function / 测试包安装判断函数 |
| `test_coverage_triton_compat_2.py` |  |
| `test_coverage_utils.py` |  |
| `test_coverage_utils_2.py` |  |
| `test_coverage_utils_3.py` |  |
