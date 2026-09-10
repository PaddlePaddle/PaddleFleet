# Fusions Tests / 融合算子模块测试

Unit tests for PaddleFleet fusion operators including bias activation, layer norm, RMS norm, softmax, and SwiGLU.
PaddleFleet 融合算子的单元测试，包括偏置激活、层归一化、RMS 归一化、softmax 和 SwiGLU。

## Test Files

| File | Description / 描述 |
|------|-------------------|
| `test_coverage_clampswiglu_align.py` |  |
| `test_coverage_fused_bias_dropout.py` | Tests for _bias_dropout_add_func, bias_dropout_add_unfused / 测试融合偏置 dropout 加法 |
| `test_coverage_fused_bias_dropout_2.py` |  |
| `test_coverage_fused_bias_geglu.py` |  |
| `test_coverage_fused_bias_geglu_2.py` |  |
| `test_coverage_fused_bias_geglu_3.py` |  |
| `test_coverage_fused_bias_geglu_4.py` |  |
| `test_coverage_fused_bias_geglu_5.py` |  |
| `test_coverage_fused_bias_geglu_6.py` |  |
| `test_coverage_fused_bias_gelu.py` |  |
| `test_coverage_fused_bias_gelu_2.py` |  |
| `test_coverage_fused_bias_swiglu.py` | Tests for swiglu, bias_swiglu, and weighted_swiglu output shapes / 测试 SwiGLU 输出形状 |
| `test_coverage_fused_bias_swiglu_2.py` |  |
| `test_coverage_fused_bias_swiglu_3.py` |  |
| `test_coverage_fused_bias_swiglu_4.py` |  |
| `test_coverage_fused_bias_swiglu_use_accuracy_compatible.py` |  |
| `test_coverage_fused_layer_norm.py` | Tests for FusedLayerNorm initialization and reset_parameters / 测试融合层归一化初始化与参数重置 |
| `test_coverage_fused_layer_norm_2.py` |  |
| `test_coverage_fused_mla_yarn_rope_apply.py` |  |
| `test_coverage_fused_rms_norm.py` | Tests for FusedRmsNorm initialization and sequence parallel flag / 测试融合 RMS 归一化初始化与序列并行 |
| `test_coverage_fused_rms_norm_2.py` |  |
| `test_coverage_fused_softmax.py` | Tests for SoftmaxOne initialization and forward pass / 测试 SoftmaxOne 初始化与前向传播 |
| `test_coverage_fused_softmax_2.py` |  |
| `test_coverage_fused_softmax_3.py` |  |
| `test_coverage_fused_swiglu_scale.py` | Tests for fused_swiglu_scale_forward CPU fallback and CUDA paths / 测试融合 SwiGLU 缩放的前向传播 |
| `test_coverage_fused_swiglu_scale_2.py` |  |
| `test_coverage_fused_swiglu_scale_3.py` |  |
| `test_coverage_fusions_module_structure.py` | Tests for fusions module imports / 测试融合模块导入结构 |
| `test_coverage_hysparse_dsa_online_precision.py` |  |
| `test_csa_sparse_attn_backends.py` |  |
| `test_csa_sparse_attn_sub_512_latent.py` |  |
| `test_csa_sparse_attn_sub_tile_heads.py` |  |
| `test_csa_sparse_attn_utils.py` |  |
