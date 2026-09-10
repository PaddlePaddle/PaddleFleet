# Transformer Tests / Transformer 模块测试

Unit tests for PaddleFleet transformer module including attention, MLP, MoE, dot product attention, multi-latent attention, and transformer layer/encoder.
PaddleFleet Transformer 模块的单元测试，包括注意力、MLP、MoE、点积注意力、多潜在注意力和 Transformer 层/编码器。

## Test Files

| File | Description / 描述 |
|------|-------------------|
| `test_coverage_attention.py` | Tests for SelfAttentionSublayersSpec dataclass / 测试自注意力子层规格 |
| `test_coverage_attention_2.py` |  |
| `test_coverage_attention_3.py` |  |
| `test_coverage_attention_4.py` |  |
| `test_coverage_attention_5.py` |  |
| `test_coverage_block_attn_res.py` |  |
| `test_coverage_csa_attention_compact.py` |  |
| `test_coverage_dot_product_attention.py` | Tests for DotProductAttention constructor / 测试点积注意力构造器 |
| `test_coverage_dot_product_attention_2.py` |  |
| `test_coverage_dot_product_attention_3.py` |  |
| `test_coverage_dot_product_attention_4.py` |  |
| `test_coverage_dot_product_attention_5.py` |  |
| `test_coverage_dot_product_attention_6.py` |  |
| `test_coverage_dot_product_attention_softmax_offset.py` |  |
| `test_coverage_dot_product_attention_softmax_scale.py` |  |
| `test_coverage_dot_product_attention_use_accuracy_compatible.py` |  |
| `test_coverage_dsa_attention.py` | Tests for hadamard_transform function / 测试 Hadamard 变换函数 |
| `test_coverage_dsv4_hybrid_attention_recompute.py` |  |
| `test_coverage_fp8_utils.py` |  |
| `test_coverage_fp8_utils_2.py` |  |
| `test_coverage_fp8_utils_3.py` |  |
| `test_coverage_fp8_utils_4.py` |  |
| `test_coverage_fp8_utils_5.py` |  |
| `test_coverage_fp8_utils_6.py` |  |
| `test_coverage_fused_a2a.py` |  |
| `test_coverage_fusion_layer_utils.py` |  |
| `test_coverage_gated_delta_net.py` | Tests for _l2norm helper function / 测试 L2 范数辅助函数 |
| `test_coverage_hyper_connection_transformer_layer.py` |  |
| `test_coverage_mlp.py` | Tests for MLP constructor / 测试 MLP 构造器 |
| `test_coverage_mlp_2.py` |  |
| `test_coverage_mlp_3.py` |  |
| `test_coverage_mlp_4.py` |  |
| `test_coverage_moe_expert.py` |  |
| `test_coverage_moe_expert_2.py` |  |
| `test_coverage_moe_layer.py` |  |
| `test_coverage_moe_layer_2.py` |  |
| `test_coverage_moe_layer_4.py` |  |
| `test_coverage_moe_layer_5.py` |  |
| `test_coverage_moe_layer_6.py` |  |
| `test_coverage_moe_layer_7.py` |  |
| `test_coverage_moe_utils.py` |  |
| `test_coverage_moe_utils_2.py` |  |
| `test_coverage_moe_utils_3.py` |  |
| `test_coverage_moe_utils_4.py` |  |
| `test_coverage_moe_utils_5.py` |  |
| `test_coverage_moe_utils_6.py` |  |
| `test_coverage_moe_utils_7.py` |  |
| `test_coverage_moe_utils_8.py` |  |
| `test_coverage_multi_latent_attention.py` | Tests for MLASelfAttentionSublayersSpec / 测试多潜在自注意力子层规格 |
| `test_coverage_multi_latent_attention_2.py` |  |
| `test_coverage_multi_latent_attention_3.py` |  |
| `test_coverage_multi_latent_attention_4.py` |  |
| `test_coverage_multi_latent_attention_softmax_scale.py` |  |
| `test_coverage_paddle_norm.py` | Tests for RMSNorm layer / 测试 RMS 归一化层 |
| `test_coverage_paddle_norm_2.py` |  |
| `test_coverage_paddle_norm_3.py` |  |
| `test_coverage_swa_high_precision_norm.py` |  |
| `test_coverage_token_dispatcher.py` |  |
| `test_coverage_token_dispatcher_2.py` |  |
| `test_coverage_token_dispatcher_3.py` |  |
| `test_coverage_transformer_block.py` | Tests for TransformerBlockSublayersSpec / 测试 Transformer 块子层规格 |
| `test_coverage_transformer_block_2.py` |  |
| `test_coverage_transformer_config.py` | Tests for TransformerConfig defaults / 测试 Transformer 配置默认值 |
| `test_coverage_transformer_config_2.py` |  |
| `test_coverage_transformer_encoder.py` | Tests for build_overlapped_nodes / 测试重叠节点构建 |
| `test_coverage_transformer_encoder_2.py` |  |
| `test_coverage_transformer_encoder_3.py` |  |
| `test_coverage_transformer_encoder_4.py` |  |
| `test_coverage_transformer_encoder_5.py` |  |
| `test_coverage_transformer_encoder_6.py` |  |
| `test_coverage_transformer_encoder_7.py` |  |
| `test_coverage_transformer_encoder_8.py` |  |
| `test_coverage_transformer_encoder_9.py` |  |
| `test_coverage_transformer_layer.py` | Tests for tensors_clone utility / 测试张量克隆工具函数 |
| `test_coverage_transformer_layer_2.py` |  |
| `test_coverage_transformer_layer_3.py` |  |
| `test_coverage_transformer_layer_5.py` |  |
| `test_coverage_transformer_layer_6.py` |  |
| `test_coverage_transformer_layer_7.py` |  |
| `test_coverage_transformer_layer_8.py` |  |
| `test_coverage_transformer_layer_9.py` |  |
| `test_coverage_transformer_layer_shared_no_hook.py` |  |
| `test_coverage_utils.py` |  |
| `test_coverage_utils_2.py` |  |
| `test_coverage_utils_3.py` |  |
