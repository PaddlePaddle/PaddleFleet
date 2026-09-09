# Model Tests / 模型模块测试

Unit tests for PaddleFleet model components including GPT, CLIP, LLaVA, Qwen, Kimi, and multimodal models.
PaddleFleet 模型组件的单元测试，包括 GPT、CLIP、LLaVA、Qwen、Kimi 和多模态模型。

## Test Files

| File | Description / 描述 |
|------|-------------------|
| `test_coverage_backends.py` | Tests for BackendSpecProvider protocol and LocalSpecProvider / 测试后端规格提供者协议与本地实现 |
| `test_coverage_clip_vit_model.py` | Tests for get_num_image_embeddings with CLIP, SigLIP, InternViT / 测试 CLIP/SigLIP/InternViT 图像嵌入数量获取 |
| `test_coverage_clip_vit_model_2.py` |  |
| `test_coverage_context_parallel.py` | Tests for get_padding with SP, CP, and FP8 configurations / 测试不同并行配置下的 padding 获取 |
| `test_coverage_context_parallel_2.py` |  |
| `test_coverage_embedding.py` |  |
| `test_coverage_empty_layer.py` | Tests for EmptyLayer initialization, forward, and FleetLayer inheritance / 测试空层的初始化、前向传播与继承 |
| `test_coverage_gpt.py` |  |
| `test_coverage_gpt_config.py` | Tests for GPTConfig defaults, custom values, and TransformerConfig inheritance / 测试 GPT 配置默认值与继承 |
| `test_coverage_gpt_embedding.py` | Tests for GPTEmbeddingSpec and get_placeholder_mask / 测试 GPT 嵌入规格与占位符 mask |
| `test_coverage_gpt_model.py` | Tests for build_overlapped_nodes, GPTSublayersSpec, add_sequential_layer / 测试重叠节点构建与子层规格 |
| `test_coverage_gpt_model_2.py` |  |
| `test_coverage_gpt_model_3.py` |  |
| `test_coverage_gpt_model_4.py` |  |
| `test_coverage_gpt_model_5.py` |  |
| `test_coverage_gpt_model_6.py` |  |
| `test_coverage_gpt_model_7.py` |  |
| `test_coverage_kimi_k25_model.py` | Tests for KimiK25 vision transformer layer/model classes / 测试 Kimi K25 视觉 Transformer 层与模型 |
| `test_coverage_kimi_k25_model_2.py` |  |
| `test_coverage_language_loss.py` | Tests for subbatch utility and LanguageLoss initialization / 测试子批次工具与语言损失初始化 |
| `test_coverage_language_loss_2.py` |  |
| `test_coverage_language_loss_3.py` |  |
| `test_coverage_language_loss_4.py` |  |
| `test_coverage_language_loss_5.py` |  |
| `test_coverage_language_loss_6.py` |  |
| `test_coverage_language_loss_7.py` |  |
| `test_coverage_llava_model.py` | Tests for LLaVA model module-level constants / 测试 LLaVA 模块级常量 |
| `test_coverage_llava_model_2.py` |  |
| `test_coverage_llava_model_3.py` |  |
| `test_coverage_llava_model_4.py` |  |
| `test_coverage_llava_model_5.py` |  |
| `test_coverage_llava_model_6.py` |  |
| `test_coverage_llava_model_7.py` |  |
| `test_coverage_llava_model_8.py` |  |
| `test_coverage_llava_spec.py` | Tests for decoder_model_with_local_default_spec / 测试本地默认规格解码模型函数 |
| `test_coverage_llava_spec_2.py` |  |
| `test_coverage_lm_head.py` | Tests for GPTLMHead forward method / 测试 GPT 语言模型头前向传播 |
| `test_coverage_moe_layer_specs.py` | Tests for get_moe_layer_spec_for_backend / 测试后端 MoE 层规格获取 |
| `test_coverage_multimodal_projector.py` | Tests for MultimodalProjector with various projector types / 测试多模态投影器 |
| `test_coverage_multimodal_projector_2.py` |  |
| `test_coverage_patch_merger.py` |  |
| `test_coverage_qwen3_5_model.py` | Tests for Qwen3_5RMSNorm class / 测试 Qwen3.5 RMS 归一化层 |
| `test_coverage_qwen3_5_model_2.py` |  |
| `test_coverage_qwen3_vl_model.py` | Tests for Qwen3VL vision model sublayers and layer structure / 测试 Qwen3VL 视觉模型子层与层结构 |
| `test_coverage_radio.py` | Tests for RADIOViTModel initialization with mocked internals / 测试 RADIO ViT 模型初始化 |
| `test_coverage_radio_2.py` |  |
| `test_coverage_rope_utils.py` |  |
| `test_coverage_rope_utils_2.py` |  |
| `test_coverage_sd2_tpool_merge.py` |  |
| `test_coverage_utils.py` |  |
| `test_coverage_vision_layer.py` | Tests for VisionLayer initialization / 测试视觉层初始化 |
| `test_coverage_vit_layer_specs.py` | Tests for get_vit_layer_with_local_spec / 测试本地规格 ViT 层获取 |