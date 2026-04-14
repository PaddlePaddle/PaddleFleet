# PaddleFleet Model Tests

This directory contains unit tests for PaddleFleet model components.

## Test Files

| File | Source File | Coverage Target |
|------|-------------|-----------------|
| `test_gpt_model.py` | `src/paddlefleet/models/gpt/gpt_model.py` | Unit tests for GPTModel, GPTSublayersSpec, build_overlapped_nodes |
| `test_language_loss.py` | `src/paddlefleet/models/common/language_loss/language_loss.py` | Tests for LanguageLoss, subbatch, DistributedSoftmaxOp |
| `test_clip_vit_model.py` | `src/paddlefleet/models/vision/clip_vit_model.py` | Tests for CLIPViTModel, get_num_image_embeddings |
| `test_radio.py` | `src/paddlefleet/models/vision/radio.py` | Tests for RADIOViTModel positional encoding, constructor, forward |
| `test_multimodal_projector.py` | `src/paddlefleet/models/vision/multimodal_projector.py` | Tests for MultimodalProjector MLP/affine types |
| `test_vit_layer_specs.py` | `src/paddlefleet/models/vision/vit_layer_specs.py` | Tests for get_vit_layer_with_local_spec, _get_mlp_module_spec |
| `test_llava_model.py` | `src/paddlefleet/models/multimodal/llava_model.py` | Tests for LLaVAModel, pixel_shuffle, state dict hooks |
| `test_llava_spec.py` | `src/paddlefleet/models/multimodal/llava_spec.py` | Tests for decoder_model_with_local_default_spec |
| `test_context_parallel.py` | `src/paddlefleet/models/multimodal/context_parallel.py` | Tests for get_padding, get_packed_seq_params |
| `test_kimi_k25_model.py` | `src/paddlefleet/models/kimi_k25/kimi_k25_model.py` | Tests for KimiK25 model components |
| `test_qwen3_5_model.py` | `src/paddlefleet/models/qwen3_5/qwen3_5_model.py` | Tests for Qwen3.5 RMSNorm, vision model |

## Running Tests

```bash
# Run all model tests
python -m pytest tests/single_card_tests/ai_edited_test/models/ -v

# Run a specific test file
python -m pytest tests/single_card_tests/ai_edited_test/models/test_language_loss.py -v

# Run with unittest
python -m unittest tests.single_card_tests.ai_edited_test.models.test_language_loss
```

## Notes

- Tests requiring CUDA are decorated with `@unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")`
- Distributed-related calls are mocked to allow single-card testing
- Fleet initialization is handled via a shared `_init_fleet()` helper
