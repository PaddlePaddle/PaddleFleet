# Embedding Module Unit Tests

This directory contains unit tests for the embedding modules in PaddleFleet.

## Source Files Under Test

| Source File | Test File | Coverage Target |
|---|---|---|
| `src/paddlefleet/models/common/embeddings/rope_utils.py` | `test_ai_rope_utils.py` | Functions: `apply_rotary_pos_emb`, `get_pos_emb_on_this_cp_rank`, `_rotate_half`, `get_unsqueeze_dim`, `_apply_rotary_pos_emb_bshd`, `_apply_rotary_pos_emb_bshd_fp32`, `_apply_rotary_pos_emb_thd`, `_get_thd_freqs_on_this_cp_rank` |
| `src/paddlefleet/models/common/embeddings/language_model_embedding.py` | `test_ai_language_model_embedding.py` | Class: `LanguageModelEmbedding` |
| `src/paddlefleet/models/common/embeddings/rotary_pos_embedding.py` | `test_ai_rotary_pos_embedding.py` | Classes: `RotaryEmbedding`, `MultimodalRotaryEmbedding`, `Rope2DPosEmbRepeated` |
| `src/paddlefleet/models/common/embeddings/yarn_rotary_pos_embedding.py` | `test_ai_yarn_rotary_pos_embedding.py` | Class: `YarnRotaryEmbedding`, helper functions: `_yarn_find_correction_dim`, `_yarn_find_correction_range`, `_yarn_linear_ramp_mask`, `_yarn_get_mscale`, `_yarn_get_concentration_factor`, `_yarn_get_concentration_factor_from_config` |

## Test Categories

### rope_utils
- `_rotate_half`: Tests for both interleaved and non-interleaved modes
- `get_unsqueeze_dim`: Tests dimension inference for broadcasting
- `_apply_rotary_pos_emb_bshd`: Tests BSHD format RoPE application with various options (mscale, interleaved, high_precision, freqs transpose, partial rotary dim)
- `_apply_rotary_pos_emb_bshd_fp32`: Tests FP32 high-precision RoPE with t_pass handling
- `_get_thd_freqs_on_this_cp_rank`: Tests frequency slicing for context parallelism
- `_apply_rotary_pos_emb_thd`: Tests THD (packed sequence) RoPE with mocked distributed calls
- `get_pos_emb_on_this_cp_rank`: Tests CP rank position embedding slicing
- `apply_rotary_pos_emb`: Tests the top-level dispatch function (fused/unfused, BSHD/THD)

### language_model_embedding
- Initialization: Tests with learned_absolute, rope, tokentypes, sequence parallel constraints
- Forward: Tests with/without position embeddings, tokentype IDs, fp32 residual connection
- Sequence parallel: Tests scatter, clone, reduce_scatter paths with mocked tensor_parallel
- Properties: Tests embedding_weight property, zero_parameters method

### rotary_pos_embedding
- `RotaryEmbedding`: Tests init, forward, get_cos_sin, get_freqs_non_repeated, _apply_scaling, get_rotary_seq_len
- `MultimodalRotaryEmbedding`: Tests init, forward, apply_interleaved_mrope
- `Rope2DPosEmbRepeated`: Tests init, _precompute_freqs_cis, forward, get_cos_sin, caching

### yarn_rotary_pos_embedding
- Helper functions: Tests for all standalone math functions (_yarn_find_correction_dim, _yarn_find_correction_range, _yarn_linear_ramp_mask, _yarn_get_mscale, _yarn_get_concentration_factor, _yarn_get_concentration_factor_from_config)
- `YarnRotaryEmbedding`: Tests init, forward, caching mechanism, inheritance

## Running the Tests

```bash
# Run all embedding tests
python -m pytest tests/single_card_tests/ai_edited_test/embeddings/ -v

# Run a specific test file
python -m pytest tests/single_card_tests/ai_edited_test/embeddings/test_ai_rope_utils.py -v

# Run with unittest
python -m unittest tests.single_card_tests.ai_edited_test.embeddings.test_ai_rope_utils
```

## Dependencies

- PaddlePaddle (CPU mode is sufficient for all tests)
- Distributed calls are mocked using `unittest.mock`
