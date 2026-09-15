# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Behavior tests for the ERNIE pretrain ``ModelConfig`` dataclass.

Module under test:
``paddlefleet.cli.train.ernie_pretrain.model_config`` -- the ``ModelConfig``
dataclass. This belongs to the "configuration and runtime infrastructure"
layer: every field default is a contract that downstream fine-tuning /
quantization / training code reads, and the class is fed to an argument parser
whose flag defaults come straight from these fields. A silent change to any
default flips real training behaviour, so the tests pin exact values (content
and identity), not just attribute existence, type, or field count.

All expected values below were hand-derived by reading the dataclass field
declarations, never produced by the code under test. Booleans are checked with
``assertIs`` against ``True``/``False`` so a truthy non-bool cannot masquerade
as the documented flag, and None / numeric-zero-looking defaults are
distinguished from ``False`` where the difference is load-bearing (e.g.
``lora_rank`` defaulting to the integer ``8``, ``lora_path`` defaulting to
``None`` rather than ``False``).

These tests run on CPU. Importing the production module pulls the
``paddlefleet`` package, whose ``__init__`` chain imports Paddle; when Paddle is
absent the import raises ImportError and every test skips with an honest reason
(never silently passed).
"""

import unittest

try:
    from paddlefleet.cli.train.ernie_pretrain.model_config import ModelConfig

    _IMPORT_ERROR = None
except ImportError as exc:  # Paddle backend not installed in this env.
    ModelConfig = None
    _IMPORT_ERROR = exc


class _ModelConfigTestBase(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(
                "paddlefleet ernie model_config import failed "
                "(Paddle dependency unavailable): {!r}".format(_IMPORT_ERROR)
            )


class TestPathAndStringDefaults(_ModelConfigTestBase):
    """String / path defaults callers rely on when no CLI flag is given."""

    def test_model_and_tokenizer_paths_default_to_none(self):
        # None (not "") is the sentinel that means "not supplied"; downstream
        # code branches on it, so an empty string would be a real regression.
        config = ModelConfig()
        self.assertIsNone(config.model_name_or_path)
        self.assertIsNone(config.tokenizer_name_or_path)

    def test_stage_and_fine_tuning_string_defaults(self):
        config = ModelConfig()
        self.assertEqual(config.stage, "SFT")
        self.assertEqual(config.fine_tuning, "LoRA")

    def test_attn_and_lorapro_string_defaults(self):
        config = ModelConfig()
        self.assertEqual(config._attn_implementation, "flashmask")
        self.assertEqual(config.lorapro_x_mode, "zero")

    def test_fp8_and_pp_seg_string_defaults(self):
        config = ModelConfig()
        self.assertEqual(config.fp8_format_type, "hybrid")
        self.assertEqual(config.pp_seg_method, "layer:DecoderLayer|EmptyLayer")

    def test_aistudio_license_and_ids_default(self):
        config = ModelConfig()
        self.assertEqual(config.aistudio_repo_license, "Apache License 2.0")
        self.assertIsNone(config.aistudio_repo_id)
        self.assertIsNone(config.aistudio_token)

    def test_long_sequence_strategy_name_and_type_default_none(self):
        config = ModelConfig()
        self.assertIsNone(config.strategy_type)
        self.assertIsNone(config.strategy_name)

    def test_weight_quantize_algo_defaults_none(self):
        # None means "no quantization"; an accidental "" or a real algo name
        # would silently turn quantization on.
        self.assertIsNone(ModelConfig().weight_quantize_algo)


class TestBooleanDefaults(_ModelConfigTestBase):
    """Boolean flag defaults, identity-checked to reject truthy non-bools.

    The vast majority of feature flags default OFF; exactly three default ON
    (``continue_training``, ``aistudio_repo_private``,
    ``use_attn_mask_startend_row_indices``). Pinning both groups guards against
    a default being flipped in either direction.
    """

    def test_flags_that_default_true(self):
        config = ModelConfig()
        self.assertIs(config.continue_training, True)
        self.assertIs(config.aistudio_repo_private, True)
        self.assertIs(config.use_attn_mask_startend_row_indices, True)

    def test_peft_flags_default_false(self):
        config = ModelConfig()
        self.assertIs(config.lora, False)
        self.assertIs(config.use_quick_lora, False)
        self.assertIs(config.rslora, False)
        self.assertIs(config.pissa, False)
        self.assertIs(config.lora_use_mixer, False)
        self.assertIs(config.use_mora, False)
        self.assertIs(config.lorapro, False)
        self.assertIs(config.vera, False)
        self.assertIs(config.lokr, False)
        self.assertIs(config.prefix_tuning, False)
        self.assertIs(config.reft, False)

    def test_misc_feature_flags_default_false(self):
        config = ModelConfig()
        self.assertIs(config.use_fast_layer_norm, False)
        self.assertIs(config.save_to_aistudio, False)
        self.assertIs(config.neftune, False)
        self.assertIs(config.flash_mask, False)
        self.assertIs(config.use_long_sequence_strategies, False)

    def test_quantization_flags_default_false(self):
        config = ModelConfig()
        self.assertIs(config.qlora_weight_double_quant, False)
        self.assertIs(config.apply_hadamard, False)
        self.assertIs(config.quant_input_grad, False)
        self.assertIs(config.quant_weight_grad, False)


class TestNumericDefaults(_ModelConfigTestBase):
    """Numeric defaults, kept distinct from booleans and from each other."""

    def test_dropout_probability_defaults(self):
        config = ModelConfig()
        # Both dropout defaults are 0.1; assert they are floats, not the int 0
        # or a bool, and carry the documented magnitude.
        self.assertEqual(config.hidden_dropout_prob, 0.1)
        self.assertEqual(config.attention_probs_dropout_prob, 0.1)
        self.assertIsInstance(config.hidden_dropout_prob, float)
        self.assertIsInstance(config.attention_probs_dropout_prob, float)

    def test_float_scaling_defaults(self):
        config = ModelConfig()
        self.assertEqual(config.lora_plus_scale, 1.0)
        self.assertEqual(config.lorapro_scaling_factor, 2.0)
        self.assertEqual(config.neftune_noise_alpha, 5.0)
        self.assertEqual(config.rope_scaling_factor, 1.0)
        self.assertEqual(config.actscale_moving_rate, 0.01)

    def test_integer_dimension_defaults(self):
        config = ModelConfig()
        # These are non-zero ints; a value of 0 or False would be a regression.
        self.assertEqual(config.lora_rank, 8)
        self.assertEqual(config.vera_rank, 8)
        self.assertEqual(config.lokr_dim, 8)
        self.assertEqual(config.num_prefix_tokens, 128)
        self.assertEqual(config.qlora_weight_blocksize, 64)
        self.assertEqual(config.qlora_weight_double_quant_block_size, 256)
        self.assertEqual(config.hadamard_block_size, 32)
        self.assertEqual(config.apply_online_actscale_step, 200)
        for value in (
            config.lora_rank,
            config.num_prefix_tokens,
            config.qlora_weight_blocksize,
        ):
            self.assertIsInstance(value, int)
            self.assertNotIsInstance(value, bool)

    def test_lora_and_lokr_and_prefix_paths_default_none(self):
        config = ModelConfig()
        self.assertIsNone(config.lora_path)
        self.assertIsNone(config.lokr_path)
        self.assertIsNone(config.prefix_path)


class TestCustomValues(_ModelConfigTestBase):
    """Caller-supplied values reach the stored attributes unchanged."""

    def test_custom_values_override_defaults(self):
        config = ModelConfig(
            model_name_or_path="ernie/tiny",
            tokenizer_name_or_path="ernie/tok",
            stage="DPO",
            fine_tuning="Full",
            hidden_dropout_prob=0.25,
            lora=True,
            lora_rank=64,
            weight_quantize_algo="weight_only_int8",
            fp8_format_type="e4m3",
        )
        self.assertEqual(config.model_name_or_path, "ernie/tiny")
        self.assertEqual(config.tokenizer_name_or_path, "ernie/tok")
        self.assertEqual(config.stage, "DPO")
        self.assertEqual(config.fine_tuning, "Full")
        self.assertEqual(config.hidden_dropout_prob, 0.25)
        self.assertIs(config.lora, True)
        self.assertEqual(config.lora_rank, 64)
        self.assertEqual(config.weight_quantize_algo, "weight_only_int8")
        self.assertEqual(config.fp8_format_type, "e4m3")

    def test_true_defaults_can_be_turned_off_by_caller(self):
        # Explicitly disabling a default-ON flag must stick (guards against a
        # constructor that ignores/overrides caller input for these fields).
        config = ModelConfig(
            continue_training=False,
            aistudio_repo_private=False,
            use_attn_mask_startend_row_indices=False,
        )
        self.assertIs(config.continue_training, False)
        self.assertIs(config.aistudio_repo_private, False)
        self.assertIs(config.use_attn_mask_startend_row_indices, False)

    def test_instances_do_not_share_state(self):
        # Two constructions yield independent objects; mutating one leaves the
        # other's documented default intact.
        first = ModelConfig()
        second = ModelConfig()
        self.assertIsNot(first, second)
        first.lora = True
        first.lora_rank = 128
        self.assertIs(second.lora, False)
        self.assertEqual(second.lora_rank, 8)


if __name__ == "__main__":
    unittest.main()
