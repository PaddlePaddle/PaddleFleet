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

import importlib
import math
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import paddle

from paddlefleet.training.arguments import core_transformer_config_from_args
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.transformer_config import TransformerConfig

strategy = paddle.distributed.fleet.DistributedStrategy()
initialize_fleet(strategy=strategy)


class TestP2POverlapDwCalcValidation(unittest.TestCase):
    def test_unknown_point_raises_value_error_with_details(self):
        with self.assertRaisesRegex(
            ValueError,
            r"unknown p2p_overlap_dw_calc entries \['not_a_real_point'\]",
        ) as context:
            TransformerConfig(p2p_overlap_dw_calc=["not_a_real_point"])
        self.assertIn("attn_q_proj", str(context.exception))

    def test_unknown_point_raises_under_python_optimize(self):
        code = """
from paddlefleet.transformer.transformer_config import TransformerConfig

try:
    TransformerConfig(p2p_overlap_dw_calc=["not_a_real_point"])
except ValueError as exc:
    if "not_a_real_point" not in str(exc) or "attn_q_proj" not in str(exc):
        raise RuntimeError(f"incomplete validation error: {exc}")
else:
    raise RuntimeError("unknown p2p_overlap_dw_calc point was accepted")
"""
        result = subprocess.run(
            [sys.executable, "-O", "-c", code],
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def test_deferral_rejects_pp_without_interleaved_scheduler(self):
        for pp, vpp in ((1, None), (2, None), (2, 1)):
            with (
                self.subTest(pp=pp, vpp=vpp),
                self.assertRaisesRegex(ValueError, "virtual_pipeline"),
            ):
                TransformerConfig(
                    p2p_overlap_dw_calc=["moe_expert_up_gate_proj"],
                    pipeline_model_parallel_size=pp,
                    virtual_pipeline_model_parallel_size=vpp,
                )

    def test_deferral_accepts_interleaved_scheduler(self):
        config = TransformerConfig(
            p2p_overlap_dw_calc=["moe_expert_up_gate_proj"],
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
        )
        self.assertEqual(
            config.p2p_overlap_dw_calc, ["moe_expert_up_gate_proj"]
        )


class TestMoeLayerFreqAndFirstKDenseReplace(unittest.TestCase):
    """Tests for the moe_layer_freq / first_k_dense_replace logic in TransformerConfig.__post_init__."""

    def test_both_none_defaults_moe_layer_freq_to_1(self):
        """When both first_k_dense_replace and moe_layer_freq are None, moe_layer_freq defaults to 1."""
        config = TransformerConfig(
            first_k_dense_replace=None,
            moe_layer_freq=None,
            num_hidden_layers=12,
        )
        self.assertEqual(config.moe_layer_freq, 1)

    def test_first_k_dense_replace_with_list_moe_layer_freq_raises(self):
        """When first_k_dense_replace is set and moe_layer_freq is a list (not int), should raise ValueError."""
        with self.assertRaises(ValueError):
            TransformerConfig(
                first_k_dense_replace=2,
                moe_layer_freq=[1, 0, 1, 0],
                num_hidden_layers=4,
            )

    def test_first_k_dense_replace_with_int_moe_layer_freq(self):
        """When first_k_dense_replace is set and moe_layer_freq is an int,
        it should generate a pattern based on the frequency."""
        config = TransformerConfig(
            first_k_dense_replace=2,
            moe_layer_freq=2,
            num_hidden_layers=8,
        )
        # first 2 layers are dense (0), remaining layers follow pattern: 1 if (i % 2 == 0) else 0
        # Pattern for range(8): i=0->1, i=1->0, i=2->1, i=3->0, i=4->1, i=5->0, i=6->1, i=7->0
        expected = [0, 0, 0, 1, 0, 1, 0, 1]
        self.assertEqual(config.moe_layer_freq, expected)

    def test_first_k_dense_replace_with_moe_layer_freq_none(self):
        """When first_k_dense_replace is set and moe_layer_freq is None,
        the remaining layers should all be MoE (all 1s)."""
        config = TransformerConfig(
            first_k_dense_replace=3,
            moe_layer_freq=None,
            num_hidden_layers=8,
        )
        # both-None check won't trigger since first_k_dense_replace is set,
        # moe_layer_freq stays None (falsy).
        # else branch: moe_layer_pattern = [1] * (8 - 3) = [1, 1, 1, 1, 1]
        # final = [0, 0, 0] + [1, 1, 1, 1, 1]
        expected = [0, 0, 0, 1, 1, 1, 1, 1]
        self.assertEqual(config.moe_layer_freq, expected)

    def test_first_k_dense_replace_with_moe_layer_freq_zero(self):
        """When first_k_dense_replace is set and moe_layer_freq is 0 (falsy int),
        the pattern should fall into the else branch producing all 1s for non-dense layers."""
        config = TransformerConfig(
            first_k_dense_replace=4,
            moe_layer_freq=0,
            num_hidden_layers=10,
        )
        # moe_layer_freq=0 is falsy, so moe_layer_pattern = [1] * (10 - 4) = [1, 1, 1, 1, 1, 1]
        expected = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        self.assertEqual(config.moe_layer_freq, expected)

    def test_first_k_dense_replace_with_moe_layer_freq_3(self):
        """When first_k_dense_replace is set with moe_layer_freq=3,
        every 3rd layer (index % 3 == 0) should be MoE."""
        config = TransformerConfig(
            first_k_dense_replace=1,
            moe_layer_freq=3,
            num_hidden_layers=7,
        )
        # first 1 layer is dense
        # Pattern for range(7): i=0->1, i=1->0, i=2->0, i=3->1, i=4->0, i=5->0, i=6->1
        expected = [0, 0, 0, 1, 0, 0, 1]
        self.assertEqual(config.moe_layer_freq, expected)

    def test_first_k_dense_replace_equals_num_hidden_layers(self):
        """Edge case: first_k_dense_replace equals num_hidden_layers,
        all layers should be dense (all 0s)."""
        config = TransformerConfig(
            first_k_dense_replace=6,
            moe_layer_freq=0,
            num_hidden_layers=6,
        )
        # moe_layer_freq=0 is falsy, so moe_layer_pattern = [1] * (6 - 6) = []
        expected = [0, 0, 0, 0, 0, 0]
        self.assertEqual(config.moe_layer_freq, expected)

    def test_only_moe_layer_freq_int_no_first_k_dense(self):
        """When only moe_layer_freq is set (as int) and first_k_dense_replace is None,
        moe_layer_freq should remain as the integer value."""
        config = TransformerConfig(
            first_k_dense_replace=None,
            moe_layer_freq=2,
            num_hidden_layers=8,
        )
        self.assertEqual(config.moe_layer_freq, 2)

    def test_only_first_k_dense_replace_no_moe_layer_freq(self):
        """When first_k_dense_replace is set and moe_layer_freq is not specified (defaults to None),
        should generate the correct pattern."""
        config = TransformerConfig(
            first_k_dense_replace=2,
            num_hidden_layers=6,
        )
        # moe_layer_freq=None => both-None check won't trigger because first_k_dense_replace is set
        # After the None-None check, moe_layer_freq is still None
        # first_k_dense_replace is truthy => enter the block
        # moe_layer_freq is None (falsy) => moe_layer_pattern = [1] * (6-2) = [1,1,1,1]
        expected = [0, 0, 1, 1, 1, 1]
        self.assertEqual(config.moe_layer_freq, expected)

    def test_first_k_dense_replace_1_moe_layer_freq_1(self):
        """first_k_dense_replace=1, moe_layer_freq=1: first layer dense, rest all MoE."""
        config = TransformerConfig(
            first_k_dense_replace=1,
            moe_layer_freq=1,
            num_hidden_layers=5,
        )
        # moe_layer_freq=1 is truthy int, pattern: 1 if (i % 1 == 0) else 0 => all 1s
        expected = [0, 1, 1, 1, 1]
        self.assertEqual(config.moe_layer_freq, expected)

    # --- first_k_dense_replace vs moe_n_hash_layers mutual exclusivity ---

    def test_hash_layers_with_first_k_dense_raises(self):
        """moe_n_hash_layers > 0 and first_k_dense_replace > 0 should raise ValueError."""
        with self.assertRaisesRegex(
            ValueError,
            "first_k_dense_replace.*moe_n_hash_layers.*mutually exclusive",
        ):
            TransformerConfig(
                num_hidden_layers=12,
                first_k_dense_replace=4,
                moe_n_hash_layers=2,
                actual_vocab_size=32000,
                num_experts_per_tok=2,
                n_routed_experts=8,
            )

    def test_hash_layers_with_zero_first_k_dense_ok(self):
        """moe_n_hash_layers > 0 with first_k_dense_replace=0 is valid."""
        config = TransformerConfig(
            num_hidden_layers=12,
            first_k_dense_replace=0,
            moe_n_hash_layers=4,
            actual_vocab_size=32000,
            num_experts_per_tok=2,
            n_routed_experts=8,
        )
        self.assertEqual(config.moe_n_hash_layers, 4)

    def test_first_k_dense_with_zero_hash_layers_ok(self):
        """first_k_dense_replace > 0 with moe_n_hash_layers=0 (default) is valid."""
        config = TransformerConfig(
            num_hidden_layers=12,
            first_k_dense_replace=2,
        )
        self.assertEqual(config.first_k_dense_replace, 2)
        self.assertEqual(config.moe_n_hash_layers, 0)


class TestRoutedScalingFactorConfig(unittest.TestCase):
    """Tests for the routed_scaling_factor and routed_scaling_factor_learnable fields
    in TransformerConfig."""

    def test_routed_scaling_factor_default_is_1(self):
        """routed_scaling_factor defaults to 1.0 when not specified."""
        config = TransformerConfig(num_hidden_layers=4)
        self.assertAlmostEqual(config.routed_scaling_factor, 1.0)

    def test_routed_scaling_factor_learnable_default_is_false(self):
        """routed_scaling_factor_learnable defaults to False when not specified."""
        config = TransformerConfig(num_hidden_layers=4)
        self.assertFalse(config.routed_scaling_factor_learnable)

    def test_routed_scaling_factor_float(self):
        """routed_scaling_factor accepts a float value (e.g., 2.5 for DeepSeek-V3)."""
        config = TransformerConfig(
            num_hidden_layers=4, routed_scaling_factor=2.5
        )
        self.assertAlmostEqual(config.routed_scaling_factor, 2.5)

    def test_routed_scaling_factor_learnable_true(self):
        """routed_scaling_factor_learnable can be set to True."""
        config = TransformerConfig(
            num_hidden_layers=4,
            routed_scaling_factor=2.5,
            routed_scaling_factor_learnable=True,
        )
        self.assertAlmostEqual(config.routed_scaling_factor, 2.5)
        self.assertTrue(config.routed_scaling_factor_learnable)


class TestMoETokenDispatcherConfig(unittest.TestCase):
    def test_default_dispatcher_type_is_alltoall(self):
        # Pins the schema default: every dispatcher-specific branch in MoELayer
        # keys off this field, so a silent change of default would reroute all
        # MoE traffic.
        field = TransformerConfig.__dataclass_fields__[
            "moe_token_dispatcher_type"
        ]
        self.assertEqual(field.default, "alltoall")
        self.assertEqual(
            TransformerConfig(
                num_hidden_layers=4, n_routed_experts=8
            ).moe_token_dispatcher_type,
            "alltoall",
        )

    def test_ringmoe_dispatcher_type_is_preserved(self):
        # 'ringmoe' is accepted as a non-default value and survives
        # __post_init__ unrewritten -- MoELayer decides the ring vs flat path
        # from exactly this string.
        config = TransformerConfig(
            num_hidden_layers=4,
            n_routed_experts=8,
            moe_token_dispatcher_type="ringmoe",
        )

        self.assertEqual(config.moe_token_dispatcher_type, "ringmoe")
        # The ring ignores the gate-overlap prefetch, but the flag keeps its
        # default rather than being force-corrected behind the user's back.
        self.assertTrue(config.moe_allgather_gate_overlap)

    def test_documented_dispatcher_types_are_all_accepted(self):
        """The field docstring is the only list of accepted values, so pin it.

        Fails if a dispatcher is wired up without being documented, or
        documented without being accepted by the schema.
        """
        import inspect
        import re

        # Field docstrings are not kept at runtime; read them from the source.
        doc = re.search(
            r'moe_token_dispatcher_type: str = "alltoall"\s*"""(.*?)"""',
            inspect.getsource(TransformerConfig),
            re.DOTALL,
        )
        self.assertIsNotNone(doc, "field docstring not found")
        documented = set(re.findall(r"'([a-z0-9]+)'", doc.group(1)))
        self.assertIn("ringmoe", documented)
        for dispatcher_type in sorted(documented):
            with self.subTest(dispatcher_type=dispatcher_type):
                config = TransformerConfig(
                    num_hidden_layers=4,
                    n_routed_experts=8,
                    moe_token_dispatcher_type=dispatcher_type,
                )
                self.assertEqual(
                    config.moe_token_dispatcher_type, dispatcher_type
                )

    def test_hybridep_dispatcher_type_is_preserved(self):
        config = TransformerConfig(
            num_hidden_layers=4,
            n_routed_experts=8,
            moe_token_dispatcher_type="hybridep",
        )

        self.assertEqual(config.moe_token_dispatcher_type, "hybridep")
        self.assertTrue(config.moe_use_fusion_node)


class TestMagicInit(unittest.TestCase):
    """Tests for magic_init sigma calculation and init method assignment."""

    def test_magic_init_true_sigma_calculation(self):
        hidden_size = 768
        config = TransformerConfig(
            num_hidden_layers=12,
            hidden_size=hidden_size,
            magic_init=True,
        )

        expected_sigma = math.sqrt(0.3333 / hidden_size)
        self.assertFalse(config.use_truncated_normal_init)
        self.assertAlmostEqual(config.init_method_std, expected_sigma, places=6)
        self.assertIsNotNone(config.init_method)

    def test_magic_init_true_zero_hidden_size_raises(self):
        with self.assertRaisesRegex(
            ValueError, "hidden_size must be non-zero when magic_init is True."
        ):
            TransformerConfig(
                num_hidden_layers=12,
                hidden_size=0,
                magic_init=True,
            )


class TestTruncateNormInit(unittest.TestCase):
    """Tests for the use_truncated_normal_init functionality in TransformerConfig."""

    def test_truncate_norm_sigma_calculation(self):
        hidden_size = 1024
        config = TransformerConfig(
            num_hidden_layers=12,
            hidden_size=hidden_size,
            use_truncated_normal_init=True,
        )

        self.assertAlmostEqual(
            config.init_method_std,
            0.02,
            places=6,
        )

    def test_truncate_norm_takes_precedence_over_magic_init(self):
        hidden_size = 1024
        config = TransformerConfig(
            num_hidden_layers=12,
            hidden_size=hidden_size,
            magic_init=True,
            use_truncated_normal_init=True,
        )

        self.assertAlmostEqual(
            config.init_method_std,
            0.02,
            places=6,
        )
        self.assertIs(config.init_method, config.output_layer_init_method)
        self.assertIs(config.init_method, config.embedding_init_method)

    def test_truncate_norm_init_method_restores_default_dtype(self):
        from paddlefleet.utils import truncated_init_method_normal

        original_dtype = paddle.get_default_dtype()
        weight = paddle.empty([8, 8], dtype="float32")
        init_method = truncated_init_method_normal(0.5, truncate_factor=2.0)

        try:
            paddle.set_default_dtype("float64")
            init_method(weight)
            self.assertEqual(paddle.get_default_dtype(), "float64")
        finally:
            paddle.set_default_dtype(original_dtype)

    def test_truncate_norm_init_method_restores_default_dtype_on_error(self):
        from paddlefleet.utils import truncated_init_method_normal

        original_dtype = paddle.get_default_dtype()
        init_method = truncated_init_method_normal(0.5, truncate_factor=2.0)
        weight = Mock()

        try:
            paddle.set_default_dtype("float64")
            with (
                patch("paddle.nn.init.trunc_normal_", side_effect=RuntimeError),
                self.assertRaises(RuntimeError),
            ):
                init_method(weight)
            self.assertEqual(paddle.get_default_dtype(), "float64")
        finally:
            paddle.set_default_dtype(original_dtype)

    def test_truncate_norm_zero_hidden_size_falls_back_to_init_std(self):
        config = TransformerConfig(
            num_hidden_layers=12,
            hidden_size=0,
            use_truncated_normal_init=True,
        )

        self.assertEqual(config.init_method_std, 0.02)

    def test_truncate_norm_none_init_std_zero_hidden_size_raises(self):
        with self.assertRaisesRegex(
            ValueError,
            "hidden_size must be non-zero when init_method_std is None and use_truncated_normal_init is True.",
        ):
            TransformerConfig(
                num_hidden_layers=12,
                hidden_size=0,
                init_method_std=None,
                use_truncated_normal_init=True,
            )

    def test_truncate_norm_raises_on_non_positive_factor(self):
        with self.assertRaisesRegex(
            ValueError,
            "truncated_normal_init_factor must be positive when use_truncated_normal_init is True.",
        ):
            TransformerConfig(
                num_hidden_layers=12,
                hidden_size=1024,
                use_truncated_normal_init=True,
                truncated_normal_init_factor=0,
            )


class TestPadTokenId(unittest.TestCase):
    """Tests for the pad_token_id field on TransformerConfig."""

    def test_default_is_zero(self):
        config = TransformerConfig(num_hidden_layers=2)
        self.assertEqual(config.pad_token_id, 0)

    def test_override_value(self):
        config = TransformerConfig(num_hidden_layers=2, pad_token_id=151643)
        self.assertEqual(config.pad_token_id, 151643)


class FakeDictConfig(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class TestYamlArguments(unittest.TestCase):
    def _load_yaml_arguments_with_fake_omegaconf(self):
        class FakeOmegaConf:
            @staticmethod
            def create(value):
                return FakeDictConfig(value)

            @staticmethod
            def to_container(value, resolve=True):
                return dict(value)

        fake_omegaconf = types.SimpleNamespace(
            DictConfig=FakeDictConfig,
            OmegaConf=FakeOmegaConf,
        )

        module_name = "paddlefleet.training.yaml_arguments"
        module_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "paddlefleet"
            / "training"
            / "yaml_arguments.py"
        )
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        yaml_arguments = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"omegaconf": fake_omegaconf}):
            spec.loader.exec_module(yaml_arguments)
        return yaml_arguments

    def test_deepep_buffer_configs_keeps_dict_value(self):
        yaml_arguments = self._load_yaml_arguments_with_fake_omegaconf()
        cfg = FakeDictConfig(
            {
                "model": FakeDictConfig(
                    {
                        "num_hidden_layers": 2,
                        "deepep_buffer_configs": FakeDictConfig(
                            {
                                "num_sms": 24,
                                "dispatch_config": [60, 256],
                                "combine_config": [20, 256],
                            }
                        ),
                    }
                )
            }
        )

        result = yaml_arguments._flatten_configs(cfg)

        self.assertEqual(result.num_hidden_layers, 2)
        self.assertEqual(
            result.deepep_buffer_configs,
            {
                "num_sms": 24,
                "dispatch_config": [60, 256],
                "combine_config": [20, 256],
            },
        )
        self.assertFalse(hasattr(result, "num_sms"))

    def test_regular_nested_config_still_flattens(self):
        yaml_arguments = self._load_yaml_arguments_with_fake_omegaconf()
        cfg = FakeDictConfig(
            {
                "model": FakeDictConfig({"num_hidden_layers": 2}),
                "training": FakeDictConfig({"micro_batch_size": 4}),
            }
        )

        result = yaml_arguments._flatten_configs(cfg)

        self.assertEqual(result.num_hidden_layers, 2)
        self.assertEqual(result.micro_batch_size, 4)
        self.assertFalse(hasattr(result, "model"))
        self.assertFalse(hasattr(result, "training"))

    def test_core_config_receives_deepep_buffer_configs(self):
        yaml_arguments = self._load_yaml_arguments_with_fake_omegaconf()
        args = yaml_arguments._flatten_configs(
            FakeDictConfig(
                {
                    "model": FakeDictConfig(
                        {
                            "num_hidden_layers": 2,
                            "deepep_buffer_configs": FakeDictConfig(
                                {"num_sms": 24}
                            ),
                        }
                    )
                }
            )
        )

        config = core_transformer_config_from_args(args)

        self.assertEqual(config.deepep_buffer_configs, {"num_sms": 24})


if __name__ == "__main__":
    unittest.main()
