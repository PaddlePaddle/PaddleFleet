# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock

from paddle import nn

from paddlefleet.transformers import model_utils
from paddlefleet.transformers.aoa_config_base import MoEAOAConfigGenerator
from paddlefleet.transformers.configuration_utils import PretrainedConfig
from paddlefleet.transformers.model_utils import (
    _legacy_autoregressive_mtp_export_config,
)


class TestLegacyAutoregressiveMtpExportConfig(TestCase):
    def test_zero_legacy_field_keeps_canonical_config(self):
        for value in (None, 0):
            with self.subTest(value=value):
                config = SimpleNamespace(
                    mtp_num_layers=value, num_nextn_predict_layers=2
                )
                with _legacy_autoregressive_mtp_export_config(config):
                    self.assertEqual(config.mtp_num_layers, value)
                    self.assertEqual(config.num_nextn_predict_layers, 2)

    def test_invalid_legacy_value_is_rejected(self):
        for value, error in (
            ("1", TypeError),
            (True, TypeError),
            (-1, ValueError),
        ):
            with self.subTest(value=value):
                config = SimpleNamespace(
                    mtp_num_layers=value, num_nextn_predict_layers=2
                )
                with (
                    self.assertRaisesRegex(error, "mtp_num_layers"),
                    _legacy_autoregressive_mtp_export_config(config),
                ):
                    self.fail("Invalid MTP configuration entered export")
                self.assertEqual(config.mtp_num_layers, value)
                self.assertEqual(config.num_nextn_predict_layers, 2)

    def test_invalid_canonical_value_is_rejected_before_mutation(self):
        for value in (None, "2", True):
            with self.subTest(value=value):
                config = SimpleNamespace(
                    mtp_num_layers=1, num_nextn_predict_layers=value
                )
                with (
                    self.assertRaisesRegex(
                        TypeError, "num_nextn_predict_layers"
                    ),
                    _legacy_autoregressive_mtp_export_config(config),
                ):
                    self.fail("Invalid MTP configuration entered export")
                self.assertEqual(config.mtp_num_layers, 1)
                self.assertEqual(config.num_nextn_predict_layers, value)


class _ExportModel(model_utils.PretrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.projection = nn.Linear(2, 3)


class _InverseExportModel(_ExportModel):
    def _gen_inv_aoa_config(self, config):
        return MoEAOAConfigGenerator.gen_inv_aoa_config(config)


class _ForwardExportModel(_ExportModel):
    def _gen_aoa_config(self, config):
        return MoEAOAConfigGenerator.gen_aoa_config(config)


class TestMtpSavePretrained(TestCase):
    """Exercise export orchestration and real config I/O, without collectives."""

    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.save_dir = stack.enter_context(TemporaryDirectory())
        self.saver = stack.enter_context(
            mock.patch.object(model_utils, "HFFormatFullParamSaver")
        )
        self.world_size = stack.enter_context(
            mock.patch.object(
                model_utils.dist, "get_world_size", return_value=1
            )
        )
        stack.enter_context(mock.patch.object(model_utils.dist, "barrier"))
        self.config = PretrainedConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            mtp_num_layers=1,
            num_nextn_predict_layers=2,
        )
        self.model = _InverseExportModel(self.config)

    def test_export_uses_legacy_layer_count_and_restores_runtime_config(self):
        self.model.save_pretrained(self.save_dir)

        model, aoa = self.saver.call_args.args
        self.assertIs(model, self.model)
        mtp_projections = [
            item for item in aoa["aoa_statements"] if ".eh_proj." in item
        ]
        self.assertEqual(
            mtp_projections,
            [
                "model.layers.1.eh_proj.weight^T -> model.layers.1.eh_proj.weight"
            ],
        )
        saved = PretrainedConfig.from_pretrained(self.save_dir)
        self.assertEqual(saved.num_nextn_predict_layers, 1)
        self.assertEqual(saved.mtp_num_layers, 2)
        self.assertEqual(self.config.num_nextn_predict_layers, 2)
        self.assertEqual(self.config.mtp_num_layers, 1)
        self.saver.return_value.save_checkpoint.assert_called_once_with(
            self.save_dir, "10GB"
        )

    def test_canonical_only_config_can_be_exported(self):
        del self.config.mtp_num_layers

        self.model.save_pretrained(self.save_dir)

        saved = PretrainedConfig.from_pretrained(self.save_dir)
        self.assertEqual(saved.num_nextn_predict_layers, 2)
        self.assertFalse(hasattr(self.config, "mtp_num_layers"))

    def test_forward_aoa_is_reversed_with_legacy_layer_count(self):
        model = _ForwardExportModel(self.config)

        model.save_pretrained(self.save_dir)

        saved_model, aoa = self.saver.call_args.args
        self.assertIs(saved_model, model)
        self.assertTrue(aoa["aoa_config_reverse"])
        self.assertEqual(
            [item for item in aoa["aoa_statements"] if ".eh_proj." in item],
            [
                "model.layers.1.eh_proj.weight^T -> model.layers.1.eh_proj.weight"
            ],
        )
        saved = PretrainedConfig.from_pretrained(self.save_dir)
        self.assertEqual(saved.num_nextn_predict_layers, 1)
        self.assertEqual(self.config.mtp_num_layers, 1)
        self.assertEqual(self.config.num_nextn_predict_layers, 2)

    def test_missing_aoa_generator_restores_runtime_config(self):
        model = _ExportModel(self.config)

        with self.assertRaisesRegex(RuntimeError, "must implement either"):
            model.save_pretrained(self.save_dir)

        self.saver.assert_not_called()
        self.assertEqual(self.config.mtp_num_layers, 1)
        self.assertEqual(self.config.num_nextn_predict_layers, 2)

    def test_saver_failure_restores_runtime_config(self):
        self.saver.return_value.save_checkpoint.side_effect = RuntimeError(
            "checkpoint failed"
        )

        with self.assertRaisesRegex(RuntimeError, "checkpoint failed"):
            self.model.save_pretrained(self.save_dir)

        self.assertEqual(self.config.mtp_num_layers, 1)
        self.assertEqual(self.config.num_nextn_predict_layers, 2)

    def test_sonic_saver_failure_restores_runtime_config(self):
        self.config.using_sonic_moe = True
        with mock.patch.object(
            model_utils, "SonicMoEHFFormatFullParamSaver"
        ) as sonic_saver:
            sonic_saver.return_value.save_checkpoint.side_effect = RuntimeError(
                "sonic checkpoint failed"
            )

            with self.assertRaisesRegex(
                RuntimeError, "sonic checkpoint failed"
            ):
                self.model.save_pretrained(self.save_dir)

            sonic_saver.return_value.save_checkpoint.assert_called_once_with(
                self.save_dir, "10GB"
            )

        self.saver.assert_not_called()
        self.assertEqual(self.config.mtp_num_layers, 1)
        self.assertEqual(self.config.num_nextn_predict_layers, 2)

    def test_model_export_config_is_copied_before_tp_normalization(self):
        self.model.config_to_save = PretrainedConfig(
            tensor_model_parallel_size=2,
            num_nextn_predict_layers=1,
            architectures=["_ExportModel"],
        )

        self.model.save_pretrained(self.save_dir)

        saved = PretrainedConfig.from_pretrained(self.save_dir)
        self.assertEqual(saved.tensor_model_parallel_size, 1)
        self.assertEqual(saved.num_nextn_predict_layers, 1)
        self.assertEqual(saved.architectures, ["_ExportModel"])
        self.assertEqual(
            self.model.config_to_save.tensor_model_parallel_size, 2
        )
        self.assertEqual(self.config.mtp_num_layers, 1)
        self.assertEqual(self.config.num_nextn_predict_layers, 2)

    def test_non_main_process_saves_weights_without_config_files(self):
        self.model.save_pretrained(self.save_dir, is_main_process=False)

        self.saver.return_value.save_checkpoint.assert_called_once_with(
            self.save_dir, "10GB"
        )
        self.assertEqual(list(Path(self.save_dir).iterdir()), [])
        self.assertEqual(self.config.mtp_num_layers, 1)
        self.assertEqual(self.config.num_nextn_predict_layers, 2)

    def test_config_save_failure_restores_runtime_config(self):
        with (
            mock.patch.object(
                PretrainedConfig,
                "save_pretrained",
                side_effect=OSError("config save failed"),
            ),
            self.assertRaisesRegex(OSError, "config save failed"),
        ):
            self.model.save_pretrained(self.save_dir)

        self.assertEqual(self.config.mtp_num_layers, 1)
        self.assertEqual(self.config.num_nextn_predict_layers, 2)

    def test_parallel_export_passes_groups_and_shard_to_saver(self):
        pp = SimpleNamespace(nranks=2)
        ep = SimpleNamespace(nranks=4)
        sharding = SimpleNamespace(nranks=2, rank=1)
        hcg = SimpleNamespace(
            get_pipe_parallel_group=lambda: pp,
            get_expert_parallel_group=lambda: ep,
            get_moe_sharding_parallel_group=lambda: sharding,
        )
        self.world_size.return_value = 16

        with (
            mock.patch.object(model_utils.dist.fleet, "_hcg", hcg, create=True),
            mock.patch.object(
                model_utils.dist.fleet,
                "get_hybrid_communicate_group",
                return_value=hcg,
            ),
        ):
            self.model.save_pretrained(self.save_dir)

        kwargs = self.saver.call_args.kwargs
        self.assertIs(kwargs["model"], self.model)
        self.assertIs(kwargs["h_group"], ep)
        self.assertIs(kwargs["v_group"], pp)
        self.assertEqual(kwargs["num_splits"], 2)
        self.assertEqual(kwargs["shard_idx"], 1)

    def test_non_moe_hcg_uses_serial_saver(self):
        hcg = SimpleNamespace(
            get_pipe_parallel_group=lambda: SimpleNamespace(nranks=1)
        )
        self.world_size.return_value = 2

        with (
            mock.patch.object(model_utils.dist.fleet, "_hcg", hcg, create=True),
            mock.patch.object(
                model_utils.dist.fleet,
                "get_hybrid_communicate_group",
                return_value=hcg,
            ),
        ):
            self.model.save_pretrained(self.save_dir)

        self.assertIs(self.saver.call_args.args[0], self.model)
        self.assertEqual(
            self.saver.call_args.kwargs, {"memory_growth_threshold": 8 * 2**30}
        )

    def test_group_lookup_failure_propagates_and_restores_config(self):
        self.world_size.return_value = 4
        for method in (
            "get_pipe_parallel_group",
            "get_expert_parallel_group",
            "get_moe_sharding_parallel_group",
        ):
            hcg = SimpleNamespace(
                get_pipe_parallel_group=lambda: SimpleNamespace(nranks=2),
                get_expert_parallel_group=lambda: SimpleNamespace(nranks=2),
                get_moe_sharding_parallel_group=lambda: SimpleNamespace(
                    nranks=1, rank=0
                ),
            )
            setattr(hcg, method, mock.Mock(side_effect=RuntimeError(method)))
            with (
                self.subTest(method=method),
                mock.patch.object(
                    model_utils.dist.fleet, "_hcg", hcg, create=True
                ),
                mock.patch.object(
                    model_utils.dist.fleet,
                    "get_hybrid_communicate_group",
                    return_value=hcg,
                ),
                self.assertRaisesRegex(RuntimeError, method),
            ):
                self.model.save_pretrained(self.save_dir)

            self.saver.assert_not_called()
            self.assertEqual(self.config.mtp_num_layers, 1)
            self.assertEqual(self.config.num_nextn_predict_layers, 2)


if __name__ == "__main__":
    import unittest

    unittest.main()
