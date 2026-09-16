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

"""Tests for the sharding_machine_balanced_2d_partition switch."""

import dataclasses
import types
import unittest
from unittest.mock import patch

from paddlefleet.trainer import TrainingArguments
from paddlefleet.trainer.argparser import PdArgumentParser


class ReachedFleetInit(Exception):
    """Indicates that TrainingArguments reached fleet.init."""


class FakeStrategy:
    """Minimal strategy that emulates Paddle's hybrid_configs setter."""

    def __init__(self, sharding_configs):
        self.sharding_configs = sharding_configs
        self._hybrid_configs = {}
        self.use_muon_sharding = False

    @property
    def hybrid_configs(self):
        return self._hybrid_configs

    @hybrid_configs.setter
    def hybrid_configs(self, configs):
        # Paddle normally creates sharding_configs when this dict is assigned.
        # Inject it here so the test does not depend on the installed Paddle
        # wheel containing the new proto field.
        configs = dict(configs)
        configs["sharding_configs"] = self.sharding_configs
        self._hybrid_configs = configs


class TestShardingMachineBalanced2DPartition(unittest.TestCase):
    def _run_training_args(self, sharding_configs, **overrides):
        args_dict = {
            "output_dir": "/tmp/paddlefleet_test_output",
            "bf16": True,
            "sharding": "stage1",
            "sharding_parallel_size": 8,
            "amp_master_grad": True,
            "split_param": True,
        }
        args_dict.update(overrides)

        strategy = FakeStrategy(sharding_configs)
        parser = PdArgumentParser((TrainingArguments,))

        with (
            patch(
                "paddlefleet.trainer.training_args.fleet.DistributedStrategy",
                return_value=strategy,
            ),
            patch(
                "paddlefleet.trainer.training_args.fleet.init",
                side_effect=ReachedFleetInit,
            ),
            patch(
                "paddlefleet.trainer.training_args.dist.get_world_size",
                return_value=8,
            ),
            patch(
                "paddle.distributed.parallel.parallel_helper."
                "_is_parallel_ctx_initialized",
                return_value=False,
            ),
        ):
            try:
                parser.parse_dict(args_dict)
            except ReachedFleetInit:
                pass

        return strategy

    def test_field_defaults_to_false(self):
        fields = {f.name: f for f in dataclasses.fields(TrainingArguments)}
        self.assertIn("sharding_machine_balanced_2d_partition", fields)
        self.assertFalse(
            fields["sharding_machine_balanced_2d_partition"].default
        )

    def test_default_off_does_not_trigger_validation(self):
        # Even if the optimizer is not Muon and Paddle does not have the
        # proto field, the check must be skipped when the flag is False.
        sharding_configs = types.SimpleNamespace()

        self._run_training_args(
            sharding_configs,
            optim="adamw",
            sharding_machine_balanced_2d_partition=False,
        )

    def test_non_muon_raises_value_error(self):
        # Covers the first validation branch.
        sharding_configs = types.SimpleNamespace(
            machine_balanced_2d_partition=False
        )

        with self.assertRaisesRegex(ValueError, "only supports Muon"):
            self._run_training_args(
                sharding_configs,
                optim="adamw",
                sharding_machine_balanced_2d_partition=True,
            )

    def test_old_paddle_raises_value_error(self):
        # Covers the validation branch for an older Paddle proto.
        sharding_configs = types.SimpleNamespace()

        with self.assertRaisesRegex(ValueError, "not supported by current"):
            self._run_training_args(
                sharding_configs,
                optim="muon",
                sharding_machine_balanced_2d_partition=True,
            )

    def test_muon_sets_machine_balanced_flag(self):
        # Covers the assignment in the successful path.
        sharding_configs = types.SimpleNamespace(
            machine_balanced_2d_partition=False
        )

        strategy = self._run_training_args(
            sharding_configs,
            optim="muon",
            sharding_machine_balanced_2d_partition=True,
        )

        self.assertTrue(
            strategy.hybrid_configs[
                "sharding_configs"
            ].machine_balanced_2d_partition
        )


if __name__ == "__main__":
    unittest.main()
