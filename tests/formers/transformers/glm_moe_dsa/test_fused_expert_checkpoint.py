# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
"""Focused tests for fused-MoE save at sharding_parallel_size=1 and HF export cadence."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from paddlefleet.trainer.trainer import Trainer
from paddlefleet.trainer.trainer_callback import (
    DefaultFlowCallback,
    TrainerControl,
    TrainerState,
)
from paddlefleet.trainer.trainer_utils import IntervalStrategy
from paddlefleet.trainer.training_args import _resolve_save_hf_steps


class TestRestoreFusedExpert3DLayout(unittest.TestCase):
    def test_restores_flattened_grouped_gemm_weight(self):
        import paddle
        from paddle.distributed import ShardedWeight

        from paddlefleet.trainer.trainer import restore_fused_expert_3d_layout

        key = "model.layers.3.mlp.grouped_gemm_experts.weight1"
        param = paddle.zeros([2, 4, 6], dtype="float32")
        flat = param.reshape([8, 6])
        shard = ShardedWeight(
            key=key,
            local_tensor=flat,
            local_shape=tuple(flat.shape),
            global_shape=tuple(flat.shape),
            global_offset=(0, 0),
        )
        model = MagicMock()
        model.named_parameters.return_value = [(key, param)]

        restore_fused_expert_3d_layout(model, {key: shard})

        self.assertEqual(tuple(shard.local_tensor.shape), (2, 4, 6))
        self.assertEqual(shard.local_shape, (2, 4, 6))
        self.assertEqual(shard.global_shape, (2, 4, 6))
        self.assertEqual(shard.global_offset, (0, 0, 0))

    def expert_shard(self, key, global_shape, global_offset):
        import paddle
        from paddle.distributed import ShardedWeight

        param = paddle.zeros([2, 4, 6], dtype="float32")
        flat = param.reshape([8, 6])
        shard = ShardedWeight(
            key=key,
            local_tensor=flat,
            local_shape=tuple(flat.shape),
            global_shape=global_shape,
            global_offset=global_offset,
        )
        return param, shard

    def test_keeps_expert_parallel_global_coordinates(self):
        from paddlefleet.trainer.trainer import restore_fused_expert_3d_layout

        key = "model.layers.3.mlp.grouped_gemm_experts.weight1"
        # EP rank 1 of 2: experts [2, 4) of 4, flattened to rows [8, 16).
        param, shard = self.expert_shard(key, (16, 6), (8, 0))
        model = MagicMock()
        model.named_parameters.return_value = [(key, param)]

        restore_fused_expert_3d_layout(model, {key: shard})

        self.assertEqual(shard.local_shape, (2, 4, 6))
        self.assertEqual(shard.global_shape, (4, 4, 6))
        self.assertEqual(shard.global_offset, (2, 0, 0))

    def test_rejects_shards_that_split_an_expert(self):
        from paddlefleet.trainer.trainer import restore_fused_expert_3d_layout

        key = "model.layers.3.mlp.grouped_gemm_experts.weight1"
        param, shard = self.expert_shard(key, (16, 6), (6, 0))
        model = MagicMock()
        model.named_parameters.return_value = [(key, param)]

        with self.assertRaisesRegex(ValueError, "whole"):
            restore_fused_expert_3d_layout(model, {key: shard})

    def test_resolves_pipeline_parameter_names(self):
        from paddlefleet.trainer.trainer import restore_fused_expert_3d_layout

        single_key = "model.layers.3.mlp.grouped_gemm_experts.weight1"
        pp_key = "4.mlp.grouped_gemm_experts.weight1"
        param, shard = self.expert_shard(single_key, (8, 6), (0, 0))
        model = MagicMock()
        model.named_parameters.return_value = [(pp_key, param)]
        model._pipeline_name_mapping = {single_key: pp_key}

        restore_fused_expert_3d_layout(model, {single_key: shard})

        self.assertEqual(shard.local_shape, (2, 4, 6))
        self.assertEqual(shard.global_shape, (2, 4, 6))

    def test_rejects_missing_or_non_3d_parameter(self):
        import paddle
        from paddle.distributed import ShardedWeight
        from paddlefleet.trainer.trainer import restore_fused_expert_3d_layout

        key = "model.layers.3.mlp.grouped_gemm_experts.weight1"
        shard = ShardedWeight(
            key=key,
            local_tensor=paddle.zeros([8, 6], dtype="float32"),
            local_shape=(8, 6),
            global_shape=(8, 6),
            global_offset=(0, 0),
        )
        model = MagicMock()
        model.named_parameters.return_value = []
        with self.assertRaisesRegex(ValueError, "no matching model parameter"):
            restore_fused_expert_3d_layout(model, {key: shard})

        model.named_parameters.return_value = [
            (key, paddle.zeros([8, 6], dtype="float32"))
        ]
        with self.assertRaisesRegex(ValueError, "3-D model parameter"):
            restore_fused_expert_3d_layout(model, {key: shard})

    def test_skips_shards_already_in_3d_layout(self):
        import paddle
        from paddle.distributed import ShardedWeight
        from paddlefleet.trainer.trainer import restore_fused_expert_3d_layout

        # Qwen3-VL keeps grouped-GEMM experts 3-D and names them under
        # ``model.language_model``; nothing needs restoring, so the model's
        # parameters must not even be looked up.
        key = "model.language_model.layers.0.mlp.grouped_gemm_experts.weight1"
        local = paddle.zeros([2, 4, 6], dtype="float32")
        shard = ShardedWeight(
            key=key,
            local_tensor=local,
            local_shape=(2, 4, 6),
            global_shape=(2, 4, 6),
            global_offset=(0, 0, 0),
        )

        restore_fused_expert_3d_layout(SimpleNamespace(), {key: shard})

        self.assertIs(shard.local_tensor, local)
        self.assertEqual(shard.local_shape, (2, 4, 6))
        self.assertEqual(shard.global_offset, (0, 0, 0))


class TestFusedExpertOptimizerSave(unittest.TestCase):
    def make_trainer(self, dtype="bfloat16"):
        import paddle
        from paddle.distributed import ShardedWeight

        class Model(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.grouped_gemm_experts = paddle.nn.Layer()
                self.grouped_gemm_experts.add_parameter(
                    "weight1",
                    self.create_parameter(
                        [2, 4, 6],
                        dtype=dtype,
                        default_initializer=paddle.nn.initializer.Constant(
                            0.25
                        ),
                    ),
                )
                self.add_parameter(
                    "unrelated",
                    self.create_parameter(
                        [2, 3, 4],
                        dtype=dtype,
                        default_initializer=paddle.nn.initializer.Constant(0.5),
                    ),
                )

            def sharded_state_dict(self):
                result = {}
                for key, param in self.named_parameters():
                    tensor = (
                        param.reshape([-1, param.shape[-1]])
                        if key.startswith("grouped_gemm_experts")
                        else param
                    )
                    tensor.name = param.name
                    result[key] = ShardedWeight(
                        key,
                        tensor,
                        tuple(tensor.shape),
                        tuple(tensor.shape),
                        (0,) * tensor.ndim,
                    )
                return result

        trainer = object.__new__(Trainer)
        trainer.model = Model()
        trainer.optimizer = paddle.optimizer.AdamW(
            learning_rate=0.01,
            parameters=trainer.model.parameters(),
            multi_precision=True,
        )
        trainer.args = SimpleNamespace(replicate_saved_into_local=False)
        self.step(trainer)
        return trainer

    @staticmethod
    def step(trainer):
        loss = sum(
            (param.astype("float32") ** 2).sum()
            for param in trainer.model.parameters()
        )
        loss.backward()
        trainer.optimizer.step()
        trainer.optimizer.clear_grad()

    @staticmethod
    def snapshot(optimizer):
        return [
            (mapping, key, value, tuple(value.shape))
            for mapping in [
                *optimizer._accumulators.values(),
                optimizer._master_weights,
            ]
            for key, value in mapping.items()
        ]

    def assert_unchanged(self, snapshot):
        for mapping, key, tensor, shape in snapshot:
            self.assertIs(mapping[key], tensor)
            self.assertEqual(tuple(tensor.shape), shape)

    def test_save_load_then_step_matches_uninterrupted(self):
        import tempfile
        from pathlib import Path

        import numpy as np
        import paddle.distributed as dist

        from paddlefleet.trainer.trainer import (
            MASTER_WEIGHT_DIC,
            OPTIMIZER_STATE_DIC,
            _fused_expert_optimizer_save_views,
        )

        for dtype in ("float32", "bfloat16"):
            with (
                self.subTest(dtype=dtype),
                tempfile.TemporaryDirectory() as directory,
            ):
                trainer = self.make_trainer(dtype)
                snapshot = self.snapshot(trainer.optimizer)
                trainer._save_flex_optimizer_state(directory)
                self.assert_unchanged(snapshot)
                self.assertTrue((Path(directory) / "saved_signal_0").is_file())
                resumed = self.make_trainer(dtype)
                self.step(resumed)
                resumed.model.set_state_dict(trainer.model.state_dict())
                resumed_snapshot = self.snapshot(resumed.optimizer)
                with _fused_expert_optimizer_save_views(
                    resumed.model,
                    resumed.model.sharded_state_dict(),
                    resumed.optimizer,
                ):
                    shards = resumed.optimizer.sharded_state_dict(
                        resumed.model.sharded_state_dict()
                    )
                    dist.load_state_dict(
                        {
                            k: v
                            for k, v in shards.items()
                            if not k.endswith(".w_0")
                        },
                        str(Path(directory) / OPTIMIZER_STATE_DIC),
                    )
                    if dtype == "bfloat16":
                        dist.load_state_dict(
                            {
                                k: v
                                for k, v in shards.items()
                                if k.endswith(".w_0")
                            },
                            str(Path(directory) / MASTER_WEIGHT_DIC),
                        )
                self.assert_unchanged(resumed_snapshot)
                self.step(trainer)
                self.step(resumed)
                for left, right in zip(
                    trainer.model.parameters(), resumed.model.parameters()
                ):
                    np.testing.assert_array_equal(left.numpy(), right.numpy())
                for left_mapping, right_mapping in zip(
                    [
                        *trainer.optimizer._accumulators.values(),
                        trainer.optimizer._master_weights,
                    ],
                    [
                        *resumed.optimizer._accumulators.values(),
                        resumed.optimizer._master_weights,
                    ],
                ):
                    for left, right in zip(
                        left_mapping.values(), right_mapping.values()
                    ):
                        np.testing.assert_array_equal(
                            left.numpy(), right.numpy()
                        )

    def test_failures_restore_original_objects_and_shapes(self):
        import tempfile
        from unittest.mock import patch

        for location in ("sharded_state_dict", "save_state_dict"):
            with (
                self.subTest(location=location),
                tempfile.TemporaryDirectory() as directory,
            ):
                trainer = self.make_trainer()
                snapshot = self.snapshot(trainer.optimizer)
                target = (
                    trainer.optimizer
                    if location == "sharded_state_dict"
                    else __import__(
                        "paddle.distributed", fromlist=["save_state_dict"]
                    )
                )
                with (
                    patch.object(
                        target,
                        location,
                        side_effect=RuntimeError("save failure"),
                    ),
                    self.assertRaisesRegex(RuntimeError, "save failure"),
                ):
                    trainer._save_flex_optimizer_state(directory)
                self.assert_unchanged(snapshot)
                self.step(trainer)


class TestDefaultFlowCallbackSaveHf(unittest.TestCase):
    def args(self, save_hf_steps):
        return SimpleNamespace(
            logging_first_step=False,
            logging_strategy=IntervalStrategy.NO,
            logging_steps=1,
            evaluation_strategy=IntervalStrategy.NO,
            eval_steps=1,
            save_strategy=IntervalStrategy.STEPS,
            save_steps=5,
            flash_device_save_steps=0,
            save_last_step=False,
            save_hf_steps=save_hf_steps,
        )

    def test_save_to_hf_reuses_save_steps_when_save_hf_steps_default(self):
        save_hf_steps = _resolve_save_hf_steps(-1, 5, True)
        self.assertEqual(save_hf_steps, 5)
        state = TrainerState(global_step=5, max_steps=5)
        control = TrainerControl()
        DefaultFlowCallback().on_step_end(
            self.args(save_hf_steps), state, control
        )
        self.assertTrue(control.should_save_hf)
        self.assertTrue(control.should_save)

    def test_save_hf_stays_off_when_save_to_hf_false(self):
        save_hf_steps = _resolve_save_hf_steps(-1, 5, False)
        self.assertEqual(save_hf_steps, -1)
        state = TrainerState(global_step=5, max_steps=5)
        control = TrainerControl()
        DefaultFlowCallback().on_step_end(
            self.args(save_hf_steps), state, control
        )
        self.assertFalse(control.should_save_hf)

    def test_explicit_save_hf_steps_wins(self):
        self.assertEqual(_resolve_save_hf_steps(10, 5, True), 10)
        self.assertEqual(_resolve_save_hf_steps(-1, 0, True), -1)


if __name__ == "__main__":
    unittest.main()
