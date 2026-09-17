# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import os
import shutil
import tempfile
import unittest

from tensorboard.backend.event_processing import event_accumulator
from visualdl import LogReader

from paddlefleet.trainer import TrainerControl, TrainerState, TrainingArguments
from paddlefleet.trainer.integrations import (
    SwanLabCallback,
    TensorBoardCallback,
    VisualDLCallback,
    WandbCallback,
)
from tests.formers.trainer.trainer_utils import (
    RegressionModelConfig,
    RegressionPretrainedModel,
)


class _SingleCardGpuEnv(unittest.TestCase):
    """Replicate the launcher-provided single-card GPU environment.

    ``TrainingArguments(bf16=True)`` runs ``__post_init__`` which probes
    ``paddle.device.get_device_capability()`` (via the ``is_sm100`` check). On
    the H20/A100 CI runner the launcher exports ``FLAGS_selected_gpus`` and pins
    a card, but a bare pytest process leaves the current device as CPU, so the
    capability probe calls ``get_device_properties(Place(cpu))`` and raises
    ``ValueError``. Pin an explicit single card as the launcher would (and
    restore afterwards). These constructions are GPU-only paths, so skip
    honestly on a CPU-only build rather than faking a pass.
    """

    def setUp(self):
        import paddle

        if (
            not paddle.is_compiled_with_cuda()
            or paddle.device.cuda.device_count() == 0
        ):
            self.skipTest(
                "TrainingArguments(bf16=True) probes GPU device capability; "
                "this build has no usable CUDA device"
            )
        self._orig_device = paddle.get_device()
        self._orig_selected_gpus = os.environ.get("FLAGS_selected_gpus")
        os.environ["FLAGS_selected_gpus"] = "0"
        paddle.set_device("gpu:0")
        self.addCleanup(self._restore_gpu_env)

    def _restore_gpu_env(self):
        import paddle

        paddle.set_device(self._orig_device)
        if self._orig_selected_gpus is None:
            os.environ.pop("FLAGS_selected_gpus", None)
        else:
            os.environ["FLAGS_selected_gpus"] = self._orig_selected_gpus


class TestWandbCallback(_SingleCardGpuEnv):
    def test_wandbcallback(self):
        output_dir = tempfile.mkdtemp()
        args = TrainingArguments(
            output_dir=output_dir,
            max_steps=200,
            logging_steps=20,
            run_name="test_wandbcallback",
            logging_dir=output_dir,
            bf16=True,
        )
        state = TrainerState(trial_name="PaddleFleet")
        control = TrainerControl()
        config = RegressionModelConfig(a=1, b=1)
        model = RegressionPretrainedModel(config)
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"
        os.environ["WANDB_MODE"] = "offline"
        wandbcallback = WandbCallback()
        self.assertFalse(wandbcallback._initialized)
        wandbcallback.on_train_begin(args, state, control)
        self.assertTrue(wandbcallback._initialized)
        self.assertEqual(wandbcallback._wandb.run.name, state.trial_name)
        self.assertEqual(wandbcallback._wandb.run.group, args.run_name)
        for global_step in range(args.max_steps):
            state.global_step = global_step
            if global_step % args.logging_steps == 0:
                log = {
                    "loss": 100 - 0.4 * global_step,
                    "learning_rate": 0.1,
                    "global_step": global_step,
                }
                wandbcallback.on_log(args, state, control, logs=log)
                self.assertEqual(
                    wandbcallback._wandb.run.summary["train/loss"], log["loss"]
                )
                self.assertEqual(
                    wandbcallback._wandb.run.summary["train/learning_rate"],
                    log["learning_rate"],
                )
                self.assertEqual(
                    wandbcallback._wandb.run.summary["train/global_step"],
                    log["global_step"],
                )
        wandbcallback.on_train_end(args, state, control, model=model)
        wandbcallback._wandb.finish()
        os.environ.pop("WANDB_LOG_MODEL", None)
        os.environ.pop("WANDB_MODE", None)
        shutil.rmtree(output_dir)


class TestSwanlabCallback(_SingleCardGpuEnv):
    def test_swanlabcallback(self):
        output_dir = tempfile.mkdtemp()
        args = TrainingArguments(
            output_dir=output_dir,
            max_steps=200,
            logging_steps=20,
            run_name="test_swanlabcallback",
            logging_dir=output_dir,
            bf16=True,
        )
        state = TrainerState(trial_name="PaddleFleet")
        control = TrainerControl()
        config = RegressionModelConfig(a=1, b=1)
        model = RegressionPretrainedModel(config)
        os.environ["SWANLAB_MODE"] = "disabled"
        swanlabcallback = SwanLabCallback()
        self.assertFalse(swanlabcallback._initialized)
        swanlabcallback.on_train_begin(args, state, control)
        self.assertTrue(swanlabcallback._initialized)
        for global_step in range(args.max_steps):
            state.global_step = global_step
            if global_step % args.logging_steps == 0:
                log = {
                    "loss": 100 - 0.4 * global_step,
                    "learning_rate": 0.1,
                    "global_step": global_step,
                }
                swanlabcallback.on_log(args, state, control, logs=log)
        swanlabcallback.on_train_end(args, state, control, model=model)
        # In disabled mode, finish() should not be called as there is no active run
        if not swanlabcallback._disabled:
            swanlabcallback._swanlab.finish()
        os.environ.pop("SWANLAB_MODE", None)
        shutil.rmtree(output_dir)


class TestTensorboardCallback(_SingleCardGpuEnv):
    def test_tbcallback(self):
        output_dir = tempfile.mkdtemp()
        args = TrainingArguments(
            output_dir=output_dir,
            max_steps=200,
            logging_steps=20,
            run_name="test_tbcallback",
            logging_dir=output_dir,
            bf16=True,
        )
        state = TrainerState(trial_name="PaddleFleet")
        control = TrainerControl()
        tensorboard_callback = TensorBoardCallback()
        self.assertIsNone(tensorboard_callback.tb_writer)
        tensorboard_callback.on_train_begin(args, state, control)
        try:
            log_directory = tensorboard_callback.tb_writer.logdir
        except AttributeError:
            log_directory = tensorboard_callback.tb_writer.log_dir
        self.assertEqual(log_directory, output_dir)
        for global_step in range(args.max_steps):
            state.global_step = global_step
            if global_step % args.logging_steps == 0:
                log = {
                    "loss": 100 - 0.4 * global_step,
                    "learning_rate": 0.1,
                    "global_step": global_step,
                }
                tensorboard_callback.on_log(args, state, control, logs=log)
        ea = event_accumulator.EventAccumulator(output_dir)
        ea.Reload()
        loss_scalars = ea.Scalars("train/loss")
        learning_rate_scalars = ea.Scalars("train/learning_rate")
        global_step_scalars = ea.Scalars("train/global_step")
        for i, scalar in enumerate(loss_scalars):
            expected_loss = 100 - 0.4 * scalar.step
            self.assertAlmostEqual(scalar.value, expected_loss, places=5)
        for i, scalar in enumerate(learning_rate_scalars):
            expected_lr = 0.1
            self.assertAlmostEqual(scalar.value, expected_lr, places=5)
        for i, scalar in enumerate(global_step_scalars):
            expected_step = i * args.logging_steps
            self.assertEqual(scalar.value, expected_step)
        tensorboard_callback.on_train_end(args, state, control)
        self.assertIsNone(tensorboard_callback.tb_writer)
        shutil.rmtree(output_dir)


class TestVisualDLCallback(_SingleCardGpuEnv):
    def test_vdlcallback(self):
        output_dir = tempfile.mkdtemp()
        args = TrainingArguments(
            output_dir=output_dir,
            max_steps=200,
            logging_steps=20,
            run_name="test_vdlcallback",
            logging_dir=output_dir,
            bf16=True,
        )
        state = TrainerState(trial_name="PaddleFleet")
        control = TrainerControl()
        visualdl_callback = VisualDLCallback()
        self.assertIsNone(visualdl_callback.vdl_writer)
        visualdl_callback.on_train_begin(args, state, control)
        self.assertEqual(visualdl_callback.vdl_writer.logdir, output_dir)
        for global_step in range(args.max_steps):
            state.global_step = global_step
            if global_step % args.logging_steps == 0:
                log = {
                    "loss": 100 - 0.4 * global_step,
                    "learning_rate": 0.1,
                    "global_step": global_step,
                }
                visualdl_callback.on_log(args, state, control, logs=log)
        reader = LogReader(file_path=visualdl_callback.vdl_writer.file_name)
        loss_scalars = reader.get_data("scalar", "train/loss")
        learning_rate_scalars = reader.get_data("scalar", "train/learning_rate")
        global_step_scalars = reader.get_data("scalar", "train/global_step")
        for i, scalar in enumerate(loss_scalars):
            expected_loss = 100 - 0.4 * args.logging_steps * i
            self.assertAlmostEqual(scalar.value, expected_loss, places=5)
        for i, scalar in enumerate(learning_rate_scalars):
            expected_lr = 0.1
            self.assertAlmostEqual(scalar.value, expected_lr, places=5)
        for i, scalar in enumerate(global_step_scalars):
            expected_step = i * args.logging_steps
            self.assertEqual(scalar.value, expected_step)
        visualdl_callback.on_train_end(args, state, control)
        self.assertIsNone(visualdl_callback.vdl_writer)
        shutil.rmtree(output_dir)
