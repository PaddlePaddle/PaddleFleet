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

"""ZCC ema-interval asserts run only when zero-cost checkpoint is on."""

import tempfile
from unittest.mock import patch

import pytest

from paddlefleet.trainer.training_args import TrainingArguments


def _args(**overrides):
    kwargs = {
        "output_dir": tempfile.mkdtemp(),
        "report_to": [],
        "bf16": True,
        "enable_zero_cost_checkpoint": True,
        "fuse_optimizer_states": True,
        "save_steps": 20,
        "flash_device_save_steps": 20,
        "zcc_ema_interval": 20,
    }
    kwargs.update(overrides)
    with (
        patch("paddlefleet.trainer.training_args.fleet.init"),
        patch("paddlefleet.trainer.training_args.initialize_fleet"),
        patch("paddle.distributed.get_world_size", return_value=1),
    ):
        return TrainingArguments(**kwargs)


def test_zcc_accepts_interval_multiples():
    _args()


def test_zcc_rejects_non_multiple_save_steps():
    with pytest.raises(AssertionError, match=r"save_steps\[4\]"):
        _args(save_steps=4)


def test_zcc_rejects_non_multiple_flash_device_save_steps():
    with pytest.raises(AssertionError, match=r"flash_device_save_steps\[10\]"):
        _args(flash_device_save_steps=10)


def test_zcc_off_skips_interval_check():
    _args(
        enable_zero_cost_checkpoint=False,
        save_steps=4,
        flash_device_save_steps=0,
    )
