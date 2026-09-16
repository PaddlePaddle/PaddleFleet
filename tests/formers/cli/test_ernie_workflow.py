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

"""Behavior tests for the ERNIE pretrain workflow.

Module under test:
``paddlefleet.cli.train.ernie_pretrain.workflow``. In the repository module map
this workflow belongs to the "Trainer 训练引擎" layer: a workflow assembles the
model config + args and launches the Trainer. These tests verify the workflow's
*real* config-plumbing decisions with hand-derived expectations, never by
re-invoking the code under test to produce the expected values.

What is genuinely exercised here:

* ``update_model_config_from_args`` -- the production code decides, per key,
  via ``hasattr(config, k)`` whether to ``setattr`` an override or skip it with
  a warning. A *real* lightweight config object (not a MagicMock) is used so the
  genuine ``hasattr``/``setattr`` path runs: existing keys must be overwritten
  to their exact new values, an unknown key must NOT be injected onto the
  object, and the same object instance must be returned (identity, not a copy).

* ``get_tp_split_ckpt`` -- the tensor-parallel checkpoint path construction.
  The degree/rank are zero-padded to two digits and the ``tp_degree > 1`` branch
  is a real fork versus the flat single-shard path. ``tensor_parallel_rank`` is
  clamped to a floor of 0 via ``max(rank, 0)``. Expected paths are written as
  literal strings derived by hand from the production format specifiers.

* ``ExpConfig`` -- the dataclass field contract: exactly the fields
  ``max_steps``, ``name`` and ``config``, carrying the exact values supplied,
  round-tripping through ``dataclasses.asdict``.

These tests run on CPU. Importing the production module pulls the ``paddlefleet``
package, which imports Paddle at import time; when Paddle is absent the import
raises ``ImportError`` and every test skips (recorded, not silently passed). No
production code is modified by this file.
"""

import dataclasses
import os
import types
import unittest

try:
    from paddlefleet.cli.train.ernie_pretrain.workflow import (
        ExpConfig,
        get_tp_split_ckpt,
        update_model_config_from_args,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # Paddle backend / paddlefleet not installed.
    ExpConfig = None
    get_tp_split_ckpt = None
    update_model_config_from_args = None
    _IMPORT_ERROR = exc


class _WorkflowTestBase(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(
                "paddlefleet import failed (dependency unavailable): "
                f"{_IMPORT_ERROR!r}"
            )


class _PlainConfig:
    """A real, minimal stand-in for ErnieMoEConfig.

    It exposes only ordinary attributes so that ``hasattr``/``setattr`` -- the
    exact primitives the production function relies on -- run for real. Using a
    plain object (rather than a MagicMock) is deliberate: a MagicMock answers
    ``hasattr`` truthy for every name and would mask a regression where an
    unknown key is silently attached.
    """

    def __init__(self, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers = num_layers


class TestUpdateModelConfigFromArgs(_WorkflowTestBase):
    def test_existing_keys_overwritten_unknown_key_skipped(self):
        config = _PlainConfig(hidden_size=768, num_layers=2)
        model_args = {
            "hidden_size": 1024,  # existing -> must be overwritten
            "num_layers": 4,  # existing -> must be overwritten
            "unknown_flag": 42,  # absent   -> must NOT be attached
        }

        result = update_model_config_from_args(config, model_args)

        # Same object returned (mutated in place, not copied).
        self.assertIs(result, config)
        # Existing keys take the exact override values.
        self.assertEqual(config.hidden_size, 1024)
        self.assertEqual(config.num_layers, 4)
        # Unknown key is rejected -- it must not appear on the object.
        self.assertFalse(hasattr(config, "unknown_flag"))

    def test_empty_args_leaves_config_untouched(self):
        config = _PlainConfig(hidden_size=768, num_layers=2)

        result = update_model_config_from_args(config, {})

        self.assertIs(result, config)
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.num_layers, 2)

    def test_falsy_override_value_is_still_applied(self):
        # ``hasattr`` gates the write, not the truthiness of the value, so a
        # falsy override (0) must replace a truthy existing value.
        config = _PlainConfig(hidden_size=768, num_layers=2)

        update_model_config_from_args(config, {"num_layers": 0})

        self.assertEqual(config.num_layers, 0)


class TestGetTpSplitCkpt(_WorkflowTestBase):
    def test_degree_one_returns_flat_single_shard_path(self):
        args = types.SimpleNamespace(
            tensor_model_parallel_size=1, tensor_parallel_rank=0
        )
        self.assertEqual(
            get_tp_split_ckpt(args, "/some/path"),
            os.path.join("/some/path", "model_state.pdparams"),
        )

    def test_degree_two_rank_zero_two_digit_padding(self):
        args = types.SimpleNamespace(
            tensor_model_parallel_size=2, tensor_parallel_rank=0
        )
        # Hand-derived: tp02 dir, tp00 shard.
        self.assertEqual(
            get_tp_split_ckpt(args, "/some/path"),
            os.path.join("/some/path", "tp02", "model_state.tp00.pdparams"),
        )

    def test_two_digit_padding_distinguishes_degree_and_rank(self):
        # Degree 16 / rank 10 keeps both fields distinct and confirms the
        # zero-pad width is two (no truncation, no over-padding).
        args = types.SimpleNamespace(
            tensor_model_parallel_size=16, tensor_parallel_rank=10
        )
        self.assertEqual(
            get_tp_split_ckpt(args, "/model/dir"),
            os.path.join("/model/dir", "tp16", "model_state.tp10.pdparams"),
        )

    def test_negative_rank_clamped_to_zero_in_multi_shard_path(self):
        # ``max(rank, 0)`` floors the rank; degree stays >1 so we are in the
        # tp-split branch and the shard index must be 00, not a negative value.
        args = types.SimpleNamespace(
            tensor_model_parallel_size=8, tensor_parallel_rank=-5
        )
        self.assertEqual(
            get_tp_split_ckpt(args, "/model/dir"),
            os.path.join("/model/dir", "tp08", "model_state.tp00.pdparams"),
        )

    def test_negative_rank_with_degree_one_uses_flat_path(self):
        args = types.SimpleNamespace(
            tensor_model_parallel_size=1, tensor_parallel_rank=-1
        )
        self.assertEqual(
            get_tp_split_ckpt(args, "/some/path"),
            os.path.join("/some/path", "model_state.pdparams"),
        )


class TestExpConfig(_WorkflowTestBase):
    def test_fields_and_values_round_trip(self):
        config = ExpConfig(
            max_steps=1000,
            name="ernie_exp",
            config={"lr": "1e-4", "warmup": "100"},
        )

        # Exact field surface -- catches rename/removal/addition.
        field_names = {f.name for f in dataclasses.fields(ExpConfig)}
        self.assertEqual(field_names, {"max_steps", "name", "config"})

        # Exact stored values.
        self.assertEqual(config.max_steps, 1000)
        self.assertEqual(config.name, "ernie_exp")
        self.assertEqual(config.config, {"lr": "1e-4", "warmup": "100"})

        # asdict reproduces the full mapping with the same values.
        self.assertEqual(
            dataclasses.asdict(config),
            {
                "max_steps": 1000,
                "name": "ernie_exp",
                "config": {"lr": "1e-4", "warmup": "100"},
            },
        )


if __name__ == "__main__":
    unittest.main()
