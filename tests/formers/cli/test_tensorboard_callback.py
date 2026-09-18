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

"""Behavior tests for the ERNIE pretrain TensorBoardCallback.

The production module imports ``paddlefleet.*`` (which loads ``paddle`` at
import time), so importing the callback requires a working paddle install.
The local dev environment has NO paddle; those runs skip with an honest
reason. All expected values below are derived by hand from the production
source, not copied from any other test. The SummaryWriter is treated as a
collaborator: a fake writer records the scalars/texts it receives so we can
assert exactly what content the callback emits.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import paddle  # noqa: F401

    from paddlefleet.cli.train.ernie_pretrain.src.callbacks import (
        tensorboard_callback as tb_module,
    )
    from paddlefleet.cli.train.ernie_pretrain.src.callbacks.tensorboard_callback import (
        TensorBoardCallback,
        is_tensorboard_available,
        rewrite_logs,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # honest probe: only missing-module triggers skip
    tb_module = None
    TensorBoardCallback = None
    is_tensorboard_available = None
    rewrite_logs = None
    _IMPORT_ERROR = repr(exc)

_SKIP_REASON = f"paddle/production import unavailable (local env has no paddle): {_IMPORT_ERROR}"


class _NumelStub:
    """Mimics the object returned by ``paddle.Tensor.numel()``."""

    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _ParamStub:
    """A stand-in for a model parameter consumed by the callback __init__."""

    def __init__(self, numel, stop_gradient):
        self._numel = numel
        self.stop_gradient = stop_gradient

    def numel(self):
        return _NumelStub(self._numel)


class _ModelStub:
    """A collaborator model that only needs named_parameters()."""

    def __init__(self, params):
        self._params = list(params)

    def named_parameters(self):
        return list(self._params)


class _FakeWriter:
    """A SummaryWriter collaborator that records everything it receives."""

    def __init__(self):
        self.scalars = []
        self.texts = []
        self.flush_calls = 0
        self.close_calls = 0

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def add_text(self, tag, text):
        self.texts.append((tag, text))

    def flush(self):
        self.flush_calls += 1

    def close(self):
        self.close_calls += 1


@unittest.skipIf(rewrite_logs is None, _SKIP_REASON)
class RewriteLogsBehaviorTest(unittest.TestCase):
    """rewrite_logs routes keys by their leading prefix and strips it."""

    def test_default_keys_get_train_prefix(self):
        self.assertEqual(
            rewrite_logs({"loss": 0.5, "lr": 0.1}),
            {"train/loss": 0.5, "train/lr": 0.1},
        )

    def test_eval_prefix_stripped(self):
        self.assertEqual(rewrite_logs({"eval_loss": 0.3}), {"eval/loss": 0.3})

    def test_test_prefix_stripped(self):
        self.assertEqual(rewrite_logs({"test_acc": 0.9}), {"test/acc": 0.9})

    def test_prefix_only_matched_at_start(self):
        # "my_eval_x" does not START with "eval_", so it is a train metric.
        self.assertEqual(
            rewrite_logs({"my_eval_x": 1.0}), {"train/my_eval_x": 1.0}
        )

    def test_bare_prefix_yields_empty_suffix(self):
        self.assertEqual(
            rewrite_logs({"eval_": 2.0, "test_": 3.0}),
            {"eval/": 2.0, "test/": 3.0},
        )

    def test_mixed_prefixes_all_routed(self):
        self.assertEqual(
            rewrite_logs({"loss": 0.5, "eval_loss": 0.3, "test_acc": 0.9}),
            {"train/loss": 0.5, "eval/loss": 0.3, "test/acc": 0.9},
        )

    def test_empty_maps_to_empty(self):
        self.assertEqual(rewrite_logs({}), {})


@unittest.skipIf(is_tensorboard_available is None, _SKIP_REASON)
class IsTensorboardAvailableBehaviorTest(unittest.TestCase):
    """is_tensorboard_available OR-combines two find_spec probes into a bool."""

    def _run_with_specs(self, present):
        def fake_find_spec(name, *args, **kwargs):
            return object() if name in present else None

        with patch("importlib.util.find_spec", side_effect=fake_find_spec):
            return is_tensorboard_available()

    def test_true_when_tensorboard_present(self):
        self.assertIs(self._run_with_specs({"tensorboard"}), True)

    def test_true_when_only_tensorboardx_present(self):
        self.assertIs(self._run_with_specs({"tensorboardX"}), True)

    def test_true_when_both_present(self):
        self.assertIs(
            self._run_with_specs({"tensorboard", "tensorboardX"}), True
        )

    def test_false_when_neither_present(self):
        self.assertIs(self._run_with_specs(set()), False)


@unittest.skipIf(TensorBoardCallback is None, _SKIP_REASON)
class TensorBoardCallbackOnLogBehaviorTest(unittest.TestCase):
    """on_log / on_train_begin emit exactly the scalars and text expected.

    The callback __init__ guards on is_tensorboard_available() and imports a
    SummaryWriter class. Neither is the behavior under test here, so the env
    probe is patched to True and an explicit fake writer is injected via the
    tb_writer argument; the real scalar-routing and numel-filtering logic is
    left intact and observed.
    """

    def _make_callback(self, model, writer, log_flops=False, log_tokens=False):
        with patch.object(
            tb_module, "is_tensorboard_available", return_value=True
        ):
            return TensorBoardCallback(
                SimpleNamespace(),
                model,
                tb_writer=writer,
                log_flops_per_step=log_flops,
                log_tokens_per_step=log_tokens,
            )

    def test_model_numel_excludes_frozen_and_embedding_params(self):
        # Hand-derived: only params with stop_gradient False AND whose name
        # contains neither "embeddings" nor "embed_tokens" are summed.
        params = [
            ("layer.0.weight", _ParamStub(100, False)),  # counted
            ("embeddings.weight", _ParamStub(50, False)),  # name excluded
            ("embed_tokens.weight", _ParamStub(40, False)),  # name excluded
            ("layer.1.bias", _ParamStub(10, True)),  # frozen, excluded
            ("layer.2.weight", _ParamStub(5, False)),  # counted
        ]
        cb = self._make_callback(_ModelStub(params), _FakeWriter())
        self.assertEqual(cb.model_numel, 105)

    def test_on_log_emits_prefixed_scalars_with_token_and_flop_axes(self):
        params = [
            ("layer.0.weight", _ParamStub(100, False)),
            ("embeddings.weight", _ParamStub(50, False)),
            ("embed_tokens.weight", _ParamStub(40, False)),
            ("layer.1.bias", _ParamStub(10, True)),
            ("layer.2.weight", _ParamStub(5, False)),
        ]
        writer = _FakeWriter()
        cb = self._make_callback(
            _ModelStub(params), writer, log_flops=True, log_tokens=True
        )
        self.assertEqual(cb.model_numel, 105)

        args = SimpleNamespace(
            train_batch_size=2,
            gradient_accumulation_steps=3,
            reeao_dataset_world_size=4,
            max_seq_len=8,
        )
        state = SimpleNamespace(is_world_process_zero=True, global_step=7)
        logs = {
            "loss": 0.5,
            "learning_rate": 0.001,
            "eval_loss": 0.25,
            "note": "hello",  # non-numeric -> dropped
        }
        cb.on_log(args, state, None, logs=logs)

        # Hand-derived constants:
        #   total_tokens_per_step = 2 * 3 * 4 * 8              = 192
        #   flops_per_step        = model_numel * tokens * 6   = 105*192*6 = 120960
        # xaxis scalars are emitted only for the "train/loss" key.
        self.assertEqual(
            writer.scalars,
            [
                ("train/loss", 0.5, 7),
                ("train/loss_xaxis_tokens", 0.5, 7 * 192),
                ("train/loss_xaxis_flops", 0.5, 7 * 120960),
                ("train/learning_rate", 0.001, 7),
                ("eval/loss", 0.25, 7),
            ],
        )
        self.assertEqual(writer.flush_calls, 1)
        # rewrite_logs builds a new dict; with no inputs the caller dict is intact.
        self.assertEqual(
            logs,
            {
                "loss": 0.5,
                "learning_rate": 0.001,
                "eval_loss": 0.25,
                "note": "hello",
            },
        )

    def test_on_log_without_flags_emits_only_base_scalars(self):
        writer = _FakeWriter()
        cb = self._make_callback(
            _ModelStub([("w", _ParamStub(3, False))]), writer
        )
        args = SimpleNamespace(
            train_batch_size=2,
            gradient_accumulation_steps=2,
            reeao_dataset_world_size=1,
            max_seq_len=4,
        )
        state = SimpleNamespace(is_world_process_zero=True, global_step=5)
        cb.on_log(args, state, None, logs={"loss": 0.75, "acc": 0.9})
        self.assertEqual(
            writer.scalars,
            [("train/loss", 0.75, 5), ("train/acc", 0.9, 5)],
        )
        self.assertEqual(writer.flush_calls, 1)

    def test_on_log_skips_non_zero_process(self):
        writer = _FakeWriter()
        cb = self._make_callback(
            _ModelStub([("w", _ParamStub(10, False))]), writer
        )
        args = SimpleNamespace(
            train_batch_size=1,
            gradient_accumulation_steps=1,
            reeao_dataset_world_size=1,
            max_seq_len=1,
        )
        state = SimpleNamespace(is_world_process_zero=False, global_step=3)
        cb.on_log(args, state, None, logs={"loss": 1.0})
        self.assertEqual(writer.scalars, [])
        self.assertEqual(writer.flush_calls, 0)

    def test_on_train_begin_writes_args_json_for_process_zero(self):
        writer = _FakeWriter()
        cb = self._make_callback(
            _ModelStub([("w", _ParamStub(1, False))]), writer
        )
        args = SimpleNamespace(to_json_string=lambda: "ARGS_JSON")
        state = SimpleNamespace(is_world_process_zero=True)
        cb.on_train_begin(args, state, None)
        self.assertEqual(writer.texts, [("args", "ARGS_JSON")])

    def test_on_train_begin_skips_non_zero_process(self):
        writer = _FakeWriter()
        cb = self._make_callback(
            _ModelStub([("w", _ParamStub(1, False))]), writer
        )
        args = SimpleNamespace(to_json_string=lambda: "ARGS_JSON")
        state = SimpleNamespace(is_world_process_zero=False)
        cb.on_train_begin(args, state, None)
        self.assertEqual(writer.texts, [])


if __name__ == "__main__":
    unittest.main()
