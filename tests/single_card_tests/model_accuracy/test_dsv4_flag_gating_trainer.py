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

"""``TransformerConfig.use_dsv4_accuracy`` must gate every DSV4 replay call site in ``trainer/``.

This companion to ``test_dsv4_accuracy_flag_gating.py`` (which pins the model /
kernel / dataset call sites, including the ``trainer_utils`` cosine-schedule
tail) covers the trainer-package call sites that can be isolated on a single
card without an initialized distributed process group:

* ``recompute_utils.install_recompute_p2p_overlap`` - the flag turns the
  ``RecomputeStore`` runtime-support guard on; with the flag off the guard is
  skipped entirely so the historical config path is unchanged.
* ``trainer_utils.init_optimizer`` - for the sharded (V2 / Muon 1D / Muon 2D)
  branches the flag makes the DSV4 replay ignore ``HACK_CONVERT_CKPT`` and
  always require the optimizer state to be present in the checkpoint metadata.
* ``trainer_callback.MoECorrectionBiasAdjustCallback.on_optimizer_end`` - the
  flag takes the frozen DSV4 path (a single all-reduce then return) instead of
  the historical fleet-group reduction that updates the correction bias.
* ``trainer.Trainer.training_pipeline_step`` - the flag installs the DSV4 loss
  accumulation-step scale (``set_loss_acc_steps``) at the top of the step.

Each test pins BOTH sides of the flag with an observable difference (a raised
error, a captured parameter list, a mutated tensor, or a wrapped call).

Several ``trainer.py`` call sites are intentionally NOT covered here because
they are only reachable from inside the full distributed training loop and
cannot be isolated on a single card:

* ``trainer.py:3408`` (``flush_sequence_first_wgrad``) - lives in the middle of
  ``_inner_training_loop`` after ``training_step`` and gradient accumulation;
  needs a real model/optimizer, an initialized ``hcg`` and grad-sync.
* ``trainer.py:3566`` (per-parameter grad rescale) - same loop, guarded by
  ``gradient_accumulation_steps > 1`` and iterating real model parameters with
  ``main_grad``; not constructible without the training loop.
* ``trainer.py:5981`` (``set_pipeline_loss_scale``) - only reached after the
  ``_pp_data_buffer`` fills and the pipeline dataset-preparation stage runs,
  which requires a real ``PipelineParallel`` model
  (``_prepare_pipeline_inputs_func`` / ``_prepare_training`` /
  ``forward_backward_pipeline``) and PP > 1.
"""

from __future__ import annotations

import os
import types
import unittest
from unittest.mock import patch

import paddle

from paddlefleet import accuracy_compatible_patch, recompute_utils
from paddlefleet.trainer import trainer, trainer_callback, trainer_utils


def _dsv4_flag(module, enabled):
    return patch.object(
        module, "use_dsv4_accuracy_compatible", return_value=enabled
    )


def _p2p_config():
    """A config that satisfies every downstream ``install_recompute_p2p_overlap``
    guard, so only the DSV4 ``RecomputeStore`` support check is left to toggle."""
    return types.SimpleNamespace(
        p2p_overlap_recompute=True,
        recompute_granularity="selective",
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
    )


class TestRecomputeP2POverlapGating(unittest.TestCase):
    """recompute_utils.py:43 - the flag adds the ``RecomputeStore`` support guard.

    With the flag on and no runtime ``RecomputeStore`` support the setup must
    fail loudly; with the flag off the exact same config must sail through the
    historical path (the guard never runs) and simply arm ``RecomputeStore``.
    """

    def setUp(self):
        self._saved_enabled = recompute_utils.RecomputeStore.enabled

    def tearDown(self):
        recompute_utils.RecomputeStore.enabled = self._saved_enabled

    def test_flag_on_requires_recompute_store_support(self):
        with (
            _dsv4_flag(recompute_utils, True),
            patch.object(recompute_utils, "HAS_RECOMPUTE_STORE", False),
            self.assertRaises(RuntimeError),
        ):
            recompute_utils.install_recompute_p2p_overlap(_p2p_config())

    def test_flag_off_skips_the_store_requirement(self):
        recompute_utils.RecomputeStore.enabled = False
        with (
            _dsv4_flag(recompute_utils, False),
            patch.object(recompute_utils, "HAS_RECOMPUTE_STORE", False),
        ):
            recompute_utils.install_recompute_p2p_overlap(_p2p_config())
        self.assertTrue(recompute_utils.RecomputeStore.enabled)

    def test_flag_on_with_store_support_passes(self):
        # Pins that the guard keys off the *store*, not the flag alone: with
        # support present the flag-on path no longer raises.
        recompute_utils.RecomputeStore.enabled = False
        with (
            _dsv4_flag(recompute_utils, True),
            patch.object(recompute_utils, "HAS_RECOMPUTE_STORE", True),
        ):
            recompute_utils.install_recompute_p2p_overlap(_p2p_config())
        self.assertTrue(recompute_utils.RecomputeStore.enabled)


class _FakeShardingV2:
    """Stand-in for ``DygraphShardingOptimizerV2`` (isinstance target only)."""


class _FakeMuon:
    """Stand-in for ``MuonShardingOptimizer`` (isinstance target only)."""


class _GradView:
    def __init__(self, param_buffer, begin, end):
        self._param_buffer = param_buffer
        self._param_begin = begin
        self._param_end = end


def _sharded_meta(param_name, struct_name):
    """One ``model_sharded_state_dict`` entry mapping a static name to a struct."""
    local_tensor = types.SimpleNamespace(name=param_name)
    return {struct_name: types.SimpleNamespace(local_tensor=local_tensor)}


def _capturing_optimizer(**attrs):
    captured = {}

    def _create_accumulators(block, parameter_list):
        captured["parameter_list"] = list(parameter_list)

    opt = types.SimpleNamespace(
        _create_accumulators=_create_accumulators, **attrs
    )
    return opt, captured


class TestInitOptimizerShardingV2Gating(unittest.TestCase):
    """trainer_utils.py:1720 - DygraphShardingOptimizerV2 ignores HACK_CONVERT_CKPT.

    With ``HACK_CONVERT_CKPT=1`` and no optimizer state in the metadata, the
    historical path (flag off) keeps the parameter, while the DSV4 replay (flag
    on) drops it because the state is required.
    """

    def _run(self, flag_enabled):
        grad_view = _GradView(paddle.arange(4, dtype="float32"), 0, 2)
        buffer = types.SimpleNamespace(
            _sharding_param_grad_view={"p0": grad_view}
        )
        opt, captured = _capturing_optimizer(
            _inner_opt=_FakeShardingV2(), _comm_buffer_list=[buffer]
        )
        meta = _sharded_meta("p0", "struct.p0")
        with (
            patch.object(
                trainer_utils, "DygraphShardingOptimizerV2", _FakeShardingV2
            ),
            _dsv4_flag(trainer_utils, flag_enabled),
            patch.object(
                trainer_utils, "has_optimizer_state", return_value=False
            ),
            patch.dict(os.environ, {"HACK_CONVERT_CKPT": "1"}),
        ):
            trainer_utils.init_optimizer(opt, meta, state_dict_metadata=set())
        return captured["parameter_list"]

    def test_flag_off_honours_hack_convert_ckpt(self):
        self.assertEqual(len(self._run(False)), 1)

    def test_flag_on_still_requires_the_optimizer_state(self):
        self.assertEqual(len(self._run(True)), 0)


class TestInitOptimizerMuon1DGating(unittest.TestCase):
    """trainer_utils.py:1773 - the Muon 1D branch ignores HACK_CONVERT_CKPT too."""

    def _run(self, flag_enabled):
        grad_view = _GradView(paddle.arange(4, dtype="float32"), 0, 2)
        buffer = types.SimpleNamespace(
            _sharding_param_grad_view={"p0": grad_view}
        )
        opt, captured = _capturing_optimizer(
            _inner_opt=_FakeMuon(),
            _comm_buffer_list=[buffer],
            _params_2d_by_color={},
        )
        meta = _sharded_meta("p0", "struct.p0")
        with (
            patch.object(trainer_utils, "MuonShardingOptimizer", _FakeMuon),
            _dsv4_flag(trainer_utils, flag_enabled),
            patch.object(
                trainer_utils, "has_optimizer_state", return_value=False
            ),
            patch.dict(os.environ, {"HACK_CONVERT_CKPT": "1"}),
        ):
            trainer_utils.init_optimizer(opt, meta, state_dict_metadata=set())
        return captured["parameter_list"]

    def test_flag_off_honours_hack_convert_ckpt(self):
        self.assertEqual(len(self._run(False)), 1)

    def test_flag_on_still_requires_the_optimizer_state(self):
        self.assertEqual(len(self._run(True)), 0)


class TestInitOptimizerMuon2DGating(unittest.TestCase):
    """trainer_utils.py:1823 - the Muon 2D-by-color branch ignores HACK_CONVERT_CKPT."""

    def _run(self, flag_enabled):
        color = "c0"
        param = types.SimpleNamespace(name="p2d")
        opt, captured = _capturing_optimizer(
            _inner_opt=_FakeMuon(),
            _comm_buffer_list=[],
            _params_2d_by_color={color: object()},
            _rank2params_2d_by_color={color: {0: [param]}},
            _color_to_group_info={color: {"rank": 0}},
        )
        meta = _sharded_meta("p2d", "struct.p2d")
        with (
            patch.object(trainer_utils, "MuonShardingOptimizer", _FakeMuon),
            _dsv4_flag(trainer_utils, flag_enabled),
            patch.object(
                trainer_utils, "has_optimizer_state", return_value=False
            ),
            patch.dict(os.environ, {"HACK_CONVERT_CKPT": "1"}),
        ):
            trainer_utils.init_optimizer(opt, meta, state_dict_metadata=set())
        return captured["parameter_list"]

    def test_flag_off_honours_hack_convert_ckpt(self):
        self.assertEqual(len(self._run(False)), 1)

    def test_flag_on_still_requires_the_optimizer_state(self):
        self.assertEqual(len(self._run(True)), 0)


class _FakeGate:
    """A ``noaux_tc`` gate that the callback should pick up via ``model.apply``."""

    def __init__(self):
        self.topk_method = "noaux_tc"
        self.e_score_correction_bias = paddle.to_tensor(
            [1.0, 2.0], dtype="float32"
        )
        self.expert_usage = paddle.to_tensor([3.0, 1.0], dtype="float32")
        self.weight = types.SimpleNamespace(stop_gradient=False)


class _FakeModel:
    def __init__(self, layers):
        self._layers = layers

    def apply(self, fn):
        for layer in self._layers:
            fn(layer)


class TestMoECorrectionBiasCallbackGating(unittest.TestCase):
    """trainer_callback.py:1065 - the flag selects the frozen DSV4 callback path.

    Flag on: a single ``dist.all_reduce`` on the stacked usage then an early
    return - the correction bias and usage counters are left untouched, and the
    fleet hybrid-comm group is never consulted. Flag off (single-card groups):
    the historical path updates the bias and zeros the usage.
    """

    def _run(self, flag_enabled):
        gate = _FakeGate()
        model = _FakeModel([gate])
        callback = trainer_callback.MoECorrectionBiasAdjustCallback(
            lr=0.1, use_mp=False
        )
        args = types.SimpleNamespace(freeze_training=False)
        bias_before = gate.e_score_correction_bias.numpy().copy()
        single_card_group = types.SimpleNamespace(nranks=1)
        hcg = types.SimpleNamespace(
            get_model_parallel_group=lambda: single_card_group,
            get_data_parallel_group=lambda: single_card_group,
            get_sharding_parallel_group=lambda: single_card_group,
        )
        with (
            _dsv4_flag(trainer_callback, flag_enabled),
            patch.object(trainer_callback, "PretrainedMoEGate", _FakeGate),
            patch.object(trainer_callback.dist, "all_reduce") as all_reduce,
            patch.object(trainer_callback, "fleet") as fleet,
            patch.object(
                trainer_callback, "get_lr_ratio_fn", return_value=None
            ),
        ):
            if flag_enabled:
                fleet.get_hybrid_communicate_group.side_effect = AssertionError(
                    "fleet must not be consulted on DSV4 path"
                )
            else:
                fleet._hcg = object()
                fleet.get_hybrid_communicate_group.return_value = hcg
            callback.on_optimizer_end(
                args, None, None, model=model, optimizer=None
            )
        bias_after = gate.e_score_correction_bias.numpy()
        return (
            all_reduce.call_count,
            bias_before,
            bias_after,
            gate.expert_usage.numpy(),
        )

    def test_flag_on_all_reduces_then_returns_without_updating(self):
        calls, before, after, usage = self._run(True)
        self.assertEqual(calls, 1)
        self.assertTrue((before == after).all())  # bias untouched
        self.assertTrue((usage == [3.0, 1.0]).all())  # usage not zeroed

    def test_flag_off_updates_the_correction_bias(self):
        calls, before, after, usage = self._run(False)
        self.assertEqual(calls, 0)  # single-card groups -> no all-reduce
        self.assertFalse((before == after).all())  # bias updated
        self.assertTrue((usage == [0.0, 0.0]).all())  # usage zeroed


class TestPipelineStepLossAccStepsGating(unittest.TestCase):
    """trainer.py:5914 - the flag installs the DSV4 loss accumulation-step scale.

    ``training_pipeline_step`` returns early while the ``_pp_data_buffer`` is
    still filling (``gradient_accumulation_steps=2``, first micro-step), which
    isolates the top-of-step flag check from the pipeline forward/backward.
    """

    def _run(self, flag_enabled):
        fake_self = types.SimpleNamespace(
            args=types.SimpleNamespace(gradient_accumulation_steps=2)
        )
        with (
            _dsv4_flag(trainer, flag_enabled),
            patch.object(
                accuracy_compatible_patch, "set_loss_acc_steps"
            ) as set_loss_acc_steps,
        ):
            trainer.Trainer.training_pipeline_step(
                fake_self,
                model=None,
                inputs={"input_ids": 0},
                data_buffer_prepared=False,
            )
        return set_loss_acc_steps

    def test_flag_on_sets_the_loss_acc_steps(self):
        set_loss_acc_steps = self._run(True)
        set_loss_acc_steps.assert_called_once_with(2)

    def test_flag_off_leaves_the_loss_acc_steps_alone(self):
        set_loss_acc_steps = self._run(False)
        set_loss_acc_steps.assert_not_called()


class TestInitOptimizerShardingV2HackOffGating(unittest.TestCase):
    """trainer_utils.py:1731 - the flag-off DygraphShardingOptimizerV2 branch.

    ``TestInitOptimizerShardingV2Gating`` pins the flag-off path with
    ``HACK_CONVERT_CKPT=1`` (the ``elif`` guard is False so line 1731 never
    runs and the param is kept). With the hack disabled the guard falls through
    to the ``has_optimizer_state`` check at line 1731: a missing state drops the
    param, a present state keeps it.
    """

    def _run(self, has_state):
        grad_view = _GradView(paddle.arange(4, dtype="float32"), 0, 2)
        buffer = types.SimpleNamespace(
            _sharding_param_grad_view={"p0": grad_view}
        )
        opt, captured = _capturing_optimizer(
            _inner_opt=_FakeShardingV2(), _comm_buffer_list=[buffer]
        )
        meta = _sharded_meta("p0", "struct.p0")
        with (
            patch.object(
                trainer_utils, "DygraphShardingOptimizerV2", _FakeShardingV2
            ),
            _dsv4_flag(trainer_utils, False),
            patch.object(
                trainer_utils, "has_optimizer_state", return_value=has_state
            ),
            patch.dict(os.environ, {"HACK_CONVERT_CKPT": "0"}),
        ):
            trainer_utils.init_optimizer(opt, meta, state_dict_metadata=set())
        return captured["parameter_list"]

    def test_missing_state_drops_the_param(self):
        self.assertEqual(len(self._run(False)), 0)

    def test_present_state_keeps_the_param(self):
        self.assertEqual(len(self._run(True)), 1)


class TestInitOptimizerMuon1DHackOffGating(unittest.TestCase):
    """trainer_utils.py:1784 - the flag-off Muon 1D branch honours the missing
    optimizer state when HACK_CONVERT_CKPT is off (line 1784)."""

    def _run(self, has_state):
        grad_view = _GradView(paddle.arange(4, dtype="float32"), 0, 2)
        buffer = types.SimpleNamespace(
            _sharding_param_grad_view={"p0": grad_view}
        )
        opt, captured = _capturing_optimizer(
            _inner_opt=_FakeMuon(),
            _comm_buffer_list=[buffer],
            _params_2d_by_color={},
        )
        meta = _sharded_meta("p0", "struct.p0")
        with (
            patch.object(trainer_utils, "MuonShardingOptimizer", _FakeMuon),
            _dsv4_flag(trainer_utils, False),
            patch.object(
                trainer_utils, "has_optimizer_state", return_value=has_state
            ),
            patch.dict(os.environ, {"HACK_CONVERT_CKPT": "0"}),
        ):
            trainer_utils.init_optimizer(opt, meta, state_dict_metadata=set())
        return captured["parameter_list"]

    def test_missing_state_drops_the_param(self):
        self.assertEqual(len(self._run(False)), 0)

    def test_present_state_keeps_the_param(self):
        self.assertEqual(len(self._run(True)), 1)


class TestInitOptimizerMuon2DHackOffGating(unittest.TestCase):
    """trainer_utils.py:1834 - the flag-off Muon 2D-by-color branch honours the
    missing optimizer state when HACK_CONVERT_CKPT is off (line 1834)."""

    def _run(self, has_state):
        color = "c0"
        param = types.SimpleNamespace(name="p2d")
        opt, captured = _capturing_optimizer(
            _inner_opt=_FakeMuon(),
            _comm_buffer_list=[],
            _params_2d_by_color={color: object()},
            _rank2params_2d_by_color={color: {0: [param]}},
            _color_to_group_info={color: {"rank": 0}},
        )
        meta = _sharded_meta("p2d", "struct.p2d")
        with (
            patch.object(trainer_utils, "MuonShardingOptimizer", _FakeMuon),
            _dsv4_flag(trainer_utils, False),
            patch.object(
                trainer_utils, "has_optimizer_state", return_value=has_state
            ),
            patch.dict(os.environ, {"HACK_CONVERT_CKPT": "0"}),
        ):
            trainer_utils.init_optimizer(opt, meta, state_dict_metadata=set())
        return captured["parameter_list"]

    def test_missing_state_drops_the_param(self):
        self.assertEqual(len(self._run(False)), 0)

    def test_present_state_keeps_the_param(self):
        self.assertEqual(len(self._run(True)), 1)


if __name__ == "__main__":
    unittest.main()
