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

"""Behavior tests for ``paddlefleet.trainer.plugins.npu_plugin``.

The NPU accelerate plugin rebinds an optimizer's ``step`` to a variant that
flattens parameter gradients before applying the update. Two layers of behavior
are CPU-observable and are exercised here with the real production functions:

* ``npu_accelerate_plugin`` must rebind ``step`` so that later calls route into
  ``_optimizer_step_with_flatten_param_grads`` bound to that optimizer instance.
* ``_optimizer_step_with_flatten_param_grads`` must reject dict-style param
  groups, filter out ``stop_gradient`` params and params without a gradient,
  gate the flatten step on ``regularization``/``_grad_clip``, and forward the
  surviving ``(param, grad)`` pairs to ``_apply_optimize``.
* ``_flatten_param_grads`` must abort (returning the original pairs unchanged)
  when a param has ``need_clip is False`` or a ``regularizer``.

The final ``coalesce_tensor`` numerics require NPU/GPU hardware and are NOT
asserted here; that path is deliberately not driven. When paddle is not
installed (no accelerator CI host) every case skips with an explicit reason
instead of pretending to pass.
"""

import unittest
from unittest import mock

try:
    from paddlefleet.trainer.plugins import npu_plugin as _npu_plugin_mod
    from paddlefleet.trainer.plugins.npu_plugin import (
        _flatten_param_grads,
        _optimizer_step_with_flatten_param_grads,
        npu_accelerate_plugin,
    )

    _IMPORT_ERROR = None
except (
    ImportError
) as exc:  # paddle/paddlefleet absent (e.g. CPU-only, no NPU/GPU)
    _npu_plugin_mod = None
    _flatten_param_grads = None
    _optimizer_step_with_flatten_param_grads = None
    npu_accelerate_plugin = None
    _IMPORT_ERROR = exc

_HAS_NPU_PLUGIN = _IMPORT_ERROR is None
_SKIP_REASON = (
    "paddlefleet.trainer.plugins.npu_plugin could not be imported "
    f"({_IMPORT_ERROR!r}); paddle is not installed in this environment. "
    "These CPU-runnable behavior tests are skipped rather than faked."
)

_UNSET = object()  # sentinel: attribute intentionally left unset on _FakeParam


class _FakeGrad:
    """Stand-in for a gradient tensor: carries identity and a persistable flag.

    Only the attributes touched by the production code (``persistable``) are
    modelled; identity is asserted so we can prove the exact objects flow
    through the orchestration.
    """

    def __init__(self, name):
        self.name = name
        self.persistable = False


class _FakeParam:
    """Stand-in for an optimizer parameter.

    Exposes only what ``_optimizer_step_with_flatten_param_grads`` and
    ``_flatten_param_grads`` read from a parameter: ``stop_gradient``,
    ``_grad_ivar()``, and the optional ``need_clip``/``regularizer`` attributes
    consulted via ``getattr``. ``need_clip``/``regularizer`` are left unset by
    default so the production ``getattr`` defaults (``True``/``None``) apply.
    """

    def __init__(
        self,
        name,
        stop_gradient=False,
        grad=None,
        need_clip=_UNSET,
        regularizer=_UNSET,
    ):
        self.name = name
        self.stop_gradient = stop_gradient
        self._grad = grad
        if need_clip is not _UNSET:
            self.need_clip = need_clip
        if regularizer is not _UNSET:
            self.regularizer = regularizer

    def _grad_ivar(self):
        return self._grad


class _RecordingOptimizer:
    """Collaborator optimizer that records the ``_apply_optimize`` contract.

    ``_apply_optimize`` is a collaborator of the function under test (not the
    function under test itself), so recording its exact call arguments is a
    faithful way to observe what the orchestration forwards.
    """

    def __init__(self, param_groups, regularization=None, grad_clip=None):
        self._param_groups = param_groups
        self.regularization = regularization
        self._grad_clip = grad_clip
        self.helper = None
        self.apply_calls = []

    def step(self):
        # Sentinel so a test can prove the plugin actually replaced this method.
        return "ORIGINAL_STEP"

    def _apply_optimize(
        self, *, loss, startup_program, params_grads, param_group_idx
    ):
        self.apply_calls.append(
            {
                "loss": loss,
                "startup_program": startup_program,
                "params_grads": params_grads,
                "param_group_idx": param_group_idx,
            }
        )
        return "APPLIED"


@unittest.skipUnless(_HAS_NPU_PLUGIN, _SKIP_REASON)
class TestNpuAcceleratePluginBinding(unittest.TestCase):
    """npu_accelerate_plugin rebinds step to the flatten variant."""

    def test_rebinds_step_to_flatten_function_on_this_optimizer(self):
        opt = _RecordingOptimizer(param_groups=[_FakeParam("p0")])
        original_step = opt.step
        self.assertEqual(original_step(), "ORIGINAL_STEP")

        npu_accelerate_plugin(opt)

        # Identity contract: step is now the production flatten function, bound
        # to this exact optimizer instance -- not the original bound method.
        self.assertIsNot(opt.step, original_step)
        self.assertIs(
            opt.step.__func__, _optimizer_step_with_flatten_param_grads
        )
        self.assertIs(opt.step.__self__, opt)

    def test_rebound_step_actually_routes_into_flatten_logic(self):
        # Behavioral proof (not just __func__ inspection): after the plugin,
        # calling step() on a dict-param-group optimizer must hit the flatten
        # variant, which rejects dict groups. The original step would not raise.
        opt = _RecordingOptimizer(param_groups=[{"params": []}])
        self.assertEqual(opt.step(), "ORIGINAL_STEP")  # pre-condition

        npu_accelerate_plugin(opt)

        with self.assertRaises(RuntimeError):
            opt.step()
        self.assertEqual(opt.apply_calls, [])  # never reached _apply_optimize


@unittest.skipUnless(_HAS_NPU_PLUGIN, _SKIP_REASON)
class TestFlattenStepOrchestration(unittest.TestCase):
    """_optimizer_step_with_flatten_param_grads filtering and forwarding."""

    def test_dict_param_groups_raise_with_clear_message(self):
        opt = _RecordingOptimizer(param_groups=[{"params": []}])
        with self.assertRaises(RuntimeError) as ctx:
            _optimizer_step_with_flatten_param_grads(opt)
        self.assertIn("not supported", str(ctx.exception))
        self.assertIn("dict", str(ctx.exception))
        self.assertEqual(opt.apply_calls, [])

    def test_filters_frozen_and_gradless_params_and_forwards_survivors(self):
        # regularization is set (not None), so the flatten block is skipped and
        # the surviving pairs must reach _apply_optimize verbatim. This targets
        # the filter + forward contract, not the flatten numerics.
        g1, g2 = _FakeGrad("g1"), _FakeGrad("g2")
        p_frozen = _FakeParam(
            "frozen", stop_gradient=True, grad=_FakeGrad("gx")
        )
        p_nograd = _FakeParam("nograd", stop_gradient=False, grad=None)
        p_ok1 = _FakeParam("ok1", stop_gradient=False, grad=g1)
        p_ok2 = _FakeParam("ok2", stop_gradient=False, grad=g2)

        opt = _RecordingOptimizer(
            param_groups=[p_frozen, p_nograd, p_ok1, p_ok2],
            regularization=object(),  # non-None -> flatten branch skipped
        )

        result = _optimizer_step_with_flatten_param_grads(opt)

        self.assertEqual(result, "APPLIED")
        self.assertEqual(len(opt.apply_calls), 1)
        call = opt.apply_calls[0]
        self.assertIsNone(call["loss"])
        self.assertIsNone(call["startup_program"])
        self.assertEqual(call["param_group_idx"], 0)

        forwarded = call["params_grads"]
        # Exactly the two survivors, in order, with identity preserved and the
        # correct param<->grad pairing (frozen + grad-less excluded).
        self.assertEqual(len(forwarded), 2)
        self.assertIs(forwarded[0][0], p_ok1)
        self.assertIs(forwarded[0][1], g1)
        self.assertIs(forwarded[1][0], p_ok2)
        self.assertIs(forwarded[1][1], g2)

    def test_non_globalnorm_grad_clip_skips_flatten(self):
        # regularization is None but _grad_clip is neither None nor
        # ClipGradByGlobalNorm, so the flatten step must be skipped and the raw
        # (unflattened) pairs forwarded. If flatten wrongly ran, it would build
        # brand-new coalesced tensors (or fail on these stubs) -- so asserting
        # the original objects flow through proves the gate held.
        g1 = _FakeGrad("g1")
        p_ok1 = _FakeParam("ok1", stop_gradient=False, grad=g1)

        class _NotGlobalNorm:
            pass

        opt = _RecordingOptimizer(
            param_groups=[p_ok1],
            regularization=None,
            grad_clip=_NotGlobalNorm(),
        )

        _optimizer_step_with_flatten_param_grads(opt)

        forwarded = opt.apply_calls[0]["params_grads"]
        self.assertEqual(len(forwarded), 1)
        self.assertIs(forwarded[0][0], p_ok1)
        self.assertIs(forwarded[0][1], g1)

    @unittest.expectedFailure
    def test_all_params_frozen_should_forward_empty_but_currently_crashes(self):
        # Robustness gap: when every param is filtered out (all frozen), the
        # orchestration still calls _flatten_param_grads([]), which indexes
        # need_flatten_params[0] on an empty list and raises IndexError instead
        # of forwarding an empty pair list to _apply_optimize. Documented as an
        # expected failure; no production edit is made.
        opt = _RecordingOptimizer(
            param_groups=[_FakeParam("frozen", stop_gradient=True)],
            regularization=None,
            grad_clip=None,
        )
        _optimizer_step_with_flatten_param_grads(opt)
        self.assertEqual(opt.apply_calls[0]["params_grads"], [])


@unittest.skipUnless(_HAS_NPU_PLUGIN, _SKIP_REASON)
class TestFlattenParamGradsEarlyReturn(unittest.TestCase):
    """_flatten_param_grads aborts when a param opts out of clipping."""

    def test_returns_original_pairs_when_need_clip_false(self):
        opt = _RecordingOptimizer(param_groups=[])
        g = _FakeGrad("g")
        p = _FakeParam("p", grad=g, need_clip=False)
        params_grads = [(p, g)]

        with mock.patch.object(_npu_plugin_mod, "logger") as logger:
            result = _flatten_param_grads(opt, params_grads)

        # Same list object returned unchanged; the coalesce path is not reached.
        self.assertIs(result, params_grads)
        # Side effect that happens before the early return is still observable.
        self.assertTrue(g.persistable)
        logger.warning.assert_called_once()

    def test_returns_original_pairs_when_regularizer_set(self):
        opt = _RecordingOptimizer(param_groups=[])
        g = _FakeGrad("g")
        p = _FakeParam("p", grad=g, regularizer=object())
        params_grads = [(p, g)]

        with mock.patch.object(_npu_plugin_mod, "logger"):
            result = _flatten_param_grads(opt, params_grads)

        self.assertIs(result, params_grads)
        self.assertTrue(g.persistable)

    def test_processes_prefix_then_aborts_on_later_bad_param(self):
        # First param is clip-eligible (gets marked persistable and queued),
        # second opts out -> the whole original list is returned and no coalesce
        # occurs. Both grads still get persistable=True (set before the check).
        opt = _RecordingOptimizer(param_groups=[])
        g_good, g_bad = _FakeGrad("good"), _FakeGrad("bad")
        p_good = _FakeParam("good", grad=g_good)  # need_clip defaults True
        p_bad = _FakeParam("bad", grad=g_bad, need_clip=False)
        params_grads = [(p_good, g_good), (p_bad, g_bad)]

        with mock.patch.object(_npu_plugin_mod, "logger"):
            result = _flatten_param_grads(opt, params_grads)

        self.assertIs(result, params_grads)
        self.assertTrue(g_good.persistable)
        self.assertTrue(g_bad.persistable)


if __name__ == "__main__":
    unittest.main()
