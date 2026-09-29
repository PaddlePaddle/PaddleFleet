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

"""Behavior tests for ernie_pretrain sequence_parallel_utils.

Module under test:
    paddlefleet.cli.train.ernie_pretrain.models.sequence_parallel_utils

Scope and oracle policy
-----------------------
These tests exercise only the parts of ``sequence_parallel_utils`` whose
behavior is fully determined in a single CPU process WITHOUT a real
hybrid-parallel collective:

* ``mark_as_sequence_parallel_parameter`` / ``is_sequence_parallel_parameter``
  -- the marking round-trip on a *real* paddle parameter. The mark must set
  the attribute to exactly ``True``; the predicate must return the stored
  attribute (defaulting to ``False`` when absent). Expected values are derived
  by hand from the parameter's attribute state, not from the functions under
  test.
* ``is_fused_matmul_bias_supported`` -- the CPU branch. A paddle build that is
  not compiled with CUDA cannot provide the CUDA-only ``fused_gemm_epilogue``
  epilogue, so support MUST be ``False``. This expectation is derived from the
  build capability, and the real paddle build is used (paddle is NOT mocked --
  mocking the dependency that decides the branch would verify nothing).
* ``MPScale`` -- the scale PyLayer. Forward divides by ``mp_degree``; backward
  passes the upstream gradient through unchanged (identity), NOT divided by
  ``mp_degree``. Both the forward output and the backward gradient are compared
  against hand-derived expectations.

``get_hcg`` is a thin delegation to ``fleet.get_hybrid_communicate_group()``;
faithfully checking it needs a real initialized hybrid-parallel group from a
distributed launch (multi-rank), so it is skipped with an honest reason rather
than faked (faking fleet to return a sentinel and asserting the sentinel
round-trips would be self-referential).

Paddle is an optional heavy dependency and the production module imports it at
top level, so every test skips (with an honest reason) when the import fails.
The local environment has no paddle installed.
"""

import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.cli.train.ernie_pretrain.models.sequence_parallel_utils import (
        MPScale,
        is_fused_matmul_bias_supported,
        is_sequence_parallel_parameter,
        mark_as_sequence_parallel_parameter,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle (or the module) unavailable in this env
    paddle = None
    MPScale = None
    is_fused_matmul_bias_supported = None
    is_sequence_parallel_parameter = None
    mark_as_sequence_parallel_parameter = None
    _IMPORT_ERROR = exc


def _skip_if_no_paddle(test_case):
    if _IMPORT_ERROR is not None:
        test_case.skipTest(
            f"paddle/sequence_parallel_utils import unavailable: {_IMPORT_ERROR!r}"
        )


class SequenceParallelParameterTest(unittest.TestCase):
    """mark_as_sequence_parallel_parameter / is_sequence_parallel_parameter."""

    def setUp(self):
        _skip_if_no_paddle(self)

    def _make_param(self):
        # A real paddle parameter is the faithful target of the marking API
        # (production marks real bias parameters). Not a mock.
        return paddle.create_parameter(shape=[2, 3], dtype="float32")

    def test_default_is_exactly_false(self):
        param = self._make_param()
        # A fresh parameter carries no sequence_parallel attribute, so the
        # predicate must fall back to exactly False (not merely falsy).
        self.assertFalse(hasattr(param, "sequence_parallel"))
        self.assertIs(is_sequence_parallel_parameter(param), False)

    def test_mark_sets_true_and_is_detected(self):
        param = self._make_param()
        mark_as_sequence_parallel_parameter(param)
        # The mark must set exactly True, and the predicate must report True.
        self.assertIs(param.sequence_parallel, True)
        self.assertIs(is_sequence_parallel_parameter(param), True)

    def test_explicit_false_reports_false(self):
        param = self._make_param()
        param.sequence_parallel = False
        self.assertIs(is_sequence_parallel_parameter(param), False)

    def test_predicate_returns_stored_value_verbatim(self):
        # The predicate is a plain attribute read: it forwards whatever is
        # stored rather than coercing to bool. A distinguishable sentinel makes
        # a bool-coercing reimplementation observable.
        param = self._make_param()
        param.sequence_parallel = "sp-tag"
        self.assertEqual(is_sequence_parallel_parameter(param), "sp-tag")


class FusedMatmulBiasSupportTest(unittest.TestCase):
    """is_fused_matmul_bias_supported: CPU branch and return-type contract."""

    def setUp(self):
        _skip_if_no_paddle(self)

    def test_cpu_build_reports_unsupported(self):
        # Hand-derived expectation tied to the build: without CUDA there is no
        # CUDA-only fused_gemm_epilogue, so support is False. Uses the real
        # paddle build (no mocking of is_compiled_with_cuda).
        if paddle.is_compiled_with_cuda():
            self.skipTest(
                "CPU-only paddle build required to assert the unsupported branch"
            )
        self.assertIs(is_fused_matmul_bias_supported(), False)

    def test_returns_strict_bool(self):
        # The function has several return points (False literals and a
        # hasattr(...)); on any build it must return a genuine bool, never None
        # or a truthy core proxy.
        result = is_fused_matmul_bias_supported()
        self.assertIsInstance(result, bool)


class MPScaleTest(unittest.TestCase):
    """MPScale PyLayer: forward scaling and identity backward."""

    def setUp(self):
        _skip_if_no_paddle(self)

    def test_forward_divides_by_mp_degree(self):
        x = paddle.to_tensor([2.0, 4.0, 6.0, 8.0], dtype="float32")
        out = MPScale.apply(x, 2)
        # Hand-derived: forward returns x * (1/mp_degree) == x / 2.
        np.testing.assert_allclose(
            out.numpy(),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            rtol=1e-6,
            atol=0.0,
        )

    def test_backward_is_identity_not_scaled(self):
        # The load-bearing contract: forward scales by 1/mp_degree, but backward
        # returns the upstream gradient UNCHANGED (identity), independent of
        # mp_degree. A naive symmetric implementation that also divided the
        # gradient by mp_degree would be rejected here.
        mp_degree = 4
        x = paddle.to_tensor([3.0, -6.0, 9.0], dtype="float32")
        x.stop_gradient = False
        out = MPScale.apply(x, mp_degree)

        upstream = paddle.to_tensor([1.0, 2.0, -0.5], dtype="float32")
        out.backward(upstream)

        self.assertIsNotNone(x.grad)
        # Identity backward => x.grad equals the upstream grad exactly.
        np.testing.assert_allclose(
            x.grad.numpy(), upstream.numpy(), rtol=1e-6, atol=0.0
        )
        # Explicitly NOT the mp_degree-scaled gradient.
        scaled = upstream.numpy() / mp_degree
        self.assertFalse(np.allclose(x.grad.numpy(), scaled))


class GetHcgTest(unittest.TestCase):
    """get_hcg: delegation that needs a real distributed group to verify."""

    def setUp(self):
        _skip_if_no_paddle(self)

    def test_requires_real_hybrid_parallel_group(self):
        # get_hcg() only forwards to fleet.get_hybrid_communicate_group().
        # A faithful check needs a real initialized hybrid-parallel group,
        # which exists only under a real distributed (multi-rank) launch.
        # Mocking fleet to return a sentinel and asserting the sentinel comes
        # back would verify nothing about production (self-referential), so we
        # skip honestly instead of faking a collective.
        self.skipTest(
            "get_hcg delegates to fleet.get_hybrid_communicate_group; "
            "verifying it requires a real distributed hybrid-parallel group "
            "(multi-card launch), unavailable in a single CPU process."
        )


if __name__ == "__main__":
    unittest.main()
