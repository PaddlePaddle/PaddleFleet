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

"""Behavior tests for the CPU-decidable helpers in ``paddlefleet.utils``
(defined in ``paddlefleet/utils/_fleet_utils.py`` and re-exported).

Scope. ``paddlefleet.utils`` bundles ~66 helpers; siblings cover the rest
concurrently. This file takes the THIRD slice and restricts itself to the
following helpers, chosen so the slice stays disjoint:

  * ensure_divisibility / divide         (pure integer arithmetic + guard)
  * init_method_normal                   (functools.partial configuration)
  * scaled_init_method_normal            (std = sigma / sqrt(multiplier*L))
  * prepare_input_tensors_for_wgrad_compute (contiguous + 3D->2D reshape)
  * deprecate_inference_params           (warn-and-return selection logic)
  * get_pg_size / get_pg_rank            (NOT-initialized short circuit only)

Every expected value below is hand-derived from the source arithmetic, not
read back from the function under test. No collective is faked; get_pg_size /
get_pg_rank are exercised ONLY on their ``paddle.distributed`` NOT-initialized
short circuit -- the real multi-rank ``group.nranks`` / ``group.rank`` branch
requires a genuine process group and belongs to a multi-card test (see
antipattern #13). The tests document this boundary explicitly.
"""

import math
import os
import sys
import unittest
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Honest capability probe. paddlefleet.utils imports paddle at module load.
# Only a genuine missing dependency (ImportError) may skip; any other error
# must surface as a real failure rather than a fake pass.
try:
    import paddle

    from paddlefleet.utils import (
        deprecate_inference_params,
        divide,
        ensure_divisibility,
        get_pg_rank,
        get_pg_size,
        init_method_normal,
        prepare_input_tensors_for_wgrad_compute,
        scaled_init_method_normal,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on environment
    paddle = None
    _IMPORT_ERROR = exc

_SKIP_MSG = (
    "paddlefleet.utils could not be imported "
    f"(missing dependency): {_IMPORT_ERROR}"
)


class _FakeGroup:
    """Plain stand-in for a distributed Group.

    Carries only the integer attributes the helpers read (``ranks``,
    ``nranks``, ``rank``). It is used EXCLUSIVELY on the not-initialized short
    circuit, where the helpers return before ever touching a collective, so
    this is not a mocked collective (cf. antipattern #13).
    """

    def __init__(self, ranks, rank):
        self.ranks = list(ranks)
        self.nranks = len(ranks)
        self.rank = rank


@unittest.skipUnless(paddle is not None, _SKIP_MSG)
class TestEnsureDivisibility(unittest.TestCase):
    """ensure_divisibility asserts numerator % denominator == 0."""

    def test_divisible_returns_none_without_raising(self):
        # 10 % 5 == 0; contract is to return normally (None), not raise.
        self.assertIsNone(ensure_divisibility(10, 5))

    def test_not_divisible_raises_with_informative_message(self):
        # 10 % 3 == 1 != 0 -> AssertionError carrying both operands.
        with self.assertRaises(AssertionError) as ctx:
            ensure_divisibility(10, 3)
        self.assertIn("10", str(ctx.exception))
        self.assertIn("3", str(ctx.exception))
        self.assertIn("not divisible", str(ctx.exception))

    def test_negative_numerator_still_divisible(self):
        # -10 % 5 == 0 in Python; guard must accept it without raising.
        self.assertIsNone(ensure_divisibility(-10, 5))


@unittest.skipUnless(paddle is not None, _SKIP_MSG)
class TestDivide(unittest.TestCase):
    """divide guards divisibility then returns the integer quotient."""

    def test_returns_exact_integer_quotient(self):
        # 20 // 4 == 5, 63 // 9 == 7; result must be the exact int, not float.
        result = divide(20, 4)
        self.assertEqual(result, 5)
        self.assertIsInstance(result, int)
        self.assertEqual(divide(63, 9), 7)

    def test_non_divisible_input_raises_via_guard(self):
        # divide delegates to ensure_divisibility, so 10 / 3 must raise.
        with self.assertRaises(AssertionError):
            divide(10, 3)


@unittest.skipUnless(paddle is not None, _SKIP_MSG)
class TestInitMethodNormal(unittest.TestCase):
    """init_method_normal binds paddle.nn.init.normal_ with mean=0, std=sigma."""

    def test_partial_binds_normal_with_hand_derived_keywords(self):
        sigma = 0.02
        method = init_method_normal(sigma)
        # The load-bearing contract is WHICH initializer is bound and with
        # WHAT keyword arguments, not merely that a callable exists.
        self.assertIs(method.func, paddle.nn.init.normal_)
        self.assertEqual(method.args, ())
        self.assertEqual(method.keywords, {"mean": 0.0, "std": 0.02})


@unittest.skipUnless(paddle is not None, _SKIP_MSG)
class TestScaledInitMethodNormal(unittest.TestCase):
    """scaled_init_method_normal computes std = sigma / sqrt(multiplier*L)."""

    def test_std_matches_hand_derived_default_multiplier(self):
        # sigma=0.1, num_layers=8, multiplier default 2.0:
        #   std = 0.1 / sqrt(2.0 * 8) = 0.1 / sqrt(16) = 0.1 / 4 = 0.025
        method = scaled_init_method_normal(0.1, 8)
        self.assertIs(method.func, paddle.nn.init.normal_)
        self.assertEqual(method.keywords["mean"], 0.0)
        expected = 0.1 / math.sqrt(2.0 * 8)
        self.assertAlmostEqual(method.keywords["std"], expected, places=12)
        self.assertAlmostEqual(method.keywords["std"], 0.025, places=12)

    def test_multiplier_changes_std(self):
        # Same sigma/num_layers, multiplier=8.0:
        #   std = 0.1 / sqrt(8.0 * 8) = 0.1 / sqrt(64) = 0.1 / 8 = 0.0125
        method = scaled_init_method_normal(0.1, 8, multiplier=8.0)
        expected = 0.1 / math.sqrt(8.0 * 8)
        self.assertAlmostEqual(method.keywords["std"], expected, places=12)
        self.assertAlmostEqual(method.keywords["std"], 0.0125, places=12)
        # And it must differ from the default-multiplier result, proving the
        # parameter is actually consumed in the denominator.
        default_std = scaled_init_method_normal(0.1, 8).keywords["std"]
        self.assertNotAlmostEqual(method.keywords["std"], default_std, places=6)


@unittest.skipUnless(paddle is not None, _SKIP_MSG)
class TestPrepareInputTensorsForWgradCompute(unittest.TestCase):
    """prepare_input_tensors_for_wgrad_compute keeps 2D, reshapes 3D to 2D."""

    def test_2d_inputs_preserve_content_and_shape(self):
        # 2D path must not reshape; content and layout unchanged.
        grad = paddle.arange(12, dtype="float32").reshape([3, 4])
        inp = (paddle.arange(12, dtype="float32") + 100.0).reshape([3, 4])
        g_out, i_out = prepare_input_tensors_for_wgrad_compute(grad, inp)
        self.assertEqual(g_out.shape, [3, 4])
        self.assertEqual(i_out.shape, [3, 4])
        # Distinct, non-overlapping content for grad vs input so a swap would
        # be caught; verify exact values, not merely shape.
        import numpy as np

        np.testing.assert_array_equal(
            g_out.numpy(), np.arange(12, dtype="float32").reshape(3, 4)
        )
        np.testing.assert_array_equal(
            i_out.numpy(),
            (np.arange(12, dtype="float32") + 100.0).reshape(3, 4),
        )

    def test_3d_inputs_reshaped_to_2d_preserving_row_major_content(self):
        # 3D [2,3,4] -> 2D [2*3, 4] = [6, 4], row-major. Hand-derive with
        # numpy reshape; content (not just [6,4] shape) must be preserved.
        import numpy as np

        grad_np = np.arange(24, dtype="float32").reshape(2, 3, 4)
        inp_np = (np.arange(24, dtype="float32") + 1000.0).reshape(2, 3, 4)
        grad = paddle.to_tensor(grad_np)
        inp = paddle.to_tensor(inp_np)
        g_out, i_out = prepare_input_tensors_for_wgrad_compute(grad, inp)
        self.assertEqual(g_out.shape, [6, 4])
        self.assertEqual(i_out.shape, [6, 4])
        np.testing.assert_array_equal(g_out.numpy(), grad_np.reshape(6, 4))
        np.testing.assert_array_equal(i_out.numpy(), inp_np.reshape(6, 4))


@unittest.skipUnless(paddle is not None, _SKIP_MSG)
class TestDeprecateInferenceParams(unittest.TestCase):
    """deprecate_inference_params selects context vs params and warns once."""

    def test_context_present_returns_context_without_warning(self):
        ctx = object()
        params = object()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = deprecate_inference_params(ctx, params)
        # context wins; params ignored; no deprecation warning issued.
        self.assertIs(result, ctx)
        self.assertEqual(len(caught), 0)

    def test_none_context_with_params_warns_and_returns_params(self):
        params = object()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = deprecate_inference_params(None, params)
        # Only this branch warns; it returns the legacy params object itself.
        self.assertIs(result, params)
        self.assertEqual(len(caught), 1)
        message = str(caught[0].message)
        self.assertIn("inference_params", message)
        self.assertIn("inference_context", message)

    def test_both_none_returns_none_without_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = deprecate_inference_params(None, None)
        self.assertIsNone(result)
        self.assertEqual(len(caught), 0)

    def test_context_present_params_none_returns_context(self):
        ctx = object()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = deprecate_inference_params(ctx, None)
        self.assertIs(result, ctx)
        self.assertEqual(len(caught), 0)


@unittest.skipUnless(paddle is not None, _SKIP_MSG)
class TestGetPgSizeRankUninitialized(unittest.TestCase):
    """get_pg_size / get_pg_rank short-circuit when distributed is NOT init.

    These tests validate ONLY the ``not paddle.distributed.is_initialized()``
    branch, which returns 1 / 0 regardless of the group argument. The real
    ``group.nranks`` / ``group.rank`` branches require a genuine multi-rank
    process group and are out of scope here (antipattern #13). If the test
    process happens to have distributed initialized, we skip rather than assert
    a value that would then be reached via a different branch.
    """

    def setUp(self):
        if paddle.distributed.is_initialized():
            self.skipTest(
                "paddle.distributed is initialized; the not-initialized "
                "short circuit under test is unreachable in this process"
            )

    def test_get_pg_size_returns_one_when_not_initialized(self):
        # group=None and a multi-rank fake group both return 1 solely because
        # distributed is not initialized (the first OR clause). A fake group
        # with 8 ranks would return 8 IF initialized, so this distinguishes
        # the short circuit from the group-is-None clause.
        self.assertEqual(get_pg_size(None), 1)
        multi = _FakeGroup(ranks=[0, 1, 2, 3, 4, 5, 6, 7], rank=3)
        self.assertEqual(get_pg_size(multi), 1)

    def test_get_pg_rank_returns_zero_when_not_initialized(self):
        # Likewise rank 0; the fake group carries rank=5 which would surface
        # IF initialized, confirming the short circuit takes precedence.
        self.assertEqual(get_pg_rank(None), 0)
        group = _FakeGroup(ranks=[0, 1, 2, 3, 4, 5], rank=5)
        self.assertEqual(get_pg_rank(group), 0)


if __name__ == "__main__":
    unittest.main()
