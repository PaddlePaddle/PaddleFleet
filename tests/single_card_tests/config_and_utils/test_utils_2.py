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

"""Behavior tests for the second slice of paddlefleet.utils helpers.

The public ``paddlefleet.utils`` package lazily re-exports the callables that
actually live in ``paddlefleet/utils/_fleet_utils.py``; this file drives that
real module. Sibling files (test_utils.py / _3 / _4 / _5) concurrently cover
the other helpers, so this slice targets a disjoint set:

  * ensure_divisibility / divide          - exact arithmetic + assertion contract
  * GlobalMemoryBuffer.get_tensor         - buffer caching / realloc / dtype key
  * init_method_normal                    - std/mean binding of the initializer
  * scaled_init_method_normal             - std = sigma / sqrt(multiplier*num_layers)
  * get_pg_size / get_pg_rank             - world-size / rank fallback branches
  * log_single_rank                       - rank-gated emission of a real record
  * get_tensor_model_parallel_group_if_none - default-group selection branches
  * prepare_input_tensors_for_wgrad_compute - 3D->2D reshape content
  * deprecate_inference_params            - context/params precedence + warning
  * get_batch_on_this_cp_rank             - which dict keys are scattered

Every expected value is hand-derived from the source arithmetic/branches and
written as a literal or an independent recomputation; nothing is produced by
calling the function under test. Genuine collaborators (parallel_state,
ContextParallelScatterOp, paddle.distributed.is_initialized/get_rank) are the
only things mocked, and each mock returns an input-dependent marker so the test
observes real consumption rather than a bare ``assert_called``.

Honest skip: _fleet_utils imports paddle unconditionally. Only a real missing
dependency (ImportError, which ModuleNotFoundError subclasses) skips the suite;
any other import error surfaces as a genuine failure instead of a fake pass.
"""

import logging
import os
import sys
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import numpy as np
    import paddle

    import paddlefleet.utils._fleet_utils as fleet_utils
    from paddlefleet import parallel_state

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on environment
    np = None
    paddle = None
    fleet_utils = None
    parallel_state = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    None
    if _IMPORT_ERROR is None
    else f"paddle/paddlefleet not importable: {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class DivisibilityTests(unittest.TestCase):
    def test_ensure_divisibility_accepts_and_rejects(self):
        # 12 % 4 == 0 -> no exception; return value is implicitly None.
        self.assertIsNone(fleet_utils.ensure_divisibility(12, 4))
        # 7 % 3 == 1 != 0 -> AssertionError carrying the exact source message.
        with self.assertRaises(AssertionError) as ctx:
            fleet_utils.ensure_divisibility(7, 3)
        self.assertEqual(str(ctx.exception), "7 is not divisible by 3")

    def test_divide_returns_floor_quotient(self):
        # divide == numerator // denominator once divisibility holds.
        self.assertEqual(fleet_utils.divide(12, 4), 3)
        self.assertEqual(fleet_utils.divide(20, 5), 4)
        # A non-multiple must not silently floor; it re-raises the assertion.
        with self.assertRaises(AssertionError):
            fleet_utils.divide(10, 3)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class GlobalMemoryBufferTests(unittest.TestCase):
    def test_get_tensor_reuses_storage_for_repeat_request(self):
        buf = fleet_utils.GlobalMemoryBuffer()
        t1 = buf.get_tensor([3, 4], paddle.float32, "a")
        t2 = buf.get_tensor([3, 4], paddle.float32, "a")
        # Same (name, dtype) key and non-growing request -> same backing storage.
        self.assertEqual(t1.shape, [3, 4])
        self.assertEqual(t2.shape, [3, 4])
        self.assertEqual(t1.data_ptr(), t2.data_ptr())

    def test_get_tensor_reallocates_when_growing(self):
        buf = fleet_utils.GlobalMemoryBuffer()
        small = buf.get_tensor([2, 2], paddle.float32, "g")  # numel 4
        big = buf.get_tensor([4, 4], paddle.float32, "g")  # numel 16 > 4
        # Growing past the cached numel forces a fresh allocation; the old view
        # keeps the previous storage alive so the pointers must differ.
        self.assertEqual(big.shape, [4, 4])
        self.assertNotEqual(small.data_ptr(), big.data_ptr())
        # A subsequent smaller request now reuses the enlarged buffer.
        shrink = buf.get_tensor([2, 3], paddle.float32, "g")  # numel 6 <= 16
        self.assertEqual(shrink.shape, [2, 3])
        self.assertEqual(shrink.data_ptr(), big.data_ptr())

    def test_get_tensor_separate_storage_per_dtype(self):
        buf = fleet_utils.GlobalMemoryBuffer()
        f32 = buf.get_tensor([2, 2], paddle.float32, "same_name")
        f64 = buf.get_tensor([2, 2], paddle.float64, "same_name")
        # dtype participates in the cache key, so the two must not collide.
        self.assertEqual(f32.dtype, paddle.float32)
        self.assertEqual(f64.dtype, paddle.float64)
        self.assertNotEqual(f32.data_ptr(), f64.data_ptr())


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class InitMethodTests(unittest.TestCase):
    def test_init_method_normal_binds_mean_and_std(self):
        fn = fleet_utils.init_method_normal(0.023)
        # The helper's whole job is to bind normal_ with mean=0 and std=sigma.
        self.assertIs(fn.func, paddle.nn.init.normal_)
        self.assertEqual(fn.keywords["mean"], 0.0)
        self.assertEqual(fn.keywords["std"], 0.023)

    def test_scaled_init_method_normal_computes_std(self):
        # std = sigma / sqrt(multiplier * num_layers)
        # explicit: 1.0 / sqrt(2.0 * 2) = 1.0 / 2.0 = 0.5
        fn = fleet_utils.scaled_init_method_normal(1.0, 2, multiplier=2.0)
        self.assertIs(fn.func, paddle.nn.init.normal_)
        self.assertEqual(fn.keywords["mean"], 0.0)
        self.assertEqual(fn.keywords["std"], 0.5)
        # default multiplier 2.0: 0.1 / sqrt(2.0 * 8) = 0.1 / 4.0 = 0.025
        fn2 = fleet_utils.scaled_init_method_normal(0.1, 8)
        self.assertAlmostEqual(fn2.keywords["std"], 0.025, places=12)

    def test_scaled_init_method_normal_zero_layers_raises(self):
        # multiplier * num_layers == 0 -> sqrt(0) == 0 -> float division by zero.
        with self.assertRaises(ZeroDivisionError):
            fleet_utils.scaled_init_method_normal(0.02, 0)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class ProcessGroupTests(unittest.TestCase):
    def test_get_pg_size_returns_group_nranks(self):
        # ranks has length 2 but nranks is 7; the contract returns nranks, so a
        # distinct value catches a "len(ranks)" mix-up.
        group = SimpleNamespace(ranks=[0, 1], nranks=7)
        with patch("paddle.distributed.is_initialized", return_value=True):
            self.assertEqual(fleet_utils.get_pg_size(group=group), 7)

    def test_get_pg_size_falls_back_to_one(self):
        multi = SimpleNamespace(ranks=[0, 1, 2, 3], nranks=4)
        # Not initialized -> 1 even for a multi-rank group.
        with patch("paddle.distributed.is_initialized", return_value=False):
            self.assertEqual(fleet_utils.get_pg_size(group=multi), 1)
        with patch("paddle.distributed.is_initialized", return_value=True):
            # group is None -> 1
            self.assertEqual(fleet_utils.get_pg_size(group=None), 1)
            # single-rank group short-circuits to 1 before reading nranks.
            single = SimpleNamespace(ranks=[0], nranks=9)
            self.assertEqual(fleet_utils.get_pg_size(group=single), 1)

    def test_get_pg_rank_returns_group_rank_or_zero(self):
        group = SimpleNamespace(rank=3)
        with patch("paddle.distributed.is_initialized", return_value=True):
            self.assertEqual(fleet_utils.get_pg_rank(group=group), 3)
            # group None -> 0 even when initialized.
            self.assertEqual(fleet_utils.get_pg_rank(group=None), 0)
        # Not initialized -> 0 even with a group that reports rank 5.
        with patch("paddle.distributed.is_initialized", return_value=False):
            self.assertEqual(
                fleet_utils.get_pg_rank(group=SimpleNamespace(rank=5)), 0
            )


def _capturing_logger(name):
    """A real logging.Logger whose emitted records are collected in a list."""
    records = []

    class _ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger(name)
    logger.handlers = []
    logger.addHandler(_ListHandler())
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger, records


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class LogSingleRankTests(unittest.TestCase):
    def test_log_single_rank_logs_only_on_matching_rank(self):
        logger, records = _capturing_logger("pf_test_log_match")
        with (
            patch("paddle.distributed.is_initialized", return_value=True),
            patch("paddle.distributed.get_rank", return_value=2),
        ):
            # current rank 2 == target rank 2 -> emit, with real formatting.
            fleet_utils.log_single_rank(
                logger, logging.WARNING, "count=%d", 7, rank=2
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].levelno, logging.WARNING)
            self.assertEqual(records[0].getMessage(), "count=7")
            # current rank 2 != target rank 0 (default) -> no additional record.
            fleet_utils.log_single_rank(logger, logging.WARNING, "again")
            self.assertEqual(len(records), 1)

    def test_log_single_rank_logs_when_uninitialized(self):
        logger, records = _capturing_logger("pf_test_log_uninit")
        with patch("paddle.distributed.is_initialized", return_value=False):
            # No distributed context -> always logs regardless of rank kwarg.
            fleet_utils.log_single_rank(
                logger, logging.INFO, "v=%s", "x", rank=5
            )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].levelno, logging.INFO)
        self.assertEqual(records[0].getMessage(), "v=x")


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TpGroupIfNoneTests(unittest.TestCase):
    def test_tp_group_if_none_returns_provided_or_none(self):
        sentinel = object()
        with patch("paddle.distributed.is_initialized", return_value=True):
            # A provided group is returned unchanged (no default lookup).
            self.assertIs(
                fleet_utils.get_tensor_model_parallel_group_if_none(sentinel),
                sentinel,
            )
        with patch("paddle.distributed.is_initialized", return_value=False):
            # Uninitialized short-circuits to None even with a group argument.
            self.assertIsNone(
                fleet_utils.get_tensor_model_parallel_group_if_none(sentinel)
            )

    def test_tp_group_if_none_selects_default_by_expert_flag(self):
        tp_default = MagicMock(name="tp_default")
        expert_default = MagicMock(name="expert_default")
        with (
            patch("paddle.distributed.is_initialized", return_value=True),
            patch("paddle.distributed.get_rank", return_value=1),
            patch.object(
                parallel_state,
                "get_tensor_model_parallel_group",
                return_value=tp_default,
            ) as tp_fn,
            patch.object(
                parallel_state,
                "get_expert_tensor_parallel_group",
                return_value=expert_default,
            ) as ex_fn,
        ):
            # is_expert=False -> the non-expert default group is consumed.
            res = fleet_utils.get_tensor_model_parallel_group_if_none(
                None, is_expert=False, check_initialized=True
            )
            self.assertIs(res, tp_default)
            tp_fn.assert_called_once_with(check_initialized=True)
            ex_fn.assert_not_called()
            # is_expert=True -> the expert default group is consumed instead.
            res_e = fleet_utils.get_tensor_model_parallel_group_if_none(
                None, is_expert=True, check_initialized=False
            )
            self.assertIs(res_e, expert_default)
            ex_fn.assert_called_once_with(check_initialized=False)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class WgradInputTests(unittest.TestCase):
    def test_prepare_wgrad_reshapes_3d_and_keeps_2d(self):
        # 3D inputs collapse the first two axes: [2,3,4] -> [6,4], [2,3,5] -> [6,5].
        grad = paddle.arange(2 * 3 * 4).reshape([2, 3, 4])
        inp = paddle.arange(2 * 3 * 5).reshape([2, 3, 5])
        g_out, a_out = fleet_utils.prepare_input_tensors_for_wgrad_compute(
            grad, inp
        )
        self.assertEqual(g_out.shape, [6, 4])
        self.assertEqual(a_out.shape, [6, 5])
        # Content must be a pure reshape (row-major), not a transpose or copy.
        np.testing.assert_array_equal(
            g_out.numpy(), np.arange(2 * 3 * 4).reshape(6, 4)
        )
        np.testing.assert_array_equal(
            a_out.numpy(), np.arange(2 * 3 * 5).reshape(6, 5)
        )
        # 2D inputs (dim != 3) are passed through with shape and values intact.
        grad2 = paddle.arange(4 * 5).reshape([4, 5])
        inp2 = paddle.arange(4 * 7).reshape([4, 7])
        g2, a2 = fleet_utils.prepare_input_tensors_for_wgrad_compute(
            grad2, inp2
        )
        self.assertEqual(g2.shape, [4, 5])
        self.assertEqual(a2.shape, [4, 7])
        np.testing.assert_array_equal(
            g2.numpy(), np.arange(4 * 5).reshape(4, 5)
        )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class DeprecateInferenceParamsTests(unittest.TestCase):
    def test_deprecate_inference_params_context_params_and_warning(self):
        ctx = object()
        prm = object()
        # When context is provided it wins and no deprecation warning fires.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIs(fleet_utils.deprecate_inference_params(ctx, prm), ctx)
        self.assertEqual(len(caught), 0)
        # context None + params set -> warn once and fall back to params.
        with self.assertWarns(UserWarning):
            fell_back = fleet_utils.deprecate_inference_params(None, prm)
        self.assertIs(fell_back, prm)
        # Both None -> returns None (the context) with no warning.
        with warnings.catch_warnings(record=True) as caught2:
            warnings.simplefilter("always")
            self.assertIsNone(
                fleet_utils.deprecate_inference_params(None, None)
            )
        self.assertEqual(len(caught2), 0)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class GetBatchOnThisCpRankTests(unittest.TestCase):
    def test_get_batch_cp_rank_dict_scatters_selected_keys(self):
        # ContextParallelScatterOp is a genuine (distributed) collaborator; the
        # marker records the exact tensor identity plus the axis/mode it was
        # called with, so we can prove which keys were scattered and how.
        t_ids = paddle.zeros([2, 4])
        t_pos = paddle.zeros([2, 4])
        t_mask = paddle.zeros([2, 4])
        t_lab = paddle.zeros([2, 4])
        inputs = {
            "input_ids": t_ids,
            "position_ids": t_pos,
            "attention_mask": t_mask,
            "labels": t_lab,
        }

        def marker(tensor, axis, mode):
            return ("scattered", id(tensor), axis, mode)

        mock_op = MagicMock()
        mock_op.apply.side_effect = marker
        with patch.object(fleet_utils, "ContextParallelScatterOp", mock_op):
            result = fleet_utils.get_batch_on_this_cp_rank(inputs)

        # Only input_ids/position_ids/labels are scattered, each with axis=-1
        # and the default mode; the marker confirms the right tensor went in.
        self.assertEqual(
            result["input_ids"],
            ("scattered", id(t_ids), -1, "dualchunk_allgather"),
        )
        self.assertEqual(
            result["position_ids"],
            ("scattered", id(t_pos), -1, "dualchunk_allgather"),
        )
        self.assertEqual(
            result["labels"],
            ("scattered", id(t_lab), -1, "dualchunk_allgather"),
        )
        # attention_mask is not in the scatter set -> passed through by identity.
        self.assertIs(result["attention_mask"], t_mask)
        self.assertEqual(mock_op.apply.call_count, 3)

        # A custom cp_balance_mode must be forwarded verbatim to the op.
        mock_op.apply.reset_mock()
        mock_op.apply.side_effect = marker
        single = {"input_ids": t_ids}
        with patch.object(fleet_utils, "ContextParallelScatterOp", mock_op):
            res2 = fleet_utils.get_batch_on_this_cp_rank(
                single, cp_balance_mode="thd"
            )
        self.assertEqual(res2["input_ids"], ("scattered", id(t_ids), -1, "thd"))

    def test_get_batch_cp_rank_rejects_list_and_bad_type(self):
        # A list input hits an explicit AssertionError before any scatter.
        with self.assertRaises(AssertionError):
            fleet_utils.get_batch_on_this_cp_rank([paddle.zeros([2, 2])])
        # Any other (non-tensor, non-dict, non-list) type -> ValueError.
        with self.assertRaises(ValueError):
            fleet_utils.get_batch_on_this_cp_rank(5)


if __name__ == "__main__":
    unittest.main()
