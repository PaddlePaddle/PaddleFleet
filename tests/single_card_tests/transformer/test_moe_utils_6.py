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

"""No-card behavior tests for paddlefleet.transformer.moe.moe_utils.

Only CPU-executable, single-process logic is exercised here, with expected
values derived independently by hand / numpy from the production source:

  * global_moe_balance_training_logs_enabled -- None / non-callable / call()
  * _all_gather_local_tokens (group=None)     -- flatten then reshape [1, -1]
  * sort_chunks_by_idxs                        -- exact chunk split + reorder
  * all_gather_group / reduce_scatter_group    -- nranks==1 local clone path
  * barrier_ep                                 -- forwards the group arg
  * _log_summary                               -- max/min/var/median/mean stats
  * _log_tokens_per_expert                     -- count division + zero guard

``get_global_training_logs`` and ``paddle.distributed.barrier`` are genuine
not-under-test collaborators (a real training-log singleton / a real process
group); they are stubbed so the pure CPU logic can run. The nranks==1 branches
of the collective helpers are the interface's single-rank local fallback ONLY:
real cross-rank all-gather / reduce-scatter numerics are NOT verified here and
require a real EP process group (multi-card).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.moe.moe_utils import (
        _all_gather_local_tokens,
        _log_summary,
        _log_tokens_per_expert,
        all_gather_group,
        barrier_ep,
        global_moe_balance_training_logs_enabled,
        reduce_scatter_group,
        sort_chunks_by_idxs,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest, precise skip
    _IMPORT_ERROR = exc


_SKIP_REASON = (
    ""
    if _IMPORT_ERROR is None
    else f"moe_utils import failed: {_IMPORT_ERROR!r}"
)

_LOGS_PATH = "paddlefleet.transformer.moe.moe_utils.get_global_training_logs"
_BARRIER_PATH = (
    "paddlefleet.transformer.moe.moe_utils.paddle.distributed.barrier"
)


class _LogsSpy:
    """Stand-in for the training-logs singleton; records update() kwargs."""

    def __init__(self):
        self.calls = []

    def update(self, **kwargs):
        self.calls.append(kwargs)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestGlobalMoeBalanceTrainingLogsEnabled(unittest.TestCase):
    """global_moe_balance_training_logs_enabled == logs present AND flag call.

    Contract (moe_utils.py:663-669): return False when the logs singleton is
    None or has no callable ``is_moe_balance_logs_enabled``; otherwise return
    the *result of calling* that method (not the method object itself).
    """

    def test_false_when_logs_is_none(self):
        with patch(_LOGS_PATH, return_value=None):
            self.assertFalse(global_moe_balance_training_logs_enabled())

    def test_false_when_flag_attribute_missing(self):
        # SimpleNamespace has no is_moe_balance_logs_enabled -> getattr None.
        with patch(_LOGS_PATH, return_value=SimpleNamespace()):
            self.assertFalse(global_moe_balance_training_logs_enabled())

    def test_false_when_flag_attribute_not_callable(self):
        # A truthy but non-callable attr must NOT be treated as enabled;
        # guards against ``return getattr(...) is not None``.
        logs = SimpleNamespace(is_moe_balance_logs_enabled=True)
        with patch(_LOGS_PATH, return_value=logs):
            self.assertFalse(global_moe_balance_training_logs_enabled())

    def test_true_only_when_call_returns_true(self):
        logs = SimpleNamespace(is_moe_balance_logs_enabled=lambda: True)
        with patch(_LOGS_PATH, return_value=logs):
            self.assertTrue(global_moe_balance_training_logs_enabled())

    def test_false_when_call_returns_false(self):
        # Distinguishes "calls the method" from "returns the callable object";
        # a callable that returns False must yield False.
        logs = SimpleNamespace(is_moe_balance_logs_enabled=lambda: False)
        with patch(_LOGS_PATH, return_value=logs):
            self.assertFalse(global_moe_balance_training_logs_enabled())


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestAllGatherLocalTokens(unittest.TestCase):
    """_all_gather_local_tokens(x, None) flattens x then reshapes to [1, -1]."""

    def test_1d_input_preserved_as_single_row(self):
        tokens = paddle.to_tensor([1, 2, 3, 4], dtype="int64")
        result = _all_gather_local_tokens(tokens, None)
        self.assertEqual(list(result.shape), [1, 4])
        self.assertEqual(result.dtype, paddle.int64)
        np.testing.assert_array_equal(result.numpy(), [[1, 2, 3, 4]])

    def test_2d_input_is_flattened_row_major(self):
        # Locks the reshape([-1]) then reshape([1, -1]) ordering: a 2x3 tensor
        # must collapse row-major into one row, not stay [1, 2, 3].
        tokens = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype="int64")
        result = _all_gather_local_tokens(tokens, None)
        self.assertEqual(list(result.shape), [1, 6])
        np.testing.assert_array_equal(result.numpy(), [[1, 2, 3, 4, 5, 6]])


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestSortChunksByIdxs(unittest.TestCase):
    """sort_chunks_by_idxs splits along axis 0 then concatenates in idx order."""

    def test_reorders_chunks_by_index(self):
        # Rows carry distinguishable content so a wrong split boundary or a
        # reversed / mis-indexed reorder cannot pass on shape alone.
        data = np.arange(20).reshape(10, 2).astype("float32")
        input_tensor = paddle.to_tensor(data)
        split_sizes = paddle.to_tensor([3, 4, 3], dtype="int64")
        sorted_idxs = paddle.to_tensor([2, 0, 1], dtype="int64")

        output, permuted_probs = sort_chunks_by_idxs(
            input_tensor, split_sizes, sorted_idxs
        )

        chunks = [data[0:3], data[3:7], data[7:10]]
        expected = np.concatenate([chunks[i] for i in [2, 0, 1]], axis=0)
        np.testing.assert_array_equal(output.numpy(), expected)
        self.assertIsNone(permuted_probs)

    def test_identity_order_returns_original(self):
        data = np.arange(12).reshape(6, 2).astype("float32")
        input_tensor = paddle.to_tensor(data)
        split_sizes = paddle.to_tensor([2, 2, 2], dtype="int64")
        sorted_idxs = paddle.to_tensor([0, 1, 2], dtype="int64")

        output, permuted_probs = sort_chunks_by_idxs(
            input_tensor, split_sizes, sorted_idxs
        )
        np.testing.assert_array_equal(output.numpy(), data)
        self.assertIsNone(permuted_probs)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestCollectiveSingleRankLocalPath(unittest.TestCase):
    """nranks==1 local fallback of the EP collective helpers.

    This is the interface's single-rank path only: it must return a *copy* of
    the input with identical contents. Real multi-rank all-gather / reduce-
    scatter semantics need a real process group and are NOT tested here.
    """

    def test_all_gather_group_clones_input(self):
        data = np.arange(32).reshape(4, 8).astype("float32")
        input_tensor = paddle.to_tensor(data)
        group = SimpleNamespace(nranks=1)
        result = all_gather_group(input_tensor, group=group)
        self.assertIsNot(result, input_tensor)  # a clone, not the same object
        np.testing.assert_array_equal(result.numpy(), data)

    def test_reduce_scatter_group_clones_input(self):
        data = np.arange(32).reshape(4, 8).astype("float32")
        input_tensor = paddle.to_tensor(data)
        group = SimpleNamespace(nranks=1)
        result = reduce_scatter_group(input_tensor, group=group)
        self.assertIsNot(result, input_tensor)
        np.testing.assert_array_equal(result.numpy(), data)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestBarrierEp(unittest.TestCase):
    """barrier_ep forwards its group argument to paddle.distributed.barrier."""

    def test_forwards_group_positionally(self):
        ep_group = SimpleNamespace(tag="ep")
        with patch(_BARRIER_PATH) as mock_barrier:
            barrier_ep(ep_group)
        mock_barrier.assert_called_once()
        (called_arg,), called_kwargs = mock_barrier.call_args
        self.assertIs(called_arg, ep_group)  # exact group object, not a copy
        self.assertEqual(called_kwargs, {})


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestLogSummary(unittest.TestCase):
    """_log_summary writes hand-derived stats under the expected key prefix."""

    def test_computes_summary_statistics(self):
        # [1,2,3,4,5]: max=5, min=1, mean=3, median=3 (odd length, unambiguous),
        # unbiased var (paddle.var default unbiased=True) = 2.5.
        logs = _LogsSpy()
        with patch(_LOGS_PATH, return_value=logs):
            _log_summary("gate", 0, paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
        self.assertEqual(len(logs.calls), 1)
        kw = logs.calls[0]
        self.assertAlmostEqual(kw["gate_layer_0_max"], 5.0, places=5)
        self.assertAlmostEqual(kw["gate_layer_0_min"], 1.0, places=5)
        self.assertAlmostEqual(kw["gate_layer_0_mean"], 3.0, places=5)
        self.assertAlmostEqual(kw["gate_layer_0_median"], 3.0, places=5)
        self.assertAlmostEqual(kw["gate_layer_0_var"], 2.5, places=4)
        self.assertAlmostEqual(
            kw["gate_layer_0_max_mean_ratio"], 5.0 / 3.0, places=5
        )
        self.assertAlmostEqual(
            kw["gate_layer_0_min_mean_ratio"], 1.0 / 3.0, places=5
        )

    def test_skips_empty_summary_data(self):
        # numel == 0 guard (moe_utils.py:731-732): update must NOT be called.
        logs = _LogsSpy()
        with patch(_LOGS_PATH, return_value=logs):
            _log_summary("gate", 0, paddle.to_tensor([], dtype="float32"))
        self.assertEqual(logs.calls, [])

    def test_mtp_layer_switches_key_prefix(self):
        # is_mtp_layer=True must produce "mtp_layer" instead of "layer".
        logs = _LogsSpy()
        with patch(_LOGS_PATH, return_value=logs):
            _log_summary(
                "gate", 2, paddle.to_tensor([1.0, 3.0]), is_mtp_layer=True
            )
        kw = logs.calls[0]
        self.assertIn("gate_mtp_layer_2_max", kw)
        self.assertNotIn("gate_layer_2_max", kw)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestLogTokensPerExpert(unittest.TestCase):
    """_log_tokens_per_expert divides by count and logs raw + averaged stats."""

    def test_divides_summary_by_count(self):
        # count=2 -> avg = summary/2. Two _log_summary calls fire: the "_avg"
        # key carries the divided mean, the raw key the undivided mean.
        logs = _LogsSpy()
        with patch(_LOGS_PATH, return_value=logs):
            _log_tokens_per_expert(
                0,
                "tpe",
                paddle.to_tensor([2.0, 4.0, 6.0]),
                paddle.to_tensor([2]),
            )
        self.assertEqual(len(logs.calls), 2)
        merged = {}
        for call in logs.calls:
            merged.update(call)
        self.assertAlmostEqual(merged["tpe_avg_layer_0_mean"], 2.0, places=5)
        self.assertAlmostEqual(merged["tpe_layer_0_mean"], 4.0, places=5)

    def test_zero_count_replaced_with_one(self):
        # count == 0 (moe_utils.py:766): replaced by ones, so avg == summary
        # instead of dividing by zero.
        logs = _LogsSpy()
        with patch(_LOGS_PATH, return_value=logs):
            _log_tokens_per_expert(
                0,
                "tpe",
                paddle.to_tensor([2.0, 4.0, 6.0]),
                paddle.to_tensor([0]),
            )
        merged = {}
        for call in logs.calls:
            merged.update(call)
        self.assertAlmostEqual(merged["tpe_avg_layer_0_mean"], 4.0, places=5)
        self.assertAlmostEqual(merged["tpe_layer_0_mean"], 4.0, places=5)


if __name__ == "__main__":
    unittest.main()
