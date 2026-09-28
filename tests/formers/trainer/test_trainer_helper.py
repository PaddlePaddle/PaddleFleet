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

"""Behavior tests for ``paddlefleet.trainer.utils.helper``.

Module mapping (unit-test-rules.md): the pad/concat/detach/numpify/truncate
helpers and the world-size<=1 short circuits belong to the "Trainer 训练引擎"
data-collection path; the file-synchronization helpers
(``distributed_isfile``/``distributed_file``/``_get_actual_path``) serve
"Checkpoint 与权重管理". These are exercised as 无卡 (CPU) cases.

Oracle strategy: every expected value is hand-derived from the documented
contract of each helper. Inputs are small and content-distinguishable so the
padded gap positions, concatenation order, detached values, upcast dtype,
truncation slice and resolved paths are each computed independently in the
test (by hand or with plain NumPy) -- never by calling the function under test
to produce its own expected output, and never asserting shape alone.

Not covered here (require a real process group; cross-rank numerics remain
unverified without real multi-card runs): ``distributed_concat`` (dist
all_gather), the multi-node branches of ``distributed_isfile``/
``distributed_file`` (all_reduce / all_gather / broadcast), and the
``broadcast_dp=False`` all-gather path of ``broadcast_moe_optimizer``.
"""

import os
import tempfile
import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.trainer.utils.helper import (
        _get_actual_path,
        broadcast_dataset_rank0_model,
        broadcast_dp_optimizer,
        distributed_file,
        distributed_isfile,
        nested_concat,
        nested_detach,
        nested_numpify,
        nested_truncate,
        numpy_pad_and_concatenate,
        paddle_pad_and_concatenate,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not installed on this host
    paddle = None
    _IMPORT_ERROR = exc


requires_paddle = unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}",
)


class _CpuHelperTest(unittest.TestCase):
    """Base that pins execution to CPU (无卡)."""

    @classmethod
    def setUpClass(cls):
        if _IMPORT_ERROR is None:
            paddle.set_device("cpu")


@requires_paddle
class TestPaddlePadAndConcatenate(_CpuHelperTest):
    def test_1d_concatenates_in_order(self):
        """1D path is a plain axis-0 concat; oracle = manual [1,2,3,4,5]."""
        t1 = paddle.to_tensor([1.0, 2.0, 3.0])
        t2 = paddle.to_tensor([4.0, 5.0])
        out = paddle_pad_and_concatenate(t1, t2)
        np.testing.assert_array_equal(out.numpy(), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_equal_width_concatenates_rows(self):
        t1 = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        t2 = paddle.to_tensor([[7.0, 8.0, 9.0]])
        out = paddle_pad_and_concatenate(t1, t2)
        np.testing.assert_array_equal(
            out.numpy(), [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        )

    def test_unequal_width_pads_second_axis(self):
        """Different column counts pad the short rows with padding_index.

        Hand-filled oracle: new shape (3, 3); t1 lands at [:2, :2], t2 at
        [2:, :3]; the single gap column is the padding value -100.
        """
        t1 = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])  # 2x2
        t2 = paddle.to_tensor([[5.0, 6.0, 7.0]])  # 1x3
        out = paddle_pad_and_concatenate(t1, t2, padding_index=-100).numpy()
        expected = np.array(
            [[1.0, 2.0, -100.0], [3.0, 4.0, -100.0], [5.0, 6.0, 7.0]]
        )
        np.testing.assert_array_equal(out, expected)

    def test_padding_index_fills_only_gap_positions(self):
        """padding_index must appear only at padded positions, not overwrite
        real data; oracle places 9.0 at row0 cols 2-3 only."""
        t1 = paddle.to_tensor([[1.0, 2.0]])  # 1x2
        t2 = paddle.to_tensor([[3.0, 4.0, 5.0, 6.0]])  # 1x4
        out = paddle_pad_and_concatenate(t1, t2, padding_index=9.0).numpy()
        expected = np.array([[1.0, 2.0, 9.0, 9.0], [3.0, 4.0, 5.0, 6.0]])
        np.testing.assert_array_equal(out, expected)


@requires_paddle
class TestNumpyPadAndConcatenate(_CpuHelperTest):
    def test_1d_concatenates_in_order(self):
        a1 = np.array([1.0, 2.0, 3.0])
        a2 = np.array([4.0, 5.0])
        out = numpy_pad_and_concatenate(a1, a2)
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_equal_width_concatenates_rows(self):
        a1 = np.array([[1.0, 2.0, 3.0]])
        a2 = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        out = numpy_pad_and_concatenate(a1, a2)
        np.testing.assert_array_equal(out, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_unequal_width_pads_second_axis(self):
        """Hand-filled oracle: new shape (3, 3), pad value -1 at the gaps."""
        a1 = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2x2
        a2 = np.array([[5.0, 6.0, 7.0]])  # 1x3
        out = numpy_pad_and_concatenate(a1, a2, padding_index=-1)
        expected = np.array([[1, 2, -1], [3, 4, -1], [5, 6, 7]])
        np.testing.assert_array_equal(out, expected)


@requires_paddle
class TestNestedConcat(_CpuHelperTest):
    def test_dispatches_to_tensor_padding(self):
        t1 = paddle.to_tensor([[1.0, 2.0]])
        t2 = paddle.to_tensor([[3.0, 4.0, 5.0]])
        out = nested_concat(t1, t2, padding_index=0.0).numpy()
        np.testing.assert_array_equal(out, [[1, 2, 0], [3, 4, 5]])

    def test_dispatches_to_numpy(self):
        a1 = np.array([[1.0, 2.0, 3.0]])
        a2 = np.array([[4.0, 5.0, 6.0]])
        out = nested_concat(a1, a2)
        self.assertIsInstance(out, np.ndarray)
        np.testing.assert_array_equal(out, [[1, 2, 3], [4, 5, 6]])

    def test_preserves_list_structure_and_content(self):
        """List leaves are recursed element-wise; oracle keeps pairing:
        leaf0 is a 1D concat, leaf1 an equal-width row concat."""
        t1 = [paddle.to_tensor([1.0, 2.0]), paddle.to_tensor([[1.0, 2.0]])]
        t2 = [paddle.to_tensor([3.0]), paddle.to_tensor([[3.0, 4.0]])]
        out = nested_concat(t1, t2, padding_index=0.0)
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        np.testing.assert_array_equal(out[0].numpy(), [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(out[1].numpy(), [[1, 2], [3, 4]])

    def test_type_mismatch_raises_assertion(self):
        t = paddle.to_tensor([[1.0, 2.0]])
        a = np.array([[1.0, 2.0]])
        with self.assertRaises(AssertionError):
            nested_concat(t, a)

    def test_unsupported_leaf_type_raises_typeerror(self):
        """Equal python-int types pass the type assertion, then fall through
        to the else-branch TypeError for non tensor/ndarray leaves."""
        with self.assertRaises(TypeError):
            nested_concat(5, 6)


@requires_paddle
class TestNestedDetach(_CpuHelperTest):
    def test_detaches_tensor_keeps_values(self):
        """detach() must stop gradient while preserving the data verbatim."""
        t = paddle.to_tensor([1.0, 2.0, 3.0])
        t.stop_gradient = False
        out = nested_detach(t)
        self.assertTrue(out.stop_gradient)
        np.testing.assert_array_equal(out.numpy(), [1.0, 2.0, 3.0])

    def test_preserves_tuple_structure_and_values(self):
        t1 = paddle.to_tensor([1.0])
        t1.stop_gradient = False
        t2 = paddle.to_tensor([[2.0, 3.0]])
        t2.stop_gradient = False
        out = nested_detach((t1, t2))
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertTrue(out[0].stop_gradient)
        self.assertTrue(out[1].stop_gradient)
        np.testing.assert_array_equal(out[1].numpy(), [[2.0, 3.0]])

    def test_preserves_list_structure(self):
        t = paddle.to_tensor([5.0])
        t.stop_gradient = False
        out = nested_detach([t])
        self.assertIsInstance(out, list)
        self.assertTrue(out[0].stop_gradient)
        np.testing.assert_array_equal(out[0].numpy(), [5.0])


@requires_paddle
class TestNestedNumpify(_CpuHelperTest):
    def test_returns_ndarray_with_same_values(self):
        t = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        out = nested_numpify(t)
        self.assertIsInstance(out, np.ndarray)
        np.testing.assert_array_equal(out, [[1.0, 2.0], [3.0, 4.0]])

    def test_float16_is_upcast_to_float32(self):
        """Contract: float16 leaves are cast to float32 before .numpy();
        oracle dtype is float32 and the (exactly representable) values hold."""
        t = paddle.to_tensor([1.5, 2.5], dtype=paddle.float16)
        out = nested_numpify(t)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_equal(
            out, np.array([1.5, 2.5], dtype=np.float32)
        )

    def test_preserves_list_of_arrays(self):
        t1 = paddle.to_tensor([1.0, 2.0])
        t2 = paddle.to_tensor([[3.0]])
        out = nested_numpify([t1, t2])
        self.assertIsInstance(out, list)
        self.assertIsInstance(out[0], np.ndarray)
        np.testing.assert_array_equal(out[0], [1.0, 2.0])
        np.testing.assert_array_equal(out[1], [[3.0]])


@requires_paddle
class TestNestedTruncate(_CpuHelperTest):
    def test_truncates_tensor_content(self):
        """Truncation keeps the first ``limit`` entries in order."""
        t = paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        out = nested_truncate(t, 4)
        np.testing.assert_array_equal(out.numpy(), [0.0, 1.0, 2.0, 3.0])

    def test_truncates_first_axis_only(self):
        t = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        out = nested_truncate(t, 2)
        np.testing.assert_array_equal(out.numpy(), [[1.0, 2.0], [3.0, 4.0]])

    def test_limit_beyond_length_returns_all(self):
        t = paddle.to_tensor([1.0, 2.0, 3.0])
        out = nested_truncate(t, 10)
        np.testing.assert_array_equal(out.numpy(), [1.0, 2.0, 3.0])

    def test_preserves_list_structure_and_truncates_each(self):
        t1 = paddle.to_tensor([0.0, 1.0, 2.0, 3.0])
        t2 = paddle.to_tensor([9.0, 8.0, 7.0])
        out = nested_truncate([t1, t2], 2)
        self.assertIsInstance(out, list)
        np.testing.assert_array_equal(out[0].numpy(), [0.0, 1.0])
        np.testing.assert_array_equal(out[1].numpy(), [9.0, 8.0])


@requires_paddle
class TestGetActualPath(_CpuHelperTest):
    def test_prefers_json_over_bin(self):
        """When both exist, .json wins because it is probed first."""
        with tempfile.TemporaryDirectory() as d:
            base = os.path.join(d, "model")
            open(base + ".json", "w").close()
            open(base + ".bin", "w").close()
            self.assertEqual(_get_actual_path(base + ".bin"), base + ".json")

    def test_falls_back_to_bin_when_no_json(self):
        with tempfile.TemporaryDirectory() as d:
            base = os.path.join(d, "model")
            open(base + ".bin", "w").close()
            self.assertEqual(_get_actual_path(base + ".json"), base + ".bin")

    def test_returns_original_when_neither_exists(self):
        with tempfile.TemporaryDirectory() as d:
            missing = os.path.join(d, "model.json")
            self.assertEqual(_get_actual_path(missing), missing)


class _SingleNodeEnv(_CpuHelperTest):
    """Pin PADDLE_TRAINERS_NUM=1 (single-node) with guaranteed restore."""

    def setUp(self):
        self._orig = os.environ.get("PADDLE_TRAINERS_NUM")
        os.environ["PADDLE_TRAINERS_NUM"] = "1"
        self.addCleanup(self._restore_env)

    def _restore_env(self):
        if self._orig is None:
            os.environ.pop("PADDLE_TRAINERS_NUM", None)
        else:
            os.environ["PADDLE_TRAINERS_NUM"] = self._orig


@requires_paddle
class TestDistributedIsfileSingleNode(_SingleNodeEnv):
    def test_existing_json_returns_true(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.json")
            with open(p, "w") as f:
                f.write("x")
            self.assertIs(distributed_isfile(p), True)

    def test_missing_file_returns_false(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIs(
                distributed_isfile(os.path.join(d, "none.json")), False
            )

    def test_json_request_matches_existing_bin(self):
        """.json/.bin are interchangeable: asking for .json while only .bin
        exists still reports True (targets = both extensions)."""
        with tempfile.TemporaryDirectory() as d:
            base = os.path.join(d, "w")
            with open(base + ".bin", "w") as f:
                f.write("x")
            self.assertIs(distributed_isfile(base + ".json"), True)

    def test_plain_extension_checks_exact_path(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "note.txt")
            with open(p, "w") as f:
                f.write("x")
            self.assertIs(distributed_isfile(p), True)


@requires_paddle
class TestDistributedFileSingleNode(_SingleNodeEnv):
    def test_returns_path_unchanged_for_plain_name(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "note.txt")
            self.assertEqual(distributed_file(p), p)

    def test_resolves_json_request_to_existing_bin(self):
        with tempfile.TemporaryDirectory() as d:
            base = os.path.join(d, "w")
            with open(base + ".bin", "w") as f:
                f.write("x")
            self.assertEqual(distributed_file(base + ".json"), base + ".bin")

    def test_returns_json_when_present(self):
        with tempfile.TemporaryDirectory() as d:
            base = os.path.join(d, "w")
            with open(base + ".json", "w") as f:
                f.write("x")
            self.assertEqual(distributed_file(base + ".json"), base + ".json")


@requires_paddle
class TestSingleProcessBroadcast(_SingleNodeEnv):
    def test_dp_optimizer_returns_same_object_when_world_size_one(self):
        """world_size <= 1 short-circuits: the exact input dict is returned
        unchanged (identity), no collective is invoked."""
        state = {"moment1": paddle.to_tensor([1.0, 2.0])}
        out = broadcast_dp_optimizer(state)
        self.assertIs(out, state)

    def test_broadcast_dataset_rank0_model_noop_when_world_size_one(self):
        """world_size <= 1 returns None without touching the model object."""
        sentinel = object()
        self.assertIsNone(broadcast_dataset_rank0_model(sentinel))


if __name__ == "__main__":
    unittest.main()
