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

"""Behavior tests for the ``filter_scores`` / ``filter_scores_grad`` custom ops
in the ``paddlefleet_ops._extensions.ops`` extension.

These ops implement a *stream-compaction with gather/scatter* used by the MoE
router (see ``paddlefleet.transformer.moe.moe_utils.FilterScores``):

  forward  ``filter_scores(probs, indices)``
      Returns a 1-D tensor holding ``probs[i]`` for every flattened position
      ``i`` where ``indices[i] != -1``, kept in row-major order. Positions whose
      index is ``-1`` are dropped entirely. Output length == count of valid
      positions.

  backward ``filter_scores_grad(indices, grad_topk_scores)``
      Scatters the compact gradient back to ``indices``-shaped ``grad_probs``:
      valid positions receive the corresponding compact grad (same order),
      dropped (``-1``) positions receive ``0``.

The expected values below are hand-derived from a fixed, deliberately
non-uniform input that mixes valid and ``-1`` positions across rows, so an
implementation that reorders, mis-routes the scatter, forgets to zero the
dropped slots, or swaps rows/cols would be rejected. We never call the op to
produce its own expected value.

The kernels are CUDA-only (``FilterScoresGPU`` hard-checks a GPU place), and
``paddlefleet_ops`` imports ``paddle`` at import time. The local CI image for
this file may lack ``paddle`` or a GPU, so the import is guarded and the whole
suite skips honestly instead of erroring at collection or faking a pass.
"""

import unittest

try:
    import numpy as np
    import paddle

    # Importing ``paddlefleet_ops`` runs ``import_custom_ops(...)`` which
    # registers the compiled ``filter_scores`` / ``filter_scores_grad`` custom
    # ops with paddle so they are reachable via ``_run_custom_op``.
    import paddlefleet_ops  # noqa: F401

    _IMPORT_ERROR = None
except ImportError as exc:  # only a genuine missing dependency, never broad
    np = None
    paddle = None
    _IMPORT_ERROR = exc


def _gpu_available():
    """Honest capability probe: op requires a real CUDA device."""
    if paddle is None:
        return False
    if not paddle.is_compiled_with_cuda():
        return False
    try:
        return paddle.device.cuda.device_count() > 0
    except Exception:
        return False


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle / paddlefleet_ops not importable: {_IMPORT_ERROR!r}",
)
class TestFilterScoresOp(unittest.TestCase):
    """Numeric compaction/scatter behavior of the filter_scores custom op."""

    def setUp(self):
        if not _gpu_available():
            self.skipTest(
                "filter_scores is a CUDA-only op (FilterScoresGPU checks a GPU "
                "place); no compiled-with-CUDA paddle or GPU device present."
            )
        paddle.set_device("gpu")

    def _run_filter_scores(self, probs, indices):
        return paddle._C_ops._run_custom_op("filter_scores", probs, indices)[0]

    def _run_filter_scores_grad(self, indices, grad_topk_scores):
        return paddle._C_ops._run_custom_op(
            "filter_scores_grad", indices, grad_topk_scores
        )[0]

    def test_forward_compacts_valid_positions_in_order(self):
        # Fixed, non-uniform 2x3 input. Valid positions (index != -1) in
        # row-major order are: (0,0), (0,2), (1,1), (1,2).
        probs = paddle.to_tensor(
            [[0.10, 0.20, 0.30], [0.40, 0.50, 0.60]], dtype="float32"
        )
        indices = paddle.to_tensor([[5, -1, 7], [-1, 2, 9]], dtype="int64")

        out = self._run_filter_scores(probs, indices)

        # Hand-derived: gather probs at the four valid slots, drop the -1 slots.
        expected = np.array([0.10, 0.30, 0.50, 0.60], dtype=np.float32)
        self.assertEqual(list(out.shape), [4])
        np.testing.assert_allclose(
            out.numpy().astype(np.float32), expected, rtol=1e-6, atol=1e-6
        )

    def test_forward_all_minus_one_yields_empty(self):
        probs = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        indices = paddle.to_tensor([[-1, -1], [-1, -1]], dtype="int64")

        out = self._run_filter_scores(probs, indices)

        # Every position dropped -> empty output (length 0).
        self.assertEqual(int(out.numel()), 0)

    def test_forward_preserves_value_identity_not_just_count(self):
        # Two inputs with identical validity mask but different values must
        # yield different compacted outputs; a count-only impl would not.
        indices = paddle.to_tensor([[0, -1, 3, -1]], dtype="int64")
        probs_a = paddle.to_tensor([[1.0, 9.0, 2.0, 9.0]], dtype="float32")
        probs_b = paddle.to_tensor([[7.0, 9.0, 8.0, 9.0]], dtype="float32")

        out_a = self._run_filter_scores(probs_a, indices).numpy()
        out_b = self._run_filter_scores(probs_b, indices).numpy()

        np.testing.assert_allclose(out_a, [1.0, 2.0], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(out_b, [7.0, 8.0], rtol=1e-6, atol=1e-6)

    def test_backward_scatters_grad_and_zeros_dropped(self):
        # Same validity layout as the forward test. The compact upstream grad
        # (length 4, one per valid slot in row-major order) must be scattered
        # back to the valid positions, with 0 at every -1 position.
        indices = paddle.to_tensor([[5, -1, 7], [-1, 2, 9]], dtype="int64")
        grad_topk = paddle.to_tensor([10.0, 20.0, 30.0, 40.0], dtype="float32")

        grad_probs = self._run_filter_scores_grad(indices, grad_topk)

        # Hand-derived scatter:
        #   (0,0)<-10, (0,1)=0, (0,2)<-20, (1,0)=0, (1,1)<-30, (1,2)<-40
        expected = np.array(
            [[10.0, 0.0, 20.0], [0.0, 30.0, 40.0]], dtype=np.float32
        )
        self.assertEqual(list(grad_probs.shape), [2, 3])
        np.testing.assert_allclose(
            grad_probs.numpy().astype(np.float32),
            expected,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_forward_backward_roundtrip_through_pylayer(self):
        # Exercise the production autograd path (FilterScores PyLayer) end to
        # end: forward gathers, backward scatters. Expected grad is hand
        # derived, not read back from the op.
        from paddlefleet.transformer.moe.moe_utils import filter_scores

        probs = paddle.to_tensor(
            [[0.10, 0.20, 0.30], [0.40, 0.50, 0.60]], dtype="float32"
        )
        probs.stop_gradient = False
        indices = paddle.to_tensor([[5, -1, 7], [-1, 2, 9]], dtype="int64")

        topk = filter_scores(probs, indices)
        np.testing.assert_allclose(
            topk.numpy().astype(np.float32),
            np.array([0.10, 0.30, 0.50, 0.60], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

        # Non-uniform upstream grad exposes mis-routing / order errors.
        topk.backward(paddle.to_tensor([1.0, 2.0, 3.0, 4.0], dtype="float32"))
        self.assertIsNotNone(probs.grad)
        expected_grad = np.array(
            [[1.0, 0.0, 2.0], [0.0, 3.0, 4.0]], dtype=np.float32
        )
        np.testing.assert_allclose(
            probs.grad.numpy().astype(np.float32),
            expected_grad,
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
