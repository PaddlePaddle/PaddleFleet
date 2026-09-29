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

"""Behavior tests for the fused MoE TopK Triton op (branch set 2).

Module under test: ``paddlefleet.triton_ops.moe_topk_fusion`` (repository
module map: "计算优化" / Fused Ops). This file deliberately exercises
branches that are NOT touched by the base coverage of this module:

  * ``return_alpha=True`` forward path -- the extra cutoff output equal to
    the (k+1)-th largest choice value per token (the ``return_alpha`` block
    of ``_fwd_kernel`` and the 3-tuple return of ``MoETopkFusion.forward``).
  * Choice-topk tie handling -- when two experts share the max choice value
    the smaller index must be selected first (``tl.min`` over the argmax
    candidates).
  * Node-limit selection with ``topk_group > 1`` -- multi-group selection by
    per-group top-2 sum, whereas the plain path uses ``topk_group == 1``.
  * Routing-map OR reduction for a repeated expert id in a single token --
    a duplicated index must set the expert bit exactly once, so the dispatch
    count is 1 rather than 2 (the ``_bitwise_or`` reduction in
    ``_routing_map_fwd_bitmap_kernel``).

Every expected value below is hand-derived from the documented selection and
routing semantics, independently of the kernel implementation.

Environment note: these kernels are Triton/CUDA only and have no CPU
implementation. There is no import-time pure-Python logic in the module that
can be observed without ``paddle`` and ``triton`` (the module imports both at
top level and every observable value is produced on the GPU). When paddle is
not installed, or no CUDA GPU with an active Triton runtime is present, the
whole suite is skipped honestly rather than fake-passing on CPU.
"""

import unittest

try:
    import numpy as np
    import paddle
    import triton

    _IMPORT_OK = True
    _IMPORT_ERR = ""
except ImportError as exc:  # missing dependency -> honest skip, not swallowed
    paddle = None
    triton = None
    _IMPORT_OK = False
    _IMPORT_ERR = str(exc)

if _IMPORT_OK:
    # A genuine API/compile error here must surface, so only ImportError is
    # treated as a "missing dependency" skip reason.
    try:
        from paddlefleet.triton_ops.moe_topk_fusion import (
            MoETopkFusion,
            routing_map_fusion_forward,
        )
    except ImportError as exc:
        _IMPORT_OK = False
        _IMPORT_ERR = str(exc)


def _gpu_ready():
    """True only when paddle+triton import and a CUDA GPU with an active
    Triton runtime is present. Hardware absence is a skip, not a pass."""
    if not _IMPORT_OK:
        return False
    if not paddle.is_compiled_with_cuda():
        return False
    try:
        triton.runtime.driver.active.get_current_device()
    except Exception:
        # No usable GPU/Triton runtime -> genuine hardware/runtime absence.
        return False
    return True


_SKIP_REASON = (
    "fused MoE topk kernels require paddle + triton and a CUDA GPU with an "
    "active Triton runtime; there is no CPU implementation to validate "
    "(import error: {})".format(_IMPORT_ERR or "none")
)


@unittest.skipUnless(_gpu_ready(), _SKIP_REASON)
class TestMoETopkFusionBranches2(unittest.TestCase):
    def setUp(self):
        paddle.device.set_device("gpu:0")

    def test_forward_return_alpha_is_cutoff_value(self):
        # return_alpha=True adds a per-token cutoff equal to the (k+1)-th
        # largest choice value. Hand-derived:
        #   row0 sorted desc: 0.9, 0.7, 0.3, 0.1 -> top-2 {0.9@1, 0.7@3},
        #                     3rd largest = 0.3 @ idx2  -> alpha = 0.3
        #   row1 sorted desc: 0.8, 0.5, 0.2, 0.1 -> top-2 {0.8@2, 0.5@0},
        #                     3rd largest = 0.2 @ idx1  -> alpha = 0.2
        gate_probs = paddle.to_tensor(
            [[0.1, 0.9, 0.3, 0.7], [0.5, 0.2, 0.8, 0.1]],
            dtype="float32",
        )
        probs_for_choice = gate_probs.clone()

        topk_probs, topk_indices, alpha = MoETopkFusion.apply(
            gate_probs, probs_for_choice, 2, False, 1, 1, False, True
        )

        # The extra alpha output only exists on this branch.
        self.assertEqual(topk_indices.numpy().tolist(), [[1, 3], [2, 0]])
        np.testing.assert_allclose(
            topk_probs.numpy(),
            [[0.9, 0.7], [0.8, 0.5]],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            alpha.numpy(), [0.3, 0.2], rtol=1e-6, atol=1e-6
        )

    def test_forward_breaks_choice_ties_toward_smaller_index(self):
        # idx0 and idx1 both hold the max value 0.5; the kernel resolves the
        # tie by argmin of the equal-max positions, so idx0 is picked first,
        # then idx1. A "largest index wins" bug would yield [[1, 0]] here.
        gate_probs = paddle.to_tensor([[0.5, 0.5, 0.2, 0.1]], dtype="float32")
        probs_for_choice = gate_probs.clone()

        topk_probs, topk_indices = MoETopkFusion.apply(
            gate_probs, probs_for_choice, 2, False, 1, 1, False
        )

        self.assertEqual(topk_indices.numpy().tolist(), [[0, 1]])
        np.testing.assert_allclose(
            topk_probs.numpy(), [[0.5, 0.5]], rtol=1e-6, atol=1e-6
        )

    def test_node_limit_selects_multiple_groups_by_top2_sum(self):
        # n_experts=8, n_group=4 (experts-per-group=2), topk_group=2.
        # Group score = sum of the top-2 choice values in the group:
        #   G0 {0,1}: 0.1+0.1 = 0.2
        #   G1 {2,3}: 0.9+0.8 = 1.7  (rank 1)
        #   G2 {4,5}: 0.2+0.2 = 0.4
        #   G3 {6,7}: 0.7+0.6 = 1.3  (rank 2)
        # Selected groups {G1, G3} -> allowed experts {2,3,6,7}.
        # Choice top-2 among allowed: 0.9@2, 0.8@3 -> indices [[2, 3]].
        probs_for_choice = paddle.to_tensor(
            [[0.1, 0.1, 0.9, 0.8, 0.2, 0.2, 0.7, 0.6]],
            dtype="float32",
        )
        # Distinct gate values verify probs are fetched from gate, not choice.
        gate_probs = paddle.arange(8, dtype="float32").reshape([1, 8]) / 10

        topk_probs, topk_indices = MoETopkFusion.apply(
            gate_probs, probs_for_choice, 2, True, 4, 2, False
        )

        self.assertEqual(topk_indices.numpy().tolist(), [[2, 3]])
        np.testing.assert_allclose(
            topk_probs.numpy(), [[0.2, 0.3]], rtol=1e-6, atol=1e-6
        )

    def test_routing_map_repeated_expert_id_counted_once(self):
        # Row 0 selects expert 1 twice. The OR/bit reduction must set the
        # expert bit exactly once, so routing_map row0 = [0,1,0,0] and the
        # dispatch count for expert 1 is 1 (a sum reduction would give 2).
        gate_probs = paddle.ones([2, 4], dtype="float32")
        topk_indices = paddle.to_tensor([[1, 1], [0, 2]], dtype="int64")

        routing_map, topk_indices_out, dispatch_mask = (
            routing_map_fusion_forward(gate_probs, topk_indices)
        )

        self.assertEqual(
            routing_map.numpy().tolist(),
            [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]],
        )
        # No masks supplied: indices pass through unchanged.
        self.assertEqual(topk_indices_out.numpy().tolist(), [[1, 1], [0, 2]])
        # expert0:1, expert1:1 (not 2), expert2:1, expert3:0
        self.assertEqual(dispatch_mask.numpy().tolist(), [1, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
