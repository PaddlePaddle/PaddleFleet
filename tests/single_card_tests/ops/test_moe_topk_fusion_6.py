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

"""Behavior tests for the fused MoE TopK Triton op (variant _6).

Module under test: ``paddlefleet.triton_ops.moe_topk_fusion``. In the
repository module map this is the "计算优化 / Fused Ops" boundary: a Triton
kernel selects the top-k experts per token, optionally normalizes the picked
gate probabilities, and a separate bitmap kernel builds the routing map and
dispatch mask.

This variant deliberately targets branches NOT exercised by the base / _2 / _4
behavior tests nor by ``custom_ops/test_moe_topk_fusion_triton.py``:

  * forward ``norm_gate_logits=True`` -- the picked gate probabilities are
    divided by their per-token sum so each token's returned weights sum to 1
    (the ``norm_gate_logits`` normalization block of ``_fwd_kernel``);
  * the ``denom = max(total_sum, 1e-12)`` clamp on that path -- a token whose
    selected gate values sum to exactly zero must yield finite zeros, not a
    0/0 NaN;
  * backward on the ``norm_gate_logits=True`` path -- the normalized-gradient
    formula ``grad = (grad_out - dot(grad_out, normed)) / sigma`` scattered
    into the selected slots (``_bwd_kernel`` normed branch), which is distinct
    from the plain scatter backward covered elsewhere;
  * routing map driven by ``input_ids`` padding alone (padding-only branch,
    ``has_input_ids=True`` / ``has_pure_text_mask=False``);
  * routing map with ``n_experts > 32`` so the bitmap kernel runs more than one
    expert-dim tile (``pid_n > 0``, ``base = 32``) and accumulates the dispatch
    mask across tiles via ``atomic_add``.

Every expected value below is hand-derived from small, per-row-distinct inputs
so that swapped indices, a missing normalization/clamp, a wrong normalized
gradient, a dropped mask term, or a second-tile addressing bug would be
rejected. The kernels compile to PTX and only execute on a CUDA GPU with an
active Triton runtime, and this environment has no ``paddle`` installed, so the
cases skip honestly rather than assert on a faked ``triton.language`` shim.
They are real GPU behavior tests, not source-introspection or shape-only checks.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

_IMPORT_OK = True
_IMPORT_ERR = ""
try:  # capability probe: only ImportError means "dependency absent".
    import numpy as np
    import paddle
    import triton
except ImportError as exc:  # pragma: no cover - env-dependent
    _IMPORT_OK = False
    _IMPORT_ERR = str(exc)

if _IMPORT_OK:
    # A genuine API/compile error must surface; only ImportError is a skip.
    try:
        from paddlefleet.triton_ops.moe_topk_fusion import (
            MoETopkFusion,
            routing_map_fusion_forward,
        )
    except ImportError as exc:  # pragma: no cover - env-dependent
        _IMPORT_OK = False
        _IMPORT_ERR = str(exc)


def _gpu_ready():
    """True only when paddle+triton import and a CUDA GPU with an active
    Triton runtime is present. Hardware/runtime absence is a skip, not a pass.
    """
    if not _IMPORT_OK:
        return False
    if not paddle.device.is_compiled_with_cuda():
        return False
    if paddle.device.cuda.device_count() <= 0:
        return False
    try:
        triton.runtime.driver.active.get_current_device()
    except Exception:
        return False
    return True


_SKIP_REASON = (
    "fused MoE topk kernels require paddle + triton and a CUDA GPU with an "
    "active Triton runtime; the kernels compile to PTX and have no CPU "
    "implementation to validate (import error: {})".format(
        _IMPORT_ERR or "none"
    )
)


@unittest.skipUnless(_gpu_ready(), _SKIP_REASON)
class TestMoETopkFusionForwardNormalized(unittest.TestCase):
    """Forward with norm_gate_logits=True (per-token weight normalization)."""

    def setUp(self):
        paddle.device.set_device("gpu:0")

    def test_norm_gate_logits_divides_selected_probs_by_their_sum(self):
        # Choice scores decide WHICH experts win; gate_probs supply the
        # weights that are then normalized. Keeping them distinct makes a
        # gate/choice mix-up observable.
        #   row0 choice [0.1,0.4,0.3,0.2] -> top-2 idx1(0.4), idx2(0.3)
        #        gate   [1.0,3.0,1.0,5.0] -> picked 3.0, 1.0; sum 4.0
        #        normalized -> [0.75, 0.25]
        #   row1 choice [0.7,0.2,0.5,0.1] -> top-2 idx0(0.7), idx2(0.5)
        #        gate   [2.0,9.0,6.0,9.0] -> picked 2.0, 6.0; sum 8.0
        #        normalized -> [0.25, 0.75]
        probs_for_choice = paddle.to_tensor(
            [[0.1, 0.4, 0.3, 0.2], [0.7, 0.2, 0.5, 0.1]], dtype="float32"
        )
        gate_probs = paddle.to_tensor(
            [[1.0, 3.0, 1.0, 5.0], [2.0, 9.0, 6.0, 9.0]], dtype="float32"
        )

        topk_probs, topk_indices = MoETopkFusion.apply(
            gate_probs,
            probs_for_choice,
            2,  # moe_k
            False,  # use_node_limit
            1,  # n_group
            1,  # topk_group
            True,  # norm_gate_logits -> normalized weights returned
        )

        self.assertEqual(topk_indices.numpy().tolist(), [[1, 2], [0, 2]])
        weights = topk_probs.numpy()
        np.testing.assert_allclose(
            weights,
            np.array([[0.75, 0.25], [0.25, 0.75]], dtype="float32"),
            rtol=1e-6,
            atol=1e-6,
        )
        # Defining property of this branch: each token's weights sum to 1.
        np.testing.assert_allclose(
            weights.sum(axis=1),
            np.ones(2, dtype="float32"),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_norm_gate_logits_zero_sum_is_clamped_not_nan(self):
        # Selected gate values sum to exactly 0. The kernel clamps the
        # denominator to 1e-12, so the normalized weights are finite zeros;
        # without the clamp the division would be 0/0 -> NaN.
        #   choice [0.4,0.3,0.2,0.1] -> top-2 idx0(0.4), idx1(0.3)
        #   gate   [0.0,0.0,5.0,5.0] -> picked 0.0, 0.0; sum 0.0
        probs_for_choice = paddle.to_tensor(
            [[0.4, 0.3, 0.2, 0.1]], dtype="float32"
        )
        gate_probs = paddle.to_tensor([[0.0, 0.0, 5.0, 5.0]], dtype="float32")

        topk_probs, topk_indices = MoETopkFusion.apply(
            gate_probs, probs_for_choice, 2, False, 1, 1, True
        )

        self.assertEqual(topk_indices.numpy().tolist(), [[0, 1]])
        weights = topk_probs.numpy()
        self.assertTrue(np.isfinite(weights).all())
        np.testing.assert_array_equal(
            weights, np.zeros((1, 2), dtype="float32")
        )


@unittest.skipUnless(_gpu_ready(), _SKIP_REASON)
class TestMoETopkFusionBackwardNormalized(unittest.TestCase):
    """Backward on the norm_gate_logits=True path (normalized gradient)."""

    def setUp(self):
        paddle.device.set_device("gpu:0")

    def test_normed_backward_matches_normalization_jacobian(self):
        # Same forward selection as the normalized-forward case above:
        #   row0 sigma=4, normed=[0.75,0.25], slots idx1,idx2
        #   row1 sigma=8, normed=[0.25,0.75], slots idx0,idx2
        # The normed backward computes, per token,
        #   grad_i = (grad_out_i - dot(grad_out, normed)) / sigma
        # scattered into the selected slots; unselected slots stay zero.
        #   row0: dot = 1*0.75 + 2*0.25 = 1.25
        #         grad = ([1,2]-1.25)/4 = [-0.0625, 0.1875] at idx1,idx2
        #   row1: dot = 3*0.25 + 4*0.75 = 3.75
        #         grad = ([3,4]-3.75)/8 = [-0.09375, 0.03125] at idx0,idx2
        probs_for_choice = paddle.to_tensor(
            [[0.1, 0.4, 0.3, 0.2], [0.7, 0.2, 0.5, 0.1]], dtype="float32"
        )
        gate_probs = paddle.to_tensor(
            [[1.0, 3.0, 1.0, 5.0], [2.0, 9.0, 6.0, 9.0]], dtype="float32"
        )
        gate_probs.stop_gradient = False

        topk_probs, topk_indices = MoETopkFusion.apply(
            gate_probs, probs_for_choice, 2, False, 1, 1, True
        )
        # Distinguishable upstream grad aligned with the topk slot order.
        upstream = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        topk_probs.backward(upstream)
        grad = gate_probs.grad.numpy()

        expected = np.array(
            [
                [0.0, -0.0625, 0.1875, 0.0],
                [-0.09375, 0.0, 0.03125, 0.0],
            ],
            dtype="float32",
        )
        # Scale-sensitive exact comparison: a 0.5x / 2x factor on the gradient
        # or a dropped -dot term would fail here.
        np.testing.assert_allclose(grad, expected, rtol=1e-5, atol=1e-6)


@unittest.skipUnless(_gpu_ready(), _SKIP_REASON)
class TestRoutingMapPaddingMaskOnly(unittest.TestCase):
    """Routing map with the input_ids padding mask active and no text mask."""

    def setUp(self):
        paddle.device.set_device("gpu:0")

    def test_padding_mask_only_invalidates_pad_rows(self):
        # has_input_ids=True, has_pure_text_mask=False. Row 1 is padding
        # (token == pad_token_id) and must be zeroed out; its indices become
        # the -1 sentinel. Valid rows route to their selected experts.
        gate_probs = paddle.zeros([3, 4], dtype="float32")
        topk_indices = paddle.to_tensor([[0, 2], [1, 3], [2, 3]], dtype="int64")
        input_ids = paddle.to_tensor([5, 0, 7], dtype="int64")

        routing_map, topk_out, dispatch_mask = routing_map_fusion_forward(
            gate_probs,
            topk_indices,
            input_ids=input_ids,
            is_pure_text_line=None,
            pad_token_id=0,
        )

        np.testing.assert_array_equal(
            routing_map.numpy(),
            np.array(
                [[1, 0, 1, 0], [0, 0, 0, 0], [0, 0, 1, 1]], dtype="float32"
            ),
        )
        # Column sums over valid rows only: e0=1, e1=0, e2=2, e3=1.
        np.testing.assert_array_equal(
            dispatch_mask.numpy(), np.array([1, 0, 2, 1], dtype="int64")
        )
        np.testing.assert_array_equal(
            topk_out.numpy(),
            np.array([[0, 2], [-1, -1], [2, 3]], dtype=topk_out.numpy().dtype),
        )


@unittest.skipUnless(_gpu_ready(), _SKIP_REASON)
class TestRoutingMapMultipleExpertTiles(unittest.TestCase):
    """Routing map when n_experts spans more than one BLOCK_N=32 tile."""

    def setUp(self):
        paddle.device.set_device("gpu:0")

    def test_experts_beyond_first_tile_route_and_accumulate(self):
        # n_experts=40 > BLOCK_N(32) -> a second expert-dim tile (base=32)
        # runs. Experts 32,33,35,39 live in that tile; the relative index
        # base subtraction and the cross-tile atomic_add into dispatch must
        # both be correct. All rows valid (no masks).
        n_experts = 40
        seq_len = 3
        gate_probs = paddle.zeros([seq_len, n_experts], dtype="float32")
        selected = [[1, 35], [32, 33], [0, 39]]
        topk_indices = paddle.to_tensor(selected, dtype="int64")

        routing_map, topk_out, dispatch_mask = routing_map_fusion_forward(
            gate_probs, topk_indices
        )

        expected_rm = np.zeros((seq_len, n_experts), dtype="float32")
        expected_disp = np.zeros(n_experts, dtype="int64")
        for row, cols in enumerate(selected):
            for col in cols:
                expected_rm[row, col] = 1.0
                expected_disp[col] += 1

        np.testing.assert_array_equal(routing_map.numpy(), expected_rm)
        np.testing.assert_array_equal(dispatch_mask.numpy(), expected_disp)
        # No masks supplied -> indices pass through unchanged, including the
        # >=32 expert ids that exercise the second tile.
        np.testing.assert_array_equal(
            topk_out.numpy(),
            np.array(selected, dtype=topk_out.numpy().dtype),
        )


if __name__ == "__main__":
    unittest.main()
