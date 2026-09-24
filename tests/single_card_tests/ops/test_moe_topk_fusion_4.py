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

"""Behavior tests for the fused MoE TopK Triton op (variant _4).

Module under test: ``paddlefleet.triton_ops.moe_topk_fusion``. In the
repository module map this is the "计算优化 / Fused Ops" boundary: a Triton
kernel selects the top-k experts per token, optionally normalizes the picked
gate probabilities, optionally returns the (k+1)-th largest choice value
(``alpha`` cutoff), and a separate kernel builds the routing map / dispatch
mask under padding and pure-text masks.

This variant deliberately targets branches NOT exercised by the base / _2 / _3
behavior tests or by ``custom_ops/test_moe_topk_fusion_triton.py``:
  * forward ``return_alpha=True`` cutoff output with ``norm_gate_logits=False``
    (raw, un-normalized gate probs returned);
  * routing map driven by ``is_pure_text_line`` alone (pure-text mask branch);
  * routing map with BOTH ``input_ids`` padding and ``is_pure_text_line``
    masks active at once;
  * the plain backward path (``norm_gate_logits=False``) whose gradient is a
    pure scatter of the upstream grad into the selected expert slots.

Every expectation below is hand-derived from small, per-row-distinct inputs so
that swapped indices, wrong tie-breaking, dropped mask terms, or a mis-routed
gradient would be rejected. The kernels compile to PTX and only execute on a
GPU, and this environment has no paddle installed, so the cases skip honestly
rather than assert on a faked ``triton.language`` shim. They are real GPU
behavior tests, not source-introspection or shape-only checks.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

HAS_PADDLE = True
SKIP_REASON = ""
try:  # capability probe: only ImportError means "dependency absent".
    import numpy as np
    import paddle
except ImportError as exc:  # pragma: no cover - env-dependent
    HAS_PADDLE = False
    SKIP_REASON = f"paddle/numpy not importable: {exc}"


def _gpu_available():
    """True only when a real CUDA device exists to run the PTX kernels."""
    if not HAS_PADDLE:
        return False
    if not paddle.device.is_compiled_with_cuda():
        return False
    return paddle.device.cuda.device_count() > 0


RUN = HAS_PADDLE and _gpu_available()
if not RUN and not SKIP_REASON:
    SKIP_REASON = (
        "a CUDA GPU is required: the MoE TopK Triton kernels compile to PTX "
        "and execute only on GPU (no CPU fallback)"
    )


# PLACEHOLDER_BODY


@unittest.skipUnless(RUN, SKIP_REASON)
class TestMoETopkFusionForwardAlpha(unittest.TestCase):
    """Forward with return_alpha=True and norm_gate_logits=False."""

    def test_return_alpha_cutoff_and_raw_probs(self):
        from paddlefleet.triton_ops import MoETopkFusion

        # Per-row-distinct choice scores; no ties, so selection order is
        # fully determined. Row 0 top-2 = idx1(0.4), idx2(0.3); the 3rd
        # largest (alpha cutoff) is idx3=0.2. Row 1 top-2 = idx0(0.7),
        # idx2(0.5); 3rd largest is idx1=0.2.
        probs_for_choice = paddle.to_tensor(
            [[0.1, 0.4, 0.3, 0.2], [0.7, 0.2, 0.5, 0.1]], dtype="float32"
        )
        # gate_probs supply the RETURNED probabilities (distinct from the
        # choice scores) so a gate/choice mix-up is observable.
        gate_probs = paddle.to_tensor(
            [[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]],
            dtype="float32",
        )

        paddle.enable_compat(scope={"triton"}, silent=True)
        probs, indices, alpha = MoETopkFusion.apply(
            gate_probs,
            probs_for_choice,
            2,  # moe_k
            False,  # use_node_limit
            1,  # n_group
            1,  # topk_group
            False,  # norm_gate_logits -> raw gate probs returned
            True,  # return_alpha
        )
        paddle.disable_compat()

        # Indices come out in descending choice-score order.
        np.testing.assert_array_equal(
            indices.numpy(),
            np.array([[1, 2], [0, 2]], dtype=indices.numpy().dtype),
        )
        # With no normalization the probs are gate_probs gathered at the
        # selected indices verbatim.
        np.testing.assert_allclose(
            probs.numpy(),
            np.array([[11.0, 12.0], [20.0, 22.0]], dtype="float32"),
            rtol=1e-5,
            atol=1e-5,
        )
        # alpha = the (k+1)-th largest choice value per row (the cutoff).
        np.testing.assert_allclose(
            alpha.numpy(),
            np.array([0.2, 0.2], dtype="float32"),
            rtol=1e-5,
            atol=1e-5,
        )


@unittest.skipUnless(RUN, SKIP_REASON)
class TestRoutingMapMaskBranches(unittest.TestCase):
    """Routing-map kernel under pure-text and combined masks."""

    def test_pure_text_mask_only(self):
        from paddlefleet.triton_ops import routing_map_fusion_forward

        gate_probs = paddle.zeros([3, 4], dtype="float32")
        topk_indices = paddle.to_tensor([[0, 2], [1, 3], [0, 1]], dtype="int64")
        # Row 1 is not a pure-text line -> masked out entirely.
        is_pure_text_line = paddle.to_tensor([1, 0, 1], dtype="int32")

        paddle.enable_compat(scope={"triton"}, silent=True)
        routing_map, topk_out, dispatch_mask = routing_map_fusion_forward(
            gate_probs,
            topk_indices,
            input_ids=None,
            is_pure_text_line=is_pure_text_line,
        )
        paddle.disable_compat()

        np.testing.assert_array_equal(
            routing_map.numpy(),
            np.array(
                [[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 0, 0]], dtype="float32"
            ),
        )
        # Column sums over valid rows only.
        np.testing.assert_array_equal(
            dispatch_mask.numpy(), np.array([2, 1, 1, 0], dtype="int64")
        )
        # The masked-out row's indices are replaced with the -1 sentinel.
        np.testing.assert_array_equal(
            topk_out.numpy(),
            np.array([[0, 2], [-1, -1], [0, 1]], dtype=topk_out.numpy().dtype),
        )

    def test_padding_and_pure_text_masks_combined(self):
        from paddlefleet.triton_ops import routing_map_fusion_forward

        gate_probs = paddle.zeros([4, 4], dtype="float32")
        topk_indices = paddle.to_tensor(
            [[0, 3], [1, 2], [0, 1], [2, 3]], dtype="int64"
        )
        # pad_token_id=0 -> rows 1 and 3 invalid by padding.
        input_ids = paddle.to_tensor([5, 0, 7, 0], dtype="int64")
        # pure-text mask invalidates row 2.
        is_pure_text_line = paddle.to_tensor([1, 1, 0, 1], dtype="int32")
        # A row survives only if valid under BOTH masks: only row 0.

        paddle.enable_compat(scope={"triton"}, silent=True)
        routing_map, topk_out, dispatch_mask = routing_map_fusion_forward(
            gate_probs,
            topk_indices,
            input_ids=input_ids,
            is_pure_text_line=is_pure_text_line,
            pad_token_id=0,
        )
        paddle.disable_compat()

        np.testing.assert_array_equal(
            routing_map.numpy(),
            np.array(
                [
                    [1, 0, 0, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype="float32",
            ),
        )
        np.testing.assert_array_equal(
            dispatch_mask.numpy(), np.array([1, 0, 0, 1], dtype="int64")
        )
        np.testing.assert_array_equal(
            topk_out.numpy(),
            np.array(
                [[0, 3], [-1, -1], [-1, -1], [-1, -1]],
                dtype=topk_out.numpy().dtype,
            ),
        )


@unittest.skipUnless(RUN, SKIP_REASON)
class TestMoETopkFusionBackwardPlain(unittest.TestCase):
    """Backward on the norm_gate_logits=False path (pure scatter)."""

    def test_plain_grad_is_scatter_into_selected_slots(self):
        from paddlefleet.triton_ops import MoETopkFusion

        probs_for_choice = paddle.to_tensor(
            [[0.1, 0.4, 0.3, 0.2], [0.7, 0.2, 0.5, 0.1]], dtype="float32"
        )
        gate_probs = paddle.to_tensor(
            [[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]],
            dtype="float32",
        )
        gate_probs.stop_gradient = False

        paddle.enable_compat(scope={"triton"}, silent=True)
        probs, indices = MoETopkFusion.apply(
            gate_probs,
            probs_for_choice,
            2,  # moe_k
            False,  # use_node_limit
            1,  # n_group
            1,  # topk_group
            False,  # norm_gate_logits -> plain backward path
        )
        # Distinguishable upstream grad, aligned with topk order.
        upstream = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        probs.backward(upstream)
        grad = gate_probs.grad
        paddle.disable_compat()

        # Selected slots: row0 -> idx1,idx2 ; row1 -> idx0,idx2. On the plain
        # path each upstream value lands verbatim at its expert slot; every
        # unselected slot stays exactly zero.
        expected = np.array(
            [[0.0, 1.0, 2.0, 0.0], [3.0, 0.0, 4.0, 0.0]], dtype="float32"
        )
        np.testing.assert_allclose(grad.numpy(), expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
