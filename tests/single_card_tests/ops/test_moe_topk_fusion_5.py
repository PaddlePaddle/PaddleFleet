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

"""Behavior tests for the fused MoE TopK Triton op (variant _5).

Module under test: ``paddlefleet.triton_ops.moe_topk_fusion`` (repository
module map: "计算优化 / Fused Ops"). A Triton kernel selects the top-k experts
per token, optionally normalizes the picked gate probabilities, optionally
returns the (k+1)-th largest choice value (``alpha`` cutoff); a separate
kernel builds the routing map / dispatch mask under padding / pure-text masks.

This variant deliberately targets branches NOT exercised by the base / _2 / _3
/ _4 behavior tests nor by ``custom_ops/test_moe_topk_fusion_triton.py``:

  * The NORMALIZED backward path (``norm_gate_logits=True`` in ``_bwd_kernel``):
    the gradient is the Jacobian of the softmax-free normalization
    ``p_i = g_i / S`` contracted with the upstream grad, i.e.
    ``dL/dg_j = grad_out_j / S - (sum_m grad_out_m * p_m) / S`` scattered into
    the selected expert slots and zero elsewhere. (_4 only covers the plain
    ``norm_gate_logits=False`` scatter backward.)
  * ``return_alpha=True`` combined with ``norm_gate_logits=True`` forward:
    ``alpha`` is read from the choice values *before* normalization, so it is
    unaffected by it, while the returned probs are normalized to sum 1.
    (_2 and _4 both use ``norm_gate_logits=False`` with alpha.)
  * Routing map driven by ``input_ids`` padding ALONE with a NON-default
    ``pad_token_id`` (the ``has_input_ids`` branch and the ``pad_token_id``
    runtime argument). (_4 covers pure-text alone and the both-masks case.)

Every expected value below is hand-derived from small, per-row-distinct inputs
so swapped indices, a wrong normalization Jacobian, a dropped mask term, or a
mis-consumed ``pad_token_id`` would be rejected. These kernels compile to PTX
and execute only on a GPU; there is no CPU implementation and no import-time
pure-Python logic observable without ``paddle``+``triton``. When paddle/triton
are missing, or no CUDA GPU with an active Triton runtime is present, the suite
skips honestly rather than fake-passing on CPU.
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
    # A genuine API/compile error must surface; only ImportError is a
    # "missing dependency" skip reason.
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
        # No usable GPU / Triton runtime -> genuine hardware absence.
        return False
    return True


_SKIP_REASON = (
    "fused MoE topk kernels require paddle + triton and a CUDA GPU with an "
    "active Triton runtime; there is no CPU implementation to validate "
    "(import error: {})".format(_IMPORT_ERR or "none")
)


def _normalized_grad_reference(gate_row, sel_idx, dy_row):
    """Independent reference for the normalized backward gradient of one row.

    p = gate[sel] / S, and dL/dg_j = dy_j / S - (sum_m dy_m p_m) / S for the
    selected slots, 0 elsewhere. Derived from the normalization Jacobian, not
    from the production backward kernel.
    """
    gate_row = np.asarray(gate_row, dtype=np.float64)
    dy_row = np.asarray(dy_row, dtype=np.float64)
    picked = gate_row[sel_idx]
    s = picked.sum()
    p = picked / s
    dot = float((dy_row * p).sum())
    grad = np.zeros_like(gate_row)
    for k, j in enumerate(sel_idx):
        grad[j] = dy_row[k] / s - dot / s
    return grad


@unittest.skipUnless(_gpu_ready(), _SKIP_REASON)
class TestMoETopkFusionBranches5(unittest.TestCase):
    def setUp(self):
        paddle.device.set_device("gpu:0")

    def test_return_alpha_with_normalization(self):
        # return_alpha=True AND norm_gate_logits=True. alpha comes from the
        # choice values before normalization, so it is unaffected; the probs
        # are the gate values at the selected slots, normalized to sum 1.
        #   row0 choice desc: 0.90@3, 0.70@5, 0.50@1, ... -> top-2 {3,5},
        #         alpha = 3rd largest choice = 0.50; gate[3,5]=0.44,0.66,
        #         S=1.10 -> normalized [0.4, 0.6].
        #   row1 choice desc: 0.80@0, 0.65@2, 0.45@4, ... -> top-2 {0,2},
        #         alpha = 0.45; gate[0,2]=0.66,0.44, S=1.10 -> [0.6, 0.4].
        gate_probs = paddle.to_tensor(
            [
                [0.11, 0.22, 0.33, 0.44, 0.55, 0.66],
                [0.66, 0.55, 0.44, 0.33, 0.22, 0.11],
            ],
            dtype="float32",
        )
        probs_for_choice = paddle.to_tensor(
            [
                [0.10, 0.50, 0.30, 0.90, 0.20, 0.70],
                [0.80, 0.15, 0.65, 0.05, 0.45, 0.25],
            ],
            dtype="float32",
        )

        topk_probs, topk_indices, alpha = MoETopkFusion.apply(
            gate_probs, probs_for_choice, 2, False, 1, 1, True, True
        )

        self.assertEqual(topk_indices.dtype, paddle.int64)
        self.assertEqual(topk_indices.numpy().tolist(), [[3, 5], [0, 2]])
        np.testing.assert_allclose(
            topk_probs.numpy(),
            [[0.4, 0.6], [0.6, 0.4]],
            rtol=1e-5,
            atol=1e-5,
        )
        # Normalized rows sum to 1.
        np.testing.assert_allclose(
            topk_probs.numpy().sum(axis=-1), [1.0, 1.0], rtol=1e-6, atol=1e-6
        )
        # alpha unchanged by normalization: still the (k+1)-th largest choice.
        np.testing.assert_allclose(
            alpha.numpy(), [0.50, 0.45], rtol=1e-6, atol=1e-6
        )

    def test_backward_normalized_gradient_matches_jacobian(self):
        # Normalized backward path: grad wrt gate_probs is the normalization
        # Jacobian contracted with the fixed upstream grad, scattered into the
        # selected slots and exactly zero elsewhere. A plain-scatter bug (using
        # the unnormalized path) would place dy directly and fail here.
        gate_np = np.array(
            [
                [0.11, 0.22, 0.33, 0.44, 0.55, 0.66],
                [0.66, 0.55, 0.44, 0.33, 0.22, 0.11],
            ],
            dtype=np.float32,
        )
        choice_np = np.array(
            [
                [0.10, 0.50, 0.30, 0.90, 0.20, 0.70],
                [0.80, 0.15, 0.65, 0.05, 0.45, 0.25],
            ],
            dtype=np.float32,
        )
        gate_probs = paddle.to_tensor(gate_np)
        gate_probs.stop_gradient = False
        probs_for_choice = paddle.to_tensor(choice_np)

        topk_probs, _ = MoETopkFusion.apply(
            gate_probs, probs_for_choice, 2, False, 1, 1, True
        )

        # Fixed, per-position-distinct upstream gradient.
        dy = paddle.to_tensor([[0.1, -0.2], [0.3, 0.05]], dtype="float32")
        topk_probs.backward(dy)
        grad = gate_probs.grad.numpy()

        # Selection order is by descending choice value.
        row0 = _normalized_grad_reference(gate_np[0], [3, 5], [0.1, -0.2])
        row1 = _normalized_grad_reference(gate_np[1], [0, 2], [0.3, 0.05])
        expected = np.stack([row0, row1]).astype(np.float64)

        np.testing.assert_allclose(grad, expected, rtol=1e-4, atol=1e-6)
        # Non-selected experts get exactly zero gradient.
        non_selected = np.ones_like(gate_np, dtype=bool)
        non_selected[0, [3, 5]] = False
        non_selected[1, [0, 2]] = False
        np.testing.assert_array_equal(
            grad[non_selected], np.zeros(non_selected.sum())
        )

    def test_routing_map_padding_only_with_nonzero_pad_token_id(self):
        # has_input_ids branch with a NON-default pad_token_id=7: rows whose
        # input id equals 7 are padding -> routing all zero and indices -1;
        # valid rows pass through. dispatch_mask is the column sum over valid
        # rows only.
        gate_probs = paddle.ones([4, 6], dtype="float32")
        topk_indices = paddle.to_tensor(
            [[0, 3], [1, 4], [2, 5], [0, 1]], dtype="int64"
        )
        input_ids = paddle.to_tensor([7, 2, 7, 5], dtype="int64")

        routing_map, topk_indices_out, dispatch_mask = (
            routing_map_fusion_forward(
                gate_probs,
                topk_indices,
                input_ids=input_ids,
                pad_token_id=7,
            )
        )

        expected_routing = np.zeros((4, 6), dtype=np.float32)
        expected_routing[1, [1, 4]] = 1.0
        expected_routing[3, [0, 1]] = 1.0
        np.testing.assert_array_equal(routing_map.numpy(), expected_routing)
        self.assertEqual(
            topk_indices_out.numpy().tolist(),
            [[-1, -1], [1, 4], [-1, -1], [0, 1]],
        )
        # expert0: row3; expert1: row1+row3; expert4: row1.
        self.assertEqual(dispatch_mask.numpy().tolist(), [1, 2, 0, 0, 1, 0])


if __name__ == "__main__":
    unittest.main()
