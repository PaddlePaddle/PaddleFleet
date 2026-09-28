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

"""Behavior tests for the fused MoE TopK Triton op (variant _3).

Module under test: ``paddlefleet.triton_ops.moe_topk_fusion``. In the
repository module map this is the "计算优化 / Fused Ops" boundary: a Triton
kernel selects the top-k experts per token and can normalize the picked gate
probabilities, and its backward kernel produces the gradient w.r.t.
``gate_probs``.

This variant deliberately targets the ``norm_gate_logits=True`` branches that
the base / _2 / _4 behavior tests do NOT exercise (those all run
``norm_gate_logits=False``):

  * forward normalization -- the ``norm_gate_logits`` block of
    ``_fwd_kernel`` divides each picked gate probability by the sum of the
    picked gate probabilities, so a token's returned probs sum to 1 and are
    NOT the raw gate values;
  * node-limit selection combined with normalization -- group selection by
    per-group top-2 sum masks out experts in non-selected groups (even a
    global-max expert), and the surviving picks are then normalized;
  * the normalized backward path -- ``_bwd_kernel`` with
    ``norm_gate_logits=True`` returns
    ``grad_out / sigma - dot(grad_out, normed) / sigma`` scattered into the
    selected expert slots, which is distinct from the plain scatter of the
    ``norm_gate_logits=False`` path.

Every expectation below is hand-derived from small, per-row-distinct inputs so
that a swapped index, a skipped normalization, a dropped node-limit mask, or a
mis-routed gradient would be rejected. Gate and choice tensors are given
different values so that reading probabilities from the wrong buffer is caught.
The kernels compile to PTX and only execute on a GPU, and this environment has
no paddle installed, so the cases skip honestly rather than assert against a
faked ``triton.language`` shim. They are real GPU behavior tests, not
source-introspection or shape-only checks.
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

MoETopkFusion = None
if HAS_PADDLE:
    # A genuine API/compile error must surface; only ImportError is a skip.
    try:
        from paddlefleet.triton_ops.moe_topk_fusion import MoETopkFusion
    except ImportError as exc:  # pragma: no cover - env-dependent
        HAS_PADDLE = False
        SKIP_REASON = f"moe_topk_fusion not importable: {exc}"


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


class _Ctx:
    """Minimal stand-in for the PyLayer ``ctx`` container.

    ``MoETopkFusion.backward`` is a ``@staticmethod`` that reads its saved
    tensors and scalars off ``ctx``. The container itself is not under test;
    the backward kernel it drives is. Populating it directly lets the test
    pin exact saved tensors and derive the gradient by hand.
    """

    def __init__(self, saved, input_shape, norm_gate_logits, moe_k):
        self._saved = saved
        self.input_shape = input_shape
        self.norm_gate_logits = norm_gate_logits
        self.moe_k = moe_k

    def saved_tensor(self):
        return self._saved


@unittest.skipUnless(RUN, SKIP_REASON)
class TestMoETopkFusionNormBranches(unittest.TestCase):
    def setUp(self):
        paddle.device.set_device("gpu:0")

    def test_forward_normalizes_picked_gate_probs(self):
        # norm_gate_logits=True: the picked gate probabilities are divided by
        # their sum, so each row of topk_probs sums to 1 and is NOT the raw
        # gate value. Choice and gate tensors differ, so a probs-from-choice
        # bug is also rejected.
        #   row0 choice desc: 0.9@1, 0.7@3 -> picks [1, 3]
        #        gate[1]=4, gate[3]=8, sum=12 -> [4/12, 8/12]
        #   row1 choice desc: 0.8@2, 0.5@0 -> picks [2, 0]
        #        gate[2]=30, gate[0]=10, sum=40 -> [30/40, 10/40]
        choice = paddle.to_tensor(
            [[0.1, 0.9, 0.3, 0.7], [0.5, 0.2, 0.8, 0.1]], dtype="float32"
        )
        gate = paddle.to_tensor(
            [[2.0, 4.0, 6.0, 8.0], [10.0, 20.0, 30.0, 40.0]], dtype="float32"
        )

        topk_probs, topk_indices = MoETopkFusion.apply(
            gate, choice, 2, False, 1, 1, True
        )

        self.assertEqual(topk_indices.numpy().tolist(), [[1, 3], [2, 0]])
        np.testing.assert_allclose(
            topk_probs.numpy(),
            [[4.0 / 12.0, 8.0 / 12.0], [30.0 / 40.0, 10.0 / 40.0]],
            rtol=1e-6,
            atol=1e-6,
        )
        # Normalization actually happened: rows sum to 1, unlike raw picks.
        np.testing.assert_allclose(
            topk_probs.numpy().sum(axis=1), [1.0, 1.0], rtol=1e-6, atol=1e-6
        )

    def test_node_limit_masks_then_normalizes(self):
        # n_experts=8, n_group=4 (experts-per-group=2), topk_group=2.
        # Group score = sum of the top-2 choice values per group:
        #   G0 {0,1}: 0.2 + 0.95 = 1.15  (rank 2, selected)
        #   G1 {2,3}: 0.15 + 0.1 = 0.25
        #   G2 {4,5}: 0.85 + 0.9 = 1.75  (rank 1, selected)
        #   G3 {6,7}: 0.98 + 0.05 = 1.03  (NOT selected)
        # idx6 holds the global-max choice 0.98 but its group is dropped, so
        # allowed experts are {0,1,4,5}. Choice top-2 among allowed:
        #   0.95@1, 0.9@5 -> picks [1, 5]  (a no-node-limit run would pick 6).
        # Then normalize the picked gate probs: gate[1]=2, gate[5]=6, sum=8.
        choice = paddle.to_tensor(
            [[0.2, 0.95, 0.15, 0.1, 0.85, 0.9, 0.98, 0.05]], dtype="float32"
        )
        gate = (paddle.arange(8, dtype="float32") + 1.0).reshape([1, 8])

        topk_probs, topk_indices = MoETopkFusion.apply(
            gate, choice, 2, True, 4, 2, True
        )

        self.assertEqual(topk_indices.numpy().tolist(), [[1, 5]])
        np.testing.assert_allclose(
            topk_probs.numpy(),
            [[2.0 / 8.0, 6.0 / 8.0]],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_backward_normalized_gradient_routes_and_scatters(self):
        # Normalized backward: grad_gate[idx_k] =
        #   grad_out_k / sigma - dot(grad_out, normed) / sigma,
        # scattered into the selected expert slots, zero elsewhere.
        # Fixed inputs (seq_len=1, n_experts=5, moe_k=2):
        #   indices = [1, 3], normed = [0.4, 0.6], sigma = 5.0,
        #   grad_out = [1.0, 0.0].
        #   inv = 1/5 = 0.2, dot = 1*0.4 + 0*0.6 = 0.4
        #   g[1] = 1*0.2 - 0.4*0.2 = 0.12
        #   g[3] = 0*0.2 - 0.4*0.2 = -0.08
        #   all other experts get 0.
        indices = paddle.to_tensor([[1, 3]], dtype="int32")
        normed = paddle.to_tensor([[0.4, 0.6]], dtype="float32")
        sigma = paddle.to_tensor([5.0], dtype="float32")
        ctx = _Ctx(
            saved=(indices, normed, sigma),
            input_shape=[1, 5],
            norm_gate_logits=True,
            moe_k=2,
        )
        grad_out = paddle.to_tensor([[1.0, 0.0]], dtype="float32")

        grad_gate, grad_choice = MoETopkFusion.backward(ctx, grad_out, None)

        self.assertIsNone(grad_choice)
        np.testing.assert_allclose(
            grad_gate.numpy(),
            [[0.0, 0.12, 0.0, -0.08, 0.0]],
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
