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
"""Tests for the ``"hf"`` accumulation order in ``transformer/moe/moe_utils``.

``Qwen3_5MoeExperts`` is a Python loop over experts, so on the reference side a
token's ``top_k`` contributions are combined with one ``index_add_`` per expert:
each add lands in a BF16 buffer and is rounded before the next one. A single
``sum(axis=1)`` is FP32-accumulated and rounds only once, so the two differ in
the last mantissa bit -- and the gradient side additionally differs in *direction*,
because torch accumulates a multiply-used tensor's gradient in reverse
module-creation order (highest expert index first).

Three ``PyLayer``s carry that difference, and since the module-level env flag was
removed they each receive the target as a parameter. The tests therefore check
both the arithmetic and the plumbing: the same call with ``"megatron"`` must stay
on the FP32-reduction branch.
"""

import os
import sys
import unittest

import numpy as np
import paddle

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

from paddlefleet.transformer.moe.moe_utils import (
    ApplyPermutedProbs,
    permute,
    unpermute,
)


def _fixed_topk_routing_map(num_tokens, num_experts, topk, seed=0):
    """A ``[tokens, experts]`` bool map with exactly ``topk`` experts per token."""
    rng = np.random.default_rng(seed)
    m = np.zeros((num_tokens, num_experts), dtype=bool)
    for t in range(num_tokens):
        m[t, rng.choice(num_experts, size=topk, replace=False)] = True
    return paddle.to_tensor(m)


class TestApplyPermutedProbsTargetDispatch(unittest.TestCase):
    """The probs gradient's dtype handling differs between the two references."""

    def setUp(self):
        paddle.seed(20260908)
        self.tokens = paddle.randn([16, 8], dtype=paddle.bfloat16)
        self.probs = paddle.rand([16], dtype=paddle.float32).astype(
            paddle.bfloat16
        )

    def _run(self, target):
        t = self.tokens.detach()
        t.stop_gradient = False
        p = self.probs.detach()
        p.stop_gradient = False
        out = ApplyPermutedProbs.apply(t, p, target)
        g = paddle.randn(out.shape, dtype=paddle.bfloat16)
        gt, gp = paddle.grad([out], [t, p], grad_outputs=[g])
        return out, gt, gp

    def test_forward_is_target_independent(self):
        """Only the backward differs; the forward product must be identical."""
        outs = [
            self._run(t)[0].astype("float32").numpy()
            for t in ("hf", "megatron")
        ]
        np.testing.assert_array_equal(outs[0], outs[1])

    def test_hf_multiplies_in_activation_dtype(self):
        """HF grad equals a BF16 product reduced with an FP32 accumulator."""
        t = self.tokens.detach()
        t.stop_gradient = False
        p = self.probs.detach()
        p.stop_gradient = False
        out = ApplyPermutedProbs.apply(t, p, "hf")
        g = paddle.randn(out.shape, dtype=paddle.bfloat16)
        _, gp = paddle.grad([out], [t, p], grad_outputs=[g])
        with paddle.amp.auto_cast(False):
            expected = (g * self.tokens).sum(axis=-1, dtype="float32")
        np.testing.assert_array_equal(
            gp.astype("float32").numpy(),
            expected.astype(self.probs.dtype).astype("float32").numpy(),
        )
        self.assertEqual(gp.dtype, self.probs.dtype)

    def test_megatron_upcasts_both_operands_first(self):
        """The Megatron branch promotes both operands before multiplying."""
        t = self.tokens.detach()
        t.stop_gradient = False
        p = self.probs.detach()
        p.stop_gradient = False
        out = ApplyPermutedProbs.apply(t, p, "megatron")
        g = paddle.randn(out.shape, dtype=paddle.bfloat16)
        _, gp = paddle.grad([out], [t, p], grad_outputs=[g])
        expected = (self.tokens.cast("float32") * g.cast("float32")).sum(
            axis=-1
        )
        np.testing.assert_array_equal(
            gp.astype("float32").numpy(),
            expected.astype(self.probs.dtype).astype("float32").numpy(),
        )

    def test_hf_and_megatron_probs_grads_differ_in_bf16(self):
        """Guard against the target being ignored: the branches must diverge."""
        paddle.seed(3)
        t = paddle.randn([64, 32], dtype=paddle.bfloat16)
        p = paddle.rand([64], dtype=paddle.float32).astype(paddle.bfloat16)
        g = paddle.randn([64, 32], dtype=paddle.bfloat16)
        grads = []
        for target in ("hf", "megatron"):
            ti = t.detach()
            ti.stop_gradient = False
            pi = p.detach()
            pi.stop_gradient = False
            out = ApplyPermutedProbs.apply(ti, pi, target)
            _, gp = paddle.grad([out], [ti, pi], grad_outputs=[g])
            grads.append(gp.astype("float32").numpy().copy())
        np.testing.assert_allclose(grads[0], grads[1], rtol=5e-2, atol=1e-2)
        self.assertFalse(np.array_equal(grads[0], grads[1]))

    def test_default_target_is_megatron(self):
        """The historical two-argument call must not change meaning."""
        paddle.seed(5)
        t = paddle.randn([8, 4], dtype=paddle.bfloat16)
        p = paddle.rand([8], dtype=paddle.float32).astype(paddle.bfloat16)
        g = paddle.randn([8, 4], dtype=paddle.bfloat16)
        outs = []
        for args in ((t, p), (t, p, "megatron"), (t, p, True)):
            ti = args[0].detach()
            ti.stop_gradient = False
            pi = args[1].detach()
            pi.stop_gradient = False
            call = (ti, pi, *tuple(args[2:]))
            out = ApplyPermutedProbs.apply(*call)
            _, gp = paddle.grad([out], [ti, pi], grad_outputs=[g])
            outs.append(gp.astype("float32").numpy().copy())
        np.testing.assert_array_equal(outs[0], outs[1])
        np.testing.assert_array_equal(outs[0], outs[2])

    def test_grad_tokens_is_target_independent(self):
        """``grad_tokens`` is elementwise, so the target must not touch it."""
        paddle.seed(9)
        t = paddle.randn([8, 4], dtype=paddle.bfloat16)
        p = paddle.rand([8], dtype=paddle.float32).astype(paddle.bfloat16)
        g = paddle.randn([8, 4], dtype=paddle.bfloat16)
        gts = []
        for target in ("hf", "megatron"):
            ti = t.detach()
            ti.stop_gradient = False
            pi = p.detach()
            pi.stop_gradient = False
            out = ApplyPermutedProbs.apply(ti, pi, target)
            gt, _ = paddle.grad([out], [ti, pi], grad_outputs=[g])
            gts.append(gt.astype("float32").numpy().copy())
        np.testing.assert_array_equal(gts[0], gts[1])


class TestUnpermuteHFCombine(unittest.TestCase):
    """``unpermute``'s aligned gather/sum: per-slot BF16 adds vs one FP32 sum."""

    def setUp(self):
        paddle.seed(20260908)
        # ``topk`` has to be large enough for the per-add rounding to be
        # observable: with topk=3 the sequential BF16 adds and the single
        # FP32 reduction happen to agree on every element, so the test
        # would pass without exercising the difference it is about.
        self.num_tokens, self.num_experts, self.topk, self.hidden = (
            16,
            8,
            6,
            64,
        )
        self.routing_map = _fixed_topk_routing_map(
            self.num_tokens, self.num_experts, self.topk, seed=1
        )
        self.tokens = paddle.randn(
            [self.num_tokens, self.hidden], dtype=paddle.bfloat16
        )

    def _roundtrip(self, target):
        tokens_per_expert = self.routing_map.sum(axis=0)
        permuted, sorted_indices = permute(
            self.tokens,
            self.routing_map,
            tokens_per_expert,
            use_accuracy_compatible=target,
        )
        return unpermute(
            permuted,
            sorted_indices,
            restore_shape=self.tokens.shape,
            routing_map=self.routing_map,
            use_accuracy_compatible=target,
        )

    def test_roundtrip_sums_topk_copies(self):
        """Without probs, each token comes back scaled by ``topk``."""
        for target in ("hf", "megatron"):
            with self.subTest(target=target):
                out = self._roundtrip(target)
                np.testing.assert_allclose(
                    out.astype("float32").numpy(),
                    (self.tokens.astype("float32") * self.topk).numpy(),
                    rtol=8e-3,
                    atol=1e-2,
                )

    def test_hf_combine_differs_from_fp32_reduction(self):
        """The per-slot BF16 accumulation must not collapse to one FP32 sum."""
        hf = self._roundtrip("hf").astype("float32").numpy()
        mg = self._roundtrip("megatron").astype("float32").numpy()
        # Same value to within a BF16 ULP of the magnitudes involved, but
        # not bit-identical -- that gap is the whole point of the branch.
        np.testing.assert_allclose(hf, mg, rtol=0, atol=0.25)
        self.assertFalse(np.array_equal(hf, mg))

    def test_output_shape_and_dtype(self):
        out = self._roundtrip("hf")
        self.assertEqual(out.shape, list(self.tokens.shape))
        self.assertEqual(out.dtype, self.tokens.dtype)

    def test_hf_backward_accumulates_in_reverse_slot_order(self):
        """``_PermuteAlignedPyLayer``'s HF backward walks slots high-to-low."""
        tokens_per_expert = self.routing_map.sum(axis=0)
        # One shared upstream gradient: ``paddle.randn`` advances the RNG, so
        # drawing it inside the loop would compare two different problems.
        g = paddle.randn(
            [int(tokens_per_expert.sum()), self.hidden], dtype=paddle.bfloat16
        )
        grads = []
        for target in ("hf", "megatron"):
            t = self.tokens.detach()
            t.stop_gradient = False
            permuted, _ = permute(
                t,
                self.routing_map,
                tokens_per_expert,
                use_accuracy_compatible=target,
            )
            self.assertEqual(permuted.shape, list(g.shape))
            gt = paddle.grad([permuted], [t], grad_outputs=[g])[0]
            grads.append(gt.astype("float32").numpy().copy())
        # A handful of near-zero elements differ by a single BF16 ULP, where a
        # relative tolerance is meaningless; bound the absolute error instead.
        np.testing.assert_allclose(grads[0], grads[1], rtol=0, atol=0.25)
        self.assertFalse(np.array_equal(grads[0], grads[1]))

    def test_hf_backward_masks_padding_rows(self):
        """A token routed nowhere must contribute no gradient on the HF path."""
        m = _fixed_topk_routing_map(8, 4, 2, seed=2).numpy()
        m[3, :] = False
        routing_map = paddle.to_tensor(m)
        tokens = paddle.randn([8, 8], dtype=paddle.bfloat16)
        tokens.stop_gradient = False
        tokens_per_expert = routing_map.sum(axis=0)
        permuted, _ = permute(
            tokens, routing_map, tokens_per_expert, use_accuracy_compatible="hf"
        )
        g = paddle.ones(permuted.shape, dtype=paddle.bfloat16)
        (gt,) = paddle.grad([permuted], [tokens], grad_outputs=[g])
        np.testing.assert_allclose(
            gt[3].astype("float32").numpy(),
            np.zeros(8, dtype=np.float32),
            rtol=0,
            atol=0,
        )

    def test_padding_rows_stay_exactly_zero(self):
        """A token routed to no expert must produce a hard zero, not epsilon."""
        m = _fixed_topk_routing_map(8, 4, 2, seed=2).numpy()
        m[3, :] = False
        routing_map = paddle.to_tensor(m)
        tokens = paddle.randn([8, 8], dtype=paddle.bfloat16)
        tokens_per_expert = routing_map.sum(axis=0)
        permuted, sorted_indices = permute(
            tokens, routing_map, tokens_per_expert, use_accuracy_compatible="hf"
        )
        out = unpermute(
            permuted,
            sorted_indices,
            restore_shape=tokens.shape,
            routing_map=routing_map,
            use_accuracy_compatible="hf",
        )
        np.testing.assert_allclose(
            out[3].astype("float32").numpy(),
            np.zeros(8, dtype=np.float32),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
