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

"""Unit tests for the expert GEMM paths of ``transformer.moe.moe_expert``.

``GroupedMLPExpert.forward`` picks one of four kernels: the per-expert TN
GEMM used under ``use_accuracy_compatible``, the DeepGEMM grouped BMM, the
plain grouped BMM, and a zero-token path that only exists to keep the
expert weights in the graph. Each is driven here with small real tensors,
with the two grouped-BMM PyLayers replaced by recording stubs that compute
the same grouped reference (the fused kernels need Blackwell / a built
paddlefleet_ops, and this is a single-card unit test).

``_UACExpertFp32WgradCapture`` is tested on its own: it is an identity in
the forward direction and its only effect is the fp32 ``X.T @ dY`` it
deposits into ``weight.main_grad[expert]``.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import paddle
import paddle.nn.functional as F

from paddlefleet.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
)
from paddlefleet.transformer.moe import moe_expert as me
from paddlefleet.transformer.transformer_config import TransformerConfig

HIDDEN = 3
INTER = 2
EXPERTS = 2


def _param(array):
    array = np.asarray(array, dtype="float32")
    return paddle.create_parameter(
        shape=list(array.shape),
        dtype="float32",
        default_initializer=paddle.nn.initializer.Assign(array),
    )


def _ramp(*shape, lo=-1.0, hi=1.5):
    total = int(np.prod(shape))
    return np.linspace(lo, hi, total, dtype="float32").reshape(shape)


def _tensor(array, stop_gradient=False):
    out = paddle.to_tensor(np.asarray(array, dtype="float32"))
    out.stop_gradient = stop_gradient
    return out


def _grouped_reference(hidden_states, weight1, weight2, tokens, probs=None):
    """act(x @ w1[e]) [* probs] @ w2[e], concatenated over the experts."""
    parts = []
    start = 0
    for expert, count in enumerate(tokens):
        if count == 0:
            continue
        block = hidden_states[start : start + count]
        inner = F.relu(paddle.matmul(block, weight1[expert]))
        if probs is not None:
            inner = inner * probs[start : start + count].unsqueeze(-1)
        parts.append(paddle.matmul(inner, weight2[expert]))
        start += count
    return paddle.concat(parts, axis=0)


class UacExpertWgradCaptureTest(unittest.TestCase):
    """``_UACExpertFp32WgradCapture``: identity forward, fp32 wgrad side."""

    def setUp(self):
        self.weight = _param(_ramp(EXPERTS, HIDDEN, INTER))
        self.weight.main_grad = None
        self.weight.grad_added_to_main_grad = False

    def test_forward_returns_the_gemm_output_unchanged(self):
        x = _tensor(_ramp(2, HIDDEN))
        # The capture always wraps a GEMM output, never a leaf tensor.
        gemm_out = _tensor(_ramp(2, INTER)) * 1.0

        out = me._UACExpertFp32WgradCapture.apply(gemm_out, x, self.weight, 1)

        np.testing.assert_array_equal(out.numpy(), gemm_out.numpy())

    def test_backward_passes_dy_through_and_fills_main_grad(self):
        x = _tensor(_ramp(2, HIDDEN))
        upstream = _tensor(_ramp(2, INTER))
        gemm_out = upstream * 1.0

        out = me._UACExpertFp32WgradCapture.apply(gemm_out, x, self.weight, 1)
        (out * 2.0).sum().backward()

        # dY reaches the upstream tensor unchanged (E-476: reconstructing dX
        # here regressed the alignment), ...
        np.testing.assert_array_equal(
            upstream.grad.numpy(), np.full((2, INTER), 2.0, dtype="float32")
        )
        # ... and the only side effect is fp32 X.T @ dY on expert 1.
        expected = _ramp(2, HIDDEN).T @ np.full((2, INTER), 2.0, "float32")
        self.assertEqual(self.weight.main_grad.dtype, paddle.float32)
        np.testing.assert_allclose(
            self.weight.main_grad.numpy()[1], expected, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_array_equal(
            self.weight.main_grad.numpy()[0],
            np.zeros((HIDDEN, INTER), dtype="float32"),
        )
        self.assertTrue(self.weight.grad_added_to_main_grad)

    def test_backward_accumulates_into_an_existing_buffer(self):
        self.weight.main_grad = paddle.ones(self.weight.shape, dtype="float32")
        x = _tensor(_ramp(2, HIDDEN))
        gemm_out = _tensor(_ramp(2, INTER)) * 1.0

        out = me._UACExpertFp32WgradCapture.apply(gemm_out, x, self.weight, 0)
        out.sum().backward()

        expected = 1.0 + _ramp(2, HIDDEN).T @ np.ones(
            (2, INTER), dtype="float32"
        )
        np.testing.assert_allclose(
            self.weight.main_grad.numpy()[0], expected, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_array_equal(
            self.weight.main_grad.numpy()[1],
            np.ones((HIDDEN, INTER), dtype="float32"),
        )

    def test_empty_shard_returns_the_grad_without_touching_main_grad(self):
        ctx = SimpleNamespace(
            saved_tensor=lambda: (paddle.zeros([0, HIDDEN], "float32"),),
            weight=self.weight,
            expert_index=0,
        )
        dy = paddle.ones([0, INTER], dtype="float32")

        grads = me._UACExpertFp32WgradCapture.backward(ctx, dy)

        self.assertIs(grads[0], dy)
        self.assertEqual(grads[1:], (None, None))
        # No buffer is allocated for a shard that owns no token.
        self.assertIsNone(self.weight.main_grad)
        self.assertFalse(self.weight.grad_added_to_main_grad)


class ExpertForwardTestBase(unittest.TestCase):
    """Shared float32 stub of the parts of ``GroupedMLPExpert`` used."""

    def setUp(self):
        self.weight1 = _param(_ramp(EXPERTS, HIDDEN, INTER))
        self.weight2 = _param(_ramp(EXPERTS, INTER, HIDDEN, lo=0.5, hi=-1.0))
        self.weight1.main_grad = None
        self.weight2.main_grad = None

    def _expert(
        self,
        *,
        use_accuracy_compatible=True,
        moe_deep_gemm=False,
        activation_recompute=False,
    ):
        return SimpleNamespace(
            config=SimpleNamespace(
                use_accuracy_compatible=use_accuracy_compatible
            ),
            weight1=self.weight1,
            weight2=self.weight2,
            activation_func=F.relu,
            activation_recompute=activation_recompute,
            moe_deep_gemm=moe_deep_gemm,
        )

    def _forward(self, expert, hidden_states, tokens, **kwargs):
        return me.GroupedMLPExpert.forward(
            expert,
            hidden_states,
            paddle.to_tensor(tokens, dtype="int64"),
            **kwargs,
        )


class GemmTnExpertPathTest(ExpertForwardTestBase):
    """Per-expert TN GEMM, the path ``use_accuracy_compatible`` selects."""

    def test_per_expert_blocks_match_the_grouped_reference(self):
        hidden_states = _tensor(_ramp(5, HIDDEN))

        out, bias = self._forward(self._expert(), hidden_states, [2, 3])

        self.assertIsNone(bias)
        self.assertEqual(list(out.shape), [5, HIDDEN])
        np.testing.assert_allclose(
            out.numpy(),
            _grouped_reference(
                hidden_states, self.weight1, self.weight2, [2, 3]
            ).numpy(),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_expert_without_tokens_is_skipped(self):
        hidden_states = _tensor(_ramp(3, HIDDEN))

        out, _ = self._forward(self._expert(), hidden_states, [0, 3])

        self.assertEqual(list(out.shape), [3, HIDDEN])
        np.testing.assert_allclose(
            out.numpy(),
            _grouped_reference(
                hidden_states, self.weight1, self.weight2, [0, 3]
            ).numpy(),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_permuted_probs_are_folded_before_fc2(self):
        hidden_states = _tensor(_ramp(4, HIDDEN))
        probs = _tensor([0.5, 2.0, -1.5, 0.25])

        out, _ = self._forward(
            self._expert(), hidden_states, [1, 3], permuted_probs=probs
        )

        expected = _grouped_reference(
            hidden_states, self.weight1, self.weight2, [1, 3], probs=probs
        )
        np.testing.assert_allclose(
            out.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6
        )
        self.assertEqual(out.dtype, hidden_states.dtype)

    def test_row_owner_splits_each_expert_without_moving_the_value(self):
        hidden_states = _tensor(_ramp(5, HIDDEN))
        # Expert 0 owns rows 0-1 (both on shard 0), expert 1 owns rows 2-4
        # split over shard 0 and shard 1.
        row_owner = paddle.to_tensor([0, 0, 0, 1, 1], dtype="int32")

        split, _ = self._forward(
            self._expert(), hidden_states, [2, 3], row_owner=row_owner
        )
        unsplit, _ = self._forward(self._expert(), hidden_states, [2, 3])

        self.assertEqual(list(split.shape), [5, HIDDEN])
        np.testing.assert_allclose(
            split.numpy(), unsplit.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_row_owner_splits_carry_their_probs_slice(self):
        hidden_states = _tensor(_ramp(4, HIDDEN))
        probs = _tensor([0.5, 2.0, -1.5, 0.25])
        row_owner = paddle.to_tensor([0, 0, 1, 1], dtype="int32")

        out, _ = self._forward(
            self._expert(),
            hidden_states,
            [1, 3],
            permuted_probs=probs,
            row_owner=row_owner,
        )

        expected = _grouped_reference(
            hidden_states, self.weight1, self.weight2, [1, 3], probs=probs
        )
        np.testing.assert_allclose(
            out.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_both_gemms_capture_their_fp32_expert_wgrad(self):
        hidden_states = _tensor(_ramp(3, HIDDEN))

        out, _ = self._forward(self._expert(), hidden_states, [1, 2])
        out.sum().backward()

        # One capture per GEMM per expert, so both parameters own a fp32
        # buffer with a non-zero row for each expert that saw tokens.
        for weight in (self.weight1, self.weight2):
            self.assertIsNotNone(weight.main_grad)
            self.assertEqual(weight.main_grad.dtype, paddle.float32)
            self.assertEqual(list(weight.main_grad.shape), weight.shape)
            for expert in range(EXPERTS):
                self.assertGreater(
                    float(paddle.abs(weight.main_grad[expert]).sum().item()),
                    0.0,
                )
        # The fc2 capture of expert 0 is exactly its own fc2 input
        # transposed times the incoming dY (ones, from ``out.sum()``).
        fc2_input = F.relu(
            paddle.matmul(_tensor(_ramp(3, HIDDEN))[:1], self.weight1[0])
        )
        expected = paddle.matmul(
            fc2_input.t(), paddle.ones([1, HIDDEN], dtype="float32")
        )
        np.testing.assert_allclose(
            self.weight2.main_grad.numpy()[0],
            expected.numpy(),
            rtol=1e-6,
            atol=1e-6,
        )


def _grouped_bmm_stub(x, weight, batch_sizes, *unused):
    """Grouped BMM reference standing in for the fused PyLayers."""
    if isinstance(batch_sizes, paddle.Tensor):
        counts = [int(value) for value in batch_sizes.numpy().tolist()]
    else:
        counts = [int(value) for value in batch_sizes]
    parts = []
    start = 0
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        parts.append(paddle.matmul(x[start : start + count], weight[expert]))
        start += count
    return paddle.concat(parts, axis=0)


class GroupedBmmExpertPathTest(ExpertForwardTestBase):
    """Plain grouped BMM path (``use_accuracy_compatible`` off)."""

    def test_runtime_weights_and_restore_hooks_reach_the_gemms(self):
        hidden_states = _tensor(_ramp(5, HIDDEN))
        probs = _tensor([0.5, 2.0, -1.5, 0.25, 1.0])
        restore1, restore2 = (lambda: None), (lambda: None)
        runtime = me.RuntimeExpertWeights(
            tensors=(self.weight1, self.weight2),
            restore_before_backward=(restore1, restore2),
        )

        with patch.object(me, "BMMFunction") as bmm:
            bmm.apply.side_effect = _grouped_bmm_stub
            out, bias = self._forward(
                self._expert(use_accuracy_compatible=False),
                hidden_states,
                [2, 3],
                expert_weights=runtime,
                permuted_probs=probs,
            )

        self.assertIsNone(bias)
        self.assertEqual(bmm.apply.call_count, 2)
        first, second = bmm.apply.call_args_list
        self.assertIs(first.args[1], self.weight1)
        self.assertIs(second.args[1], self.weight2)
        self.assertEqual(first.args[2], [2, 3])
        # Each GEMM gets the restore hook of the weight it consumes.
        self.assertIs(first.args[4], restore1)
        self.assertIs(second.args[4], restore2)
        np.testing.assert_allclose(
            out.numpy(),
            _grouped_reference(
                hidden_states, self.weight1, self.weight2, [2, 3], probs=probs
            ).numpy(),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_plain_weight_tuple_is_used_as_is(self):
        hidden_states = _tensor(_ramp(3, HIDDEN))
        other1 = _param(_ramp(EXPERTS, HIDDEN, INTER, lo=0.25, hi=2.0))
        other2 = _param(_ramp(EXPERTS, INTER, HIDDEN, lo=-0.75, hi=1.0))

        with patch.object(me, "BMMFunction") as bmm:
            bmm.apply.side_effect = _grouped_bmm_stub
            out, _ = self._forward(
                self._expert(use_accuracy_compatible=False),
                hidden_states,
                [1, 2],
                expert_weights=(other1, other2),
            )

        first, second = bmm.apply.call_args_list
        self.assertIs(first.args[1], other1)
        self.assertIs(second.args[1], other2)
        # No restore hook exists on the bare tuple form.
        self.assertIsNone(first.args[4])
        np.testing.assert_allclose(
            out.numpy(),
            _grouped_reference(hidden_states, other1, other2, [1, 2]).numpy(),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_activation_recompute_is_refused(self):
        hidden_states = _tensor(_ramp(3, HIDDEN))

        with patch.object(me, "BMMFunction") as bmm:
            bmm.apply.side_effect = _grouped_bmm_stub
            with self.assertRaisesRegex(
                NotImplementedError, "Recompute in GroupedMLPExpert"
            ):
                self._forward(
                    self._expert(
                        use_accuracy_compatible=False,
                        activation_recompute=True,
                    ),
                    hidden_states,
                    [1, 2],
                )


class DeepGemmExpertPathTest(ExpertForwardTestBase):
    """DeepGEMM grouped BMM path (``moe_deep_gemm`` on)."""

    def test_token_counts_are_passed_as_int32_and_probs_folded(self):
        hidden_states = _tensor(_ramp(4, HIDDEN))
        probs = _tensor([0.5, 2.0, -1.5, 0.25])

        with patch.object(me, "DeepGEMMBMMFunction") as deep_gemm:
            deep_gemm.apply.side_effect = _grouped_bmm_stub
            out, bias = self._forward(
                self._expert(use_accuracy_compatible=False, moe_deep_gemm=True),
                hidden_states,
                [1, 3],
                permuted_probs=probs,
            )

        self.assertIsNone(bias)
        self.assertEqual(deep_gemm.apply.call_count, 2)
        for call in deep_gemm.apply.call_args_list:
            batch_sizes = call.args[2]
            self.assertEqual(batch_sizes.dtype, paddle.int32)
            self.assertEqual(batch_sizes.numpy().tolist(), [1, 3])
        np.testing.assert_allclose(
            out.numpy(),
            _grouped_reference(
                hidden_states, self.weight1, self.weight2, [1, 3], probs=probs
            ).numpy(),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_activation_recompute_is_refused(self):
        hidden_states = _tensor(_ramp(3, HIDDEN))

        with patch.object(me, "DeepGEMMBMMFunction") as deep_gemm:
            deep_gemm.apply.side_effect = _grouped_bmm_stub
            with self.assertRaisesRegex(
                NotImplementedError, "Recompute in GroupedMLPExpert"
            ):
                self._forward(
                    self._expert(
                        use_accuracy_compatible=False,
                        moe_deep_gemm=True,
                        activation_recompute=True,
                    ),
                    hidden_states,
                    [1, 2],
                )


class NoLocalTokenPathTest(ExpertForwardTestBase):
    """Zero-token path: shapes only, but the weights must stay in the graph."""

    def test_probs_are_applied_and_both_weights_keep_a_gradient(self):
        hidden_states = _tensor(np.zeros((0, HIDDEN), dtype="float32"))
        probs = _tensor(np.zeros((0,), dtype="float32"))

        out, bias = self._forward(
            self._expert(use_accuracy_compatible=False),
            hidden_states,
            [0, 0],
            permuted_probs=probs,
        )
        out.sum().backward()

        self.assertIsNone(bias)
        self.assertEqual(list(out.shape), [0, HIDDEN])
        self.assertEqual(out.dtype, hidden_states.dtype)
        for weight in (self.weight1, self.weight2):
            self.assertIsNotNone(weight.grad)
            self.assertEqual(list(weight.grad.shape), weight.shape)
            np.testing.assert_array_equal(
                weight.grad.numpy(),
                np.zeros(weight.shape, dtype="float32"),
            )


class AccuracyCompatibleConstructionTest(unittest.TestCase):
    """The UAC constructor claims ``main_grad`` on both expert weights."""

    def _expert(self, use_accuracy_compatible):
        model_parallel_cuda_manual_seed(2026)
        config = TransformerConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=8,
            moe_intermediate_size=4,
            gated_linear_unit=True,
            hidden_act=F.silu,
            params_dtype="bfloat16",
            use_accuracy_compatible=use_accuracy_compatible,
        )
        return me.GroupedMLPExpert(
            num_local_experts=EXPERTS, config=config, moe_deep_gemm=False
        )

    def test_accuracy_compatible_claims_both_buffers(self):
        expert = self._expert(True)

        # Claimed but empty: the fp32 capture allocates on first use, and
        # MixPrecision skips a Parameter that already owns main_grad.
        self.assertIsNone(expert.weight1.main_grad)
        self.assertIsNone(expert.weight2.main_grad)

    def test_default_leaves_the_buffers_to_mixprecision(self):
        expert = self._expert(False)

        self.assertFalse(hasattr(expert.weight1, "main_grad"))
        self.assertFalse(hasattr(expert.weight2, "main_grad"))


if __name__ == "__main__":
    unittest.main()
