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
"""Dtype handling in ``LinearWithGradAccumulationAndAsyncCommunication.backward``.

The Function differentiates its gemms by hand, and its backward runs with AMP
disabled (activation recompute turns AMP off explicitly). So the cast that AMP
would have applied inside the forward gemm has to be reproduced in backward:
a fp32 activation (e.g. the fp32 embedding output produced by
``fp32_residual_connection``) feeding a bf16 weight would otherwise make the
wgrad gemm fail on mixed dtypes. The returned dgrad is cast back to the
original input dtype, because PyLayer requires the gradient dtype to match the
forward input dtype.

The backward is exercised directly with a stub ``ctx`` so that the
sequence-parallel and all-reduce branches can be reached on a single card.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 3)
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np
import paddle

from paddlefleet.tensor_parallel.layers import (
    LinearWithGradAccumulationAndAsyncCommunication as LinearFn,
)

SEQ, BATCH, H_IN, H_OUT = 4, 2, 16, 8


class FakeCtx:
    """Stand-in for the PyLayer context saved by ``forward``.

    ``DEFAULTS`` must stay in sync with every ``ctx.<field> = ...`` assignment
    in ``LinearFn.forward``; ``TestFakeCtxParity`` enforces that so a new field
    upstream fails one explicit test instead of every backward test with an
    opaque ``AttributeError``.
    """

    DEFAULTS = {
        "bf16_input_saved": True,
        "fp8_saved": False,
        "fp8_input_stashed": False,
        "fp8": False,
        "fp8_wgrad": False,
        "main_grad": None,
        "use_bias": False,
        "grad_output_buffer": None,
        "wgrad_deferral_limit": 0,
        "tp_group": None,
        "input_stop_gradient": False,
        "gradient_accumulation_fusion": False,
        "sequence_parallel": False,
        "allreduce_dgrad": False,
        "input_dtype": paddle.float32,
        "input_shape": (SEQ, BATCH, H_IN),
        "use_accuracy_compatible": False,
        "inp_quant_func": None,
        "weight_quant_func": None,
        "use_pow2_scale": False,
        "use_ue8m0": False,
        "save_original_input": True,
    }

    def __init__(self, saved, **overrides):
        self._saved = tuple(saved)
        attrs = dict(self.DEFAULTS)
        attrs.update(overrides)
        for key, value in attrs.items():
            setattr(self, key, value)

    def saved_tensor(self):
        return self._saved


class TestFakeCtxParity(unittest.TestCase):
    """The stub context must expose everything ``forward`` stores."""

    def test_all_forward_ctx_fields_are_stubbed(self):
        import inspect
        import re

        source = inspect.getsource(LinearFn.forward)
        stored = set(re.findall(r"\bctx\.([A-Za-z_][A-Za-z_0-9]*)\s*=", source))
        # set inside the fp8 branches of forward, mirrored in DEFAULTS too
        missing = stored - set(FakeCtx.DEFAULTS)

        self.assertEqual(
            missing,
            set(),
            "FakeCtx.DEFAULTS is missing fields that LinearFn.forward stores "
            f"on ctx: {sorted(missing)}",
        )


def _make_tensors():
    """fp32 activation + bf16 weight: the mixed-dtype case AMP hides."""
    input_fp32 = paddle.randn([SEQ, BATCH, H_IN], dtype="float32")
    weight_bf16 = paddle.randn([H_IN, H_OUT]).astype("bfloat16")
    grad_output = paddle.randn([SEQ, BATCH, H_OUT]).astype("bfloat16")
    return input_fp32, weight_bf16, grad_output


class TestBackwardDtypeCast(unittest.TestCase):
    def test_fp32_activation_with_bf16_weight(self):
        input_fp32, weight, grad_output = _make_tensors()
        ctx = FakeCtx([input_fp32, weight])

        grad_input, grad_weight = LinearFn.backward(ctx, grad_output)

        # dgrad is returned in the forward input dtype ...
        self.assertEqual(grad_input.dtype, paddle.float32)
        self.assertEqual(list(grad_input.shape), [SEQ, BATCH, H_IN])
        # ... while wgrad stays in the weight dtype
        self.assertEqual(grad_weight.dtype, paddle.bfloat16)
        self.assertEqual(list(grad_weight.shape), [H_IN, H_OUT])

        expected = paddle.matmul(grad_output, weight.t()).astype("float32")
        np.testing.assert_allclose(
            grad_input.numpy(), expected.numpy(), rtol=1e-2, atol=1e-2
        )

    def test_matching_dtypes_are_left_alone(self):
        input_bf16 = paddle.randn([SEQ, BATCH, H_IN]).astype("bfloat16")
        weight = paddle.randn([H_IN, H_OUT]).astype("bfloat16")
        grad_output = paddle.randn([SEQ, BATCH, H_OUT]).astype("bfloat16")
        ctx = FakeCtx([input_bf16, weight], input_dtype=paddle.bfloat16)

        grad_input, grad_weight = LinearFn.backward(ctx, grad_output)

        self.assertEqual(grad_input.dtype, paddle.bfloat16)
        self.assertEqual(grad_weight.dtype, paddle.bfloat16)

    def test_deferred_wgrad_publishes_main_grad(self):
        """gradient_accumulation_fusion still binds main_grad to the weight."""
        input_fp32, weight, grad_output = _make_tensors()
        main_grad = paddle.zeros([H_IN, H_OUT], dtype="float32")
        buffer = []
        ctx = FakeCtx(
            [input_fp32, weight],
            gradient_accumulation_fusion=True,
            main_grad=main_grad,
            grad_output_buffer=buffer,
            wgrad_deferral_limit=0,
        )

        grad_input, grad_weight = LinearFn.backward(ctx, grad_output)

        self.assertIs(weight.main_grad, main_grad)
        # wgrad was deferred: grad_output stashed, no weight gradient produced
        self.assertEqual(len(buffer), 1)
        self.assertIsNone(grad_weight)
        self.assertEqual(grad_input.dtype, paddle.float32)


class TestBackwardCollectiveBranches(unittest.TestCase):
    """SP reduce-scatter and dgrad all-reduce also cast back to input dtype."""

    def _memory_buffer_patch(self):
        buffer = MagicMock()
        buffer.get_tensor = MagicMock(
            side_effect=lambda shape, dtype, name: paddle.zeros(
                shape, dtype=dtype
            )
        )
        return patch(
            "paddlefleet.tensor_parallel.layers.get_global_memory_buffer",
            return_value=buffer,
        )

    def test_sequence_parallel_sub_grad_input_cast(self):
        input_fp32, weight, grad_output = _make_tensors()
        bias_present = True
        ctx = FakeCtx(
            [input_fp32, weight],
            sequence_parallel=True,
            use_bias=bias_present,
            tp_group=MagicMock(world_size=1),
        )

        with (
            self._memory_buffer_patch(),
            patch("paddlefleet.tensor_parallel.layers.dist") as mock_dist,
            patch(
                "paddlefleet.tensor_parallel.layers._reduce_scatter_base"
            ) as mock_rs,
        ):
            mock_dist.stream.all_gather.return_value = MagicMock()
            mock_rs.return_value = MagicMock()
            grad_input, grad_weight, grad_bias = LinearFn.backward(
                ctx, grad_output
            )

        mock_rs.assert_called_once()
        self.assertEqual(grad_input.dtype, paddle.float32)
        self.assertEqual(list(grad_input.shape), [SEQ, BATCH, H_IN])
        self.assertEqual(list(grad_weight.shape), [H_IN, H_OUT])
        self.assertEqual(list(grad_bias.shape), [H_OUT])

    def test_sequence_parallel_without_bias(self):
        input_bf16 = paddle.randn([SEQ, BATCH, H_IN]).astype("bfloat16")
        weight = paddle.randn([H_IN, H_OUT]).astype("bfloat16")
        grad_output = paddle.randn([SEQ, BATCH, H_OUT]).astype("bfloat16")
        ctx = FakeCtx(
            [input_bf16, weight],
            sequence_parallel=True,
            input_dtype=paddle.bfloat16,
            tp_group=MagicMock(world_size=1),
        )

        with (
            self._memory_buffer_patch(),
            patch("paddlefleet.tensor_parallel.layers.dist") as mock_dist,
            patch(
                "paddlefleet.tensor_parallel.layers._reduce_scatter_base"
            ) as mock_rs,
        ):
            mock_dist.stream.all_gather.return_value = MagicMock()
            mock_rs.return_value = MagicMock()
            outputs = LinearFn.backward(ctx, grad_output)

        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].dtype, paddle.bfloat16)

    def test_allreduce_dgrad_cast(self):
        input_fp32, weight, grad_output = _make_tensors()
        ctx = FakeCtx(
            [input_fp32, weight],
            allreduce_dgrad=True,
            tp_group=MagicMock(world_size=1),
        )

        with patch("paddlefleet.tensor_parallel.layers.dist") as mock_dist:
            mock_dist.all_reduce.return_value = MagicMock()
            grad_input, grad_weight = LinearFn.backward(ctx, grad_output)

        mock_dist.all_reduce.assert_called_once()
        self.assertEqual(grad_input.dtype, paddle.float32)
        self.assertEqual(list(grad_weight.shape), [H_IN, H_OUT])


if __name__ == "__main__":
    unittest.main()
