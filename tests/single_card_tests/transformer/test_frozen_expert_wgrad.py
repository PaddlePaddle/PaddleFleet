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

"""Guards for skipping expert weight gradients when the experts are frozen.

MoE expert weight gradients are written straight into ``main_grad`` / ``grad``
instead of being returned through autograd, so ``stop_gradient`` is not honored
automatically and every hand-written backward has to check it. DSv4 phase 2
(``csa_train_indexer_only``) freezes the whole backbone, which makes those wgrad
GEMMs and their fp32 buffers pure waste.

The subtlety these tests pin down is *where* ``stop_gradient`` may be trusted:

* On a PyLayer's own forward inputs (``ctx.saved_tensor()`` round-trips the flag
  faithfully) it is reliable.
* On anything else it is not. A plain tensor defaults to
  ``stop_gradient=True``, and so does a ``_slice()`` view of a **trainable**
  parameter -- which is exactly what the sliced / subbatch deep_gemm path hands
  to ``bf16_weight_grad``. Treating those as frozen silently zeroed out real
  weight gradients (``tgt=0`` against a non-zero reference in the
  ``test_moe_*subbatch_deep_gemm*`` suites).

These run on CPU except where a kernel is explicitly monkeypatched away.
"""

import unittest
from types import SimpleNamespace

import paddle

from paddlefleet.transformer.moe.fp8_utils import (
    expert_weights_all_frozen,
    slice_expert_weight,
)


def _param(trainable, shape=(2, 4)):
    param = paddle.create_parameter(list(shape), dtype="float32")
    param.stop_gradient = not trainable
    return param


class _Parent:
    """Stand-in for ``GroupedMLPExpert``: stacked weight1 / weight2 parameters."""

    def __init__(self, trainable):
        self.weight1 = _param(trainable, shape=(2, 4, 8))
        self.weight2 = _param(trainable, shape=(2, 8, 4))


class TestExpertWeightsAllFrozen(unittest.TestCase):
    """Only an actual frozen ``EagerParamBase`` counts as frozen."""

    def test_frozen_parameter(self):
        self.assertTrue(expert_weights_all_frozen(_param(trainable=False)))

    def test_trainable_parameter(self):
        self.assertFalse(expert_weights_all_frozen(_param(trainable=True)))

    def test_plain_tensor_is_not_frozen(self):
        # A plain tensor defaults to stop_gradient=True but says nothing about
        # whether the parameter behind it is trainable.
        tensor = paddle.randn([2, 4])
        self.assertTrue(tensor.stop_gradient)
        self.assertFalse(expert_weights_all_frozen(tensor))

    def test_slice_view_of_trainable_parameter_is_not_frozen(self):
        # Regression guard: the sliced deep_gemm path passes per-expert
        # ``parent._slice(i, i + 1)`` views. They are plain tensors with
        # stop_gradient=True even though the parent is trainable, and skipping
        # their wgrad zeroes out a real gradient.
        parent = _param(trainable=True)
        view = parent._slice(0, 1)
        self.assertTrue(view.stop_gradient)
        self.assertFalse(expert_weights_all_frozen(view))

    def test_list_all_frozen(self):
        self.assertTrue(
            expert_weights_all_frozen(
                [_param(trainable=False), _param(trainable=False)]
            )
        )

    def test_list_mixed_keeps_original_behavior(self):
        self.assertFalse(
            expert_weights_all_frozen(
                [_param(trainable=False), _param(trainable=True)]
            )
        )

    def test_list_with_slice_view_is_not_frozen(self):
        parent = _param(trainable=True)
        self.assertFalse(
            expert_weights_all_frozen(
                [_param(trainable=False), parent._slice(0, 1)]
            )
        )

    def test_none_and_empty(self):
        self.assertFalse(expert_weights_all_frozen(None))
        self.assertFalse(expert_weights_all_frozen([]))
        self.assertFalse(expert_weights_all_frozen([None]))


class TestSlicedViewResolvesToParent(unittest.TestCase):
    """The sliced / subbatch path must consult the parent parameter.

    Both per-expert slicing sites go through ``slice_expert_weight``, which stamps
    ``_parent`` on the view; ``_PerExpertWeightView`` stores the same thing. So
    ``bf16_weight_grad`` can uniformly do
    ``getattr(weights, "_parent", weights)`` before asking
    :func:`expert_weights_all_frozen`, because a view's own ``stop_gradient`` is
    always True.
    """

    def _view(self, parent_trainable):
        parent = _Parent(trainable=parent_trainable)
        return slice_expert_weight(parent.weight1, 0)

    def _resolve(self, weights):
        """The expression `bf16_weight_grad` uses."""
        return expert_weights_all_frozen(getattr(weights, "_parent", weights))

    def test_view_alone_is_never_frozen(self):
        # Without resolving through _parent the answer is always "not frozen",
        # which is safe but loses the optimization.
        view = self._view(parent_trainable=False)
        self.assertTrue(view.stop_gradient)
        self.assertFalse(expert_weights_all_frozen(view))

    def test_resolved_view_of_frozen_expert_is_frozen(self):
        self.assertTrue(self._resolve(self._view(parent_trainable=False)))

    def test_resolved_view_of_trainable_expert_is_not_frozen(self):
        self.assertFalse(self._resolve(self._view(parent_trainable=True)))

    def test_plain_parameter_needs_no_resolution(self):
        self.assertTrue(self._resolve(_param(trainable=False)))
        self.assertFalse(self._resolve(_param(trainable=True)))

    def test_fp8_view_carries_its_parent(self):
        # _PerExpertWeightView is not a Tensor and has no stop_gradient at all,
        # but it stores the parent parameter under the same attribute name.
        from paddlefleet.transformer.moe.fp8_utils import _PerExpertWeightView

        parent = _Parent(trainable=False)
        view = _PerExpertWeightView(parent.weight1, 0, 2)
        self.assertIs(view._parent, parent.weight1)
        self.assertTrue(self._resolve(view))


class TestStaticSubbatchNodeCarriesParent(unittest.TestCase):
    """The static per-expert deep_gemm node must not lose the parent pointer.

    ``ExpertsGroupGemmContiguousNode.__init__`` slices the stacked weights for a
    single expert (static subbatch + deep_gemm, bf16 weights). It used to stamp
    ``_parent`` only on the outer sliced object, so ``bf16_weight_grad`` saw a raw
    view and kept computing the wgrad of a frozen expert.
    """

    def _node(self, trainable):
        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        parent = _Parent(trainable=trainable)
        custom_map = SimpleNamespace(
            grouped_gemm_experts=parent,
            moe_rank=0,
            num_experts_per_device=2,
            experts=None,
        )
        node = ExpertsGroupGemmContiguousNode(
            custom_map,
            expert_id=1,
            moe_deep_gemm=True,
            use_bf16_gemm_weight_grad=True,
        )
        return parent, node

    def test_view_points_at_parent_parameter(self):
        parent, node = self._node(trainable=True)
        ge = node.grouped_gemm_experts
        self.assertIs(ge.weight1._parent, parent.weight1)
        self.assertIs(ge.weight2._parent, parent.weight2)
        # The view itself still reports stop_gradient=True.
        self.assertTrue(ge.weight1.stop_gradient)

    def test_frozen_parent_skips_wgrad_kernel(self):
        parent, node = self._node(trainable=False)
        ge = node.grouped_gemm_experts
        called = []
        node.bf16_gemm = lambda *a, **k: called.append("gemm")
        for weights in (ge.weight1, ge.weight2):
            self.assertIsNone(node.bf16_weight_grad(None, None, weights))
        self.assertEqual(called, [])
        self.assertIsNone(getattr(parent.weight1, "main_grad", None))
        self.assertIsNone(parent.weight1.grad)

    def test_trainable_parent_does_not_skip(self):
        _, node = self._node(trainable=True)
        ge = node.grouped_gemm_experts
        # Not frozen, so the early return must not fire: the call proceeds past
        # the guard and fails on the stubbed-out internals instead (the node
        # never set up ``tokens_per_expert_tensor``).
        with self.assertRaisesRegex(AttributeError, "tokens_per_expert_tensor"):
            node.bf16_weight_grad(None, None, ge.weight1)


class _FakeCtx:
    """Stand-in for a PyLayer ctx. ``stop_gradient`` round-trips faithfully.

    Verified against a real ``paddle.autograd.PyLayer``: a trainable parameter
    saved with ``save_for_backward`` comes back with ``stop_gradient=False`` and a
    frozen one with ``True``, so reading the flag off ``saved_tensor()`` is a
    valid way for a hand-written backward to decide whether to produce a grad.
    """

    def __init__(self, x, y, batch_sizes):
        self._saved = (x, y)
        self.batch_sizes = batch_sizes

    def saved_tensor(self):
        return self._saved


class TestDeepGEMMBMMBackwardStopGradient(unittest.TestCase):
    """``DeepGEMMBMMFunction.backward`` must return None per frozen input.

    Paddle's PyLayer contract rejects a gradient for a ``stop_gradient`` input,
    and the corresponding GEMM would be wasted work. The kernels are replaced by
    recorders so the branch logic can be checked on any device.
    """

    def setUp(self):
        from paddlefleet.transformer.moe import moe_expert

        self.moe_expert = moe_expert
        self.calls = []

        class _StubDeepGemm:
            @staticmethod
            def m_grouped_bf16_gemm_nt_contiguous(*args, **kwargs):
                self.calls.append("dx")

        self._orig_dg = getattr(moe_expert, "paddlefleet_deep_gemm", None)
        self._orig_k = moe_expert.k_grouped_bf16_gemm_tn_contiguous_aligned
        moe_expert.paddlefleet_deep_gemm = _StubDeepGemm

        def _stub_k(*args, **kwargs):
            self.calls.append("dy")

        moe_expert.k_grouped_bf16_gemm_tn_contiguous_aligned = _stub_k

    def tearDown(self):
        if self._orig_dg is None:
            self.moe_expert.__dict__.pop("paddlefleet_deep_gemm", None)
        else:
            self.moe_expert.paddlefleet_deep_gemm = self._orig_dg
        self.moe_expert.k_grouped_bf16_gemm_tn_contiguous_aligned = self._orig_k

    def _run(self, x_frozen, y_frozen):
        x = paddle.randn([4, 8]).astype("bfloat16")
        y = paddle.randn([1, 8, 8]).astype("bfloat16")
        x.stop_gradient = x_frozen
        y.stop_gradient = y_frozen
        batch_sizes = paddle.to_tensor([4], dtype="int64")
        grad = paddle.randn([4, 8]).astype("bfloat16")
        ctx = _FakeCtx(x, y, batch_sizes)
        return self.moe_expert.DeepGEMMBMMFunction.backward(ctx, grad)

    def test_both_trainable_produces_both_grads(self):
        dx, dy = self._run(x_frozen=False, y_frozen=False)
        self.assertIsNotNone(dx)
        self.assertIsNotNone(dy)
        self.assertEqual(sorted(self.calls), ["dx", "dy"])

    def test_frozen_weight_skips_wgrad(self):
        dx, dy = self._run(x_frozen=False, y_frozen=True)
        self.assertIsNotNone(dx)
        self.assertIsNone(dy)
        self.assertEqual(self.calls, ["dx"])

    def test_frozen_input_skips_dgrad(self):
        dx, dy = self._run(x_frozen=True, y_frozen=False)
        self.assertIsNone(dx)
        self.assertIsNotNone(dy)
        self.assertEqual(self.calls, ["dy"])

    def test_both_frozen_skips_everything(self):
        dx, dy = self._run(x_frozen=True, y_frozen=True)
        self.assertIsNone(dx)
        self.assertIsNone(dy)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
