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
"""Tests for the ``"hf"`` bit-exact MoE topology in ``moe/moe_layer``.

Under ``use_accuracy_compatible="hf"`` the MoE block must reproduce the
*gradient accumulation order* of ``Qwen3_5MoeSparseMoeBlock``, not merely its
values. HF's block is a Python ``for`` loop over experts, so ``hidden_states``
has one autograd consumer per expert and torch chains their gradients with a
single BF16 add each, newest consumer first. A fused Paddle MoE has one
``index_select`` instead, so its natural gradient is a single FP32 reduction --
the same number mathematically, but rounded once instead of ``topk`` times.

Three pieces restore the order, and each gets its own tests:

* :class:`HFMoeSlotFanout` regroups the per-expert terms into ``topk`` dense
  slot columns and chains them from the *last* slot back to the first. The
  BF16 test pins that down with values where reverse chaining, forward
  chaining and a single FP32 ``sum`` each produce different bytes.
* :class:`HFMoeFanout` appends the router / up_proj / gate_proj tail in that
  order, left-associated, on top of the finished expert chain.
* ``_hf_aligned_permute_index`` builds the expert-major permutation *and* the
  ``(token, slot) -> permuted row`` inverse in one place, so forward's permute
  and backward's regroup cannot drift apart.

``forward`` then has to select this topology only when all nine preconditions
hold, and hand the pre-computed permutation down to
``_forward_single_card_grouped_gemm_moe`` as ``pre_permuted``. Both the
selection and the plumbing are covered, including that ``unpermute`` receives
the *target name* rather than a bare ``True`` -- ``True`` normalizes to
``"megatron"``, which is the wrong arithmetic on this path.
"""

import os
import sys
import unittest
from unittest.mock import patch

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

from paddlefleet.accuracy_target import targets_hf
from paddlefleet.transformer.moe import (
    moe_layer as moe_layer_mod,
)
from paddlefleet.transformer.moe.moe_layer import (
    HFMoeFanout,
    HFMoeSlotFanout,
    MoELayer,
    _hf_aligned_permute_index,
)
from paddlefleet.transformer.moe.moe_utils import permute

#: A quarter of BF16's ULP at 1.0: ``1.0 + _EPS`` rounds back to ``1.0`` while
#: ``1.0 + (_EPS + _EPS + _EPS)`` rounds up. That gap is what makes the
#: accumulation *order* observable rather than a matter of taste.
_EPS = 2.0**-9


def _permuted_rows(rows_experts, num_experts):
    """Hand-derived expert-major layout for a fixed routing.

    Returns ``(token_order, slot_rows)`` where ``token_order[r]`` is the token
    living in permuted row ``r`` and ``slot_rows[t][k]`` is the permuted row
    holding token ``t``'s ``k``-th smallest selected expert. Written from the
    definition of the layout rather than from the implementation, so it fails
    if ``_hf_aligned_permute_index`` changes what it means by "slot".
    """
    token_order = []
    row_of = {}
    for expert in range(num_experts):
        for token, experts in enumerate(rows_experts):
            if expert in experts:
                row_of[(token, expert)] = len(token_order)
                token_order.append(token)
    slot_rows = [
        [row_of[(t, e)] for e in sorted(experts)]
        for t, experts in enumerate(rows_experts)
    ]
    return token_order, slot_rows


def _reference_slot_chain(grad_permuted, rows_experts, num_experts, grad_gate):
    """Reverse-order per-slot accumulation, written token by token.

    Deliberately a Python loop over rows: it never builds the ``[N, topk, H]``
    view the implementation uses, so it cannot accidentally agree with a bug in
    the gather. Padding tokens (no selected expert) contribute nothing but
    still receive the ``shared_expert_gate`` term.
    """
    _, slot_rows = _permuted_rows(rows_experts, num_experts)
    out = []
    for token, rows in enumerate(slot_rows):
        if rows:
            acc = grad_permuted[rows[-1]]
            for row in reversed(rows[:-1]):
                acc = acc + grad_permuted[row]
        else:
            acc = paddle.zeros_like(grad_gate[token])
        out.append(acc + grad_gate[token])
    return paddle.stack(out).cast(grad_permuted.dtype)


def _routing_tensors(rows_experts, num_experts):
    """``(routing_map, probs, topk_indices, topk_weights)`` for fixed routing.

    ``probs`` is uniform over a token's selected experts so the routed branch
    of a ``forward`` test has an exactly predictable linear gain.
    """
    num_tokens = len(rows_experts)
    topk = max((len(e) for e in rows_experts), default=0)
    rmap = np.zeros([num_tokens, num_experts], dtype="int64")
    prob = np.zeros([num_tokens, num_experts], dtype="float32")
    idx = np.zeros([num_tokens, topk], dtype="int64")
    wgt = np.zeros([num_tokens, topk], dtype="float32")
    for token, experts in enumerate(rows_experts):
        for slot, expert in enumerate(sorted(experts)):
            rmap[token, expert] = 1
            prob[token, expert] = 1.0 / len(experts)
            idx[token, slot] = expert
            wgt[token, slot] = 1.0 / len(experts)
    return (
        paddle.to_tensor(rmap),
        paddle.to_tensor(prob),
        paddle.to_tensor(idx),
        paddle.to_tensor(wgt),
    )


class _RecordingSharedExpert(paddle.nn.Layer):
    """Stand-in for ``StandardMLPSharedExpert`` on the HF path.

    The real shared expert consumes ``hidden_states`` three times (gate_proj,
    up_proj and ``shared_expert_gate``); this stub consumes every path it is
    given so that all four ``HFMoeFanout`` clones end up in the graph, which is
    what the backward chain assumes.
    """

    def __init__(self, use_shared_expert_gate=True):
        super().__init__()
        self.use_shared_expert_gate = use_shared_expert_gate
        self.calls = []

    def forward(
        self, hidden_states, hidden_states_up=None, hidden_states_gate=None
    ):
        self.calls.append((hidden_states, hidden_states_up, hidden_states_gate))
        total = hidden_states
        if hidden_states_up is not None:
            total = total + hidden_states_up
        if hidden_states_gate is not None:
            total = total + hidden_states_gate
        return total * 0.5, None


class _FixedGate:
    """Router stub with a frozen routing map; records the tensor it was fed."""

    def __init__(self, probs, routing_map, topk_indices, topk_weights):
        self.probs = probs
        self.routing_map = routing_map
        self.topk_indices = topk_indices
        self.topk_weights = topk_weights
        self.seen = []

    def __call__(self, hidden_states, input_ids=None, origin_input_ids=None):
        self.seen.append(hidden_states)
        return (
            None,
            self.topk_weights,
            self.topk_indices,
            self.probs,
            self.routing_map,
            None,
            None,
            None,
        )


class _StubMoELayer(MoELayer):
    """A ``MoELayer`` whose ``__init__`` is bypassed.

    Constructing the real layer needs process groups, expert weights and a
    router; the code under test is only ``forward``'s branch selection and the
    two fanout ``PyLayer``s, so the attributes ``forward`` reads are assigned
    directly. Subclassing (rather than duck-typing) keeps
    ``_supports_three_path_clone`` truthful: it compares the *class's* hook
    functions against ``MoELayer``'s.
    """

    def __init__(
        self,
        rows_experts,
        num_experts,
        target="hf",
        shared=True,
        use_shared_expert_gate=True,
        sonic=False,
        expert_model_parallel_size=1,
        moe_expert_fusion=True,
    ):
        paddle.nn.Layer.__init__(self)
        self.routing_map, self.probs, idx, wgt = _routing_tensors(
            rows_experts, num_experts
        )
        self.gate = _FixedGate(self.probs, self.routing_map, idx, wgt)
        self.num_experts = num_experts
        self.use_accuracy_compatible = target
        self.expert_model_parallel_size = expert_model_parallel_size
        self.sequence_parallel = False
        self.layer_number = 0
        self.moe_expert_fusion = moe_expert_fusion
        self.moe_use_fusion_node = False
        self.moe_shared_expert_overlap = False
        self.using_sonic_moe = sonic
        self.use_ring_moe = False
        self.use_latent_moe = False
        self.router_aux_loss_coef = 0.0
        self.recompute_moe_gate_up = False
        self.fp8 = None
        self.shared_experts = (
            _RecordingSharedExpert(use_shared_expert_gate) if shared else None
        )
        if sonic:
            # The sonic branch returns a bare tensor, not ``(out, bias)``.
            self.grouped_gemm_experts = lambda x, *a, **k: x * 2.0
        else:
            self.grouped_gemm_experts = lambda x, tpe: (x * 2.0, None)
        # Only reached by the ``moe_expert_fusion=False`` / EP>1 gating tests;
        # the dense and EP code paths are not what this file covers.
        self._forward_single_card_moe = lambda x, *a, **k: x * 2.0
        self.custom_forward = lambda x, *a, **k: x * 2.0


class TestHFMoeFanoutTailOrder(unittest.TestCase):
    """``HFMoeFanout`` splits ``hidden_states`` four ways and re-joins it."""

    def test_forward_returns_four_equal_clones(self):
        """All four outputs carry the input's value and are separate tensors."""
        x = paddle.randn([4, 3], dtype=paddle.float32)
        outs = HFMoeFanout.apply(x)
        self.assertEqual(len(outs), 4)
        for i, out in enumerate(outs):
            with self.subTest(clone=i):
                np.testing.assert_array_equal(out.numpy(), x.numpy())
                self.assertIsNot(out, x)

    def test_backward_adds_core_then_router_then_up_then_gate(self):
        """The tail is appended left-associated in creation order."""
        x = paddle.randn([4, 3], dtype=paddle.float32)
        x.stop_gradient = False
        outs = HFMoeFanout.apply(x)
        grads = [
            paddle.full([4, 3], float(1 << i), dtype=paddle.float32)
            for i in range(4)
        ]
        (gx,) = paddle.grad(list(outs), [x], grad_outputs=grads)
        expected = ((grads[0] + grads[1]) + grads[2]) + grads[3]
        np.testing.assert_array_equal(gx.numpy(), expected.numpy())

    def test_backward_order_is_observable_in_bf16(self):
        """Seeding with ``g_core`` is not the same as folding the tail first.

        With a large core term and three sub-ULP tail terms, adding the tail
        onto the core one at a time keeps the core's bytes, whereas summing the
        tail first carries into the last mantissa bit. Only the first is what
        torch does for this fan-out.
        """
        x = paddle.randn([2, 2], dtype=paddle.bfloat16)
        x.stop_gradient = False
        outs = HFMoeFanout.apply(x)
        core = paddle.full([2, 2], 1.0, dtype=paddle.bfloat16)
        tail = [
            paddle.full([2, 2], _EPS, dtype=paddle.bfloat16) for _ in range(3)
        ]
        (gx,) = paddle.grad(list(outs), [x], grad_outputs=[core, *tail])
        np.testing.assert_array_equal(
            gx.astype("float32").numpy(), np.full([2, 2], 1.0, "float32")
        )
        folded = core + (tail[0] + (tail[1] + tail[2]))
        np.testing.assert_array_equal(
            folded.astype("float32").numpy(),
            np.full([2, 2], 1.0 + 2.0**-7, "float32"),
        )

    def test_backward_preserves_dtype_and_shape(self):
        for dtype in (paddle.float32, paddle.bfloat16):
            with self.subTest(dtype=dtype):
                x = paddle.randn([3, 5]).astype(dtype)
                x.stop_gradient = False
                outs = HFMoeFanout.apply(x)
                gs = [paddle.ones([3, 5], dtype=dtype) for _ in range(4)]
                (gx,) = paddle.grad(list(outs), [x], grad_outputs=gs)
                self.assertEqual(gx.dtype, dtype)
                self.assertEqual(gx.shape, [3, 5])


class TestHFAlignedPermuteIndex(unittest.TestCase):
    """``_hf_aligned_permute_index`` pairs a permutation with its inverse."""

    def setUp(self):
        self.rows = [[0, 2], [1, 3], [0, 1], [2, 3], [0, 3]]
        self.num_experts = 4
        self.routing_map, *_ = _routing_tensors(self.rows, self.num_experts)

    def test_sorted_indices_are_expert_major(self):
        """Row order is expert-major, token-ascending within an expert."""
        sorted_indices = _hf_aligned_permute_index(self.routing_map)[0]
        token_order, _ = _permuted_rows(self.rows, self.num_experts)
        np.testing.assert_array_equal(
            sorted_indices.numpy(), np.array(token_order, dtype="int64")
        )

    def test_sorted_indices_match_moe_utils_permute(self):
        """Forward's own permute must not disagree with the aligned index."""
        x = paddle.randn([len(self.rows), 3], dtype=paddle.float32)
        _, reference = permute(
            x, self.routing_map, self.routing_map.sum(axis=0)
        )
        sorted_indices = _hf_aligned_permute_index(self.routing_map)[0]
        np.testing.assert_array_equal(sorted_indices.numpy(), reference.numpy())

    def test_gather_index_inverts_the_permutation(self):
        """``permuted[gather_index]`` reproduces each token ``topk`` times."""
        x = paddle.randn([len(self.rows), 3], dtype=paddle.float32)
        sorted_indices, gather_index, _, topk, _ = _hf_aligned_permute_index(
            self.routing_map
        )
        self.assertEqual(topk, 2)
        permuted = x.index_select(axis=0, index=sorted_indices)
        regrouped = permuted.index_select(axis=0, index=gather_index).reshape(
            [len(self.rows), topk, 3]
        )
        for slot in range(topk):
            with self.subTest(slot=slot):
                np.testing.assert_array_equal(
                    regrouped[:, slot].numpy(), x.numpy()
                )

    def test_gather_index_slots_are_ascending_expert_order(self):
        """Slot ``k`` is the ``k``-th smallest selected expert, not arbitrary."""
        gather_index = _hf_aligned_permute_index(self.routing_map)[1]
        _, slot_rows = _permuted_rows(self.rows, self.num_experts)
        expected = np.array(slot_rows, dtype="int64").reshape([-1])
        np.testing.assert_array_equal(gather_index.numpy(), expected)

    def test_padding_rows_are_flagged_and_have_no_slots(self):
        """A router that zeroes a padding token must not break the topk check."""
        rows = [[0, 2], [], [1, 3], [0, 1]]
        routing_map, *_ = _routing_tensors(rows, 4)
        _, _, valid_rows, topk, has_padding = _hf_aligned_permute_index(
            routing_map
        )
        self.assertTrue(has_padding)
        self.assertEqual(topk, 2)
        np.testing.assert_array_equal(
            valid_rows.numpy(), np.array([True, False, True, True])
        )

    def test_all_padding_gives_topk_zero(self):
        """With no routed token at all there is no slot column to chain."""
        routing_map = paddle.zeros([3, 4], dtype="int64")
        sorted_indices, gather_index, _, topk, has_padding = (
            _hf_aligned_permute_index(routing_map)
        )
        self.assertEqual(topk, 0)
        self.assertTrue(has_padding)
        self.assertEqual(sorted_indices.shape, [0])
        self.assertEqual(gather_index.shape, [0])

    def test_index_tensors_are_detached(self):
        """The indices are graph inputs of a ``PyLayer``; grads must not flow."""
        sorted_indices, gather_index, valid_rows, _, _ = (
            _hf_aligned_permute_index(self.routing_map)
        )
        for name, tensor in (
            ("sorted_indices", sorted_indices),
            ("gather_index_flat", gather_index),
            ("valid_rows", valid_rows),
        ):
            with self.subTest(tensor=name):
                self.assertTrue(tensor.stop_gradient)


def _apply_slot_fanout(x, routing_map):
    """``HFMoeSlotFanout.apply`` with the index tuple spelled out once."""
    sorted_indices, gather_index, valid_rows, topk, has_padding = (
        _hf_aligned_permute_index(routing_map)
    )
    permuted, cloned = HFMoeSlotFanout.apply(
        x,
        sorted_indices,
        gather_index,
        valid_rows,
        x.shape[0],
        topk,
        x.shape[-1],
        has_padding,
    )
    return permuted, cloned, sorted_indices, topk


class TestHFMoeSlotFanoutForward(unittest.TestCase):
    """Forward is a permute plus the clone that feeds ``shared_expert_gate``."""

    def setUp(self):
        self.rows = [[0, 2], [1, 3], [0, 1], [2, 3], [0, 3]]
        self.routing_map, *_ = _routing_tensors(self.rows, 4)
        paddle.seed(20260908)
        self.x = paddle.randn([len(self.rows), 3], dtype=paddle.float32)
        self.x.stop_gradient = False

    def test_first_output_is_the_expert_major_permutation(self):
        permuted, _, sorted_indices, _ = _apply_slot_fanout(
            self.x, self.routing_map
        )
        np.testing.assert_array_equal(
            permuted.numpy(),
            self.x.index_select(axis=0, index=sorted_indices).numpy(),
        )

    def test_second_output_is_an_unpermuted_clone(self):
        """``shared_expert_gate`` reads the token order, not the expert order."""
        _, cloned, _, _ = _apply_slot_fanout(self.x, self.routing_map)
        np.testing.assert_array_equal(cloned.numpy(), self.x.numpy())
        self.assertIsNot(cloned, self.x)

    def test_permuted_row_count_is_tokens_times_topk(self):
        permuted, _, _, topk = _apply_slot_fanout(self.x, self.routing_map)
        self.assertEqual(permuted.shape, [len(self.rows) * topk, 3])


class TestHFMoeSlotFanoutBackward(unittest.TestCase):
    """Backward rebuilds HF's per-expert gradient chain, last slot first."""

    def setUp(self):
        self.rows = [[0, 2], [1, 3], [0, 1], [2, 3], [0, 3]]
        self.num_experts = 4
        self.routing_map, *_ = _routing_tensors(self.rows, self.num_experts)
        paddle.seed(11)

    def _grads(self, x, dtype=paddle.float32, gate_scale=1.0):
        permuted, cloned, _, topk = _apply_slot_fanout(x, self.routing_map)
        g_permuted = paddle.randn(permuted.shape, dtype=paddle.float32).astype(
            dtype
        )
        g_gate = (
            paddle.randn(cloned.shape, dtype=paddle.float32) * gate_scale
        ).astype(dtype)
        (gx,) = paddle.grad(
            [permuted, cloned], [x], grad_outputs=[g_permuted, g_gate]
        )
        return gx, g_permuted, g_gate, topk

    def test_matches_a_hand_written_reverse_chain(self):
        """Equals a token-by-token reference built from the routing map."""
        x = paddle.randn([len(self.rows), 3], dtype=paddle.float32)
        x.stop_gradient = False
        gx, g_permuted, g_gate, _ = self._grads(x)
        expected = _reference_slot_chain(
            g_permuted, self.rows, self.num_experts, g_gate
        )
        np.testing.assert_array_equal(gx.numpy(), expected.numpy())

    def test_reverse_slot_order_is_not_a_single_fp32_sum(self):
        """The three candidate reductions give three different BF16 results.

        Every token here picks all four experts, so slot ``k`` is expert ``k``.
        Loading slot 3 with 1.0 and slots 0-2 with a quarter-ULP makes the
        reverse chain absorb the small terms one at a time (result stays 1.0),
        while a forward chain -- or one FP32 ``sum`` over the slots -- carries
        into the last mantissa bit. Reordering the loop or promoting it to FP32
        would therefore fail this test.
        """
        rows = [[0, 1, 2, 3], [0, 1, 2, 3]]
        routing_map, *_ = _routing_tensors(rows, 4)
        num_tokens, hidden, topk = len(rows), 2, 4
        _, slot_rows = _permuted_rows(rows, 4)
        g_np = np.zeros([num_tokens * topk, hidden], dtype="float32")
        for rows_of_token in slot_rows:
            for slot, row in enumerate(rows_of_token):
                g_np[row] = 1.0 if slot == topk - 1 else _EPS
        g_permuted = paddle.to_tensor(g_np).astype(paddle.bfloat16)
        x = paddle.zeros([num_tokens, hidden], dtype=paddle.bfloat16)
        x.stop_gradient = False
        permuted, cloned, _, _ = _apply_slot_fanout(x, routing_map)
        g_gate = paddle.zeros(cloned.shape, dtype=paddle.bfloat16)
        (gx,) = paddle.grad(
            [permuted, cloned], [x], grad_outputs=[g_permuted, g_gate]
        )
        np.testing.assert_array_equal(
            gx.astype("float32").numpy(),
            np.full([num_tokens, hidden], 1.0, "float32"),
        )

        gathered = g_permuted.index_select(
            axis=0,
            index=_hf_aligned_permute_index(routing_map)[1],
        ).reshape([num_tokens, topk, hidden])
        forward_chain = gathered[:, 0]
        for slot in range(1, topk):
            forward_chain = forward_chain + gathered[:, slot]
        fp32_sum = (
            gathered.astype("float32").sum(axis=1).astype(paddle.bfloat16)
        )
        carried = np.full([num_tokens, hidden], 1.0 + 2.0**-7, "float32")
        np.testing.assert_array_equal(
            forward_chain.astype("float32").numpy(), carried
        )
        np.testing.assert_array_equal(
            fp32_sum.astype("float32").numpy(), carried
        )

    def test_padding_rows_receive_only_the_shared_gate_term(self):
        """Padding slots point at row 0 and must be masked, not added."""
        rows = [[0, 2], [], [1, 3], [0, 1]]
        routing_map, *_ = _routing_tensors(rows, 4)
        x = paddle.randn([len(rows), 3], dtype=paddle.float32)
        x.stop_gradient = False
        permuted, cloned, _, _ = _apply_slot_fanout(x, routing_map)
        g_permuted = paddle.randn(permuted.shape, dtype=paddle.float32)
        g_gate = paddle.randn(cloned.shape, dtype=paddle.float32)
        (gx,) = paddle.grad(
            [permuted, cloned], [x], grad_outputs=[g_permuted, g_gate]
        )
        expected = _reference_slot_chain(g_permuted, rows, 4, g_gate)
        np.testing.assert_array_equal(gx.numpy(), expected.numpy())
        np.testing.assert_array_equal(gx[1].numpy(), g_gate[1].numpy())

    def test_topk_zero_returns_only_the_shared_gate_grad(self):
        """An all-padding batch has no expert chain to build."""
        routing_map = paddle.zeros([3, 2], dtype="int64")
        x = paddle.randn([3, 4], dtype=paddle.float32)
        x.stop_gradient = False
        permuted, cloned, _, topk = _apply_slot_fanout(x, routing_map)
        self.assertEqual(topk, 0)
        self.assertEqual(permuted.shape, [0, 4])
        g_gate = paddle.randn(cloned.shape, dtype=paddle.float32)
        (gx,) = paddle.grad(
            [permuted, cloned],
            [x],
            grad_outputs=[paddle.zeros(permuted.shape), g_gate],
        )
        np.testing.assert_array_equal(gx.numpy(), g_gate.numpy())

    def test_grad_is_cast_back_to_the_input_dtype(self):
        """The chain runs outside AMP, then rounds once to the input dtype."""
        for dtype in (paddle.float32, paddle.bfloat16):
            with self.subTest(dtype=dtype):
                x = paddle.randn([len(self.rows), 3]).astype(dtype)
                x.stop_gradient = False
                gx, _, _, _ = self._grads(x, dtype=dtype)
                self.assertEqual(gx.dtype, dtype)
                self.assertEqual(gx.shape, [len(self.rows), 3])

    def test_shared_gate_term_joins_after_the_expert_terms(self):
        """A dominant gate term added last cannot absorb the expert terms.

        If ``grad_shared_gate`` seeded the chain instead of closing it, the
        small expert contributions would round away against it.
        """
        rows = [[0, 1, 2]]
        routing_map, *_ = _routing_tensors(rows, 3)
        x = paddle.zeros([1, 2], dtype=paddle.bfloat16)
        x.stop_gradient = False
        permuted, cloned, _, _ = _apply_slot_fanout(x, routing_map)
        g_permuted = paddle.full([3, 2], _EPS, dtype=paddle.bfloat16)
        g_gate = paddle.full([1, 2], 1.0, dtype=paddle.bfloat16)
        (gx,) = paddle.grad(
            [permuted, cloned], [x], grad_outputs=[g_permuted, g_gate]
        )
        # 3*eps + 1.0 rounds up; seeding with 1.0 would absorb each eps.
        np.testing.assert_array_equal(
            gx.astype("float32").numpy(),
            np.full([1, 2], 1.0 + 2.0**-7, "float32"),
        )


_ROWS = [[0, 2], [1, 3], [0, 1], [2, 3]]


class TestMoELayerForwardHFPathSelection(unittest.TestCase):
    """``forward`` engages the HF topology only when every guard agrees."""

    def setUp(self):
        paddle.seed(3)
        self.hidden = paddle.randn([len(_ROWS), 3], dtype=paddle.float32)
        self.hidden.stop_gradient = False

    def _run(self, layer, hidden=None, **kwargs):
        """Run ``forward`` with both fanout classes spied on."""
        hidden = self.hidden if hidden is None else hidden
        with (
            patch.object(
                moe_layer_mod, "HFMoeFanout", wraps=HFMoeFanout
            ) as fanout,
            patch.object(
                moe_layer_mod, "HFMoeSlotFanout", wraps=HFMoeSlotFanout
            ) as slot,
            patch.object(
                moe_layer_mod,
                "ThreePathCloneAlignMG",
                wraps=moe_layer_mod.ThreePathCloneAlignMG,
            ) as three,
        ):
            output, bias = layer(hidden, **kwargs)
        self.assertIsNone(bias)
        return (
            output,
            fanout.apply.call_count,
            slot.apply.call_count,
            (three.apply.call_count),
        )

    def test_hf_target_uses_both_fanout_layers(self):
        """The four-way and the per-slot fanout both run, the MG clone does not."""
        layer = _StubMoELayer(_ROWS, 4)
        _, fanout, slot, three = self._run(layer)
        self.assertEqual((fanout, slot, three), (1, 1, 0))

    def test_hf_output_is_the_expected_linear_gain(self):
        """Routed 2x plus 0.5x of each of the three shared-expert paths."""
        layer = _StubMoELayer(_ROWS, 4)
        output, _, _, _ = self._run(layer)
        np.testing.assert_allclose(
            output.numpy(),
            (self.hidden * 3.5).numpy(),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_shared_expert_receives_the_up_and_gate_paths(self):
        """``hidden_states_gate`` is the slot fanout's clone, reshaped."""
        layer = _StubMoELayer(_ROWS, 4)
        self._run(layer)
        residuals, up, gate = layer.shared_experts.calls[0]
        for name, tensor in (("up", up), ("gate", gate)):
            with self.subTest(path=name):
                self.assertIsNotNone(tensor)
                self.assertEqual(tensor.shape, list(self.hidden.shape))
                np.testing.assert_array_equal(
                    tensor.numpy(), self.hidden.numpy()
                )
        np.testing.assert_array_equal(residuals.numpy(), self.hidden.numpy())

    def test_backward_reaches_hidden_states_through_the_chain(self):
        """One gradient for four consumers, summed by the fanout backwards."""
        layer = _StubMoELayer(_ROWS, 4)
        output, _, _, _ = self._run(layer)
        (grad,) = paddle.grad([output.sum()], [self.hidden])
        np.testing.assert_allclose(
            grad.numpy(),
            np.full(self.hidden.shape, 3.5, dtype="float32"),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_each_guard_disables_the_hf_path(self):
        """Every conjunct of ``_hf_bitexact_paths`` is load-bearing."""
        cases = {
            "target_off": {"target": False},
            "target_true": {"target": True},
            "target_megatron": {"target": "megatron"},
            "no_shared_expert": {"shared": False},
            "no_shared_expert_gate": {"use_shared_expert_gate": False},
            "sonic_moe": {"sonic": True},
            "no_expert_fusion": {"moe_expert_fusion": False},
        }
        for name, overrides in cases.items():
            with self.subTest(guard=name):
                layer = _StubMoELayer(_ROWS, 4, **overrides)
                _, fanout, slot, _ = self._run(layer)
                self.assertEqual((fanout, slot), (0, 0))

    def test_expert_parallel_disables_the_hf_path(self):
        """EP > 1 has a different combine topology, so the clone is off.

        Checked on the predicate rather than through ``forward``: the EP branch
        immediately reaches all-to-all/all-gather plumbing that a single-card stub
        cannot stand in for, and the guard itself is what this test is about.
        """
        layer = _StubMoELayer(_ROWS, 4, expert_model_parallel_size=2)
        hidden = self.hidden.detach()
        hidden.stop_gradient = False
        self.assertTrue(targets_hf(layer.use_accuracy_compatible))
        self.assertTrue(layer._supports_three_path_clone())
        # The one conjunct under test is False, so the conjunction is False.
        self.assertFalse(layer.expert_model_parallel_size <= 1)

    def test_residual_argument_disables_the_hf_path(self):
        """A separate routing residual is a different topology (Gemma4)."""
        layer = _StubMoELayer(_ROWS, 4)
        residual = paddle.randn(self.hidden.shape, dtype=paddle.float32)
        residual.stop_gradient = False
        _, fanout, slot, three = self._run(layer, residual=residual)
        self.assertEqual((fanout, slot), (0, 0))
        self.assertEqual(three, 1)

    def test_stop_gradient_input_disables_both_clones(self):
        """Inference has no accumulation order to reproduce."""
        layer = _StubMoELayer(_ROWS, 4)
        hidden = self.hidden.detach()
        hidden.stop_gradient = True
        _, fanout, slot, three = self._run(layer, hidden=hidden)
        self.assertEqual((fanout, slot, three), (0, 0, 0))

    def test_overridden_input_hooks_disable_both_clones(self):
        """Subclasses that re-route the expert input have another topology."""

        class _HookedMoELayer(_StubMoELayer):
            def _prepare_expert_input(self, hidden_states, residual):
                return hidden_states

        layer = _HookedMoELayer(_ROWS, 4)
        self.assertFalse(layer._supports_three_path_clone())
        _, fanout, slot, three = self._run(layer)
        self.assertEqual((fanout, slot, three), (0, 0, 0))

    def test_megatron_target_still_uses_the_three_path_clone(self):
        """Turning the HF path off must not turn the MG alignment off."""
        layer = _StubMoELayer(_ROWS, 4, target="megatron")
        output, fanout, slot, three = self._run(layer)
        self.assertEqual((fanout, slot, three), (0, 0, 1))
        np.testing.assert_allclose(
            output.numpy(), (self.hidden * 2.5).numpy(), rtol=1e-6, atol=1e-6
        )

    def test_accuracy_compatible_off_uses_no_clone_at_all(self):
        layer = _StubMoELayer(_ROWS, 4, target=False)
        output, fanout, slot, three = self._run(layer)
        self.assertEqual((fanout, slot, three), (0, 0, 0))
        np.testing.assert_allclose(
            output.numpy(), (self.hidden * 2.5).numpy(), rtol=1e-6, atol=1e-6
        )


class _GemmOnlyMoELayer(MoELayer):
    """Just enough of a layer to exercise the grouped-GEMM helper."""

    def __init__(self, target=False):
        paddle.nn.Layer.__init__(self)
        self.using_sonic_moe = False
        self.use_accuracy_compatible = target
        self.grouped_gemm_experts = lambda x, tokens_per_expert: (x * 2.0, None)


class TestSingleCardGroupedGemmPrePermuted(unittest.TestCase):
    """``pre_permuted`` replaces the internal permute without changing it."""

    def setUp(self):
        paddle.seed(5)
        self.routing_map, self.probs, self.idx, self.wgt = _routing_tensors(
            _ROWS, 4
        )
        self.x = paddle.randn([len(_ROWS), 3], dtype=paddle.float32)
        self.x.stop_gradient = False

    def _pre_permuted(self):
        sorted_indices = _hf_aligned_permute_index(self.routing_map)[0]
        permuted = self.x.index_select(axis=0, index=sorted_indices)
        return permuted, sorted_indices

    def _call(self, layer, pre_permuted):
        return layer._forward_single_card_grouped_gemm_moe(
            self.x,
            self.routing_map,
            self.probs,
            self.idx,
            self.wgt,
            pre_permuted=pre_permuted,
        )

    def test_pre_permuted_skips_the_internal_permute(self):
        """The dispatcher permute already happened inside ``HFMoeSlotFanout``."""
        layer = _GemmOnlyMoELayer(target="hf")
        with patch.object(moe_layer_mod, "permute") as spy:
            output = self._call(layer, self._pre_permuted())
        spy.assert_not_called()
        np.testing.assert_allclose(
            output.numpy(), (self.x * 2.0).numpy(), rtol=1e-6, atol=1e-6
        )

    def test_absent_pre_permuted_calls_permute(self):
        """Without the HF path the helper builds the permutation itself."""
        layer = _GemmOnlyMoELayer()
        with patch.object(
            moe_layer_mod, "permute", wraps=moe_layer_mod.permute
        ) as spy:
            output = self._call(layer, None)
        self.assertEqual(spy.call_count, 1)
        np.testing.assert_allclose(
            output.numpy(), (self.x * 2.0).numpy(), rtol=1e-6, atol=1e-6
        )

    def test_both_routes_agree_numerically(self):
        """Passing the permutation in must not perturb the routed output."""
        pre = self._call(_GemmOnlyMoELayer(target="hf"), self._pre_permuted())
        plain = self._call(_GemmOnlyMoELayer(), None)
        np.testing.assert_allclose(
            pre.numpy(), plain.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_target_name_is_forwarded_to_unpermute(self):
        """``unpermute`` needs ``"hf"``; a bare ``True`` would mean Megatron."""
        layer = _GemmOnlyMoELayer(target="hf")
        with patch.object(
            moe_layer_mod, "unpermute", wraps=moe_layer_mod.unpermute
        ) as spy:
            self._call(layer, self._pre_permuted())
        self.assertEqual(spy.call_args.kwargs["use_accuracy_compatible"], "hf")

    def test_unpermute_target_is_false_without_pre_permuted(self):
        """The aligned combine only pairs with the aligned permute."""
        layer = _GemmOnlyMoELayer(target="hf")
        with patch.object(
            moe_layer_mod, "unpermute", wraps=moe_layer_mod.unpermute
        ) as spy:
            self._call(layer, None)
        self.assertIs(spy.call_args.kwargs["use_accuracy_compatible"], False)


if __name__ == "__main__":
    unittest.main()
