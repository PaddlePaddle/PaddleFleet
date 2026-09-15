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
"""Behavior tests for the ``use_accuracy_compatible`` bf16 MoE path in
``paddlefleet.transformer.moe.fp8_utils.ExpertsGroupGemmContiguousNode``.

The ``use_accuracy_compatible`` flag switches the pure-bf16, split-group
(``moe_expert_fusion=False``, ``use_fp8_mlp=False``) expert path onto a
"compute in fp32, round once" formulation so per-expert GEMMs and the
SwiGLU activation match a Megatron SequentialMLP reference bit-for-bit.

Each test drives a real node built through the production constructor
(only the weight-holding ``custom_map`` collaborator is mocked) and
compares against an expected value derived by hand in NumPy from the
mathematical definition of the operation -- never by calling the
implementation under test. Because the split-group path is dtype-agnostic
(plain slicing + matmul/linear + concat + analytic SwiGLU), the tests use
float32 inputs so the hand-derived references are exact to fp32 rounding.

Paddle is an optional heavy dependency; when it (or the production module)
cannot be imported the whole module is skipped with an honest reason
rather than reporting a false pass.
"""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)
# Also make the in-tree ``src`` layout importable when the package is not
# pip-installed into the environment.
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        ),
        "src",
    ),
)

_IMPORT_ERROR = None
try:
    from unittest.mock import MagicMock

    import numpy as np
    import paddle
    import paddle.nn.functional as F

    from paddlefleet.transformer.moe.fp8_utils import (
        FP8_ALIGN,
        ExpertsGroupGemmContiguousNode,
        moe_token_padding_alignment,
    )

    _DEPS_OK = True
except ImportError as exc:  # honest skip: real missing dependency only
    _DEPS_OK = False
    _IMPORT_ERROR = repr(exc)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _silu(x):
    return x * _sigmoid(x)


def _silu_grad(x):
    s = _sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def _make_split_group_node(
    use_accuracy_compatible, activation_type="swiglu", clamp_value=None
):
    """Build a real split-group bf16 node.

    ``moe_expert_fusion=False`` + ``use_fp8_mlp=False`` gives
    ``is_split_group_gemm=True`` and routes through the per-expert Python
    loop that the flag under test modifies. Only ``custom_map`` (the weight
    container, not code under test) is mocked; ``config=None`` keeps the
    activation-beta lookups on their real defaults.
    """
    custom_map = MagicMock()
    custom_map.experts = [MagicMock()]
    custom_map.config = None
    return ExpertsGroupGemmContiguousNode(
        custom_map,
        use_fp8_mlp=False,
        moe_expert_fusion=False,
        use_accuracy_compatible=use_accuracy_compatible,
        activation_type=activation_type,
        clamp_value=clamp_value,
    )


@unittest.skipUnless(
    _DEPS_OK, f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}"
)
class TestMoeTokenPaddingAlignment(unittest.TestCase):
    """``moe_token_padding_alignment`` returns 1 (no padding) only on the
    pure-bf16 non-grouped-gemm accuracy-compatible path; every other
    combination must keep the ``FP8_ALIGN`` (128) alignment."""

    def test_all_flag_combinations(self):
        for use_acc in (True, False):
            for use_fp8 in (True, False):
                for grouped in (True, False):
                    got = moe_token_padding_alignment(
                        use_fp8_mlp=use_fp8,
                        moe_grouped_gemm=grouped,
                        use_accuracy_compatible=use_acc,
                    )
                    # Independent restatement of the contract, not a copy of
                    # the branch: skip padding iff accuracy-compatible AND
                    # pure bf16 AND not grouped gemm.
                    expected = (
                        1
                        if (use_acc and not use_fp8 and not grouped)
                        else FP8_ALIGN
                    )
                    self.assertEqual(
                        got,
                        expected,
                        msg=(
                            f"use_acc={use_acc} use_fp8={use_fp8} "
                            f"grouped={grouped}"
                        ),
                    )

    def test_fp8_align_constant_is_128(self):
        # The alignment target is a hardware GEMM tile constant; pin it so a
        # silent change is caught by the combination test above.
        self.assertEqual(FP8_ALIGN, 128)


@unittest.skipUnless(
    _DEPS_OK, f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}"
)
class TestBwdGateUpInputBf16(unittest.TestCase):
    """``bwd_gate_up_input_bf16`` splits ``do1`` by ``tokens_per_expert`` and,
    for each expert i, computes ``do1_i @ expert_w1[i].T``. The accuracy-
    compatible branch uses ``paddle.matmul`` and the default uses
    ``F.linear``; both must equal the per-expert reference and each other,
    and zero-token experts must be skipped without shifting the split."""

    def _fixed_inputs(self):
        # N=4 (do1 feature dim), K=5 (dx feature dim), 3 tokens.
        do1 = paddle.to_tensor(
            [
                [0.5, -1.0, 2.0, 0.25],
                [1.5, 0.5, -0.5, 1.0],
                [-2.0, 1.0, 0.0, 0.75],
            ],
            dtype="float32",
        )
        # expert_w1[i] is [K, N]; the code transposes to [N, K].
        w1_0 = paddle.to_tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [-0.5, 0.6, 0.7, -0.8],
                [0.9, -1.0, 1.1, 1.2],
                [1.3, 1.4, -1.5, 1.6],
                [0.05, -0.15, 0.25, -0.35],
            ],
            dtype="float32",
        )
        w1_1 = paddle.to_tensor(
            [
                [2.0, -0.5, 0.5, 1.0],
                [0.3, 0.7, -0.2, 0.9],
                [-1.1, 1.2, 0.4, -0.6],
                [0.8, -0.9, 1.3, 0.2],
                [1.5, 0.1, -0.7, 0.6],
            ],
            dtype="float32",
        )
        return do1, [w1_0, w1_1]

    def _reference_dx(self, do1, expert_w1, tokens_per_expert):
        do1_np = do1.astype("float32").numpy().astype(np.float64)
        rows = []
        start = 0
        for i, n in enumerate(tokens_per_expert):
            if n == 0:
                continue
            chunk = do1_np[start : start + n]
            w = expert_w1[i].astype("float32").numpy().astype(np.float64)
            # dx_i = do1_i @ w1_i^T
            rows.append(chunk @ w.T)
            start += n
        return np.concatenate(rows, axis=0)

    def test_compatible_matches_hand_reference(self):
        do1, expert_w1 = self._fixed_inputs()
        node = _make_split_group_node(True)
        node.tokens_per_expert = [2, 1]
        dx = node.bwd_gate_up_input_bf16(do1, expert_w1)
        ref = self._reference_dx(do1, expert_w1, [2, 1])
        self.assertEqual(dx.shape, [3, 5])
        np.testing.assert_allclose(
            dx.astype("float32").numpy(), ref, rtol=1e-5, atol=1e-6
        )

    def test_compatible_and_default_branches_agree(self):
        do1, expert_w1 = self._fixed_inputs()
        node_c = _make_split_group_node(True)
        node_c.tokens_per_expert = [2, 1]
        node_d = _make_split_group_node(False)
        node_d.tokens_per_expert = [2, 1]
        dx_c = node_c.bwd_gate_up_input_bf16(do1, expert_w1)
        dx_d = node_d.bwd_gate_up_input_bf16(do1, expert_w1)
        np.testing.assert_allclose(
            dx_c.astype("float32").numpy(),
            dx_d.astype("float32").numpy(),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_zero_token_expert_is_skipped(self):
        do1, expert_w1 = self._fixed_inputs()
        # Only 2 tokens present; first expert receives none.
        do1 = do1[:2]
        node = _make_split_group_node(True)
        node.tokens_per_expert = [0, 2]
        dx = node.bwd_gate_up_input_bf16(do1, expert_w1)
        ref = self._reference_dx(do1, expert_w1, [0, 2])
        self.assertEqual(dx.shape, [2, 5])
        np.testing.assert_allclose(
            dx.astype("float32").numpy(), ref, rtol=1e-5, atol=1e-6
        )


@unittest.skipUnless(
    _DEPS_OK, f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}"
)
class TestFwdDownBf16SwigluAccuracyCompatible(unittest.TestCase):
    """With the flag set, the split-group SwiGLU forward computes
    ``o2 = silu(gate) * up * probs`` in fp32 and rounds once. Using an
    identity down-proj weight makes ``o3 == o2``, so ``fwd_down_bf16`` output
    equals a hand-written silu reference (and must NOT equal a gelu one)."""

    def _fixed_inputs(self):
        tokens, inter = 3, 2
        # o1 = [gate | up], each [tokens, inter].
        o1 = paddle.to_tensor(
            [
                [0.5, -1.0, 2.0, 0.25],
                [1.5, 0.5, -0.5, 1.0],
                [-2.0, 1.0, 0.75, -0.5],
            ],
            dtype="float32",
        )
        probs = paddle.to_tensor([[0.2], [0.7], [1.3]], dtype="float32")
        expert_w2 = [paddle.eye(inter, dtype="float32")]
        return tokens, inter, o1, probs, expert_w2

    def test_forward_equals_silu_reference(self):
        tokens, inter, o1, probs, expert_w2 = self._fixed_inputs()
        node = _make_split_group_node(True, "swiglu")
        node.tokens_per_expert = [tokens]
        o3 = node.fwd_down_bf16(o1, probs, expert_w2)

        o1_np = o1.numpy().astype(np.float64)
        gate, up = o1_np[:, :inter], o1_np[:, inter:]
        p = probs.numpy().astype(np.float64)
        silu_ref = _silu(gate) * up * p
        self.assertEqual(o3.shape, [tokens, inter])
        np.testing.assert_allclose(
            o3.astype("float32").numpy(), silu_ref, rtol=1e-5, atol=1e-6
        )
        # A gelu-based activation gives a different result, so the branch is
        # genuinely silu, not accidentally matching another activation.
        gelu = (
            0.5
            * gate
            * (
                1.0
                + np.tanh(np.sqrt(2.0 / np.pi) * (gate + 0.044715 * gate**3))
            )
        )
        gelu_ref = gelu * up * p
        self.assertFalse(
            np.allclose(
                o3.astype("float32").numpy(), gelu_ref, rtol=1e-3, atol=1e-3
            )
        )


@unittest.skipUnless(
    _DEPS_OK, f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}"
)
class TestBwdDownInputBf16SwigluAccuracyCompatible(unittest.TestCase):
    """The accuracy-compatible split-group SwiGLU backward:
      1. derives ``do2_s`` per expert via ``unzipped_grad_i @ expert_w2[i].T``;
      2. runs autograd through ``o2 = silu(gate) * up * probs`` in fp32.
    ``bwd_down_input_bf16`` returns ``(do1, o2_s, probs_grad)``; each is
    checked against an independent analytic SwiGLU gradient."""

    def _fixed_inputs(self):
        total, hidden, inter = 3, 3, 2
        # unzipped_grad has ``hidden`` cols; expert_w2[i] is [inter, hidden],
        # transposed to [hidden, inter] so do2_s has ``inter`` cols.
        unzipped_grad = paddle.to_tensor(
            [
                [0.5, -1.0, 0.25],
                [1.5, 0.5, -0.5],
                [-2.0, 1.0, 0.75],
            ],
            dtype="float32",
        )
        w2_0 = paddle.to_tensor(
            [[0.1, 0.2, 0.3], [-0.4, 0.5, -0.6]], dtype="float32"
        )
        w2_1 = paddle.to_tensor(
            [[0.7, -0.8, 0.9], [1.0, 0.15, -0.25]], dtype="float32"
        )
        # o1 = [gate | up], each [total, inter].
        o1 = paddle.to_tensor(
            [
                [0.5, -1.0, 2.0, 0.25],
                [1.5, 0.5, -0.5, 1.0],
                [-2.0, 1.0, 0.75, -0.5],
            ],
            dtype="float32",
        )
        probs = paddle.to_tensor([[0.2], [0.7], [1.3]], dtype="float32")
        return total, hidden, inter, unzipped_grad, [w2_0, w2_1], o1, probs

    def _reference(self, unzipped_grad, expert_w2, o1, probs, inter, split):
        ug = unzipped_grad.numpy().astype(np.float64)
        do2_rows = []
        start = 0
        for i, n in enumerate(split):
            if n == 0:
                continue
            w = expert_w2[i].numpy().astype(np.float64)  # [inter, hidden]
            do2_rows.append(ug[start : start + n] @ w.T)  # [n, inter]
            start += n
        do2_s = np.concatenate(do2_rows, axis=0)  # [total, inter]

        o1_np = o1.numpy().astype(np.float64)
        gate, up = o1_np[:, :inter], o1_np[:, inter:]
        p = probs.numpy().astype(np.float64)  # [total, 1]

        o2_s = _silu(gate) * up * p
        d_gate = do2_s * _silu_grad(gate) * up * p
        d_up = do2_s * _silu(gate) * p
        do1 = np.concatenate([d_gate, d_up], axis=-1)
        probs_grad = (_silu(gate) * up * do2_s).sum(axis=-1, keepdims=True)
        return do1, o2_s, probs_grad

    def test_backward_matches_analytic_swiglu(self):
        total, hidden, inter, ug, expert_w2, o1, probs = self._fixed_inputs()
        node = _make_split_group_node(True, "swiglu")
        node.tokens_per_expert = [2, 1]
        do1, o2_s, probs_grad = node.bwd_down_input_bf16(
            expert_w2, ug, o1, probs
        )

        ref_do1, ref_o2s, ref_pg = self._reference(
            ug, expert_w2, o1, probs, inter, [2, 1]
        )
        self.assertEqual(do1.shape, [total, 2 * inter])
        np.testing.assert_allclose(
            do1.astype("float32").numpy(), ref_do1, rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            o2_s.astype("float32").numpy(), ref_o2s, rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            probs_grad.astype("float32").numpy(),
            ref_pg,
            rtol=1e-4,
            atol=1e-5,
        )

    def test_backward_probs_grad_is_scale_sensitive(self):
        # Guard against a regression that drops the ``up`` (linear) factor from
        # dL/dprobs: the reference below deliberately omits it and must differ.
        total, hidden, inter, ug, expert_w2, o1, probs = self._fixed_inputs()
        node = _make_split_group_node(True, "swiglu")
        node.tokens_per_expert = [2, 1]
        _, _, probs_grad = node.bwd_down_input_bf16(expert_w2, ug, o1, probs)

        _, _, ref_pg = self._reference(ug, expert_w2, o1, probs, inter, [2, 1])
        # Wrong reference: silu(gate) * do2_s summed (missing the ``up`` term).
        ug_np = ug.numpy().astype(np.float64)
        do2_rows, start = [], 0
        for i, n in enumerate([2, 1]):
            w = expert_w2[i].numpy().astype(np.float64)
            do2_rows.append(ug_np[start : start + n] @ w.T)
            start += n
        do2_s = np.concatenate(do2_rows, axis=0)
        gate = o1.numpy().astype(np.float64)[:, :inter]
        wrong_pg = (_silu(gate) * do2_s).sum(axis=-1, keepdims=True)

        np.testing.assert_allclose(
            probs_grad.astype("float32").numpy(), ref_pg, rtol=1e-4, atol=1e-5
        )
        self.assertFalse(
            np.allclose(
                probs_grad.astype("float32").numpy(),
                wrong_pg,
                rtol=1e-3,
                atol=1e-3,
            )
        )


if __name__ == "__main__":
    unittest.main()
