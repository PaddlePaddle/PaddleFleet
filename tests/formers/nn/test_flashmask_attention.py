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

"""Behavior tests for ``flashmask_attention_forward``.

Module under test:
    ``paddlefleet.nn.attention.flashmask_attention.flashmask_attention_forward``

Scope and environment (see unit-test-rules.md, "模型与训练目标" / 无卡):
    The numerical attention kernels ``flashmask_attention`` (paddle flash-attn v2/v3)
    and ``sink_attention_forward`` are GPU-only kernels; their visibility/softmax
    numerics cannot be executed on CPU and are NOT validated here. That coverage
    lives in ``tests/formers/nn/test_attention.py::test_correctness`` (single-card).

    What IS CPU-testable, and what this file verifies with independent expected
    values (not shape-only), is the orchestration/control logic that
    ``flashmask_attention_forward`` owns:
      * the [b, h, l, d] -> [b, l, h, d] layout transform applied to q/k/v,
      * the ``is_causal`` decision derived from the FlashMask sparse-index shape
        (``shape[-1]==1`` -> causal, ``shape[-1]==4`` -> non-causal, ``==2`` keeps
        the caller's value), and from the module / seq-len when no indices are
        given,
      * the 3-D -> 4-D unsqueeze of the sparse indices with content preserved,
      * forwarding of the exact sparse indices to the kernel,
      * collaborator selection (plain flashmask vs. sink path) and the
        dropout/scale/causal arguments forwarded to the sink kernel,
      * the final [b, l, h, d] -> [b, l, h*d] output flatten.

    The GPU kernels are mocked with input-derived markers so we can assert the
    real arguments they receive and that their output flows through the reshape
    unchanged. We explicitly do NOT assert any attention numerics.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import paddle

FLASH_MODULE = "paddlefleet.nn.attention.flashmask_attention"


def _module(is_causal=True, with_attr=True):
    """Minimal stand-in for the attention nn.Layer: only ``is_causal`` is read."""
    if with_attr:
        return SimpleNamespace(is_causal=is_causal)
    return (
        SimpleNamespace()
    )  # exercises getattr(module, "is_causal", True) default


def _arange(shape):
    """Deterministic, position-distinguishable tensor of the given shape."""
    n = int(np.prod(shape))
    return paddle.arange(n, dtype="float32").reshape(shape)


class _KernelSpy:
    """Records the arguments a mocked kernel receives and returns a marker.

    The marker is derived from the query it receives ([b, l, h, d]) so the test
    can independently predict the flattened output and confirm the kernel output
    is what flows out of ``flashmask_attention_forward``.
    """

    def __init__(self):
        self.calls = 0
        self.q = self.k = self.v = None
        self.sink = None
        self.kwargs = None
        self.marker = None

    def flash(self, query, key, value, **kwargs):
        self.calls += 1
        self.q, self.k, self.v = query, key, value
        self.kwargs = kwargs
        self.marker = (
            _arange(query.shape) + 7.0
        )  # +offset: distinct from inputs
        return self.marker

    def sink_fwd(self, query, key, value, sink, **kwargs):
        self.calls += 1
        self.q, self.k, self.v, self.sink = query, key, value, sink
        self.kwargs = kwargs
        self.marker = _arange(query.shape) + 13.0
        return self.marker


class TestFlashmaskAttentionForward(unittest.TestCase):
    # Distinct sizes so an axis swap or wrong-dim reshape cannot pass silently.
    B, H, L, D = 2, 3, 4, 5

    def setUp(self):
        paddle.set_device("cpu")
        # Inputs are in [b, h, l, d] layout (as the attention interface delivers).
        self.query = _arange([self.B, self.H, self.L, self.D])
        self.key = _arange([self.B, self.H, self.L, self.D]) + 100.0
        self.value = _arange([self.B, self.H, self.L, self.D]) + 200.0

    def _call(self, spy, **kwargs):
        from paddlefleet.nn.attention.flashmask_attention import (
            flashmask_attention_forward,
        )

        return flashmask_attention_forward(
            kwargs.pop("module", _module()),
            self.query,
            self.key,
            self.value,
            **kwargs,
        )

    # --- layout transform -------------------------------------------------

    def test_qkv_transposed_to_blhd_layout(self):
        """q/k/v must be transposed [b, h, l, d] -> [b, l, h, d] before the kernel.

        Independent reference uses perm=[0, 2, 1, 3]; we compare full content,
        not just shape, so a wrong axis swap is rejected.
        """
        spy = _KernelSpy()
        indices = paddle.zeros([self.B, self.H, self.L, 1], dtype="int32")
        with mock.patch(
            f"{FLASH_MODULE}.flashmask_attention", side_effect=spy.flash
        ):
            self._call(spy, attn_mask_startend_row_indices=indices)

        exp_q = paddle.transpose(self.query, perm=[0, 2, 1, 3])
        exp_k = paddle.transpose(self.key, perm=[0, 2, 1, 3])
        exp_v = paddle.transpose(self.value, perm=[0, 2, 1, 3])
        self.assertEqual(spy.q.shape, [self.B, self.L, self.H, self.D])
        np.testing.assert_array_equal(spy.q.numpy(), exp_q.numpy())
        np.testing.assert_array_equal(spy.k.numpy(), exp_k.numpy())
        np.testing.assert_array_equal(spy.v.numpy(), exp_v.numpy())

    def test_output_is_kernel_output_flattened(self):
        """Output must be the kernel output reshaped [b, l, h, d] -> [b, l, h*d]."""
        spy = _KernelSpy()
        indices = paddle.zeros([self.B, self.H, self.L, 1], dtype="int32")
        with mock.patch(
            f"{FLASH_MODULE}.flashmask_attention", side_effect=spy.flash
        ):
            out, weights = self._call(
                spy, attn_mask_startend_row_indices=indices
            )

        self.assertIsNone(weights)
        expected = spy.marker.reshape([self.B, self.L, self.H * self.D])
        self.assertEqual(out.shape, [self.B, self.L, self.H * self.D])
        np.testing.assert_array_equal(out.numpy(), expected.numpy())

    # --- is_causal derived from the sparse-index shape --------------------

    def test_indices_shape1_forces_causal_true(self):
        """FlashMask bound_num==1 => unidirectional (causal=True), overriding caller."""
        spy = _KernelSpy()
        indices = paddle.zeros([self.B, self.H, self.L, 1], dtype="int32")
        with mock.patch(
            f"{FLASH_MODULE}.flashmask_attention", side_effect=spy.flash
        ):
            self._call(
                spy,
                attn_mask_startend_row_indices=indices,
                is_causal=False,  # explicit False must be overridden to True
            )
        self.assertIs(spy.kwargs["causal"], True)

    def test_indices_shape4_forces_causal_false(self):
        """FlashMask bound_num==4 => bidirectional (causal=False), overriding caller."""
        spy = _KernelSpy()
        indices = paddle.zeros([self.B, self.H, self.L, 4], dtype="int32")
        with mock.patch(
            f"{FLASH_MODULE}.flashmask_attention", side_effect=spy.flash
        ):
            self._call(
                spy,
                attn_mask_startend_row_indices=indices,
                is_causal=True,  # explicit True must be overridden to False
            )
        self.assertIs(spy.kwargs["causal"], False)

    def test_indices_shape2_preserves_explicit_causal(self):
        """bound_num==2 is ambiguous: neither override fires, caller value is kept."""
        for explicit in (True, False):
            with self.subTest(is_causal=explicit):
                spy = _KernelSpy()
                indices = paddle.zeros(
                    [self.B, self.H, self.L, 2], dtype="int32"
                )
                with mock.patch(
                    f"{FLASH_MODULE}.flashmask_attention", side_effect=spy.flash
                ):
                    self._call(
                        spy,
                        attn_mask_startend_row_indices=indices,
                        is_causal=explicit,
                    )
                self.assertIs(spy.kwargs["causal"], explicit)

    # --- 3-D indices unsqueeze + content forwarding -----------------------

    def test_3d_indices_unsqueezed_with_content_preserved(self):
        """3-D indices become 4-D via unsqueeze(-1); values must be preserved."""
        spy = _KernelSpy()
        # Distinguishable content so unsqueeze correctness (not just ndim) is checked.
        base = _arange([self.B, self.H, self.L]).astype("int32")
        with mock.patch(
            f"{FLASH_MODULE}.flashmask_attention", side_effect=spy.flash
        ):
            self._call(spy, attn_mask_startend_row_indices=base)

        passed = spy.kwargs["startend_row_indices"]
        self.assertEqual(passed.ndim, 4)
        self.assertEqual(passed.shape, [self.B, self.H, self.L, 1])
        np.testing.assert_array_equal(
            passed.numpy(), base.unsqueeze(-1).numpy()
        )
        # bound_num==1 after unsqueeze -> causal forced True.
        self.assertIs(spy.kwargs["causal"], True)

    def test_4d_indices_forwarded_unchanged(self):
        """4-D indices reach the kernel with identical shape and content."""
        spy = _KernelSpy()
        indices = (_arange([self.B, self.H, self.L, 4])).astype("int32")
        with mock.patch(
            f"{FLASH_MODULE}.flashmask_attention", side_effect=spy.flash
        ):
            self._call(spy, attn_mask_startend_row_indices=indices)

        passed = spy.kwargs["startend_row_indices"]
        self.assertEqual(passed.shape, [self.B, self.H, self.L, 4])
        np.testing.assert_array_equal(passed.numpy(), indices.numpy())

    # --- is_causal inference without indices ------------------------------

    def test_no_indices_infers_causal_from_module_and_seqlen(self):
        """With no indices/explicit flag: causal = (seq_len>1) and module.is_causal."""
        cases = [
            # (module, seq_len, expected_causal, note)
            (
                _module(is_causal=True),
                self.L,
                True,
                "causal module, multi-token",
            ),
            (_module(is_causal=False), self.L, False, "non-causal module"),
            (_module(is_causal=True), 1, False, "single token -> not causal"),
            (
                _module(with_attr=False),
                self.L,
                True,
                "missing attr -> default True",
            ),
        ]
        for module, seq_len, expected, note in cases:
            with self.subTest(note=note):
                spy = _KernelSpy()
                q = _arange([self.B, self.H, seq_len, self.D])
                k = _arange([self.B, self.H, seq_len, self.D])
                v = _arange([self.B, self.H, seq_len, self.D])
                with mock.patch(
                    f"{FLASH_MODULE}.flashmask_attention", side_effect=spy.flash
                ):
                    from paddlefleet.nn.attention.flashmask_attention import (
                        flashmask_attention_forward,
                    )

                    flashmask_attention_forward(
                        module, q, k, v, attn_mask_startend_row_indices=None
                    )
                self.assertIs(spy.kwargs["causal"], expected)

    def test_explicit_is_causal_without_indices_propagates(self):
        """An explicit is_causal (no indices) is passed straight through."""
        for explicit in (True, False):
            with self.subTest(is_causal=explicit):
                spy = _KernelSpy()
                with mock.patch(
                    f"{FLASH_MODULE}.flashmask_attention", side_effect=spy.flash
                ):
                    self._call(
                        spy,
                        attn_mask_startend_row_indices=None,
                        is_causal=explicit,
                    )
                self.assertIs(spy.kwargs["causal"], explicit)

    # --- collaborator selection: sink path --------------------------------

    def test_sink_routes_to_sink_kernel_with_params(self):
        """A non-None sink routes to sink_attention_forward, not plain flashmask,
        and forwards dropout_p / softmax_scale / the derived causal flag.
        """
        spy = _KernelSpy()
        flash_spy = _KernelSpy()
        indices = paddle.zeros([self.B, self.H, self.L, 4], dtype="int32")
        sink = _arange([self.H])
        with (
            mock.patch(
                f"{FLASH_MODULE}.sink_attention_forward",
                side_effect=spy.sink_fwd,
            ),
            mock.patch(
                f"{FLASH_MODULE}.flashmask_attention",
                side_effect=flash_spy.flash,
            ),
        ):
            out, weights = self._call(
                spy,
                attn_mask_startend_row_indices=indices,
                sink=sink,
                scaling=0.25,
                dropout=0.1,
            )

        self.assertEqual(spy.calls, 1)
        self.assertEqual(flash_spy.calls, 0)  # plain kernel must NOT run
        self.assertIs(spy.sink, sink)
        self.assertEqual(spy.kwargs["dropout_p"], 0.1)
        self.assertEqual(spy.kwargs["softmax_scale"], 0.25)
        # bound_num==4 -> causal False also drives the sink kernel.
        self.assertIs(spy.kwargs["causal"], False)
        # sink indices forwarded intact.
        np.testing.assert_array_equal(
            spy.kwargs["startend_row_indices"].numpy(), indices.numpy()
        )
        # Output still flows through the flatten.
        self.assertIsNone(weights)
        expected = spy.marker.reshape([self.B, self.L, self.H * self.D])
        np.testing.assert_array_equal(out.numpy(), expected.numpy())


if __name__ == "__main__":
    unittest.main()
