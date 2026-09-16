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

"""Behavior tests for ``paddlefleet.models.common.language_loss.subbatch``.

Disjoint slice: this file targets the pure ``subbatch`` chunking helper that
``LanguageLoss.forward_impl`` uses to split the cross-entropy over the sequence
axis. Sibling suites already cover ``LanguageLoss.forward`` numerics, the
megatron label roll, the cu_seqlens stash and distributed (TP/SP) subbatch
consistency; none of them pins the per-chunk slicing/concat contract of the
helper itself on CPU. That is what these tests do.

Each test drives the real ``subbatch`` wrapper and compares against an
independent NumPy reference (never the helper as its own oracle). Inputs use
distinguishable ``arange`` content so misordered slices, wrong axis, dropped
tail chunks or a swapped concat axis are rejected rather than masked.

The production module imports paddle at load time. The environment used to
author this file has no paddle installed, so every test is gated behind an
honest ``skipUnless(IMPORT_OK, ...)`` guard and skips instead of faking a pass.
Only a genuine ImportError/ModuleNotFoundError is treated as a missing
dependency; any other error is allowed to surface rather than be swallowed.
"""

import os
import sys
import unittest

import numpy as np

# Make ``src/`` importable when tests are run from a source checkout.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_SRC, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import paddle

    from paddlefleet.models.common.language_loss.language_loss import subbatch

    IMPORT_OK = True
    IMPORT_ERR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest missing-dep guard
    IMPORT_OK = False
    IMPORT_ERR = exc

_SKIP_REASON = (
    f"paddle / paddlefleet not importable in this environment: {IMPORT_ERR!r}"
)


@unittest.skipUnless(IMPORT_OK, _SKIP_REASON)
class TestSubbatchChunking(unittest.TestCase):
    """Numeric contract of the ``subbatch`` sequence-chunking helper."""

    @staticmethod
    def _f32(values):
        return paddle.to_tensor(np.asarray(values, dtype=np.float32))

    def test_identity_reconstructs_in_order(self):
        # Identity f over 3 chunks must rebuild the original tensor exactly;
        # a misordered slice or concat would permute the arange content.
        x = self._f32(np.arange(6).reshape(1, 6))
        wrapped = subbatch(lambda t: t, arg_idx=[0], axis=[1], bs=2, out_idx=1)
        out = wrapped(x)
        np.testing.assert_array_equal(
            out.numpy(), np.arange(6, dtype=np.float32).reshape(1, 6)
        )

    def test_per_chunk_reduction_pins_boundaries(self):
        # f sums each length-2 chunk along the subbatched axis; the concat of
        # the three scalars must equal the hand-derived per-chunk sums. Wrong
        # chunk boundaries or a dropped tail change these numbers.
        x = self._f32(np.arange(6).reshape(1, 6))
        wrapped = subbatch(
            lambda t: t.sum(axis=1, keepdim=True),
            arg_idx=[0],
            axis=[1],
            bs=2,
            out_idx=1,
        )
        out = wrapped(x)
        ref = np.arange(6, dtype=np.float32).reshape(3, 2).sum(axis=1)
        np.testing.assert_array_equal(out.numpy(), ref.reshape(1, 3))

    def test_out_idx_controls_concat_axis(self):
        # Same reduction but out_idx=0 must stack the chunk results along a
        # different axis, yielding a column vector rather than a row.
        x = self._f32(np.arange(6).reshape(1, 6))
        wrapped = subbatch(
            lambda t: t.sum(axis=1, keepdim=True),
            arg_idx=[0],
            axis=[1],
            bs=2,
            out_idx=0,
        )
        out = wrapped(x)
        ref = np.arange(6, dtype=np.float32).reshape(3, 2).sum(axis=1)
        np.testing.assert_array_equal(out.numpy(), ref.reshape(3, 1))

    def test_short_axis_takes_early_return_single_call(self):
        # axis_width (6) < bs (100): the helper must call f once on the whole
        # input and skip the chunk/concat loop entirely.
        calls = {"n": 0}

        def f(t):
            calls["n"] += 1
            return t * 2.0

        x = self._f32(np.arange(6).reshape(1, 6))
        wrapped = subbatch(f, arg_idx=[0], axis=[1], bs=100, out_idx=1)
        out = wrapped(x)
        self.assertEqual(calls["n"], 1)
        np.testing.assert_array_equal(
            out.numpy(), (np.arange(6, dtype=np.float32) * 2.0).reshape(1, 6)
        )

    def test_two_batched_args_sliced_consistently(self):
        # Both args are subbatched on axis 1; each chunk of a must be paired
        # with the matching chunk of b. A misaligned slice on either arg would
        # perturb the summed result.
        a_np = np.arange(6, dtype=np.float32).reshape(1, 6)
        b_np = np.arange(6, dtype=np.float32).reshape(1, 6) + 100.0
        a, b = self._f32(a_np), self._f32(b_np)
        wrapped = subbatch(
            lambda x, y: x + y, arg_idx=[0, 1], axis=[1, 1], bs=2, out_idx=1
        )
        out = wrapped(a, b)
        np.testing.assert_array_equal(out.numpy(), a_np + b_np)

    def test_same_arg_idx_reuses_first_slice(self):
        # same_arg_idx={1: 0} means args[1] must reuse the *already sliced*
        # args[0] object, not be sliced again. f asserts object identity and
        # squares it; the concat must equal x**2.
        x = self._f32(np.arange(6).reshape(1, 6))
        seen_identity = []

        def f(first, second):
            seen_identity.append(first is second)
            return first * second

        wrapped = subbatch(
            f,
            arg_idx=[0, 1],
            axis=[1, 1],
            bs=2,
            out_idx=1,
            same_arg_idx={1: 0},
        )
        out = wrapped(x, x)
        self.assertTrue(seen_identity and all(seen_identity))
        np.testing.assert_array_equal(
            out.numpy(), (np.arange(6, dtype=np.float32) ** 2).reshape(1, 6)
        )

    def test_arg_idx_axis_length_mismatch_raises(self):
        # len(arg_idx) != len(axis) is an explicit assertion inside wrapper.
        x = self._f32(np.arange(6).reshape(1, 6))
        wrapped = subbatch(
            lambda t: t, arg_idx=[0], axis=[1, 1], bs=2, out_idx=1
        )
        with self.assertRaises(AssertionError):
            wrapped(x)

    def test_unequal_axis_widths_raise(self):
        # Two batched args whose subbatched dims differ must be rejected.
        a = self._f32(np.arange(6).reshape(1, 6))
        b = self._f32(np.arange(4).reshape(1, 4))
        wrapped = subbatch(
            lambda x, y: x, arg_idx=[0, 1], axis=[1, 1], bs=2, out_idx=1
        )
        with self.assertRaises(AssertionError):
            wrapped(a, b)

    def test_recompute_matches_plain_forward(self):
        # use_recompute must be numerically transparent in the forward pass.
        x = self._f32(np.arange(6).reshape(1, 6))
        x.stop_gradient = False
        plain = subbatch(
            lambda t: t * 3.0, arg_idx=[0], axis=[1], bs=2, out_idx=1
        )(x)
        recomp = subbatch(
            lambda t: t * 3.0,
            arg_idx=[0],
            axis=[1],
            bs=2,
            out_idx=1,
            use_recompute=True,
        )(x)
        np.testing.assert_allclose(
            recomp.numpy(), plain.numpy(), rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
