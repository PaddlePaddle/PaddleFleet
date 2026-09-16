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

"""Numeric contract tests for ``subbatch`` in
``paddlefleet.models.common.language_loss.language_loss``.

``subbatch`` is the memory-saving orchestrator the language-loss forward path
wraps around the cross-entropy call: it slices the batched positional args
along a chosen axis in fixed-size chunks, applies the wrapped function to each
chunk, and stitches the per-chunk outputs back together along ``out_idx``. Its
correctness is therefore purely about the *orchestration* -- slice boundaries,
lock-step slicing of several args, ``same_arg_idx`` reuse, pass-through of
non-batched args/kwargs, and the below-threshold short-circuit -- not about the
mathematics of the wrapped function.

The tests exercise the real ``subbatch`` (never a re-implementation) with a
small position-distinguishable ``arange`` fixture and hand-derived NumPy
references, so a wrong slice axis, dropped/duplicated chunk, mis-ordered concat
or ignored kwarg is rejected. ``subbatch`` slices real Paddle tensors and calls
``paddle.cat``; the whole suite is skipped with an honest reason when Paddle is
not importable in this environment (this machine is CPU-only with no Paddle
installed), so no assertion is faked into a pass.
"""

import unittest

try:
    import numpy as np
    import paddle

    # subbatch calls ``paddle.cat``, which paddlefleet exposes as an alias of
    # ``paddle.concat`` via this patch module at framework init. Import it so
    # the runtime precondition production relies on is established here too.
    try:
        import paddlefleet.utils.paddle_patch  # noqa: F401
    except Exception:
        pass

    from paddlefleet.models.common.language_loss.language_loss import subbatch

    _IMPORT_ERROR = None
except ImportError as exc:  # precise: only missing deps => honest skip
    subbatch = None
    _IMPORT_ERROR = exc

_PADDLE_AVAILABLE = _IMPORT_ERROR is None


@unittest.skipUnless(
    _PADDLE_AVAILABLE,
    f"paddle/paddlefleet not importable in this CPU-only environment: "
    f"{_IMPORT_ERROR}",
)
class SubbatchContractTest(unittest.TestCase):
    """subbatch slice/stitch orchestration against independent references."""

    def setUp(self):
        # Pure orchestration logic; pin to CPU for deterministic, device-free
        # execution when a GPU happens to be present under single_card_tests.
        try:
            paddle.set_device("cpu")
        except Exception:
            pass
        if not hasattr(paddle, "cat"):
            # Should have been provided by paddle_patch; establish the exact
            # documented alias rather than skip, so the stitch step can run.
            paddle.cat = paddle.concat

    def test_ragged_chunks_reconstruct_full_input(self):
        # arange => every element distinct, so a dropped/duplicated/reordered
        # chunk or a wrong slice start is visible. axis_width=5, bs=2 forces a
        # ragged final chunk [4:5] on top of [0:2] and [2:4].
        data = paddle.arange(15, dtype="float32").reshape([5, 3])
        wrapped = subbatch(lambda x: x, arg_idx=[0], axis=[0], bs=2, out_idx=0)
        out = wrapped(data)
        np.testing.assert_array_equal(
            out.numpy(), np.arange(15, dtype="float32").reshape(5, 3)
        )

    def test_below_threshold_calls_function_on_full_unsliced_args(self):
        # axis_width=3 < bs=8 => short-circuit: f must see the WHOLE tensor
        # exactly once, with no slicing applied.
        data = paddle.arange(12, dtype="float32").reshape([3, 4])
        seen = []

        def spy(x):
            seen.append(x.numpy().copy())
            return x * 2.0

        wrapped = subbatch(spy, arg_idx=[0], axis=[0], bs=8, out_idx=0)
        out = wrapped(data)

        self.assertEqual(len(seen), 1)
        np.testing.assert_array_equal(
            seen[0], np.arange(12, dtype="float32").reshape(3, 4)
        )
        np.testing.assert_array_equal(
            out.numpy(), np.arange(12, dtype="float32").reshape(3, 4) * 2.0
        )

    def test_two_args_sliced_in_lockstep_along_axis1(self):
        # Mirrors the real language-loss call: arg_idx=[0,1], axis=[1,1],
        # out_idx=1. logits[2,5,3] and labels[2,5] must be sliced together on
        # axis 1 and the outputs concatenated on axis 1.
        logits = paddle.arange(30, dtype="float32").reshape([2, 5, 3])
        labels = paddle.arange(10, dtype="float32").reshape([2, 5])

        def combine(lg, lb):
            return lg.sum(axis=-1) + lb

        wrapped = subbatch(
            combine, arg_idx=[0, 1], axis=[1, 1], bs=2, out_idx=1
        )
        out = wrapped(logits, labels)

        lg_ref = np.arange(30, dtype="float32").reshape(2, 5, 3)
        lb_ref = np.arange(10, dtype="float32").reshape(2, 5)
        expected = lg_ref.sum(axis=-1) + lb_ref
        self.assertEqual(list(out.shape), [2, 5])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_same_arg_idx_reuses_sliced_first_arg(self):
        # same_arg_idx={1:0} => positional arg 1 is REPLACED by the sliced
        # arg 0 each chunk; whatever is passed as arg 1 must be ignored.
        base = paddle.arange(8, dtype="float32").reshape([4, 2])
        decoy = paddle.full([4, 2], -999.0, dtype="float32")

        wrapped = subbatch(
            lambda a, b: a + b,
            arg_idx=[0],
            axis=[0],
            bs=2,
            out_idx=0,
            same_arg_idx={1: 0},
        )
        out = wrapped(base, decoy)

        # a + b with b := sliced a  =>  2 * base, decoy fully ignored.
        expected = 2.0 * np.arange(8, dtype="float32").reshape(4, 2)
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_non_batched_positional_and_kwargs_pass_through_unsliced(self):
        # Only arg 0 is batched; the scalar positional `scale` and keyword
        # `bias` must reach f untouched on every chunk.
        data = paddle.arange(6, dtype="float32").reshape([3, 2])

        def affine(x, scale, bias=0.0):
            return x * scale + bias

        wrapped = subbatch(affine, arg_idx=[0], axis=[0], bs=2, out_idx=0)
        out = wrapped(data, 3.0, bias=1.0)

        expected = np.arange(6, dtype="float32").reshape(3, 2) * 3.0 + 1.0
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_arg_idx_axis_length_mismatch_raises(self):
        data = paddle.arange(8, dtype="float32").reshape([4, 2])
        wrapped = subbatch(
            lambda a, b: a + b, arg_idx=[0, 1], axis=[0], bs=2, out_idx=0
        )
        with self.assertRaises(AssertionError):
            wrapped(data, data)

    def test_unequal_batched_widths_raise(self):
        a = paddle.arange(8, dtype="float32").reshape([4, 2])
        b = paddle.arange(6, dtype="float32").reshape([3, 2])
        wrapped = subbatch(
            lambda x, y: x.sum() + y.sum(),
            arg_idx=[0, 1],
            axis=[0, 0],
            bs=2,
            out_idx=0,
        )
        with self.assertRaises(AssertionError):
            wrapped(a, b)


if __name__ == "__main__":
    unittest.main()
