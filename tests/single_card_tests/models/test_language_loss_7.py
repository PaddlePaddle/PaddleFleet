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

"""Unit tests for the ``LanguageLoss`` base module numerics.

Scope (disjoint from the separate-MTP-head plumbing exercised elsewhere):
the CPU-executable, non-distributed pieces of
``paddlefleet.models.common.language_loss.language_loss``:

- ``LanguageLoss.forward_impl`` — the real cross-entropy entry: label
  shift is already applied by the caller, so here we verify the CE math,
  the ``!= ignored_index`` loss mask, and token-mean normalization
  (sum / valid-token-count), against a hand-derived NumPy reference.
- ``subbatch`` — the chunked-apply wrapper: correct per-chunk slices,
  order-preserving concatenation, and its guard assertions.
- ``_tensor_md5`` — the debug hash, against an independent hashlib ref.
- ``_loss_md5_enabled`` / ``_use_accuracy_compatible_kernel`` — the exact
  ``== "1"`` env gates.
- ``_print_scalar_loss_md5`` — prints only when enabled, with the hash.

Every case needs Paddle for the real ops; without it the file skips with an
honest missing-dependency reason rather than faking a pass.
"""

import hashlib
import io
import os
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

try:
    import numpy as np
    import paddle

    from paddlefleet.models.common.language_loss.language_loss import (
        LanguageLoss,
        _loss_md5_enabled,
        _print_scalar_loss_md5,
        _tensor_md5,
        _use_accuracy_compatible_kernel,
        subbatch,
    )
except ImportError as exc:  # honest: only a missing dependency skips
    np = None
    paddle = None
    LanguageLoss = None
    _loss_md5_enabled = None
    _print_scalar_loss_md5 = None
    _tensor_md5 = None
    _use_accuracy_compatible_kernel = None
    subbatch = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

_HAS_PADDLE = _IMPORT_ERROR is None
_SKIP_REASON = f"paddle/paddlefleet import unavailable: {_IMPORT_ERROR}"


def _make_language_loss():
    """Build a ``LanguageLoss`` whose ``forward_impl`` runs the real CE path.

    ``__new__`` skips ``FleetLayer.__init__`` (which needs the distributed
    runtime); only the plain scaffold attributes ``forward_impl`` reads are
    populated, and ``loss_func`` is a *real* ``CrossEntropyLoss``. No test
    asserts these scaffold values — they exist solely so the production
    method executes; correctness is judged by comparing its numeric output
    to an independent NumPy reference.
    """
    layer = LanguageLoss.__new__(LanguageLoss)
    layer.config = SimpleNamespace(
        gpt_model_use_experimental_version=False,
        sequence_parallel=False,
        cp_balance_mode="contiguous_allgather",
        fused_linear_ce_loss_chunk=0,
    )
    layer.ignored_index = -100
    layer.enable_parallel_cross_entropy = False
    layer.use_accuracy_compatible = False
    layer.use_subbatch = False
    layer.loss_subbatch_sequence_length = 0
    layer.loss_func = paddle.nn.CrossEntropyLoss(reduction="none")
    return layer


def _numpy_token_mean_ce(logits, labels, ignore_index=-100):
    """Independent token-averaged cross-entropy (natural log).

    Mirrors the non-fused, non-CP, non-experimental contract of
    ``forward_impl``: per-token CE over classes, ignore ``-100`` targets,
    then sum over valid tokens divided by the valid-token count.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    bsz, seq, _ = logits.shape
    shifted = logits - logits.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=-1)) + logits.max(axis=-1)
    total = 0.0
    count = 0
    for b in range(bsz):
        for s in range(seq):
            lab = int(labels[b, s])
            if lab == ignore_index:
                continue
            total += float(logsumexp[b, s] - logits[b, s, lab])
            count += 1
    return total / count, count


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestLanguageLossForwardImpl(unittest.TestCase):
    """``forward_impl`` computes masked, token-averaged cross-entropy."""

    def test_matches_independent_token_mean_ce(self):
        layer = _make_language_loss()
        # Distinguishable logits; targets pick specific, differing classes so
        # a label/logit misalignment would move the gathered CE terms.
        logits = paddle.to_tensor(
            [
                [
                    [2.0, 0.5, -1.0, 0.0],
                    [0.0, 3.0, 1.0, -2.0],
                    [1.0, 1.0, 4.0, 0.0],
                ],
                [
                    [-1.0, 2.0, 0.0, 1.0],
                    [3.0, -1.0, 0.5, 2.0],
                    [0.0, 0.0, 0.0, 5.0],
                ],
            ],
            dtype="float32",
        )
        labels = paddle.to_tensor([[0, 1, 2], [3, 0, 3]], dtype="int64")

        out = layer.forward_impl(logits, labels)
        ref, count = _numpy_token_mean_ce(logits.numpy(), labels.numpy())
        self.assertEqual(count, 6)  # all six tokens valid
        np.testing.assert_allclose(
            float(out.numpy()), ref, rtol=1e-4, atol=1e-5
        )

    def test_ignored_tokens_excluded_from_mean(self):
        layer = _make_language_loss()
        logits = paddle.to_tensor(
            [
                [
                    [2.0, 0.5, -1.0, 0.0],
                    [0.0, 3.0, 1.0, -2.0],
                    [1.0, 1.0, 4.0, 0.0],
                ],
                [
                    [-1.0, 2.0, 0.0, 1.0],
                    [3.0, -1.0, 0.5, 2.0],
                    [0.0, 0.0, 0.0, 5.0],
                ],
            ],
            dtype="float32",
        )
        # Three of six positions are masked out with -100.
        labels = paddle.to_tensor(
            [[0, -100, 2], [-100, 0, -100]], dtype="int64"
        )

        out = layer.forward_impl(logits, labels)
        ref, count = _numpy_token_mean_ce(logits.numpy(), labels.numpy())
        self.assertEqual(count, 3)
        np.testing.assert_allclose(
            float(out.numpy()), ref, rtol=1e-4, atol=1e-5
        )

        # Negative control: the denominator must be the valid-token count (3),
        # not the total token count (6). Dividing by 6 would give ref*3/6.
        wrong_if_divided_by_total = ref * count / 6.0
        self.assertNotAlmostEqual(
            float(out.numpy()), wrong_if_divided_by_total, places=4
        )

    def test_all_ignored_returns_zero(self):
        layer = _make_language_loss()
        logits = paddle.to_tensor(
            [[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]], dtype="float32"
        )
        labels = paddle.full([1, 2], -100, dtype="int64")

        out = layer.forward_impl(logits, labels)
        self.assertEqual(float(out.numpy()), 0.0)


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestSubbatch(unittest.TestCase):
    """``subbatch`` chunks the batched axis, applies ``f``, and reassembles."""

    def test_short_circuit_when_axis_smaller_than_bs(self):
        # axis_width (2) < bs (5): no chunking, f is called exactly once on
        # the whole input and its result returned unchanged.
        calls = []

        def f(x):
            calls.append(x.numpy().copy())
            return x + 100.0

        inp = paddle.to_tensor([[1.0, 2.0]], dtype="float32")
        wrapped = subbatch(f, arg_idx=[0], axis=[1], bs=5, out_idx=1)
        out = wrapped(inp)

        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0], [[1.0, 2.0]])
        np.testing.assert_array_equal(out.numpy(), [[101.0, 102.0]])

    def test_multi_chunk_slices_in_order_and_concatenates(self):
        # 6 columns, bs=2 -> exactly three chunks at columns [0:2],[2:4],[4:6].
        # arange content makes any wrong slice / wrong concat order visible.
        original = np.arange(6, dtype=np.float32).reshape([1, 6])
        expected_chunks = [original[:, 0:2], original[:, 2:4], original[:, 4:6]]
        seen = []

        def f(x):
            seen.append(x.numpy().copy())
            return x * 2.0

        wrapped = subbatch(f, arg_idx=[0], axis=[1], bs=2, out_idx=1)
        out = wrapped(paddle.to_tensor(original))

        self.assertEqual(len(seen), 3)
        for got, want in zip(seen, expected_chunks):
            np.testing.assert_array_equal(got, want)
        # Reassembled along axis 1 in slice order == elementwise f(original).
        np.testing.assert_array_equal(out.numpy(), original * 2.0)

    def test_rejects_mismatched_arg_idx_and_axis(self):
        wrapped = subbatch(
            lambda x: x, arg_idx=[0, 1], axis=[1], bs=2, out_idx=1
        )
        with self.assertRaises(AssertionError):
            wrapped(paddle.zeros([1, 4]), paddle.zeros([1, 4]))

    def test_rejects_unequal_batched_widths(self):
        # Two batched args with different sizes along their batched axes.
        wrapped = subbatch(
            lambda a, b: a, arg_idx=[0, 1], axis=[1, 1], bs=2, out_idx=1
        )
        with self.assertRaises(AssertionError):
            wrapped(paddle.zeros([1, 4]), paddle.zeros([1, 6]))


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestTensorMd5(unittest.TestCase):
    """``_tensor_md5`` hashes the cast tensor bytes; independent hashlib ref."""

    def test_default_float32_hash(self):
        vals = [[1.5, -2.25], [0.0, 3.0]]
        t = paddle.to_tensor(vals, dtype="float32")
        expected = hashlib.md5(
            np.asarray(vals, dtype=np.float32).tobytes()
        ).hexdigest()
        self.assertEqual(_tensor_md5(t), expected)

    def test_dtype_argument_changes_byte_width(self):
        # Casting to float64 doubles the byte width, so the hash must match a
        # float64 reference and differ from the float32 one.
        vals = [1.5, -2.25, 0.0, 3.0]
        t = paddle.to_tensor(vals, dtype="float32")
        ref64 = hashlib.md5(
            np.asarray(vals, dtype=np.float64).tobytes()
        ).hexdigest()
        ref32 = hashlib.md5(
            np.asarray(vals, dtype=np.float32).tobytes()
        ).hexdigest()
        self.assertEqual(_tensor_md5(t, dtype="float64"), ref64)
        self.assertNotEqual(ref64, ref32)

    def test_detaches_grad_tracked_tensor(self):
        t = paddle.to_tensor([0.25, -0.75], dtype="float32")
        t.stop_gradient = False
        expected = hashlib.md5(
            np.asarray([0.25, -0.75], dtype=np.float32).tobytes()
        ).hexdigest()
        # Must not raise on a grad-tracked tensor and must ignore grad state.
        self.assertEqual(_tensor_md5(t), expected)


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestEnvGates(unittest.TestCase):
    """The two env switches trigger only on the exact string ``"1"``."""

    def test_loss_md5_enabled_exact_match(self):
        with mock.patch.dict(os.environ, {"LOG_LOSS_MD5": "1"}):
            self.assertTrue(_loss_md5_enabled())
        with mock.patch.dict(os.environ, {"LOG_LOSS_MD5": "0"}):
            self.assertFalse(_loss_md5_enabled())
        with mock.patch.dict(os.environ, {"LOG_LOSS_MD5": "2"}):
            self.assertFalse(_loss_md5_enabled())  # only "1" enables
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_loss_md5_enabled())  # absent -> default "0"

    def test_accuracy_compatible_kernel_exact_match(self):
        key = "FLAGS_use_accuracy_compatible_kernel"
        with mock.patch.dict(os.environ, {key: "1"}):
            self.assertTrue(_use_accuracy_compatible_kernel())
        with mock.patch.dict(os.environ, {key: "true"}):
            self.assertFalse(_use_accuracy_compatible_kernel())  # not "1"
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_use_accuracy_compatible_kernel())


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestPrintScalarLossMd5(unittest.TestCase):
    """``_print_scalar_loss_md5`` emits nothing unless the env gate is on."""

    def test_silent_when_disabled(self):
        loss = paddle.to_tensor(1.25, dtype="float32")
        buf = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"LOG_LOSS_MD5": "0"}),
            redirect_stdout(buf),
        ):
            self.assertIsNone(_print_scalar_loss_md5("PFX", "lm_loss", loss))
        self.assertEqual(buf.getvalue(), "")

    def test_prints_name_and_matching_md5_when_enabled(self):
        loss = paddle.to_tensor(1.25, dtype="float32")
        # Production hashes loss.detach().cast("float32").reshape([1]).
        expected_md5 = hashlib.md5(
            np.asarray([1.25], dtype=np.float32).tobytes()
        ).hexdigest()
        buf = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"LOG_LOSS_MD5": "1"}),
            redirect_stdout(buf),
        ):
            _print_scalar_loss_md5("PFX", "lm_loss", loss)
        text = buf.getvalue()
        self.assertIn("lm_loss=1.25", text)
        self.assertIn(f"lm_loss_md5={expected_md5}", text)


if __name__ == "__main__":
    unittest.main()
