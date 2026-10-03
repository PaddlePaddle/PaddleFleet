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

"""Unit tests for the module-level helpers of ``language_loss``.

These helpers implement deferred token normalization (report the divided
loss, keep the logits gradient unnormalized) plus the two vocab-parallel
cross-entropy wrappers. They are plain module functions, so they are
exercised directly: no process group, no LanguageLoss instance.
"""

import unittest
from unittest.mock import patch

import numpy as np
import paddle

from paddlefleet import tensor_parallel
from paddlefleet.models.common.language_loss import language_loss as ll
from paddlefleet.tensor_parallel import cross_entropy as tp_cross_entropy


def _scalar(value, dtype="float32"):
    """A 0-d tensor, the shape the reporting numerator must have."""
    return paddle.to_tensor(value, dtype=dtype)


class _ModuleStateTestCase(unittest.TestCase):
    """The helpers keep process-wide state; reset it around every test."""

    def setUp(self):
        self._reset_module_state()

    def tearDown(self):
        self._reset_module_state()

    @staticmethod
    def _reset_module_state():
        ll._MAIN_REPORTING_CONTEXT = None
        ll._PENDING_GRADIENT_DIVISOR.pop("value", None)
        ll._LOCAL_MAIN_VALID_TOKENS.pop("value", None)


class MainReportingContextTest(_ModuleStateTestCase):
    """begin / _record / consume bookkeeping of the MAIN numerator."""

    def test_begin_rejects_unconsumed_context(self):
        ll.begin_main_reporting_microbatch(3, 1)
        self.assertEqual(
            ll._MAIN_REPORTING_CONTEXT,
            {"step": 3, "microbatch": 1, "pending": None},
        )
        with self.assertRaisesRegex(RuntimeError, "unconsumed"):
            ll.begin_main_reporting_microbatch(4, 0)
        # The rejected call must not overwrite the live context.
        self.assertEqual(ll._MAIN_REPORTING_CONTEXT["step"], 3)
        self.assertEqual(ll._MAIN_REPORTING_CONTEXT["microbatch"], 1)

    def test_record_without_context_is_a_noop(self):
        ll._record_main_reporting_sum(_scalar(6.0), 3.0)
        self.assertIsNone(ll._MAIN_REPORTING_CONTEXT)

    def test_record_rejects_second_numerator(self):
        ll.begin_main_reporting_microbatch(1, 0)
        ll._record_main_reporting_sum(_scalar(6.0), 3.0)
        with self.assertRaisesRegex(RuntimeError, "duplicate MAIN numerator"):
            ll._record_main_reporting_sum(_scalar(9.0), 5.0)
        # The first numerator stays; the duplicate is dropped.
        pending = ll._MAIN_REPORTING_CONTEXT["pending"]
        self.assertEqual(float(pending["sum"]), 6.0)
        self.assertEqual(pending["count"], 3.0)

    def test_record_requires_native_fp32_scalar(self):
        ll.begin_main_reporting_microbatch(1, 0)
        rejected = (
            _scalar(6.0, "float64"),
            _scalar(6.0, "float16"),
            paddle.to_tensor([6.0], dtype="float32"),
        )
        for bad in rejected:
            with (
                self.subTest(dtype=str(bad.dtype), shape=list(bad.shape)),
                self.assertRaisesRegex(RuntimeError, "FP32 scalar"),
            ):
                ll._record_main_reporting_sum(bad, 3.0)
        self.assertIsNone(ll._MAIN_REPORTING_CONTEXT["pending"])

    def test_consume_returns_detached_receipt_and_clears_context(self):
        ll.begin_main_reporting_microbatch(9, 2)
        loss_sum = _scalar(6.0)
        loss_sum.stop_gradient = False
        ll._record_main_reporting_sum(loss_sum, 3.0)

        receipt = ll.consume_main_reporting_microbatch(9, 2)

        self.assertEqual(float(receipt["sum"]), 6.0)
        self.assertTrue(receipt["sum"].stop_gradient)
        self.assertEqual(receipt["count"], 3.0)
        self.assertEqual(receipt["step"], 9)
        self.assertEqual(receipt["microbatch"], 2)
        self.assertIsNone(ll._MAIN_REPORTING_CONTEXT)

    def test_consume_returns_none_when_nothing_was_recorded(self):
        ll.begin_main_reporting_microbatch(5, 0)
        self.assertIsNone(ll.consume_main_reporting_microbatch(5, 0))
        self.assertIsNone(ll._MAIN_REPORTING_CONTEXT)

    def test_consume_rejects_stale_then_missing_context(self):
        ll.begin_main_reporting_microbatch(5, 0)
        for step, microbatch in ((5, 1), (6, 0)):
            with (
                self.subTest(step=step, microbatch=microbatch),
                self.assertRaisesRegex(RuntimeError, "missing or stale"),
            ):
                ll.consume_main_reporting_microbatch(step, microbatch)
        # A stale request must leave the context intact for its real owner.
        self.assertIsNotNone(ll._MAIN_REPORTING_CONTEXT)
        ll.consume_main_reporting_microbatch(5, 0)
        with self.assertRaisesRegex(RuntimeError, "missing or stale"):
            ll.consume_main_reporting_microbatch(5, 0)


class PendingGradientDivisorTest(_ModuleStateTestCase):
    """The divisor other ranks read back after the loss rank published it."""

    def test_publish_read_and_clear_divisor(self):
        self.assertIsNone(ll.get_pending_gradient_divisor())

        ll.set_pending_gradient_divisor(44)
        published = ll.get_pending_gradient_divisor()
        self.assertEqual(published, 44.0)
        self.assertIsInstance(published, float)

        ll.clear_pending_gradient_divisor()
        self.assertIsNone(ll.get_pending_gradient_divisor())
        # Clearing twice must stay a no-op (trainer may not have a divisor).
        ll.clear_pending_gradient_divisor()
        self.assertIsNone(ll.get_pending_gradient_divisor())


class NormalizeLossByTokensTest(_ModuleStateTestCase):
    """Forward divides; under UAC the gradient share is deferred."""

    def test_plain_division_when_not_accuracy_compatible(self):
        out = ll._normalize_loss_by_tokens(_scalar(7.0), 4.0)
        np.testing.assert_array_equal(
            out.numpy(), np.float32(7.0) / np.float32(4.0)
        )
        self.assertIsNone(ll.get_pending_gradient_divisor())
        self.assertIsNone(ll.get_local_main_valid_tokens())

    def test_plain_division_when_no_valid_tokens(self):
        out = ll._normalize_loss_by_tokens(
            _scalar(7.0), 0.0, use_accuracy_compatible=True
        )
        self.assertTrue(bool(np.isinf(out.numpy())))
        self.assertIsNone(ll.get_pending_gradient_divisor())

    def test_deferred_normalization_publishes_reporting_numerator(self):
        ll.begin_main_reporting_microbatch(1, 0)
        loss_sum = _scalar(11.0)
        loss_sum.stop_gradient = False

        out = ll._normalize_loss_by_tokens(
            loss_sum, 4.0, use_accuracy_compatible=True
        )

        np.testing.assert_array_equal(
            out.numpy(), np.float32(11.0) / np.float32(4.0)
        )
        self.assertEqual(ll.get_pending_gradient_divisor(), 4.0)
        self.assertEqual(ll.get_local_main_valid_tokens(), 4.0)
        receipt = ll.consume_main_reporting_microbatch(1, 0)
        self.assertEqual(float(receipt["sum"]), 11.0)
        self.assertEqual(receipt["count"], 4.0)
        out.backward()
        # main_tokens defaults to valid_tokens: gradient scale is exactly 1.
        np.testing.assert_array_equal(loss_sum.grad.numpy(), np.float32(1.0))

    def test_deferred_normalization_keeps_gradient_unnormalized(self):
        loss_sum = _scalar(6.0)
        loss_sum.stop_gradient = False

        out = ll._normalize_loss_by_tokens(
            loss_sum, 4.0, 8.0, use_accuracy_compatible=True
        )

        np.testing.assert_array_equal(
            out.numpy(), np.float32(6.0) / np.float32(4.0)
        )
        out.backward()
        # The forward divided by 4, the backward only applies main/valid.
        np.testing.assert_array_equal(loss_sum.grad.numpy(), np.float32(2.0))
        # An explicit main_tokens means this rank is not the MAIN owner.
        self.assertIsNone(ll.get_pending_gradient_divisor())
        self.assertIsNone(ll.get_local_main_valid_tokens())

    def test_defer_op_divides_by_a_zero_dim_tensor(self):
        loss_sum = _scalar(5.0)
        loss_sum.stop_gradient = False

        out = ll.DeferTokenNormalizationOp.apply(loss_sum, 44.0, 0.25)

        expected = loss_sum.detach() / paddle.full(
            [], 44.0, dtype=loss_sum.dtype
        )
        np.testing.assert_array_equal(out.numpy(), expected.numpy())
        self.assertEqual(list(out.shape), [])
        out.backward()
        np.testing.assert_array_equal(loss_sum.grad.numpy(), np.float32(0.25))


class CrossEntropyWrapperTest(unittest.TestCase):
    """The two vocab-parallel cross-entropy wrappers."""

    def test_accuracy_compatible_ce_masks_ignored_labels(self):
        logits = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        labels = paddle.to_tensor([[-100, 1, 0]], dtype="int64")
        seen = {}

        def fake_ce(passed_logits, passed_labels):
            seen["labels"] = passed_labels
            return passed_logits.sum(-1) + passed_labels.astype(
                passed_logits.dtype
            )

        with patch.object(
            tensor_parallel, "vocab_parallel_cross_entropy", fake_ce
        ):
            out = ll._accuracy_compatible_cross_entropy(logits, labels, -100)

        # The sentinel becomes index 0; every other label is untouched.
        np.testing.assert_array_equal(
            seen["labels"].numpy(), np.array([[0, 1, 0]])
        )
        np.testing.assert_array_equal(labels.numpy(), np.array([[-100, 1, 0]]))
        np.testing.assert_array_equal(
            out.numpy(), np.array([[3.0, 8.0, 11.0]], dtype="float32")
        )

    def test_uac_ce_runs_in_sbv_layout_and_restores_bs(self):
        batch, seq, vocab = 2, 3, 4
        logits = paddle.arange(batch * seq * vocab, dtype="float32").reshape(
            [batch, seq, vocab]
        )
        labels = paddle.arange(batch * seq, dtype="int64").reshape([batch, seq])
        seen = {}

        def fake_ce(passed_logits, passed_labels):
            seen["logits"] = passed_logits
            seen["labels"] = passed_labels
            return passed_logits[..., 0] * 100.0 + passed_labels.astype(
                passed_logits.dtype
            )

        with patch.object(
            tp_cross_entropy, "vocab_parallel_cross_entropy", fake_ce
        ):
            out = ll._uac_vocab_parallel_ce(logits, labels)

        # Megatron layout on the way in ...
        self.assertEqual(list(seen["logits"].shape), [seq, batch, vocab])
        self.assertEqual(list(seen["labels"].shape), [seq, batch])
        np.testing.assert_array_equal(
            seen["logits"].numpy(), logits.numpy().transpose(1, 0, 2)
        )
        # ... trainer layout on the way out, values still per (b, s).
        self.assertEqual(list(out.shape), [batch, seq])
        expected = logits.numpy()[..., 0] * 100.0 + labels.numpy().astype(
            "float32"
        )
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_uac_ce_keeps_non_3d_layout_untouched(self):
        rows, vocab = 3, 4
        logits = paddle.arange(rows * vocab, dtype="float32").reshape(
            [rows, vocab]
        )
        labels = paddle.arange(rows, dtype="int64")
        seen = {}

        def fake_ce(passed_logits, passed_labels):
            seen["logits"] = passed_logits
            seen["labels"] = passed_labels
            return passed_logits[..., 0] + passed_labels.astype(
                passed_logits.dtype
            )

        with patch.object(
            tp_cross_entropy, "vocab_parallel_cross_entropy", fake_ce
        ):
            out = ll._uac_vocab_parallel_ce(logits, labels)

        self.assertIs(seen["logits"], logits)
        self.assertIs(seen["labels"], labels)
        expected = logits.numpy()[:, 0] + labels.numpy().astype("float32")
        np.testing.assert_array_equal(out.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
