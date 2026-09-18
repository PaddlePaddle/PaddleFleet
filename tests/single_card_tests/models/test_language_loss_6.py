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

"""Single-card coverage for the separate MTP head-loss classes
``MainLanguageLoss`` / ``MTPLanguageLoss`` on the **ernie5 (non-megatron,
use_erndata=False) path**.

Scope (disjoint slice):
  * ``MTPLanguageLoss.forward``: per-depth label slicing
    ``labels[:, d+1 : d+1 + (L-K)]``, the in-place ``dict_args`` mutation
    (``mtp_logits`` popped, ``mtp_loss`` inserted, other keys kept, same object
    returned), the number of produced per-depth losses, and the assertion
    guards (missing ``mtp_logits`` / ``labels``, ``num_nextn_predict_layers``,
    ``mtp_load_weight_only``, ``mtp_distillation_loss``).
  * ``MainLanguageLoss.forward``: main-label trim ``labels[:, :-K]``, the
    ``add_loss`` scalar arithmetic for both ``add_mtp_loss=True`` (adds
    ``scaling * mean(mtp_loss)``) and ``add_mtp_loss=False`` (value is the LM
    loss only, gradient-only MTP flow), the ``train_mtp_only`` short-circuit,
    and population of the ``mtp_loss_tracker``.

Out of scope (covered elsewhere / by siblings): the inherited base
``LanguageLoss._forward`` / ``forward_impl`` cross-entropy numerics
(``test_language_loss_2/4/5.py``, ``subbatch`` / md5 / softmax helpers) and the
``use_erndata=True`` megatron per-doc roll path
(``test_separate_headloss_megatron.py``). The inherited ``_forward`` is
therefore isolated with a recording stub that returns a distinguishable scalar,
so this file observes the subclass orchestration/arithmetic — NOT the base
cross-entropy value.

Reference values are hand-derived with NumPy. There is no Paddle build in this
environment, so every case is gated with ``skipUnless`` and an honest reason;
none of them silently pass.
"""

from __future__ import annotations

import unittest
from unittest import mock
from unittest.mock import MagicMock

import numpy as np

try:  # Precise import gate: paddle is genuinely absent on this CPU box.
    import paddle
    from paddle.distributed.fleet.meta_parallel import ScheduleNode

    from paddlefleet.models.common.language_loss.language_loss import (
        LanguageLoss,  # noqa: F401
        MainLanguageLoss,
        MTPLanguageLoss,
    )

    PADDLE_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - exercised only without paddle
    PADDLE_AVAILABLE = False
    IMPORT_ERROR = repr(exc)

SKIP_REASON = (
    f"paddle / paddlefleet not importable in this environment: {IMPORT_ERROR}"
)

IGNORED = -100


def _record_forward(lm_value=None):
    """Return ``(stub, recorded)``.

    ``stub`` mimics the inherited ``LanguageLoss._forward`` collaborator: it
    records the labels tensor handed to it and returns a distinguishable scalar
    (``lm_value`` when given, otherwise the running call index) so the caller's
    real orchestration can be observed.
    """
    recorded = []

    def _stub(logits, labels):
        recorded.append(labels)
        val = float(len(recorded)) if lm_value is None else lm_value
        return paddle.to_tensor(val, dtype="float32")

    return _stub, recorded


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestMTPLanguageLossErnie5Forward(unittest.TestCase):
    """``MTPLanguageLoss.forward`` on the ernie5 (use_erndata=False) path."""

    def _make(self, K, mtp_load_weight_only=False, mtp_distillation_loss=False):
        loss = MTPLanguageLoss.__new__(MTPLanguageLoss)
        cfg = MagicMock()
        cfg.num_nextn_predict_layers = K
        cfg.mtp_load_weight_only = mtp_load_weight_only
        cfg.use_erndata = False
        cfg.mtp_distillation_loss = mtp_distillation_loss
        loss.config = cfg
        loss.ignored_index = IGNORED
        return loss

    def test_per_depth_labels_sliced_and_dict_mutated(self):
        # seq_length = L - K; MTP depth d predicts labels[:, d+1 : d+1+seq_length].
        K, B, L, V = 2, 1, 8, 5
        loss = self._make(K)
        stub, recorded = _record_forward()
        loss._forward = stub

        labels_np = np.arange(L, dtype="int64").reshape([B, L])
        labels = paddle.to_tensor(labels_np)
        mtp_logits = [
            paddle.randn([B, L, V], dtype="float32") for _ in range(K)
        ]
        dict_args = {"mtp_logits": mtp_logits, "labels": labels}

        out = loss.forward(dict_args)

        # Same dict object returned, mutated in place.
        self.assertIs(out, dict_args)
        self.assertNotIn("mtp_logits", out)
        self.assertIn("mtp_loss", out)
        self.assertIn("labels", out)  # untouched
        self.assertEqual(len(out["mtp_loss"]), K)

        # One recorded labels tensor per depth, each the hand-derived slice.
        seq_length = L - K
        self.assertEqual(len(recorded), K)
        for depth in range(K):
            got = recorded[depth].numpy()
            ref = labels_np[:, (depth + 1) : (depth + 1 + seq_length)]
            self.assertEqual(list(got.shape), [B, seq_length])
            np.testing.assert_array_equal(got, ref)

    def test_slice_windows_are_distinct_across_depths(self):
        # Guards against every depth receiving the same (e.g. depth-0) window.
        K, B, L, V = 3, 1, 10, 4
        loss = self._make(K)
        stub, recorded = _record_forward()
        loss._forward = stub

        labels_np = np.arange(L, dtype="int64").reshape([B, L])
        labels = paddle.to_tensor(labels_np)
        mtp_logits = [
            paddle.randn([B, L, V], dtype="float32") for _ in range(K)
        ]
        loss.forward({"mtp_logits": mtp_logits, "labels": labels})

        seq_length = L - K
        firsts = [int(recorded[d].numpy()[0, 0]) for d in range(K)]
        # First target token of each depth window is d+1: strictly increasing.
        self.assertEqual(firsts, [1, 2, 3])
        # And the windows do not collapse to a single shape/content.
        for d in range(K):
            ref = labels_np[:, (d + 1) : (d + 1 + seq_length)]
            np.testing.assert_array_equal(recorded[d].numpy(), ref)

    def test_missing_mtp_logits_asserts(self):
        loss = self._make(K=2)
        loss._forward, _ = _record_forward()
        labels = paddle.arange(8, dtype="int64").reshape([1, 8])
        with self.assertRaises(AssertionError):
            loss.forward({"labels": labels})

    def test_missing_labels_asserts(self):
        loss = self._make(K=2)
        loss._forward, _ = _record_forward()
        mtp_logits = [
            paddle.randn([1, 8, 5], dtype="float32") for _ in range(2)
        ]
        with self.assertRaises(AssertionError):
            loss.forward({"mtp_logits": mtp_logits})

    def test_zero_predict_layers_asserts(self):
        loss = self._make(K=0)
        loss._forward, _ = _record_forward()
        labels = paddle.arange(8, dtype="int64").reshape([1, 8])
        mtp_logits = [paddle.randn([1, 8, 5], dtype="float32")]
        with self.assertRaises(AssertionError):
            loss.forward({"mtp_logits": mtp_logits, "labels": labels})

    def test_load_weight_only_asserts(self):
        loss = self._make(K=2, mtp_load_weight_only=True)
        loss._forward, _ = _record_forward()
        labels = paddle.arange(8, dtype="int64").reshape([1, 8])
        mtp_logits = [
            paddle.randn([1, 8, 5], dtype="float32") for _ in range(2)
        ]
        with self.assertRaises(AssertionError):
            loss.forward({"mtp_logits": mtp_logits, "labels": labels})

    def test_distillation_loss_rejected(self):
        loss = self._make(K=2, mtp_distillation_loss=True)
        loss._forward, _ = _record_forward()
        labels = paddle.arange(8, dtype="int64").reshape([1, 8])
        mtp_logits = [
            paddle.randn([1, 8, 5], dtype="float32") for _ in range(2)
        ]
        with self.assertRaises(AssertionError):
            loss.forward({"mtp_logits": mtp_logits, "labels": labels})


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestMainLanguageLossErnie5Forward(unittest.TestCase):
    """``MainLanguageLoss.forward`` on the ernie5 (use_erndata=False) path."""

    def _make(
        self,
        K,
        add_mtp_loss=True,
        train_mtp_only=False,
        scaling=1.0,
        mtp_distillation_loss=False,
        mtp_load_weight_only=False,
    ):
        loss = MainLanguageLoss.__new__(MainLanguageLoss)
        cfg = MagicMock()
        cfg.num_nextn_predict_layers = K
        cfg.mtp_load_weight_only = mtp_load_weight_only
        cfg.use_erndata = False
        cfg.mtp_distillation_loss = mtp_distillation_loss
        cfg.train_mtp_only = train_mtp_only
        cfg.add_mtp_loss = add_mtp_loss
        cfg.mtp_loss_scaling_factor = scaling
        loss.config = cfg
        loss.ignored_index = IGNORED
        # Isolate the global training-log sink (a non-tested collaborator).
        p = mock.patch(
            "paddlefleet.models.common.language_loss.language_loss."
            "get_global_training_logs",
            return_value=None,
        )
        p.start()
        self.addCleanup(p.stop)
        # mtp_loss_tracker is class-level global state: snapshot & restore.
        orig = dict(MainLanguageLoss.mtp_loss_tracker)
        self.addCleanup(
            lambda: (
                MainLanguageLoss.mtp_loss_tracker.clear(),
                MainLanguageLoss.mtp_loss_tracker.update(orig),
            )
        )
        return loss

    def test_main_label_trimmed_to_L_minus_K(self):
        K, B, L, V = 2, 1, 8, 5
        loss = self._make(K)
        stub, recorded = _record_forward(lm_value=2.0)
        loss._forward = stub

        labels_np = np.arange(L, dtype="int64").reshape([B, L])
        labels = paddle.to_tensor(labels_np)
        dict_args = {
            "logits": paddle.randn([B, L - K, V], dtype="float32"),
            "mtp_loss": [paddle.to_tensor(0.0, dtype="float32")],
        }
        loss.forward(dict_args, labels)

        # Main LM label is labels[:, :-K] (length L-K), not the full length.
        self.assertEqual(len(recorded), 1)
        got = recorded[0].numpy()
        self.assertEqual(list(got.shape), [B, L - K])
        np.testing.assert_array_equal(got, labels_np[:, :-K])

    def test_add_mtp_loss_true_adds_scaled_mean(self):
        # final = lm + scaling * mean(mtp_loss)
        K = 2
        loss = self._make(K, add_mtp_loss=True, scaling=0.5)
        stub, _ = _record_forward(lm_value=2.0)
        loss._forward = stub

        labels = paddle.arange(8, dtype="int64").reshape([1, 8])
        mtp_loss = [
            paddle.to_tensor(1.0, dtype="float32"),
            paddle.to_tensor(3.0, dtype="float32"),
        ]
        dict_args = {
            "logits": paddle.randn([1, 6, 5], dtype="float32"),
            "mtp_loss": mtp_loss,
        }
        out = loss.forward(dict_args, labels)
        # 2.0 + 0.5 * ((1.0 + 3.0) / 2) = 3.0
        np.testing.assert_allclose(float(out), 3.0, rtol=0, atol=1e-6)

    def test_add_mtp_loss_false_value_is_lm_only(self):
        # add_mtp_loss=False -> value == lm (loss + loss - loss.detach()); the
        # MTP term contributes gradient only, so the scalar must ignore it.
        K = 2
        loss = self._make(K, add_mtp_loss=False, scaling=10.0)
        stub, _ = _record_forward(lm_value=2.0)
        loss._forward = stub

        labels = paddle.arange(8, dtype="int64").reshape([1, 8])
        mtp_loss = [
            paddle.to_tensor(1.0, dtype="float32"),
            paddle.to_tensor(3.0, dtype="float32"),
        ]
        out = loss.forward(
            {
                "logits": paddle.randn([1, 6, 5], dtype="float32"),
                "mtp_loss": mtp_loss,
            },
            labels,
        )
        # Despite scaling=10, the reported value equals the LM loss (2.0).
        np.testing.assert_allclose(float(out), 2.0, rtol=0, atol=1e-6)

    def test_train_mtp_only_zeroes_lm_and_skips_forward(self):
        K = 1
        loss = self._make(
            K, add_mtp_loss=True, train_mtp_only=True, scaling=1.0
        )
        stub, recorded = _record_forward(lm_value=99.0)
        loss._forward = stub

        labels = paddle.arange(8, dtype="int64").reshape([1, 8])
        mtp_loss = [paddle.to_tensor(4.0, dtype="float32")]
        out = loss.forward(
            {
                "logits": paddle.randn([1, 7, 5], dtype="float32"),
                "mtp_loss": mtp_loss,
            },
            labels,
        )
        # lm_loss short-circuits to 0.0, so _forward is never called and
        # final == scaling * mean(mtp_loss) == 4.0.
        self.assertEqual(len(recorded), 0)
        np.testing.assert_allclose(float(out), 4.0, rtol=0, atol=1e-6)

    def test_tracker_records_detached_per_depth_losses(self):
        K = 3
        loss = self._make(K, add_mtp_loss=True, scaling=1.0)
        stub, _ = _record_forward(lm_value=0.0)
        loss._forward = stub

        labels = paddle.arange(8, dtype="int64").reshape([1, 8])
        mtp_loss = [
            paddle.to_tensor(1.5, dtype="float32"),
            paddle.to_tensor(2.5, dtype="float32"),
            paddle.to_tensor(3.5, dtype="float32"),
        ]
        loss.forward(
            {
                "logits": paddle.randn([1, 5, 5], dtype="float32"),
                "mtp_loss": mtp_loss,
            },
            labels,
        )
        tracker = MainLanguageLoss.mtp_loss_tracker
        for i, expected in enumerate([1.5, 2.5, 3.5]):
            key = f"mtp_{i + 1}_loss"
            self.assertIn(key, tracker)
            self.assertTrue(bool(tracker[key].stop_gradient))  # detached
            np.testing.assert_allclose(
                float(tracker[key]), expected, rtol=0, atol=1e-6
            )

    def test_zero_predict_layers_asserts(self):
        loss = self._make(K=0)
        loss._forward, _ = _record_forward()
        labels = paddle.arange(8, dtype="int64").reshape([1, 8])
        with self.assertRaises(AssertionError):
            loss.forward(
                {
                    "logits": paddle.randn([1, 8, 5], dtype="float32"),
                    "mtp_loss": [],
                },
                labels,
            )

    def test_distillation_loss_rejected(self):
        loss = self._make(K=2, mtp_distillation_loss=True)
        loss._forward, _ = _record_forward(lm_value=1.0)
        labels = paddle.arange(8, dtype="int64").reshape([1, 8])
        dict_args = {
            "logits": paddle.randn([1, 6, 5], dtype="float32"),
            "mtp_loss": [paddle.to_tensor(1.0, dtype="float32")],
        }
        with self.assertRaises(AssertionError):
            loss.forward(dict_args, labels)


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestBuildScheduleNode(unittest.TestCase):
    """``build_schedule_node`` wraps the instance ``forward`` in a ScheduleNode."""

    def test_main_and_mtp_build_distinct_schedule_nodes(self):
        main = MainLanguageLoss.__new__(MainLanguageLoss)
        mtp = MTPLanguageLoss.__new__(MTPLanguageLoss)
        main_node = main.build_schedule_node()
        mtp_node = mtp.build_schedule_node()
        self.assertIsInstance(main_node, ScheduleNode)
        self.assertIsInstance(mtp_node, ScheduleNode)
        self.assertIsNot(main_node, mtp_node)


if __name__ == "__main__":
    unittest.main()
