# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""Behavior tests for paddlefleet.transformers.contrastive_loss.

These tests exercise the real loss entries (SimpleContrastiveLoss,
MatryoshkaContrastiveLoss) on CPU with fixed, distinguishable tiny
embeddings and compare the produced scalar loss against an INDEPENDENT
numpy reference (softmax cross-entropy hand-derived below). The goal is to
verify the actual training-objective math: similarity scores, temperature
scaling, the in-batch positive-target selection (target = i * group_size),
and the cross-entropy MEAN reduction -- not just tensor shape.

Environment: 无卡 (CPU only). SimpleContrastiveLoss / MatryoshkaContrastiveLoss
are pure paddle ops (matmul, cross entropy, L2 normalize) that run on CPU.
The Inf-CL variants (SimpleInfclLoss / MatryoshkaInfclLoss) call a triton GPU
kernel (paddlefleet_kernel.triton.inf_cl.cal_inf_loss); their numeric path is
NOT validated here and is explicitly skipped with reason. There is no
cross-rank gather in this module, so no multi-card path applies.
"""

import unittest

import numpy as np
import paddle


# --------------------------------------------------------------------------
# Independent numpy references (no call into the production loss).
# --------------------------------------------------------------------------
def _log_softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def _ce_mean(scores, targets):
    """Softmax cross-entropy with reduction='mean', hand-derived."""
    log_probs = _log_softmax(np.asarray(scores, dtype=np.float64))
    picked = log_probs[np.arange(len(targets)), targets]
    return float((-picked).mean())


def _ce_sum(scores, targets):
    log_probs = _log_softmax(np.asarray(scores, dtype=np.float64))
    picked = log_probs[np.arange(len(targets)), targets]
    return float((-picked).sum())


def _contrastive_ref(q, p, temperature):
    """Independent reference matching SimpleContrastiveLoss semantics."""
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    scores = (q @ p.T) / temperature
    batch_size = q.shape[0]
    group_size = p.shape[0] // batch_size
    targets = np.arange(batch_size) * group_size
    return _ce_mean(scores, targets)


def _l2_normalize(x):
    x = np.asarray(x, dtype=np.float64)
    norm = np.sqrt((x * x).sum(axis=-1, keepdims=True))
    return x / norm


def _matryoshka_ref(q, p, temperature, dims):
    """Independent reference: per-dim L2-normalize then sum simple losses."""
    total = 0.0
    for d in dims:
        rq = _l2_normalize(np.asarray(q, dtype=np.float64)[:, :d])
        rp = _l2_normalize(np.asarray(p, dtype=np.float64)[:, :d])
        total += _contrastive_ref(rq, rp, temperature)
    return total


class TestSimpleContrastiveLoss(unittest.TestCase):
    """Numeric behavior of the real SimpleContrastiveLoss entry (CPU)."""

    @classmethod
    def setUpClass(cls):
        paddle.set_device("cpu")

    def _loss(self, q_np, p_np, temperature):
        from paddlefleet.transformers.contrastive_loss import (
            SimpleContrastiveLoss,
        )

        loss_fn = SimpleContrastiveLoss(embedding_temperature=temperature)
        q = paddle.to_tensor(q_np, dtype="float32")
        p = paddle.to_tensor(p_np, dtype="float32")
        return loss_fn(q, p)

    def test_forward_group_size_one_matches_reference(self):
        # group_size = 1: one positive passage per query, target = [0, 1].
        q = np.array(
            [[1.0, 2.0, -1.0, 0.5], [0.5, -1.0, 2.0, 1.0]], dtype=np.float32
        )
        p = np.array(
            [[0.9, 1.5, -0.5, 0.2], [0.1, -0.8, 1.7, 0.6]], dtype=np.float32
        )
        loss = self._loss(q, p, temperature=0.5)
        self.assertEqual(loss.shape, [])
        expected = _contrastive_ref(q, p, 0.5)
        np.testing.assert_allclose(loss.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_forward_group_size_two_positive_target(self):
        # group_size = 2: positives are passages 0 and 2 (target = i*group).
        q = np.array([[1.0, 0.5, -1.0], [0.2, 2.0, 0.7]], dtype=np.float32)
        p = np.array(
            [
                [1.1, 0.4, -0.9],  # positive for query 0
                [0.3, -1.0, 0.5],  # negative
                [0.1, 2.1, 0.6],  # positive for query 1
                [-0.7, 0.2, 1.3],  # negative
            ],
            dtype=np.float32,
        )
        loss = self._loss(q, p, temperature=0.5)
        expected = _contrastive_ref(q, p, 0.5)
        np.testing.assert_allclose(loss.numpy(), expected, rtol=1e-5, atol=1e-6)

        # Negative control: if target ignored group_size (target = [0, 1]),
        # the loss would differ. This proves target = i*group_size is used.
        scores = (q.astype(np.float64) @ p.astype(np.float64).T) / 0.5
        wrong = _ce_mean(scores, np.array([0, 1]))
        self.assertFalse(
            np.isclose(expected, wrong, rtol=1e-4, atol=1e-6),
            "fixture must distinguish correct grouped target from naive target",
        )

    def test_temperature_is_consumed(self):
        q = np.array(
            [[1.0, 2.0, -1.0, 0.5], [0.5, -1.0, 2.0, 1.0]], dtype=np.float32
        )
        p = np.array(
            [[0.9, 1.5, -0.5, 0.2], [0.1, -0.8, 1.7, 0.6]], dtype=np.float32
        )

        loss_hi = self._loss(q, p, temperature=0.5)
        loss_lo = self._loss(q, p, temperature=0.1)

        np.testing.assert_allclose(
            loss_hi.numpy(), _contrastive_ref(q, p, 0.5), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            loss_lo.numpy(), _contrastive_ref(q, p, 0.1), rtol=1e-5, atol=1e-6
        )
        # Different temperatures must produce different scaled losses.
        self.assertFalse(
            np.isclose(loss_hi.numpy(), loss_lo.numpy(), rtol=1e-4, atol=1e-6)
        )

    def test_default_temperature_is_point_zero_two(self):
        # Small-magnitude inputs so scores/0.02 stay in a safe numeric range.
        from paddlefleet.transformers.contrastive_loss import (
            SimpleContrastiveLoss,
        )

        q = np.array(
            [[0.01, 0.02, -0.01], [0.00, -0.02, 0.03]], dtype=np.float32
        )
        p = np.array(
            [[0.02, 0.01, -0.02], [-0.01, 0.03, 0.00]], dtype=np.float32
        )
        loss_fn = SimpleContrastiveLoss()  # default temperature
        loss = loss_fn(
            paddle.to_tensor(q, dtype="float32"),
            paddle.to_tensor(p, dtype="float32"),
        )
        expected = _contrastive_ref(q, p, 0.02)
        np.testing.assert_allclose(loss.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_reduction_is_mean_not_sum(self):
        # B = 3 so mean and sum differ; verifies reduction='mean'.
        q = np.array(
            [[1.0, 0.5, -1.0], [0.2, 2.0, 0.7], [-1.0, 0.3, 1.5]],
            dtype=np.float32,
        )
        p = np.array(
            [[0.9, 0.4, -0.8], [0.1, 1.8, 0.6], [-0.9, 0.2, 1.4]],
            dtype=np.float32,
        )
        loss = self._loss(q, p, temperature=0.5)

        scores = (q.astype(np.float64) @ p.astype(np.float64).T) / 0.5
        targets = np.array([0, 1, 2])
        mean_ref = _ce_mean(scores, targets)
        sum_ref = _ce_sum(scores, targets)

        np.testing.assert_allclose(loss.numpy(), mean_ref, rtol=1e-5, atol=1e-6)
        # Guard: sum reduction would give a clearly different (3x larger) value.
        self.assertFalse(
            np.isclose(loss.numpy(), sum_ref, rtol=1e-3, atol=1e-6)
        )


class TestMatryoshkaContrastiveLoss(unittest.TestCase):
    """Numeric behavior of MatryoshkaContrastiveLoss (CPU)."""

    @classmethod
    def setUpClass(cls):
        paddle.set_device("cpu")

    def test_forward_with_dims_matches_reference(self):
        from paddlefleet.transformers.contrastive_loss import (
            MatryoshkaContrastiveLoss,
        )

        dims = [2, 4]
        q = np.array(
            [
                [1.0, 2.0, -1.0, 0.5, 0.3, -0.2],
                [0.5, -1.0, 2.0, 1.0, -0.4, 0.8],
            ],
            dtype=np.float32,
        )
        p = np.array(
            [
                [0.9, 1.5, -0.5, 0.2, 0.1, -0.3],
                [0.1, -0.8, 1.7, 0.6, -0.5, 0.7],
            ],
            dtype=np.float32,
        )
        loss_fn = MatryoshkaContrastiveLoss(
            embedding_temperature=0.5, embedding_matryoshka_dims=dims
        )
        loss = loss_fn(
            paddle.to_tensor(q, dtype="float32"),
            paddle.to_tensor(p, dtype="float32"),
        )
        expected = _matryoshka_ref(q, p, 0.5, dims)
        np.testing.assert_allclose(loss.numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_dims_are_summed_not_averaged(self):
        from paddlefleet.transformers.contrastive_loss import (
            MatryoshkaContrastiveLoss,
        )

        dims = [2, 4]
        q = np.array(
            [
                [1.0, 2.0, -1.0, 0.5, 0.3, -0.2],
                [0.5, -1.0, 2.0, 1.0, -0.4, 0.8],
            ],
            dtype=np.float32,
        )
        p = np.array(
            [
                [0.9, 1.5, -0.5, 0.2, 0.1, -0.3],
                [0.1, -0.8, 1.7, 0.6, -0.5, 0.7],
            ],
            dtype=np.float32,
        )
        loss_fn = MatryoshkaContrastiveLoss(
            embedding_temperature=0.5, embedding_matryoshka_dims=dims
        )
        loss = loss_fn(
            paddle.to_tensor(q, dtype="float32"),
            paddle.to_tensor(p, dtype="float32"),
        )

        # Per-dim independent losses.
        per_dim = [
            _contrastive_ref(
                _l2_normalize(q[:, :d]), _l2_normalize(p[:, :d]), 0.5
            )
            for d in dims
        ]
        summed = sum(per_dim)
        averaged = summed / len(dims)

        np.testing.assert_allclose(loss.numpy(), summed, rtol=1e-5, atol=1e-5)
        # Accumulation must be a sum: differs from any single dim and from mean.
        self.assertFalse(
            np.isclose(loss.numpy(), per_dim[0], rtol=1e-3, atol=1e-6)
        )
        self.assertFalse(
            np.isclose(loss.numpy(), averaged, rtol=1e-3, atol=1e-6)
        )

    def test_no_dims_falls_back_to_simple_reference(self):
        from paddlefleet.transformers.contrastive_loss import (
            MatryoshkaContrastiveLoss,
        )

        q = np.array(
            [[1.0, 2.0, -1.0, 0.5], [0.5, -1.0, 2.0, 1.0]], dtype=np.float32
        )
        p = np.array(
            [[0.9, 1.5, -0.5, 0.2], [0.1, -0.8, 1.7, 0.6]], dtype=np.float32
        )
        # dims=None -> stored as [] -> forward uses plain SimpleContrastiveLoss.
        loss_fn = MatryoshkaContrastiveLoss(
            embedding_temperature=0.5, embedding_matryoshka_dims=None
        )
        loss = loss_fn(
            paddle.to_tensor(q, dtype="float32"),
            paddle.to_tensor(p, dtype="float32"),
        )
        # Compare against the independent simple reference (no matryoshka
        # normalization is applied on the fallback path).
        expected = _contrastive_ref(q, p, 0.5)
        np.testing.assert_allclose(loss.numpy(), expected, rtol=1e-5, atol=1e-6)


class TestInfclLossSkips(unittest.TestCase):
    """Inf-CL variants depend on a triton GPU kernel; numeric path skipped."""

    @classmethod
    def setUpClass(cls):
        paddle.set_device("cpu")

    def test_simple_infcl_forward_import_contract(self):
        # When the triton kernel is unavailable (CPU env), forward must raise a
        # helpful ImportError. If the kernel IS importable we cannot exercise
        # its GPU numeric path on CPU, so skip with reason.
        from paddlefleet.transformers.contrastive_loss import SimpleInfclLoss

        try:
            import paddlefleet_kernel.triton.inf_cl  # noqa: F401

            self.skipTest(
                "paddlefleet_kernel.triton.inf_cl is importable; its GPU "
                "triton numeric path is not validated on CPU (无卡)."
            )
        except ImportError:
            pass

        loss_fn = SimpleInfclLoss()
        q = paddle.to_tensor(
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            dtype="float32",
        )
        p = paddle.to_tensor(
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            dtype="float32",
        )
        with self.assertRaises(ImportError) as ctx:
            loss_fn(q, p)
        self.assertIn("inf_cl loss cannot be used", str(ctx.exception))

    @unittest.skip(
        "Inf-CL numeric path (cal_inf_loss) requires the triton GPU kernel "
        "paddlefleet_kernel.triton.inf_cl; not runnable in 无卡 CPU test. "
        "Validate SimpleInfclLoss/MatryoshkaInfclLoss numeric loss on GPU."
    )
    def test_infcl_numeric_requires_gpu_kernel(self):
        pass


if __name__ == "__main__":
    unittest.main()
