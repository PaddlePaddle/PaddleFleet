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

"""Single-card coverage for the eval token-CE observer hook.

``LanguageLoss.forward_impl`` reduces the per-token cross-entropy before
returning, so a caller that needs per-row / per-source attribution cannot
recover it from the returned scalar. The optional
``_eval_token_loss_hook`` attribute is read just before the ``lossmask``
reduction in both loss paths and, when set, is handed the unreduced token
loss together with the labels it belongs to.

The hook must be observation-only. These tests drive the REAL
``forward_impl`` via ``__new__`` + MagicMock config and a stubbed
``loss_func``, and assert on both sides of the contract:

- the hook sees the unreduced token loss, shaped like the labels, with the
  values the loss function produced (``test_observer_*``);
- installing the hook changes neither the returned scalar nor the
  gradients that flow from it (``test_*_unchanged``).

Both call sites are covered: the plain ``loss_func`` path, and the fused
linear+CE path taken when ``GPTLMHead`` emits a tuple. The latter stands in
a fake ``LigerFusedLinearCrossEntropyFunction`` for the triton kernel, so
the observer line is reached without a triton build.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import paddle

from paddlefleet.models.common.language_loss.language_loss import LanguageLoss

IGNORED = -100


def _make_loss():
    """A ``LanguageLoss`` whose ``loss_func`` returns a known CE matrix."""
    loss = LanguageLoss.__new__(LanguageLoss)
    cfg = MagicMock()
    cfg.gpt_model_use_experimental_version = False
    cfg.sequence_parallel = False
    cfg.fused_linear_ce_loss_chunk = 0
    cfg.cp_balance_mode = "dualchunk_allgather"
    loss.config = cfg
    loss.ignored_index = IGNORED
    loss.enable_parallel_cross_entropy = False
    loss.loss_subbatch_sequence_length = 0
    loss.use_subbatch = False
    # Normally set in __init__, which __new__ skips; the plain reduction
    # branch reads it directly.
    loss.use_accuracy_compatible = False

    def _stub_loss_func(logits, labels):
        # Per-token CE shaped like labels, distinct per position so a
        # reduced tensor cannot pass for an unreduced one.
        n = int(np.prod(labels.shape))
        base = paddle.arange(1, n + 1, dtype="float32")
        return base.reshape(labels.shape) * 0.5

    loss.loss_func = _stub_loss_func
    return loss


def _inputs(B=2, L=4, V=5, masked_rows=()):
    logits = paddle.randn([B, L, V], dtype="float32")
    labels_np = np.arange(B * L, dtype="int64").reshape([B, L]) % V
    for row in masked_rows:
        labels_np[row, :] = IGNORED
    return logits, paddle.to_tensor(labels_np)


class _FakeFusedCE:
    """Stand-in for the triton fused linear+CE kernel.

    Returns a flat per-token loss so ``forward_impl`` reshapes it to
    ``[B, S]`` exactly as it does for the real kernel.
    """

    captured = None

    @staticmethod
    def apply(_input, _weight, _labels, *args):
        n = _input.shape[0]
        return paddle.arange(1, n + 1, dtype="float32") * 0.25


def _fake_triton_module():
    mod = types.ModuleType("paddlefleet.triton_ops.fused_linear_cross_entropy")
    mod.LigerFusedLinearCrossEntropyFunction = _FakeFusedCE
    return mod


class TestEvalTokenLossObserver(unittest.TestCase):
    """The plain ``loss_func`` path (language_loss.py:390-392)."""

    def test_observer_sees_unreduced_token_loss(self) -> None:
        loss = _make_loss()
        logits, labels = _inputs()
        seen = []
        loss._eval_token_loss_hook = lambda tl, lb: seen.append((tl, lb))

        loss.forward_impl(logits, labels)

        self.assertEqual(len(seen), 1, "observer must run exactly once")
        token_loss, main_labels = seen[0]
        self.assertEqual(list(token_loss.shape), list(labels.shape))
        np.testing.assert_allclose(
            token_loss.numpy(),
            loss.loss_func(logits, labels).numpy(),
            rtol=0,
            atol=0,
        )
        np.testing.assert_array_equal(main_labels.numpy(), labels.numpy())

    def test_observer_runs_when_every_label_is_masked(self) -> None:
        # The all-masked branch returns early; the hook is read before it,
        # so a fully masked row set must still be reported rather than
        # silently dropped from the caller's accounting.
        loss = _make_loss()
        logits, labels = _inputs(B=1, masked_rows=(0,))
        seen = []
        loss._eval_token_loss_hook = lambda tl, lb: seen.append((tl, lb))

        out = loss.forward_impl(logits, labels)

        self.assertEqual(len(seen), 1)
        self.assertAlmostEqual(float(out), 0.0, places=6)

    def test_returned_loss_unchanged(self) -> None:
        logits, labels = _inputs()
        without = _make_loss().forward_impl(logits, labels)
        instrumented = _make_loss()
        instrumented._eval_token_loss_hook = lambda tl, lb: None
        with_hook = instrumented.forward_impl(logits, labels)
        np.testing.assert_allclose(
            with_hook.numpy(), without.numpy(), rtol=0, atol=0
        )

    def test_gradients_unchanged(self) -> None:
        # The hook receives live tensors; make sure merely reading them
        # (and holding a reference) does not perturb the backward pass.
        _logits, labels = _inputs()

        def _grad(hook):
            x = paddle.ones([1], dtype="float32")
            x.stop_gradient = False
            loss = _make_loss()
            loss.loss_func = lambda lg, lb: (
                x * paddle.ones(lb.shape, dtype="float32")
            )
            if hook is not None:
                loss._eval_token_loss_hook = hook
            out = loss.forward_impl(paddle.randn([2, 4, 5]), labels)
            out.backward()
            return x.grad.numpy().copy()

        held = []
        np.testing.assert_allclose(
            _grad(lambda tl, lb: held.append((tl, lb))),
            _grad(None),
            rtol=0,
            atol=0,
        )
        self.assertEqual(len(held), 1)

    def test_absent_hook_is_not_required(self) -> None:
        loss = _make_loss()
        self.assertFalse(hasattr(loss, "_eval_token_loss_hook"))
        logits, labels = _inputs()
        self.assertEqual(
            loss.forward_impl(logits, labels).dtype, paddle.float32
        )


class TestEvalTokenLossObserverFusedPath(unittest.TestCase):
    """The fused linear+CE path (language_loss.py:311-313)."""

    def _run(self, *, with_hook):
        loss = _make_loss()
        loss.config.fused_linear_ce_loss_chunk = 128
        B, L, H = 2, 4, 3
        hidden = paddle.randn([B, L, H], dtype="float32")
        weight = paddle.randn([H, 5], dtype="float32")
        _logits, labels = _inputs(B=B, L=L)
        seen = []
        if with_hook:
            loss._eval_token_loss_hook = lambda tl, lb: seen.append((tl, lb))
        mods = {
            "paddlefleet.triton_ops.fused_linear_cross_entropy": (
                _fake_triton_module()
            )
        }
        with mock.patch.dict(sys.modules, mods):
            out = loss.forward_impl((hidden, weight, None), labels)
        return out, seen, labels

    def test_observer_sees_token_loss_reshaped_to_labels(self) -> None:
        out, seen, labels = self._run(with_hook=True)
        self.assertEqual(len(seen), 1)
        token_loss, main_labels = seen[0]
        self.assertEqual(list(token_loss.shape), list(labels.shape))
        expected = (
            np.arange(1, int(np.prod(labels.shape)) + 1, dtype="float32") * 0.25
        ).reshape(labels.shape)
        np.testing.assert_allclose(token_loss.numpy(), expected, rtol=0, atol=0)
        np.testing.assert_array_equal(main_labels.numpy(), labels.numpy())
        self.assertEqual(list(out.shape), [] if out.ndim == 0 else [1])

    def test_returned_loss_unchanged(self) -> None:
        paddle.seed(0)
        with_hook, seen, _ = self._run(with_hook=True)
        paddle.seed(0)
        without, _, _ = self._run(with_hook=False)
        self.assertEqual(len(seen), 1)
        np.testing.assert_allclose(
            with_hook.numpy(), without.numpy(), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
