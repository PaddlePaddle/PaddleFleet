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
import pytest

import paddlefleet.models.common.language_loss.language_loss as ll
from paddlefleet.models.common.language_loss.language_loss import LanguageLoss

IGNORED = -100


@pytest.fixture(autouse=True)
def _default_compatibility_mode(monkeypatch):
    monkeypatch.setenv("FLAGS_use_accuracy_compatible_kernel", "0")


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


def _real_loss(
    cls=LanguageLoss,
    *,
    experimental=False,
    fused=False,
    recomputing=False,
    added=True,
    depths=2,
):
    """Construct the real layer and CE, with explicit (non-MagicMock) flags."""
    cfg = types.SimpleNamespace(
        parallel_output=False,
        use_accuracy_compatible=False,
        loss_subbatch_sequence_length=0,
        sequence_parallel=False,
        gpt_model_use_experimental_version=experimental,
        fused_linear_ce_loss_chunk=1 if fused else 0,
        cp_balance_mode="contiguous_allgather",
        experimental_dataflow=False,
        recompute_modules=["loss_fn"] if recomputing else [],
        recompute_num_layers=-1,
        num_nextn_predict_layers=depths,
        mtp_load_weight_only=False,
        use_erndata=False,
        train_mtp_only=False,
        mtp_distillation_loss=False,
        add_mtp_loss=added,
        mtp_loss_scaling_factor=0.3,
    )
    return cls(cfg, pg_collection=types.SimpleNamespace())


def _real_inputs(*, fused=False, depths=2):
    paddle.seed(20260917)
    heads, leaves, expected_logits = [], [], []
    for _ in range(depths + 1):
        if fused:
            hidden = paddle.randn([2, 4, 8]) * 0.1
            weight = paddle.randn([16, 8]) * 0.1
            bias = paddle.randn([16]) * 0.1
            for tensor in (hidden, weight, bias):
                tensor.stop_gradient = False
            heads.append((hidden, weight, bias))
            leaves.extend((hidden, weight, bias))
            expected_logits.append(
                (
                    paddle.matmul(hidden, weight, transpose_y=True) + bias
                ).detach()
            )
        else:
            logits = paddle.randn([2, 4, 16]) * 0.1
            logits.stop_gradient = False
            heads.append(logits)
            leaves.append(logits)
            expected_logits.append(logits.detach())
    labels = paddle.to_tensor(
        [
            [0, 1, 2, 3, 4, 5][: 4 + depths],
            [6, -100, 7, -100, 8, -100][: 4 + depths],
        ],
        dtype="int64",
    )
    return heads, labels, leaves, expected_logits


def _save_observations(seen):
    def observe(loss, labels):
        seen.append((loss.detach().clone(), labels.detach().clone()))

    return observe


@pytest.mark.parametrize("experimental", [False, True])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("recompute_mode", [None, True, False])
@pytest.mark.parametrize("added", [False, True])
def test_real_mtp_forward_backward_observer_contract(
    experimental, fused, recompute_mode, added
):
    """Real CE (including Triton), MTP, autograd and both recompute engines."""
    results = []
    original_recompute = ll.recompute
    for observing in (False, True):
        layer = _real_loss(
            experimental=experimental,
            fused=fused,
            added=added,
            recomputing=recompute_mode is not None,
        )
        heads, labels, leaves, expected_logits = _real_inputs(fused=fused)
        seen = []
        if observing:
            layer._eval_token_loss_hook = _save_observations(seen)
        with mock.patch.object(
            ll,
            "recompute",
            side_effect=lambda f, *args: original_recompute(
                f, *args, use_reentrant=recompute_mode
            ),
        ):
            output = layer.forward(heads, labels)
            assert len(seen) == (3 if observing else 0)
            output.backward()
        assert len(seen) == (3 if observing else 0)
        results.append(
            (output.numpy().copy(), [x.grad.numpy().copy() for x in leaves])
        )
        for depth, (token_loss, observed_labels) in enumerate(seen):
            target = labels[:, depth : depth + 4]
            reference = paddle.nn.functional.cross_entropy(
                expected_logits[depth], target, reduction="none"
            ).reshape(target.shape)
            np.testing.assert_array_equal(
                observed_labels.numpy(), target.numpy()
            )
            np.testing.assert_allclose(
                token_loss.numpy(), reference.numpy(), rtol=2e-5, atol=2e-6
            )
    np.testing.assert_array_equal(results[0][0], results[1][0])
    for without, with_hook in zip(results[0][1], results[1][1]):
        np.testing.assert_array_equal(without, with_hook)


@pytest.mark.parametrize("experimental", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_separate_main_and_mtp_heads(experimental, fused):
    main = _real_loss(
        ll.MainLanguageLoss,
        experimental=experimental,
        fused=fused,
        recomputing=True,
    )
    mtp = _real_loss(
        ll.MTPLanguageLoss,
        experimental=experimental,
        fused=fused,
        recomputing=True,
    )
    heads, labels, _, _ = _real_inputs(fused=fused)
    main_seen, mtp_seen = [], []
    main._eval_token_loss_hook = _save_observations(main_seen)
    mtp._eval_token_loss_hook = _save_observations(mtp_seen)
    data = mtp.forward({"mtp_logits": heads[1:], "labels": labels})
    output = main.forward(
        {"logits": heads[0], "mtp_loss": data["mtp_loss"]}, labels
    )
    assert (len(main_seen), len(mtp_seen)) == (1, 2)
    output.backward()
    assert (len(main_seen), len(mtp_seen)) == (1, 2)
    np.testing.assert_array_equal(
        main_seen[0][1].numpy(), labels[:, :4].numpy()
    )
    for depth, (_, target) in enumerate(mtp_seen):
        np.testing.assert_array_equal(
            target.numpy(), labels[:, depth + 1 : depth + 5].numpy()
        )


@pytest.mark.parametrize("reentrant", [True, False])
def test_multiple_pending_microbatches_and_changed_observer(reentrant):
    layer = _real_loss(recomputing=True, depths=0)
    original_recompute = ll.recompute
    first, second, late = [], [], []
    heads, labels, _, _ = _real_inputs(depths=0)
    with mock.patch.object(
        ll,
        "recompute",
        side_effect=lambda f, *args: original_recompute(
            f, *args, use_reentrant=reentrant
        ),
    ):
        layer._eval_token_loss_hook = _save_observations(first)
        a = layer.forward(heads[0], labels)
        layer._eval_token_loss_hook = _save_observations(second)
        b = layer.forward(heads[0] * 2, labels)
        layer._eval_token_loss_hook = _save_observations(late)
        (a + b).backward()
    assert (len(first), len(second), len(late)) == (1, 1, 0)
    assert not ll._token_loss_replaying.get()


def test_observer_added_after_unobserved_forward_is_not_called_by_replay():
    layer = _real_loss(recomputing=True, depths=0)
    heads, labels, _, _ = _real_inputs(depths=0)
    output = layer.forward(heads[0], labels)
    seen = []
    layer._eval_token_loss_hook = _save_observations(seen)
    output.backward()
    assert seen == []


def test_subbatch_notifies_once_after_all_chunks():
    layer = _real_loss(depths=0)
    layer.use_subbatch = True
    layer.loss_subbatch_sequence_length = 2
    heads, labels, _, _ = _real_inputs(depths=0)
    seen = []
    layer._eval_token_loss_hook = _save_observations(seen)
    layer.forward(heads[0], labels).backward()
    assert len(seen) == 1
    reference = paddle.nn.functional.cross_entropy(
        heads[0], labels, reduction="none"
    )
    np.testing.assert_allclose(
        seen[0][0].numpy(), reference.reshape(labels.shape).numpy()
    )


def test_distillation_observes_main_ce_only():
    layer = _real_loss(recomputing=True)
    layer.config.mtp_distillation_loss = True
    heads, labels, _, expected_logits = _real_inputs()
    seen = []
    layer._eval_token_loss_hook = _save_observations(seen)
    layer.forward(heads, labels).backward()
    assert len(seen) == 1
    reference = paddle.nn.functional.cross_entropy(
        expected_logits[0], labels[:, :4], reduction="none"
    ).reshape([2, 4])
    np.testing.assert_allclose(seen[0][0].numpy(), reference.numpy())


@pytest.mark.parametrize("fused", [False, True])
def test_all_masked_main_and_mtp_still_notify(fused):
    layer = _real_loss(experimental=True, fused=fused)
    heads, labels, _, _ = _real_inputs(fused=fused)
    labels[:] = -100
    seen = []
    layer._eval_token_loss_hook = _save_observations(seen)
    output = layer.forward(heads, labels)
    output.backward()
    assert len(seen) == 3
    assert float(output) == 0.0
    assert all(list(loss.shape) == [2, 4] for loss, _ in seen)


def test_observer_failure_does_not_poison_next_forward():
    layer = _real_loss(recomputing=True, depths=0)
    heads, labels, _, _ = _real_inputs(depths=0)

    def fail(*args):
        raise RuntimeError("observer failure")

    layer._eval_token_loss_hook = fail
    with pytest.raises(RuntimeError, match="observer failure"):
        layer.forward(heads[0], labels)
    assert not ll._token_loss_replaying.get()
    seen = []
    layer._eval_token_loss_hook = _save_observations(seen)
    layer.forward(heads[0], labels).backward()
    assert len(seen) == 1


def test_experimental_nonfused_mtp_observes_after_cp_gather():
    """Check gather placement only; communication itself is mocked here."""
    layer = _real_loss(experimental=True)
    heads, labels, _, _ = _real_inputs()
    seen = []
    layer._eval_token_loss_hook = _save_observations(seen)

    def gather(value, **kwargs):
        return paddle.concat([value, value], axis=1)

    with (
        mock.patch.object(
            ll, "get_context_parallel_world_size", return_value=2
        ),
        mock.patch.object(
            ll.ContextParallelScatterOp, "apply", side_effect=lambda x, **kw: x
        ),
        mock.patch.object(
            ll.ContextParallelGatherOp, "apply", side_effect=gather
        ),
    ):
        layer.forward(heads, labels)
    assert len(seen) == 3
    for depth, (loss, target) in enumerate(seen):
        assert list(loss.shape) == [2, 8]
        expected = paddle.concat([labels[:, depth : depth + 4]] * 2, axis=1)
        np.testing.assert_array_equal(target.numpy(), expected.numpy())


if __name__ == "__main__":
    unittest.main()
