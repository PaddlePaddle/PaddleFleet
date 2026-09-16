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

"""Direct unit tests for the numeric and data-replay helpers in
``paddlefleet.accuracy_compatible_patch``.

The module bundles two very different families of code:

* Single-card numeric kernels - PyLayers and plain functions that back the
  DSV4 accuracy-compatible replay (softmax-with-sink, grouped O projection,
  RMS norms, seq-first weight-gradient accumulation, the learned output
  contract, and so on). Several of them hand their backward pass to torch
  through ``_to_torch`` / ``_to_paddle`` dlpack bridges, so every PyLayer here
  is exercised on BOTH forward and backward to cover those torch round-trips.
* Trainer / data-pipeline helpers - the ``LOAD_FIXED_DATA_PATH`` replay loader
  and its file-resolution, iteration and loss-scaling companions.

Everything below runs on a single card with no distributed process group.

Deliberately NOT exercised here (they mutate global paddle/fleet optimizer and
fusion classes and would pollute the shared pytest process):
``_install_fusion_patch``, ``_install_sharding_shape_patch``,
``_install_adamw_patch``, ``install_accuracy_compatible_paddle_patches`` and
``_group_indices``. ``te_matmul`` and ``CompatibleHadamard`` need
``transformer_engine`` / ``fast_hadamard_transform`` which are not importable
in this environment, so their tests are skipped with a reason.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import paddle

from paddlefleet import accuracy_compatible_patch as acp
from paddlefleet.accuracy_compatible_patch import FixedTrainingData


def setUpModule():
    if paddle.is_compiled_with_cuda():
        paddle.set_device("gpu")


try:
    acp._import_torch()
    _TORCH_OK = True
    _TORCH_SKIP = ""
except Exception as exc:  # pragma: no cover - torch is expected to be present
    _TORCH_OK = False
    _TORCH_SKIP = f"torch is not importable: {exc!r}"

requires_torch = unittest.skipUnless(_TORCH_OK, _TORCH_SKIP)


def _grad_enabled(tensor):
    tensor.stop_gradient = False
    return tensor


class TestSumForSmallRows(unittest.TestCase):
    """``sum_for_small_rows`` matches a plain row-sum in every branch."""

    def test_small_row_count_is_padded_but_result_matches_plain_sum(self):
        value = paddle.arange(0, 20, dtype="float32").reshape([4, 5])
        out = acp.sum_for_small_rows(value)
        self.assertEqual(out.shape, [4, 1])
        np.testing.assert_allclose(
            out.numpy(), value.numpy().sum(-1, keepdims=True), rtol=1e-6
        )

    def test_tall_matrix_takes_the_plain_sum_path(self):
        paddle.seed(0)
        value = paddle.randn([32, 3], dtype="float32")
        out = acp.sum_for_small_rows(value)
        # A 3-element row sum can land near zero, so pin with an absolute
        # tolerance too rather than relative-only.
        np.testing.assert_allclose(
            out.numpy(),
            value.numpy().sum(-1, keepdims=True),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_non_2d_input_takes_the_plain_sum_path(self):
        value = paddle.randn([2, 3, 4], dtype="float32")
        out = acp.sum_for_small_rows(value)
        self.assertEqual(out.shape, [2, 3, 1])


class TestLossScaleBeforeBackward(unittest.TestCase):
    """``set_acc_steps`` / ``scale`` divide a loss by the clamped step count."""

    def tearDown(self):
        acp.LossScaleBeforeBackward.set_acc_steps(1)

    def test_scale_divides_by_the_accumulation_steps(self):
        acp.LossScaleBeforeBackward.set_acc_steps(4)
        scaled = acp.LossScaleBeforeBackward.scale(paddle.to_tensor(8.0))
        self.assertAlmostEqual(float(scaled.numpy()), 2.0, places=6)

    def test_single_step_is_a_no_op(self):
        acp.LossScaleBeforeBackward.set_acc_steps(1)
        loss = paddle.to_tensor(8.0)
        self.assertIs(acp.LossScaleBeforeBackward.scale(loss), loss)

    def test_acc_steps_are_clamped_to_at_least_one(self):
        acp.LossScaleBeforeBackward.set_acc_steps(0)
        self.assertEqual(acp.LossScaleBeforeBackward._acc_steps, 1)


@requires_torch
class TestCompatibleCSASinkSoftmax(unittest.TestCase):
    """Softmax-with-sink forward matches numpy; backward flows to both inputs."""

    def test_forward_matches_reference_and_backward_is_finite(self):
        paddle.seed(0)
        scores = _grad_enabled(paddle.randn([2, 3, 4], dtype="float32"))
        sink = _grad_enabled(paddle.randn([2, 3, 1], dtype="float32"))

        out = acp.CompatibleCSASinkSoftmax.apply(scores, sink)

        s = scores.numpy()
        k = sink.numpy()
        m = np.maximum(s.max(-1, keepdims=True), k)
        exp_s = np.exp(s - m)
        ref = exp_s / (exp_s.sum(-1, keepdims=True) + np.exp(k - m))
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5, atol=1e-5)

        out.sum().backward()
        self.assertTrue(bool(paddle.isfinite(scores.grad).all()))
        self.assertTrue(bool(paddle.isfinite(sink.grad).all()))


@requires_torch
class TestCompatibleEinsum(unittest.TestCase):
    """``compatible_einsum`` contracts scores against q as ``sbht,sbhd->tbd``."""

    def test_matches_numpy_einsum(self):
        paddle.seed(1)
        grad_scores = paddle.randn([2, 2, 3, 4], dtype="float32")
        q = paddle.randn([2, 2, 3, 5], dtype="float32")

        out = acp.compatible_einsum(grad_scores, q)

        ref = np.einsum("sbht,sbhd->tbd", grad_scores.numpy(), q.numpy())
        self.assertEqual(out.shape, [4, 2, 5])
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-4, atol=1e-4)


@requires_torch
class TestCompatibleOGroupProjection(unittest.TestCase):
    """Grouped O projection: forward matches an einsum, backward is finite."""

    def test_forward_matches_reference_and_backward_stashes_wgrad(self):
        paddle.seed(2)
        b, s, g, d, r = 2, 3, 2, 4, 5
        x = _grad_enabled(paddle.randn([b, s, g, d], dtype="float32"))
        weight = _grad_enabled(paddle.randn([g, r, d], dtype="float32"))

        out = acp.CompatibleOGroupProjection.apply(x, weight, g, r, 0, 0)

        ref = np.einsum("bsgd,grd->bsgr", x.numpy(), weight.numpy())
        self.assertEqual(out.shape, [b, s, g, r])
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-4, atol=1e-4)

        out.sum().backward()
        self.assertTrue(bool(paddle.isfinite(x.grad).all()))
        self.assertTrue(bool(paddle.isfinite(weight.grad).all()))
        # backward also accumulates a seq-first wgrad on the weight object.
        stashed = getattr(weight, "_dsv4_attn_o_group_seqfirst_wgrad", None)
        self.assertIsNotNone(stashed)
        self.assertEqual(stashed.dtype, paddle.float32)


@requires_torch
class TestCompatibleQRMSNorm(unittest.TestCase):
    """QK RMS norm forward matches numpy; backward is finite."""

    def test_forward_matches_reference_and_backward_is_finite(self):
        paddle.seed(3)
        eps = 1e-6
        q = _grad_enabled(paddle.randn([3, 8], dtype="float32"))

        out = acp.CompatibleQRMSNorm.apply(q, eps)

        qn = q.numpy()
        ref = qn * (1.0 / np.sqrt((qn**2).mean(-1, keepdims=True) + eps))
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5, atol=1e-5)

        (out * out).sum().backward()
        self.assertTrue(bool(paddle.isfinite(q.grad).all()))


@requires_torch
class TestCompatibleEmbeddingIndexBackward(unittest.TestCase):
    """Embedding gather forward matches indexing; backward scatters counts."""

    def test_forward_gathers_rows_and_backward_accumulates_counts(self):
        paddle.seed(4)
        weight = _grad_enabled(paddle.randn([6, 4], dtype="float32"))
        idx = paddle.to_tensor([1, 3, 5, 3], dtype="int64")

        out = acp.CompatibleEmbeddingIndexBackward.apply(idx, weight)

        np.testing.assert_allclose(
            out.numpy(), weight.numpy()[idx.numpy()], rtol=1e-6
        )

        out.sum().backward()
        grad = weight.grad.numpy()
        # ones grad on rows [1, 3, 5, 3] -> row 3 seen twice.
        np.testing.assert_allclose(grad[3], np.ones(4) * 2.0)
        np.testing.assert_allclose(grad[1], np.ones(4))
        np.testing.assert_allclose(grad[5], np.ones(4))
        np.testing.assert_allclose(grad[0], np.zeros(4))


class TestTeMatmul(unittest.TestCase):
    """``te_matmul`` needs transformer_engine's general_gemm."""

    def test_te_matmul_requires_transformer_engine(self):
        try:
            import transformer_engine  # noqa: F401
        except Exception as exc:
            self.skipTest(f"transformer_engine not importable: {exc!r}")
        grad_output = paddle.randn([4, 8], dtype="bfloat16")
        weight = paddle.randn([8, 4], dtype="bfloat16")
        out = acp.te_matmul(grad_output, weight)
        self.assertTrue(bool(paddle.isfinite(out.cast("float32")).all()))


@requires_torch
class TestLinearSeqfirstWgrad(unittest.TestCase):
    """``linear_seqfirst_wgrad`` returns ``input^T @ grad`` or ``None``."""

    def _reference(self, inp, grad):
        return np.matmul(
            inp.cast("float32").numpy().reshape(-1, inp.shape[-1]).T,
            grad.cast("float32").numpy().reshape(-1, grad.shape[-1]),
        )

    def test_two_dim_path(self):
        paddle.seed(5)
        inp = paddle.randn([5, 6], dtype="float32").cast("bfloat16")
        grad = paddle.randn([5, 4], dtype="float32").cast("bfloat16")
        weight = paddle.randn([6, 4], dtype="float32").cast("bfloat16")

        out = acp.linear_seqfirst_wgrad(inp, grad, weight)

        self.assertEqual(out.shape, [6, 4])
        np.testing.assert_allclose(
            out.cast("float32").numpy(),
            self._reference(inp, grad),
            rtol=6e-2,
            atol=6e-2,
        )

    def test_three_dim_seqfirst_transpose_path(self):
        paddle.seed(6)
        inp = paddle.randn([2, 7, 6], dtype="float32").cast("bfloat16")
        grad = paddle.randn([2, 7, 4], dtype="float32").cast("bfloat16")
        weight = paddle.randn([6, 4], dtype="float32").cast("bfloat16")

        out = acp.linear_seqfirst_wgrad(inp, grad, weight)

        self.assertEqual(out.shape, [6, 4])
        np.testing.assert_allclose(
            out.cast("float32").numpy(),
            self._reference(inp, grad),
            rtol=6e-2,
            atol=6e-2,
        )

    def test_four_dim_path(self):
        paddle.seed(7)
        inp = paddle.randn([2, 3, 2, 6], dtype="float32").cast("bfloat16")
        grad = paddle.randn([2, 3, 2, 4], dtype="float32").cast("bfloat16")
        weight = paddle.randn([6, 4], dtype="float32").cast("bfloat16")

        out = acp.linear_seqfirst_wgrad(inp, grad, weight)
        self.assertEqual(out.shape, [6, 4])

    def test_non_bf16_weight_returns_none(self):
        inp = paddle.randn([5, 6], dtype="bfloat16")
        grad = paddle.randn([5, 4], dtype="bfloat16")
        weight = paddle.randn([6, 4], dtype="float32")
        self.assertIsNone(acp.linear_seqfirst_wgrad(inp, grad, weight))

    def test_rank_mismatch_returns_none(self):
        inp = paddle.randn([5, 6], dtype="bfloat16")
        grad = paddle.randn([2, 5, 4], dtype="bfloat16")
        weight = paddle.randn([6, 4], dtype="bfloat16")
        self.assertIsNone(acp.linear_seqfirst_wgrad(inp, grad, weight))


class TestMoEInputBranches(unittest.TestCase):
    """Fan-out clone whose backward sums the three incoming grads."""

    def test_forward_clones_and_backward_sums_grads(self):
        x = _grad_enabled(paddle.randn([3, 4], dtype="float32"))
        a, b, c = acp.MoEInputBranches.apply(x)
        np.testing.assert_allclose(a.numpy(), x.numpy())
        (a.sum() + b.sum() + c.sum()).backward()
        np.testing.assert_allclose(x.grad.numpy(), np.full([3, 4], 3.0))


class TestIndicesToMultihot(unittest.TestCase):
    """Expert indices become a boolean routing map plus a probs map."""

    def test_multihot_and_probs(self):
        indices = paddle.to_tensor([[0, 1, -1], [2, -1, -1]], dtype="int64")
        probs = paddle.to_tensor(
            [[0.5, 0.3, 0.0], [0.7, 0.0, 0.0]], dtype="float32"
        )

        routing_map, probs_map = acp.indices_to_multihot(indices, probs, 4)

        self.assertEqual(routing_map.dtype, paddle.bool)
        np.testing.assert_array_equal(
            routing_map.numpy(),
            np.array([[1, 1, 0, 0], [0, 0, 1, 0]], dtype=bool),
        )
        np.testing.assert_allclose(
            probs_map.numpy(),
            np.array([[0.5, 0.3, 0.0, 0.0], [0.0, 0.0, 0.7, 0.0]]),
            rtol=1e-6,
        )


@requires_torch
class TestHcMappingSeqfirstWgrad(unittest.TestCase):
    """The HC-mapping seq-first weight-gradient helpers and their grad hook."""

    def test_hc_mapping_wgrad_matches_reference(self):
        paddle.seed(8)
        x_sf = paddle.randn([6, 8], dtype="float32")
        grad_sf = paddle.randn([6, 4], dtype="float32")
        weight = paddle.randn([8, 4], dtype="float32")

        out = acp._hc_mapping_wgrad(x_sf, grad_sf, weight)

        ref = np.matmul(x_sf.numpy().T, grad_sf.numpy())
        self.assertEqual(out.shape, [8, 4])
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-4, atol=1e-4)

    def test_accumulate_starts_then_adds(self):
        weight = paddle.randn([8, 4], dtype="float32")
        g1 = paddle.ones([8, 4], dtype="float32")
        acp._accumulate_hc_mapping_seqfirst_wgrad(weight, g1)
        np.testing.assert_allclose(
            weight._dsv4_hc_mapping_seqfirst_wgrad.numpy(), np.ones([8, 4])
        )
        acp._accumulate_hc_mapping_seqfirst_wgrad(weight, g1 * 2.0)
        np.testing.assert_allclose(
            weight._dsv4_hc_mapping_seqfirst_wgrad.numpy(),
            np.full([8, 4], 3.0),
        )

    def test_accumulate_ignores_none(self):
        weight = paddle.randn([8, 4], dtype="float32")
        acp._accumulate_hc_mapping_seqfirst_wgrad(weight, None)
        self.assertIsNone(
            getattr(weight, "_dsv4_hc_mapping_seqfirst_wgrad", None)
        )

    def test_grad_hook_fires_for_multi_row_3d_input(self):
        paddle.seed(9)
        b, s, hidden, out = 2, 3, 8, 4
        x = _grad_enabled(paddle.randn([b, s, hidden], dtype="float32"))
        weight = paddle.randn([hidden, out], dtype="float32")
        proj_2d = paddle.matmul(x.reshape([-1, hidden]), weight)

        acp._register_hc_mapping_seqfirst_wgrad_hook(proj_2d, x, weight)
        proj_2d.sum().backward()

        self.assertIsNotNone(
            getattr(weight, "_dsv4_hc_mapping_seqfirst_wgrad", None)
        )

    def test_grad_hook_is_skipped_for_single_row_batch(self):
        x = _grad_enabled(paddle.randn([1, 3, 8], dtype="float32"))
        weight = paddle.randn([8, 4], dtype="float32")
        proj_2d = paddle.matmul(x.reshape([-1, 8]), weight)
        acp._register_hc_mapping_seqfirst_wgrad_hook(proj_2d, x, weight)
        proj_2d.sum().backward()
        self.assertIsNone(
            getattr(weight, "_dsv4_hc_mapping_seqfirst_wgrad", None)
        )


@requires_torch
class TestTorchOrderRmsScale(unittest.TestCase):
    """RMS scaling factor forward matches numpy; backward is finite."""

    def test_forward_matches_reference_and_backward_is_finite(self):
        import math

        paddle.seed(10)
        eps = 1e-6
        x = _grad_enabled(paddle.randn([4, 8], dtype="float32"))

        r = acp.TorchOrderRmsScale.apply(x, eps)

        xn = x.numpy()
        norm = np.linalg.norm(xn, axis=-1, keepdims=True)
        ref = 1.0 / (norm / math.sqrt(8) + eps)
        np.testing.assert_allclose(r.numpy(), ref, rtol=1e-5, atol=1e-5)

        (r * r).sum().backward()
        self.assertTrue(bool(paddle.isfinite(x.grad).all()))


@requires_torch
class TestCompatibleProjectionAndNorm(unittest.TestCase):
    """The fused projection + RMS-scale helper and its wgrad hook."""

    def test_projection_matches_matmul_and_hook_accumulates(self):
        paddle.seed(11)
        b, s, nC, out = 2, 3, 8, 4
        x = _grad_enabled(paddle.randn([b, s, nC], dtype="float32"))
        weight = paddle.randn([nC, out], dtype="float32")

        proj, r = acp.compatible_projection_and_norm(x, weight, 1e-6)

        self.assertEqual(proj.shape, [b, s, out])
        self.assertEqual(r.shape, [b, s, 1])
        ref = np.matmul(x.numpy().reshape(-1, nC), weight.numpy()).reshape(
            b, s, out
        )
        np.testing.assert_allclose(proj.numpy(), ref, rtol=1e-4, atol=1e-4)

        proj.sum().backward()
        self.assertIsNotNone(
            getattr(weight, "_dsv4_hc_mapping_seqfirst_wgrad", None)
        )


@requires_torch
class TestCompatibleSinkhornBackward(unittest.TestCase):
    """Sinkhorn backward produces a finite grad of the input shape."""

    def test_backward_is_finite(self):
        paddle.seed(12)
        logits = paddle.randn([4, 4], dtype="float32")
        grad_output = paddle.randn([4, 4], dtype="float32")

        grad_input = acp.compatible_sinkhorn_backward(
            logits, grad_output, num_iterations=2, eps=1e-6
        )

        self.assertEqual(grad_input.shape, [4, 4])
        self.assertTrue(bool(paddle.isfinite(grad_input.cast("float32")).all()))


@requires_torch
class TestCompatibleLearnedOutputContract(unittest.TestCase):
    """Learned output contract forward + backward on 2D and 3D inputs."""

    def _apply(self, hidden):
        n = 2
        head = _grad_enabled(paddle.randn([hidden.shape[-1], n], "float32"))
        base = _grad_enabled(paddle.randn([n], dtype="float32"))
        scale = _grad_enabled(paddle.randn([n], dtype="float32"))
        out = acp.CompatibleLearnedOutputContract.apply(
            hidden, head, base, scale, n, 1e-6, paddle.float32
        )
        return out, head, base, scale

    def test_two_dim_forward_backward(self):
        paddle.seed(13)
        hidden = _grad_enabled(paddle.randn([4, 8], dtype="float32"))
        out, head, base, scale = self._apply(hidden)
        self.assertEqual(out.shape, [4, 4])
        (out * out).sum().backward()
        for tensor in (hidden, head, base, scale):
            self.assertTrue(bool(paddle.isfinite(tensor.grad).all()))

    def test_three_dim_multi_row_takes_seqfirst_head_grad(self):
        paddle.seed(14)
        hidden = _grad_enabled(paddle.randn([2, 3, 8], dtype="float32"))
        out, head, base, scale = self._apply(hidden)
        self.assertEqual(out.shape, [2, 3, 4])
        (out * out).sum().backward()
        for tensor in (hidden, head, base, scale):
            self.assertTrue(bool(paddle.isfinite(tensor.grad).all()))


import json
import types


def _training_args(seq_len=4, accum=1):
    return types.SimpleNamespace(
        max_seq_len=seq_len, gradient_accumulation_steps=accum
    )


class TestResolveFixedTrainingFiles(unittest.TestCase):
    """File resolution walks the micro/step/manifest/glob fallbacks."""

    def _touch(self, d, name):
        np.save(os.path.join(d, name), np.array([0]))

    def test_micro_step_naming(self):
        with tempfile.TemporaryDirectory() as d:
            self._touch(d, "tokens_step0_micro0_rank0_seq4.npy")
            self._touch(d, "labels_step0_micro0_rank0_seq4.npy")
            tokens, labels = acp._resolve_fixed_training_files(d, 0, 0, 0, 4)
            self.assertTrue(
                tokens.endswith("tokens_step0_micro0_rank0_seq4.npy")
            )
            self.assertTrue(
                labels.endswith("labels_step0_micro0_rank0_seq4.npy")
            )

    def test_step_only_naming_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            self._touch(d, "tokens_step1_rank0_seq4.npy")
            self._touch(d, "labels_step1_rank0_seq4.npy")
            tokens, _ = acp._resolve_fixed_training_files(d, 1, 0, 0, 4)
            self.assertTrue(tokens.endswith("tokens_step1_rank0_seq4.npy"))

    def test_manifest_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            self._touch(d, "custom_tokens.npy")
            self._touch(d, "custom_labels.npy")
            with open(
                os.path.join(d, "manifest_rank0.jsonl"), "w", encoding="utf-8"
            ) as fh:
                fh.write(
                    json.dumps(
                        {
                            "step": 0,
                            "micro": 0,
                            "rank": 0,
                            "tokens_file": "custom_tokens.npy",
                            "labels_file": "custom_labels.npy",
                        }
                    )
                    + "\n"
                )
            tokens, labels = acp._resolve_fixed_training_files(d, 0, 0, 0, 4)
            self.assertTrue(tokens.endswith("custom_tokens.npy"))
            self.assertTrue(labels.endswith("custom_labels.npy"))

    def test_glob_single_match_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            self._touch(d, "tokens_step0_micro0_rank0_seq99.npy")
            self._touch(d, "labels_step0_micro0_rank0_seq99.npy")
            # seq_len 4 misses the exact seq99 names, so only the glob
            # (which ignores the seq length) can resolve the pair.
            tokens, labels = acp._resolve_fixed_training_files(d, 0, 0, 0, 4)
            self.assertTrue(tokens.endswith("seq99.npy"))
            self.assertTrue(
                labels.endswith("labels_step0_micro0_rank0_seq99.npy")
            )

    def test_missing_files_raise(self):
        with (
            tempfile.TemporaryDirectory() as d,
            self.assertRaises(FileNotFoundError),
        ):
            acp._resolve_fixed_training_files(d, 0, 0, 0, 4)


class TestLoadFixedTrainingData(unittest.TestCase):
    """``LOAD_FIXED_DATA_PATH`` drives the 1D and 2D replay loaders."""

    def setUp(self):
        self._saved_count = acp._LOAD_FIXED_DATA_CALL_COUNT
        acp._LOAD_FIXED_DATA_CALL_COUNT = 0
        self._saved_env = os.environ.get("LOAD_FIXED_DATA_PATH")

    def tearDown(self):
        acp._LOAD_FIXED_DATA_CALL_COUNT = self._saved_count
        if self._saved_env is None:
            os.environ.pop("LOAD_FIXED_DATA_PATH", None)
        else:
            os.environ["LOAD_FIXED_DATA_PATH"] = self._saved_env

    def test_returns_none_without_the_env_var(self):
        os.environ.pop("LOAD_FIXED_DATA_PATH", None)
        self.assertIsNone(
            acp.load_fixed_training_data(_training_args(), 0, lambda n, a: n)
        )

    def test_one_dim_arrays(self):
        with tempfile.TemporaryDirectory() as d:
            np.save(
                os.path.join(d, "tokens_step0_micro0_rank0_seq4.npy"),
                np.array([1, 2, 3, 4]),
            )
            np.save(
                os.path.join(d, "labels_step0_micro0_rank0_seq4.npy"),
                np.array([5, 6, 7, 8]),
            )
            os.environ["LOAD_FIXED_DATA_PATH"] = d
            fixed = acp.load_fixed_training_data(
                _training_args(), 0, lambda n, a: n * 10
            )
        self.assertFalse(fixed.batched)
        self.assertEqual(fixed.input_ids, [1, 2, 3, 4])
        self.assertEqual(fixed.labels, [5, 6, 7, 8])
        self.assertEqual(fixed.position_ids, [0, 1, 2, 3])
        self.assertEqual(fixed.max_seq_len, 40)

    def test_two_dim_arrays(self):
        with tempfile.TemporaryDirectory() as d:
            np.save(
                os.path.join(d, "tokens_step0_micro0_rank0_seq4.npy"),
                np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            )
            np.save(
                os.path.join(d, "labels_step0_micro0_rank0_seq4.npy"),
                np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            )
            os.environ["LOAD_FIXED_DATA_PATH"] = d
            fixed = acp.load_fixed_training_data(
                _training_args(), 0, lambda n, a: n
            )
        self.assertTrue(fixed.batched)
        self.assertEqual(fixed.input_ids, [[1, 2, 3, 4], [5, 6, 7, 8]])
        self.assertEqual(fixed.position_ids, [[0, 1, 2, 3], [0, 1, 2, 3]])


class TestFixedDataIterAndSample(unittest.TestCase):
    """Batched / non-batched iteration and sampling views."""

    def test_batched_iter_and_sample(self):
        fixed = FixedTrainingData(
            input_ids=[[1, 2], [3, 4]],
            labels=[[1, 2], [3, 4]],
            position_ids=[[0, 1], [0, 1]],
            max_seq_len=2,
            batched=True,
        )
        self.assertEqual(acp.fixed_data_iter(fixed, ["x"]), [None, None])
        pos, ids, labels, pos2 = acp.fixed_data_sample(fixed, 1)
        self.assertEqual(ids, [[3, 4]])
        self.assertEqual(labels, [[3, 4]])
        self.assertEqual(pos, [[0, 1]])
        self.assertEqual(pos2, [[0, 1]])

    def test_non_batched_iter_and_sample(self):
        fixed = FixedTrainingData(
            input_ids=[1, 2],
            labels=[3, 4],
            position_ids=[0, 1],
            max_seq_len=2,
            batched=False,
        )
        self.assertEqual(acp.fixed_data_iter(fixed, ["x"]), ["x"])
        self.assertEqual(acp.fixed_data_iter(None, ["x"]), ["x"])
        pos, ids, labels, pos2 = acp.fixed_data_sample(fixed, 0)
        self.assertEqual(ids, [[1, 2]])
        self.assertEqual(labels, [[3, 4]])


class TestParamGradAndFlush(unittest.TestCase):
    """``_get_param_grad`` and ``flush_sequence_first_wgrad`` plumbing."""

    def test_get_param_grad_prefers_main_grad(self):
        weight = paddle.create_parameter([2, 2], dtype="float32")
        weight.main_grad = paddle.ones([2, 2], dtype="float32")
        np.testing.assert_allclose(
            acp._get_param_grad(weight).numpy(), np.ones([2, 2])
        )

    def test_flush_moves_stashed_wgrad_into_grad(self):
        linear = paddle.nn.Linear(4, 4)
        param = linear.weight
        # Give the parameter a real grad, then stash a seq-first wgrad.
        x = paddle.randn([3, 4], dtype="float32")
        linear(x).sum().backward()
        param._dsv4_hc_mapping_seqfirst_wgrad = paddle.ones_like(param) * 7.0

        acp.flush_sequence_first_wgrad(linear)

        np.testing.assert_allclose(param.grad.numpy(), np.full([4, 4], 7.0))
        self.assertIsNone(param._dsv4_hc_mapping_seqfirst_wgrad)

    def test_flush_skips_params_without_stash(self):
        linear = paddle.nn.Linear(4, 4)
        x = paddle.randn([3, 4], dtype="float32")
        linear(x).sum().backward()
        before = linear.weight.grad.numpy().copy()
        acp.flush_sequence_first_wgrad(linear)
        np.testing.assert_allclose(linear.weight.grad.numpy(), before)

    def test_flush_skips_param_with_stash_but_no_grad(self):
        # A fresh Linear has never run backward, so its weight grad is None.
        # With a stash present this hits the ``grad is None`` skip (line 1141):
        # the stash must be left untouched.
        linear = paddle.nn.Linear(4, 4)
        param = linear.weight
        self.assertIsNone(acp._get_param_grad(param))
        param._dsv4_attn_o_group_seqfirst_wgrad = paddle.ones_like(param) * 5.0

        acp.flush_sequence_first_wgrad(linear)

        self.assertIsNotNone(
            getattr(param, "_dsv4_attn_o_group_seqfirst_wgrad", None)
        )


class TestLossScaleHelpers(unittest.TestCase):
    """Loss-scaling entry points used by the trainer."""

    def tearDown(self):
        acp.LossScaleBeforeBackward.set_acc_steps(1)

    def test_set_loss_acc_steps_forwards_to_the_scaler(self):
        acp.set_loss_acc_steps(8)
        scaled = acp.LossScaleBeforeBackward.scale(paddle.to_tensor(16.0))
        self.assertAlmostEqual(float(scaled.numpy()), 2.0, places=6)

    def test_set_pipeline_loss_scale_single_step_is_noop(self):
        from paddlefleet.transformer.multi_token_prediction import (
            MTPLossAutoScaler,
        )

        saved = getattr(MTPLossAutoScaler, "main_loss_backward_scale", None)
        try:
            acp.set_pipeline_loss_scale(1)
            self.assertIs(
                getattr(MTPLossAutoScaler, "main_loss_backward_scale", None),
                saved,
            )
        finally:
            if saved is not None:
                MTPLossAutoScaler.set_loss_scale(saved)

    def test_set_pipeline_loss_scale_sets_the_autoscalers(self):
        from paddlefleet.transformer.dsa_attention import (
            DSAIndexerLossAutoScaler,
        )
        from paddlefleet.transformer.multi_token_prediction import (
            MTPLossAutoScaler,
        )

        saved_mtp = getattr(MTPLossAutoScaler, "main_loss_backward_scale", None)
        saved_dsa = getattr(
            DSAIndexerLossAutoScaler, "main_loss_backward_scale", None
        )
        try:
            acp.set_pipeline_loss_scale(4)
            self.assertAlmostEqual(
                float(MTPLossAutoScaler.main_loss_backward_scale.numpy()),
                0.25,
                places=6,
            )
        finally:
            if saved_mtp is not None:
                MTPLossAutoScaler.set_loss_scale(saved_mtp)
            if saved_dsa is not None:
                DSAIndexerLossAutoScaler.set_loss_scale(saved_dsa)


class TestHasOptimizerState(unittest.TestCase):
    """Pure predicate over checkpoint metadata."""

    def test_true_when_any_state_name_present(self):
        self.assertTrue(
            acp.has_optimizer_state(
                "layer.w", {"layer.w.moment1": 1}, [".moment1", ".moment2"]
            )
        )

    def test_false_when_absent(self):
        self.assertFalse(acp.has_optimizer_state("layer.w", {}, [".moment1"]))


@requires_torch
class TestCompatibleOGroupProjectionAccumulate(unittest.TestCase):
    """Running backward twice on one weight hits the accumulate branch (467)."""

    def test_backward_twice_accumulates_the_stash(self):
        g, r, d = 2, 5, 4
        b, s = 2, 3
        w_np = np.random.randn(g, r, d).astype("float32")
        x1_np = np.random.randn(b, s, g, d).astype("float32")
        x2_np = np.random.randn(b, s, g, d).astype("float32")

        def single(x_np):
            weight = _grad_enabled(paddle.to_tensor(w_np))
            x = _grad_enabled(paddle.to_tensor(x_np))
            acp.CompatibleOGroupProjection.apply(
                x, weight, g, r, 0, 0
            ).sum().backward()
            return weight._dsv4_attn_o_group_seqfirst_wgrad.numpy().copy()

        contribution1 = single(x1_np)
        contribution2 = single(x2_np)

        # Reuse a SINGLE weight object across two passes: the second backward
        # sees ``prev is not None`` and adds into the stash (466-467).
        weight = _grad_enabled(paddle.to_tensor(w_np))
        x1 = _grad_enabled(paddle.to_tensor(x1_np))
        acp.CompatibleOGroupProjection.apply(
            x1, weight, g, r, 0, 0
        ).sum().backward()
        first = weight._dsv4_attn_o_group_seqfirst_wgrad.numpy().copy()
        x2 = _grad_enabled(paddle.to_tensor(x2_np))
        acp.CompatibleOGroupProjection.apply(
            x2, weight, g, r, 0, 0
        ).sum().backward()
        accumulated = weight._dsv4_attn_o_group_seqfirst_wgrad.numpy()

        np.testing.assert_allclose(first, contribution1, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(
            accumulated, contribution1 + contribution2, rtol=1e-4, atol=1e-4
        )


@requires_torch
class TestLinearSeqfirstWgradElseBranches(unittest.TestCase):
    """The ``first >= second`` non-transposed 3D/4D flatten paths."""

    def _reference(self, inp, grad):
        return np.matmul(
            inp.cast("float32").numpy().reshape(-1, inp.shape[-1]).T,
            grad.cast("float32").numpy().reshape(-1, grad.shape[-1]),
        )

    def test_three_dim_first_ge_second_non_transposed(self):
        paddle.seed(30)
        hidden = 6
        inp = paddle.randn([4, 2, hidden], dtype="float32").cast("bfloat16")
        grad = paddle.randn([4, 2, 4], dtype="float32").cast("bfloat16")
        weight = paddle.randn([hidden, 4], dtype="float32").cast("bfloat16")

        out = acp.linear_seqfirst_wgrad(inp, grad, weight)

        self.assertEqual(out.shape, [hidden, 4])
        np.testing.assert_allclose(
            out.cast("float32").numpy(),
            self._reference(inp, grad),
            rtol=6e-2,
            atol=6e-2,
        )

    def test_four_dim_first_ge_second_non_transposed(self):
        paddle.seed(31)
        hidden, streams = 6, 3
        inp = paddle.randn([4, 2, streams, hidden], dtype="float32").cast(
            "bfloat16"
        )
        grad = paddle.randn([4, 2, streams, 4], dtype="float32").cast(
            "bfloat16"
        )
        weight = paddle.randn([hidden, 4], dtype="float32").cast("bfloat16")

        out = acp.linear_seqfirst_wgrad(inp, grad, weight)

        self.assertEqual(out.shape, [hidden, 4])
        np.testing.assert_allclose(
            out.cast("float32").numpy(),
            self._reference(inp, grad),
            rtol=6e-2,
            atol=6e-2,
        )


# APPEND_MARKER_GROUP_B
class TestResolveFixedTrainingFilesManifestBranches(unittest.TestCase):
    """Manifest mismatch ``continue`` paths (970/972/974) and break (978)."""

    def _touch(self, d, name):
        np.save(os.path.join(d, name), np.array([0]))

    def _write_manifest(self, d, records):
        with open(
            os.path.join(d, "manifest_rank0.jsonl"), "w", encoding="utf-8"
        ) as fh:
            fh.writelines(json.dumps(record) + "\n" for record in records)

    def test_manifest_skips_step_micro_rank_mismatches(self):
        with tempfile.TemporaryDirectory() as d:
            self._touch(d, "custom_tokens.npy")
            self._touch(d, "custom_labels.npy")
            self._write_manifest(
                d,
                [
                    {
                        "step": 9,
                        "micro": 0,
                        "rank": 0,
                        "tokens_file": "x",
                        "labels_file": "y",
                    },
                    {
                        "step": 0,
                        "micro": 9,
                        "rank": 0,
                        "tokens_file": "x",
                        "labels_file": "y",
                    },
                    {
                        "step": 0,
                        "micro": 0,
                        "rank": 9,
                        "tokens_file": "x",
                        "labels_file": "y",
                    },
                    {
                        "step": 0,
                        "micro": 0,
                        "rank": 0,
                        "tokens_file": "custom_tokens.npy",
                        "labels_file": "custom_labels.npy",
                    },
                ],
            )

            tokens, labels = acp._resolve_fixed_training_files(d, 0, 0, 0, 4)

            self.assertTrue(tokens.endswith("custom_tokens.npy"))
            self.assertTrue(labels.endswith("custom_labels.npy"))

    def test_manifest_break_on_missing_file_names(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_manifest(
                d,
                [
                    {
                        "step": 0,
                        "micro": 0,
                        "rank": 0,
                        "tokens_file": "",
                        "labels_file": "",
                    },
                ],
            )
            with self.assertRaises(FileNotFoundError):
                acp._resolve_fixed_training_files(d, 0, 0, 0, 4)


# APPEND_MARKER_GROUP_B
class TestLoadFixedTrainingDataRaises(unittest.TestCase):
    """``load_fixed_training_data`` rejects >2D and shape-mismatched arrays."""

    def setUp(self):
        self._saved_count = acp._LOAD_FIXED_DATA_CALL_COUNT
        acp._LOAD_FIXED_DATA_CALL_COUNT = 0
        self._saved_env = os.environ.get("LOAD_FIXED_DATA_PATH")

    def tearDown(self):
        acp._LOAD_FIXED_DATA_CALL_COUNT = self._saved_count
        if self._saved_env is None:
            os.environ.pop("LOAD_FIXED_DATA_PATH", None)
        else:
            os.environ["LOAD_FIXED_DATA_PATH"] = self._saved_env

    def test_three_dim_tokens_raise_value_error(self):
        with tempfile.TemporaryDirectory() as d:
            np.save(
                os.path.join(d, "tokens_step0_micro0_rank0_seq4.npy"),
                np.zeros([1, 2, 4], dtype="int64"),
            )
            np.save(
                os.path.join(d, "labels_step0_micro0_rank0_seq4.npy"),
                np.zeros([1, 2, 4], dtype="int64"),
            )
            os.environ["LOAD_FIXED_DATA_PATH"] = d
            acp._LOAD_FIXED_DATA_CALL_COUNT = 0
            with self.assertRaises(ValueError):
                acp.load_fixed_training_data(
                    _training_args(), 0, lambda n, a: n
                )

    def test_label_shape_mismatch_raises_value_error(self):
        with tempfile.TemporaryDirectory() as d:
            np.save(
                os.path.join(d, "tokens_step0_micro0_rank0_seq4.npy"),
                np.zeros([2, 4], dtype="int64"),
            )
            np.save(
                os.path.join(d, "labels_step0_micro0_rank0_seq4.npy"),
                np.zeros([2, 3], dtype="int64"),
            )
            os.environ["LOAD_FIXED_DATA_PATH"] = d
            acp._LOAD_FIXED_DATA_CALL_COUNT = 0
            with self.assertRaises(ValueError):
                acp.load_fixed_training_data(
                    _training_args(), 0, lambda n, a: n
                )


if __name__ == "__main__":
    unittest.main()
