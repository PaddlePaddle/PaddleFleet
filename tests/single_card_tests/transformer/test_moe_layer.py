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

import contextlib
import hashlib
import io
import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.moe import moe_layer
    from paddlefleet.transformer.moe.moe_layer import (
        GradDtypeGuard,
        GradDtypeUnguard,
        MoESublayers,
    )
    from paddlefleet.transformer.transformer_layer import TransformerLayer

    _IMPORT_ERROR = None
except (
    ImportError,
    ModuleNotFoundError,
) as exc:  # CPU-only env may lack paddle
    paddle = None
    moe_layer = None
    GradDtypeGuard = GradDtypeUnguard = MoESublayers = None
    TransformerLayer = None
    _IMPORT_ERROR = exc

HAS_PADDLE = _IMPORT_ERROR is None
_SKIP_REASON = f"paddle/paddlefleet not importable on this env: {_IMPORT_ERROR}"


class _RecordingUnguardCtx:
    """Stand-in for a ctx that exposes ``set_grad_in_dtype_consistent``.

    Records the exact value the production forward pushes into it so the test
    can assert the flag is set (not merely that a method exists).
    """

    def __init__(self):
        self.calls = []

    def set_grad_in_dtype_consistent(self, value):
        self.calls.append(value)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestGradDtypeGuardStaticMethods(unittest.TestCase):
    """Directly exercise the real PyLayer static methods (forward/backward).

    These are the production functions; the test observes their actual returns
    rather than reimplementing them.
    """

    def test_guard_forward_emits_empty_status_of_requested_dtype(self):
        x = paddle.to_tensor([1.0, 2.0, 3.0], dtype="float32")

        status, saved = GradDtypeGuard.forward(None, x, "float16")

        # status is an empty carrier tensor whose dtype is the *requested*
        # dtype, not x's dtype -- proves the dtype argument is consumed.
        self.assertEqual(list(status.shape), [0])
        self.assertEqual(status.dtype, paddle.float16)
        # The original tensor is stashed by identity for the unguard step.
        self.assertIs(saved["x"], x)

    def test_guard_forward_status_dtype_tracks_argument(self):
        x = paddle.to_tensor([0.5], dtype="float32")

        status_fp32, _ = GradDtypeGuard.forward(None, x, "float32")
        status_bf16, _ = GradDtypeGuard.forward(None, x, "bfloat16")

        self.assertEqual(status_fp32.dtype, paddle.float32)
        self.assertEqual(status_bf16.dtype, paddle.bfloat16)

    def test_guard_backward_is_identity_on_gradient(self):
        grad = paddle.to_tensor([3.0, -1.0, 4.0], dtype="float32")

        out = GradDtypeGuard.backward(None, grad)

        # Passes the incoming gradient straight through (same object, values).
        self.assertIs(out, grad)
        np.testing.assert_array_equal(out.numpy(), [3.0, -1.0, 4.0])

    def test_unguard_forward_restores_saved_tensor_and_sets_flag(self):
        x = paddle.to_tensor([7.0, 8.0], dtype="float32")
        status, saved = GradDtypeGuard.forward(None, x, "float32")
        ctx = _RecordingUnguardCtx()

        # Production signature is forward(ctx, x, status); the paired call site
        # passes (status_tensor, saved_dict), so saved is bound to `status`.
        restored = GradDtypeUnguard.forward(ctx, status, saved)

        self.assertIs(restored, x)
        np.testing.assert_array_equal(restored.numpy(), [7.0, 8.0])
        # Flag is set exactly once, to False (grad dtype no longer consistent).
        self.assertEqual(ctx.calls, [False])

    def test_unguard_forward_without_flag_hook_still_restores(self):
        x = paddle.to_tensor([9.0], dtype="float32")
        _, saved = GradDtypeGuard.forward(None, x, "float32")

        # A ctx lacking set_grad_in_dtype_consistent must not raise; the
        # hasattr branch simply skips the flag update.
        restored = GradDtypeUnguard.forward(object(), None, saved)

        self.assertIs(restored, x)

    def test_unguard_backward_is_identity_on_gradient(self):
        grad = paddle.to_tensor([2.0, 5.0], dtype="float32")

        out = GradDtypeUnguard.backward(None, grad)

        self.assertIs(out, grad)
        np.testing.assert_array_equal(out.numpy(), [2.0, 5.0])


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestGradDtypeGuardApplyRoundTrip(unittest.TestCase):
    """Drive the public PyLayer.apply entry (forward path) end to end."""

    def test_apply_round_trip_preserves_values_and_dtype(self):
        x = paddle.to_tensor([1.0, 2.0, 3.0], dtype="float32")
        x.stop_gradient = False

        status, saved = GradDtypeGuard.apply(x, "float32")
        self.assertEqual(list(status.shape), [0])
        self.assertEqual(status.dtype, paddle.float32)
        self.assertIs(saved["x"], x)

        restored = GradDtypeUnguard.apply(status, saved)

        # Round trip is a value-preserving identity on the guarded tensor.
        np.testing.assert_array_equal(
            restored.numpy(), np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestMoESublayers(unittest.TestCase):
    def test_default_mlp_spec_is_none(self):
        self.assertIsNone(MoESublayers().mlp_spec)

    def test_custom_mlp_spec_is_stored_by_identity(self):
        class _MLPSpec:
            pass

        self.assertIs(MoESublayers(mlp_spec=_MLPSpec).mlp_spec, _MLPSpec)
        sentinel = object()
        self.assertIs(MoESublayers(mlp_spec=sentinel).mlp_spec, sentinel)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestLogMoeMd5(unittest.TestCase):
    """Behavior of the MD5 probe gate in moe_layer._log_moe_md5.

    Each gate (global flag, experimental version, MTP-skip) is toggled in
    isolation, and the printed line is checked against an independently
    computed MD5 rather than the production function's own output.
    """

    def setUp(self):
        self.addCleanup(
            setattr, moe_layer, "_LOG_LAYER_MD5", moe_layer._LOG_LAYER_MD5
        )
        self.addCleanup(
            setattr,
            TransformerLayer,
            "_gpt_model_use_experimental_version",
            TransformerLayer._gpt_model_use_experimental_version,
        )
        self.addCleanup(
            setattr,
            TransformerLayer,
            "_skip_mtp_probes",
            TransformerLayer._skip_mtp_probes,
        )

    @staticmethod
    def _capture(tensor, name, layer_idx):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            moe_layer._log_moe_md5(tensor, name, layer_idx)
        return buffer.getvalue()

    @staticmethod
    def _independent_md5(values):
        # Reproduce the production byte layout from numpy directly, without
        # calling _log_moe_md5, so the reference is not self-derived.
        data = np.asarray(values, dtype=np.float32).tobytes()
        return hashlib.md5(data).hexdigest()

    def test_prints_exact_line_when_all_gates_open(self):
        moe_layer._LOG_LAYER_MD5 = True
        TransformerLayer._gpt_model_use_experimental_version = True
        TransformerLayer._skip_mtp_probes = False

        tensor = paddle.to_tensor([1.0, 2.0], dtype="float32")
        output = self._capture(tensor, "hidden", 7)

        expected_md5 = self._independent_md5([1.0, 2.0])
        self.assertEqual(
            output,
            f"[MD5 MoE] Rank=0 Layer=7 hidden MD5={expected_md5} shape=[2]\n",
        )

    def test_layer_idx_none_omits_layer_field(self):
        moe_layer._LOG_LAYER_MD5 = True
        TransformerLayer._gpt_model_use_experimental_version = True
        TransformerLayer._skip_mtp_probes = False

        tensor = paddle.to_tensor([1.0, 2.0], dtype="float32")
        output = self._capture(tensor, "hidden", None)

        expected_md5 = self._independent_md5([1.0, 2.0])
        self.assertNotIn("Layer=", output)
        self.assertEqual(
            output,
            f"[MD5 MoE] Rank=0 hidden MD5={expected_md5} shape=[2]\n",
        )

    def test_md5_reflects_tensor_content(self):
        moe_layer._LOG_LAYER_MD5 = True
        TransformerLayer._gpt_model_use_experimental_version = True
        TransformerLayer._skip_mtp_probes = False

        out_a = self._capture(
            paddle.to_tensor([1.0, 2.0], dtype="float32"), "hidden", 3
        )
        out_b = self._capture(
            paddle.to_tensor([1.0, 2.5], dtype="float32"), "hidden", 3
        )

        # Different content -> different MD5, matching independent references.
        self.assertIn(f"MD5={self._independent_md5([1.0, 2.0])}", out_a)
        self.assertIn(f"MD5={self._independent_md5([1.0, 2.5])}", out_b)
        self.assertNotEqual(out_a, out_b)

    def test_skip_mtp_probes_suppresses_output(self):
        moe_layer._LOG_LAYER_MD5 = True
        TransformerLayer._gpt_model_use_experimental_version = True
        TransformerLayer._skip_mtp_probes = True

        output = self._capture(
            paddle.to_tensor([1.0], dtype="float32"), "hidden", 7
        )

        self.assertEqual(output, "")

    def test_global_flag_off_suppresses_output(self):
        moe_layer._LOG_LAYER_MD5 = False
        TransformerLayer._gpt_model_use_experimental_version = True
        TransformerLayer._skip_mtp_probes = False

        output = self._capture(
            paddle.to_tensor([1.0], dtype="float32"), "hidden", 7
        )

        self.assertEqual(output, "")

    def test_non_experimental_version_suppresses_output(self):
        moe_layer._LOG_LAYER_MD5 = True
        TransformerLayer._gpt_model_use_experimental_version = False
        TransformerLayer._skip_mtp_probes = False

        output = self._capture(
            paddle.to_tensor([1.0], dtype="float32"), "hidden", 7
        )

        self.assertEqual(output, "")


if __name__ == "__main__":
    unittest.main()
