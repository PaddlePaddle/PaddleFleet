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

"""``FLAGS_use_dsv4_accuracy`` must gate every DSV4 replay site in the mHC
hyper-connection stack (and the MTP contraction that reuses it).

The flag defaults to 0. With it off the mHC numeric paths have to stay exactly
where they were before the DSV4 replay landed: the historical Sinkhorn
normalization, the FP32 mapping-projection parameters, the native recompute
backward, the ``compute_h`` epsilon, the un-transposed ``H_res`` layout and the
reference learned output contraction. Turning the flag on swaps each of those
for its Megatron-aligned DSV4 twin. Every test below pins one
``hyper_connection.py`` call site on both sides of the flag with an observable
effect - a different dtype, a different epsilon, a different code path
(``assert_called`` / ``assert_not_called``) or a different numeric result.

The two multi_token_prediction.py call sites (:869 and :1077) are deliberately
NOT covered here; see ``TestMtpCallSitesAreDistributedOnly`` for why they cannot
be isolated on a single card.
"""

from __future__ import annotations

import types
import unittest
from unittest.mock import patch

import numpy as np
import paddle

from paddlefleet import accuracy_compatible_patch
from paddlefleet.transformer import hyper_connection


def _dsv4_flag(module, enabled):
    return patch.object(
        module, "use_dsv4_accuracy_compatible", return_value=enabled
    )


def _mhc_config(params_dtype="float32"):
    """Minimal config that drives ``HyperConnectionModule`` on a single card.

    ``mhc_single_stream_init=True`` keeps the projection init deterministic and
    off the model-parallel RNG-tracker fork (this box reports world_size==4
    even without an initialized comm group), and ``use_fused_mhc=False`` keeps
    every op on the native reference path so the flag is the only variable.
    """
    return types.SimpleNamespace(
        num_residual_streams=2,
        hidden_size=4,
        mhc_sinkhorn_iterations=4,
        mhc_single_stream_init=True,
        params_dtype=params_dtype,
        mhc_init_gating_factor=1.0,
        use_fused_mhc=False,
        high_precision_mhc=False,
        sequence_parallel=False,
    )


def _contract_config(params_dtype="float32"):
    """Minimal config for ``HyperConnectionContractLayer`` (a plain FleetLayer)."""
    return types.SimpleNamespace(
        num_residual_streams=2,
        hidden_size=4,
        num_nextn_predict_layers=0,
        enable_mtp_magic_send=False,
        separate_mtp_input=False,
        params_dtype=params_dtype,
        sequence_parallel=False,
    )


def _build_module(params_dtype="float32"):
    # Build with the flag OFF so the module lands on the native reference ops;
    # individual tests then pin the flag around the specific method under test.
    with _dsv4_flag(hyper_connection, False):
        return hyper_connection.HyperConnectionModule(
            _mhc_config(params_dtype), layer_number=0
        )


class TestSinkhornForwardGating(unittest.TestCase):
    """hyper_connection.py:107 - ``SinkhornKnopp.forward`` picks the DSV4
    exp-normalize replay only when the flag is on; otherwise it must keep the
    historical ``_sinkhorn_normalize`` softmax path."""

    def setUp(self):
        paddle.seed(0)
        self.logits = paddle.randn([1, 2, 2], dtype="float32")

    def test_flag_off_uses_the_historical_sinkhorn_normalize(self):
        with (
            _dsv4_flag(hyper_connection, False),
            patch.object(
                hyper_connection.SinkhornKnopp,
                "_sinkhorn_normalize",
                wraps=hyper_connection.SinkhornKnopp._sinkhorn_normalize,
            ) as normalize,
        ):
            out = hyper_connection.native_sinkhorn(self.logits, 4, 1e-6)

        normalize.assert_called_once()
        self.assertEqual(out.shape, [1, 2, 2])
        self.assertTrue(bool(paddle.isfinite(out).all()))

    def test_flag_on_takes_the_exp_normalize_replay(self):
        with (
            _dsv4_flag(hyper_connection, True),
            patch.object(
                hyper_connection.SinkhornKnopp, "_sinkhorn_normalize"
            ) as normalize,
        ):
            out = hyper_connection.native_sinkhorn(self.logits, 4, 1e-6)

        normalize.assert_not_called()
        # The replay still returns a well-formed non-negative mixing matrix.
        self.assertEqual(out.shape, [1, 2, 2])
        self.assertTrue(bool(paddle.isfinite(out).all()))
        self.assertTrue(bool((out >= 0).all()))


class TestSinkhornBackwardGating(unittest.TestCase):
    """hyper_connection.py:144 - ``SinkhornKnopp.backward`` routes to the
    ``compatible_sinkhorn_backward`` (torch) replay only when the flag is on;
    otherwise it recomputes the gradient with the native autograd path."""

    def setUp(self):
        paddle.seed(1)
        self.grad = paddle.randn([1, 2, 2], dtype="float32")

    def _run_backward(self):
        logits = paddle.randn([1, 2, 2], dtype="float32")
        logits.stop_gradient = False
        out = hyper_connection.native_sinkhorn(logits, 4, 1e-6)
        (out * self.grad).sum().backward()
        return logits.grad

    def test_flag_on_calls_the_compatible_backward(self):
        sentinel = paddle.full([1, 2, 2], 7.0, dtype="float32")
        with (
            _dsv4_flag(hyper_connection, True),
            patch.object(
                accuracy_compatible_patch,
                "compatible_sinkhorn_backward",
                return_value=sentinel,
            ) as compatible,
        ):
            grad = self._run_backward()

        compatible.assert_called_once()
        np.testing.assert_array_equal(grad.numpy(), sentinel.numpy())

    def test_flag_off_recomputes_without_the_compatible_backward(self):
        with (
            _dsv4_flag(hyper_connection, False),
            patch.object(
                accuracy_compatible_patch, "compatible_sinkhorn_backward"
            ) as compatible,
        ):
            grad = self._run_backward()

        compatible.assert_not_called()
        self.assertTrue(bool(paddle.isfinite(grad).all()))


class TestMappingProjectionParamDtype(unittest.TestCase):
    """hyper_connection.py:394 - ``HyperConnectionModule`` keeps its mapping
    parameters in FP32 unless the flag is on, in which case they follow
    ``config.params_dtype`` (the Megatron accuracy-compatible contract)."""

    def test_flag_off_forces_fp32_parameters(self):
        with _dsv4_flag(hyper_connection, False):
            module = hyper_connection.HyperConnectionModule(
                _mhc_config("bfloat16"), layer_number=0
            )
        self.assertEqual(module.mapping_proj.weight.dtype, paddle.float32)
        self.assertEqual(module.alpha_pre.dtype, paddle.float32)
        self.assertEqual(module.bias.dtype, paddle.float32)

    def test_flag_on_keeps_the_model_dtype(self):
        with _dsv4_flag(hyper_connection, True):
            module = hyper_connection.HyperConnectionModule(
                _mhc_config("bfloat16"), layer_number=0
            )
        self.assertEqual(module.mapping_proj.weight.dtype, paddle.bfloat16)
        self.assertEqual(module.alpha_pre.dtype, paddle.bfloat16)
        self.assertEqual(module.bias.dtype, paddle.bfloat16)


class TestProjectionAndNormGating(unittest.TestCase):
    """hyper_connection.py:557 - ``_projection_and_get_norm`` routes to the
    ``compatible_projection_and_norm`` replay only when the flag is on;
    otherwise it uses the native reference projection."""

    def setUp(self):
        self.module = _build_module("float32")
        paddle.seed(2)
        self.x = paddle.randn(
            [3, self.module.n * self.module.hidden_size], dtype="float32"
        )
        self.out_dim = self.module.n * self.module.n + 2 * self.module.n

    def test_flag_on_uses_the_compatible_projection(self):
        sentinel = (
            paddle.zeros([3, self.out_dim], dtype="float32"),
            paddle.ones([3, 1], dtype="float32"),
        )
        with (
            _dsv4_flag(hyper_connection, True),
            patch.object(
                accuracy_compatible_patch,
                "compatible_projection_and_norm",
                return_value=sentinel,
            ) as compatible,
        ):
            proj, r = self.module._projection_and_get_norm(self.x)

        compatible.assert_called_once()
        np.testing.assert_array_equal(proj.numpy(), sentinel[0].numpy())

    def test_flag_off_uses_the_native_projection(self):
        with (
            _dsv4_flag(hyper_connection, False),
            patch.object(
                accuracy_compatible_patch, "compatible_projection_and_norm"
            ) as compatible,
        ):
            proj, r = self.module._projection_and_get_norm(self.x)

        compatible.assert_not_called()
        self.assertEqual(proj.shape, [3, self.out_dim])
        self.assertEqual(r.shape, [3, 1])


class TestComputeHEpsilonGating(unittest.TestCase):
    """hyper_connection.py:618 - ``_compute_h`` feeds ``compute_h_eps`` to the
    mapping head normally, but the DSV4 replay forces the epsilon to ``0.0``.
    The epsilon is added straight onto ``h_pre = sigmoid(...) + eps``, so the
    two sides differ by exactly ``compute_h_eps``."""

    def setUp(self):
        self.module = _build_module("float32")
        paddle.seed(3)
        out_dim = self.module.n * self.module.n + 2 * self.module.n
        self.proj = paddle.randn([3, out_dim], dtype="float32")
        self.r = paddle.rand([3, 1], dtype="float32")

    def _eps_seen(self, enabled):
        with (
            _dsv4_flag(hyper_connection, enabled),
            patch.object(
                self.module,
                "_compute_h_op",
                wraps=self.module._compute_h_op,
            ) as compute_h_op,
        ):
            h_pre, _, _ = self.module._compute_h(self.proj, self.r)
        return compute_h_op.call_args.args[7], h_pre

    def test_flag_on_pins_the_epsilon_to_zero(self):
        eps_on, h_pre_on = self._eps_seen(True)
        eps_off, h_pre_off = self._eps_seen(False)

        self.assertEqual(eps_on, 0.0)
        self.assertEqual(eps_off, self.module.compute_h_eps)
        # h_pre = sigmoid(...) + eps, so off exceeds on by exactly the epsilon
        # (up to float32 rounding of the added constant).
        np.testing.assert_allclose(
            (h_pre_off - h_pre_on).numpy(),
            np.full(h_pre_on.shape, self.module.compute_h_eps),
            rtol=0,
            atol=1e-7,
        )


class TestApplyHResLayoutGating(unittest.TestCase):
    """hyper_connection.py:701 - ``apply_h_res`` applies ``H_res`` un-transposed
    under the DSV4 replay, but transposes it (``H_res^T @ residual``) on the
    historical path. With an asymmetric ``H_res`` the two layouts disagree."""

    def setUp(self):
        self.module = _build_module("float32")
        n, C = self.module.n, self.module.hidden_size
        # Asymmetric mixing matrix so transpose vs. no-transpose is observable.
        self.h_res = paddle.to_tensor(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32").reshape(
                [1, n, n]
            )
        )
        self.residual = paddle.to_tensor(
            np.arange(1, 1 + n * C, dtype="float32").reshape([1, n * C])
        )

    def _reference(self, transpose):
        n, C = self.module.n, self.module.hidden_size
        mat = self.h_res.reshape([n, n])
        if transpose:
            mat = mat.t()
        resid = self.residual.reshape([n, C])
        return paddle.matmul(mat, resid).reshape([1, n * C]).numpy()

    def test_flag_on_applies_h_res_without_transpose(self):
        with _dsv4_flag(hyper_connection, True):
            out = self.module.apply_h_res(self.h_res, self.residual)
        np.testing.assert_allclose(
            out.numpy(), self._reference(transpose=False)
        )

    def test_flag_off_applies_the_transposed_h_res(self):
        with _dsv4_flag(hyper_connection, False):
            out = self.module.apply_h_res(self.h_res, self.residual)
        np.testing.assert_allclose(out.numpy(), self._reference(transpose=True))

    def test_the_two_layouts_are_actually_distinguishable(self):
        with _dsv4_flag(hyper_connection, True):
            on = self.module.apply_h_res(self.h_res, self.residual)
        with _dsv4_flag(hyper_connection, False):
            off = self.module.apply_h_res(self.h_res, self.residual)
        self.assertFalse(np.array_equal(on.numpy(), off.numpy()))


class TestLearnedOutputContractGating(unittest.TestCase):
    """hyper_connection.py:896 - ``learned_output_contract`` routes to the
    ``CompatibleLearnedOutputContract`` replay only when the flag is on;
    otherwise it runs the native rsqrt / sigmoid-gated contraction."""

    def setUp(self):
        self.n = 2
        self.hidden = 4
        paddle.seed(4)
        self.hidden_states = paddle.randn(
            [2, 3, self.n * self.hidden], dtype="float32"
        )
        self.head_fn = paddle.randn(
            [self.n * self.hidden, self.n], dtype="float32"
        )
        self.base = paddle.zeros([self.n], dtype="float32")
        self.scale = paddle.ones([1], dtype="float32")

    def _call(self):
        return hyper_connection.HyperConnectionModule.learned_output_contract(
            self.hidden_states,
            self.head_fn,
            self.base,
            self.scale,
            self.n,
            1e-6,
        )

    def test_flag_on_uses_the_compatible_contract(self):
        sentinel = paddle.full([2, 3, self.hidden], 5.0, dtype="float32")
        fake = types.SimpleNamespace(apply=lambda *a, **k: sentinel)
        with (
            _dsv4_flag(hyper_connection, True),
            patch.object(
                accuracy_compatible_patch,
                "CompatibleLearnedOutputContract",
                fake,
            ),
        ):
            out = self._call()
        np.testing.assert_array_equal(out.numpy(), sentinel.numpy())

    def test_flag_off_uses_the_native_contract(self):
        with (
            _dsv4_flag(hyper_connection, False),
            patch.object(
                accuracy_compatible_patch, "CompatibleLearnedOutputContract"
            ) as compatible,
        ):
            out = self._call()
        compatible.apply.assert_not_called()
        self.assertEqual(out.shape, [2, 3, self.hidden])


class TestContractLayerParamDtype(unittest.TestCase):
    """hyper_connection.py:1115 - ``HyperConnectionContractLayer`` keeps its
    learned-contraction parameters in FP32 unless the flag is on, in which case
    they follow ``config.params_dtype``."""

    def test_flag_off_forces_fp32_parameters(self):
        with _dsv4_flag(hyper_connection, False):
            layer = hyper_connection.HyperConnectionContractLayer(
                _contract_config("bfloat16")
            )
        self.assertEqual(layer.hc_head_fn.dtype, paddle.float32)
        self.assertEqual(layer.hc_head_base.dtype, paddle.float32)
        self.assertEqual(layer.hc_head_scale.dtype, paddle.float32)

    def test_flag_on_keeps_the_model_dtype(self):
        with _dsv4_flag(hyper_connection, True):
            layer = hyper_connection.HyperConnectionContractLayer(
                _contract_config("bfloat16")
            )
        self.assertEqual(layer.hc_head_fn.dtype, paddle.bfloat16)
        self.assertEqual(layer.hc_head_base.dtype, paddle.bfloat16)
        self.assertEqual(layer.hc_head_scale.dtype, paddle.bfloat16)


class TestMtpCallSitesAreDistributedOnly(unittest.TestCase):
    """multi_token_prediction.py:869 and :1077 are intentionally not covered.

    Both live inside ``MultiTokenPredictionLayer``. Its ``__init__`` calls
    ``ProcessGroupCollection.use_mpu_process_groups()`` and builds tensor/
    column-parallel sublayers (``e_proj`` / ``h_proj`` via ``build_spec_layer``),
    which require an initialized model-parallel process group - so :869 (the
    ``hc_param_dtype`` gate) cannot be reached without a constructed distributed
    layer. :1077 sits deep in that layer's ``forward`` after ``hnorm`` /
    stream-reshape / mask handling and, on the flag-off branch, uses
    ``deferrable_linear`` and ``gather_from_tensor_model_parallel_region``.
    Neither site can be isolated on a single card without an initialized
    distributed process group, so per the single-card rule they are skipped.
    """

    @unittest.skip(
        "MTP :869/:1077 need an initialized model-parallel process group "
        "(ProcessGroupCollection + parallel sublayers); not single-card safe."
    )
    def test_mtp_sites_skipped(self):
        pass


if __name__ == "__main__":
    unittest.main()
