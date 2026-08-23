#!/usr/bin/env python3
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
"""Single-card tests for the SiTU-GLU FP8 MoE path.

Covers the code added by ``fp8 support situ activation func``:

* ``_weighted_situ_fp32`` -- fp32-throughout ``situ_glu(o1) * probs``.
* ``fuse_weighted_situ_fp8_quant_python`` -- activation + blockwise FP8 quant,
  including the ``clamp_value`` guard that rejects clamping.
* ``ExpertsGroupGemmContiguousNode.fwd_down`` -- ``geglu`` is now the only
  activation rejected on the FP8 path; ``situ`` dispatches to ``fwd_down_fp8``.
* ``fwd_down_fp8`` / ``bwd_down_input`` SiTU branches, exercised end-to-end
  through ``FusionMoePyLayer`` with real DeepGEMM FP8 kernels.
* ``MoELayer``'s narrowed SiTU+FP8 guard (now only w4a8 / SonicMoE raise).

The end-to-end tests need the repository copy of PaddleFleet, not the one in
site-packages::

    PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=0 \
        python -m pytest tests/single_card_tests/fp8/test_situ_fp8.py -v
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("FLAGS_use_virtual_memory_auto_growth", "True")
os.environ.setdefault("FLAGS_cudnn_deterministic", "True")

import numpy as np
import paddle
from paddle import nn

from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.transformer.activations import situ, situ_glu_scale_forward
from paddlefleet.transformer.moe.fp8_utils import (
    ExpertsGroupGemmContiguousNode,
    _weighted_situ_fp32,
    fuse_weighted_situ_fp8_quant_python,
    tilewise_quant,
)
from paddlefleet.transformer.transformer_config import TransformerConfig

_HAS_GPU = (
    paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
)
_REQUIRE_GPU = unittest.skipUnless(_HAS_GPU, "Requires a CUDA device")
# The DeepGEMM grouped FP8 GEMM used by the MoE path is SM90+.
_SM90_PLUS = _HAS_GPU and paddle.device.cuda.get_device_capability()[0] >= 9
_REQUIRE_SM90 = unittest.skipUnless(_SM90_PLUS, "Requires an SM90+ GPU")

if _HAS_GPU:
    model_parallel_cuda_manual_seed(1234)

BETA = 1.5
LINEAR_BETA = 2.0


def _situ_glu_reference(o1, beta, linear_beta):
    """Independent numpy reference for ``situ_glu`` on ``[gate, up]`` halves.

    Deliberately not written in terms of ``situ_glu``: reusing the function
    under test would make the assertion vacuous.
    """
    x = o1.astype("float32").numpy()
    half = x.shape[-1] // 2
    gate, up = x[:, :half], x[:, half:]
    gate_act = beta * np.tanh(gate / beta) / (1.0 + np.exp(-gate))
    up_act = (
        up if linear_beta is None else linear_beta * np.tanh(up / linear_beta)
    )
    return gate_act * up_act


def _dequant_blockwise(q, sf):
    """Undo 1x128 blockwise FP8 quantization: ``[M, K]`` x ``[M, K/128]``."""
    qf = q.astype("float32").numpy()
    s = sf.astype("float32").numpy()
    return qf * np.repeat(s, qf.shape[1] // s.shape[1], axis=1)


@_REQUIRE_GPU
class TestWeightedSituFp32(unittest.TestCase):
    """``_weighted_situ_fp32``: fp32 throughout, 1-D probs broadcast."""

    def setUp(self):
        paddle.seed(2026)
        self.o1 = paddle.randn([64, 256], dtype="bfloat16")

    def test_matches_independent_reference(self):
        probs = paddle.rand([64, 1])
        got = _weighted_situ_fp32(self.o1, probs, BETA, LINEAR_BETA)
        want = _situ_glu_reference(self.o1, BETA, LINEAR_BETA) * probs.numpy()
        np.testing.assert_allclose(got.numpy(), want, rtol=1e-5, atol=1e-5)

    def test_stays_fp32_for_bf16_input(self):
        """No bf16 down-cast may sneak in before quantization."""
        out = _weighted_situ_fp32(
            self.o1, paddle.rand([64, 1]), BETA, LINEAR_BETA
        )
        self.assertEqual(out.dtype, paddle.float32)

    def test_1d_probs_are_broadcast_per_row(self):
        """The end-to-end path passes ``unzipped_probs`` as 1-D ``[M]``."""
        probs_1d = paddle.rand([64])
        got = _weighted_situ_fp32(self.o1, probs_1d, BETA, LINEAR_BETA)
        want = _weighted_situ_fp32(
            self.o1, probs_1d.reshape([-1, 1]), BETA, LINEAR_BETA
        )
        self.assertEqual(list(got.shape), [64, 128])
        np.testing.assert_array_equal(got.numpy(), want.numpy())

    def test_linear_beta_none_leaves_up_branch_linear(self):
        probs = paddle.rand([64, 1])
        got = _weighted_situ_fp32(self.o1, probs, BETA, None)
        want = _situ_glu_reference(self.o1, BETA, None) * probs.numpy()
        np.testing.assert_allclose(got.numpy(), want, rtol=1e-5, atol=1e-5)


@_REQUIRE_GPU
class TestFuseWeightedSituFp8QuantPython(unittest.TestCase):
    """``fuse_weighted_situ_fp8_quant_python``: activation then 1x128 FP8."""

    def setUp(self):
        paddle.seed(2026)
        # I = 256 so the 1x128 scale factor tensor has more than one column;
        # a single column would hide per-block scale bugs.
        self.o1 = paddle.randn([64, 512], dtype="bfloat16")
        self.probs = paddle.rand([64])

    def _quant(self, **kw):
        kw.setdefault("using_ue8m0_scale", True)
        return fuse_weighted_situ_fp8_quant_python(
            self.o1, self.probs, BETA, LINEAR_BETA, **kw
        )

    def test_shapes_and_dtypes(self):
        q, sf = self._quant()
        self.assertEqual(q.dtype, paddle.float8_e4m3fn)
        self.assertEqual(sf.dtype, paddle.float32)
        self.assertEqual(list(q.shape), [64, 256])
        self.assertEqual(list(sf.shape), [64, 2])

    def test_dequant_recovers_fp32_activation(self):
        """q * sf must approximate the pre-quant fp32 activation.

        Tolerance is set by FP8 e4m3 itself: 4 total mantissa bits give a
        relative step of 2^-4, and UE8M0 rounds the block scale up to a power
        of two, so the worst case is ~2x that. 15% on the block maximum is a
        loose-but-meaningful bound -- a wrong scale or a transposed block
        blows past it by orders of magnitude.
        """
        q, sf = self._quant()
        want = _weighted_situ_fp32(
            self.o1, self.probs, BETA, LINEAR_BETA
        ).numpy()
        got = _dequant_blockwise(q, sf)
        scale = max(np.abs(want).max(), 1e-9)
        self.assertLess(np.abs(got - want).max() / scale, 0.15)

    def test_ue8m0_scales_are_powers_of_two(self):
        _, sf = self._quant(using_ue8m0_scale=True)
        exponent = np.log2(sf.numpy())
        np.testing.assert_allclose(
            exponent, np.round(exponent), rtol=0, atol=1e-6
        )

    def test_plain_scales_are_not_forced_to_powers_of_two(self):
        """``using_ue8m0_scale=False`` keeps the raw ``amax/448`` scale."""
        _, sf = self._quant(using_ue8m0_scale=False)
        exponent = np.log2(sf.numpy())
        self.assertTrue(np.abs(exponent - np.round(exponent)).max() > 1e-6)

    def test_clamp_value_is_rejected(self):
        """SiTU has no clamp-aware backward, so a positive clamp must fail."""
        with self.assertRaises(AssertionError) as cm:
            self._quant(clamp_value=7.0)
        self.assertIn("does not support clamp", str(cm.exception))

    def test_non_positive_clamp_is_accepted(self):
        """``clamp_value=0``/negative means "disabled", not an error."""
        for value in (None, 0.0, -1.0):
            with self.subTest(clamp_value=value):
                q, _ = self._quant(clamp_value=value)
                self.assertEqual(q.dtype, paddle.float8_e4m3fn)

    def test_activation_matches_situ_glu_scale_forward(self):
        """Same formula as the bf16 forward op used by the non-FP8 path."""
        q, sf = self._quant()
        got = _dequant_blockwise(q, sf)
        want = situ_glu_scale_forward(
            self.o1.astype("float32"),
            self.probs,
            beta=BETA,
            linear_beta=LINEAR_BETA,
            situ_glu_fusion=False,
        ).numpy()
        scale = max(np.abs(want).max(), 1e-9)
        self.assertLess(np.abs(got - want).max() / scale, 0.15)


class TestFwdDownActivationDispatch(unittest.TestCase):
    """``fwd_down`` gating: only ``geglu`` is still rejected on the FP8 path.

    Built with ``object.__new__`` and the handful of attributes the dispatcher
    reads. A real node would need stacked expert weights and a process group,
    which says nothing about the branch under test.
    """

    def _node(self, activation_type, use_fp8_mlp=True):
        node = object.__new__(ExpertsGroupGemmContiguousNode)
        node.activation_type = activation_type
        node.use_fp8_mlp = use_fp8_mlp
        node.calls = []
        node.fwd_down_fp8 = lambda *a, **k: node.calls.append("fp8") or "fp8"
        node.fwd_down_bf16 = lambda *a, **k: node.calls.append("bf16") or "bf16"
        return node

    def test_geglu_is_rejected_on_fp8_path(self):
        node = self._node("geglu")
        with self.assertRaises(ValueError) as cm:
            node.fwd_down(None, None, None, 1)
        message = str(cm.exception)
        self.assertIn("'swiglu' or", message)
        self.assertIn("situ", message)
        self.assertIn("geglu", message)

    def test_situ_and_swiglu_reach_fwd_down_fp8(self):
        for activation_type in ("situ", "swiglu"):
            with self.subTest(activation_type=activation_type):
                node = self._node(activation_type)
                self.assertEqual(node.fwd_down(None, None, None, 1), "fp8")
                self.assertEqual(node.calls, ["fp8"])

    def test_bf16_path_ignores_activation_type(self):
        """Without FP8 even ``geglu`` is fine -- the guard is FP8-only."""
        node = self._node("geglu", use_fp8_mlp=False)
        self.assertEqual(node.fwd_down(None, None, None, 1), "bf16")


class TestMoELayerSituFp8Guard(unittest.TestCase):
    """``MoELayer`` now rejects SiTU+FP8 only on w4a8 / SonicMoE."""

    @staticmethod
    def _config(**kw):
        config = TransformerConfig(
            hidden_size=256,
            moe_intermediate_size=128,
            gated_linear_unit=True,
            hidden_act=situ,
            n_routed_experts=2,
            num_experts_per_tok=1,
            fp8=True,
        )
        for key, value in kw.items():
            setattr(config, key, value)
        return config

    def _assert_guard_raises(self, **kw):
        from paddlefleet.transformer.moe.moe_layer import MoELayer

        with self.assertRaises(ValueError) as cm:
            MoELayer(self._config(**kw))
        self.assertIn("DeepGEMM fp8 path", str(cm.exception))

    def test_w4a8_backend_still_rejects_situ_fp8(self):
        self._assert_guard_raises(use_w4a8=True)

    def test_sonic_moe_backend_still_rejects_situ_fp8(self):
        self._assert_guard_raises(using_sonic_moe=True)


SEQ, TOPK, HIDDEN, INTER, N_EXPERT = 512, 4, 1024, 512, 4


class _FakeSituMoELayer(nn.Layer):
    """Minimal stand-in for ``MoELayer`` accepted by ``FusionMoePyLayer``.

    Mirrors ``FakeDeepGemmMOELayer`` in ``test_moe_subbatch_deep_gemm.py``, plus
    the ``config`` attribute: ``ExpertsGroupGemmContiguousNode`` reads
    ``activation_situ_beta`` / ``activation_situ_linear_beta`` /
    ``situ_glu_fusion`` off ``custom_map.config``, and silently falls back to
    ``beta=1.0, linear_beta=None`` when it is missing -- which would test the
    wrong formula.
    """

    def __init__(self, tokens_per_expert, linear_beta, situ_glu_fusion):
        super().__init__()
        from paddlefleet.transformer.moe.moe_expert import GroupedMLPExpert

        config = TransformerConfig(
            hidden_size=HIDDEN,
            moe_intermediate_size=INTER,
            gated_linear_unit=True,
            hidden_act=situ,
        )
        config.activation_situ_beta = BETA
        config.activation_situ_linear_beta = linear_beta
        config.situ_glu_fusion = situ_glu_fusion
        self.config = config
        self._activation_type = "situ"
        self.grouped_gemm_experts = GroupedMLPExpert(
            num_local_experts=N_EXPERT, config=config, moe_deep_gemm=True
        )
        with paddle.no_grad():
            for weight in (
                self.grouped_gemm_experts.weight1,
                self.grouped_gemm_experts.weight2,
            ):
                weight.set_value(
                    paddle.randn(weight.shape, dtype="bfloat16") * 0.02
                )
        self.token_dispatcher = SimpleNamespace(
            _comm_manager=SimpleNamespace(tokens_per_expert=tokens_per_expert)
        )
        self.experts = None

    def clear_main_grad(self):
        self.grouped_gemm_experts.weight1.main_grad = None
        self.grouped_gemm_experts.weight2.main_grad = None


@_REQUIRE_SM90
class TestSituFp8EndToEnd(unittest.TestCase):
    """SiTU + FP8 through ``FusionMoePyLayer`` with real DeepGEMM kernels.

    This is what covers the SiTU branches inside ``fwd_down_fp8`` and
    ``bwd_down_input``; they sit behind stacked-weight quantization and a
    grouped GEMM, so there is no honest way to reach them without running the
    real path.
    """

    def _run(
        self,
        use_fp8,
        use_ue8m0=False,
        linear_beta=LINEAR_BETA,
        situ_glu_fusion=True,
        activation_type="situ",
        clamp_value=None,
    ):
        """One forward+backward; returns (out, hs_grad, probs_grad, w2_grad)."""
        paddle.seed(2026)
        np.random.seed(2026)
        hidden = paddle.randn([SEQ, HIDDEN], "bfloat16")
        out_grad = paddle.randn_like(hidden)
        hidden_fp8, scale = tilewise_quant(hidden)
        probs = paddle.randn([SEQ, TOPK])

        indices = np.full([SEQ, TOPK], -1, dtype=np.int64)
        tokens_per_expert = [0] * N_EXPERT
        for row in range(SEQ):
            chosen = np.sort(
                np.random.choice(N_EXPERT, size=TOPK, replace=False)
            )
            indices[row] = chosen
            for expert in chosen:
                tokens_per_expert[expert] += 1

        # The bf16 reference must not be fed the FP8 tensor: the DeepGEMM bf16
        # grouped GEMM asserts on the dtype.
        x = hidden_fp8 if use_fp8 else hidden
        x.stop_gradient = False
        probs.stop_gradient = False

        layer = _FakeSituMoELayer(
            tokens_per_expert, linear_beta, situ_glu_fusion
        )
        layer = paddle.amp.decorate(layer, level="O2", dtype="bfloat16")
        layer.clear_main_grad()

        from paddlefleet.transformer.moe.fusion_layer_utils import (
            FusionMoePyLayer,
        )

        out = FusionMoePyLayer.apply(
            x,
            probs,
            paddle.to_tensor(indices),
            layer,
            TOPK,
            use_fp8_mlp=use_fp8,
            moe_deep_gemm=True,
            recompute_moe_gate_up=True,
            dequant_input=use_fp8,
            moe_expert_fusion=True,
            recompute_moe_premute=False,
            use_bf16_gemm_weight_grad=True,
            use_ue8m0=use_ue8m0,
            fp8_dispatched_handle={"scale": scale} if use_fp8 else None,
            use_auto_subbatch=False,
            activation_type=activation_type,
            clamp_value=clamp_value,
        )
        paddle.autograd.backward(out, out_grad)
        return (
            out,
            x.grad,
            probs.grad,
            layer.grouped_gemm_experts.weight2.main_grad,
        )

    _NAMES = ("out", "hidden_grad", "probs_grad", "weight2_grad")

    def test_forward_is_finite_for_both_ue8m0_settings(self):
        """Regression: the SiTU FP8 forward must not depend on ``use_ue8m0``.

        The SwiGLU branch pins ``using_pow2_scaling=True`` in its CUDA op, so
        the DeepGEMM SM100 GEMM always receives power-of-two block scales. If
        the SiTU branch forwards ``self.use_ue8m0`` into ``quant_blockwize``
        instead, then at the default ``use_ue8m0=False`` the kernel reads raw
        fp32 scales as packed UE8M0 and every output element comes back NaN.
        """
        for use_ue8m0 in (False, True):
            with self.subTest(use_ue8m0=use_ue8m0):
                out = self._run(True, use_ue8m0=use_ue8m0)[0]
                values = out.astype("float32").numpy()
                self.assertFalse(
                    np.isnan(values).any(),
                    f"{int(np.isnan(values).sum())}/{values.size} outputs are "
                    f"NaN at use_ue8m0={use_ue8m0}",
                )
                self.assertGreater(np.abs(values).max(), 0.0)

    def _assert_close_to_bf16(self, tgt, tol=0.15):
        """FP8 vs bf16 SiTU, compared per tensor against its own magnitude.

        ``tol`` is relative to each tensor's max absolute value. FP8 e4m3 alone
        costs a few percent, and the FP8 run additionally quantizes its input,
        so ~10% is expected here; the bound exists to catch a wrong formula or
        a mis-scaled block, not to certify precision.
        """
        ref = self._run(False)
        for name, a, b in zip(self._NAMES, ref, tgt):
            with self.subTest(tensor=name):
                self.assertIsNotNone(a, f"{name} missing in bf16 reference")
                self.assertIsNotNone(b, f"{name} missing in fp8 run")
                an = a.astype("float32").numpy()
                bn = b.astype("float32").numpy()
                self.assertEqual(an.shape, bn.shape, name)
                scale = max(np.abs(an).max(), 1e-9)
                self.assertLess(np.abs(an - bn).max() / scale, tol, name)

    def test_matches_bf16_reference(self):
        """Forward and all three gradients track the bf16 SiTU path."""
        self._assert_close_to_bf16(self._run(True, use_ue8m0=True))

    def test_matches_bf16_reference_without_triton_fusion(self):
        """``situ_glu_fusion=False`` exercises the pure-paddle backward."""
        ref = self._run(False, situ_glu_fusion=False)
        tgt = self._run(True, use_ue8m0=True, situ_glu_fusion=False)
        for name, a, b in zip(self._NAMES, ref, tgt):
            with self.subTest(tensor=name):
                an = a.astype("float32").numpy()
                bn = b.astype("float32").numpy()
                scale = max(np.abs(an).max(), 1e-9)
                self.assertLess(np.abs(an - bn).max() / scale, 0.15, name)

    def test_linear_beta_none_runs_end_to_end(self):
        """``linear_beta=None`` keeps the up branch linear on the FP8 path."""
        out, hidden_grad, probs_grad, weight_grad = self._run(
            True, use_ue8m0=True, linear_beta=None
        )
        for name, tensor in zip(
            self._NAMES, (out, hidden_grad, probs_grad, weight_grad)
        ):
            with self.subTest(tensor=name):
                self.assertIsNotNone(tensor, name)
                values = tensor.astype("float32").numpy()
                self.assertTrue(np.isfinite(values).all(), name)

    def test_swiglu_fp8_path_still_reachable(self):
        """The new SiTU branch must not divert SwiGLU, with or without clamp.

        Both SwiGLU sub-branches share the ``if situ / elif clamp / else``
        chain the commit rewrote, so a mis-ordered condition would silently
        route SwiGLU through the SiTU activation.
        """
        for clamp_value in (None, 7.0):
            with self.subTest(clamp_value=clamp_value):
                out = self._run(
                    True,
                    use_ue8m0=True,
                    activation_type="swiglu",
                    clamp_value=clamp_value,
                )[0]
                values = out.astype("float32").numpy()
                self.assertTrue(np.isfinite(values).all())
                self.assertGreater(np.abs(values).max(), 0.0)


if __name__ == "__main__":
    unittest.main()
