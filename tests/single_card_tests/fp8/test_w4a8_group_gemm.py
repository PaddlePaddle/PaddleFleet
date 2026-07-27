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
"""Unit tests for the w4a8 MoE MLP path (fp4 weights x fp8 activations,
1x32 blockwise online quantization).

Two parts:
1. Python scattered quant/dequant helpers in
   ``paddlefleet.transformer.moe.fp8_utils`` (any CUDA device).
2. End-to-end ``ExpertsGroupGemmContiguousNode(use_w4a8=True)``
   forward/backward against bf16/fp32 references (requires DeepGEMM with
   ``m_grouped_fp8_fp4_gemm_nt_contiguous`` and SM100).

Run:
    source <DeepGEMM-with-fp8xfp4 env>/bin/activate
    PYTHONPATH=packages/paddlefleet_ops/src:src python -m pytest \
        tests/single_card_tests/fp8/test_w4a8_group_gemm.py -v
"""

import unittest

import numpy as np
import paddle
import paddle.nn.functional as F

from paddlefleet.transformer.moe import fp8_utils
from paddlefleet.transformer.moe.fp8_utils import (
    W4A8_QUANT_BLOCK,
    ceil_to_ue8m0,
    fuse_stack_fp8_quant_python,
    fuse_stack_transpose_fp8_quant_python,
    fuse_weighted_swiglu_fp8_quant_clamp_python,
    fuse_weighted_swiglu_fp8_quant_python,
    fused_act_dequant_python,
    quant_blockwize,
)

try:
    from paddlefleet_ops import deep_gemm as _pf_deep_gemm

    HAS_FP8_FP4_GEMM = hasattr(
        _pf_deep_gemm, "m_grouped_fp8_fp4_gemm_nt_contiguous"
    )
except (ImportError, RuntimeError):
    HAS_FP8_FP4_GEMM = False

IS_SM100 = (
    paddle.is_compiled_with_cuda()
    and paddle.device.cuda.get_device_capability()[0] == 10
)

FP4_E2M1_MAX = 6.0
# e2m1 code (0~7) magnitudes
_FP4_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _unpack_fp4_to_float(packed, sf):
    """Unpack fp4 int8 [.., C/2] + scale [.., C/32] back to fp32 [.., C]."""
    raw = packed.numpy().astype(np.uint8)
    lo = raw & 0x0F
    hi = raw >> 4
    codes = np.stack([lo, hi], axis=-1).reshape(*raw.shape[:-1], -1)
    mag = _FP4_VALUES[codes & 0x7]
    val = np.where(codes >= 8, -mag, mag)
    sf_np = sf.numpy().astype(np.float64)
    val = val.reshape(*sf_np.shape, W4A8_QUANT_BLOCK) * sf_np[..., None]
    return val.reshape(*raw.shape[:-1], raw.shape[-1] * 2)


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _bits_equal(a, b):
    """Bitwise comparison for fp8/bf16 tensors."""
    if a.dtype == paddle.bfloat16:
        return bool((a.view("int16") == b.view("int16")).all())
    return bool((a.view("int8") == b.view("int8")).all())


class TestCeilToUe8m0(unittest.TestCase):
    def test_values(self):
        """ceil_to_ue8m0 rounds scales up to powers of two."""
        x = paddle.to_tensor(
            [0.5, 1.0, 1.5, 3.0, 0.3, 1e-4, 448.0], dtype="float32"
        )
        y = ceil_to_ue8m0(x).numpy()
        np.testing.assert_allclose(y, [0.5, 1.0, 2.0, 4.0, 0.5, 2**-13, 512.0])
        exp = np.log2(y)
        np.testing.assert_allclose(exp, np.round(exp))


class TestQuantBlockwize(unittest.TestCase):
    def setUp(self):
        paddle.seed(2026)
        np.random.seed(2026)

    def test_fp8_1x32_dequant_roundtrip(self):
        """fp8 1x32 quant-dequant roundtrip keeps relative error small."""
        x = paddle.randn([64, 256], dtype="bfloat16")
        x_fp8, sf = quant_blockwize(x, quant_method="1x32", quant_dtype="fp8")
        self.assertEqual(x_fp8.dtype, paddle.float8_e4m3fn)
        self.assertEqual(list(sf.shape), [64, 256 // W4A8_QUANT_BLOCK])
        # ue8m0: scales are powers of 2
        exp = np.log2(sf.numpy())
        np.testing.assert_allclose(exp, np.round(exp))
        x_dq = fused_act_dequant_python(x_fp8, sf)
        err = (x_dq.astype("float32") - x.astype("float32")).abs()
        ref = x.astype("float32").abs().clip(min=1e-2)
        self.assertLess(float((err / ref).mean()), 0.06)

    def test_fp8_no_ue8m0_exact_scale(self):
        """using_ue8m0_scale=False keeps the exact amax/448 scale."""
        x = paddle.randn([16, 64], dtype="bfloat16")
        x_fp8, sf = quant_blockwize(
            x, quant_dtype="fp8", using_ue8m0_scale=False
        )
        x_view = x.astype("float32").reshape([16, -1, W4A8_QUANT_BLOCK]).numpy()
        amax = np.clip(np.abs(x_view).max(axis=-1), 1e-4, None)
        np.testing.assert_allclose(sf.numpy(), amax / 448.0, rtol=1e-6)
        self.assertEqual(x_fp8.dtype, paddle.float8_e4m3fn)

    def test_empty_input(self):
        """Zero-row input returns empty tensors with correct shapes."""
        x = paddle.empty([0, 64], dtype="bfloat16")
        x_fp8, sf = quant_blockwize(x, quant_dtype="fp8")
        self.assertEqual(list(x_fp8.shape), [0, 64])
        self.assertEqual(list(sf.shape), [0, 2])
        x_fp4, sf4 = quant_blockwize(x, quant_dtype="fp4")
        self.assertEqual(list(x_fp4.shape), [0, 32])
        self.assertEqual(list(sf4.shape), [0, 2])

    def test_fp4_dequant_roundtrip(self):
        """fp4 1x32 quant-dequant roundtrip stays correlated with the input."""
        x = paddle.randn([32, 128], dtype="bfloat16")
        x_fp4, sf = quant_blockwize(x, quant_method="1x32", quant_dtype="fp4")
        self.assertEqual(x_fp4.dtype, paddle.int8)
        self.assertEqual(list(x_fp4.shape), [32, 64])
        self.assertEqual(list(sf.shape), [32, 128 // W4A8_QUANT_BLOCK])
        x_dq = _unpack_fp4_to_float(x_fp4, sf)
        x_ref = x.astype("float32").numpy()
        self.assertGreater(_cos(x_dq, x_ref), 0.98)
        block = x_ref.reshape(32, -1, W4A8_QUANT_BLOCK)
        amax = np.abs(block).max(axis=-1).clip(min=1e-4)
        self.assertTrue(
            (sf.numpy() * FP4_E2M1_MAX >= amax * 0.999).all(),
            "scale must cover per-block amax",
        )

    def test_fp4_no_ue8m0_exact_scale(self):
        """fp4 with using_ue8m0_scale=False keeps the exact amax/6 scale."""
        x = paddle.randn([8, 64], dtype="bfloat16")
        x_fp4, sf = quant_blockwize(
            x, quant_dtype="fp4", using_ue8m0_scale=False
        )
        x_view = x.astype("float32").reshape([8, -1, W4A8_QUANT_BLOCK]).numpy()
        amax = np.clip(np.abs(x_view).max(axis=-1), 1e-4, None)
        np.testing.assert_allclose(sf.numpy(), amax / FP4_E2M1_MAX, rtol=1e-6)
        # dequantized values must still track the input
        x_dq = _unpack_fp4_to_float(x_fp4, sf)
        self.assertGreater(_cos(x_dq, x_view.reshape(8, -1)), 0.98)

    def test_2d_tile_128x128_matches_manual_reference(self):
        """gran_m > 1 branch: 128x128 tile quantization for weights."""
        m, n = 256, 384
        x = paddle.randn([m, n], dtype="bfloat16")
        for use_ue8m0 in (True, False):
            q, sf = quant_blockwize(
                x,
                quant_method="128x128",
                quant_dtype="fp8",
                using_ue8m0_scale=use_ue8m0,
            )
            self.assertEqual(q.dtype, paddle.float8_e4m3fn)
            self.assertEqual(list(q.shape), [m, n])
            self.assertEqual(list(sf.shape), [m // 128, n // 128])
            # manual per-tile reference in numpy
            x32 = x.astype("float32").numpy()
            tiles = x32.reshape(m // 128, 128, n // 128, 128).transpose(
                0, 2, 1, 3
            )
            amax = np.clip(np.abs(tiles).max(axis=(2, 3)), 1e-4, None)
            ref_sf = amax / 448.0
            if use_ue8m0:
                ref_sf = np.exp2(np.ceil(np.log2(ref_sf)))
            np.testing.assert_allclose(sf.numpy(), ref_sf, rtol=1e-6)
            # dequantize back and check the error is bounded by fp8 noise
            q32 = q.astype("float32").numpy()
            q_tiles = q32.reshape(m // 128, 128, n // 128, 128).transpose(
                0, 2, 1, 3
            )
            dq = q_tiles * ref_sf[:, :, None, None]
            self.assertGreater(_cos(dq, tiles), 0.999)

    def test_2d_tile_rejects_fp4(self):
        """2D tile quantization only supports fp8; fp4 must be rejected."""
        x = paddle.randn([128, 128], dtype="bfloat16")
        with self.assertRaises(AssertionError):
            quant_blockwize(x, quant_method="128x128", quant_dtype="fp4")

    def test_2d_tile_rejects_non_divisible_m(self):
        """2D tile quantization requires M to be a multiple of gran_m."""
        x = paddle.randn([100, 128], dtype="bfloat16")
        with self.assertRaises(AssertionError):
            quant_blockwize(x, quant_method="128x128", quant_dtype="fp8")

    def test_rejects_non_divisible_k(self):
        """K must be a multiple of the quantization block size."""
        x = paddle.randn([4, 48], dtype="bfloat16")
        with self.assertRaises(AssertionError):
            quant_blockwize(x, quant_method="1x32", quant_dtype="fp8")

    def test_rejects_unsupported_dtype(self):
        """An unsupported quant_dtype raises ValueError."""
        x = paddle.randn([4, 64], dtype="bfloat16")
        with self.assertRaises(ValueError):
            quant_blockwize(x, quant_method="1x32", quant_dtype="int8")


class TestStackQuantHelpers(unittest.TestCase):
    def setUp(self):
        paddle.seed(2026)
        np.random.seed(2026)

    def test_fuse_stack_transpose_fp4(self):
        """Stacked weights are transposed then fp4-quantized along K."""
        e, k, n = 2, 64, 128
        w = paddle.randn([e, k, n], dtype="bfloat16")
        # transpose: [E, K, N] -> [E, N, K/2]
        w_fp4, sf = fuse_stack_transpose_fp8_quant_python(w, quant_dtype="fp4")
        self.assertEqual(w_fp4.dtype, paddle.int8)
        self.assertEqual(list(w_fp4.shape), [e, n, k // 2])
        self.assertEqual(list(sf.shape), [e, n, k // W4A8_QUANT_BLOCK])
        w_dq = _unpack_fp4_to_float(w_fp4, sf)
        w_ref = w.astype("float32").transpose([0, 2, 1]).numpy()
        self.assertGreater(_cos(w_dq, w_ref), 0.98)

    def test_fuse_stack_fp4(self):
        """Stacked weights are fp4-quantized along the last dim."""
        e, k, n = 2, 128, 64
        w = paddle.randn([e, k, n], dtype="bfloat16")
        w_fp4, sf = fuse_stack_fp8_quant_python(w, quant_dtype="fp4")
        self.assertEqual(list(w_fp4.shape), [e, k, n // 2])
        self.assertEqual(list(sf.shape), [e, k, n // W4A8_QUANT_BLOCK])
        w_dq = _unpack_fp4_to_float(w_fp4, sf)
        self.assertGreater(_cos(w_dq, w.astype("float32").numpy()), 0.98)

    def test_list_input(self):
        """A list of expert weights behaves exactly like the stacked tensor."""
        w_list = [paddle.randn([64, 32], dtype="bfloat16") for _ in range(3)]
        w_fp4, sf = fuse_stack_transpose_fp8_quant_python(
            w_list, quant_dtype="fp4"
        )
        self.assertEqual(list(w_fp4.shape), [3, 32, 32])
        self.assertEqual(list(sf.shape), [3, 32, 2])
        # a list must behave exactly like the stacked tensor
        w_fp4_ref, sf_ref = fuse_stack_transpose_fp8_quant_python(
            paddle.stack(w_list, axis=0), quant_dtype="fp4"
        )
        self.assertTrue(bool((w_fp4 == w_fp4_ref).all()))
        self.assertTrue(bool((sf == sf_ref).all()))

    def test_single_element_list_input(self):
        """A 1-element list holds an already-stacked [E, R, C] tensor and
        takes the w[0] branch in _stack_expert_weights."""
        w = paddle.randn([2, 64, 64], dtype="bfloat16")
        w_fp4_a, sf_a = fuse_stack_fp8_quant_python([w], quant_dtype="fp4")
        w_fp4_b, sf_b = fuse_stack_fp8_quant_python(w, quant_dtype="fp4")
        self.assertTrue(bool((w_fp4_a == w_fp4_b).all()))
        self.assertTrue(bool((sf_a == sf_b).all()))

    def test_2d_tile_layout_kept(self):
        """quant_method='128x128' keeps the 2D CUDA fused-op layout."""
        e, k, n = 2, 128, 256
        w = paddle.randn([e, k, n], dtype="bfloat16")
        q, sf = fuse_stack_fp8_quant_python(
            w, quant_method="128x128", quant_dtype="fp8"
        )
        self.assertEqual(list(q.shape), [e * k, n])
        self.assertEqual(list(sf.shape), [e * k // 128, n // 128])
        q_t, sf_t = fuse_stack_transpose_fp8_quant_python(
            w, quant_method="128x128", quant_dtype="fp8"
        )
        self.assertEqual(list(q_t.shape), [e * n, k])
        self.assertEqual(list(sf_t.shape), [e * n // 128, k // 128])


class TestWeightedSwigluQuant(unittest.TestCase):
    def setUp(self):
        paddle.seed(2026)
        np.random.seed(2026)

    def test_matches_fp32_reference(self):
        """swiglu(o1)*probs quantized output matches the fp32 reference."""
        m, h2 = 32, 128
        o1 = paddle.randn([m, h2], dtype="bfloat16")
        probs = paddle.rand([m, 1], dtype="float32")
        o2_fp8, sf = fuse_weighted_swiglu_fp8_quant_python(o1, probs)
        self.assertEqual(list(o2_fp8.shape), [m, h2 // 2])
        self.assertEqual(list(sf.shape), [m, h2 // 2 // W4A8_QUANT_BLOCK])
        x32 = o1.astype("float32")
        gate, up = x32[:, : h2 // 2], x32[:, h2 // 2 :]
        ref = (F.silu(gate) * up * probs).numpy()
        got = fused_act_dequant_python(o2_fp8, sf).astype("float32").numpy()
        err = np.abs(got - ref)
        denom = np.clip(np.abs(ref), 1e-2, None)
        self.assertLess(float((err / denom).mean()), 0.06)

    def test_clamp_matches_fp32_reference(self):
        """Clamped swiglu output matches the clamped fp32 reference."""
        m, h2 = 16, 128
        clamp_value = 1.0
        o1 = 5.0 * paddle.randn([m, h2], dtype="bfloat16")
        probs = paddle.rand([m, 1], dtype="float32")
        o2_fp8, sf = fuse_weighted_swiglu_fp8_quant_clamp_python(
            o1, probs, clamp_value
        )
        x32 = o1.astype("float32")
        gate = x32[:, : h2 // 2].clip(max=clamp_value)
        up = x32[:, h2 // 2 :].clip(min=-clamp_value, max=clamp_value)
        ref = (F.silu(gate) * up * probs).numpy()
        got = fused_act_dequant_python(o2_fp8, sf).astype("float32").numpy()
        err = np.abs(got - ref)
        denom = np.clip(np.abs(ref), 1e-2, None)
        self.assertLess(float((err / denom).mean()), 0.06)

    def test_probs_1d_matches_2d(self):
        """1-D probs [M] (as passed by the e2e forward) must broadcast the
        same way as [M, 1]."""
        m, h2 = 16, 128
        o1 = paddle.randn([m, h2], dtype="bfloat16")
        probs = paddle.rand([m], dtype="float32")
        q_1d, sf_1d = fuse_weighted_swiglu_fp8_quant_python(o1, probs)
        q_2d, sf_2d = fuse_weighted_swiglu_fp8_quant_python(
            o1, probs.unsqueeze(-1)
        )
        self.assertTrue(_bits_equal(q_1d, q_2d))
        self.assertTrue(bool((sf_1d == sf_2d).all()))
        clamp = 1.0
        qc_1d, sfc_1d = fuse_weighted_swiglu_fp8_quant_clamp_python(
            o1, probs, clamp
        )
        qc_2d, sfc_2d = fuse_weighted_swiglu_fp8_quant_clamp_python(
            o1, probs.unsqueeze(-1), clamp
        )
        self.assertTrue(_bits_equal(qc_1d, qc_2d))
        self.assertTrue(bool((sfc_1d == sfc_2d).all()))


class TestFusedActDequantPython(unittest.TestCase):
    def setUp(self):
        paddle.seed(2026)

    def test_1x128_granularity_inferred(self):
        """Granularity is inferred from the scale shape, so 1x128 fp8
        activations (the original fp8 path) also dequantize correctly."""
        x = paddle.randn([16, 256], dtype="bfloat16")
        x_fp8, sf = quant_blockwize(x, quant_method="1x128", quant_dtype="fp8")
        self.assertEqual(list(sf.shape), [16, 2])
        x_dq = fused_act_dequant_python(x_fp8, sf)
        err = (x_dq.astype("float32") - x.astype("float32")).abs()
        ref = x.astype("float32").abs().clip(min=1e-2)
        self.assertLess(float((err / ref).mean()), 0.06)


HAS_K_GROUPED_BF16_GEMM = HAS_FP8_FP4_GEMM and hasattr(
    _pf_deep_gemm, "k_grouped_bf16_gemm_tn_contiguous"
)


@unittest.skipUnless(
    HAS_K_GROUPED_BF16_GEMM and IS_SM100,
    "requires DeepGEMM with k_grouped_bf16_gemm_tn_contiguous on SM100",
)
class TestKGroupedBf16GemmAligned(unittest.TestCase):
    """k_grouped_bf16_gemm_tn_contiguous_aligned (used by the bf16 dw path)
    now calls the new DeepGEMM binding positionally; verify per-group
    d[i] = a_i^T @ b_i against a numpy reference."""

    def setUp(self):
        paddle.seed(2026)

    def _check(self, tokens, k=128, n=256):
        """Run the aligned grouped gemm and compare with a numpy reference."""
        from paddlefleet.transformer.moe.moe_utils import (
            k_grouped_bf16_gemm_tn_contiguous_aligned,
        )

        e = len(tokens)
        m = sum(tokens)
        a = paddle.randn([m, k], dtype="bfloat16")
        b = paddle.randn([m, n], dtype="bfloat16")
        d = paddle.zeros([e, k, n], dtype="float32")
        k_grouped_bf16_gemm_tn_contiguous_aligned(
            a,
            b,
            d,
            tokens,
            paddle.to_tensor(tokens, dtype="int32"),
            paddle.zeros([e, k, n], dtype="float32"),
        )
        ref = []
        start = 0
        for t in tokens:
            ref.append(
                (
                    a[start : start + t].astype("float32").T
                    @ b[start : start + t].astype("float32")
                ).numpy()
            )
            start += t
        ref = np.stack(ref)
        rel = np.abs(d.numpy() - ref).max() / (np.abs(ref).max() + 1e-9)
        self.assertLess(rel, 1e-5)

    def test_aligned_groups(self):
        """128-aligned groups compute per-group a^T @ b correctly."""
        self._check([128, 384])

    def test_unaligned_groups_are_padded(self):
        """Groups not 128-aligned are padded internally and stay correct."""
        # internal padding handles groups that are not 128-aligned
        self._check([100, 156])


def _make_experts_and_map(num_experts, k, n):
    """Build fake stacked expert weights and the custom map object."""

    class _Experts:
        pass

    class _Map:
        pass

    experts = _Experts()
    experts.weight1 = (
        paddle.randn([num_experts, k, 2 * n], dtype="bfloat16") * 0.05
    )
    experts.weight2 = paddle.randn([num_experts, n, k], dtype="bfloat16") * 0.05
    experts.weight1.main_grad = None
    experts.weight2.main_grad = None
    custom_map = _Map()
    custom_map.grouped_gemm_experts = experts
    return experts, custom_map


@unittest.skipUnless(
    HAS_FP8_FP4_GEMM and IS_SM100,
    "requires DeepGEMM with m_grouped_fp8_fp4_gemm_nt_contiguous on SM100",
)
class TestW4A8GroupGemm(unittest.TestCase):
    """End-to-end: ExpertsGroupGemmContiguousNode(use_w4a8) vs references.

    Note: the DeepGEMM SM100 contiguous layout requires the token count of
    each group to be 128-aligned (guaranteed by the dispatcher padding in
    real MoE runs); 0-token groups are allowed.
    """

    def setUp(self):
        paddle.seed(2026)
        np.random.seed(2026)

    def _make_node(
        self,
        num_experts,
        k,
        n,
        clamp_value=None,
        dequant_input=False,
        recompute_moe_gate_up=False,
    ):
        """Build an ExpertsGroupGemmContiguousNode with use_w4a8=True."""
        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        experts, custom_map = _make_experts_and_map(num_experts, k, n)
        node = ExpertsGroupGemmContiguousNode(
            custom_map,
            use_fp8_mlp=True,
            moe_deep_gemm=True,
            moe_expert_fusion=True,
            use_w4a8=True,
            clamp_value=clamp_value,
            dequant_input=dequant_input,
            recompute_moe_gate_up=recompute_moe_gate_up,
        )
        return node, experts

    def _bf16_ref_forward(self, x, probs, tokens, experts, n):
        """fp32 per-expert reference forward for the MoE MLP."""
        ref_parts = []
        start = 0
        for i, t in enumerate(tokens):
            xi = x[start : start + t].astype("float32")
            o1 = xi @ experts.weight1[i].astype("float32")
            gate, up = o1[:, :n], o1[:, n:]
            o2 = F.silu(gate) * up * probs[start : start + t]
            ref_parts.append(o2 @ experts.weight2[i].astype("float32"))
            start += t
        return paddle.concat(ref_parts, axis=0).numpy()

    # ---------------- construction-time branches ----------------

    def test_init_requires_prerequisites(self):
        """use_w4a8 without use_fp8_mlp must be rejected."""
        experts, custom_map = _make_experts_and_map(2, 128, 128)
        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        # use_fp8_mlp=False (with moe_deep_gemm kept on so that the
        # constructor reaches the w4a8 prerequisite assert) must be rejected
        with self.assertRaises(AssertionError):
            ExpertsGroupGemmContiguousNode(
                custom_map,
                use_fp8_mlp=False,
                moe_deep_gemm=True,
                moe_expert_fusion=True,
                use_w4a8=True,
            )

    def test_init_forces_bf16_weight_grad(self):
        """use_w4a8 forces the bf16 grouped-gemm weight-grad path."""
        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        experts, custom_map = _make_experts_and_map(2, 128, 128)
        node = ExpertsGroupGemmContiguousNode(
            custom_map,
            use_fp8_mlp=True,
            moe_deep_gemm=True,
            moe_expert_fusion=True,
            use_bf16_gemm_weight_grad=False,
            use_w4a8=True,
        )
        self.assertTrue(node.use_bf16_gemm_weight_grad)

    def test_default_use_w4a8_is_off(self):
        """use_w4a8 defaults to False."""
        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        experts, custom_map = _make_experts_and_map(2, 128, 128)
        node = ExpertsGroupGemmContiguousNode(
            custom_map,
            use_fp8_mlp=True,
            moe_deep_gemm=True,
            moe_expert_fusion=True,
        )
        self.assertFalse(node.use_w4a8)

    # ---------------- forward ----------------

    def test_forward_vs_bf16(self):
        """w4a8 forward output stays close to the bf16 reference."""
        num_experts, k, n, tokens = 2, 256, 256, [128, 128]
        node, experts = self._make_node(num_experts, k, n)
        m = sum(tokens)
        x = paddle.randn([m, k], dtype="bfloat16")
        probs = paddle.rand([m, 1], dtype="float32")

        o3 = node.forward(x, probs, tokens)

        ref = self._bf16_ref_forward(x, probs, tokens, experts, n)
        got = o3.astype("float32").numpy()
        # single-layer fp4 e2m1 tolerance ~0.01 (DeepGEMM reference);
        # relaxed to 0.97 for two chained layers
        self.assertGreater(_cos(got, ref), 0.97)

    def test_forward_uneven_tokens(self):
        """Uneven and zero-token groups still match the bf16 reference."""
        num_experts, k, n, tokens = 4, 128, 256, [128, 256, 0, 128]
        node, experts = self._make_node(num_experts, k, n)
        m = sum(tokens)
        x = paddle.randn([m, k], dtype="bfloat16")
        probs = paddle.rand([m, 1], dtype="float32")

        o3 = node.forward(x, probs, tokens)
        ref = self._bf16_ref_forward(x, probs, tokens, experts, n)
        self.assertGreater(_cos(o3.astype("float32").numpy(), ref), 0.97)

    def test_forward_clamp(self):
        """Forward with clamp_value matches the clamped bf16 reference."""
        num_experts, k, n, tokens = 2, 128, 128, [128, 128]
        clamp_value = 1.0
        node, experts = self._make_node(
            num_experts, k, n, clamp_value=clamp_value
        )
        m = sum(tokens)
        # Moderate saturation: some elements hit the clamp boundary, but not
        # so many that fp4 weight noise flips the clamp decision at the
        # boundary (an inherent quantization sensitivity).
        x = 2.0 * paddle.randn([m, k], dtype="bfloat16")
        probs = paddle.rand([m, 1], dtype="float32")

        o3 = node.forward(x, probs, tokens)

        ref_parts = []
        start = 0
        for i, t in enumerate(tokens):
            xi = x[start : start + t].astype("float32")
            o1 = xi @ experts.weight1[i].astype("float32")
            gate = o1[:, :n].clip(max=clamp_value)
            up = o1[:, n:].clip(min=-clamp_value, max=clamp_value)
            o2 = F.silu(gate) * up * probs[start : start + t]
            ref_parts.append(o2 @ experts.weight2[i].astype("float32"))
            start += t
        ref = paddle.concat(ref_parts, axis=0).numpy()
        # make sure the clamp actually kicks in
        o1_full = paddle.concat(
            [
                x[sum(tokens[:i]) : sum(tokens[: i + 1])].astype("float32")
                @ experts.weight1[i].astype("float32")
                for i in range(num_experts)
            ],
            axis=0,
        )
        self.assertGreater(
            float((o1_full.abs() > clamp_value).astype("float32").mean()),
            0.05,
        )
        self.assertGreater(_cos(o3.astype("float32").numpy(), ref), 0.97)

    # ---------------- backward ----------------

    def _autograd_reference(
        self, x, probs, tokens, experts, n, out_grad, clamp_value=None
    ):
        """fp32 paddle autograd reference; returns (dx, dprobs, dw1, dw2)."""
        x_ref = x.astype("float32").detach()
        x_ref.stop_gradient = False
        probs_ref = probs.astype("float32").detach()
        probs_ref.stop_gradient = False
        w1_ref = experts.weight1.astype("float32").detach()
        w1_ref.stop_gradient = False
        w2_ref = experts.weight2.astype("float32").detach()
        w2_ref.stop_gradient = False
        ref_parts = []
        start = 0
        for i, t in enumerate(tokens):
            xi = x_ref[start : start + t]
            o1 = xi @ w1_ref[i]
            gate, up = o1[:, :n], o1[:, n:]
            if clamp_value is not None:
                gate = gate.clip(max=clamp_value)
                up = up.clip(min=-clamp_value, max=clamp_value)
            o2 = F.silu(gate) * up * probs_ref[start : start + t].unsqueeze(-1)
            ref_parts.append(o2 @ w2_ref[i])
            start += t
        o3_ref = paddle.concat(ref_parts, axis=0)
        o3_ref.backward(out_grad.astype("float32"))
        return x_ref.grad, probs_ref.grad, w1_ref.grad, w2_ref.grad

    def test_backward(self):
        """w4a8 backward grads match the fp32 autograd reference."""
        num_experts, k, n, tokens = 2, 256, 256, [128, 384]
        node, experts = self._make_node(num_experts, k, n)
        m = sum(tokens)
        x = paddle.randn([m, k], dtype="bfloat16")
        probs = paddle.rand([m], dtype="float32")

        node.forward(x, probs.unsqueeze(-1), tokens)
        out_grad = paddle.randn([m, k], dtype="bfloat16")
        dx, probs_grad = node.backward(out_grad.clone(), probs)
        self.assertEqual(list(dx.shape), [m, k])
        self.assertEqual(probs_grad.shape[0], m)
        self.assertFalse(bool(paddle.isnan(dx.astype("float32")).any()))
        # dw goes through the bf16 path; main_grad must be populated
        self.assertIsNotNone(experts.weight1.main_grad)
        self.assertIsNotNone(experts.weight2.main_grad)

        ref_dx, ref_pg, ref_dw1, ref_dw2 = self._autograd_reference(
            x, probs, tokens, experts, n, out_grad
        )
        self.assertGreater(
            _cos(dx.astype("float32").numpy(), ref_dx.numpy()), 0.97
        )
        self.assertGreater(
            _cos(probs_grad.astype("float32").numpy(), ref_pg.numpy()), 0.97
        )
        for got_dw, ref_dw in (
            (experts.weight1.main_grad, ref_dw1),
            (experts.weight2.main_grad, ref_dw2),
        ):
            got_np = got_dw.astype("float32").numpy()
            self.assertFalse(np.isnan(got_np).any())
            self.assertGreater(_cos(got_np, ref_dw.numpy()), 0.97)

    def test_backward_clamp(self):
        """Covers the clamp branch of the w4a8 swiglu backward.

        Uses mild saturation (~8% of o1 elements clamped): fp4 weight noise
        flips the clamp decision for elements sitting on the boundary, which
        also flips the clip gradient mask, so heavier saturation inherently
        degrades the gradient match.
        """
        num_experts, k, n, tokens = 2, 128, 128, [128, 128]
        clamp_value = 1.0
        node, experts = self._make_node(
            num_experts, k, n, clamp_value=clamp_value
        )
        m = sum(tokens)
        x = paddle.randn([m, k], dtype="bfloat16")
        probs = paddle.rand([m], dtype="float32")

        # make sure the clamp actually kicks in
        o1_full = paddle.concat(
            [
                x[sum(tokens[:i]) : sum(tokens[: i + 1])].astype("float32")
                @ experts.weight1[i].astype("float32")
                for i in range(num_experts)
            ],
            axis=0,
        )
        self.assertGreater(
            float((o1_full.abs() > clamp_value).astype("float32").mean()),
            0.05,
        )

        node.forward(x, probs.unsqueeze(-1), tokens)
        out_grad = paddle.randn([m, k], dtype="bfloat16")
        dx, probs_grad = node.backward(out_grad.clone(), probs)
        self.assertFalse(bool(paddle.isnan(dx.astype("float32")).any()))

        ref_dx, ref_pg, ref_dw1, ref_dw2 = self._autograd_reference(
            x, probs, tokens, experts, n, out_grad, clamp_value=clamp_value
        )
        self.assertGreater(
            _cos(dx.astype("float32").numpy(), ref_dx.numpy()), 0.95
        )
        self.assertGreater(
            _cos(probs_grad.astype("float32").numpy(), ref_pg.numpy()), 0.95
        )
        for got_dw, ref_dw in (
            (experts.weight1.main_grad, ref_dw1),
            (experts.weight2.main_grad, ref_dw2),
        ):
            self.assertGreater(
                _cos(got_dw.astype("float32").numpy(), ref_dw.numpy()), 0.95
            )

    def test_backward_dequant_input(self):
        """dequant_input only changes the backward storage format (fp8 1x32
        instead of bf16): forward output and dx must be bit-identical to the
        flag-off run; dw1 sees quant-dequant noise on x, checked via cosine.
        """
        num_experts, k, n, tokens = 2, 256, 256, [128, 384]
        m = sum(tokens)
        results = {}
        for flag in (False, True):
            paddle.seed(2026)
            np.random.seed(2026)
            node, experts = self._make_node(
                num_experts, k, n, dequant_input=flag
            )
            paddle.seed(7)
            x = paddle.randn([m, k], dtype="bfloat16")
            probs = paddle.rand([m], dtype="float32")
            out = node.forward(x, probs.unsqueeze(-1), tokens)
            if flag:
                self.assertIsNotNone(node.input_fp8, "fp8 input must be stored")
                self.assertIsNone(node.input, "bf16 input must not be stored")
            out_grad = paddle.randn([m, k], dtype="bfloat16")
            dx, probs_grad = node.backward(out_grad.clone(), probs)
            results[flag] = (
                out.astype("float32"),
                dx.astype("float32"),
                experts.weight1.main_grad.astype("float32"),
                experts.weight2.main_grad.astype("float32"),
            )
        out_a, dx_a, dw1_a, dw2_a = results[False]
        out_b, dx_b, dw1_b, dw2_b = results[True]
        # the forward path does not depend on dequant_input: bit-identical
        self.assertTrue(bool((out_a == out_b).all()), "forward mismatch")
        # dx does not depend on the stored input: bit-identical
        self.assertTrue(bool((dx_a == dx_b).all()), "dx mismatch")
        # dw2 does not depend on x at all; dw1 uses the dequantized x and
        # carries 1x32 quantization noise
        self.assertTrue(bool((dw2_a == dw2_b).all()), "dw2 mismatch")
        c1 = _cos(dw1_a.numpy(), dw1_b.numpy())
        self.assertGreater(c1, 0.99, f"dw1 cos={c1}")

    def test_backward_recompute_gate_up(self):
        """recompute_moe_gate_up re-runs the w4a8 gate_up in backward with
        x=None; both storage formats must reproduce the no-recompute grads
        bit-exactly (the recomputed o1 reuses the same quantized inputs)."""
        num_experts, k, n, tokens = 2, 256, 256, [128, 384]
        m = sum(tokens)
        results = {}
        for recompute, dequant in (
            (False, False),
            (True, False),
            (True, True),
        ):
            paddle.seed(2026)
            np.random.seed(2026)
            node, experts = self._make_node(
                num_experts,
                k,
                n,
                dequant_input=dequant,
                recompute_moe_gate_up=recompute,
            )
            paddle.seed(7)
            x = paddle.randn([m, k], dtype="bfloat16")
            probs = paddle.rand([m], dtype="float32")
            out = node.forward(x, probs.unsqueeze(-1), tokens)
            out_grad = paddle.randn([m, k], dtype="bfloat16")
            dx, probs_grad = node.backward(out_grad.clone(), probs)
            results[(recompute, dequant)] = (
                out.astype("float32"),
                dx.astype("float32"),
                probs_grad.astype("float32"),
                experts.weight2.main_grad.astype("float32"),
            )
        base = results[(False, False)]
        for key in ((True, False), (True, True)):
            out, dx, pg, dw2 = results[key]
            self.assertTrue(
                bool((base[0] == out).all()), f"forward mismatch {key}"
            )
            self.assertTrue(bool((base[1] == dx).all()), f"dx mismatch {key}")
            self.assertTrue(
                bool((base[2] == pg).all()), f"probs_grad mismatch {key}"
            )
            self.assertTrue(bool((base[3] == dw2).all()), f"dw2 mismatch {key}")

    @unittest.skipUnless(
        fp8_utils.USE_INPLACE_SWIGLU_BWD,
        "requires the inplace fused_swiglu_probs_bwd op to compare branches",
    )
    def test_backward_out_of_place_swiglu_branch(self):
        """USE_INPLACE_SWIGLU_BWD=False falls back to
        fused_swiglu_weighted_bwd; grads must match the inplace branch."""
        num_experts, k, n, tokens = 2, 256, 256, [128, 128]
        m = sum(tokens)
        results = {}
        for inplace in (True, False):
            paddle.seed(2026)
            np.random.seed(2026)
            node, experts = self._make_node(num_experts, k, n)
            paddle.seed(7)
            x = paddle.randn([m, k], dtype="bfloat16")
            probs = paddle.rand([m], dtype="float32")
            node.forward(x, probs.unsqueeze(-1), tokens)
            out_grad = paddle.randn([m, k], dtype="bfloat16")
            orig = fp8_utils.USE_INPLACE_SWIGLU_BWD
            try:
                fp8_utils.USE_INPLACE_SWIGLU_BWD = inplace
                dx, probs_grad = node.backward(out_grad.clone(), probs)
            finally:
                fp8_utils.USE_INPLACE_SWIGLU_BWD = orig
            results[inplace] = (
                dx.astype("float32").numpy(),
                probs_grad.astype("float32").numpy(),
            )
        cos_dx = _cos(results[True][0], results[False][0])
        cos_pg = _cos(results[True][1], results[False][1])
        self.assertGreater(cos_dx, 0.999, f"dx cos={cos_dx}")
        self.assertGreater(cos_pg, 0.999, f"probs_grad cos={cos_pg}")

    # ---------------- private-method branches ----------------

    def test_fwd_gate_up_rejects_fp8_dispatch_scale(self):
        """w4a8 does not support fp8-dispatched (pre-quantized) input yet."""
        node, experts = self._make_node(2, 128, 128)
        x = paddle.randn([128, 128], dtype="bfloat16")
        scale = paddle.ones([128, 1], dtype="float32")
        with self.assertRaises(AssertionError):
            node._fwd_gate_up_w4a8(x, experts.weight1, scale=scale)

    def test_empty_input_shapes(self):
        """Zero-token inputs skip the grouped GEMM but keep output shapes."""
        k, n = 128, 128
        node, experts = self._make_node(2, k, n)
        x = paddle.empty([0, k], dtype="bfloat16")
        o1 = node._fwd_gate_up_w4a8(x, experts.weight1)
        self.assertEqual(list(o1.shape), [0, 2 * n])
        probs = paddle.empty([0, 1], dtype="float32")
        o3 = node._fwd_down_w4a8(o1, probs, experts.weight2)
        self.assertEqual(list(o3.shape), [0, k])
        do1 = paddle.empty([0, 2 * n], dtype="bfloat16")
        dx = node._bwd_gate_up_input_w4a8(do1, experts.weight1)
        self.assertEqual(list(dx.shape), [0, k])
        grad = paddle.empty([0, k], dtype="bfloat16")
        probs_1d = paddle.empty([0], dtype="float32")
        bdo1, o2_s, probs_grad = node._bwd_down_input_w4a8(
            experts.weight2, grad, o1, probs_1d
        )
        self.assertEqual(list(bdo1.shape), [0, 2 * n])
        self.assertEqual(list(o2_s.shape), [0, n])
        self.assertEqual(list(probs_grad.shape), [0])

    def test_fwd_down_preallocated_o3(self):
        """A caller-provided o3 buffer (auto-subbatch path) must produce the
        same bits as a freshly allocated one."""
        num_experts, k, n, tokens = 2, 128, 128, [128, 128]
        node, experts = self._make_node(num_experts, k, n)
        m = sum(tokens)
        node.m_indices = node.gen_m_indices(tokens)
        o1 = paddle.randn([m, 2 * n], dtype="bfloat16")
        probs = paddle.rand([m, 1], dtype="float32")
        out_fresh = node._fwd_down_w4a8(o1.clone(), probs, experts.weight2)
        buf = paddle.ones([m, k], dtype="bfloat16")
        out_buf = node._fwd_down_w4a8(
            o1.clone(), probs, experts.weight2, o3=buf
        )
        self.assertTrue(out_buf is buf)
        self.assertTrue(_bits_equal(out_fresh, out_buf))

    def test_bwd_gate_up_dx_none_matches_preallocated(self):
        """A caller-provided dx buffer produces the same bits as a fresh one."""
        num_experts, k, n, tokens = 2, 128, 128, [128, 128]
        node, experts = self._make_node(num_experts, k, n)
        m = sum(tokens)
        node.m_indices = node.gen_m_indices(tokens)
        do1 = paddle.randn([m, 2 * n], dtype="bfloat16")
        dx_fresh = node._bwd_gate_up_input_w4a8(do1, experts.weight1)
        buf = paddle.ones([m, k], dtype="bfloat16")
        dx_buf = node._bwd_gate_up_input_w4a8(do1, experts.weight1, dx=buf)
        self.assertTrue(dx_buf is buf)
        self.assertTrue(_bits_equal(dx_fresh, dx_buf))


if __name__ == "__main__":
    unittest.main()
