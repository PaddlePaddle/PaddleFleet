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
"""Parity tests: python scattered quant ops vs paddlefleet_ops CUDA fused ops.

Two parts:
1. TestFusedOpParity: at the same granularity (128, pow2 scale), every python
   scattered op must be bit-exact against its CUDA fused counterpart.
2. TestEndToEndGranularityDiff: end-to-end ExpertsGroupGemmContiguousNode,
   comparing the w4a8 path (scattered ops, 1x32 quantization) and the
   original fp8 path (fused ops, 128 granularity) against a bf16 reference,
   and reporting the difference between the two paths.

Run (SM100, requires DeepGEMM with fp8*fp4 support so that
``paddlefleet_ops.deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous`` exists):
    source <DeepGEMM-with-fp8xfp4 env>/bin/activate
    PYTHONPATH=packages/paddlefleet_ops/src:src python -m pytest \
        tests/single_card_tests/fp8/test_w4a8_fused_op_parity.py -v
"""

import unittest

import numpy as np
import paddle
import paddle.nn.functional as F

from paddlefleet.transformer.moe.fp8_utils import (
    fuse_stack_fp8_quant_python,
    fuse_stack_transpose_fp8_quant_python,
    fuse_weighted_swiglu_fp8_quant_clamp_python,
    fuse_weighted_swiglu_fp8_quant_python,
    fused_act_dequant_python,
    quant_blockwize,
)

try:
    from paddlefleet_ops import (
        fuse_stack_fp8_quant,
        fuse_stack_transpose_fp8_quant,
        fuse_weighted_swiglu_fp8_quant,
        fuse_weighted_swiglu_fp8_quant_clamp,
    )

    HAS_FUSED_OPS = True
except (ImportError, RuntimeError):
    HAS_FUSED_OPS = False

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


def _bits_equal(a, b):
    """Bitwise comparison for fp8 tensors."""
    return bool((a.view("int8") == b.view("int8")).all())


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@unittest.skipUnless(
    HAS_FUSED_OPS and IS_SM100,
    "requires paddlefleet_ops CUDA fused quant ops on SM100 "
    "(pow2 scaling path)",
)
class TestFusedOpParity(unittest.TestCase):
    """Bit-exact parity at 128 granularity.

    On SM100 the CUDA fused ops use using_pow2_scaling=True (matching
    using_ue8m0_scale=True in the scattered ops); weight stack quantization
    uses 128x128 tiles and activations use 1x128.
    """

    def setUp(self):
        paddle.seed(2026)
        np.random.seed(2026)

    def test_fuse_stack_transpose_fp8_quant(self):
        """Python op is bit-exact vs CUDA fuse_stack_transpose_fp8_quant."""
        e, k, n = 2, 256, 512
        w_list = [paddle.randn([k, n], dtype="bfloat16") for _ in range(e)]
        # CUDA: (x, using_pow2_scaling, using_ue8m0_scale,
        #        output_scale_transpose)
        cq, cs = fuse_stack_transpose_fp8_quant(w_list, True, False, False)
        pq, ps = fuse_stack_transpose_fp8_quant_python(
            w_list, quant_method="128x128", quant_dtype="fp8"
        )
        self.assertEqual(list(pq.shape), list(cq.shape))
        self.assertEqual(list(ps.shape), list(cs.shape))
        self.assertTrue(bool((cs == ps).all()), "scale mismatch")
        self.assertTrue(_bits_equal(cq, pq), "fp8 payload mismatch")

    def test_fuse_stack_fp8_quant(self):
        """Python op is bit-exact vs CUDA fuse_stack_fp8_quant."""
        e, k, n = 2, 256, 512
        w_list = [paddle.randn([k, n], dtype="bfloat16") for _ in range(e)]
        cq, cs = fuse_stack_fp8_quant(w_list, True, False, False)
        pq, ps = fuse_stack_fp8_quant_python(
            w_list, quant_method="128x128", quant_dtype="fp8"
        )
        self.assertEqual(list(pq.shape), list(cq.shape))
        self.assertEqual(list(ps.shape), list(cs.shape))
        self.assertTrue(bool((cs == ps).all()), "scale mismatch")
        self.assertTrue(_bits_equal(cq, pq), "fp8 payload mismatch")

    def test_fuse_weighted_swiglu_fp8_quant(self):
        """Python op is bit-exact vs CUDA fuse_weighted_swiglu_fp8_quant."""
        m, h2 = 128, 512
        o1 = paddle.randn([m, h2], dtype="bfloat16")
        probs = paddle.rand([m, 1], dtype="float32")
        cq, cs = fuse_weighted_swiglu_fp8_quant(
            o1, probs, using_pow2_scaling=True, use_ue8m0=False
        )
        pq, ps = fuse_weighted_swiglu_fp8_quant_python(
            o1, probs, quant_method="1x128"
        )
        self.assertEqual(list(pq.shape), list(cq.shape))
        self.assertTrue(bool((cs == ps).all()), "scale mismatch")
        self.assertTrue(_bits_equal(cq, pq), "fp8 payload mismatch")

    def test_fuse_weighted_swiglu_fp8_quant_clamp(self):
        """Python op is bit-exact vs the CUDA fused clamp op."""
        m, h2 = 128, 512
        clamp_value = 1.0
        o1 = 3.0 * paddle.randn([m, h2], dtype="bfloat16")
        probs = paddle.rand([m, 1], dtype="float32")
        cq, cs = fuse_weighted_swiglu_fp8_quant_clamp(
            o1,
            probs,
            using_pow2_scaling=True,
            use_ue8m0=False,
            clamp_value=clamp_value,
        )
        pq, ps = fuse_weighted_swiglu_fp8_quant_clamp_python(
            o1, probs, clamp_value, quant_method="1x128"
        )
        self.assertEqual(list(pq.shape), list(cq.shape))
        self.assertTrue(bool((cs == ps).all()), "scale mismatch")
        self.assertTrue(_bits_equal(cq, pq), "fp8 payload mismatch")

    def test_fuse_weighted_swiglu_fp8_quant_clamp_probs_1d(self):
        """The e2e forward passes 1-D unzipped_probs [M]; the python op must
        match the CUDA op in that layout as well."""
        m, h2 = 128, 512
        clamp_value = 1.0
        o1 = 3.0 * paddle.randn([m, h2], dtype="bfloat16")
        probs = paddle.rand([m], dtype="float32")
        cq, cs = fuse_weighted_swiglu_fp8_quant_clamp(
            o1,
            probs,
            using_pow2_scaling=True,
            use_ue8m0=False,
            clamp_value=clamp_value,
        )
        pq, ps = fuse_weighted_swiglu_fp8_quant_clamp_python(
            o1, probs, clamp_value, quant_method="1x128"
        )
        self.assertTrue(bool((cs == ps).all()), "scale mismatch")
        self.assertTrue(_bits_equal(cq, pq), "fp8 payload mismatch")

    def test_quant_blockwize_vs_fp8_quant_blockwise(self):
        """quant_blockwize is bit-exact vs paddle fp8_quant_blockwise."""
        x = paddle.randn([64, 256], dtype="bfloat16")
        cq, cs = paddle.incubate.nn.functional.fp8_quant_blockwise(
            x,
            output_scale_transpose=False,
            quant_method="1x128",
            input_transpose=False,
        )
        pq, ps = quant_blockwize(x, quant_method="1x128", quant_dtype="fp8")
        self.assertTrue(bool((cs == ps).all()), "scale mismatch")
        self.assertTrue(_bits_equal(cq, pq), "fp8 payload mismatch")

    def test_fused_act_dequant(self):
        """Python dequant is bit-exact vs paddle fused_act_dequant."""
        x = paddle.randn([128, 256], dtype="bfloat16")
        xq, xs = paddle.incubate.nn.functional.fp8_quant_blockwise(
            x,
            output_scale_transpose=False,
            quant_method="1x128",
            input_transpose=False,
        )
        cd = paddle.incubate.nn.functional.fused_act_dequant(xq, xs)
        pd = fused_act_dequant_python(xq, xs)
        self.assertEqual(cd.dtype, pd.dtype)
        self.assertTrue(bool((cd == pd).all()), "dequant mismatch")


@unittest.skipUnless(
    HAS_FUSED_OPS and HAS_FP8_FP4_GEMM and IS_SM100,
    "requires CUDA fused ops + DeepGEMM with fp8*fp4 + SM100",
)
class TestEndToEndGranularityDiff(unittest.TestCase):
    """End-to-end: w4a8 (scattered ops, 1x32) vs original fp8 (fused ops,
    128 granularity) vs bf16 reference.

    The w4a8 path stores weights in fp4, so its quantization noise is
    markedly larger than the fp8 path; this test verifies both paths work
    and reports their deviation from the bf16 reference.
    """

    def setUp(self):
        paddle.seed(2026)
        np.random.seed(2026)

    @staticmethod
    def _make_node(num_experts, k, n, use_w4a8, dequant_input=False):
        """Build an ExpertsGroupGemmContiguousNode with the given quant path."""
        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        class _Experts:
            pass

        class _Map:
            pass

        experts = _Experts()
        experts.weight1 = (
            paddle.randn([num_experts, k, 2 * n], dtype="bfloat16") * 0.05
        )
        experts.weight2 = (
            paddle.randn([num_experts, n, k], dtype="bfloat16") * 0.05
        )
        experts.weight1.main_grad = None
        experts.weight2.main_grad = None
        custom_map = _Map()
        custom_map.grouped_gemm_experts = experts
        # Both paths run dw through the bf16 grouped gemm: w4a8 forces bf16
        # wgrad by design, and the fp8 path's fp8-wgrad kitchen_gemm is not
        # supported by cuBLAS on this SM100 setup.
        node = ExpertsGroupGemmContiguousNode(
            custom_map,
            use_fp8_mlp=True,
            moe_deep_gemm=True,
            moe_expert_fusion=True,
            use_bf16_gemm_weight_grad=True,
            use_w4a8=use_w4a8,
            dequant_input=dequant_input,
        )
        return node, experts

    @staticmethod
    def _clone_weights(dst_experts, src_experts):
        """Copy weights so both nodes start from identical parameters."""
        dst_experts.weight1 = src_experts.weight1.clone()
        dst_experts.weight2 = src_experts.weight2.clone()
        dst_experts.weight1.main_grad = None
        dst_experts.weight2.main_grad = None

    @staticmethod
    def _bf16_ref_forward(x, probs, tokens, experts, n):
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

    def test_forward_w4a8_vs_fp8_vs_bf16(self):
        """Both paths track the bf16 reference; w4a8 is noisier than fp8."""
        num_experts, k, n, tokens = 2, 256, 256, [128, 384]
        m = sum(tokens)

        node_w4a8, experts = self._make_node(num_experts, k, n, use_w4a8=True)
        node_fp8, experts_fp8 = self._make_node(
            num_experts, k, n, use_w4a8=False
        )
        self._clone_weights(experts_fp8, experts)

        x = paddle.randn([m, k], dtype="bfloat16")
        probs = paddle.rand([m, 1], dtype="float32")

        out_w4a8 = (
            node_w4a8.forward(x.clone(), probs, tokens)
            .astype("float32")
            .numpy()
        )
        out_fp8 = (
            node_fp8.forward(x.clone(), probs, tokens).astype("float32").numpy()
        )
        ref = self._bf16_ref_forward(x, probs, tokens, experts, n)

        cos_w4a8 = _cos(out_w4a8, ref)
        cos_fp8 = _cos(out_fp8, ref)
        cos_cross = _cos(out_w4a8, out_fp8)
        rel_w4a8 = float(np.linalg.norm(out_w4a8 - ref) / np.linalg.norm(ref))
        rel_fp8 = float(np.linalg.norm(out_fp8 - ref) / np.linalg.norm(ref))
        rel_cross = float(
            np.linalg.norm(out_w4a8 - out_fp8) / np.linalg.norm(out_fp8)
        )
        print(
            f"\n[e2e forward] cos(w4a8,bf16)={cos_w4a8:.6f} "
            f"rel_err={rel_w4a8:.4f} | cos(fp8,bf16)={cos_fp8:.6f} "
            f"rel_err={rel_fp8:.4f} | cos(w4a8,fp8)={cos_cross:.6f} "
            f"rel_diff={rel_cross:.4f}"
        )

        # fp8 1x128 path carries small quantization noise
        self.assertGreater(cos_fp8, 0.995)
        # w4a8 stacks fp4 weights in two chained layers: noticeably noisier
        # but still highly correlated
        self.assertGreater(cos_w4a8, 0.97)
        self.assertGreater(cos_cross, 0.97)
        # w4a8 must deviate more than fp8 (inherent fp4 weight noise);
        # otherwise the comparison itself is broken
        self.assertGreater(rel_w4a8, rel_fp8)

    def test_backward_w4a8_vs_fp8(self):
        """w4a8 backward grads stay correlated with the fp8 path."""
        num_experts, k, n, tokens = 2, 256, 256, [128, 384]
        m = sum(tokens)

        node_w4a8, experts = self._make_node(num_experts, k, n, use_w4a8=True)
        node_fp8, experts_fp8 = self._make_node(
            num_experts, k, n, use_w4a8=False
        )
        self._clone_weights(experts_fp8, experts)

        x = paddle.randn([m, k], dtype="bfloat16")
        probs = paddle.rand([m], dtype="float32")
        out_grad = paddle.randn([m, k], dtype="bfloat16")

        node_w4a8.forward(x.clone(), probs.unsqueeze(-1), tokens)
        dx_w4a8, pg_w4a8 = node_w4a8.backward(out_grad.clone(), probs)

        node_fp8.forward(x.clone(), probs.unsqueeze(-1), tokens)
        dx_fp8, pg_fp8 = node_fp8.backward(out_grad.clone(), probs)

        got_dx = dx_w4a8.astype("float32").numpy()
        ref_dx = dx_fp8.astype("float32").numpy()
        cos_dx = _cos(got_dx, ref_dx)
        rel_dx = float(np.linalg.norm(got_dx - ref_dx) / np.linalg.norm(ref_dx))
        cos_pg = _cos(
            pg_w4a8.astype("float32").numpy(),
            pg_fp8.astype("float32").numpy(),
        )
        print(
            f"\n[e2e backward] cos(dx: w4a8,fp8)={cos_dx:.6f} "
            f"rel_diff={rel_dx:.4f} | cos(probs_grad)={cos_pg:.6f}"
        )
        self.assertGreater(cos_dx, 0.97)
        self.assertGreater(cos_pg, 0.97)

        # dw comparison (both paths compute dw with the bf16 grouped gemm;
        # any difference comes from forward quantization noise)
        for name, w_a, w_b in (
            ("dw1", experts.weight1, experts_fp8.weight1),
            ("dw2", experts.weight2, experts_fp8.weight2),
        ):
            cos_dw = _cos(
                w_a.main_grad.astype("float32").numpy(),
                w_b.main_grad.astype("float32").numpy(),
            )
            print(f"[e2e backward] cos({name}: w4a8,fp8)={cos_dw:.6f}")
            self.assertGreater(cos_dw, 0.97)

    def test_backward_fp8_dequant_input(self):
        """fp8 path with dequant_input=True: the bf16 wgrad dequantizes the
        stored 1x128 fp8 input via fused_act_dequant. Forward, dx and dw2
        must be bit-identical to the flag-off run; dw1 only differs by the
        1x128 quant-dequant noise on x."""
        num_experts, k, n, tokens = 2, 256, 256, [128, 384]
        m = sum(tokens)
        results = {}
        for flag in (False, True):
            paddle.seed(2026)
            np.random.seed(2026)
            node, experts = self._make_node(
                num_experts, k, n, use_w4a8=False, dequant_input=flag
            )
            paddle.seed(7)
            x = paddle.randn([m, k], dtype="bfloat16")
            probs = paddle.rand([m], dtype="float32")
            out = node.forward(x.clone(), probs.unsqueeze(-1), tokens)
            out_grad = paddle.randn([m, k], dtype="bfloat16")
            dx, _ = node.backward(out_grad.clone(), probs)
            results[flag] = (
                out.astype("float32"),
                dx.astype("float32"),
                experts.weight1.main_grad.astype("float32"),
                experts.weight2.main_grad.astype("float32"),
            )
        out_a, dx_a, dw1_a, dw2_a = results[False]
        out_b, dx_b, dw1_b, dw2_b = results[True]
        self.assertTrue(bool((out_a == out_b).all()), "forward mismatch")
        self.assertTrue(bool((dx_a == dx_b).all()), "dx mismatch")
        self.assertTrue(bool((dw2_a == dw2_b).all()), "dw2 mismatch")
        c1 = _cos(dw1_a.numpy(), dw1_b.numpy())
        self.assertGreater(c1, 0.99, f"dw1 cos={c1}")


if __name__ == "__main__":
    unittest.main()
