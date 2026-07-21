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

"""
Standalone repro for the ``GroupedOutputFP8.forward`` precision outliers
observed during DSv4 training (diff > 1e-3 samples).

The training-time shadow (see ``dsv4_hybrid_attention.py`` under
``PADDLEFLEET_FP8_SHADOW_DEBUG=1``) dumps the raw bf16 ``x`` / ``weight``
tensors whenever the normalized diff exceeds a threshold. This test loads
those dumps and re-runs the exact FP8 grouped GEMM to verify:

  1. The diff is reproducible off-line from the dumped inputs.
  2. ``paddle.einsum`` and ``deep_gemm.einsum`` in bf16 agree bit-for-bit
     (confirming the reference is not the source of the diff).

Run with a custom dump dir:

    PADDLEFLEET_FP8_DUMP_DIR=/path/to/dump_data \
        python -m unittest test_fp8_grouped_output_repro
"""

import glob
import os
import unittest

import numpy as np
import paddle
from paddlefleet_ops import deep_gemm
from paddlefleet_ops.deep_gemm.utils.math import (
    ceil_div,
    per_block_cast_to_fp8,
    per_token_cast_to_fp8,
)


DEFAULT_DUMP_DIR = (
    "/root/paddlejob/share-storage/gpfs/system-public/wangxiangzhe/baidu/dump_data"
)


def _calc_diff(a: paddle.Tensor, b: paddle.Tensor):
    """Normalized diff = 1 - 2·Σab / Σ(a²+b²) (DeepGEMM-style)."""
    with paddle.no_grad():
        a_f = a.detach().astype("float64").reshape([-1])
        b_f = b.detach().astype("float64").reshape([-1])
        denom = (a_f * a_f + b_f * b_f).sum()
        if float(denom) == 0.0:
            return 0.0, 0.0
        sim = 2.0 * (a_f * b_f).sum() / denom
        diff = float(1.0 - sim)
        max_abs = float((a_f - b_f).abs().max())
        return diff, max_abs


def fp8_grouped_output_forward(
    x: paddle.Tensor,
    weight_bf16: paddle.Tensor,
    num_groups: int,
    o_lora_rank: int,
    weight_quant: str = "128x128",
):
    """Mirror of GroupedOutputFP8.forward (no autograd, no ctx save).

    x:            [b, sq, g, d]  bf16
    weight_bf16:  [g, r, d]      bf16   (= weight.reshape([g, r, d]))
    returns:      [b, sq, g, r]  bf16

    NOTE: this uses ``paddle.incubate.nn.functional.fp8_quant_blockwise``,
    the exact quant op used in production. Kept for direct A/B comparison
    against the canonical DeepGEMM quant path in
    :func:`fp8_grouped_output_forward_dg_quant`.
    """
    b, sq, _, d = x.shape
    weight_for_gemm = weight_bf16.transpose([0, 2, 1]).contiguous()  # [g, d, r]

    x_2d = x.reshape([-1, d]).contiguous()
    x_fp8, x_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
        x_2d,
        output_scale_transpose=False,
        quant_method="1x128",
        input_transpose=False,
        using_pow2_scale=True,
    )[:2]
    x_fp8 = x_fp8.reshape([b * sq, num_groups, d])
    x_scale = x_scale.reshape([b * sq, num_groups, -1])

    weight_fp8, weight_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
        weight_for_gemm.reshape([-1, weight_for_gemm.shape[-1]]),
        output_scale_transpose=False,
        quant_method=weight_quant,
        input_transpose=False,
        using_pow2_scale=True,
    )[:2]
    weight_fp8 = weight_fp8.reshape([num_groups, d, o_lora_rank])
    weight_scale = weight_scale.reshape(
        [num_groups, d // 128, o_lora_rank // 128]
    )

    out = paddle.empty(
        [b * sq, num_groups, o_lora_rank], dtype=paddle.bfloat16
    )
    deep_gemm.fp8_einsum(
        "bhd,hdr->bhr",
        (x_fp8, x_scale),
        (weight_fp8, weight_scale),
        out,
        recipe=(1, 128, 128),
    )
    return out.reshape([b, sq, num_groups, o_lora_rank])


def fp8_grouped_output_forward_dg_quant(
    x: paddle.Tensor,
    weight_bf16: paddle.Tensor,
    num_groups: int,
    o_lora_rank: int,
    use_ue8m0: bool = True,
):
    """FP8 grouped output using DeepGEMM's canonical quant helpers.

    Mirrors DeepGEMM's own ``test_fp8_bhd_hdr_bhr`` in
    ``DeepGEMM/tests/test_einsum.py``:

      - Activation: ``per_token_cast_to_fp8(x.view(-1, d))``  (1x128 blocks)
      - Weight   : per-group loop of ``per_block_cast_to_fp8(y[i])``
                   (128x128 blocks, quantized independently for each group)

    Removes any dependency on ``paddle.incubate.nn.functional.fp8_quant_blockwise``
    so we can rule out the paddle quant op as a source of numerical error.
    """
    b, sq, _, d = x.shape
    assert num_groups == weight_bf16.shape[0]
    assert o_lora_rank == weight_bf16.shape[1]
    assert d == weight_bf16.shape[2]

    y = weight_bf16.transpose([0, 2, 1]).contiguous()  # [g, d, r]

    # --- activation: per_token, view [(b*sq), h=g, d] -----------------------
    x_view = x.reshape([b * sq, num_groups, d]).contiguous()
    x_fp8_flat, x_scale_flat = per_token_cast_to_fp8(
        x_view.reshape([-1, d]), use_ue8m0=use_ue8m0
    )
    # per DeepGEMM test: reshape to (b*h, d) and (b*h, d/128)
    x_fp8 = x_fp8_flat.reshape([b * sq, num_groups, d])
    x_scale = x_scale_flat.reshape(
        [b * sq, num_groups, ceil_div(d, 128)]
    )

    # --- weight: per-group per_block ---------------------------------------
    # Note: paddle currently doesn't support indexed assignment into a
    # float8_e4m3fn tensor, so we build the stack per group and concat.
    fp8_list = []
    scale_list = []
    for gi in range(num_groups):
        f_gi, s_gi = per_block_cast_to_fp8(
            y[gi].contiguous(), use_ue8m0=use_ue8m0
        )
        fp8_list.append(f_gi)
        scale_list.append(s_gi)
    w_fp8 = paddle.stack(fp8_list, axis=0)  # [g, d, r]
    w_scale = paddle.stack(scale_list, axis=0)  # [g, d/128, r/128]

    out = paddle.empty(
        [b * sq, num_groups, o_lora_rank], dtype=paddle.bfloat16
    )
    deep_gemm.fp8_einsum(
        "bhd,hdr->bhr",
        (x_fp8, x_scale),
        (w_fp8, w_scale),
        out,
    )
    return out.reshape([b, sq, num_groups, o_lora_rank])


def fp8_grouped_output_forward_per_group_gemm(
    x: paddle.Tensor,
    weight_bf16: paddle.Tensor,
    num_groups: int,
    o_lora_rank: int,
    use_ue8m0: bool = True,
):
    """FP8 grouped output computed as ``num_groups`` independent FP8 GEMMs.

    Instead of one big grouped ``fp8_einsum('bhd,hdr->bhr', ...)`` call,
    this splits the computation into per-group ``fp8_gemm_nn`` calls:

        for h in range(num_groups):
            A_h = x[:, :, h, :].reshape(b*sq, d)           # per_token quant
            B_h = weight[h].T                              # per_block quant
            out_h = fp8_gemm_nn(A_h, B_h)                  # [b*sq, r]
        out = stack(out_h, dim=1)                          # [b*sq, g, r]

    Purpose: cross-check the grouped einsum against N independent single
    GEMMs to detect any layout/scale-interpretation issues specific to the
    grouped kernel.
    """
    b, sq, _, d = x.shape
    r = o_lora_rank
    y = weight_bf16.transpose([0, 2, 1]).contiguous()  # [g, d, r]

    per_group_outs = []
    for gi in range(num_groups):
        x_gi = (
            x[:, :, gi, :].reshape([b * sq, d]).contiguous()
        )  # [b*sq, d] bf16
        w_gi = y[gi].contiguous()  # [d, r] bf16

        x_fp8, x_scale = per_token_cast_to_fp8(x_gi, use_ue8m0=use_ue8m0)
        w_fp8, w_scale = per_block_cast_to_fp8(w_gi, use_ue8m0=use_ue8m0)

        out_gi = paddle.empty([b * sq, r], dtype=paddle.bfloat16)
        deep_gemm.fp8_gemm_nn(
            (x_fp8, x_scale), (w_fp8, w_scale), out_gi
        )
        per_group_outs.append(out_gi)

    out = paddle.stack(per_group_outs, axis=1)  # [b*sq, g, r]
    return out.reshape([b, sq, num_groups, r])


def _load_dump(path: str):
    data = np.load(path)
    place = paddle.CUDAPlace(0)
    x = paddle.to_tensor(data["x"], place=place).astype("bfloat16")
    weight = paddle.to_tensor(data["weight"], place=place).astype("bfloat16")
    meta = {
        "x_dtype": str(data["x_dtype"]),
        "weight_dtype": str(data["weight_dtype"]),
        "diff_fp8_vs_paddle": float(data["diff_fp8_vs_paddle"]),
        "diff_fp8_vs_dg": float(data["diff_fp8_vs_dg"]),
        "diff_paddle_vs_dg": float(data["diff_paddle_vs_dg"]),
    }
    return x, weight, meta


class FP8GroupedOutputReproTest(unittest.TestCase):
    """Offline repro of the fp8 grouped_output precision outliers."""

    DUMP_DIR = os.environ.get("PADDLEFLEET_FP8_DUMP_DIR", DEFAULT_DUMP_DIR)
    # allow tiny relative wiggle from reduction-order non-determinism
    REL_TOL = 0.05

    @classmethod
    def setUpClass(cls):
        if not paddle.is_compiled_with_cuda():
            raise unittest.SkipTest("CUDA required")
        paddle.set_device("gpu")
        cls.files = sorted(
            glob.glob(os.path.join(cls.DUMP_DIR, "fwd_grouped_output_*.npz"))
        )
        if not cls.files:
            raise unittest.SkipTest(
                f"No dump files found under {cls.DUMP_DIR}"
            )

    def _run_repro(self, path: str):
        print(f"\n[repro] file: {os.path.basename(path)}")
        x, weight_bf16, meta = _load_dump(path)
        b, sq, num_groups, d = x.shape
        _, o_lora_rank, _ = weight_bf16.shape
        print(
            f"[repro] x={list(x.shape)} weight={list(weight_bf16.shape)} "
            f"num_groups={num_groups} o_lora_rank={o_lora_rank} d={d}"
        )

        # 1a) fp8 forward via paddle.incubate fp8_quant_blockwise (production path)
        fp8_out_paddle_quant = fp8_grouped_output_forward(
            x, weight_bf16, num_groups, o_lora_rank, weight_quant="128x128"
        )

        # 1b) fp8 forward via DeepGEMM canonical quant helpers
        #     (per_token_cast_to_fp8 + per_block_cast_to_fp8, per-group loop).
        #     Removes any influence from paddle's fp8_quant_blockwise op.
        fp8_out_dg_quant = fp8_grouped_output_forward_dg_quant(
            x, weight_bf16, num_groups, o_lora_rank, use_ue8m0=True
        )

        # 1c) fp8 forward as N independent fp8_gemm_nn calls (one per group)
        #     and stacked. Cross-check against the single grouped einsum.
        fp8_out_per_group = fp8_grouped_output_forward_per_group_gemm(
            x, weight_bf16, num_groups, o_lora_rank, use_ue8m0=True
        )

        # 2) paddle bf16 reference
        paddle_ref = paddle.einsum("bsgd,grd->bsgr", x, weight_bf16)

        # 3) deep_gemm bf16 reference (same layout as fp8 path)
        weight_for_gemm = weight_bf16.transpose([0, 2, 1]).contiguous()
        dg_ref_2d = paddle.empty(
            [b * sq, num_groups, o_lora_rank], dtype=paddle.bfloat16
        )
        deep_gemm.einsum(
            "bhd,hdr->bhr",
            x.reshape([b * sq, num_groups, d]).contiguous(),
            weight_for_gemm,
            dg_ref_2d,
        )
        dg_ref = dg_ref_2d.reshape([b, sq, num_groups, o_lora_rank])

        # --- diffs against paddle bf16 reference ---
        diff_fp_pd_paddle_quant, max_fp_pd_paddle_quant = _calc_diff(
            fp8_out_paddle_quant, paddle_ref
        )
        diff_fp_pd_dg_quant, max_fp_pd_dg_quant = _calc_diff(
            fp8_out_dg_quant, paddle_ref
        )
        diff_fp_pd_per_group, max_fp_pd_per_group = _calc_diff(
            fp8_out_per_group, paddle_ref
        )
        # --- diff between the fp8 variants ---
        diff_paddle_vs_dg_quant, max_paddle_vs_dg_quant = _calc_diff(
            fp8_out_paddle_quant, fp8_out_dg_quant
        )
        diff_grouped_vs_per_group, max_grouped_vs_per_group = _calc_diff(
            fp8_out_dg_quant, fp8_out_per_group
        )
        # --- reference consistency ---
        diff_pd_dg, max_pd_dg = _calc_diff(paddle_ref, dg_ref)

        print(
            f"[repro] fp8(paddle_quant)   vs bf16 ref : diff={diff_fp_pd_paddle_quant:.4e} "
            f"max_abs={max_fp_pd_paddle_quant:.4e} (dump={meta['diff_fp8_vs_paddle']:.4e})"
        )
        print(
            f"[repro] fp8(dg_quant)       vs bf16 ref : diff={diff_fp_pd_dg_quant:.4e} "
            f"max_abs={max_fp_pd_dg_quant:.4e}"
        )
        print(
            f"[repro] fp8(per_group_gemm) vs bf16 ref : diff={diff_fp_pd_per_group:.4e} "
            f"max_abs={max_fp_pd_per_group:.4e}"
        )
        print(
            f"[repro] fp8(paddle_quant) vs fp8(dg_quant)    : diff={diff_paddle_vs_dg_quant:.4e} "
            f"max_abs={max_paddle_vs_dg_quant:.4e}"
        )
        print(
            f"[repro] fp8(dg_quant)     vs fp8(per_group)   : diff={diff_grouped_vs_per_group:.4e} "
            f"max_abs={max_grouped_vs_per_group:.4e}"
        )
        print(
            f"[repro] paddle_bf16 vs dg_bf16 einsum          : diff={diff_pd_dg:.4e} "
            f"max_abs={max_pd_dg:.4e} (dump={meta['diff_paddle_vs_dg']:.4e})"
        )

        # paddle_vs_dg (bf16) must be effectively 0
        self.assertLess(
            diff_pd_dg,
            1e-8,
            f"paddle.einsum vs deep_gemm.einsum diff={diff_pd_dg} unexpectedly large",
        )

        # reproduced paddle-quant diff should track the dumped value closely
        expected = meta["diff_fp8_vs_paddle"]
        rel_err = (
            abs(diff_fp_pd_paddle_quant - expected) / max(expected, 1e-12)
        )
        self.assertLess(
            rel_err,
            self.REL_TOL,
            f"reproduced diff {diff_fp_pd_paddle_quant:.4e} deviates "
            f"{rel_err*100:.2f}% from dumped {expected:.4e} "
            f"(tol={self.REL_TOL*100:.0f}%)",
        )

        # sanity: dump was captured because diff > 1e-3
        self.assertGreater(diff_fp_pd_paddle_quant, 1e-3)

        return {
            "paddle_quant": diff_fp_pd_paddle_quant,
            "dg_quant": diff_fp_pd_dg_quant,
            "per_group": diff_fp_pd_per_group,
            "quant_op_diff": diff_paddle_vs_dg_quant,
            "grouped_vs_per_group": diff_grouped_vs_per_group,
        }

    def test_repro_first_dump(self):
        """Reproduce the first dump: verify diff matches dumped value."""
        self._run_repro(self.files[0])

    def test_repro_all_dumps(self):
        """Reproduce every dump; skipped by default to keep test time short.

        Enable via env: PADDLEFLEET_FP8_REPRO_ALL=1
        """
        if os.environ.get("PADDLEFLEET_FP8_REPRO_ALL", "0") != "1":
            self.skipTest(
                "Set PADDLEFLEET_FP8_REPRO_ALL=1 to run over all dumps"
            )
        rows = []
        for f in self.files:
            rows.append(self._run_repro(f))

        def _stats(key):
            vals = [r[key] for r in rows]
            return min(vals), max(vals), sum(vals) / len(vals)

        for key in (
            "paddle_quant",
            "dg_quant",
            "per_group",
            "quant_op_diff",
            "grouped_vs_per_group",
        ):
            mn, mx, mean = _stats(key)
            print(
                f"[repro][summary] {key:>20}: n={len(rows)} "
                f"min={mn:.4e} max={mx:.4e} mean={mean:.4e}"
            )


if __name__ == "__main__":
    unittest.main()
