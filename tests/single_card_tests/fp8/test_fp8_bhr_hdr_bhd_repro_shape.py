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
Paddle port of DeepGEMM's ``test_fp8_bhr_hdr_bhd`` in
``DeepGEMM/tests/test_einsum.py``, with the shape fixed to the one used
by ``GroupedOutputFP8`` in DSv4 (matches the tensors dumped under
``PADDLEFLEET_FP8_SHADOW_DEBUG=1``):

    h = 8                       (num_groups)
    r = 4096                    (contracting dim = production ``d``)
    d = 1024                    (output dim     = production ``o_lora_rank``)
    b = 32768                   (b*sq of the training config)

By default now loads x/y from a dumped ``.npz`` (same file used by the
grouped-output repro test), reshaped to match the ``bhr,hdr->bhd``
layout. Falls back to random bf16 tensors when no dump is available.
Purpose: give a ground-truth "fp8 blockwise diff on this shape" number,
independent of any production-side code paths.
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


def _calc_diff(x: paddle.Tensor, y: paddle.Tensor):
    """Same normalized diff metric used by DeepGEMM (``calc_diff``)."""
    with paddle.no_grad():
        xf = x.detach().astype("float64").reshape([-1])
        yf = y.detach().astype("float64").reshape([-1])
        denom = (xf * xf + yf * yf).sum()
        if float(denom) == 0.0:
            return 0.0, 0.0
        sim = 2.0 * (xf * yf).sum() / denom
        diff = float(1.0 - sim)
        max_abs = float((xf - yf).abs().max())
        return diff, max_abs


def _range_relative_rmse(a: paddle.Tensor, b: paddle.Tensor):
    """TE-style range-relative RMSE (see TransformerEngine
    ``tests/pytorch/utils.py::compare_and_assert``).

    Returns ``(rmse, rmse_range, rmse / rmse_range)`` where
    ``rmse_range = max(a, b) - min(a, b)`` is the joint value span of the
    two tensors (not amax). NaN-safe: if any element is NaN the ratio
    is reported as +inf so downstream asserts fail loudly rather than
    silently comparing against a nan value.
    """
    with paddle.no_grad():
        af = a.detach().astype("float64").reshape([-1])
        bf = b.detach().astype("float64").reshape([-1])
        diff = af - bf
        rmse = float(paddle.sqrt((diff * diff).mean()))
        a_max = float(af.max())
        a_min = float(af.min())
        b_max = float(bf.max())
        b_min = float(bf.min())
        rmse_range = max(a_max, b_max) - min(a_min, b_min)
        if not np.isfinite(rmse) or not np.isfinite(rmse_range) or rmse_range <= 0.0:
            return rmse, rmse_range, float("inf")
        return rmse, rmse_range, rmse / rmse_range


def _dynamic_range_global(t: paddle.Tensor):
    """Global dynamic range in ``log2`` space (TE ``stats_computation.py``
    ``_compute_dynamic_range_top`` / ``_bottom``).

    Uses ``|t|`` restricted to nonzero elements to avoid ``log2(0)``.
    Returns ``(top, bottom, dynamic_range = top - bottom)`` in octaves.
    E4M3 fits ~17 octaves, E5M2 fits ~32 octaves.
    """
    with paddle.no_grad():
        absx = t.detach().astype("float64").abs().reshape([-1])
        nz_mask = absx > 0
        if int(nz_mask.astype("int64").sum()) == 0:
            return 0.0, 0.0, 0.0
        # Replace zeros with +inf for amin so they don't dominate.
        amax = float(absx.max())
        absx_no_zero = paddle.where(
            nz_mask,
            absx,
            paddle.full_like(absx, float("inf")),
        )
        amin = float(absx_no_zero.min())
        top = float(np.log2(amax))
        bottom = float(np.log2(amin))
        return top, bottom, top - bottom


def _max_blockwise_dynamic_range_1d(t: paddle.Tensor, block: int, dim: int):
    """Per-1D-block ``log2`` dynamic range, then take the worst block.

    Mirrors TE ``compute_max_blockwise_dynamic_range(dims=1)``. Used for
    the activation (1x128 per-token) recipe.
    """
    with paddle.no_grad():
        absx = t.detach().astype("float64").abs()
        shape = list(absx.shape)
        assert shape[dim] % block == 0, (shape[dim], block)
        # Reshape target dim into (n_blocks, block).
        new_shape = shape[:dim] + [shape[dim] // block, block] + shape[dim + 1:]
        absx = absx.reshape(new_shape)
        block_axis = dim + 1
        amax = absx.max(axis=block_axis)
        absx_no_zero = paddle.where(
            absx > 0, absx, paddle.full_like(absx, float("inf"))
        )
        amin = absx_no_zero.min(axis=block_axis)
        # If a whole block is zero, its dynamic range is undefined; mask out.
        valid = (amax > 0).astype("float64")
        top = paddle.log2(paddle.clip(amax, min=1e-300))
        bottom = paddle.log2(paddle.clip(amin, min=1e-300))
        dr = (top - bottom) * valid
        return float(dr.max())


def _max_blockwise_dynamic_range_2d(t: paddle.Tensor, block: int, dims):
    """Per-2D-tile ``log2`` dynamic range, then take the worst tile.

    Mirrors TE ``compute_max_blockwise_dynamic_range(dims=2)``. Used for
    the weight (128x128 per-block) recipe.
    """
    d0, d1 = dims
    with paddle.no_grad():
        absx = t.detach().astype("float64").abs()
        shape = list(absx.shape)
        assert shape[d0] % block == 0 and shape[d1] % block == 0
        # Insert block axes for both dims. Doing it right-to-left keeps
        # earlier dim indices valid.
        assert d1 > d0
        new_shape = (
            shape[:d0]
            + [shape[d0] // block, block]
            + shape[d0 + 1:d1]
            + [shape[d1] // block, block]
            + shape[d1 + 1:]
        )
        absx = absx.reshape(new_shape)
        # After reshape, block axes are at (d0+1) and (d1+2).
        b0 = d0 + 1
        b1 = d1 + 2
        # Reduce over both block axes.
        amax = absx.max(axis=b1).max(axis=b0)
        absx_no_zero = paddle.where(
            absx > 0, absx, paddle.full_like(absx, float("inf"))
        )
        amin = absx_no_zero.min(axis=b1).min(axis=b0)
        valid = (amax > 0).astype("float64")
        top = paddle.log2(paddle.clip(amax, min=1e-300))
        bottom = paddle.log2(paddle.clip(amin, min=1e-300))
        dr = (top - bottom) * valid
        return float(dr.max())


# E4M3 fits ~17 octaves (including subnormals); use it as the reference
# ceiling for "does one scale cover this block?" checks.
_E4M3_OCTAVES = 17.0


class TestFp8BhrHdrBhdReproShape(unittest.TestCase):
    """FP8 einsum diff on the GroupedOutputFP8 shape, using random inputs."""

    # Repro-shape parameters (see module docstring).
    B = 32768
    H = 8
    R = 4096  # contracting
    D = 1024  # output

    # If set, load x/y from this dump file. Otherwise, look for the first
    # ``fwd_grouped_output_*.npz`` under ``PADDLEFLEET_FP8_DUMP_DIR``
    # (default: ``DEFAULT_DUMP_DIR``). Set env ``PADDLEFLEET_FP8_USE_RANDOM=1``
    # to force the original random-tensor behavior.
    DUMP_DIR = os.environ.get("PADDLEFLEET_FP8_DUMP_DIR", DEFAULT_DUMP_DIR)

    @classmethod
    def setUpClass(cls):
        if not paddle.is_compiled_with_cuda():
            raise unittest.SkipTest("CUDA required")
        paddle.set_device("gpu")
        paddle.seed(0)
        np.random.seed(0)

        cls.use_random = os.environ.get("PADDLEFLEET_FP8_USE_RANDOM", "0") == "1"
        cls.dump_path = None
        if not cls.use_random:
            files = sorted(
                glob.glob(os.path.join(cls.DUMP_DIR, "fwd_grouped_output_*.npz"))
            )
            if files:
                cls.dump_path = files[0]

    def _load_xy_from_dump(self):
        """Load and reshape dump into the ``bhr,hdr->bhd`` layout.

        Dump layout (from ``GroupedOutputFP8`` shadow dump):
            x      : [b, sq, g, d]        bf16 (contracting=d, last)
            weight : [g, o_lora_rank, d]  bf16 (contracting=d, last)

        Repro (this test) layout ``bhr,hdr->bhd``:
            x   : [B, H, R]   with R = contracting = dump's d
            y   : [H, D, R]   with D = output = dump's o_lora_rank
                              and R again = dump's d
        So the reshapes are:
            x_bhr = dump.x.reshape([b*sq, g, d])   ==>  [B, H, R]
            y_hdr = dump.weight                    ==>  [H, D, R] directly
        """
        data = np.load(self.dump_path)
        x_np = data["x"]  # [b, sq, g, d]
        w_np = data["weight"]  # [g, o_lora_rank, d]
        b_, sq_, g_, d_ = x_np.shape
        g2_, r_out_, d2_ = w_np.shape
        assert g_ == g2_, (g_, g2_)
        assert d_ == d2_, (d_, d2_)
        assert b_ * sq_ == self.B, (b_ * sq_, self.B)
        assert g_ == self.H, (g_, self.H)
        assert d_ == self.R, (d_, self.R)
        assert r_out_ == self.D, (r_out_, self.D)

        place = paddle.CUDAPlace(0)
        x = (
            paddle.to_tensor(x_np, place=place)
            .astype("bfloat16")
            .reshape([self.B, self.H, self.R])
        )
        y = paddle.to_tensor(w_np, place=place).astype("bfloat16")
        return x, y

    def _run_one(self, use_ue8m0: bool):
        b, h, r, d = self.B, self.H, self.R, self.D
        if self.dump_path is not None:
            print(
                f"\n[fp8_bhr_hdr_bhd] loading dump: "
                f"{os.path.basename(self.dump_path)}"
            )
            x, y = self._load_xy_from_dump()
        else:
            print("\n[fp8_bhr_hdr_bhd] using random bf16 inputs")
            x = paddle.randn([b, h, r], dtype="bfloat16")
            y = paddle.randn([h, d, r], dtype="bfloat16")

        # ---- Dynamic-range diagnostics (先验视角) ----------------------
        # Activation recipe is 1x128 per-token along R; weight recipe is
        # 128x128 blocks on the (D, R) plane. Report both the global
        # log2 span and the worst per-block span, so we can tell whether
        # the tensor's distribution is already too wide for a single
        # FP8 scale (E4M3 fits ~17 octaves).
        x_top, x_bot, x_dr = _dynamic_range_global(x)
        y_top, y_bot, y_dr = _dynamic_range_global(y)
        x_block_dr = _max_blockwise_dynamic_range_1d(x, block=128, dim=2)
        # y: [H, D, R], block 128x128 on dims (D=1, R=2).
        y_block_dr = _max_blockwise_dynamic_range_2d(y, block=128, dims=(1, 2))
        print(
            f"[fp8_bhr_hdr_bhd] dyn_range(x): global={x_dr:.2f} oct "
            f"(top={x_top:.2f}, bot={x_bot:.2f}), "
            f"max_1x128_block={x_block_dr:.2f} oct "
            f"(E4M3~{_E4M3_OCTAVES:.0f})"
        )
        print(
            f"[fp8_bhr_hdr_bhd] dyn_range(y): global={y_dr:.2f} oct "
            f"(top={y_top:.2f}, bot={y_bot:.2f}), "
            f"max_128x128_block={y_block_dr:.2f} oct "
            f"(E4M3~{_E4M3_OCTAVES:.0f})"
        )

        # bf16 reference (paddle.einsum; established earlier that paddle
        # and deep_gemm bf16 einsums are bit-identical for this pattern).
        ref_z = paddle.einsum("bhr,hdr->bhd", x, y)

        # Activation quant: per_token on (b*h, r) view.
        x_fp8_flat, x_scale_flat = per_token_cast_to_fp8(
            x.reshape([-1, r]), use_ue8m0=use_ue8m0
        )
        x_fp8 = x_fp8_flat.reshape([b, h, r])
        x_scale = x_scale_flat.reshape([b, h, ceil_div(r, 128)])

        # Weight quant: per_block per group.
        # (Paddle currently disallows indexed assignment into a
        # float8_e4m3fn tensor, so stack the per-group results instead of
        # writing into a pre-allocated buffer as the torch version does.)
        y_fp8_list = []
        y_scale_list = []
        for i in range(h):
            f_i, s_i = per_block_cast_to_fp8(
                y[i].contiguous(), use_ue8m0=use_ue8m0
            )
            y_fp8_list.append(f_i)
            y_scale_list.append(s_i)
        y_fp8 = paddle.stack(y_fp8_list, axis=0)  # [h, d, r]
        y_scale = paddle.stack(y_scale_list, axis=0)  # [h, d/128, r/128]

        z = paddle.empty([b, h, d], dtype="bfloat16")
        deep_gemm.fp8_einsum(
            "bhr,hdr->bhd", (x_fp8, x_scale), (y_fp8, y_scale), z
        )

        diff, max_abs = _calc_diff(z, ref_z)
        # Range-relative RMSE (TE ``compare_and_assert`` style): RMSE
        # divided by the joint value span, insensitive to overall scale.
        rmse, rmse_range, rr_rmse = _range_relative_rmse(z, ref_z)
        print(
            f"\n[fp8_bhr_hdr_bhd] shape (b={b}, h={h}, r={r}, d={d}) "
            f"use_ue8m0={use_ue8m0} -> diff={diff:.4e}  max_abs={max_abs:.4e}"
        )
        print(
            f"[fp8_bhr_hdr_bhd] rmse={rmse:.4e}  rmse_range={rmse_range:.4e}  "
            f"rr_rmse={rr_rmse:.4e}"
        )
        # DeepGEMM's own assert on this pattern.
        self.assertLess(
            diff,
            1e-3,
            f"diff={diff:.4e} exceeds DeepGEMM's 1e-3 threshold",
        )
        # TE-style range-relative RMSE tolerance for FP8 GEMM. 2e-3 is
        # the tolerance used by TE's ``compare_and_assert`` when
        # ``is_fp8=True`` on similar shapes.
        self.assertLess(
            rr_rmse,
            2e-3,
            f"rr_rmse={rr_rmse:.4e} exceeds TE's 2e-3 FP8 threshold "
            f"(rmse={rmse:.4e}, range={rmse_range:.4e})",
        )
        return diff, max_abs

    def test_use_ue8m0_true(self):
        """Matches production (``using_pow2_scale=True``)."""
        self._run_one(use_ue8m0=True)

    def test_use_ue8m0_false(self):
        """FP32 scale (no ue8m0 rounding) — for comparison."""
        self._run_one(use_ue8m0=False)


if __name__ == "__main__":
    unittest.main()
