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

"""GPU numerical parity for the Triton packed prefix-LM attention.

Unlike ``test_prefix_lm_triton_core.py`` (which stubs the kernel to check the
host-side permute / guard logic on CPU), this test runs the *real* Triton
forward and backward and compares them against an fp32 dense reference built
from :func:`build_dense_mask`. It covers representative combinations of
multi-segment packs, padding, dtype and block size.

Requires a GPU with Triton; it skips cleanly otherwise.
"""

from __future__ import annotations

import unittest

import numpy as np
import paddle
import paddle.nn.functional as F

from paddlefleet.transformer.prefix_lm_mask import (
    build_dense_mask,
    build_prefix_lm_layout,
)

_GPU = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denom) if denom > 0 else 1.0


def _dense_attention(q, k, v, forbidden, scale):
    """fp32 dense reference. q/k/v: ``[B, H, T, D]``; forbidden ``[1,1,T,T]``."""
    qf, kf, vf = (t.astype("float32") for t in (q, k, v))
    score = paddle.matmul(qf, kf, transpose_y=True) * scale
    neg_inf = paddle.full_like(score, float("-inf"))
    score = paddle.where(forbidden, neg_inf, score)
    prob = F.softmax(score, axis=-1)
    return paddle.matmul(prob, vf)


# (prefix_lens, suffix_lens, pad_len, block_m, block_n, dtype, description)
# block_m == block_n throughout: block_m drives the softmax reduction tree and is
# kept equal to block_n in practice (the production default is 64/64).
_CASES = [
    ([10], [64], 0, 64, 64, "bfloat16", "single segment, no pad"),
    ([10], [64], 54, 64, 64, "bfloat16", "single segment + pad"),
    (
        [16, 24],
        [32, 32],
        24,
        32,
        32,
        "bfloat16",
        "two segments packed + pad, block 32",
    ),
    (
        [5, 5, 5],
        [8, 8, 8],
        25,
        32,
        32,
        "bfloat16",
        "three segments + pad, block 32",
    ),
    ([0, 8], [16, 16], 0, 64, 64, "bfloat16", "first segment prefix=0"),
    ([16, 24], [32, 32], 24, 64, 64, "float16", "two segments, fp16"),
]

_H, _D = 4, 64
_MIN_COS_FWD, _MAX_ABS_FWD = 0.999, 3e-2
_MIN_COS_BWD, _MAX_ABS_BWD = 0.99, 8e-2


@unittest.skipUnless(_GPU, "Triton prefix-LM attention requires a GPU")
class TestPrefixLMAttentionParity(unittest.TestCase):
    def _run_case(self, prefix, suffix, pad, block_m, block_n, dtype):
        from paddlefleet.triton_ops.prefix_lm_attention import (
            layout_from_hyperbody_dict,
            triton_prefix_lm_attention,
        )

        total = sum(int(c) + int(q) for c, q in zip(prefix, suffix))
        tp = total + pad
        scale = 1.0 / np.sqrt(_D)
        layout = layout_from_hyperbody_dict(
            build_prefix_lm_layout(prefix, suffix, pad), tp
        )
        forbidden = build_dense_mask(prefix, suffix, pad)

        paddle.seed(20260911)
        base = [paddle.randn([1, _H, tp, _D]).astype(dtype) for _ in range(3)]
        gout = paddle.randn([1, _H, tp, _D]).astype(dtype)

        # --- reference (fp32 dense) ---
        rq, rk, rv = (t.detach() for t in base)
        for t in (rq, rk, rv):
            t.stop_gradient = False
        ref_out = _dense_attention(rq, rk, rv, forbidden, scale)
        (ref_out * gout.astype("float32")).sum().backward()

        # --- triton kernel ---
        tq, tk, tv = (t.detach() for t in base)
        for t in (tq, tk, tv):
            t.stop_gradient = False
        got_out = triton_prefix_lm_attention(
            tq, tk, tv, layout, scale=scale, block_m=block_m, block_n=block_n
        )
        (got_out * gout).sum().backward()

        # Real-token region only: pad rows/cols are sliced off downstream.
        def _slice(t):
            return t.astype("float32").numpy()[:, :, :total]

        checks = {
            "out": (
                _slice(ref_out),
                _slice(got_out),
                _MIN_COS_FWD,
                _MAX_ABS_FWD,
            ),
            "dq": (
                _slice(rq.grad),
                _slice(tq.grad),
                _MIN_COS_BWD,
                _MAX_ABS_BWD,
            ),
            "dk": (
                _slice(rk.grad),
                _slice(tk.grad),
                _MIN_COS_BWD,
                _MAX_ABS_BWD,
            ),
            "dv": (
                _slice(rv.grad),
                _slice(tv.grad),
                _MIN_COS_BWD,
                _MAX_ABS_BWD,
            ),
        }
        for name, (r, g, min_cos, max_abs) in checks.items():
            cos, mabs = _cos(r, g), float(np.abs(r - g).max())
            self.assertGreaterEqual(
                cos, min_cos, f"{name}: cos={cos:.6f} (max_abs={mabs:.4f})"
            )
            self.assertLessEqual(
                mabs, max_abs, f"{name}: max_abs={mabs:.4f} (cos={cos:.6f})"
            )

    def test_parity_cases(self):
        for prefix, suffix, pad, bm, bn, dtype, desc in _CASES:
            with self.subTest(desc=desc):
                self._run_case(prefix, suffix, pad, bm, bn, dtype)


if __name__ == "__main__":
    unittest.main()
