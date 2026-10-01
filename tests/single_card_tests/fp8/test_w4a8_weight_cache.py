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
"""Tests for the W4A8 FP4 stacked-weight quant cache (release/0.4 only feature).

Why this file exists: the cache is refreshed once per optimizer step by
``moe_layer.fp8_quant_weight`` while the resident BF16 expert weight is still
alive, and consumed by the four fwd/bwd W4A8 GEMM sites via
``fp8_utils._w4a8_stack_quant``. Two properties must hold, and NEITHER is
covered by a single-step test -- which is exactly how a refresh bug can hide:

1. ``test_refresh_requantizes_every_step`` -- the per-step refresh must actually
   re-quantize. If the refresh path reads the cache it is about to overwrite,
   the assignment becomes a no-op and every W4A8 GEMM keeps using step-1 expert
   weights forever while the optimizer goes on updating the master weights.
   Needs >= 2 steps to observe. No GPU required.

2. ``test_cached_value_is_bit_identical`` -- the cache's stated contract is that
   the cached FP4 tensor is bit-identical to quantizing inside the forward.
   Requires a free GPU (SM100 + the fused 1x32 ops).

Run:
    PYTHONPATH=src python -m pytest tests/single_card_tests/fp8/test_w4a8_weight_cache.py -v
"""

import unittest
from unittest import mock

import paddle

from paddlefleet.transformer.moe import fp8_utils
from paddlefleet.transformer.moe.fp8_utils import _w4a8_stack_quant


class _FakeExpertWeight:
    """Stands in for ``grouped_gemm_experts.weight1`` -- a plain object the cache
    attributes get attached to. ``value`` models the BF16 storage the optimizer
    updates between steps."""

    def __init__(self, value):
        self.value = value


def _refresh(weight_list, weight_obj=None, use_w4a8_fused_quant=True):
    """Verbatim shape of the per-step refresh in moe_layer.fp8_quant_weight.

    Mirrors production exactly: the weight LIST is quantized (``_stack_expert_weights``
    collapses it), and ``use_cache=False`` bypasses the cache read so the refresh
    really re-quantizes instead of handing back the attribute it is about to write.
    """
    if weight_obj is None:
        weight_obj = weight_list[0]
    weight_obj.w4a8_fp4_stacked_transpose = _w4a8_stack_quant(
        weight_list,
        transpose=True,
        use_w4a8_fused_quant=use_w4a8_fused_quant,
        use_cache=False,
    )
    weight_obj.w4a8_fp4_stacked = _w4a8_stack_quant(
        weight_list,
        transpose=False,
        use_w4a8_fused_quant=use_w4a8_fused_quant,
        use_cache=False,
    )


class TestW4A8WeightCacheRefresh(unittest.TestCase):
    """CPU-only: exercises the real cache-read logic with the quantizer stubbed."""

    def setUp(self):
        self.calls = []

        def fake_quant(weights, transpose):
            # One entry per real quantization, tagged with the weight value it saw.
            self.calls.append((weights.value, transpose))
            return f"fp4({weights.value},T={transpose})"

        self.patches = [
            mock.patch.object(fp8_utils, "_use_w4a8_fused_quant", lambda *_a, **_k: True),
            # mirrors the real _stack_expert_weights list-collapsing behaviour
            mock.patch.object(
                fp8_utils,
                "_stack_expert_weights",
                lambda w: w[0] if isinstance(w, (list, tuple)) else w,
            ),
            mock.patch.object(fp8_utils, "w4a8_stack_quantize_1x32", fake_quant),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_cache_hit_avoids_requant_within_a_step(self):
        """Within one step the four fwd/bwd sites must hit the cache, not re-quantize."""
        w = _FakeExpertWeight("step1")
        _refresh([w])
        self.assertEqual(len(self.calls), 2, "refresh should quantize both orientations")
        for _ in range(4):  # the four fwd/bwd consumer sites
            _w4a8_stack_quant(w, transpose=True, use_w4a8_fused_quant=True)
            _w4a8_stack_quant(w, transpose=False, use_w4a8_fused_quant=True)
        self.assertEqual(len(self.calls), 2, "consumers must reuse the cache")

    def test_refresh_requantizes_every_step(self):
        """The per-step refresh must re-quantize from the updated BF16 weight.

        Regression test: if the refresh ever reads the cache it is about to
        overwrite, the assignment degenerates into a no-op and every W4A8 GEMM
        keeps using step-1 expert weights for the whole run. Needs >= 2 steps.
        """
        w = _FakeExpertWeight("step1")
        steps = ["step1", "step2", "step3"]
        for value in steps:
            w.value = value  # optimizer updated the master/BF16 weight
            _refresh([w])
            self.assertEqual(
                w.w4a8_fp4_stacked_transpose,
                f"fp4({value},T=True)",
                f"cache still holds a stale tensor at {value}: {w.w4a8_fp4_stacked_transpose}",
            )
            self.assertEqual(w.w4a8_fp4_stacked, f"fp4({value},T=False)")

        self.assertEqual(
            len(self.calls),
            2 * len(steps),
            f"expected 2 quantizations per step, got {self.calls}",
        )

    def test_refreshing_through_the_cache_would_be_a_noop(self):
        """Why ``use_cache=False`` is mandatory at the refresh site.

        Pins the trap: refreshing WITH the cache enabled silently stops
        re-quantizing from step 2 on. Keep this red-flag behaviour documented so
        nobody "simplifies" the refresh call back to the cached form.
        """
        w = _FakeExpertWeight("step1")
        for value in ("step1", "step2", "step3"):
            w.value = value
            # deliberately the WRONG form: cache enabled at a refresh point
            w.w4a8_fp4_stacked_transpose = _w4a8_stack_quant(
                w, transpose=True, use_w4a8_fused_quant=True
            )
        self.assertEqual(
            w.w4a8_fp4_stacked_transpose,
            "fp4(step1,T=True)",
            "cached refresh unexpectedly tracked the weight update",
        )
        self.assertEqual(len(self.calls), 1, f"expected a single quant, got {self.calls}")


@unittest.skipUnless(paddle.device.cuda.device_count() > 0, "needs a free GPU")
class TestW4A8WeightCacheBitParity(unittest.TestCase):
    """GPU: the cached FP4 must be bit-identical to quantizing in the forward."""

    def test_cached_value_is_bit_identical(self):
        E, K, N = 4, 512, 512
        paddle.set_device("gpu:0")
        w = _FakeExpertWeight(None)
        w.value = paddle.randn([E, K, N], dtype=paddle.bfloat16)

        for transpose in (True, False):
            attr = "w4a8_fp4_stacked_transpose" if transpose else "w4a8_fp4_stacked"
            # fresh (no cache attribute present yet)
            fresh = _w4a8_stack_quant(w.value, transpose=transpose, use_w4a8_fused_quant=True)
            setattr(w, attr, fresh)
            cached = _w4a8_stack_quant(w, transpose=transpose, use_w4a8_fused_quant=True)
            for a, b in zip(
                fresh if isinstance(fresh, (tuple, list)) else (fresh,),
                cached if isinstance(cached, (tuple, list)) else (cached,),
            ):
                self.assertTrue(
                    bool((a.astype("float32") == b.astype("float32")).all()),
                    f"cached FP4 differs from freshly quantized (transpose={transpose})",
                )


if __name__ == "__main__":
    unittest.main()
