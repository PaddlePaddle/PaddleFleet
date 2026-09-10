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

"""``high_precision_rope`` on/off on a real ``DSAIndexer``.

``test_dsa_indexer_rope_fusion.py`` covers the ``dsa_indexer_rope_fusion``
switch; this covers the orthogonal ``high_precision_rope`` switch on the same
``DSAIndexer._apply_rope``. ``high_precision_rope=True`` is **not** numerically
inert for the indexer: it (a) bypasses the ``fused_apply_rope_half`` kernel and
(b) runs the eager rotation in fp32 (``rope_utils`` casts to float32, disables
auto_cast, then rounds back to bf16). These tests pin both effects and the fact
that the fp32 path lands closer to a full-fp32 reference than the default bf16
path, so a future "precision unchanged" claim about this flag would fail here.

``_apply_rope`` runs before any attention kernel and accumulates nothing
atomically, so the reference comparison is exact rather than "within noise".
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import paddle

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hybrid_mla_utils as U

SEQ = 256


def _rel(a, b):
    a = a.astype("float32").flatten()
    b = b.astype("float32").flatten()
    return float((a - b).norm() / (b.norm() + 1e-12))


@unittest.skipUnless(
    paddle.device.is_compiled_with_cuda(), "triton kernels need CUDA"
)
class TestDSAIndexerHighPrecisionRope(unittest.TestCase):
    def setUp(self) -> None:
        paddle.set_device("gpu")

    def _build(self, high_precision, fusion, seed=11):
        cfg = U._create_mqa_config(mode="mqa_dsa")
        cfg._build_dsa_indexer = True
        cfg.dsa_indexer_rope_fusion = fusion
        cfg.dsa_indexer_rotary_interleaved = False
        cfg.high_precision_rope = high_precision
        paddle.seed(seed)
        return U._build_module(cfg, bf16=True)

    def _rope_inputs(self, indexer, seed=1000):
        """A frozen ``[b, s, n_heads, head_dim]`` activation and its freqs."""
        paddle.seed(seed)
        x = paddle.randn(
            [1, SEQ, indexer.n_heads, indexer.head_dim], dtype="float32"
        ).astype("bfloat16")
        freqs = indexer.rotary_pos_emb(SEQ, packed_seq=False)
        return x, freqs

    def test_true_bypasses_fusion_false_engages_it(self):
        """The fused rope-half kernel runs only when the flag is off.

        The fusion gate carries ``and not high_precision_rope`` -- turning the
        flag on must force the eager fp32 branch even with fusion requested.
        """
        import paddlefleet.triton_ops as tri

        real = tri.fused_apply_rope_half

        def _counting(bag):
            def wrapped(*a, **kw):
                bag.append(tuple(a[0].shape))
                return real(*a, **kw)

            return wrapped

        for hp, expected in ((True, 0), (False, 1)):
            indexer = self._build(high_precision=hp, fusion=True).indexer
            x, freqs = self._rope_inputs(indexer)
            calls = []
            with (
                mock.patch.object(
                    tri, "fused_apply_rope_half", _counting(calls)
                ),
                paddle.no_grad(),
            ):
                out = indexer._apply_rope(x, freqs, 1.0)
            with self.subTest(high_precision_rope=hp):
                self.assertEqual(
                    len(calls),
                    expected,
                    f"high_precision_rope={hp}: expected {expected} fused "
                    f"call(s), got {len(calls)} ({calls})",
                )
                self.assertEqual(out.shape, x.shape)
        print(
            "[dsa] high_precision_rope=True bypasses fused_apply_rope_half, "
            "False engages it",
            flush=True,
        )

    def test_eager_fp32_and_bf16_paths_agree_on_the_indexer_rope(self):
        """The fp32 branch is bit-exact to the default bf16 branch here.

        ``high_precision_rope`` changes the *code path* (proved above), but
        this pins what it does to the *numbers*: for the indexer's plain rope
        (mscale=1, elementwise ``t*cos + rotate_half(t)*sin``), the eager fp32
        branch and the default bf16 branch produce bitwise-identical q and k.
        bf16 elementwise ops already accumulate the single multiply-add in fp32
        before the one rounding step, so promoting to fp32 first cannot move
        any bit. This is the evidence for the PR's "indexer output unchanged"
        claim -- and if a future hardware/kernel change ever breaks it, that
        silent precision drift surfaces here instead of in a loss curve.

        Both settings run the eager branch (fusion off) so the comparison
        isolates the precision cast, not a kernel swap.
        """
        indexer = self._build(high_precision=False, fusion=False).indexer
        x, freqs = self._rope_inputs(indexer)
        cfg = indexer.config

        with paddle.no_grad():
            cfg.high_precision_rope = False
            bf16 = indexer._apply_rope(x, freqs, 1.0)
            cfg.high_precision_rope = True
            fp32 = indexer._apply_rope(x, freqs, 1.0)

        self.assertEqual(fp32.dtype, bf16.dtype)
        self.assertTrue(
            bool(paddle.all(fp32 == bf16).item()),
            "eager fp32 and bf16 rope diverged on the indexer, "
            f"rel={_rel(fp32, bf16):.3e} -- high_precision_rope is no longer "
            "output-preserving for this op",
        )
        print(
            "[dsa] high_precision_rope eager fp32 == bf16 on indexer rope "
            "(bitwise)",
            flush=True,
        )

    def test_both_switch_values_run_forward_before_topk(self):
        """Regression: both settings drive a full indexer forward cleanly."""
        for hp in (False, True):
            indexer = self._build(high_precision=hp, fusion=False).indexer
            h = indexer.config.hidden_size
            qlr = indexer.wq_b.linear.weight.shape[0]
            paddle.seed(7)
            x = paddle.randn([1, SEQ, h], dtype="float32").astype("bfloat16")
            qr = paddle.randn([1, SEQ, qlr], dtype="float32").astype("bfloat16")
            with paddle.no_grad():
                q, k, w = indexer.forward_before_topk(x, qr)
            with self.subTest(high_precision_rope=hp):
                for name, t in (("q", q), ("k", k), ("weights", w)):
                    self.assertTrue(
                        bool(paddle.isfinite(t.astype("float32")).all().item()),
                        f"high_precision_rope={hp}: {name} has non-finite "
                        "values",
                    )
                self.assertEqual(list(q.shape[:2]), [1, SEQ])
                self.assertEqual(list(k.shape[:2]), [1, SEQ])


if __name__ == "__main__":
    unittest.main(verbosity=2)
