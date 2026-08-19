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

"""``dsa_indexer_rope_fusion`` on/off on a real ``DSAIndexer``.

``test_rope_half_fusion.py`` proves the kernel itself; this proves the
plumbing in ``DSAIndexer._apply_rope`` -- that q and k get the right freqs and
the right ``rope_head_dim`` -- by toggling the flag on a real indexer.

``forward_before_topk`` ends before the top-k selection and before any attention
kernel, so nothing here accumulates atomically: the comparison is exact in both
directions. That is what makes it a proof rather than a "within noise"
observation, which is all a whole-layer comparison can give.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

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
class TestDSAIndexerRopeFusion(unittest.TestCase):
    def setUp(self) -> None:
        paddle.set_device("gpu")

    def _build(self, fusion, seed=11):
        cfg = U._create_mqa_config(mode="mqa_dsa")
        cfg._build_dsa_indexer = True
        cfg.dsa_indexer_rope_fusion = fusion
        paddle.seed(seed)
        return U._build_module(cfg, bf16=True)

    def _run(self, module, x0, qr0, gq, gk):
        x = x0.clone()
        qr = qr0.clone()
        x.stop_gradient = False
        qr.stop_gradient = False
        q, k, w = module.indexer.forward_before_topk(x, qr)
        loss = (q.astype("float32") * gq).sum() + (
            k.astype("float32") * gk
        ).sum()
        loss.backward()
        return (
            q.detach().clone(),
            k.detach().clone(),
            w.detach().clone(),
            x.grad.detach().clone(),
            qr.grad.detach().clone(),
            {
                n: (None if p.grad is None else p.grad.detach().clone())
                for n, p in module.indexer.named_parameters()
            },
        )

    def test_fusion_is_bitwise_forward_and_backward(self):
        ref = self._build(False)
        fused = self._build(True)
        fused.set_state_dict(ref.state_dict())

        paddle.seed(1000)
        h = ref.indexer.config.hidden_size
        qlr = ref.indexer.wq_b.linear.weight.shape[0]
        x0 = paddle.randn([1, SEQ, h], dtype="float32").astype("bfloat16")
        qr0 = paddle.randn([1, SEQ, qlr], dtype="float32").astype("bfloat16")

        with paddle.no_grad():
            pq, pk, _ = ref.indexer.forward_before_topk(x0.clone(), qr0.clone())
        gq = paddle.randn(pq.shape, dtype="float32")
        gk = paddle.randn(pk.shape, dtype="float32")

        off = self._run(ref, x0, qr0, gq, gk)
        on = self._run(fused, x0, qr0, gq, gk)

        for name, a, b in zip(
            ("q", "k", "weights", "dx", "dqr"), on[:5], off[:5]
        ):
            with self.subTest(tensor=name):
                self.assertTrue(
                    bool(paddle.all(a == b).item()),
                    f"{name} is not bitwise equal, rel={_rel(a, b):.3e}",
                )
        checked = 0
        for n in sorted(off[5]):
            if off[5][n] is None:
                continue
            with self.subTest(param=n):
                self.assertIsNotNone(on[5][n], f"{n}: grad went missing")
                self.assertTrue(
                    bool(paddle.all(on[5][n] == off[5][n]).item()),
                    f"grad {n} is not bitwise equal, "
                    f"rel={_rel(on[5][n], off[5][n]):.3e}",
                )
            checked += 1
        self.assertGreater(checked, 0, "no parameter gradients were compared")
        print(
            f"[dsa] indexer fusion on/off: q/k/weights/dx/dqr + {checked} "
            f"param grads all bitwise equal",
            flush=True,
        )

    def test_fusion_is_actually_engaged(self):
        """Without this the test above could be comparing eager with eager."""
        from unittest import mock

        import paddlefleet.triton_ops as tri

        real = tri.fused_apply_rope_half
        calls = []

        def counting(*a, **kw):
            calls.append(tuple(a[0].shape))
            return real(*a, **kw)

        fused = self._build(True)
        paddle.seed(1000)
        h = fused.indexer.config.hidden_size
        qlr = fused.indexer.wq_b.linear.weight.shape[0]
        x = paddle.randn([1, SEQ, h], dtype="float32").astype("bfloat16")
        qr = paddle.randn([1, SEQ, qlr], dtype="float32").astype("bfloat16")
        with (
            mock.patch.object(tri, "fused_apply_rope_half", counting),
            paddle.no_grad(),
        ):
            fused.indexer.forward_before_topk(x, qr)
        self.assertEqual(
            len(calls), 2, f"expected one call for q and one for k, got {calls}"
        )
        print(f"[dsa] fused_apply_rope_half shapes={calls}", flush=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
