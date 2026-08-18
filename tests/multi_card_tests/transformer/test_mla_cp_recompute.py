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

"""Task #18 -- CP x recompute equivalence for the hybrid-MLA (MHA/MQA/DSA) layer.

Proves that full-layer activation recompute is a transparent compute/memory
trade UNDER context parallel: for the CP layer, recompute ON must equal
recompute OFF elementwise on the forward output and on every parameter / input
gradient, on each CP rank. The load-bearing invariant is that the DSA
``token_indices`` -- selected from the CP-*global* index table (built at
``s_global = s * cp_size`` and sliced at ``position_offset = cp_rank * s``,
mqa_latent_attention.py:297-298) -- are re-derived bit-identically on the
recompute re-forward; a mismatch would silently differentiate a *different*
sparsity pattern and corrupt gradients with no error.

This is the CP analogue of the single-card
``test_hybrid_mla_recompute_mtp_ckpt.py::TestRecomputeEquivalence`` and closes
the recompute leg of the MTP x CP audit: an MTP layer's inner attention IS this
same TransformerLayer, recomputed via ``multi_token_prediction.py`` uniform
checkpointing (:747-753), and it inherits the process-global CP group through
``use_mpu_process_groups()`` (transformer_layer.py:233-234).

Both branches (OFF then ON) run the identical collective sequence on every
rank -- there is NO ``if rank == X`` short-circuit -- so the CP NCCL streams
stay in lockstep across the recompute re-forward's extra ``all_gather_cp``.

Run (2 GPUs)::

    PYTHONPATH=./third_party/PaddleFleet/src:./third_party/PaddleFormers \
    python -m paddle.distributed.launch --devices <a>,<b> --nnodes 1 \
        --master 127.0.0.1:<port> \
        third_party/PaddleFleet/tests/multi_card_tests/transformer/\
test_mla_cp_recompute.py
"""

import unittest

import paddle

# Reuse the proven CP harness (fleet init, cfg/layer builders, CP globals).
# Run via ``paddle.distributed.launch <thisfile>`` puts this dir on sys.path[0],
# so the sibling test module imports as a top-level name.
import test_mla_cp_contiguous_allgather as H
from paddle.distributed.fleet.utils import recompute

# Same-rank ON vs OFF is the identical function of identical inputs, so the
# forward is bit-exact up to kernel nondeterminism; grads take one extra bf16
# re-forward. Tolerances mirror the single-card recompute suite.
FWD_RTOL = 1e-5
GRAD_RTOL = 5e-3

# ``"mqa"`` (``mqa_full_causal``) accepts dense FA4 only --
# ``MQALatentAttention._assert_dense_fa4`` raises otherwise -- and a bare
# launcher process leaves ``FLAGS_flash_attn_version`` at the image default 2.
# Pin the value ``TrainingArguments.__post_init__`` derives on these SM100 boxes,
# which is also production's for the ``mha`` and ``mqa_dsa`` cases. ``_fa4_pin``
# stands down where no FA4 backend exists, because ``test_mha_recompute_cp`` is
# not ``_GPU``-gated and has to keep running on the cc 9.0 CI images. Module
# scope: nothing here asserts on the refusal
# (``test_mla_cp_contiguous_allgather``'s ``test_4`` does, which is why that
# file scopes its pin per call).
_FA4_PIN = H.U._fa4_pin()


def setUpModule():
    H.setUpModule()
    _FA4_PIN.__enter__()


def tearDownModule():
    _FA4_PIN.__exit__(None, None, None)


def _run_recompute_pair(mask_full, attn_mode, sink=False, seed=2026):
    """Build ONE CP layer, then run fwd+bwd twice on the same rank-local shard:
    recompute OFF, then recompute ON. Returns ``(out_off, out_on, g_off, g_on)``
    with grads as ``{name: fp32 tensor or None}`` (``__hidden__`` = dH)."""
    H._cp_enable()
    b, sg = 1, mask_full.shape[2]
    sl = sg // H.CP_SIZE

    paddle.seed(seed)
    cpl = H.build_mla(
        H.build_cfg(H.CP_SIZE, sink=sink, attn_mode=attn_mode),
        H.CP_GROUP,
        attn_mode,
    )

    paddle.seed(1000)
    hf = paddle.randn([b, sg, 256], dtype=H.DTYPE)
    s, e = H.CP_RANK * sl, (H.CP_RANK + 1) * sl

    def _one(use_rc):
        cpl.clear_gradients()
        h = hf[:, s:e].clone()
        h.stop_gradient = False

        def fn(x):
            out, _ = cpl(
                x, None, attn_mask_startend_row_indices=mask_full.clone()
            )
            return out

        out = recompute(fn, h) if use_rc else fn(h)
        out.cast("float32").sum().backward()
        grads = {
            n: (None if p.grad is None else p.grad.detach().cast("float32"))
            for n, p in cpl.named_parameters()
        }
        grads["__hidden__"] = (
            None if h.grad is None else h.grad.detach().cast("float32")
        )
        return out.detach().cast("float32"), grads

    out_off, g_off = _one(False)
    out_on, g_on = _one(True)
    return out_off, out_on, g_off, g_on


class TestMLACPRecompute(unittest.TestCase):
    """recompute ON == OFF, per CP rank, on output and every gradient."""

    def _check(self, attn_mode, mask, sink=False):
        out_off, out_on, g_off, g_on = _run_recompute_pair(
            mask, attn_mode, sink=sink
        )
        self.assertLess(
            H._rel(out_on, out_off),
            FWD_RTOL,
            f"{attn_mode} sink={sink}: recompute output drifted",
        )
        # recompute must not change which params receive a gradient.
        self.assertEqual(set(g_on), set(g_off), f"{attn_mode}: grad key set")
        reforwarded = 0
        for n in g_off:
            if g_off[n] is None and g_on[n] is None:
                continue
            self.assertIsNotNone(g_on[n], f"{attn_mode}: ON missing grad {n}")
            self.assertIsNotNone(g_off[n], f"{attn_mode}: OFF missing grad {n}")
            rel = H._rel(g_on[n], g_off[n])
            self.assertLess(
                rel, GRAD_RTOL, f"{attn_mode} sink={sink}: grad[{n}] rel={rel}"
            )
            reforwarded += 1
        self.assertGreater(reforwarded, 0, f"{attn_mode}: no grads compared")
        print(
            f"[cp-recompute cp{H.CP_SIZE} rank{H.CP_RANK}] {attn_mode} "
            f"sink={sink}: fwd={H._rel(out_on, out_off):.2e} "
            f"grads_ok={reforwarded}",
            flush=True,
        )

    def test_mha_recompute_cp(self):
        self._check("mha", H._row_end([128], 128))

    @H.U._GPU
    def test_mqa_recompute_cp(self):
        # documents straddle the CP=2/4 rank boundaries so the CP-global index
        # table (sliced per rank) is genuinely exercised on the re-forward.
        self._check("mqa", H._row_end([200, 150, 162], 512))

    @H.U._GPU
    def test_mqa_dsa_recompute_cp(self):
        self._check("mqa_dsa", H._row_end([200, 150, 162], 512))


if __name__ == "__main__":
    unittest.main(verbosity=2)
