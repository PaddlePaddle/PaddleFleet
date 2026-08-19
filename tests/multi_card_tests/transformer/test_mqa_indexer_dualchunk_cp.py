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

"""``mqa_indexer_cp_mode="dualchunk_p2p"`` must not change what the layer computes.

The switch rebalances *which rows* each rank's indexer scores: rank ``r`` takes
global chunks ``(2r, 2*cp_size-1-2r)`` of ``2*cp_size`` instead of its own
``(2r, 2r+1)``, so the causal work is equal across ranks instead of growing 31x
from cp0 to cp15. The permutation is undone before the layer returns, so the
output must be unchanged.

Two things make that testable as an equality rather than a tolerance:

* the per-``(row, column)`` indexer score is bit-identical between the layouts.
  A CTA covers ``q_stage * q_tokens_per_tile`` rows and both offsets are
  multiples of that, so the same global row lands in the same aligned tile and
  the kernel visits the same key blocks in the same order;
* the swap itself is pure data movement, and it is an involution, so the
  round trip restores the contiguous order exactly.

What is *not* guaranteed is the top-k tie-break: the two calls see different
``max_k`` on the THD path, so tied scores could in principle resolve
differently. Ties are vanishingly rare (a sum of ``index_n_heads`` fp32
products), which is why this asserts equality and would flag a systematic
divergence rather than paper over it with a tolerance.
"""

import types
import unittest

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from paddlefleet.transformer.cp_utils import (
    dualchunk_chunk_ids,
    dualchunk_partner,
    dualchunk_swap,
)
from tests.single_card_tests.transformer import hybrid_mla_utils as U

CP_SIZE = None
CP_RANK = None
CP_GROUP = None

S_GLOBAL = 512

_FA4_PIN = U._fa4_pin()


def setUpModule():
    global CP_SIZE, CP_RANK, CP_GROUP
    if dist.get_world_size() < 2:
        raise unittest.SkipTest(
            "MQA dual-chunk tests require >= 2 GPUs (one peer to swap with)"
        )
    _FA4_PIN.__enter__()
    world = dist.get_world_size()
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": world,
        "sep_degree": 1,
        "cp_degree": world,
        "ep_degree": world,
        "moe_sharding_degree": 1,
        "order": [
            "sharding",
            "moe_sharding",
            "pp",
            "sep",
            "cp",
            "dp",
            "ep",
            "mp",
        ],
    }
    fleet.init(is_collective=True, strategy=strategy)
    CP_GROUP = fleet.get_hybrid_communicate_group().get_context_parallel_group()
    CP_RANK = CP_GROUP.rank
    CP_SIZE = CP_GROUP.nranks


def tearDownModule():
    _FA4_PIN.__exit__(None, None, None)


def _build(dualchunk, loss_coeff=0.0, seed=7):
    """CP layer with the rebalance off/on; identical weights either way."""
    cfg = U._create_mqa_config(mode="mqa_dsa", loss_coeff=loss_coeff)
    cfg.dsa_indexer_use_sparse_loss = True
    cfg.cp_balance_mode = "contiguous_allgather"
    cfg.mqa_indexer_cp_mode = "dualchunk_p2p" if dualchunk else None
    cfg.experimental_dataflow = True
    cfg.pad_token_id = 0
    cfg.context_parallel_size = CP_SIZE
    paddle.seed(seed)
    return U._build_module(
        cfg,
        bf16=True,
        pg_collection=types.SimpleNamespace(tp=None, cp=CP_GROUP),
    )


def _inputs(s_global, seed=1234):
    """Global-length inputs, bitwise identical on every rank."""
    rng = np.random.RandomState(seed)

    def t(shape, scale):
        return paddle.to_tensor(
            (rng.standard_normal(shape) * scale).astype("float32")
        ).cast("bfloat16")

    return {
        "query": t([1, s_global, U.H, U.DK], 0.5),
        "key": t([1, s_global, 1, U.DK], 0.5),
        "w_v": t([U.DV, U.H, U.V_HEAD_DIM], 0.05),
        "x": t([1, s_global, U.HIDDEN], 0.5),
        "qr": t([1, s_global, U.Q_LORA], 0.5),
    }


def _leaf(t):
    out = t.clone().detach()
    out.stop_gradient = False
    return out


def _max_abs(a, b):
    """``max |a - b|`` in fp32, so bf16 tensors compare without overflow."""
    return float((a.astype("float32") - b.astype("float32")).abs().max())


def _rel(a, b):
    """``max |a - b| / max |a|``, the shape the dkv tolerance is quoted in."""
    scale = float(a.astype("float32").abs().max())
    return _max_abs(a, b) / max(scale, 1e-12)


# The shared cuDNN DSA backward quotes rel ~2e-3 for ``dkv`` between identical
# calls; 5e-3 leaves headroom for the max-abs tail without hiding a real drift.
_GRAD_RTOL = 5e-3


def _run(dualchunk, inp, row_end, loss_coeff):
    """One forward+backward of this rank's slice.

    Returns ``(output, input_grads, param_grads)``. The parameter grads are the
    only place the indexer loss shows up: ``_indexer_projections`` detaches
    ``x``/``qr`` before the indexer touches them, so the KL gradient reaches the
    indexer's own weights (``wq_b``, ``wk``, ``weights_proj``) and never the
    caller's leaves. Asserting on the output alone would pass even if the
    permutation corrupted ``topk_probs``, because the loss scaler forwards
    ``output`` unchanged and only injects the gradient in backward.
    """
    sl = S_GLOBAL // CP_SIZE
    off = CP_RANK * sl
    layer = _build(dualchunk, loss_coeff)
    if dualchunk:
        # Same weights on both sides: rebuilding from the reference state dict is
        # unnecessary because ``paddle.seed`` is pinned in ``_build``, but assert
        # the switch actually took so a silent no-op cannot pass this test.
        assert layer.indexer_dualchunk, "dualchunk_p2p did not take effect"
    else:
        assert not layer.indexer_dualchunk

    r = {
        "query": _leaf(inp["query"][:, off : off + sl]),
        "key": _leaf(inp["key"][:, off : off + sl]),
        "w_v": _leaf(inp["w_v"]),
        "x": _leaf(inp["x"][:, off : off + sl]),
        "qr": _leaf(inp["qr"][:, off : off + sl]),
    }
    out = layer(
        r["query"],
        r["key"],
        None,
        None,
        attn_mask_startend_row_indices=row_end.clone(),
        x=r["x"],
        qr=r["qr"],
        v_b_proj_weight=r["w_v"],
        input_ids=None,
    )
    out.sum().backward()
    return (
        out,
        {k: v.grad for k, v in r.items() if v.grad is not None},
        {n: p.grad for n, p in layer.named_parameters() if p.grad is not None},
    )


class TestMqaIndexerDualChunkCp(unittest.TestCase):
    def test_chunk_ids_balance_and_cover(self):
        """Ids sum to a constant and every chunk has exactly one owner."""
        owner = {}
        for r in range(CP_SIZE):
            lo, hi = dualchunk_chunk_ids(r, CP_SIZE)
            self.assertEqual(lo + hi, 2 * CP_SIZE - 1)
            # The chunk this rank keeps is the one contiguous CP already gave
            # it, which is what makes the exchange a single sendrecv.
            self.assertEqual(lo // 2, r)
            self.assertEqual(hi // 2, dualchunk_partner(r, CP_SIZE))
            for c in (lo, hi):
                self.assertNotIn(c, owner)
                owner[c] = r
        self.assertEqual(sorted(owner), list(range(2 * CP_SIZE)))

    def test_swap_is_an_involution(self):
        """Applying the swap twice restores the contiguous rows exactly."""
        sl = S_GLOBAL // CP_SIZE
        base = CP_RANK * sl
        rows = paddle.arange(base, base + sl, dtype="float32")
        x = rows.reshape([1, sl, 1]).expand([1, sl, 4]).clone()
        once = dualchunk_swap(x, CP_GROUP, axis=1)
        twice = dualchunk_swap(once, CP_GROUP, axis=1)
        self.assertEqual(float((twice - x).abs().max()), 0.0)
        # And the single swap really did move the expected chunks.
        m = sl // 2
        lo, hi = dualchunk_chunk_ids(CP_RANK, CP_SIZE)
        want = paddle.concat(
            [
                paddle.arange(lo * m, (lo + 1) * m, dtype="float32"),
                paddle.arange(hi * m, (hi + 1) * m, dtype="float32"),
            ]
        )
        self.assertEqual(float((once[0, :, 0] - want).abs().max()), 0.0)

    def test_odd_extent_rejected(self):
        """Two chunks per rank needs an even local row count."""
        odd = paddle.zeros([1, 2 * (S_GLOBAL // CP_SIZE) + 1, 2])
        with self.assertRaisesRegex(ValueError, "even extent"):
            dualchunk_swap(odd, CP_GROUP, axis=1)

    @U._GPU
    def test_output_unchanged(self):
        """The rebalance is a pure scheduling change: identical output.

        The output being *exactly* equal is the whole contract, and it is the
        strong assertion here: it can only hold if the two layouts selected the
        same ``token_indices``, in the same order, from the same scores.

        Gradients then flow from identical inputs through identical code, so they
        are only checked against the tolerance the shared cuDNN DSA backward
        already carries: it accumulates ``dkv`` with atomics and is not
        reproducible even between two identical ``off`` runs (documented as
        rel ~2e-3 in ``fusions/mqa_sparse_attn.py``). Both the off-vs-off and the
        off-vs-dualchunk pair are held to the same bound, which is what shows the
        tolerance belongs to the kernel and not to this switch.
        """
        inp = _inputs(S_GLOBAL)
        row_end = U._row_end([S_GLOBAL], S_GLOBAL)
        ref, gref, pref = _run(False, inp, row_end, loss_coeff=0.0)
        ref2, gref2, pref2 = _run(False, inp, row_end, loss_coeff=0.0)
        got, ggot, pgot = _run(True, inp, row_end, loss_coeff=0.0)

        self.assertEqual(ref.shape, got.shape)
        self.assertEqual(
            _max_abs(ref, got), 0.0, "dual-chunk changed the layer output"
        )
        self._assert_grads_match(
            (gref, gref2, ggot), (pref, pref2, pgot), "no-loss"
        )

    @U._GPU
    def test_indexer_loss_gradients_unchanged(self):
        """With the KL live, the indexer's own weight grads must also match.

        This is the case the output alone cannot cover: the loss scaler forwards
        ``output`` unchanged and only injects the gradient in backward, so a
        permutation that corrupted ``topk_probs`` or ``topk_indices`` would show
        up nowhere else. ``_indexer_projections`` detaches ``x``/``qr``, so the KL
        gradient lands on ``indexer.*`` parameters -- which is what gets compared
        here, against the same off-vs-off baseline.
        """
        inp = _inputs(S_GLOBAL, seed=99)
        row_end = U._row_end([S_GLOBAL // 2, S_GLOBAL // 2], S_GLOBAL)
        ref, gref, pref = _run(False, inp, row_end, loss_coeff=0.01)
        ref2, gref2, pref2 = _run(False, inp, row_end, loss_coeff=0.01)
        got, ggot, pgot = _run(True, inp, row_end, loss_coeff=0.01)

        # The KL has to actually reach the indexer, or every assertion below is
        # comparing zeros and the test is vacuous.
        indexer_grads = [
            n for n in pref if ".indexer." in n or n.startswith("indexer.")
        ]
        self.assertTrue(indexer_grads, "no indexer parameter received a grad")
        self.assertTrue(
            any(
                float(pref[n].astype("float32").abs().max()) > 0.0
                for n in indexer_grads
            ),
            "indexer parameter grads are all zero: the KL never fired, so "
            "this test would pass for any permutation",
        )

        self.assertEqual(_max_abs(ref, got), 0.0)
        self._assert_grads_match(
            (gref, gref2, ggot), (pref, pref2, pgot), "indexer-loss"
        )

    def _assert_grads_match(self, inputs, params, tag):
        """Hold off-vs-off and off-vs-dualchunk to the same relative bound."""
        for kind, (a, a2, b) in (("input", inputs), ("param", params)):
            self.assertEqual(sorted(a), sorted(b), f"{tag}: {kind} grad keys")
            for name in a:
                for label, other in (("off-vs-off", a2), ("dualchunk", b)):
                    rel = _rel(a[name], other[name])
                    self.assertLessEqual(
                        rel,
                        _GRAD_RTOL,
                        f"{tag} {kind} grad of {name} ({label}): rel "
                        f"{rel:.2e} exceeds {_GRAD_RTOL:.0e}",
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
