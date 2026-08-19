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

"""Context parallel for the latent-MQA hybrid-MLA core attention.

Gold standard: a CP=N :class:`MQALatentAttention` must reproduce the CP=1
reference on its own query slice -- forward output, input gradients and every
parameter gradient (after the SUM all-reduce over the CP group) -- given the
same weights and the same global packed batch.

The layer takes its CP state from ``pg_collection.cp`` alone, so a reference and
a CP layer coexist in one process: the reference gets ``cp=None``.
``ContextParallelAllGatherOp`` however reads the *process-global* fleet CP group
(``context_parallel_utils.py:570-578``), which is fine because the fleet hybrid
config below sets ``cp_degree == world_size``.

Covered here (see ``test_mla_cp_contiguous_allgather.py`` for the MHA layer and
the full-MLA integration):

1. ``hybrid_mla_attention="mqa_full_causal"`` (``indexer is None``) equivalence.
2. ``hybrid_mla_attention="mqa_dsa"`` equivalence, single and multi document,
   including a document that straddles a rank boundary.
3. The indexer's RoPE under CP -- ``DSAIndexer.forward_before_topk`` must be
   **bitwise** equal to the global call, sliced. This is the load-bearing rope
   check: Q takes ``freqs[position_offset : position_offset + s]`` while K is
   all-gathered right after ``wk`` and rope'd with the full ``freqs``.
4. The selected column set: the sparse kernel's ``token_indices`` under CP must
   equal the reference's rows for this rank.
5. Indexer-loss normalisation, masked (global denominator) and unmasked
   (``/cp_size`` on the phase-3 path), and the phase-2
   ``dsa_indexer_use_sparse_loss=False`` full-causal KL.
6. The attention sink under CP.
7. The ``cp_balance_mode`` guard.

Backend note: the full-causal phases (``mqa_full_causal`` and the phase-2 warmup)
have dense FA4 as their only backend -- ``_assert_dense_fa4`` raises when the
process flags do not resolve to it -- and a bare launcher process leaves
``FLAGS_flash_attn_version`` at the image default 2. ``setUpModule`` therefore
pins 4, the value ``TrainingArguments.__post_init__`` derives on the SM100 boxes
these tests are gated to. A consequence is that no ``[b, s, s]`` column table
exists on those phases, so the column-set assertion applies to the phase-3 DSA
tests only.

Run (2 GPUs), from the repository root::

    PYTHONPATH=.:./src python -m paddle.distributed.launch \
        --devices 0,1 --nnodes 1 --master 127.0.0.1:<port> \
        tests/multi_card_tests/transformer/test_mqa_dsa_cp.py

The repository root on ``PYTHONPATH`` is what makes the shared
``tests.single_card_tests.transformer.hybrid_mla_utils`` helper importable;
``ci/multi-card_test.sh`` exports it for the same reason.
"""

import types
import unittest

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from paddlefleet.transformer.dsa_attention import DSAIndexerLossLoggingHelper

# ``tests`` is a package, so the single-card helper is a normal import; the
# multi-card runner puts the repository root on ``PYTHONPATH``
# (``ci/multi-card_test.sh``).
from tests.single_card_tests.transformer import hybrid_mla_utils as U

CP_SIZE = None
CP_RANK = None
CP_GROUP = None

# bf16 tolerances. The all-gather/reduce-scatter reorders bf16 reductions and
# ``csa_sparse_attn_bwd_cudnn`` accumulates ``dkv`` atomically (a ~2e-3
# run-to-run property of the kernel itself, not of CP), so gradients get a
# looser bound than the forward.
FWD_RTOL = 5e-3
GRAD_RTOL = 2e-2

# s_global for the equivalence runs. 512 keeps the O(s^2) dense-mode index table
# small and stays inside the regime where the top-k support set is bit
# reproducible, so the index-set test below can demand exact equality.
S_GLOBAL = 512

# The full-causal phases refuse to run on anything but dense FA4, and a bare
# launcher process never constructs ``TrainingArguments``, so the flag has to be
# pinned to the value production derives on these boxes -- and left alone where
# no FA4 backend exists, or the non-``_GPU``-gated cases would break. Module
# scope: nothing here asserts on the refusal.
_FA4_PIN = U._fa4_pin()


def setUpModule():
    global CP_SIZE, CP_RANK, CP_GROUP
    if dist.get_world_size() < 2:
        raise unittest.SkipTest("MQA context-parallel tests require >= 2 GPUs")
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


def _build(mode, cp_group, loss_coeff=0.0, sink=None, sparse_loss=True, seed=7):
    """CP=1 reference (``cp_group is None``) or CP layer, identical weights."""
    cfg = U._create_mqa_config(mode=mode, loss_coeff=loss_coeff)
    cfg.dsa_indexer_use_sparse_loss = sparse_loss
    cfg.cp_balance_mode = "contiguous_allgather"
    # Production EB dataflow hands every rank the *global* input_ids, which is
    # what ``_indexer_loss_mask`` assumes when this flag is set
    # (same branch as csa_attention.py:2419-2428).
    cfg.experimental_dataflow = True
    cfg.pad_token_id = 0
    cfg.context_parallel_size = 1 if cp_group is None else cp_group.nranks
    paddle.seed(seed)
    return U._build_module(
        cfg,
        bf16=True,
        sink=sink,
        pg_collection=types.SimpleNamespace(tp=None, cp=cp_group),
    )


def _inputs(s_global, seed=1234):
    """Global-length inputs, bitwise identical on every rank.

    ``paddle.randn`` would also agree across identical devices, but numpy makes
    that independent of the device RNG state, which the two module builds move.
    """
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


def _input_ids(s_global, n_pad=37):
    """``[1, s_global]`` int64 with a padded tail, so the row mask is non-trivial.

    ``n_pad`` is deliberately not a multiple of ``s_global / cp_size``: the pad
    rows must land on the last rank only, which is exactly the asymmetry that
    catches a per-rank (instead of global) loss denominator.
    """
    ids = np.arange(1, s_global + 1, dtype="int64")
    ids[s_global - n_pad :] = 0
    return paddle.to_tensor(ids).reshape([1, s_global])


def _leaf(t):
    out = t.clone().detach()
    out.stop_gradient = False
    return out


def _rel(a, e):
    a = a.cast("float32").flatten()
    e = e.cast("float32").flatten()
    return float((a - e).norm() / e.norm().clip(min=1e-12))


def _logged_indexer_loss(layer_number=1):
    """The indexer loss this layer just pushed into the logging tracker.

    ``_forward_warmup`` / ``_forward_sparse`` reduce the KL with the coefficient
    and denominator their phase picked (phase 2 always the global row count;
    phase 3 the global valid-row count when masked, ``/cp_size`` when not) and
    hand exactly that scalar to ``DSAIndexerLossLoggingHelper``, so the tracker
    is a direct read of the normalisation -- no gradient indirection. ``0.0``
    when the step attached no loss.
    """
    values = DSAIndexerLossLoggingHelper.tracker.get("values")
    if values is None:
        return 0.0
    return float(values[layer_number - 1])


def _last_captured():
    """Last ``token_indices`` the sparse kernel saw, or ``None``.

    ``None`` means no column table was built, which on these paths means a
    full-causal phase ran: dense FA4 is their only backend. Which backend ran is
    itself an assertion target (``test_mqa_dsa_warmup_cp.py::TestDenseFA4CP``),
    so this returns ``None`` rather than raising.
    """
    return U._CAPTURED[-1] if U._CAPTURED else None


def run_core_cp(
    mode,
    doc_lens,
    s_global=S_GLOBAL,
    loss_coeff=0.0,
    sink=None,
    with_input_ids=False,
    sparse_loss=True,
    row_end=None,
):
    """CP=1 reference then CP layer, on the same global batch.

    ``attn_mask_startend_row_indices`` is handed to *both* at the global length,
    which is what the layer sees in production (``dsv4_hybrid_attention.py:634``
    for the HCA layers, and the all-gathered ``kv_s`` for the MLA ones).
    ``input_ids`` is global too (``experimental_dataflow``).

    ``row_end`` overrides the mask built from ``doc_lens``. ``U._row_end`` folds
    the trailing gap into one final document, so it can never produce a
    row-validity pad row; the padded-layout tests
    (``test_mqa_dsa_warmup_cp.py``) pass their own table instead.
    """
    sl = s_global // CP_SIZE
    off = CP_RANK * sl
    if row_end is None:
        row_end = U._row_end(doc_lens, s_global)
    inp = _inputs(s_global)
    ids = _input_ids(s_global) if with_input_ids else None

    ref = _build(mode, None, loss_coeff, sink, sparse_loss)
    cpl = _build(mode, CP_GROUP, loss_coeff, sink, sparse_loss)
    cpl.set_state_dict(ref.state_dict())

    U._CAPTURED.clear()
    DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
    ra = {k: _leaf(v) for k, v in inp.items()}
    oa = ref(
        ra["query"],
        ra["key"],
        None,
        None,
        attn_mask_startend_row_indices=row_end.clone(),
        x=ra["x"],
        qr=ra["qr"],
        v_b_proj_weight=ra["w_v"],
        input_ids=ids,
    )
    oa.sum().backward()
    idx_ref = _last_captured()
    logged_ref = _logged_indexer_loss()

    U._CAPTURED.clear()
    DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
    rb = {
        "query": _leaf(inp["query"][:, off : off + sl]),
        "key": _leaf(inp["key"][:, off : off + sl]),
        "w_v": _leaf(inp["w_v"]),
        "x": _leaf(inp["x"][:, off : off + sl]),
        "qr": _leaf(inp["qr"][:, off : off + sl]),
    }
    ob = cpl(
        rb["query"],
        rb["key"],
        None,
        None,
        attn_mask_startend_row_indices=row_end.clone(),
        x=rb["x"],
        qr=rb["qr"],
        v_b_proj_weight=rb["w_v"],
        input_ids=ids,
    )
    ob.sum().backward()
    idx_cp = _last_captured()
    logged_cp = _logged_indexer_loss()

    # Parameter grads: this rank only saw its own query rows, so the CP group's
    # SUM is the reference. (``loss_coeff == 0`` leaves the indexer out of the
    # graph entirely, so only ``softmax_offset`` shows up there.)
    ref_named = dict(ref.named_parameters())
    param_err = {}
    for n, p in cpl.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.contiguous()
        dist.all_reduce(g, group=CP_GROUP)
        rg = ref_named[n].grad
        param_err[n] = None if rg is None else _rel(g, rg)

    # ``w_v`` is a plain input tensor shared by every row, so it reduces the same
    # way; ``query``/``key``/``x``/``qr`` are per-row and compare sliced. ``key``
    # goes through the all-gather, whose backward reduce-scatters the global
    # ``dkv`` back to this rank's slice -- that is the interesting one.
    dw = rb["w_v"].grad.contiguous()
    dist.all_reduce(dw, group=CP_GROUP)
    per_pos = (
        (ob.cast("float32") - oa[:, off : off + sl].cast("float32"))
        .abs()
        .max(axis=-1)
        .flatten()
        .tolist()
    )
    return {
        "fwd": _rel(ob, oa[:, off : off + sl]),
        "dq": _rel(rb["query"].grad, ra["query"].grad[:, off : off + sl]),
        "dkv": _rel(rb["key"].grad, ra["key"].grad[:, off : off + sl]),
        "dw": _rel(dw, ra["w_v"].grad),
        "param_err": param_err,
        "per_pos": per_pos,
        "idx_ref_slice": (
            None if idx_ref is None else idx_ref[:, off : off + sl]
        ),
        "idx_cp": idx_cp,
        # Raw tensors + the logged indexer loss, for the assertions that are not
        # a relative error: bitwise mode-vs-mode output equality, pad-row
        # zeroing and the loss-denominator check
        # (``test_mqa_dsa_warmup_cp.py``).
        "out": ob.detach(),
        "ref_out": oa.detach(),
        "dq_local": rb["query"].grad.detach(),
        "ref_dq": ra["query"].grad.detach(),
        "row_end": row_end,
        "logged_ref": logged_ref,
        "logged_cp": logged_cp,
    }


def _row_sets(idx):
    """Per-row set of selected columns, dropping the ``-1`` padding."""
    return [{int(c) for c in row if c >= 0} for row in idx[0]]


# Documents chosen so that every one of them straddles a rank boundary for both
# cp=2 (boundary 256) and cp=4 (128/256/384): [0,200) [200,350) [350,512).
_STRADDLE = [200, 150, 162]


class TestMQADSACP(unittest.TestCase):
    """A CP=N layer must reproduce the CP=1 reference on its own query slice."""

    def _check(self, res, tag):
        self.assertLess(
            res["fwd"], FWD_RTOL, f"{tag}: forward {res['fwd']:.3e}"
        )
        for key in ("dq", "dkv", "dw"):
            self.assertLess(res[key], GRAD_RTOL, f"{tag}: {key} {res[key]:.3e}")
        for name, err in res["param_err"].items():
            self.assertIsNotNone(
                err, f"{tag}: reference has no grad for {name}"
            )
            self.assertLess(err, GRAD_RTOL, f"{tag}: param {name} {err:.3e}")
        worst = max(res["param_err"].values(), default=0.0)
        print(
            f"[cp{CP_SIZE} rank{CP_RANK}] {tag}: fwd={res['fwd']:.2e} "
            f"dq={res['dq']:.2e} dkv={res['dkv']:.2e} dw={res['dw']:.2e} "
            f"param_max={worst:.2e} "
            f"per_pos_max={max(res['per_pos']):.3e}",
            flush=True,
        )

    def _check_index_sets(self, res, tag):
        """The sparse kernel must be handed the reference's own columns.

        Column ids are *global* on both sides (the CP layer indexes the
        all-gathered ``kv``), so this is a plain set equality per row -- no
        offset arithmetic. Set equality rather than sequence equality because
        the top-k order churns; see the top-k reproducibility audit.

        Only applicable to the phase-3 DSA path. The full-causal phases run dense
        FA4 and build no column table at all, so ``None`` here means the caller
        pointed this assertion at a phase that cannot support it.
        """
        for key in ("idx_ref_slice", "idx_cp"):
            self.assertIsNotNone(
                res[key],
                f"{tag}: {key} is None, so no column table was built; this "
                "assertion only applies to the phase-3 sparse path",
            )
        ref_sets = _row_sets(res["idx_ref_slice"])
        cp_sets = _row_sets(res["idx_cp"])
        self.assertEqual(len(ref_sets), len(cp_sets), f"{tag}: row count")
        for row, (want, got) in enumerate(zip(ref_sets, cp_sets)):
            self.assertEqual(
                got,
                want,
                f"{tag}: row {row} (global {CP_RANK * len(ref_sets) + row}) "
                f"selected a different column set: "
                f"missing={sorted(want - got)[:8]} "
                f"extra={sorted(got - want)[:8]}",
            )

    # ------------------------------------------------------------------
    # 1. dense path (``indexer is None``)
    # ------------------------------------------------------------------
    @U._GPU
    def test_1_dense_single_doc(self):
        res = run_core_cp("mqa", [S_GLOBAL])
        self._check(res, "dense/1doc")

    @U._GPU
    def test_2_dense_straddling_docs(self):
        """The full-causal index table is built at ``s_global`` then row-sliced.

        A per-rank build would clip every row at ``q - position_offset``, i.e.
        drop the whole prefix owned by lower ranks, which shows up here as an
        O(1) forward error on rank > 0.

        Dense FA4 encodes that table as an ``O(s)`` row bound rather than a
        column list, so the forward/gradient equivalence is the whole
        observable; ``_check_index_sets`` has nothing to read on this phase.
        """
        res = run_core_cp("mqa", _STRADDLE)
        self._check(res, "dense/3doc")

    # ------------------------------------------------------------------
    # 2. DSA path
    # ------------------------------------------------------------------
    @U._GPU
    def test_3_dsa_single_doc(self):
        res = run_core_cp("mqa_dsa", [S_GLOBAL])
        self._check(res, "dsa/1doc")
        self._check_index_sets(res, "dsa/1doc")

    @U._GPU
    def test_4_dsa_straddling_docs(self):
        res = run_core_cp("mqa_dsa", _STRADDLE)
        self._check(res, "dsa/3doc")
        self._check_index_sets(res, "dsa/3doc")

    @U._GPU
    def test_5_dsa_sink(self):
        """The sink is a per-head logit, independent of the query's position.

        Its gradient is accumulated per row, so the CP group's SUM must match
        the reference -- covered by ``param_err['softmax_offset']``.
        """
        sink = np.linspace(-1.0, 1.0, U.H)
        res = run_core_cp("mqa_dsa", _STRADDLE, sink=sink)
        self._check(res, "dsa/sink")
        self.assertIn(
            "softmax_offset",
            res["param_err"],
            "the sink must receive a gradient under CP",
        )

    # ------------------------------------------------------------------
    # 3. the indexer's RoPE -- the load-bearing check
    # ------------------------------------------------------------------
    @U._GPU
    def test_6_indexer_rope_under_cp(self):
        """``forward_before_topk`` under CP == the global call, sliced.

        Q is rope'd with ``freqs[position_offset : position_offset + s]`` while
        K is all-gathered right after ``wk`` and rope'd with the *full*
        ``freqs``, so a positional off-by-``position_offset`` on either side is
        invisible in the shapes and only shows up numerically. The control at
        the end feeds ``position_offset=0`` on purpose to prove this comparison
        actually has the power to see a wrong offset.
        """
        sl = S_GLOBAL // CP_SIZE
        off = CP_RANK * sl
        inp = _inputs(S_GLOBAL)
        ref = _build("mqa_dsa", None)
        cpl = _build("mqa_dsa", CP_GROUP)
        cpl.set_state_dict(ref.state_dict())

        with paddle.no_grad():
            q_r, k_r, w_r = ref.indexer.forward_before_topk(inp["x"], inp["qr"])
            q_c, k_c, w_c = cpl.indexer.forward_before_topk(
                inp["x"][:, off : off + sl],
                inp["qr"][:, off : off + sl],
                off,
                CP_GROUP,
            )

        # K must come back at the *global* length: the top-k scores every
        # global column, so a sharded K would silently truncate the support.
        self.assertEqual(
            list(k_c.shape),
            list(k_r.shape),
            "K must be all-gathered to the global length",
        )
        self.assertEqual(int(q_c.shape[1]), sl, "Q must stay sharded")

        errs = {}
        for name, got, want in (
            ("q", q_c, q_r[:, off : off + sl]),
            ("k", k_c, k_r),
            ("w", w_c, w_r[:, off : off + sl]),
        ):
            a = got.cast("float32")
            e = want.cast("float32")
            errs[name] = float((a - e).abs().max())
            scale = max(float(e.abs().max()), 1.0)
            self.assertLess(
                errs[name],
                1e-3 * scale,
                f"indexer {name}: max|diff|={errs[name]:.3e} "
                f"(scale {scale:.3e})",
            )
        print(
            f"[cp{CP_SIZE} rank{CP_RANK}] indexer rope max|diff| "
            f"q={errs['q']:.3e} k={errs['k']:.3e} w={errs['w']:.3e}",
            flush=True,
        )

        # ``forward_before_topk`` all-gathers K, so the control call must be
        # issued on *every* rank even though only rank > 0 can observe a
        # difference. Skipping it on rank 0 would leave the CP group's
        # collective streams off by one and deadlock the next test.
        with paddle.no_grad():
            q_bad, _, _ = cpl.indexer.forward_before_topk(
                inp["x"][:, off : off + sl],
                inp["qr"][:, off : off + sl],
                0,
                CP_GROUP,
            )
        if CP_RANK == 0:
            return
        bad = float(
            (q_bad.cast("float32") - q_r[:, off : off + sl].cast("float32"))
            .abs()
            .max()
        )
        self.assertGreater(
            bad,
            100.0 * max(errs["q"], 1e-6),
            "position_offset=0 was indistinguishable from the correct offset, "
            f"so this check is vacuous (good={errs['q']:.3e} bad={bad:.3e})",
        )
        print(
            f"[cp{CP_SIZE} rank{CP_RANK}] rope control: wrong offset gives "
            f"{bad:.3e} vs {errs['q']:.3e}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 4. indexer-loss normalisation
    # ------------------------------------------------------------------
    @U._GPU
    def test_7_indexer_loss_normalisation(self):
        """With ``loss_coeff > 0`` the indexer joins the backward graph.

        That makes its parameter gradients the observable for the loss
        denominator: masked rows divide by the *global* valid-row count and
        unmasked rows by ``cp_size``, so in both cases the CP group's SUM has
        to land on the CP=1 reference. A per-rank denominator is off by
        ``cp_size`` (or by the pad imbalance, which ``_input_ids`` puts on the
        last rank only) and fails here.
        """
        for masked, sparse in ((True, True), (False, True), (True, False)):
            with self.subTest(masked=masked, sparse_loss=sparse):
                res = run_core_cp(
                    "mqa_dsa",
                    _STRADDLE,
                    loss_coeff=0.1,
                    with_input_ids=masked,
                    sparse_loss=sparse,
                )
                self.assertTrue(
                    any(n.startswith("indexer.") for n in res["param_err"]),
                    "the indexer must receive gradients when loss_coeff > 0",
                )
                self._check(res, f"loss/masked={masked}/sparse={sparse}")

    # ------------------------------------------------------------------
    # 5. the HCA-compatibility guard
    # ------------------------------------------------------------------
    def test_8_cp_balance_mode_guard(self):
        """Only ``contiguous_allgather`` is implemented.

        The hybrid model's HCA layers assert the same mode
        (``dsv4_hybrid_attention.py:607-611``); the MQA layer must refuse the
        others rather than silently attend against a permuted KV.
        """
        cfg = U._create_mqa_config(mode="mqa_dsa")
        cfg.context_parallel_size = CP_SIZE
        for mode in ("p2p", "zigzag", None):
            with self.subTest(cp_balance_mode=mode):
                cfg.cp_balance_mode = mode
                with self.assertRaises(NotImplementedError):
                    U._build_module(
                        cfg,
                        bf16=True,
                        pg_collection=types.SimpleNamespace(
                            tp=None, cp=CP_GROUP
                        ),
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
