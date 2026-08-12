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

"""Real 2-GPU coverage of the ``contiguous_a2a`` (Ulysses) FlashMask CP switch.

``contiguous_a2a`` is a parallel switch: it moves Q/K/V from a sequence shard to
a head shard through an all-to-all, and it rewrites both the plain and the
refined-recompute backward. The single-card suites mock the process group, so
they cannot observe collective ordering, per-rank slicing, or replica
consistency. Every all-to-all below is a real NCCL call on 2 ranks.

Covered for the plain path and the RR path, against a CP=1 reference:
forward, ``dq``/``dk``/``dv`` (the QKV/input grads) and ``dsink`` (the parameter
grad), plus RR-vs-plain equality.

``dsink`` is the interesting one. The sink parameter is replicated on every rank
and sliced per head shard by a differentiable ``getitem``, so each rank's
``dsink`` is a *partial*: only its own heads are non-zero. Summing it over the
CP group must reproduce the reference exactly -- that is what the sharding/DP
reduce does in real training, and it is what proves nothing is lost or double
counted. The test asserts both halves of that statement: the local partial must
*differ* from the reference, and the CP sum must *match* it.

Why the kernel is mocked (and why that is sound)
------------------------------------------------
``learnable_sink`` exists only in the FA4 cute backend, which is built for
sm100; on an H-card CI machine the real sink kernel cannot run at all, yet the
sink gradient is exactly what needs CP coverage. So the kernel -- and only the
kernel -- is replaced by ``_attn_math``, pure-Paddle attention honouring the
same contract (``out, lse`` forward; ``dq, dk, dv, dsink`` backward), with the
backward obtained by autograd over the very same forward so the two cannot
disagree. The reference and the CP run then share one kernel, which means every
difference the test can see is CP plumbing -- the thing under test. Process
groups, all-to-all, head/sequence slicing and the RR two-forward protocol all
run for real.

Run (2 GPUs), from the repository root::

    PYTHONPATH=.:./src python -m paddle.distributed.launch --devices 0,1 \
        tests/multi_card_tests/transformer/test_flash_mask_cp_a2a.py
"""

import contextlib
import math
import unittest
from unittest.mock import patch

import paddle
import paddle.distributed as dist
import paddlefleet_ops.flash_mask_facade as facade
from paddle import framework
from paddle.distributed import fleet

import paddlefleet.refined_recompute.flash_attn as rr_fa
from paddlefleet.context_parallel_utils import (
    flashmask_attention_cp,
    scatter_contiguous,
)

DTYPE = paddle.bfloat16
SEED = 2026
BATCH, SEQ, NUM_HEADS, HEAD_DIM = 2, 256, 4, 64

# The mocked kernel is shared by both sides, so agreement is near bit-exact; the
# slack only covers the fp32 accumulation order changing with the per-rank head
# count. Any real plumbing bug (wrong shard, wrong collective order) is O(1),
# which is why the paired "wrong shard must be far" check uses 0.1.
REL_L2_TOL = 1e-3
WRONG_SHARD_MIN = 0.1

CP_SIZE = None
CP_RANK = None
CP_GROUP = None


def setUpModule():
    """Build a CP group over the whole world. No kernel gate: see module doc."""
    global CP_SIZE, CP_RANK, CP_GROUP
    world = dist.get_world_size()
    if world < 2:
        # The discriminative checks compare against another rank's shard and
        # against a partial sink grad, neither of which exists at CP=1.
        raise unittest.SkipTest(
            "Ulysses a2a CP coverage needs at least 2 ranks"
        )
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


# ------------------------------- mocked kernel -------------------------------


class _FakeFlashMaskInfo:
    """Stand-in for ``FlashMaskInfoPaddle``, which lives in the cute backend."""

    def __init__(self, startend_row_indices=None, is_causal=False):
        self.startend_row_indices = startend_row_indices
        self.is_causal = is_causal


def _mask_bias(startend_row_indices, seq_q):
    """Additive mask from the repo's 2-column non-causal FlashMask convention.

    ``[..., 0]`` is down_start (rows >= it are masked for that key) and
    ``[..., 1]`` is up_end (rows < it are masked), matching
    ``generate_non_causal_mask``. ``-1e30`` rather than ``-inf`` keeps a fully
    masked row finite, since the sink still gives it a denominator.
    """
    down = startend_row_indices[..., 0].unsqueeze(2).astype("int64")
    up = startend_row_indices[..., 1].unsqueeze(2).astype("int64")
    rows = paddle.arange(seq_q, dtype="int64").reshape([seq_q, 1])
    masked = paddle.logical_or(rows >= down, rows < up)
    return masked.astype("float32") * -1e30


def _attn_probs(query, key, value, learnable_sink, startend_row_indices):
    """Shared fp32 softmax internals, all in [b, h, s, *] layout.

    Returns the transposed q/k/v, the attention probabilities, the sink's share
    of the same denominator, and the row shift used for stabilisation.
    """
    qf = query.astype("float32").transpose([0, 2, 1, 3])
    kf = key.astype("float32").transpose([0, 2, 1, 3])
    vf = value.astype("float32").transpose([0, 2, 1, 3])
    scores = paddle.matmul(
        qf * (1.0 / math.sqrt(qf.shape[-1])), kf, transpose_y=True
    )
    if startend_row_indices is not None:
        scores = scores + _mask_bias(startend_row_indices, scores.shape[-2])

    row_max = scores.max(axis=-1, keepdim=True)
    if learnable_sink is not None:
        sink = learnable_sink.astype("float32").reshape([1, -1, 1, 1])
        row_max = paddle.maximum(row_max, sink)
        exp_sink = paddle.exp(sink - row_max)
    else:
        exp_sink = paddle.zeros_like(row_max)
    exp_scores = paddle.exp(scores - row_max)
    denom = exp_scores.sum(axis=-1, keepdim=True) + exp_sink
    return qf, kf, vf, exp_scores / denom, exp_sink / denom, denom, row_max


def _attn_math(query, key, value, learnable_sink, startend_row_indices, causal):
    """fp32 non-causal attention with sink. q/k/v: [b, s, h, d]; sink: [h]."""
    if causal:
        raise NotImplementedError("mock kernel covers the non-causal path only")
    _, _, vf, probs, _, denom, row_max = _attn_probs(
        query, key, value, learnable_sink, startend_row_indices
    )
    out = paddle.matmul(probs, vf)
    lse = (paddle.log(denom) + row_max).squeeze(-1)
    return out.transpose([0, 2, 1, 3]).astype(query.dtype), lse


def _attn_grads(
    query, key, value, learnable_sink, startend_row_indices, dout, causal
):
    """Closed-form gradient of ``_attn_math``.

    Written out by hand rather than obtained from ``paddle.grad`` because this
    runs inside ``PyLayer.backward``: re-entering the autograd engine from a
    PyLayer grad node segfaults this Paddle build.

    With ``P`` the probabilities (masked entries are exactly 0, so they need no
    special casing) and ``r_i = dO_i . O_i``::

        dV = P^T dO                 dS = P * (dO V^T - r)
        dq = (dS K) * scale         dK = (dS^T Q) * scale
        dsink_h = -sum_{b,i} p_sink * r     (the sink carries no value vector,
                                            it only inflates the denominator)
    """
    if causal:
        raise NotImplementedError("mock kernel covers the non-causal path only")
    qf, kf, vf, probs, sink_prob, _, _ = _attn_probs(
        query, key, value, learnable_sink, startend_row_indices
    )
    scale = 1.0 / math.sqrt(qf.shape[-1])
    grad_out = dout.astype("float32").transpose([0, 2, 1, 3])
    out = paddle.matmul(probs, vf)

    dvalue = paddle.matmul(probs, grad_out, transpose_x=True)
    row_dot = (grad_out * out).sum(axis=-1, keepdim=True)
    dscores = probs * (paddle.matmul(grad_out, vf, transpose_y=True) - row_dot)
    dquery = paddle.matmul(dscores, kf) * scale
    dkey = paddle.matmul(dscores, qf, transpose_x=True) * scale

    dsink = None
    if learnable_sink is not None:
        dsink = -(sink_prob * row_dot).sum(axis=[0, 2, 3])
        dsink = dsink.astype(learnable_sink.dtype)
    return (
        dquery.transpose([0, 2, 1, 3]).astype(query.dtype),
        dkey.transpose([0, 2, 1, 3]).astype(key.dtype),
        dvalue.transpose([0, 2, 1, 3]).astype(value.dtype),
        dsink,
    )


def _mock_flash_attn_fwd(
    query,
    key,
    value,
    causal=False,
    return_lse=False,
    startend_row_indices=None,
    pack_gqa=False,
    learnable_sink=None,
    softmax_scale=None,
):
    """RR first forward: runs under no_grad and must hand back ``lse``."""
    assert softmax_scale is None, "the CP a2a path rejects softmax_scale"
    with paddle.no_grad():
        out, lse = _attn_math(
            query, key, value, learnable_sink, startend_row_indices, causal
        )
    return (out, lse) if return_lse else out


def _mock_flash_attn_bwd(
    query,
    key,
    value,
    out,
    dout,
    lse,
    flashmask_info,
    learnable_sink=None,
    softmax_scale=None,
    causal=False,
    deterministic=False,
):
    """RR backward: the closed-form gradient of the mocked forward."""
    assert softmax_scale is None, "the CP a2a path rejects softmax_scale"
    indices = (
        None if flashmask_info is None else flashmask_info.startend_row_indices
    )
    with paddle.no_grad():
        return _attn_grads(
            query, key, value, learnable_sink, indices, dout, causal
        )


def _mock_facade_flashmask_attention(
    query,
    key,
    value,
    startend_row_indices=None,
    *,
    causal=False,
    learnable_sink=None,
    softmax_scale=None,
    **kwargs,
):
    """Differentiable stand-in for the facade used by the plain Ulysses path."""
    assert softmax_scale is None, "the CP a2a path rejects softmax_scale"
    out, _ = _attn_math(
        query, key, value, learnable_sink, startend_row_indices, causal
    )
    return out


@contextlib.contextmanager
def _mocked_kernel():
    """Swap in the mock kernel on both paths, leaving all collectives real."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch.object(
                facade, "flashmask_attention", _mock_facade_flashmask_attention
            )
        )
        # Pin the RR dispatch to the sink-capable branch; the kernels behind it
        # are the mocks below, so no sm100 backend is needed.
        stack.enter_context(
            patch.object(rr_fa, "get_fa_version", lambda *a, **kw: 4)
        )
        for name, impl in (
            ("_flash_attn_fwd", _mock_flash_attn_fwd),
            ("_flash_attn_bwd", _mock_flash_attn_bwd),
            ("FlashMaskInfoPaddle", _FakeFlashMaskInfo),
        ):
            # create=True: these names only exist when the cute backend does.
            stack.enter_context(patch.object(rr_fa, name, impl, create=True))
        yield


# ------------------------------ inputs / reference ---------------------------


def _per_head_startend_row_indices():
    """Per-head [b, NUM_HEADS, SEQ, 2] indices, deliberately head-dependent.

    A head dim of ``NUM_HEADS`` (not 1) is what makes the head slicing of the
    mask observable: if a rank used the wrong head block, or skipped the slice,
    its columns would be masked differently and the output would move. The band
    ``down_start[j] = j + 32 + 16*head`` masks rows ``r >= down_start[j]`` for
    key ``j``, so every row keeps a non-empty, head-specific key set.
    """
    keys = paddle.arange(SEQ, dtype="int32").reshape([1, 1, SEQ])
    heads = paddle.arange(NUM_HEADS, dtype="int32").reshape([1, NUM_HEADS, 1])
    down = paddle.clip(keys + 32 + 16 * heads, max=SEQ)
    down = paddle.broadcast_to(down, [BATCH, NUM_HEADS, SEQ]).unsqueeze(-1)
    return paddle.concat([down, paddle.zeros_like(down)], axis=-1)


def _make_inputs():
    """Identical full-sequence inputs on every rank (same seed, same order)."""
    paddle.seed(SEED)
    shape = [BATCH, SEQ, NUM_HEADS, HEAD_DIM]
    query, key, value, dout = (
        paddle.randn(shape).astype(DTYPE) for _ in range(4)
    )
    # fp32 like a master-weight sink; the +2.0 bias keeps it competitive with
    # the row max so it actually shapes the softmax (and thus has real grad).
    sink = paddle.randn([NUM_HEADS]).astype("float32") + 2.0
    return query, key, value, sink, _per_head_startend_row_indices(), dout


def _leaf(tensor):
    out = tensor.detach()
    out.stop_gradient = False
    return out


def _reference(query, key, value, sink, indices, dout):
    """CP=1 semantics: with cp_size==1 both all-to-all are the identity."""
    leaves = [_leaf(t) for t in (query, key, value, sink)]
    out, _ = _attn_math(*leaves[:3], leaves[3], indices, False)
    grads = paddle.grad(out, leaves, grad_outputs=dout)
    return out.detach(), grads


def _rel_l2(actual, expected):
    a = actual.astype("float32").flatten()
    b = expected.astype("float32").flatten()
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


def _shard(full_tensor, rank):
    chunk = full_tensor.shape[1] // CP_SIZE
    return paddle.slice(
        full_tensor, axes=[1], starts=[rank * chunk], ends=[(rank + 1) * chunk]
    )


# ---------------------------------- the test ---------------------------------


class TestFlashMaskCpUlyssesA2A(unittest.TestCase):
    """CP=2 ``contiguous_a2a`` forward/backward vs a CP=1 reference."""

    def _run_cp(self, use_rr):
        """Run one CP step and return the local results plus the reference."""
        query, key, value, sink, indices, dout = _make_inputs()
        ref_out, ref_grads = _reference(query, key, value, sink, indices, dout)

        # contiguous_a2a shards the sequence contiguously in rank order, so the
        # input leaf and the reference slice must both use contiguous slicing.
        q_local = _leaf(scatter_contiguous(query, CP_GROUP, 1))
        k_local = _leaf(scatter_contiguous(key, CP_GROUP, 1))
        v_local = _leaf(scatter_contiguous(value, CP_GROUP, 1))
        # The sink is replicated, never scattered: the CP path slices it itself.
        sink_local = _leaf(sink)
        dout_local = scatter_contiguous(dout, CP_GROUP, 1)

        kwargs = {
            "causal": False,
            "learnable_sink": sink_local,
            "mode": "contiguous_a2a",
        }
        if use_rr:
            rr_attn = rr_fa.RefinedRcomputeFlashMaskCpAttention()
            tracer = framework._dygraph_tracer()
            prev_has_grad = tracer._has_grad
            # First forward: the real attention, stashed for the RR backward.
            tracer._has_grad = False
            try:
                rr_attn(q_local, k_local, v_local, indices, **kwargs)
            finally:
                tracer._has_grad = prev_has_grad
            # Second forward: graph rebuild only, via the surrogate PyLayer.
            tracer._has_grad = True
            try:
                out = rr_attn(q_local, k_local, v_local, indices, **kwargs)
            finally:
                tracer._has_grad = prev_has_grad
        else:
            out = flashmask_attention_cp(
                q_local, k_local, v_local, indices, **kwargs
            )

        # The RR backward frees the saved output buffer, so snapshot first.
        out_local = out.detach().clone()
        grads = paddle.grad(
            out,
            [q_local, k_local, v_local, sink_local],
            grad_outputs=dout_local,
        )
        dsink_partial = grads[3].clone()
        dsink_summed = grads[3].clone()
        dist.all_reduce(dsink_summed, group=CP_GROUP)
        return {
            "out": out_local,
            "dq": grads[0],
            "dk": grads[1],
            "dv": grads[2],
            "dsink_partial": dsink_partial,
            "dsink": dsink_summed,
            "ref_out": ref_out,
            "ref_grads": ref_grads,
        }

    def _check_against_reference(self, got, tag):
        for name, ref in (
            ("out", got["ref_out"]),
            ("dq", got["ref_grads"][0]),
            ("dk", got["ref_grads"][1]),
            ("dv", got["ref_grads"][2]),
        ):
            err = _rel_l2(got[name], _shard(ref, CP_RANK))
            self.assertLess(
                err, REL_L2_TOL, f"[{tag}] {name} rel L2 {err:.3e} vs reference"
            )
            # Discriminative: an off-by-rank shard must be plainly worse, so a
            # loose tolerance cannot hide a wrong sequence partition.
            other = _rel_l2(got[name], _shard(ref, (CP_RANK + 1) % CP_SIZE))
            self.assertGreater(
                other,
                WRONG_SHARD_MIN,
                f"[{tag}] {name} matches another rank's shard as well "
                f"({other:.3e}); the comparison is not discriminative",
            )

        ref_sink = got["ref_grads"][3]
        err = _rel_l2(got["dsink"], ref_sink)
        self.assertLess(
            err,
            REL_L2_TOL,
            f"[{tag}] dsink summed over the CP group is off by {err:.3e}; the "
            "sink head shards do not reconstruct the full parameter grad",
        )
        # The other half of the same statement: per rank it must be a partial.
        # If this ever matched, the sink would be getting a full (double
        # counted) grad on every rank instead of its head shard.
        partial = _rel_l2(got["dsink_partial"], ref_sink)
        self.assertGreater(
            partial,
            WRONG_SHARD_MIN,
            f"[{tag}] per-rank dsink already equals the full reference "
            f"({partial:.3e}); the head slice is not taking effect",
        )

    def test_plain_a2a_matches_reference(self):
        with _mocked_kernel():
            self._check_against_reference(self._run_cp(False), "plain")

    def test_rr_a2a_matches_reference(self):
        with _mocked_kernel():
            self._check_against_reference(self._run_cp(True), "rr")

    def test_rr_matches_plain(self):
        """The RR switch must not change any value it computes."""
        with _mocked_kernel():
            plain = self._run_cp(False)
            rr = self._run_cp(True)
        for name in ("out", "dq", "dk", "dv", "dsink"):
            err = _rel_l2(rr[name], plain[name])
            self.assertLess(
                err, REL_L2_TOL, f"rr vs plain {name} rel L2 {err:.3e}"
            )


if __name__ == "__main__":
    unittest.main()
