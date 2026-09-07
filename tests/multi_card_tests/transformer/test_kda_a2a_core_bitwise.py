#  Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Bitwise coverage of the head-sharded all-to-all KDA core, on real ranks.

``linear_attn_cp_mode="headwise"`` exists to make context parallelism *exact*
for the linear-attention layers, and this file is what pins that claim down.
The KDA core -- ``causal_conv1d`` + ``chunk_kda`` -- must be **bitwise**
identical between

    single card:  full sequence, all heads
    headwise CP:  full sequence, this rank's head shard

for the output, the input-side gradients (``dqkv``/``dalpha``/``dbeta``) *and*
the per-head parameter gradients (``dA_log``/``ddt_bias``/``dconv_weight``).  It
can be, because after the head swap the recurrence sees the whole sequence with
plain single-card semantics -- no conv halo, no cross-rank state relay -- and
the per-head parameters reduce over a token axis that CP never cuts, with each
rank's partial gradient exactly ``0.0`` outside its own heads.  The single
measured exception, on the non-production plain-gate path, is documented at
``_DEMOTED_TO_L2`` below, together with why it is a property of head splitting
in general rather than of the all-to-all.

Deliberately kept *below* the layer: no ``in_proj``, no ``out_proj``, no fleet
topology.  Those projections are GEMMs whose ``M`` differs between the two arms
(``s/cp`` vs ``s``), so cuBLAS may pick a different algorithm for each and they
can only ever be compared with a tolerance.  Mixing them in here would turn every
bitwise assertion in this file into a tolerance one and lose the signal that the
core itself is exact.  ``test_kda_head_a2a_layer.py`` is the tolerance half of
the pair; the split between the two files is the point, not an accident.

Also no ``fleet.init``: the core needs nothing but a process group, so the CP
group here is just ``dist.new_group(range(world))``.  Reproducing the production
``dp=1, sharding=cp=ep=world`` topology is the layer test's job.

Three kinds of assertion, all of which must hold:

  * ``max|diff| == 0`` against the reference (bitwise, no tolerance)
  * per-rank parameter grads must **differ** from the reference before the CP
    sum -- otherwise the head slice is not taking effect, every rank is computing
    the full gradient, and the trainer-side reduction will double-count it
  * the local output must **not** match a neighbour's sequence shard -- a
    guard against a wrong sequence partition passing because everything is zero

Launch (needs >= 2 GPUs; 4 in CI):

    python -m paddle.distributed.launch --gpus="0,1,2,3" \
        transformer/test_kda_a2a_core_bitwise.py
"""

from __future__ import annotations

import unittest

import paddle
import paddle.distributed as dist
from paddlefleet_ops.fla.modules.conv.causal_conv1d import causal_conv1d
from paddlefleet_ops.fla.ops.kda import chunk_kda

from paddlefleet.transformer.kda_head_a2a import (
    head_to_seq,
    seq_to_head,
    seq_to_head_beta,
    slice_channels_by_head,
    slice_per_head_param,
    split_qkv_seq_to_head,
)
from tests.multi_card_tests.transformer.kda_a2a_utils import (
    WRONG_SHARD_MIN,
    pin_fla_autotune,
    rel_l2,
)

SEED = 20260903
DTYPE = "bfloat16"

# GVA on purpose: HV = 2 * H.  With H == HV a bug that slices the value heads
# with the key-head count would be invisible.
SEQ = 256
NUM_KEY_HEADS = 4
NUM_VALUE_HEADS = 8
KEY_HEAD_DIM = VALUE_HEAD_DIM = 32
CONV_KERNEL = 4
# Documents of the packed sequence, deliberately not aligned to SEQ // cp_size:
# under a2a the conv needs no halo precisely because every rank sees all of them.
DOCS = [100, 60, 96]

QK_DIM = NUM_KEY_HEADS * KEY_HEAD_DIM
V_DIM = NUM_VALUE_HEADS * VALUE_HEAD_DIM
CONV_DIM = 2 * QK_DIM + V_DIM
# How ``qkv`` lays out its channels, and how many heads each block has.
QKV_DIMS = (QK_DIM, QK_DIM, V_DIM)
QKV_HEADS = (NUM_KEY_HEADS, NUM_KEY_HEADS, NUM_VALUE_HEADS)

CP_GROUP = None
CP_RANK = None
CP_SIZE = None


def setUpModule():
    global CP_GROUP, CP_RANK, CP_SIZE
    world = dist.get_world_size()
    if world < 2:
        raise unittest.SkipTest(
            "head-shard a2a coverage needs at least 2 ranks"
        )
    dist.init_parallel_env()
    CP_GROUP = dist.new_group(list(range(world)))
    CP_RANK, CP_SIZE = CP_GROUP.rank, CP_GROUP.nranks
    for name, count in (("key", NUM_KEY_HEADS), ("value", NUM_VALUE_HEADS)):
        if count % CP_SIZE:
            raise unittest.SkipTest(
                f"num_{name}_heads={count} is not divisible by cp={CP_SIZE}"
            )
    # Must happen before the first kernel launch: @autotune keys include H/HV,
    # so without pinning the two arms would legitimately pick different configs
    # and the bitwise assertions below would be testing the autotuner.
    pin_fla_autotune()


def _cu_seqlens():
    bounds, acc = [0], 0
    for doc in DOCS:
        acc += doc
        bounds.append(acc)
    assert bounds[-1] == SEQ, f"DOCS sum to {bounds[-1]}, expected SEQ={SEQ}"
    cu = paddle.to_tensor(bounds, dtype="int64")
    return cu, cu.cpu()


def _inputs():
    """Byte-identical inputs on every rank.

    Same seed on same-model GPUs would almost certainly already agree, but a
    single broadcast removes an entire class of false failure: if the two arms
    ever disagreed because rank 3 generated a different random bit, the failure
    would look exactly like a real bitwise break.
    """
    paddle.seed(SEED)
    hv_k = NUM_VALUE_HEADS * KEY_HEAD_DIM
    out = {
        # qkv is the fused in_proj output: one contiguous channel block.
        "qkv": paddle.randn([1, SEQ, CONV_DIM]).astype(DTYPE),
        # alpha is the raw gate input (use_gate_in_kernel=True), from f_b_proj.
        "alpha": paddle.randn([1, SEQ, hv_k]).astype(DTYPE),
        # beta is raw logits; the layer keeps it in fp32 (kimi_delta_attention.py:755).
        "beta": paddle.randn([1, SEQ, NUM_VALUE_HEADS], dtype="float32"),
        "conv_w": paddle.randn([CONV_DIM, CONV_KERNEL], dtype="float32") * 0.1,
        "conv_b": paddle.randn([CONV_DIM], dtype="float32") * 0.1,
        # Production init: A_log = log(U(1, 16)), see the layer's reset_parameters.
        "A_log": paddle.log(
            paddle.uniform(
                [NUM_VALUE_HEADS], min=1.0, max=16.0, dtype="float32"
            )
        ),
        "dt_bias": paddle.randn([hv_k], dtype="float32"),
        "do": paddle.randn([1, SEQ, NUM_VALUE_HEADS, VALUE_HEAD_DIM]).astype(
            DTYPE
        ),
    }
    for tensor in out.values():
        dist.broadcast(tensor, src=0, group=CP_GROUP)
    return out


def _leaf(tensor):
    leaf = tensor.detach().clone()
    leaf.stop_gradient = False
    return leaf


def _core(
    qkv, alpha, beta, conv_w, conv_b, A_log, dt_bias, cu, cu_cpu, lower_bound
):
    """conv + chunk_kda, exactly as ``kimi_delta_attention.py:785-837`` calls them.

    ``cp_context`` is ``None`` in both arms -- that is the whole point of the a2a
    route: after the head swap this rank holds the *full* sequence, so the core
    runs single-card semantics with the *global* ``cu_seqlens``.  No halo, no
    state relay, no re-chunking.
    """
    heads_v = A_log.shape[0]
    qkv, _ = causal_conv1d(
        qkv.contiguous(),
        weight=conv_w,
        bias=conv_b,
        activation="silu",
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
        cp_context=None,
    )
    heads_k = (qkv.shape[-1] - heads_v * VALUE_HEAD_DIM) // (2 * KEY_HEAD_DIM)
    query, key, value = paddle.split(
        qkv,
        [
            heads_k * KEY_HEAD_DIM,
            heads_k * KEY_HEAD_DIM,
            heads_v * VALUE_HEAD_DIM,
        ],
        axis=-1,
    )
    seq = qkv.shape[1]
    out, _ = chunk_kda(
        q=query.reshape([1, seq, heads_k, KEY_HEAD_DIM]).contiguous(),
        k=key.reshape([1, seq, heads_k, KEY_HEAD_DIM]).contiguous(),
        v=value.reshape([1, seq, heads_v, VALUE_HEAD_DIM]).contiguous(),
        g=alpha.reshape([1, seq, heads_v, KEY_HEAD_DIM]).contiguous(),
        beta=beta.contiguous(),
        A_log=A_log,
        dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=lower_bound is not None,
        lower_bound=lower_bound,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
        cp_context=None,
    )
    return out


def _reference(inp, lower_bound):
    """Single-card arm: full sequence, all heads, no communication."""
    cu, cu_cpu = _cu_seqlens()
    leaves = {n: _leaf(inp[n]) for n in ("qkv", "alpha", "beta")}
    params = {
        n: _leaf(inp[n]) for n in ("conv_w", "conv_b", "A_log", "dt_bias")
    }
    out = _core(
        leaves["qkv"],
        leaves["alpha"],
        leaves["beta"],
        params["conv_w"],
        params["conv_b"],
        params["A_log"],
        params["dt_bias"],
        cu,
        cu_cpu,
        lower_bound,
    )
    order = list(leaves) + list(params)
    grads = paddle.grad(
        out, [*leaves.values(), *params.values()], grad_outputs=inp["do"]
    )
    return out, dict(zip(order, grads))


def _cp_arm(inp, lower_bound):
    """a2a arm: this rank owns a sequence shard on the outside, a head shard inside."""
    cu, cu_cpu = _cu_seqlens()
    s_local = SEQ // CP_SIZE
    seq_shard = slice(CP_RANK * s_local, (CP_RANK + 1) * s_local)

    leaves = {n: _leaf(inp[n][:, seq_shard]) for n in ("qkv", "alpha", "beta")}
    # Parameters stay at full shape and are sliced *inside* the graph, so their
    # grads come back full-shape with exact 0.0 outside this rank's heads.  That
    # is what makes the CP sum bitwise (x + 0 == x) and leaves both
    # sharded_state_dict and the trainer-side reduction untouched -- exactly what
    # the layer does at ``kimi_delta_attention.py``'s head-a2a block.
    params = {
        n: _leaf(inp[n]) for n in ("conv_w", "conv_b", "A_log", "dt_bias")
    }

    query, key, value = split_qkv_seq_to_head(
        leaves["qkv"], QKV_DIMS, QKV_HEADS, CP_GROUP
    )
    # Re-concatenating the three channel blocks is numerically an identity for a
    # depthwise conv (groups == conv_dim): every channel is independent.
    qkv_h = paddle.concat([t.flatten(2) for t in (query, key, value)], axis=-1)
    alpha_h = seq_to_head(
        leaves["alpha"].reshape([1, s_local, NUM_VALUE_HEADS, KEY_HEAD_DIM]),
        CP_GROUP,
    ).flatten(2)
    beta_h = seq_to_head_beta(leaves["beta"], CP_GROUP)

    out_h = _core(
        qkv_h,
        alpha_h,
        beta_h,
        slice_channels_by_head(
            params["conv_w"], QKV_DIMS, QKV_HEADS, CP_RANK, CP_SIZE
        ),
        slice_channels_by_head(
            params["conv_b"], QKV_DIMS, QKV_HEADS, CP_RANK, CP_SIZE
        ),
        slice_per_head_param(
            params["A_log"], NUM_VALUE_HEADS, CP_RANK, CP_SIZE
        ),
        slice_per_head_param(
            params["dt_bias"], NUM_VALUE_HEADS, CP_RANK, CP_SIZE
        ),
        cu,
        cu_cpu,
        lower_bound,
    )
    out = head_to_seq(out_h, CP_GROUP)

    order = list(leaves) + list(params)
    grads = paddle.grad(
        out,
        [*leaves.values(), *params.values()],
        grad_outputs=inp["do"][:, seq_shard],
    )
    return out, dict(zip(order, grads))


# For the one gradient that is measured not to be bitwise, see ``_DEMOTED_TO_L2``
# below.  The bound used there is the textbook forward error bound for a
# floating-point summation of ``n`` terms -- ``n * eps * sum|x_i|`` -- not a
# tolerance fitted to the observation.  ``eps = 2**-24`` is fp32's unit roundoff
# (the accumulator is fp32 even though the addends are bf16) and ``n`` is SEQ,
# the length of the reduced token axis.
FP32_EPS = 2**-24
SUM_ERR_BOUND = SEQ * FP32_EPS

# The tier-L2 gate for the one demoted tensor.  Measured 2.869e-06 at cp=4;
# 1e-05 leaves headroom for a different GPU's reduce blocking without admitting a
# real bug (a wrong head slice moves a whole block, rel-L2 ~1e-01).
REDUCE_REL_L2_TOL = 1e-5

# ``ddt_bias`` on the *non-production* gate path (``lower_bound=None``) is the one
# tensor that is not bitwise, and only from cp=4 up.  Measured: 2 of 256 elements
# off by 2.38e-07 (= 2^-22), and -- this is the part that matters -- both of them
# inside the owning rank's own head block, so the head slicing and the CP sum are
# exact; what changed is the *intra-rank* reduction.
#
# ``ops/kda/gate.py:288`` computes it as ``dg.view(-1, H * K).sum(0)``.  a2a never
# splits that reduction's ``T`` axis, which is why the disjoint-support argument
# above ("each rank's partial is exactly 0.0 outside its own heads, so the CP sum
# is x + 0") looked sufficient -- but the row *width* does change,
# ``HV*K`` -> ``HV/P*K``, and a framework reduce picks its blocking from the
# shape.  Different blocking, different fp32 accumulation order over ``T``.
#
# Note the absolute deviation is ~9e-06 of the tensor's own max magnitude, i.e.
# far more than 1 ULP *of the result* -- because the sum cancels.  Reordering a
# cancelling sum perturbs it by an ULP of the largest partial sum, so the result's
# own magnitude is the wrong yardstick; ``_assert_reduction_noise`` uses
# ``sum|terms|`` instead.
#
# It is data-dependent, not structural: the same head split performed in-process
# on a single card at these exact shapes is bitwise, and so is a randn-filled
# reduce at these widths.  Real ``dg`` spans a much wider dynamic
# range along ``K`` (it is a reverse cumsum of log-space gates), which is what
# makes the reordering visible.  So this is **not** an a2a defect -- any head
# split can hit it -- and the production path (``gate_lower_bound=-5.0``,
# ``safe_gate=True``) is exactly bitwise on 2 and 4 ranks.
_DEMOTED_TO_L2 = {None: {"dt_bias"}, -5.0: set()}


class KdaA2ACoreBitwiseTest(unittest.TestCase):
    """Bitwise equality of the KDA core between single-card and head-shard a2a."""

    def _assert_bitwise(self, got, ref, name):
        diff = (got.astype("float32") - ref.astype("float32")).abs()
        self.assertTrue(
            bool(paddle.isfinite(diff).all()),
            f"[rank {CP_RANK}] {name}: non-finite",
        )
        mismatched = int((diff != 0).sum())
        if not mismatched:
            return
        # Where the mismatches sit is the whole diagnosis for a per-head
        # parameter: inside this rank's own head block means the local reduction
        # ordering changed, outside it means another rank's supposedly-zero
        # region is not actually zero -- a real slicing bug, not a rounding one.
        where = paddle.nonzero(diff.reshape([-1]) != 0).reshape([-1])
        head = [int(i) for i in where[:8]]
        self.fail(
            f"[rank {CP_RANK}] {name}: NOT bitwise -- max|diff|={float(diff.max()):.6e}, "
            f"mismatched={mismatched}/{diff.numel()}, flat idx {head}"
            f"{' ...' if mismatched > len(head) else ''}"
        )

    def _assert_reduction_noise(self, got, ref, terms_l1, name):
        """Demote ONE tensor from L1 to L2, with the deviation's scale explained.

        The result's own magnitude is the wrong yardstick here: this sum cancels,
        so reordering it perturbs the answer by an ULP of the largest partial sum,
        not of the answer.  ``terms_l1`` (``sum_t |term_t|``, from ``dalpha``) gives
        the accumulator's scale, and the classic summation error bound
        ``n * eps * sum|terms|`` says how far a reordering may legitimately move
        the result.  The measured deviation sits at ~1.25x that estimate -- i.e.
        exactly at rounding scale, given that ``dalpha`` is only an O(1) proxy for
        the ``dg`` being reduced (they differ by the gate's chain factor
        ``A * sigmoid(...)``, and ``A = exp(A_log)`` spans [1, 16] here).
        That ratio is reported, not asserted: it is evidence about the mechanism,
        while the gate below is the stable, tier-L2 claim.
        """
        diff = (got.astype("float32") - ref.astype("float32")).abs()
        bound = SUM_ERR_BOUND * terms_l1.astype("float32").reshape(diff.shape)
        ratio = float((diff / paddle.clip(bound, min=float(FP32_EPS))).max())
        rel = rel_l2(got, ref)
        if CP_RANK == 0:
            print(
                f"    [L2] {name}: rel-L2 {rel:.3e}, max|diff| "
                f"{float(diff.max()):.6e}, {int((diff != 0).sum())}/{diff.numel()} "
                f"elements, {ratio:.2f}x the summation error bound",
                flush=True,
            )
        self.assertLess(
            rel,
            REDUCE_REL_L2_TOL,
            f"[rank {CP_RANK}] {name}: rel-L2 {rel:.3e} >= {REDUCE_REL_L2_TOL:.1e} "
            f"-- far beyond the measured reduction-order effect ({ratio:.2f}x the "
            f"summation error bound), so this is a real regression, not rounding",
        )

    def _assert_single_head_block(self, got, ref, name):
        """Every deviating element must sit in ONE rank's head block.

        This is the assertion that keeps the tolerated case honest.  A rounding
        difference in one rank's own reduction can only touch that rank's slots;
        a slicing bug -- a rank whose supposedly-zero region is not zero, or heads
        paired with the wrong ``A_log`` -- would spread across blocks.  So the
        weaker magnitude claim is fenced in by a structural one.
        """
        diff = (got.astype("float32") - ref.astype("float32")).abs()
        where = paddle.nonzero(diff.reshape([-1]) != 0).reshape([-1])
        if not int(where.numel()):
            return
        # int() on both sides: paddle's numel() returns a 0-d Tensor, and Tensor
        # floor-division would make every index its own unhashable object -- the
        # set would then have one entry per mismatch and never compare equal to 1.
        per_rank = int(diff.numel()) // CP_SIZE
        blocks = sorted({int(i) // per_rank for i in where})
        self.assertEqual(
            len(blocks),
            1,
            f"[rank {CP_RANK}] {name}: deviations span head blocks {blocks} "
            f"(block size {per_rank}) -- rounding cannot do that, so the head "
            f"slicing or the disjoint-support property is broken",
        )

    def _check(self, lower_bound):
        inp = _inputs()
        ref_out, ref_grad = _reference(inp, lower_bound)
        cp_out, cp_grad = _cp_arm(inp, lower_bound)

        s_local = SEQ // CP_SIZE
        mine = slice(CP_RANK * s_local, (CP_RANK + 1) * s_local)
        neighbour = slice(
            ((CP_RANK + 1) % CP_SIZE) * s_local,
            ((CP_RANK + 1) % CP_SIZE + 1) * s_local,
        )

        # L1a: output and input-side grads, against this rank's sequence shard.
        self._assert_bitwise(cp_out, ref_out[:, mine], "core_attn_out")
        for name in ("qkv", "alpha", "beta"):
            self._assert_bitwise(
                cp_grad[name], ref_grad[name][:, mine], f"d{name}"
            )

        # Guard against a wrong sequence partition: if the output were all zeros,
        # or if every rank computed the same thing, the check above would still
        # pass.  It must NOT match the next rank's shard.
        wrong = rel_l2(cp_out, ref_out[:, neighbour])
        self.assertGreater(
            wrong,
            WRONG_SHARD_MIN,
            f"[rank {CP_RANK}] output matches rank {(CP_RANK + 1) % CP_SIZE}'s shard "
            f"(rel-L2 {wrong:.3e}) -- the sequence partition is not taking effect",
        )

        # L1b: per-head parameter grads.  Two assertions, and the first one is the
        # one that actually catches bugs: the local partial must DIFFER from the
        # reference, otherwise the head slice is a no-op and every rank is
        # silently accumulating the full (double-counted) gradient.
        demoted = _DEMOTED_TO_L2[lower_bound]
        for name in ("conv_w", "conv_b", "A_log", "dt_bias"):
            partial, ref = cp_grad[name], ref_grad[name]
            local = rel_l2(partial, ref)
            self.assertGreater(
                local,
                WRONG_SHARD_MIN,
                f"[rank {CP_RANK}] d{name}: local partial already matches the "
                f"reference (rel-L2 {local:.3e}) -- head slicing is not taking effect",
            )
            summed = partial.clone()
            # The layer must never do this itself -- the sharding group already
            # contains the CP group, so a layer-local all_reduce would
            # double-count.  The test stands in for that trainer-side reduction.
            dist.all_reduce(summed, group=CP_GROUP)
            if name in demoted:
                # ``dalpha`` is the gate-input gradient, one chain-rule factor away
                # from the ``dg`` that gate.py:288 actually reduces, so its L1 is a
                # proxy for the true term magnitudes -- correct to an O(1) factor,
                # which is all a summation bound needs.
                terms_l1 = ref_grad["alpha"].astype("float32").abs().sum(axis=1)
                self._assert_reduction_noise(
                    summed, ref, terms_l1, f"d{name} (CP-summed)"
                )
                self._assert_single_head_block(
                    summed, ref, f"d{name} (CP-summed)"
                )
            else:
                self._assert_bitwise(summed, ref, f"d{name} (CP-summed)")

    def test_core_bitwise_safe_gate(self):
        """Production path: gate_lower_bound=-5.0 -> safe_gate=True."""
        self._check(lower_bound=-5.0)

    def test_core_bitwise_plain_gate(self):
        """softplus gate, no clamp -- a different kernel branch, so tested too."""
        self._check(lower_bound=None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
