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

"""Layer-level coverage of ``linear_attn_cp_mode="headwise"`` on 4 ranks.

The second half of a pair.  ``test_kda_a2a_core_bitwise.py`` proves the KDA core
is **bitwise** under the head swap; this file proves the whole
``KimiDeltaAttention`` layer -- projections, gate, ``out_norm``, ``out_proj``, the
real fleet topology, the real config plumbing -- computes the right answer.

It deliberately does **not** assert bitwise, and that is not a concession:
``in_proj``/``out_proj`` are GEMMs whose ``M`` is ``seq_len`` on the CP arm and
``seq_len * cp`` on the reference arm, so cuBLAS may pick a different algorithm
and the two arms' KDA inputs already differ by ~1 ULP before the core is even
reached.  Asserting ``max|diff| == 0`` here would be asserting something known to
be false, and the useful signal -- that the core itself is exact -- would drown in
it.  So: rel-L2 here, bitwise there.

What this file catches that the core test cannot:

  * ``linear_attn_cp_mode="headwise"`` actually reaching the KDA layer instead of
    being rejected by config validation, with the *global* token layout staying
    contiguous rather than flipping to dualchunk
  * ``build_cp_context`` being skipped, so ``cu_seqlens`` stays **global**.  The
    sequence-split path slices it down to this rank, which under the head swap
    would silently merge the documents; asserted directly by making the call fail
  * ``eff_seq`` vs ``seq_len_full`` reshape mix-ups around the swap
  * the per-head parameters being sliced with the layer's own ``tp``-local head
    counts rather than the global ones
  * the ``__init__`` guard that rejects head counts not divisible by ``cp``

Every weight gradient is asserted as a **pair**: the local partial must *differ*
from the reference, and only the CP sum may match it.  Without the first half a
layer that ignored the head shard entirely -- every rank computing the full
gradient -- would pass, and would then double-count once the trainer reduces over
the sharding group.

Launch (needs exactly 4 GPUs):

    python -m paddle.distributed.launch --gpus="0,1,2,3" \
        transformer/test_kda_head_a2a_layer.py
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle.distributed import fleet

import paddlefleet.parallel_state as ps
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    Linear,
    RowParallelLinear,
)
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer import kimi_delta_attention as kda_mod
from paddlefleet.transformer.kimi_delta_attention import (
    HAVE_FLA,
    KimiDeltaAttention,
    KimiDeltaAttentionSublayersSpec,
)
from paddlefleet.transformer.paddle_norm import RMSNorm
from paddlefleet.transformer.transformer_config import TransformerConfig
from tests.multi_card_tests.transformer.kda_a2a_utils import (
    WRONG_SHARD_MIN,
    pin_fla_autotune,
    rel_l2,
)

CONTEXT_PARALLEL = 4
HIDDEN_SIZE = 128
KEY_HEAD_DIM = VALUE_HEAD_DIM = 32
# GVA: HV = 2 * H.  The existing CP test uses H == HV, which cannot catch a
# value-head block sliced with the key-head count.
NUM_KEY_HEADS = 4
NUM_VALUE_HEADS = 8
CONV_KERNEL_DIM = 4
SEQ_LENGTH = 256
SEED = 1234
DOCS = [100, 60, 96]

# Layer level, so the projections' GEMM shapes differ between the arms: rel-L2,
# not bitwise.  See the module docstring.
REL_L2_TOL = 1e-3

MODE = "headwise"
# The head swap is layer-local, so the *global* token layout must stay the
# contiguous one it assumes: ``ContextParallelScatterOp`` dispatches on
# ``startswith("contiguous")`` and its else-branch silently uses the dualchunk
# layout, which would hand the gathered sequence to the recurrence permuted.
# ``TransformerConfig`` rejects the combination outright; asserting it here as
# well keeps the intent visible.
CP_BALANCE_MODE = "contiguous_allgather"


def _config(cp_size, mode):
    return TransformerConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_VALUE_HEADS,
        num_hidden_layers=1,
        hidden_act=F.silu,
        rms_norm_eps=1e-6,
        normalization="RMSNorm",
        context_parallel_size=cp_size,
        cp_balance_mode=CP_BALANCE_MODE,
        linear_attn_cp_mode=mode,
        deterministic_mode=False,
    )


def _build(config, pg_collection, **overrides):
    spec = KimiDeltaAttentionSublayersSpec(
        in_proj=ColumnParallelLinear,
        f_a_proj=Linear,
        f_b_proj=ColumnParallelLinear,
        out_norm=RMSNorm,
        out_proj=RowParallelLinear,
    )
    kwargs = {
        "config": config,
        "sublayers_spec": spec,
        "layer_number": 1,
        "pg_collection": pg_collection,
        "conv_kernel_dim": CONV_KERNEL_DIM,
        # Off by default, turned on here on purpose: the bias is the one conv
        # parameter the layer slices behind an ``if conv_bias is not None`` branch,
        # so with the default the branch would never execute in either test.
        "conv_bias": True,
        "key_head_dim": KEY_HEAD_DIM,
        "value_head_dim": VALUE_HEAD_DIM,
        "num_key_heads": NUM_KEY_HEADS,
        "num_value_heads": NUM_VALUE_HEADS,
        "gate_lora_rank": VALUE_HEAD_DIM,
        # With use_full_rank_gate the gate comes out of the fused in_proj rather
        # than g_a_proj/g_b_proj.  Either way it stays on the sequence-sharded
        # side of the swap, so this is the harder of the two shapes to get right.
        "use_full_rank_gate": True,
        "gate_lower_bound": -5.0,
    }
    kwargs.update(overrides)
    return KimiDeltaAttention(**kwargs)


def _indices():
    """[1, 1, S, 1] exclusive document ends for the *full* sequence.

    Global on every rank on purpose: under a2a each rank runs the core over the
    whole sequence, so the document boundaries it needs are the global ones.
    """
    row, end = [], 0
    for length in DOCS:
        end += length
        row += [end] * length
    assert len(row) == SEQ_LENGTH, (len(row), SEQ_LENGTH)
    return paddle.to_tensor([row], dtype="int32").reshape([1, 1, SEQ_LENGTH, 1])


@unittest.skipUnless(HAVE_FLA, "paddlefleet_ops fla kernels not available")
class KdaHeadA2ALayerTest(unittest.TestCase):
    """``linear_attn_cp_mode="headwise"`` at the layer level: correct, and really sharded."""

    @classmethod
    def setUpClass(cls):
        if dist.get_world_size() != CONTEXT_PARALLEL:
            raise unittest.SkipTest(
                f"needs exactly {CONTEXT_PARALLEL} ranks, got {dist.get_world_size()}"
            )
        # CP only exists under an EP topology: _create_hcg builds
        # EPHybridCommunicateGroup only when ep_degree > 1, otherwise
        # get_context_parallel_group() is None and cp_size collapses to 1.
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": CONTEXT_PARALLEL,
            "sep_degree": 1,
            "cp_degree": CONTEXT_PARALLEL,
            "ep_degree": CONTEXT_PARALLEL,
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
        initialize_fleet(strategy)
        # Before the first kernel launch: the two arms run the KDA core at
        # different head counts, and the autotune key contains H/HV.  Pinning does
        # not make this test bitwise (the projections still differ), but it keeps
        # the core's contribution out of the error budget.
        pin_fla_autotune()

    def _reference(self, x, indices):
        """Whole sequence, CP group of size 1 -- the ground truth for this rank."""
        layer = _build(
            _config(1, MODE),
            ProcessGroupCollection(
                tp=None, cp=dist.new_group([dist.get_rank()])
            ),
        )
        self.assertEqual(layer.cp_size, 1)
        xf = x.clone()
        xf.stop_gradient = False
        out, _ = layer(hidden_states=xf, attn_mask_startend_row_indices=indices)
        out.sum().backward()
        grads = {
            name: param.grad.clone()
            for name, param in layer.named_parameters()
            if param.grad is not None
        }
        return layer, out.detach(), xf.grad.clone(), grads

    def _cp_arm(self, ref_layer, shard, indices, cp_group):
        """This rank's sequence shard through the real CP group."""
        layer = _build(
            _config(CONTEXT_PARALLEL, MODE),
            ProcessGroupCollection(tp=None, cp=cp_group),
        )
        self.assertEqual(layer.cp_size, CONTEXT_PARALLEL)
        self.assertEqual(layer.config.linear_attn_cp_mode, MODE)
        self.assertEqual(layer.config.cp_balance_mode, CP_BALANCE_MODE)
        self.assertTrue(layer.cp_head_a2a)
        with paddle.no_grad():
            ref_params = dict(ref_layer.named_parameters())
            for name, param in layer.named_parameters():
                param.set_value(ref_params[name])
        # ``build_cp_context`` belongs to the sequence-split path: it slices the
        # global cu_seqlens down to this rank and then clears the global one.  After
        # the head swap this rank holds the *whole* sequence, so that slicing would
        # merge the three documents into one without any error -- the numbers would
        # just be wrong.  Making the call itself fail is a stronger statement than
        # comparing outputs: the mode must not even reach it.
        with patch.object(
            kda_mod,
            "build_cp_context",
            side_effect=AssertionError(
                "build_cp_context must not run under linear_attn_cp_mode="
                f"{MODE!r}: cu_seqlens has to stay global after the head swap"
            ),
        ):
            out, _ = layer(
                hidden_states=shard, attn_mask_startend_row_indices=indices
            )
        out.sum().backward()
        return layer, out.detach()

    def _assert_weight_grads(self, cp_layer, ref_grad, cp_group, cp_rank):
        """Every weight gradient as a PAIR: local partial != ref, CP sum == ref.

        The first half is the one that catches the bug that matters.  A layer that
        ignored the head shard -- every rank computing the full gradient -- would
        pass the CP-sum check after the trainer divides, and would double-count in
        production.  Only the pair pins it down.
        """
        checked = []
        for name, param in cp_layer.named_parameters():
            self.assertIn(
                name,
                ref_grad,
                f"[rank {cp_rank}] reference has no grad for {name}",
            )
            want = ref_grad[name]
            self.assertIsNotNone(
                param.grad,
                f"[rank {cp_rank}] {name}: no gradient on the CP arm",
            )
            if float(want.astype("float32").norm()) == 0.0:
                # Nothing to compare against; a zero reference makes rel-L2
                # meaningless in both directions.
                continue
            partial = param.grad.clone()
            local = rel_l2(partial, want)
            self.assertGreater(
                local,
                WRONG_SHARD_MIN,
                f"[rank {cp_rank}] d{name}: local partial already matches the "
                f"reference (rel-L2 {local:.3e}) -- this rank is computing the full "
                f"gradient, so the trainer-side CP sum will double-count it",
            )
            summed = partial.clone()
            # The layer must never do this itself: CP is a sub-factor of the
            # sharding axis, so the sharding group already contains the CP group
            # and the trainer/optimizer reduces over it.  A second reduction
            # inside the layer would double-count.  This test stands in for the
            # trainer-side one.
            dist.all_reduce(summed, group=cp_group)
            err = rel_l2(summed, want)
            self.assertLess(
                err,
                REL_L2_TOL,
                f"[rank {cp_rank}] d{name}: CP-summed rel-L2 {err:.3e} >= "
                f"{REL_L2_TOL:.1e}",
            )
            checked.append((name, local, err))
        self.assertTrue(
            checked, f"[rank {cp_rank}] no parameter gradient was checked"
        )
        # The zero-reference ``continue`` above is a real hole: a per-head parameter
        # whose gradient came back all-zero (never sliced, never used, or detached)
        # would be skipped and the test would still pass.  These four are the whole
        # point of the head shard, so require that they were genuinely compared.
        names = {name for name, _, _ in checked}
        for required in ("conv1d.weight", "conv1d.bias", "A_log", "dt_bias"):
            self.assertIn(
                required,
                names,
                f"[rank {cp_rank}] {required} was not checked -- its reference "
                f"gradient is zero or missing, so the head slicing is untested",
            )
        return checked

    def test_head_count_must_divide_cp(self):
        """``__init__`` rejects a head count the CP group cannot split evenly.

        Every rank must end up with the same number of heads: the swap is a single
        ``alltoall_single``, which is a fixed-size exchange.  ``NUM_KEY_HEADS - 2``
        (2 heads, cp=4) would silently hand some ranks an empty head block, so the
        guard has to fire in ``__init__`` rather than in the middle of a collective.
        """
        cp_group = ps.get_context_parallel_group()
        with self.assertRaises(ValueError) as caught:
            _build(
                _config(CONTEXT_PARALLEL, MODE),
                ProcessGroupCollection(tp=None, cp=cp_group),
                num_key_heads=2,
            )
        self.assertIn("cp_size", str(caught.exception))

    def test_layer_matches_full_sequence(self):
        paddle.seed(SEED)
        model_parallel_cuda_manual_seed(SEED)
        cp_group = ps.get_context_parallel_group()
        cp_rank = dist.get_rank(cp_group)
        indices = _indices()

        paddle.seed(SEED + 1)
        x = paddle.randn([1, SEQ_LENGTH, HIDDEN_SIZE])
        ref_layer, ref_out, ref_dx, ref_grad = self._reference(x, indices)

        local = SEQ_LENGTH // CONTEXT_PARALLEL
        mine = slice(cp_rank * local, (cp_rank + 1) * local)
        nxt = (cp_rank + 1) % CONTEXT_PARALLEL
        neighbour = slice(nxt * local, (nxt + 1) * local)

        shard = x[:, mine].clone()
        shard.stop_gradient = False
        cp_layer, out = self._cp_arm(ref_layer, shard, indices, cp_group)

        out_err = rel_l2(out, ref_out[:, mine])
        self.assertLess(
            out_err,
            REL_L2_TOL,
            f"[rank {cp_rank}] output rel-L2 {out_err:.3e} >= {REL_L2_TOL:.1e}",
        )
        dx_err = rel_l2(shard.grad, ref_dx[:, mine])
        self.assertLess(
            dx_err,
            REL_L2_TOL,
            f"[rank {cp_rank}] grad_x rel-L2 {dx_err:.3e} >= {REL_L2_TOL:.1e}",
        )

        # Without this an all-zero output, or every rank computing the same
        # sequence, would sail through the two checks above.
        wrong = rel_l2(out, ref_out[:, neighbour])
        self.assertGreater(
            wrong,
            WRONG_SHARD_MIN,
            f"[rank {cp_rank}] output matches rank {nxt}'s shard (rel-L2 "
            f"{wrong:.3e}) -- the sequence partition is not taking effect",
        )

        checked = self._assert_weight_grads(
            cp_layer, ref_grad, cp_group, cp_rank
        )
        if cp_rank == 0:
            worst = max(err for _, _, err in checked)
            print(
                f"  [PASS] linear_attn_cp_mode={MODE} cp={CONTEXT_PARALLEL} "
                f"out={out_err:.3e} "
                f"gx={dx_err:.3e} worst weight-grad={worst:.3e} "
                f"({len(checked)} params)"
            )
            for name, local, err in sorted(checked):
                print(
                    f"    {name:<22} local-vs-ref {local:.3e}  cp-sum {err:.3e}"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
