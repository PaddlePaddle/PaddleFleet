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

"""Adversarial validation (agent A6) of MLA under context parallel with the
``contiguous_allgather`` balance mode.

Gold standard proven here: a CP=2 MLA self-attention layer must equal the CP=1
reference **elementwise** on the forward output and on every parameter gradient
(after the ZeRO-style SUM all-reduce across the CP group), given identical
weights and the identical packed batch.

Key harness detail (the crux that makes this test correct):
``DotProductAttention`` -- the core attention of an MLA layer -- reads its CP
world size from ``paddlefleet.parallel_state.get_context_parallel_world_size()``
(a process-global), NOT from ``pg_collection.cp``. MLA's RoPE scatter reads the
same global at *forward* time. So to compare a non-CP reference against a CP
layer in one process we must:

  1. build + run the reference while the parallel_state CP group is UNSET
     (world size 1 -> plain full-sequence attention, no RoPE scatter), then
  2. install the fleet CP group into parallel_state, build + run the CP layer.

Run (2 GPUs)::

    PYTHONPATH=./third_party/PaddleFleet/src:./third_party/PaddleFormers \
    CUDA_VISIBLE_DEVICES=<a>,<b> python -m paddle.distributed.launch \
        --devices 0,1 --nnodes 1 --master 127.0.0.1:<port> \
        third_party/PaddleFleet/tests/multi_card_tests/transformer/\
test_mla_cp_contiguous_allgather.py
"""

import os
import sys
import types
import unittest

import numpy as np
import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import LayerSpec

import paddlefleet.parallel_state as ps
from paddlefleet.transformer.dot_product_attention import DotProductAttention
from paddlefleet.transformer.dsa_attention import (
    DSAIndexer,
    DSAIndexerSublayersSpec,
)
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.mqa_latent_attention import (
    MQALatentAttention,
    MQALatentAttentionSublayersSpec,
)
from paddlefleet.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSublayersSpec,
)
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import init_method_normal, scaled_init_method_normal

# ``hybrid_mla_utils`` owns the sublayer stubs and the SM100 kernel-availability
# skip shared with the single-card hybrid-MLA suites.
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "single_card_tests",
        "transformer",
    ),
)

import hybrid_mla_utils as U

DTYPE = "bfloat16"

CP_SIZE = None
CP_RANK = None
CP_GROUP = None
HCG = None

# bf16 round-off tolerances. Forward is frequently bit-exact (rank 0) but the
# all-gather + reduce-scatter reorders bf16 reductions, so we allow a small
# relative L2 error. 1.5e-2 is ~4x the largest observed (3.8e-3 param grad,
# 4.4e-3 dH) to stay robust across kernels/driver versions.
FWD_RTOL = 5e-3
GRAD_RTOL = 1.5e-2


def setUpModule():
    global CP_SIZE, CP_RANK, CP_GROUP, HCG
    if dist.get_world_size() < 2:
        raise unittest.SkipTest("MLA context-parallel tests require >= 2 GPUs")
    strategy = fleet.DistributedStrategy()
    world = dist.get_world_size()
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
    HCG = fleet.get_hybrid_communicate_group()
    CP_GROUP = HCG.get_context_parallel_group()
    CP_RANK = CP_GROUP.rank
    CP_SIZE = CP_GROUP.nranks
    # Start with parallel_state CP DISABLED so references build/run non-CP.
    ps._CONTEXT_PARALLEL_GROUP = None


def _cp_disable():
    ps._CONTEXT_PARALLEL_GROUP = None


def _cp_enable():
    ps._CONTEXT_PARALLEL_GROUP = HCG.get_context_parallel_group()


class _TestLinear(nn.Layer):
    def __init__(self, input_size, output_size, dtype=None, **kwargs):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[output_size, input_size],
            dtype=dtype or DTYPE,
            default_initializer=nn.initializer.Normal(std=0.02),
        )

    def forward(self, x):
        return paddle.matmul(x.cast(self.weight.dtype), self.weight.T), None


class _TestRMSNorm(nn.Layer):
    def __init__(self, hidden_size=None, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = self.create_parameter(
            shape=[hidden_size],
            dtype="float32",
            default_initializer=nn.initializer.Constant(1.0),
        )

    def forward(self, x, **kwargs):
        n = x * paddle.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return n * self.weight.cast(x.dtype)


# Fixture label -> production ``TransformerConfig.hybrid_mla_attention`` value.
# The label "mqa" predates the enum and means "latent MQA, no indexer", which
# the enum spells ``"mqa_full_causal"``.
_HYBRID_MLA_ATTENTION = {
    "mha": "mha",
    "mqa": "mqa_full_causal",
    "mqa_dsa": "mqa_dsa",
}


def build_cfg(cp_size, sink=False, attn_mode="mha"):
    H = 4
    c = TransformerConfig(
        num_hidden_layers=1, hidden_size=256, num_attention_heads=H
    )
    c.num_key_value_heads = H
    c.head_dim = 256
    c.experimental_attention_variant = "dsv4_hybrid"
    # ``attn_mode`` is a fixture label, NOT the config value: "mha" -> dense
    # DotProductAttention, "mqa" -> latent MQA over the full per-document causal
    # set, "mqa_dsa" -> latent MQA + DSA indexer. It maps onto the production
    # ``hybrid_mla_attention`` enum below ("mqa" -> "mqa_full_causal").
    c.hybrid_mla_attention = _HYBRID_MLA_ATTENTION[attn_mode]
    c.hybrid_mla_q_lora_rank = 64
    if attn_mode == "mha":
        c.hybrid_mla_kv_lora_rank = 128
        c.hybrid_mla_qk_nope_head_dim = 64
    else:
        # The absorbed query is [kv_lora_rank + qk_rope_head_dim] wide and the
        # absorbed value is [kv_lora_rank]. FlashMLA's sparse kernel only
        # accepts d_qk in {512, 576} with d_v == 512 (mqa_sparse_attn.py:299),
        # so the MQA fixture has to use the production latent width.
        c.hybrid_mla_kv_lora_rank = 512
        c.hybrid_mla_qk_nope_head_dim = 192
    c.hybrid_mla_qk_rope_head_dim = 64
    c.hybrid_mla_v_head_dim = 64
    c.hybrid_mla_num_attention_heads = H
    c.hybrid_mla_num_key_value_heads = H
    # Indexer dims are model-wide; only ``mqa_dsa`` actually builds one.
    c.dsa_index_n_heads = 64
    c.dsa_index_head_dim = 128
    c.dsa_index_topk = 128
    c.dsa_indexer_rotary_interleaved = False
    c.dsa_indexer_use_sparse_loss = True
    c.csa_window_size = 128
    c.add_full_attention_sink_bias = sink
    c.rope_type = "rope"
    c.rope_theta = 10000.0
    c.rotary_base = 10000.0
    c.rotary_interleaved = False
    c.rotary_percent = 1.0
    c.apply_rope_fusion = False
    c.num_nextn_predict_layers = 0
    c.mtp_num_layers = 0
    c.init_method = init_method_normal(0.02)
    c.output_layer_init_method = scaled_init_method_normal(0.02, 1, 1)
    c.rms_norm_eps = 1e-5
    c.layernorm_epsilon = 1e-5
    c.context_parallel_size = cp_size
    c.sequence_parallel = False
    c.tensor_model_parallel_size = 1
    c.cp_balance_mode = "contiguous_allgather"
    c.multi_latent_attention = True
    c.use_bias = False
    c.experimental_dataflow = True
    # FA4 mha sink kernel asserts learnable_sink.dtype == bfloat16, and MLA
    # creates softmax_offset with config.params_dtype. Match production bf16.
    c.params_dtype = "bfloat16"
    return c


def _mqa_core_spec(attn_mode):
    """``core_attention`` spec for latent MQA, per gpt_layer_specs.py:342."""
    indexer = None
    if attn_mode == "mqa_dsa":
        indexer = LayerSpec(
            layer=DSAIndexer,
            sublayers_spec=DSAIndexerSublayersSpec(
                linear_wq_b=U.BiasedLinear,
                linear_wk=U.BiasedLinear,
                k_norm=U.LayerNormStub,
                linear_weights_proj=U.BiasedLinear,
            ),
            extra_kwargs={"is_hybrid_mla_indexer": True},
        )
    return LayerSpec(
        layer=MQALatentAttention,
        sublayers_spec=MQALatentAttentionSublayersSpec(indexer=indexer),
    )


def build_mla(cfg, cp_group, attn_mode="mha"):
    spec = MLASelfAttentionSublayersSpec(
        q_a_layernorm=_TestRMSNorm,
        kv_a_layernorm=_TestRMSNorm,
        q_proj=_TestLinear,
        q_a_proj=_TestLinear,
        q_b_proj=_TestLinear,
        kv_a_proj_with_mqa=_TestLinear,
        kv_b_proj=_TestLinear,
        core_attention=(
            DotProductAttention
            if attn_mode == "mha"
            else _mqa_core_spec(attn_mode)
        ),
        o_proj=_TestLinear,
        gate_proj=None,
    )
    pg = types.SimpleNamespace(tp=None, cp=cp_group)
    layer = MLASelfAttention(
        config=cfg,
        sublayers_spec=spec,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        pg_collection=pg,
    )
    if attn_mode != "mha":
        # ``rotate_activation`` (indexer) and the sparse kernel both require
        # bf16; ``U.BiasedLinear`` / ``U.LayerNormStub`` default to fp32.
        layer.to(dtype=DTYPE)
    return layer


def _rel(a, x):
    a = a.cast("float32").flatten()
    x = x.cast("float32").flatten()
    return float((a - x).norm() / (x.norm() + 1e-12))


def _row_end(doc_lens, seqlen):
    """[1, 1, seqlen, 1] int32 exclusive per-token document end row."""
    out = np.empty([seqlen], dtype="int32")
    pos = 0
    for length in doc_lens:
        end = pos + length
        out[pos : min(end, seqlen)] = end
        pos = end
        if pos >= seqlen:
            break
    if pos < seqlen:
        out[pos:] = seqlen
    return paddle.to_tensor(out).reshape([1, 1, seqlen, 1])


def run_mla_cp(mask_full, seed=2026, sink=False, attn_mode="mha"):
    """Build+run ref (CP off) then CP layer (CP on). Returns a metrics dict.

    Raises are propagated (used by the sink tests to observe NotImplementedError).
    """
    b, sg = 1, mask_full.shape[2]
    sl = sg // CP_SIZE

    _cp_disable()
    paddle.seed(seed)
    ref = build_mla(
        build_cfg(1, sink=sink, attn_mode=attn_mode), None, attn_mode
    )

    paddle.seed(1000)
    hf = paddle.randn([b, sg, 256], dtype=DTYPE)

    ha = hf.clone()
    ha.stop_gradient = False
    oa, _ = ref(ha, None, attn_mask_startend_row_indices=mask_full.clone())
    oa.sum().backward()

    _cp_enable()
    paddle.seed(seed)
    cpl = build_mla(
        build_cfg(CP_SIZE, sink=sink, attn_mode=attn_mode), CP_GROUP, attn_mode
    )
    # ``paddle.seed`` alone does not guarantee identical weights once the two
    # builds differ in kernel-side init order, so copy explicitly.
    cpl.set_state_dict(ref.state_dict())

    s, e = CP_RANK * sl, (CP_RANK + 1) * sl
    hb = hf[:, s:e].clone()
    hb.stop_gradient = False
    ob, _ = cpl(hb, None, attn_mask_startend_row_indices=mask_full.clone())
    ob.sum().backward()

    ref_named = dict(ref.named_parameters())
    param_err = {}
    ref_sq = 0.0
    cp_sq = 0.0
    for n, p in cpl.named_parameters():
        if p.grad is None:
            continue
        gg = p.grad.contiguous()
        if CP_SIZE > 1:
            dist.all_reduce(gg, group=CP_GROUP)
        rp = ref_named[n]
        param_err[n] = None if rp.grad is None else _rel(gg, rp.grad)
        cp_sq += float(gg.cast("float32").square().sum())
        if rp.grad is not None:
            ref_sq += float(rp.grad.cast("float32").square().sum())

    per_pos = (
        (ob.cast("float32") - oa[:, s:e].cast("float32"))
        .abs()
        .max(axis=-1)
        .flatten()
        .tolist()
    )
    return {
        "fwd": _rel(ob, oa[:, s:e]),
        "dH": _rel(hb.grad, ha.grad[:, s:e]),
        "param_err": param_err,
        "per_pos": per_pos,
        "ref_gnorm": ref_sq**0.5,
        "cp_gnorm": cp_sq**0.5,
    }


class TestMLAContiguousAllgatherCP(unittest.TestCase):
    # ---- Item 2: single-doc CP=2 vs CP=1 output + every param grad ----
    def test_2_single_doc_equivalence(self):
        r = run_mla_cp(_row_end([128], 128))
        self.assertLess(r["fwd"], FWD_RTOL, "forward exceeds bf16 tol")
        self.assertLess(r["dH"], GRAD_RTOL, "dH exceeds bf16 tol")
        for n, v in r["param_err"].items():
            self.assertIsNotNone(v, f"ref grad missing for {n}")
            self.assertLess(v, GRAD_RTOL, f"param grad {n} exceeds bf16 tol")

    # ---- Item 3: boundary coverage (per-position error) ----
    def test_3_boundary_doc_spanning_rank_boundary(self):
        # doc0 = [0,96) spans the rank boundary at 64; doc1 = [96,128) inside r1.
        r = run_mla_cp(_row_end([96, 32], 128))
        self.assertLess(r["fwd"], FWD_RTOL)
        self.assertLess(max(r["per_pos"]), 5e-2)

    def test_3_boundary_doc_inside_rank1_and_first_query(self):
        # doc0 = [0,64) == rank0 exactly; doc1 = [64,128) entirely inside rank1.
        # Position 64 is the first query of rank1 AND first token of doc1.
        r = run_mla_cp(_row_end([64, 64], 128))
        self.assertLess(r["fwd"], FWD_RTOL)
        self.assertLess(max(r["per_pos"]), 5e-2)

    def test_3_boundary_many_docs(self):
        r = run_mla_cp(_row_end([16, 48, 40, 24], 128))
        self.assertLess(r["fwd"], FWD_RTOL)
        self.assertLess(max(r["per_pos"]), 5e-2)

    # ---- Item 4: learnable attention sink + CP under the default FA flag ----
    def test_4_sink_cp_default_fa_flag(self):
        flag = paddle.get_flags("FLAGS_flash_attn_version")[
            "FLAGS_flash_attn_version"
        ]
        raised = None
        try:
            run_mla_cp(_row_end([128], 128), sink=True)
        except (RuntimeError, NotImplementedError) as ex:
            raised = str(ex)
        if int(flag) not in (3, 4):
            # mha sink on FA!=4 must NOT silently drop the sink: the layer
            # raises a clear construction-time guard (RuntimeError) instead.
            self.assertIsNotNone(
                raised,
                "sink+CP on FA!=4 silently succeeded -> sink likely dropped",
            )
            self.assertIn("FLAGS_flash_attn_version", raised)

    # ---- Item 5: sink + CP with FLAGS_flash_attn_version=4 ----
    def test_5_sink_cp_fa4(self):
        from paddlefleet_ops import is_flash_mask_available

        if not is_flash_mask_available():
            self.skipTest("FA4 cute backend unavailable (needs sm100)")
        old = paddle.get_flags("FLAGS_flash_attn_version")[
            "FLAGS_flash_attn_version"
        ]
        paddle.set_flags({"FLAGS_flash_attn_version": 4})
        try:
            # CP+sink must match the CP=1 sink reference elementwise.
            r_sink = run_mla_cp(_row_end([128], 128), sink=True)
        finally:
            paddle.set_flags({"FLAGS_flash_attn_version": old})
        self.assertLess(
            r_sink["fwd"],
            FWD_RTOL,
            "FA4 sink+CP output must match CP=1 sink reference",
        )
        for n, v in r_sink["param_err"].items():
            self.assertLess(v, GRAD_RTOL, f"FA4 sink+CP grad {n} off")

    # ---- Item 7: global grad-norm magnitude CP=1 vs CP=2 ----
    def test_7_grad_norm_ratio(self):
        r = run_mla_cp(_row_end([128], 128))
        ratio = r["cp_gnorm"] / (r["ref_gnorm"] + 1e-12)
        # A loss-scaling / averaging bug would show a ratio of ~CP_SIZE or
        # ~1/CP_SIZE. Correct behaviour is ratio ~= 1.
        self.assertAlmostEqual(
            ratio, 1.0, delta=0.05, msg="grad-norm ratio deviates from 1"
        )

    # ---- Item 6: latent MQA (+DSA) CP equivalence at the MLA level ----
    #
    # This used to assert ``NotImplementedError``. MQA now implements
    # ``contiguous_allgather`` CP, so the same slot must prove equivalence
    # instead. The core-attention-only equivalence lives in
    # ``test_mqa_dsa_cp.py``; what is new here is the *whole* MLA layer: the
    # shared RoPE CP scatter (multi_latent_attention.py:1485-1545), the runtime
    # absorption (:1913-1939), ``o_proj``, and every parameter gradient after
    # the CP all-reduce.
    def _check_mqa(self, attn_mode):
        # ``mqa_full_causal`` (and the phase-2 warmup) accept dense FA4 only --
        # ``MQALatentAttention._assert_dense_fa4`` raises otherwise -- and a bare
        # launcher process leaves the flag at the image default 2. Pin it to the
        # value ``TrainingArguments.__post_init__`` derives on these SM100 boxes.
        # Call-site scoped, not module scoped: ``test_4`` asserts on the refusal
        # the *default* flag produces.
        with U._flash_attn_version(4):
            r = run_mla_cp(_row_end([200, 150, 162], 512), attn_mode=attn_mode)
        self.assertLess(r["fwd"], FWD_RTOL, f"{attn_mode}: forward")
        self.assertLess(r["dH"], GRAD_RTOL, f"{attn_mode}: dH")
        for n, v in r["param_err"].items():
            self.assertIsNotNone(v, f"{attn_mode}: ref grad missing for {n}")
            self.assertLess(v, GRAD_RTOL, f"{attn_mode}: param grad {n}")
        print(
            f"[mla-cp{CP_SIZE} rank{CP_RANK}] {attn_mode}: fwd={r['fwd']:.2e} "
            f"dH={r['dH']:.2e} "
            f"param_max={max(r['param_err'].values(), default=0.0):.2e} "
            f"per_pos_max={max(r['per_pos']):.3e}",
            flush=True,
        )
        return r

    @U._GPU
    def test_6_mqa_dense_cp_equivalence(self):
        """Absorbed MQA over the full per-document causal set, under CP.

        The documents [0,200) [200,350) [350,512) all straddle the rank
        boundaries for CP=2 (256) and CP=4 (128/256/384), so a per-rank index
        table -- one clipped at ``q - position_offset`` instead of at the global
        ``q`` -- drops the prefix owned by lower ranks and shows up as an O(1)
        forward error on rank > 0.
        """
        r = self._check_mqa("mqa")
        self.assertTrue(
            any(n.endswith("kv_b_proj.weight") for n in r["param_err"]),
            "kv_b_proj must still receive a gradient through the absorption",
        )

    @U._GPU
    def test_6b_mqa_dsa_cp_equivalence(self):
        """Same, with the DSA indexer selecting the columns.

        Additionally covers the indexer's own CP path end to end: its K is
        all-gathered to the global length while its Q stays sharded and takes
        the ``position_offset`` RoPE slice (dsa_attention.py:426-511).
        """
        r = self._check_mqa("mqa_dsa")
        self.assertTrue(
            any(".indexer." in n for n in r["param_err"]),
            "the indexer parameters must receive gradients under CP",
        )

    def test_6c_mqa_rejects_other_cp_balance_modes(self):
        """Only ``contiguous_allgather`` is implemented, same as HCA
        (``dsv4_hybrid_attention.py:607-611``)."""
        _cp_enable()
        self.assertGreater(ps.get_context_parallel_world_size(), 1)
        for mode in ("p2p", "zigzag", "dualchunk_allgather", None):
            with self.subTest(cp_balance_mode=mode):
                cfg = build_cfg(CP_SIZE, attn_mode="mqa")
                cfg.cp_balance_mode = mode
                with self.assertRaises(NotImplementedError):
                    build_mla(cfg, CP_GROUP, "mqa")


if __name__ == "__main__":
    unittest.main(verbosity=2)
