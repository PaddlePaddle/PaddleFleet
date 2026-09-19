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

"""``csa_hca_use_flashmask`` under context parallelism.

The CP HCA path keeps its raw KV in a ``prepend_prev_window`` buffer rather than
a full all-gather, so the FlashMask column bounds have to be rebased onto that
shorter buffer and the row bounds localised to this rank's query slice. These
tests check that against the non-CP full-sequence reference:

  - CP + FlashMask forward matches the slice of the non-CP FlashMask forward;
  - the same for dQ / dKV / dX and the compressor parameter grads;
  - CP + FlashMask also matches CP + ``csa_sparse_attn``, so the switch is
    numerically transparent on every rank rather than only in aggregate.

Requires FA4 (``q_head_dim == v_head_dim``); skips otherwise.

Run with:
    python -m paddle.distributed.launch --gpus 0,1 \
        tests/multi_card_tests/transformer/test_hca_flashmask_attn_cp.py
"""

import types
import unittest

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddlefleet.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)
from paddlefleet.transformer.csa_attention import (
    CompressedSparseAttention,
    CompressedSparseAttentionSublayersSpec,
    Compressor,
    CompressorSublayersSpec,
    CSADocMaskMetadata,
    CSAIndexer,
    CSAIndexerSublayersSpec,
)

CP_SIZE = None
CP_RANK = None
CP_GROUP = None

DTYPE = "bfloat16"
HEAD_DIM = 64
HIDDEN_SIZE = 256
NP_HEADS = 8
Q_LORA_RANK = 64
RATIO = 128
WINDOW_SIZE = 64
SEQLEN = 512
COS_THRESHOLD = 0.99


def setUpModule():
    global CP_SIZE, CP_RANK, CP_GROUP
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


class _TestLinear(nn.Layer):
    def __init__(self, input_size, output_size, dtype=None, **kwargs):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[output_size, input_size],
            dtype=dtype or DTYPE,
            default_initializer=nn.initializer.Normal(std=0.02),
        )

    def forward(self, x):
        return paddle.matmul(x, self.weight.T), None


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
        normed = x * paddle.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return normed * self.weight.cast(x.dtype)


def _build_config():
    return types.SimpleNamespace(
        num_attention_heads=NP_HEADS,
        v_head_dim=HEAD_DIM,
        hidden_size=HIDDEN_SIZE,
        q_lora_rank=Q_LORA_RANK,
        qk_pos_emb_head_dim=32,
        csa_window_size=WINDOW_SIZE,
        csa_compress_ratios=[RATIO],
        csa_dense_mode=True,
        dsa_index_n_heads=16,
        dsa_index_head_dim=32,
        dsa_index_topk=16,
        dsa_indexer_loss_coeff=0.0,
        dsa_indexer_use_sparse_loss=False,
        csa_indexer_backend="unfused",
        csa_sparse_attn_backend="unfused",
        csa_hca_use_flashmask=False,
        init_method=None,
        init_method_std=0.02,
        layernorm_epsilon=1e-5,
        num_hidden_layers=1,
    )


def _build_csa(config):
    rope = RotaryEmbedding(32, rotary_percent=1.0, rotary_base=160000)
    comp_spec = CompressorSublayersSpec(
        linear_wkv=_TestLinear,
        linear_wgate=_TestLinear,
        norm=_TestRMSNorm,
    )
    attn_spec = CompressedSparseAttentionSublayersSpec(
        compressor=LayerSpec(layer=Compressor, sublayers_spec=comp_spec),
        indexer=LayerSpec(
            layer=CSAIndexer,
            sublayers_spec=CSAIndexerSublayersSpec(
                linear_wq_b=_TestLinear,
                linear_weights_proj=_TestLinear,
                compressor=LayerSpec(
                    layer=Compressor, sublayers_spec=comp_spec
                ),
            ),
        ),
    )
    return CompressedSparseAttention(
        config=config,
        sublayers_spec=attn_spec,
        layer_number=1,
        attn_mask_type=None,
        attention_type="self",
        k_channels=HEAD_DIM,
        v_channels=HEAD_DIM,
        compress_ratio=RATIO,
        rotary_pos_emb=rope,
    )


def _cosine(actual, expected):
    a = actual.cast("float32").flatten()
    b = expected.cast("float32").flatten()
    return (a * b).sum().item() / (a.norm().item() * b.norm().item() + 1e-30)


def _startend(doc_lens):
    rows, cum = [], 0
    for length in doc_lens:
        cum += length
        rows += [cum] * length
    rows += [cum] * (SEQLEN - len(rows))
    return paddle.to_tensor(rows, dtype="int32").reshape([1, 1, SEQLEN, 1])


def _fa4_available() -> bool:
    from paddlefleet_ops.flash_mask_facade import get_fa_version

    bounds = paddle.zeros([1, 1, 8, 2], dtype="int32")
    return get_fa_version(HEAD_DIM, HEAD_DIM, bounds) == 4


class TestHCAFlashmaskCP(unittest.TestCase):
    """CP FlashMask HCA vs the non-CP reference and vs CP sparse attention."""

    @classmethod
    def setUpClass(cls):
        if not _fa4_available():
            raise unittest.SkipTest("csa_hca_use_flashmask needs FA4")

    def _make_inputs(self):
        paddle.seed(1000)
        return {
            "query": paddle.randn([1, SEQLEN, NP_HEADS, HEAD_DIM], dtype=DTYPE),
            "key": paddle.randn([1, SEQLEN, 1, HEAD_DIM], dtype=DTYPE),
            "x": paddle.randn([1, SEQLEN, HIDDEN_SIZE], dtype=DTYPE),
            "qr": paddle.randn([1, SEQLEN, Q_LORA_RANK], dtype=DTYPE),
        }

    def _run(self, csa, full_inputs, *, sliced, use_flashmask, docmask_meta):
        csa.use_hca_flashmask = use_flashmask and csa.is_hca_layer
        csa.clear_gradients()
        sq_local = SEQLEN // CP_SIZE
        lo, hi = CP_RANK * sq_local, (CP_RANK + 1) * sq_local
        tensors = {}
        for name, ref in full_inputs.items():
            t = (ref[:, lo:hi] if sliced else ref).clone()
            t.stop_gradient = False
            tensors[name] = t
        out = csa.forward(
            tensors["query"],
            tensors["key"],
            tensors["key"],
            None,
            x=tensors["x"],
            qr=tensors["qr"],
            docmask_meta=docmask_meta,
        )
        out.sum().backward()
        if sliced:
            for p in csa.parameters():
                if p.grad is not None:
                    g = p.grad.contiguous()
                    dist.all_reduce(g, group=CP_GROUP)
                    paddle.assign(g, p.grad)
        grads = {
            name: (None if t.grad is None else t.grad.clone())
            for name, t in tensors.items()
        }
        grads["compressor_wkv"] = csa.compressor.linear_wkv.weight.grad.clone()
        return out, grads

    def _build_pair(self):
        config = _build_config()
        paddle.seed(2026)
        csa_ref = _build_csa(config)
        paddle.seed(2026)
        csa_cp = _build_csa(config)
        for csa, enabled in ((csa_ref, False), (csa_cp, True)):
            paddle.seed(7)
            paddle.assign(
                paddle.randn([NP_HEADS], dtype="float32"), csa.attn_sink
            )
            csa.cp_group = CP_GROUP if enabled else None
            csa.cp_size = CP_SIZE if enabled else 1
            csa.cp_rank = CP_RANK if enabled else 0
            csa.cp_enabled = enabled
        return csa_ref, csa_cp

    def _compare(self, docmask_meta):
        sq_local = SEQLEN // CP_SIZE
        lo, hi = CP_RANK * sq_local, (CP_RANK + 1) * sq_local
        inputs = self._make_inputs()
        csa_ref, csa_cp = self._build_pair()

        out_ref, grads_ref = self._run(
            csa_ref,
            inputs,
            sliced=False,
            use_flashmask=True,
            docmask_meta=docmask_meta,
        )
        out_cp, grads_cp = self._run(
            csa_cp,
            inputs,
            sliced=True,
            use_flashmask=True,
            docmask_meta=docmask_meta,
        )
        out_cp_sparse, grads_cp_sparse = self._run(
            csa_cp,
            inputs,
            sliced=True,
            use_flashmask=False,
            docmask_meta=docmask_meta,
        )

        # CP FlashMask vs non-CP FlashMask (this rank's slice).
        self.assertGreater(
            _cosine(out_cp, out_ref[:, lo:hi]), COS_THRESHOLD, "fwd vs non-CP"
        )
        for name in ("query", "key", "x"):
            self.assertGreater(
                _cosine(grads_cp[name], grads_ref[name][:, lo:hi]),
                COS_THRESHOLD,
                f"{name} grad vs non-CP",
            )
        self.assertGreater(
            _cosine(grads_cp["compressor_wkv"], grads_ref["compressor_wkv"]),
            COS_THRESHOLD,
            "compressor grad vs non-CP",
        )

        # CP FlashMask vs CP sparse attention, on this rank alone.
        self.assertGreater(
            _cosine(out_cp, out_cp_sparse), COS_THRESHOLD, "fwd vs CP sparse"
        )
        for name in ("query", "key", "x", "compressor_wkv"):
            self.assertGreater(
                _cosine(grads_cp[name], grads_cp_sparse[name]),
                COS_THRESHOLD,
                f"{name} grad vs CP sparse",
            )

    def test_causal_only(self):
        self._compare(None)

    def test_document_mask(self):
        meta = CSADocMaskMetadata.build(
            RATIO, 1, SEQLEN, _startend([200, 61, 128, 100]), dense_mode=True
        )
        self._compare(meta)

    def test_document_mask_with_padding(self):
        meta = CSADocMaskMetadata.build(
            RATIO, 1, SEQLEN, _startend([130, 200, 130]), dense_mode=True
        )
        self.assertLess(meta.actual_n_compressed, SEQLEN // RATIO)
        self._compare(meta)


if __name__ == "__main__":
    unittest.main()
