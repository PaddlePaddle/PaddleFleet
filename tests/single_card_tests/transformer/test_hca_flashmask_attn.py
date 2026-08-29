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

"""``csa_hca_use_flashmask`` computes the same HCA attention as ``csa_sparse_attn``.

Runs one ``CompressedSparseAttention`` HCA layer twice -- same weights, same
inputs, only ``use_hca_flashmask`` flipped -- and compares forward output and
the dQ / dKV / dX / d(attn_sink) gradients against the ``unfused``
(pure-Paddle einsum) sparse-attention reference.

Requires FA4: the CSA geometry is ``q_head_dim == v_head_dim``, which only the
FA4 cute-DSL kernels take, so the test skips itself where
``get_fa_version`` does not answer 4.

Run with:
    python -m pytest tests/single_card_tests/transformer/test_hca_flashmask_attn.py
"""

import types
import unittest

import paddle
from paddle import nn
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

DTYPE = "bfloat16"
HEAD_DIM = 64
HIDDEN_SIZE = 256
NP_HEADS = 8
Q_LORA_RANK = 64
RATIO = 128
WINDOW_SIZE = 64
SEQLEN = 512
COS_THRESHOLD = 0.999


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
    indexer_sublayers = CSAIndexerSublayersSpec(
        linear_wq_b=_TestLinear,
        linear_weights_proj=_TestLinear,
        compressor=LayerSpec(layer=Compressor, sublayers_spec=comp_spec),
    )
    attn_spec = CompressedSparseAttentionSublayersSpec(
        compressor=LayerSpec(layer=Compressor, sublayers_spec=comp_spec),
        indexer=LayerSpec(layer=CSAIndexer, sublayers_spec=indexer_sublayers),
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


@unittest.skipUnless(
    paddle.is_compiled_with_cuda(), "CSA attention kernels need CUDA"
)
class TestHCAFlashmaskAttention(unittest.TestCase):
    """FlashMask HCA output and gradients match the unfused sparse reference."""

    @classmethod
    def setUpClass(cls):
        paddle.set_device("gpu")
        if not _fa4_available():
            raise unittest.SkipTest(
                "csa_hca_use_flashmask needs FA4 (head dim "
                f"{HEAD_DIM}/{HEAD_DIM} did not resolve to FA4)"
            )

    def _run(self, csa, inputs, use_flashmask, docmask_meta):
        csa.use_hca_flashmask = use_flashmask
        csa.clear_gradients()
        tensors = {}
        for name, ref in inputs.items():
            t = ref.clone()
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
        grads = {name: t.grad for name, t in tensors.items()}
        grads["attn_sink"] = csa.attn_sink.grad.clone()
        return out, grads

    def _compare(self, docmask_meta):
        config = _build_config()
        paddle.seed(2026)
        csa = _build_csa(config)
        self.assertTrue(csa.is_hca_layer)
        self.assertIsNone(csa.indexer)
        # A zero sink would hide any error in the sink's own contribution.
        paddle.assign(paddle.randn([NP_HEADS], dtype="float32"), csa.attn_sink)

        paddle.seed(1000)
        inputs = {
            "query": paddle.randn([1, SEQLEN, NP_HEADS, HEAD_DIM], dtype=DTYPE),
            "key": paddle.randn([1, SEQLEN, 1, HEAD_DIM], dtype=DTYPE),
            "x": paddle.randn([1, SEQLEN, HIDDEN_SIZE], dtype=DTYPE),
            "qr": paddle.randn([1, SEQLEN, Q_LORA_RANK], dtype=DTYPE),
        }

        out_ref, grads_ref = self._run(csa, inputs, False, docmask_meta)
        out_fm, grads_fm = self._run(csa, inputs, True, docmask_meta)

        self.assertGreater(
            _cosine(out_fm, out_ref), COS_THRESHOLD, "forward output"
        )
        for name, grad_ref in grads_ref.items():
            if grad_ref is None:
                # ``qr`` only feeds the Indexer, which an HCA layer does not
                # have, so neither side produces a gradient for it.
                self.assertIsNone(grads_fm[name], f"{name} grad appeared")
                continue
            self.assertIsNotNone(grads_fm[name], f"{name} grad is None")
            self.assertGreater(
                _cosine(grads_fm[name], grad_ref),
                COS_THRESHOLD,
                f"{name} grad",
            )

    def test_causal_only(self):
        self._compare(None)

    def test_document_mask(self):
        startend = _startend([200, 61, 128, 100])
        meta = CSADocMaskMetadata.build(
            RATIO, 1, SEQLEN, startend, dense_mode=True
        )
        self._compare(meta)

    def test_document_mask_with_padding(self):
        startend = _startend([130, 200, 130])
        meta = CSADocMaskMetadata.build(
            RATIO, 1, SEQLEN, startend, dense_mode=True
        )
        self.assertLess(meta.actual_n_compressed, SEQLEN // RATIO)
        self._compare(meta)

    def test_flag_is_a_no_op_without_hca(self):
        """The switch must not reroute a window-only (ratio 0) layer."""
        config = _build_config()
        config.csa_hca_use_flashmask = True
        config.csa_compress_ratios = [0]
        paddle.seed(2026)
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
        csa = CompressedSparseAttention(
            config=config,
            sublayers_spec=attn_spec,
            layer_number=1,
            attn_mask_type=None,
            attention_type="self",
            k_channels=HEAD_DIM,
            v_channels=HEAD_DIM,
            compress_ratio=0,
            rotary_pos_emb=rope,
        )
        self.assertFalse(csa.is_hca_layer)
        self.assertFalse(csa.use_hca_flashmask)


if __name__ == "__main__":
    unittest.main()
