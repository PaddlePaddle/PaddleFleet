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

"""Backend-free unit tests for the HySparse incremental-decode scoring path.

The integration tests in ``test_hysparse_inference_kv_cache.py`` and
``transformer/test_hysparse_decode_sink_parity.py`` exercise this decode path
end to end, but every case there is guarded by ``_hysparse_backend_or_skip`` /
``_tilelang_backend_or_skip`` and is skipped on the CI runners (non-SM10.x, no
FlashMask/DSA). That leaves the pure-arithmetic pieces the decode path is built
on -- :func:`decode_block_logit`, :func:`select_topk_blocks`,
:func:`_is_incremental_decode`, :meth:`_hy_sparse_decode_block_indices` and the
:class:`DynamicKVCache` SWA-window bookkeeping -- with no CI coverage at all,
even though none of them touch a TileLang/FlashMask/DSA kernel.

These tests call those units directly on the default device (CPU or any CUDA
GPU, no SM10.x needed) so they run everywhere, CI included.
"""

import types
import unittest
from unittest import mock

import numpy as np
import paddle

from paddlefleet.generation.greedy_generator import DynamicKVCache
from paddlefleet.tilelang_ops.hysparse.pipeline import (
    decode_block_logit,
    select_topk_blocks,
)
from paddlefleet.transformer.multi_latent_attention import (
    MQASelfAttention,
    MultiLatentAttention,
    _is_incremental_decode,
)


def _masked_scaled_logits(query, key, valid_range, sm_scale):
    """Eager ``[B, H, 1, S_kv]`` scaled logits with out-of-range keys at -inf.

    Independent re-derivation of the pre-block-max quantity inside
    :func:`decode_block_logit`, used as the oracle for both the LSE and the
    block-max checks.
    """
    _, _, _, d = query.shape
    s_kv = key.shape[1]
    q = query.astype("float32").transpose([0, 2, 1, 3])  # [B, H, 1, D]
    k = key.astype("float32").transpose([0, 2, 1, 3])  # [B, H, S_kv, D]
    logits = paddle.matmul(q, k, transpose_y=True) * sm_scale
    col = paddle.arange(s_kv, dtype="int64").reshape([1, 1, 1, s_kv])
    bos = valid_range[..., 0:1].astype("int64").unsqueeze(1)
    eos = valid_range[..., 1:2].astype("int64").unsqueeze(1)
    mask = (col >= bos) & (col < eos)
    return paddle.where(mask, logits, paddle.full_like(logits, float("-inf")))


def _reference_block_max(masked_logits, valid_range, block_B):
    """Numpy loop oracle for the document-relative block-max scatter.

    Mirrors the block bucketing in :func:`decode_block_logit` (block ``j`` spans
    key columns ``[bos + j*block_B, bos + (j+1)*block_B)``; masked columns stay
    at -inf and never win) with an independent implementation.
    """
    ml = masked_logits.numpy()
    b, h, _, s_kv = ml.shape
    bos = valid_range[..., 0].numpy().astype("int64")  # [B, 1]
    num_blocks = (s_kv + block_B - 1) // block_B
    out = np.full([b, h, 1, num_blocks], -np.inf, dtype="float32")
    for bi in range(b):
        bo = int(bos[bi, 0])
        for c in range(s_kv):
            rel = c - bo
            blk = 0 if rel < 0 else min(rel // block_B, num_blocks - 1)
            for hi in range(h):
                out[bi, hi, 0, blk] = max(out[bi, hi, 0, blk], ml[bi, hi, 0, c])
    return out


class TestDecodeBlockLogit(unittest.TestCase):
    """``decode_block_logit`` — the eager block-scoring row used at decode.

    Pure ``matmul`` / ``logsumexp`` / ``put_along_axis``; no HySparse backend.
    """

    def test_output_shapes(self):
        paddle.seed(0)
        b, h, d, s_kv, block_B = 2, 3, 16, 10, 4
        query = paddle.randn([b, 1, h, d])
        key = paddle.randn([b, s_kv, h, d])
        valid_range = paddle.to_tensor([[[1, 8]], [[0, 10]]], dtype="int32")
        block_logit, lse = decode_block_logit(
            query, key, valid_range, block_B=block_B
        )
        num_blocks = (s_kv + block_B - 1) // block_B
        self.assertEqual(block_logit.shape, [b, h, 1, num_blocks])
        self.assertEqual(lse.shape, [b, 1, h])

    def test_lse_matches_reference(self):
        paddle.seed(1)
        b, h, d, s_kv, block_B = 2, 4, 16, 12, 4
        query = paddle.randn([b, 1, h, d])
        key = paddle.randn([b, s_kv, h, d])
        valid_range = paddle.to_tensor([[[2, 9]], [[0, 12]]], dtype="int32")
        sm_scale = d**-0.5
        _, lse = decode_block_logit(
            query, key, valid_range, sm_scale=sm_scale, block_B=block_B
        )
        masked = _masked_scaled_logits(query, key, valid_range, sm_scale)
        ref_lse = paddle.logsumexp(masked, axis=-1).transpose([0, 2, 1])
        np.testing.assert_allclose(
            lse.numpy(), ref_lse.numpy(), atol=1e-5, rtol=1e-5
        )

    def test_block_logit_is_block_max_of_masked_logits(self):
        paddle.seed(2)
        b, h, d, s_kv, block_B = 2, 3, 16, 11, 4
        query = paddle.randn([b, 1, h, d])
        key = paddle.randn([b, s_kv, h, d])
        valid_range = paddle.to_tensor([[[1, 10]], [[0, 11]]], dtype="int32")
        sm_scale = d**-0.5
        block_logit, _ = decode_block_logit(
            query, key, valid_range, sm_scale=sm_scale, block_B=block_B
        )
        masked = _masked_scaled_logits(query, key, valid_range, sm_scale)
        ref = _reference_block_max(masked, valid_range, block_B)
        np.testing.assert_allclose(
            block_logit.numpy(), ref, atol=1e-5, rtol=1e-5
        )

    def test_block_with_no_valid_key_is_neg_inf(self):
        # bos=0, eos=5, s_kv=12, block_B=4 -> 3 blocks; block 2 spans cols
        # [8, 12), all >= eos, so it holds no valid key and must be -inf.
        paddle.seed(3)
        b, h, d, s_kv, block_B = 1, 2, 8, 12, 4
        query = paddle.randn([b, 1, h, d])
        key = paddle.randn([b, s_kv, h, d])
        valid_range = paddle.to_tensor([[[0, 5]]], dtype="int32")
        block_logit, _ = decode_block_logit(
            query, key, valid_range, block_B=block_B
        )
        self.assertEqual(block_logit.shape, [b, h, 1, 3])
        self.assertTrue(bool(paddle.isinf(block_logit[:, :, :, 2]).all()))
        self.assertTrue(bool((block_logit[:, :, :, 2] < 0).all()))
        # blocks 0 and 1 hold valid keys, so they must be finite.
        self.assertTrue(bool(paddle.isfinite(block_logit[:, :, :, :2]).all()))

    def test_rejects_multi_token_query(self):
        query = paddle.randn([1, 2, 3, 16])
        key = paddle.randn([1, 8, 3, 16])
        valid_range = paddle.to_tensor([[[0, 8]]], dtype="int32")
        with self.assertRaises(ValueError):
            decode_block_logit(query, key, valid_range, block_B=4)


class TestSelectTopkBlocks(unittest.TestCase):
    """``select_topk_blocks`` — the non-differentiable block TopK selector.

    Scores are monotonic in ``block_logit`` (``exp(logit - lse)``), so a fixed
    ``block_logit`` pins the selected ids without any kernel.
    """

    def test_selects_highest_scoring_blocks(self):
        # 4 blocks, block 2 then 3 carry the largest logits.
        block_logit = paddle.to_tensor([[[[1.0, 2.0, 5.0, 3.0]]]])
        lse = paddle.to_tensor([[[6.0]]])
        valid_range = paddle.to_tensor([[[0, 16]]], dtype="int32")
        idx = select_topk_blocks(block_logit, lse, valid_range, 2, block_B=4)
        self.assertEqual(idx.shape, [1, 1, 2])
        self.assertEqual(idx.dtype, paddle.int32)
        np.testing.assert_array_equal(np.sort(idx.numpy(), axis=-1), [[[2, 3]]])

    def test_pads_when_topk_exceeds_num_blocks(self):
        block_logit = paddle.to_tensor([[[[1.0, 4.0]]]])  # 2 blocks
        lse = paddle.to_tensor([[[5.0]]])
        valid_range = paddle.to_tensor([[[0, 8]]], dtype="int32")
        idx = select_topk_blocks(block_logit, lse, valid_range, 4, block_B=4)
        # width stays topk=4; the 2 surplus slots are -1 padding.
        self.assertEqual(idx.shape, [1, 1, 4])
        np.testing.assert_array_equal(idx.numpy(), [[[1, 0, -1, -1]]])

    def test_invalid_blocks_marked_minus_one(self):
        block_logit = paddle.to_tensor([[[[1.0, 4.0]]]])
        lse = paddle.to_tensor([[[5.0]]])
        # eos=3 -> block 1 (starts at col 4) holds no valid key.
        valid_range = paddle.to_tensor([[[0, 3]]], dtype="int32")
        idx = select_topk_blocks(block_logit, lse, valid_range, 2, block_B=4)
        np.testing.assert_array_equal(idx.numpy(), [[[0, -1]]])

    def test_rejects_non_positive_topk(self):
        block_logit = paddle.to_tensor([[[[1.0, 2.0]]]])
        lse = paddle.to_tensor([[[3.0]]])
        valid_range = paddle.to_tensor([[[0, 8]]], dtype="int32")
        with self.assertRaises(ValueError):
            select_topk_blocks(block_logit, lse, valid_range, 0, block_B=4)


class TestIsIncrementalDecode(unittest.TestCase):
    """``_is_incremental_decode`` — pure Python decode/prefill discriminator.

    True only once the layer has written its prefill KV into the cache, so a
    one-token prompt is still a prefill.
    """

    def test_false_without_use_cache(self):
        cache = DynamicKVCache(num_layers=2)
        self.assertFalse(_is_incremental_decode(cache, 0, False))

    def test_false_without_cache_object(self):
        self.assertFalse(_is_incremental_decode(None, 0, True))

    def test_false_without_layer_idx(self):
        cache = DynamicKVCache(num_layers=2)
        self.assertFalse(_is_incremental_decode(cache, None, True))

    def test_false_when_cache_lacks_has_layer_cache(self):
        class NoProtocol:
            pass

        self.assertFalse(_is_incremental_decode(NoProtocol(), 0, True))

    def test_false_before_prefill_true_after(self):
        cache = DynamicKVCache(num_layers=1)
        self.assertFalse(_is_incremental_decode(cache, 0, True))
        cache.update(
            paddle.randn([1, 4, 8]), paddle.randn([1, 4, 8]), layer_idx=0
        )
        self.assertTrue(_is_incremental_decode(cache, 0, True))


class TestHySparseDecodeBlockIndices(unittest.TestCase):
    """``MultiLatentAttention._hy_sparse_decode_block_indices``.

    Only reads ``config.hy_sparse_block_size`` / ``config.hy_sparse_topk`` /
    ``softmax_scale`` and the cached keys, then runs the two pure-paddle
    scoring ops. A lightweight stub ``self`` avoids building a full attention
    layer (which would drag in the TileLang/FlashMask backends).
    """

    def _stub(self, block_B, topk, softmax_scale):
        config = types.SimpleNamespace(
            hy_sparse_block_size=block_B, hy_sparse_topk=topk
        )
        return types.SimpleNamespace(config=config, softmax_scale=softmax_scale)

    def test_shape_and_dtype(self):
        paddle.seed(4)
        b, h, d, s_kv, block_B, topk = 2, 4, 16, 9, 4, 2
        cache = DynamicKVCache(num_layers=1)
        cache.update(
            paddle.randn([b, s_kv, h, d]),
            paddle.randn([b, s_kv, h, d]),
            layer_idx=0,
        )
        stub = self._stub(block_B, topk, d**-0.5)
        query = paddle.randn([b, 1, h, d])
        idx = MultiLatentAttention._hy_sparse_decode_block_indices(
            stub, query, cache, 0
        )
        self.assertEqual(idx.shape, [b, 1, topk])
        self.assertEqual(idx.dtype, paddle.int32)

    def test_matches_pipeline_reference(self):
        # The method must equal decode_block_logit + select_topk_blocks over the
        # cached keys with a column-0-anchored [0, kv_len) valid_range.
        paddle.seed(5)
        b, h, d, s_kv, block_B, topk = 1, 2, 16, 12, 4, 2
        cache = DynamicKVCache(num_layers=1)
        key = paddle.randn([b, s_kv, h, d])
        cache.update(key, paddle.randn([b, s_kv, h, d]), layer_idx=0)
        stub = self._stub(block_B, topk, d**-0.5)
        query = paddle.randn([b, 1, h, d])

        got = MultiLatentAttention._hy_sparse_decode_block_indices(
            stub, query, cache, 0
        )
        valid_range = paddle.to_tensor([[[0, s_kv]]], dtype="int32")
        block_logit, lse = decode_block_logit(
            query, key, valid_range, sm_scale=d**-0.5, block_B=block_B
        )
        expected = select_topk_blocks(
            block_logit, lse, valid_range, topk, block_B
        )
        np.testing.assert_array_equal(got.numpy(), expected.numpy())


class TestDynamicKVCacheDecodeState(unittest.TestCase):
    """The cache bookkeeping the SWA/MQA decode branch consumes.

    The ``is_decode`` branch of ``MQASelfAttention.forward`` itself ends in the
    TileLang ``sliding_window_mqa_attention`` kernel (SM10.x only, covered by
    the backend-gated integration tests). The parts it depends on that *are*
    backend-free -- the full layer reading its own cached keys via
    :meth:`get_layer_kv`, the SWA layer's window truncation, and the
    ``bos = max(0, kv_s - window_size)`` cache-local window it derives -- are
    pinned here so a regression in them fails in CI rather than only on
    Blackwell.
    """

    def test_get_layer_kv_returns_stored_cache(self):
        # greedy_generator.DynamicKVCache.get_layer_kv: what the full layer
        # reads back to re-score blocks at decode.
        cache = DynamicKVCache(num_layers=1)
        k = paddle.randn([1, 5, 2, 8])
        v = paddle.randn([1, 5, 2, 8])
        cache.update(k, v, layer_idx=0)
        got_k, got_v = cache.get_layer_kv(0)
        np.testing.assert_array_equal(got_k.numpy(), k.numpy())
        np.testing.assert_array_equal(got_v.numpy(), v.numpy())

    def test_swa_truncation_and_decode_window(self):
        window = 4
        cache = DynamicKVCache(
            num_layers=1, swa_layers=[True], window_size=window
        )
        prompt_len = 6
        cache.update(
            paddle.randn([1, prompt_len, 8]),
            paddle.randn([1, prompt_len, 8]),
            layer_idx=0,
        )
        # Prefill keeps only the trailing window in the SWA slot, but the
        # absolute sequence length is tracked in full.
        self.assertEqual(cache.k[0].shape[1], window)
        self.assertEqual(cache.get_seq_len(0), prompt_len)

        full_k, _ = cache.update(
            paddle.randn([1, 1, 8]), paddle.randn([1, 1, 8]), layer_idx=0
        )
        # The decode step sees window + 1 tokens (the kept window plus the new
        # one) before the post-update truncation.
        self.assertEqual(full_k.shape[1], window + 1)
        self.assertEqual(cache.k[0].shape[1], window)
        self.assertEqual(cache.get_seq_len(0), prompt_len + 1)

        # The cache-local sliding window the decode branch builds.
        kv_s = full_k.shape[1]
        self.assertEqual(max(0, kv_s - window), 1)

    def test_update_shared_concat_branch(self):
        # greedy_generator.DynamicKVCache.update_shared else/concat branch:
        # the second call must grow the shared latent along the seq axis.
        cache = DynamicKVCache(num_layers=1)
        prefill = paddle.randn([2, 8, 1, 16])
        first = cache.update_shared(prefill, 0)
        self.assertEqual(first.shape, [2, 8, 1, 16])
        step = paddle.randn([2, 1, 1, 16])
        grown = cache.update_shared(step, 0)
        self.assertEqual(grown.shape, [2, 9, 1, 16])
        np.testing.assert_array_equal(grown[:, :8].numpy(), prefill.numpy())


class TestMLAHySparseForwardRouting(unittest.TestCase):
    def _stub(self, seq_len):
        query = paddle.randn([1, seq_len, 1, 3])
        key = paddle.randn([1, seq_len, 1, 3])
        value = paddle.randn([1, seq_len, 1, 1])
        kv_compressed = paddle.randn([1, seq_len, 2])
        k_pos_emb = paddle.randn([1, seq_len, 1, 1])
        core_attention = mock.Mock(
            side_effect=lambda query, *_args, **_kwargs: paddle.zeros(
                [query.shape[0], query.shape[1], 1]
            )
        )
        core_attention.config = types.SimpleNamespace()
        stub = types.SimpleNamespace(
            config=types.SimpleNamespace(
                sequence_parallel=False,
                enable_hy_sparse_attention=True,
            ),
            attn_mask_type=None,
            core_attention=core_attention,
            # HySparse layers are dense/SWA MLA, never the hybrid-MLA
            # non-absorbed MQA path.
            mqa_latent=False,
            recompute_core_attention=False,
            recompute_qkv_up_porj_and_rope=False,
            training=False,
            use_rr_flash_attention=False,
            gated_attention=False,
            use_vha_postmix=False,
            layer_number=0,
            o_proj=lambda x: (x, None),
            get_query_key_value_tensors=lambda *_, **__: (
                query,
                key,
                value,
                None,
                kv_compressed,
                k_pos_emb,
            ),
        )
        return stub

    def _forward(self, stub, shared_kv, cache):
        with (
            mock.patch(
                "paddlefleet.transformer.multi_latent_attention.get_context_parallel_world_size",
                return_value=1,
            ),
            mock.patch(
                "paddlefleet.transformer.transformer_layer.TransformerLayer._log_md5"
            ),
        ):
            return MultiLatentAttention.forward(
                stub,
                paddle.zeros(
                    [1, stub.get_query_key_value_tensors()[0].shape[1], 1]
                ),
                None,
                shared_kv=shared_kv,
                past_key_values=cache,
                layer_idx=0,
                use_cache=True,
            )

    def test_prefill_seeds_cache_and_decode_refreshes_shared_indices(self):
        cache = DynamicKVCache(num_layers=1)
        prefill = self._stub(seq_len=2)
        prefill_indices = paddle.full([1, 2, 1, 1], 7, dtype="int32")
        prefill._hy_sparse_full_attention = mock.Mock(
            return_value=(paddle.zeros([1, 2, 1]), prefill_indices)
        )
        prefill._hy_sparse_decode_block_indices = mock.Mock()
        prefill_shared = []

        self._forward(prefill, prefill_shared, cache)

        self.assertEqual(cache.get_layer_kv(0)[0].shape[1], 2)
        self.assertEqual(prefill_shared[0].shape, [1, 2, 1, 3])
        self.assertIs(prefill_shared[1], prefill_indices)
        prefill._hy_sparse_full_attention.assert_called_once()

        decode = self._stub(seq_len=1)
        decode_indices = paddle.full([1, 1, 1, 1], 9, dtype="int32")
        decode._hy_sparse_full_attention = mock.Mock()
        decode._hy_sparse_decode_block_indices = mock.Mock(
            return_value=decode_indices
        )
        decode_shared = []

        self._forward(decode, decode_shared, cache)

        decode._hy_sparse_full_attention.assert_not_called()
        decode._hy_sparse_decode_block_indices.assert_called_once_with(
            mock.ANY, cache, 0
        )
        self.assertEqual(decode_shared[0].shape, [1, 3, 1, 3])
        self.assertIs(decode_shared[1], decode_indices)


class TestMQAHySparseForwardRouting(unittest.TestCase):
    def _stub(self, seq_len):
        config = types.SimpleNamespace(
            sliding_window=(2,),
            hy_sparse_block_size=2,
            hy_sparse_block_sparse_use_tilelang=True,
            kv_lora_rank=2,
            cp_balance_mode="dualchunk_allgather",
        )
        query = paddle.randn([1, seq_len, 1, 3])
        key = paddle.randn([1, seq_len, 1, 3])
        value = paddle.randn([1, seq_len, 1, 2])
        stub = types.SimpleNamespace(
            is_mqa=True,
            pg_collection=types.SimpleNamespace(tp=None),
            config=config,
            softmax_scale=0.5,
            num_attention_heads_per_partition=1,
            qk_nope_head_dim=1,
            v_head_dim=1,
            kv_b_proj=types.SimpleNamespace(weight=paddle.eye(2)),
            gated_attention=False,
            use_vha_postmix=False,
            o_proj=lambda x: (x, None),
            layer_number=0,
            attn_mask_type=None,
            get_query_key_value_tensors=lambda *_, **__: (
                query,
                key,
                value,
                None,
                None,
                None,
            ),
        )
        return stub

    def _run_forward(self, stub, shared_kv, **kwargs):
        calls = {}

        def sliding(query, key, value, valid_range, **_):
            calls["window_range"] = valid_range
            shape = [query.shape[0], query.shape[1], query.shape[2], 2]
            return paddle.zeros(shape), None

        def sparse(query, key, block_indices, valid_range, **_):
            calls["doc_range"] = valid_range
            shape = [query.shape[0], query.shape[1], query.shape[2], 2]
            return paddle.zeros(shape), None

        patches = (
            mock.patch(
                "paddlefleet.transformer.multi_latent_attention.get_pg_size",
                return_value=1,
            ),
            mock.patch(
                "paddlefleet.transformer.transformer_layer.TransformerLayer._log_md5"
            ),
            mock.patch(
                "paddlefleet.tilelang_ops.hysparse.sliding_window_mqa_attention",
                side_effect=sliding,
            ),
            mock.patch(
                "paddlefleet.tilelang_ops.hysparse.block_sparse_mqa_tl.block_sparse_mqa_attention_tl",
                side_effect=sparse,
            ),
        )
        with patches[0], patches[1], patches[2], patches[3]:
            output, bias = MQASelfAttention.forward(
                stub,
                paddle.zeros(
                    [1, stub.get_query_key_value_tensors()[0].shape[1], 1]
                ),
                None,
                shared_kv=shared_kv,
                **kwargs,
            )
        return output, bias, calls

    def test_decode_uses_cached_window_and_full_shared_history(self):
        stub = self._stub(seq_len=1)
        cache = DynamicKVCache(num_layers=1, swa_layers=[True], window_size=2)
        cache.update(
            paddle.randn([1, 3, 3]), paddle.randn([1, 3, 2]), layer_idx=0
        )
        shared_key = paddle.randn([1, 4, 1, 3])
        block_indices = paddle.zeros([1, 1, 1, 1], dtype="int32")

        with mock.patch(
            "paddlefleet.transformer.multi_latent_attention.get_context_parallel_world_size",
            return_value=1,
        ):
            output, bias, calls = self._run_forward(
                stub,
                [shared_key, block_indices],
                past_key_values=cache,
                layer_idx=0,
                use_cache=True,
            )

        self.assertEqual(output.shape, [1, 1, 1])
        self.assertIsNone(bias)
        np.testing.assert_array_equal(calls["window_range"].numpy(), [[[1, 3]]])
        np.testing.assert_array_equal(calls["doc_range"].numpy(), [[[0, 4]]])

    def test_decode_rejects_chunked_prefill(self):
        stub = self._stub(seq_len=2)
        cache = DynamicKVCache(num_layers=1)
        cache.update(
            paddle.randn([1, 1, 3]), paddle.randn([1, 1, 2]), layer_idx=0
        )
        shared_key = paddle.randn([1, 3, 1, 3])
        block_indices = paddle.zeros([1, 2, 1, 1], dtype="int32")

        with (
            mock.patch(
                "paddlefleet.transformer.multi_latent_attention.get_context_parallel_world_size",
                return_value=1,
            ),
            self.assertRaisesRegex(ValueError, "single query token"),
        ):
            self._run_forward(
                stub,
                [shared_key, block_indices],
                past_key_values=cache,
                layer_idx=0,
                use_cache=True,
            )

    def test_decode_rejects_context_parallel(self):
        stub = self._stub(seq_len=1)
        cache = DynamicKVCache(num_layers=1)
        cache.update(
            paddle.randn([1, 1, 3]), paddle.randn([1, 1, 2]), layer_idx=0
        )
        shared_key = paddle.randn([1, 2, 1, 3])
        block_indices = paddle.zeros([1, 1, 1, 1], dtype="int32")

        with (
            mock.patch(
                "paddlefleet.transformer.multi_latent_attention.get_context_parallel_world_size",
                return_value=2,
            ),
            mock.patch(
                "paddlefleet.transformer.multi_latent_attention.ContextParallelAllGatherOp.apply",
                side_effect=lambda tensor, *_: tensor,
            ),
            self.assertRaisesRegex(ValueError, "doc_valid_range"),
        ):
            self._run_forward(
                stub,
                [shared_key, block_indices],
                past_key_values=cache,
                layer_idx=0,
                use_cache=True,
            )

    def test_prefill_builds_and_scatters_both_valid_ranges(self):
        stub = self._stub(seq_len=2)
        shared_key = paddle.randn([1, 2, 1, 3])
        block_indices = paddle.zeros([1, 2, 1, 1], dtype="int32")

        with (
            mock.patch(
                "paddlefleet.transformer.multi_latent_attention.get_context_parallel_world_size",
                return_value=2,
            ),
            mock.patch(
                "paddlefleet.transformer.multi_latent_attention.ContextParallelAllGatherOp.apply",
                side_effect=lambda tensor, *_: tensor,
            ),
            mock.patch(
                "paddlefleet.transformer.multi_latent_attention.ContextParallelScatterOp.apply",
                side_effect=lambda tensor, *_: tensor,
            ) as scatter,
        ):
            _, _, calls = self._run_forward(stub, [shared_key, block_indices])

        self.assertEqual(scatter.call_count, 2)
        self.assertEqual(calls["window_range"].shape, [1, 2, 2])
        self.assertEqual(calls["doc_range"].shape, [1, 2, 2])

    def test_sparse_layer_requires_shared_block_indices(self):
        stub = self._stub(seq_len=2)
        shared_key = paddle.randn([1, 2, 1, 3])

        with (
            mock.patch(
                "paddlefleet.transformer.multi_latent_attention.get_context_parallel_world_size",
                return_value=1,
            ),
            self.assertRaisesRegex(ValueError, "top-k block indices"),
        ):
            self._run_forward(stub, [shared_key, None])


if __name__ == "__main__":
    unittest.main()
