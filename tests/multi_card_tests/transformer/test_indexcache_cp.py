# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Two-rank CP regression for IndexCache F/S Replay ordering and gradients.

Run with:
    PYTHONPATH=.:./src python -m paddle.distributed.launch --devices 0,1 \
        tests/multi_card_tests/transformer/test_indexcache_cp.py
"""

import unittest
from types import SimpleNamespace

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle.distributed import fleet

from paddlefleet.transformer.cp_utils import all_gather_cp
from paddlefleet.transformer.csa_attention import CompressedSparseAttention
from paddlefleet.transformer.indexcache_state import apply_stop_gradient_mask


CP_SIZE = None
CP_RANK = None
CP_GROUP = None


def setUpModule():
    global CP_SIZE, CP_RANK, CP_GROUP
    world_size = dist.get_world_size()
    if world_size != 2:
        raise unittest.SkipTest("IndexCache CP regression requires 2 ranks")

    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": world_size,
        "sep_degree": 1,
        "cp_degree": world_size,
        "ep_degree": world_size,
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


class _CollectiveCompressor:
    def __call__(self, x, cp_group=None, docmask_meta=None):
        del docmask_meta
        local_compressed = x[:, ::4, :1].contiguous()
        return all_gather_cp(local_compressed, dim=1, group=cp_group)


class _IndexCacheCPHarness:
    def __init__(self, action, producer_scale=None):
        self.action = action
        self.config = SimpleNamespace()
        self.cp_enabled = True
        self.cp_size = CP_SIZE
        self.cp_rank = CP_RANK
        self.cp_group = CP_GROUP
        self.tp_group = None
        self.is_hca_layer = False
        self.compressor = _CollectiveCompressor()
        self.compress_ratio = 4
        self.indexer = object()
        self.window_size = 2
        self.indexer_backend = "unfused"
        self.sparse_attn_backend = "unfused"
        self.attn_sink = paddle.zeros([1], dtype="float32")
        self.softmax_scale = 1.0
        self.training = True
        self.layer_number = 1 if action == "F" else 2
        self.producer_scale = producer_scale
        self.native_topk = None
        self.replay_topk = None
        self.replay_input = None
        self.attention_topk = None
        self.kv_full_shape = None

    def _indexcache_next_c4_action(self):
        return 0 if self.action == "F" else 1, self.action, "FS"

    @staticmethod
    def _indexcache_has_future_served_layer(pattern, c4_ordinal):
        return pattern == "FS" and c4_ordinal == 0

    @staticmethod
    def _indexcache_served_count(pattern, c4_ordinal):
        assert pattern == "FS" and c4_ordinal == 0
        return 1

    @staticmethod
    def _indexcache_scaled_loss_coeff(served_count):
        assert served_count == 1
        return 1.0

    def _compute_indexer_compressed_topk_idxs_cp(
        self,
        query,
        x,
        qr,
        compressed_kv_global,
        n_compressed_global,
        offset,
        q_positions,
        position_offset,
        **_kwargs,
    ):
        del x, qr, compressed_kv_global, n_compressed_global
        del q_positions, position_offset
        assert self.action == "F"
        batch, seq = query.shape[:2]
        self.native_topk = paddle.full(
            [batch, seq, 2],
            offset + CP_RANK,
            dtype="int32",
        )
        logits = paddle.stack(
            [self.producer_scale, -self.producer_scale], axis=-1
        ).reshape([1, 1, 2])
        topk_probs = F.softmax(logits, axis=-1).expand([batch, seq, 2])
        loss_state = SimpleNamespace(
            topk_probs=topk_probs,
            target=paddle.full_like(topk_probs, 0.5),
            indexer_loss_coeff=1.0,
        )
        return self.native_topk, None, loss_state, 1.0, True, 1

    def _indexcache_cache_topk(
        self,
        compress_topk_idxs,
        _c4_ordinal,
        _pattern,
        tilelang_indexer_loss_state,
        *_args,
    ):
        assert bool(paddle.equal_all(compress_topk_idxs, self.native_topk))
        topk_probs = tilelang_indexer_loss_state.topk_probs
        return apply_stop_gradient_mask(
            (
                compress_topk_idxs,
                paddle.zeros([1], dtype="float32"),
                paddle.zeros([1], dtype="float32"),
                paddle.zeros([1], dtype="float32"),
                paddle.zeros([1], dtype="int32"),
                topk_probs,
                paddle.full([1], self.layer_number, dtype="int64"),
                paddle.full([1], 1, dtype="int64"),
            )
        )

    def _indexcache_reuse_topk(
        self,
        _batch,
        _seq,
        _c4_ordinal,
        _pattern,
        indexcache_state=None,
    ):
        assert self.action == "S"
        self.native_topk = indexcache_state[0]
        return self.native_topk

    def _indexcache_served_distill_state(
        self,
        query,
        _compressed_kv,
        _c4_ordinal,
        _pattern,
        indexcache_state=None,
        **_kwargs,
    ):
        batch, seq = query.shape[:2]
        target = paddle.zeros([batch, seq, 2], dtype="float32")
        target[..., 1] = 1.0
        return (
            indexcache_state[5],
            target,
            1.0,
            None,
            None,
            self.layer_number,
            1,
            False,
        )

    def _postprocess_indexer_replay(
        self,
        compress_topk_idxs,
        _n_compressed,
        offset,
        **_kwargs,
    ):
        self.replay_input = compress_topk_idxs
        self.replay_topk = paddle.full(
            compress_topk_idxs.shape,
            offset + CP_SIZE + CP_RANK,
            dtype="int32",
        )
        return self.replay_topk

    def compressed_sparse_attn(
        self,
        query,
        kv_full,
        _attn_sink,
        topk_idxs,
        _softmax_scale,
        **_kwargs,
    ):
        self.attention_topk = topk_idxs[..., -2:]
        self.kv_full_shape = list(kv_full.shape)
        # Keep the CP all-gather in the backward graph without changing values.
        return query.cast("float32").sum(axis=-1) + kv_full.mean() * 0.0

    @staticmethod
    def _indexcache_attach_indexer_loss(
        output,
        _indexer_loss,
        _tilelang_indexer_loss_state,
        _producer_loss_fused,
    ):
        return output

    @staticmethod
    def _indexcache_clear_cached_state():
        return None


class TestIndexCacheContextParallel(unittest.TestCase):
    def test_f_s_replay_preserves_state_and_producer_gradient(self):
        batch, seq = 1, 8
        query = paddle.randn([batch, seq, 1, 1], dtype="float32")
        key = paddle.randn([batch, seq, 1, 1], dtype="float32")
        x = paddle.randn([batch, seq, 1], dtype="float32")
        qr = paddle.randn([batch, seq, 1], dtype="float32")
        for tensor in (query, key, x, qr):
            tensor.stop_gradient = False

        producer_scale = paddle.to_tensor([0.25], dtype="float32")
        producer_scale.stop_gradient = False
        producer = _IndexCacheCPHarness("F", producer_scale)
        producer_output, state = CompressedSparseAttention._forward_cp(
            producer,
            query,
            key,
            x,
            qr,
        )

        self.assertIs(producer.replay_input, producer.native_topk)
        self.assertTrue(paddle.equal_all(state[0], producer.native_topk).item())
        self.assertTrue(
            paddle.equal_all(producer.attention_topk, producer.replay_topk).item()
        )

        served = _IndexCacheCPHarness("S")
        served_output, cleared_state = CompressedSparseAttention._forward_cp(
            served,
            query,
            key,
            x,
            qr,
            indexcache_state=state,
        )
        self.assertIsNone(cleared_state)
        self.assertIs(served.replay_input, served.native_topk)
        self.assertTrue(paddle.equal_all(served.native_topk, state[0]).item())
        self.assertTrue(
            paddle.equal_all(served.attention_topk, served.replay_topk).item()
        )
        expected_kv_length = seq * CP_SIZE + (seq // 4) * CP_SIZE
        self.assertEqual(producer.kv_full_shape[1], expected_kv_length)
        self.assertEqual(served.kv_full_shape[1], expected_kv_length)

        (served_output.sum() + producer_output.sum() * 0.0).backward()
        self.assertIsNotNone(producer_scale.grad)
        self.assertGreater(abs(float(producer_scale.grad.item())), 0.0)

        dist.barrier()


if __name__ == "__main__":
    unittest.main()
