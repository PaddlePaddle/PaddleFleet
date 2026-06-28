# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from paddlefleet.context_parallel_utils import (
    all_gather_balance,
    scatter_balance,
)
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.dot_product_attention import (
    DotProductAttention,
)
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.sink_impl import sink_attention
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import init_method_normal, scaled_init_method_normal


def _make_config(context_parallel_size=1):
    return TransformerConfig(
        num_hidden_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=64,
        softmax_scale=None,
        use_bias=True,
        recompute_granularity=None,
        recompute_modules=None,
        init_method=init_method_normal(0.02),
        output_layer_init_method=scaled_init_method_normal(0.02, 1, 2.0),
        rms_norm_eps=1e-5,
        context_parallel_size=context_parallel_size,
        sequence_parallel=False,
        apply_query_key_layer_scaling=False,
        sliding_window=None,
        window_attn_skip_freq=None,
        fp16=False,
        bf16=True,
        masked_softmax_fusion=False,
        attention_softmax_in_fp32=True,
        attention_dropout=0.0,
        softmax_type="learnable",
        fa_version=4,
        params_dtype=paddle.bfloat16,
    )


def _initialize_cp_fleet(cp_size):
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": cp_size,
        "sep_degree": 1,
        "cp_degree": cp_size,
        "ep_degree": cp_size,
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
    initialize_fleet(strategy=strategy)


def _make_full_qkv_and_sink():
    paddle.seed(2026)
    np.random.seed(2026)
    batch_size = 1
    seq_len = 4096
    num_heads = 4
    head_dim = 64
    query = paddle.randn(
        [batch_size, seq_len, num_heads, head_dim], dtype=paddle.bfloat16
    )
    key = paddle.randn(
        [batch_size, seq_len, num_heads, head_dim], dtype=paddle.bfloat16
    )
    value = paddle.randn(
        [batch_size, seq_len, num_heads, head_dim], dtype=paddle.bfloat16
    )
    sink = paddle.randn([num_heads], dtype=paddle.bfloat16)
    return query, key, value, sink


def _set_sink(attn, sink):
    with paddle.no_grad():
        attn.softmax_offset.set_value(sink.astype(attn.softmax_offset.dtype))
    attn.softmax_offset.stop_gradient = False


def _clone_for_grad(tensor):
    cloned = tensor.detach().clone()
    cloned.stop_gradient = False
    return cloned


class TestCPDotProductAttentionSink(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not paddle.is_compiled_with_cuda():
            raise unittest.SkipTest("Requires CUDA for CP flashmask attention.")
        cls.world_size = dist.get_world_size()
        if cls.world_size < 2:
            raise unittest.SkipTest("Requires at least 2 ranks.")
        _initialize_cp_fleet(cls.world_size)

    def test_forward_backward_matches_full_sequence_sink_attention(self):
        rank = dist.get_rank()
        paddle.device.set_device(f"gpu:{rank}")

        query_seed, key_seed, value_seed, sink_seed = _make_full_qkv_and_sink()
        query_seed = query_seed.cuda()
        key_seed = key_seed.cuda()
        value_seed = value_seed.cuda()
        sink_seed = sink_seed.cuda()

        query_full = _clone_for_grad(query_seed)
        key_full = _clone_for_grad(key_seed)
        value_full = _clone_for_grad(value_seed)
        baseline_sink = sink_seed.detach().clone()
        baseline_sink.stop_gradient = False

        # Drive the baseline through the same FA4 + learnable_sink kernel
        # used inside the CP path (cp_flashmask_allgatherkv_balance_forward
        # ultimately calls _flash_attn_fwd with learnable_sink=sink). Without
        # this, the cp=1 branch would dispatch to SDPA / manual baddbmm and
        # produce numerically different results that mask whether the CP
        # implementation is correct.
        seq_len_full = query_full.shape[1]
        no_mask_indices = paddle.concat(
            [
                paddle.full(
                    [1, 1, seq_len_full, 1],
                    fill_value=seq_len_full,
                    dtype=paddle.int32,
                ),
                paddle.to_tensor(
                    np.arange(seq_len_full), dtype=paddle.int32
                ).reshape([1, 1, seq_len_full, 1]),
            ],
            axis=-1,
        ).cuda()
        expected = sink_attention(
            query_full,
            key_full,
            value_full,
            sink=baseline_sink,
            startend_row_indices=no_mask_indices,
            dropout=0.0,
            causal=False,
        )
        # sink_attention returns [B, S, H, D]; reshape to [B, S, H*D] to match
        # DotProductAttention's flashmask path output shape.
        expected = expected.reshape([expected.shape[0], expected.shape[1], -1])
        expected.astype("float32").sum().backward()

        cp_group = (
            fleet.get_hybrid_communicate_group().get_context_parallel_group()
        )
        query = _clone_for_grad(
            scatter_balance(query_seed, axis=1, group=cp_group)
        )
        key = _clone_for_grad(scatter_balance(key_seed, axis=1, group=cp_group))
        value = _clone_for_grad(
            scatter_balance(value_seed, axis=1, group=cp_group)
        )
        query.retain_grads()
        key.retain_grads()
        value.retain_grads()

        cp_attn = DotProductAttention(
            config=_make_config(context_parallel_size=self.world_size),
            layer_number=1,
            attn_mask_type=AttnMaskType.no_mask,
            attention_type="self",
        )
        cp_attn.eval()
        _set_sink(cp_attn, sink_seed)
        actual_local = cp_attn(
            query,
            key,
            value,
            None,
            attn_mask_type=AttnMaskType.no_mask,
        )
        actual_local.astype("float32").sum().backward()
        actual = all_gather_balance(actual_local, axis=1, group=cp_group)

        np.testing.assert_allclose(
            np.array(actual.astype("float32")),
            np.array(expected.astype("float32")),
            rtol=3e-2,
            atol=3e-2,
        )

        query_grad = all_gather_balance(query.grad, axis=1, group=cp_group)
        key_grad = all_gather_balance(key.grad, axis=1, group=cp_group)
        value_grad = all_gather_balance(value.grad, axis=1, group=cp_group)
        for actual_grad, expected_grad, name in [
            (query_grad, query_full.grad, "query_grad"),
            (key_grad, key_full.grad, "key_grad"),
            (value_grad, value_full.grad, "value_grad"),
        ]:
            np.testing.assert_allclose(
                np.array(actual_grad.astype("float32")),
                np.array(expected_grad.astype("float32")),
                rtol=3e-2,
                atol=3e-2,
                err_msg=name,
            )

        sink_grad = cp_attn.softmax_offset.grad.clone()
        dist.all_reduce(sink_grad, group=cp_group)
        np.testing.assert_allclose(
            np.array(sink_grad.astype("float32")),
            np.array(baseline_sink.grad.astype("float32")),
            rtol=3e-2,
            atol=3e-2,
            err_msg="sink_grad",
        )


if __name__ == "__main__":
    unittest.main()
