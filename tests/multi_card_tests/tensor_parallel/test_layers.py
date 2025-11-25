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

# Refer to NVIDIA Megatron-LM https://github.com/NVIDIA/Megatron-LM.git
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import paddle
import paddle.distributed as dist
from paddle.distributed import ShardedWeight

import paddlefleet.parallel_state as ps
from paddlefleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    linear_with_frozen_weight,
)
from paddlefleet.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.transformer.transformer_config import TransformerConfig
from tests.multi_card_tests.tensor_parallel.test_utilities import Utils


def test_LinearWithFrozenWeight(tensor_parallel, allreduce_dgrad):
    size_per_partition = int(8 / tensor_parallel)

    # Input is an 8x8 identity matrix.
    input_data = paddle.eye(8).cuda()
    input_data.requires_grad = True

    # Weight is an 8x8 matrix of all ones. If tensor parallelism > 1, the weight is partitioned evenly across GPUs.
    weight = paddle.ones((8, size_per_partition)).cuda()

    # Bias is a vector of length 8 of all zeros. If tensor parallelism > 1, the bias is partitioned evenly across GPUs
    bias = paddle.zeros(size_per_partition).cuda()

    gradient_accumulation_fusion = False
    sequence_parallel = False
    grad_output_buffer = None
    wgrad_deferral_limit = None

    weight.stop_gradient = True
    bias.stop_gradient = True
    output_parallel = linear_with_frozen_weight(
        input_data,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
    )
    output = gather_from_tensor_model_parallel_region(
        output_parallel
    )  # no-op if tensor_parallel == 1.
    output.sum().backward()

    expected_output = paddle.ones([8, 8]).cuda()
    expected_grad = 8 * paddle.ones([8, 8]).cuda()

    assert paddle.allclose(output, expected_output)
    assert paddle.allclose(input_data.grad, expected_grad)


def column_parallel_baseline():
    transformer_config = TransformerConfig(
        num_layers=1,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )
    paddle.manual_seed(42)

    tp1_group = dist.new_group([dist.get_rank()])
    col_tp1 = ColumnParallelLinear(
        input_size=8,
        output_size=8,
        init_method=transformer_config.init_method,
        bias=True,
        config=transformer_config,
        skip_bias_add=False,
        gather_output=False,
        tp_group=tp1_group,
    )

    # Input is an 8x8 identity matrix.
    input_data = paddle.arange(64).reshape((8, 8)) * 0.1
    input_data.requires_grad = True

    output, _ = col_tp1(input_data)
    output.sum().backward()

    return output, input_data.grad, col_tp1.weight.grad, col_tp1.bias.grad


def test_ColumnParallelLinear(
    tensor_parallel,
    output_baseline,
    input_grad_baseline,
    weight_grad_baseline,
    bias_grad_baseline,
):
    transformer_config = TransformerConfig(
        num_layers=1,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )

    paddle.manual_seed(42)
    model_parallel_cuda_manual_seed(42)
    size_per_partition = int(8 / tensor_parallel)
    col_tp4 = ColumnParallelLinear(
        input_size=8,
        output_size=8,
        init_method=transformer_config.init_method,
        bias=True,
        config=transformer_config,
        skip_bias_add=False,
        gather_output=True,
    )

    input_data = paddle.arange(64).reshape((8, 8)) * 0.1
    input_data.requires_grad = True

    output, _ = col_tp4(input_data)
    output.sum().backward()

    rank = ps.get_tensor_model_parallel_rank()
    assert paddle.equal_all(output, output_baseline)
    assert paddle.allclose(input_data.grad, input_grad_baseline)
    assert paddle.allclose(
        col_tp4.weight.grad, weight_grad_baseline[:, rank * 2 : (rank + 1) * 2]
    )
    assert paddle.allclose(
        col_tp4.bias.grad, bias_grad_baseline[rank * 2 : (rank + 1) * 2]
    )

    sharded_dict = col_tp4.sharded_state_dict()
    assert "bias" in sharded_dict
    bias_shard = sharded_dict["bias"]
    assert isinstance(bias_shard, ShardedWeight)
    assert "weight" in sharded_dict
    weight_shard = sharded_dict["weight"]
    assert isinstance(weight_shard, ShardedWeight)

    in_f, out_f = col_tp4.input_size, col_tp4.output_size
    assert weight_shard.global_shape == (in_f, out_f)
    assert weight_shard.local_shape == (in_f, out_f // tensor_parallel)
    assert weight_shard.global_offset == (
        0,
        rank * (out_f // tensor_parallel),
    )
    assert bias_shard.global_shape == (out_f,)
    assert bias_shard.local_shape == (out_f // tensor_parallel,)
    assert bias_shard.global_offset == (rank * (out_f // tensor_parallel),)


def row_parallel_baseline():
    transformer_config = TransformerConfig(
        num_layers=1,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )
    paddle.manual_seed(42)

    tp1_group = dist.new_group([dist.get_rank()])
    row_tp1 = RowParallelLinear(
        input_size=8,
        output_size=8,
        init_method=transformer_config.init_method,
        bias=True,
        input_is_parallel=False,
        config=transformer_config,
        skip_bias_add=False,
        tp_group=tp1_group,
    )

    input_data = paddle.arange(64).reshape((8, 8)) * 0.1
    input_data.requires_grad = True

    output, _ = row_tp1(input_data)
    output.sum().backward()

    return output, input_data.grad, row_tp1.weight.grad, row_tp1.bias.grad


def test_RowParallelLinear(
    tensor_parallel,
    output_baseline,
    input_grad_baseline,
    weight_grad_baseline,
    bias_grad_baseline,
):
    transformer_config = TransformerConfig(
        num_layers=1,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )

    paddle.manual_seed(42)
    model_parallel_cuda_manual_seed(42)
    size_per_partition = int(8 / tensor_parallel)
    row_tp4 = RowParallelLinear(
        input_size=8,
        output_size=8,
        init_method=transformer_config.init_method,
        bias=True,
        config=transformer_config,
        skip_bias_add=False,
        input_is_parallel=True,
    )

    input_data = paddle.arange(64).reshape((8, 8)) * 0.1
    input_data.requires_grad = True
    rank = ps.get_tensor_model_parallel_rank()
    scattered_input = scatter_to_tensor_model_parallel_region(input_data)

    output, _ = row_tp4(scattered_input)
    output.sum().backward()

    assert paddle.allclose(output, output_baseline, atol=1e-7)
    assert paddle.allclose(input_data.grad, input_grad_baseline)
    assert paddle.allclose(
        row_tp4.weight.grad, weight_grad_baseline[rank * 2 : (rank + 1) * 2, :]
    )
    assert paddle.allclose(row_tp4.bias.grad, bias_grad_baseline)

    sharded_dict = row_tp4.sharded_state_dict()
    assert "bias" in sharded_dict
    bias_shard = sharded_dict["bias"]
    assert isinstance(bias_shard, ShardedWeight)
    assert "weight" in sharded_dict
    weight_shard = sharded_dict["weight"]
    assert isinstance(weight_shard, ShardedWeight)

    in_f, out_f = row_tp4.input_size, row_tp4.output_size
    assert weight_shard.global_shape == (in_f, out_f)
    assert weight_shard.local_shape == (in_f // tensor_parallel, out_f)
    assert weight_shard.global_offset == (
        rank * (in_f // tensor_parallel),
        0,
    )
    assert bias_shard.global_shape == [out_f]
    assert bias_shard.local_shape == bias_shard.global_shape
    assert bias_shard.global_offset == (0,)


def embedding_baseline():
    transformer_config = TransformerConfig(
        num_layers=1,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )
    paddle.manual_seed(42)

    tp1_group = dist.new_group([dist.get_rank()])
    emb_tp1 = VocabParallelEmbedding(
        num_embeddings=16,
        embedding_dim=4,
        init_method=transformer_config.init_method,
        config=transformer_config,
        tp_group=tp1_group,
    )

    input_data = paddle.tensor(
        [[6, 3, 4, 1, 7, 13, 8, 0], [0, 5, 12, 11, 9, 2, 1, 15]]
    )
    input_data.requires_grad = True

    output = emb_tp1(input_data)
    output.sum().backward()

    return output, emb_tp1.weight.grad


def test_VocabParallelEmbedding(
    tensor_parallel, output_baseline, weight_grad_baseline
):
    transformer_config = TransformerConfig(
        num_layers=1,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )

    paddle.manual_seed(42)
    model_parallel_cuda_manual_seed(42)
    emb_tp4 = VocabParallelEmbedding(
        num_embeddings=16,
        embedding_dim=4,
        init_method=transformer_config.init_method,
        config=transformer_config,
    )

    input_data = paddle.tensor(
        [[6, 3, 4, 1, 7, 13, 8, 0], [0, 5, 12, 11, 9, 2, 1, 15]]
    )
    input_data.requires_grad = True

    output = emb_tp4(input_data)
    output.sum().backward()

    rank = dist.get_rank()
    assert paddle.equal_all(output, output_baseline)
    assert paddle.allclose(
        emb_tp4.weight.grad, weight_grad_baseline[rank * 4 : (rank + 1) * 4, :]
    )

    sharded_dict = emb_tp4.sharded_state_dict()
    assert "bias" not in sharded_dict
    assert "weight" in sharded_dict
    weight_shard = sharded_dict["weight"]
    assert isinstance(weight_shard, ShardedWeight)
    assert weight_shard.global_shape == (
        emb_tp4.num_embeddings,
        emb_tp4.embedding_dim,
    )
    assert weight_shard.local_shape == (
        emb_tp4.num_embeddings // tensor_parallel,
        emb_tp4.embedding_dim,
    )
    assert weight_shard.global_offset == (
        rank * (emb_tp4.num_embeddings // tensor_parallel),
        0,
    )


if __name__ == "__main__":
    tensor_parallel = 4
    Utils.initialize_model_parallel(tensor_parallel, 1)
    test_LinearWithFrozenWeight(4, True)
    output_tp1, input_grad_tp1, weight_grad_tp1, bias_grad_tp1 = (
        column_parallel_baseline()
    )
    test_ColumnParallelLinear(
        tensor_parallel,
        output_tp1,
        input_grad_tp1,
        weight_grad_tp1,
        bias_grad_tp1,
    )
    output_tp1, input_grad_tp1, weight_grad_tp1, bias_grad_tp1 = (
        row_parallel_baseline()
    )
    test_RowParallelLinear(
        tensor_parallel,
        output_tp1,
        input_grad_tp1,
        weight_grad_tp1,
        bias_grad_tp1,
    )
    output_tp1, weight_grad_tp1 = embedding_baseline()
    test_VocabParallelEmbedding(4, output_tp1, weight_grad_tp1)
