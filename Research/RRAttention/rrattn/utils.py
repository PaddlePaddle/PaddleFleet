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

import paddle


def find_blocks_chunked(
    input_tensor,
    current_index,
    threshold,
    num_to_choose,
    decoding: bool,
    mode: str = "both",
    causal=True,
):
    assert threshold is None or num_to_choose is None
    batch_size, head_num, chunk_num, block_num = input_tensor.shape
    if mode == "prefill" and decoding:
        return paddle.ones_like(input_tensor, dtype=paddle.bool)
    if mode == "decode" and not decoding:
        mask = paddle.ones_like(input_tensor, dtype=paddle.bool)
        if causal:
            mask[:, :, :, current_index : current_index + chunk_num] = (
                paddle.tril(
                    paddle.ones(
                        1,
                        head_num,
                        chunk_num,
                        chunk_num,
                        device=input_tensor.device,
                    )
                )
            )
            mask[:, :, current_index + chunk_num :, :] = 0
            return paddle.concat(
                [
                    paddle.ones_like(input_tensor, dtype=paddle.bool)[
                        :, :, 0 : current_index + 1
                    ],
                    paddle.zeros_like(input_tensor, dtype=paddle.bool)[
                        :, :, current_index + 1 :
                    ],
                ],
                dim=-1,
            )
        return mask

    input_tensor = input_tensor.astype(paddle.float32)

    if threshold is None:
        raise NotImplementedError("block num chunk prefill not implemented")

    total_sum = input_tensor.sum(dim=-1, keepdim=True)
    if isinstance(threshold, paddle.Tensor):
        threshold = threshold.astype(paddle.float32)
        required_sum = total_sum * threshold.unsqueeze(0).unsqueeze(
            -1
        ).unsqueeze(-1).expand((batch_size, head_num, chunk_num, 1)).to(
            input_tensor.device
        )
    else:
        required_sum = total_sum * threshold

    if causal:
        mask = paddle.zeros_like(input_tensor, dtype=paddle.bool)
        mask[:, :, :, 0] = 1
        mask[:, :, :, current_index : current_index + chunk_num] = (
            paddle.eye(chunk_num, device=mask.device)
            .astype(paddle.bool)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(1, head_num, chunk_num, chunk_num)
        )
        other_values = input_tensor.masked_fill(mask, 0)
        sorted_values, _ = paddle.compat.sort(
            other_values, dim=-1, descending=True
        )
        sorted_values = sorted_values.to(input_tensor.device)

        sorted_values = paddle.concat(
            [
                paddle.zeros(
                    (batch_size, head_num, chunk_num, 1),
                    device=input_tensor.device,
                ),
                paddle.where(
                    mask, input_tensor, paddle.zeros_like(input_tensor)
                ).sum(dim=-1, keepdim=True),
                sorted_values[:, :, :, :-2],
            ],
            dim=-1,
        )

        _, index = paddle.compat.sort(
            paddle.where(mask, 100000 * (1 + input_tensor), input_tensor),
            dim=-1,
            descending=True,
        )
        cumulative_sum_without_self = paddle.concat(
            [
                paddle.zeros(
                    (batch_size, head_num, chunk_num, 1),
                    device=input_tensor.device,
                ),
                sorted_values[:, :, :, 0:-1],
            ],
            dim=-1,
        ).cumsum(dim=-1)

        index_mask = cumulative_sum_without_self < required_sum
        index = paddle.where(index_mask, index, paddle.zeros_like(index))
        mask = mask.view(batch_size, head_num * chunk_num, block_num)
        index = index.view(batch_size, head_num * chunk_num, block_num)
        mask[
            :,
            paddle.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1),
            index,
        ] = True
        mask = mask.view(batch_size, head_num, chunk_num, block_num)
    else:
        mask = paddle.zeros_like(input_tensor, dtype=paddle.bool)
        sorted_values, index = paddle.compat.sort(
            input_tensor, dim=-1, descending=True
        )
        sorted_values = sorted_values.to(input_tensor.device)
        cumulative_sum_without_self = paddle.concat(
            [
                paddle.zeros(
                    (batch_size, head_num, chunk_num, 1),
                    device=input_tensor.device,
                ),
                sorted_values[:, :, :, 0:-1],
            ],
            dim=-1,
        ).cumsum(dim=-1)
        index_mask = cumulative_sum_without_self < required_sum
        index = paddle.where(index_mask, index, paddle.zeros_like(index))
        mask = mask.view(batch_size, head_num * chunk_num, block_num)
        index = index.view(batch_size, head_num * chunk_num, block_num)
        mask[
            :,
            paddle.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1),
            index,
        ] = True
        mask = mask.view(batch_size, head_num, chunk_num, block_num)

    try:
        if causal:
            assert (~mask[:, :, :, current_index + chunk_num :]).all()
    except Exception:
        mask[:, :, :, current_index + chunk_num :] = False

    if causal:
        if decoding:
            assert mask[:, :, :, 0].all() and mask[:, :, :, -1].all()
        else:
            lambda_mask = paddle.zeros_like(
                input_tensor, dtype=paddle.bool, device=input_tensor.device
            )
            lambda_mask[:, :, :, 0] = 1
            lambda_mask[:, :, :, current_index : current_index + chunk_num] = (
                paddle.eye(chunk_num, device=lambda_mask.device)
                .astype(paddle.bool)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(1, head_num, chunk_num, chunk_num)
            )
            assert paddle.where(lambda_mask, mask, paddle.ones_like(mask)).all()

    return mask
