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

"""Triton fused gen_m_indices kernel."""

import paddle
import triton
import triton.language as tl

from .utils import enable_compat_on_triton_kernel


@enable_compat_on_triton_kernel
@triton.jit
def fused_gen_m_indices_kernel(
    out_ptr,
    expert_offset: tl.constexpr,
    token_offset,
    num_tokens,
    count0,
    count1,
    count2,
    count3,
    count4,
    count5,
    count6,
    count7,
    count8,
    count9,
    count10,
    count11,
    count12,
    count13,
    count14,
    BLOCK_SIZE: tl.constexpr,
):
    """Generate repeated expert ids for at most 16 experts in one launch."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_tokens

    end1 = count0
    end2 = end1 + count1
    end3 = end2 + count2
    end4 = end3 + count3
    end5 = end4 + count4
    end6 = end5 + count5
    end7 = end6 + count6
    end8 = end7 + count7
    end9 = end8 + count8
    end10 = end9 + count9
    end11 = end10 + count10
    end12 = end11 + count11
    end13 = end12 + count12
    end14 = end13 + count13
    end15 = end14 + count14

    expert_idx = tl.full((BLOCK_SIZE,), expert_offset, tl.int32)
    expert_idx = tl.where(offsets >= end1, expert_offset + 1, expert_idx)
    expert_idx = tl.where(offsets >= end2, expert_offset + 2, expert_idx)
    expert_idx = tl.where(offsets >= end3, expert_offset + 3, expert_idx)
    expert_idx = tl.where(offsets >= end4, expert_offset + 4, expert_idx)
    expert_idx = tl.where(offsets >= end5, expert_offset + 5, expert_idx)
    expert_idx = tl.where(offsets >= end6, expert_offset + 6, expert_idx)
    expert_idx = tl.where(offsets >= end7, expert_offset + 7, expert_idx)
    expert_idx = tl.where(offsets >= end8, expert_offset + 8, expert_idx)
    expert_idx = tl.where(offsets >= end9, expert_offset + 9, expert_idx)
    expert_idx = tl.where(offsets >= end10, expert_offset + 10, expert_idx)
    expert_idx = tl.where(offsets >= end11, expert_offset + 11, expert_idx)
    expert_idx = tl.where(offsets >= end12, expert_offset + 12, expert_idx)
    expert_idx = tl.where(offsets >= end13, expert_offset + 13, expert_idx)
    expert_idx = tl.where(offsets >= end14, expert_offset + 14, expert_idx)
    expert_idx = tl.where(offsets >= end15, expert_offset + 15, expert_idx)

    tl.store(out_ptr + token_offset + offsets, expert_idx, mask=mask)


def gen_m_indices_fusion(tokens_per_expert: list[int]) -> paddle.Tensor:
    """Generate a 1D tensor by repeating each expert id by its token count."""
    num_tokens = sum(tokens_per_expert)
    if num_tokens == 0:
        return paddle.empty([0], dtype="int32")

    chunk_size = 16
    block_size = 1024

    token_offset = 0
    expert_offset = 0
    num_experts = len(tokens_per_expert)

    out = paddle.empty([num_tokens], dtype="int32")

    while expert_offset < num_experts:
        counts = []
        chunk_tokens = 0
        for i in range(expert_offset, expert_offset + chunk_size):
            count = tokens_per_expert[i] if i < num_experts else 0
            counts.append(count)
            chunk_tokens += count

        if chunk_tokens > 0:
            grid = (triton.cdiv(chunk_tokens, block_size),)
            fused_gen_m_indices_kernel[grid](
                out,
                expert_offset,
                token_offset,
                chunk_tokens,
                *counts[:-1],
                BLOCK_SIZE=block_size,
            )

        token_offset += chunk_tokens
        expert_offset += chunk_size

    return out


if __name__ == "__main__":

    def _reference(tokens_per_expert):
        counts = paddle.to_tensor(tokens_per_expert, dtype="int32")
        if counts.shape[0] == 0:
            return paddle.empty([0], dtype="int32")
        return paddle.repeat_interleave(
            paddle.arange(counts.shape[0], dtype="int32"),
            counts,
        )

    def _check(tokens_per_expert):
        paddle.base.core.nvprof_nvtx_push("triton")
        actual = gen_m_indices_fusion(tokens_per_expert)
        paddle.base.core.nvprof_nvtx_pop()

        paddle.base.core.nvprof_nvtx_push("paddle")
        expected = _reference(tokens_per_expert)
        paddle.base.core.nvprof_nvtx_pop()

        assert actual.shape == expected.shape, (actual.shape, expected.shape)
        assert paddle.all(actual == expected).item(), (
            tokens_per_expert,
            actual.numpy().tolist(),
            expected.numpy().tolist(),
        )

    test_cases = [
        [
            13263,
            2441,
            4968,
            8104,
            7841,
            6604,
            6164,
            7054,
            9330,
            3376,
            5591,
            3946,
            7798,
            15858,
            6794,
            12351,
            11652,
            7874,
            2130,
            2388,
            7182,
            6572,
            14851,
            9255,
            11480,
            16232,
            13388,
            10421,
            15141,
            16182,
            11254,
            9579,
        ]
    ] * 10

    paddle.base.core.nvprof_start()
    for case in test_cases:
        paddle.base.core.nvprof_nvtx_push("case")
        _check(case)
        paddle.base.core.nvprof_nvtx_pop()
    paddle.base.core.nvprof_stop()

    print("gen_m_indices_fusion tests passed")
