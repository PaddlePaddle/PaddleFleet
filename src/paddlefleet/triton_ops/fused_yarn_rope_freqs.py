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

"""
Fused Triton kernel for YarnRotaryEmbedding frequency generation.

Replaces: arange + cast + add(offset) + outer(seq, inv_freq) + cat(freqs, freqs)
With: a single kernel that produces the final emb tensor directly.

Output shape: [1, max_seq_len, 1, dim] where dim = head_dim (full, after cat).
"""

import paddle
from paddle import Tensor

from .utils import enable_compat_on_triton_kernel, is_torch_compat_available

if is_torch_compat_available():
    paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl


@enable_compat_on_triton_kernel
@triton.jit
def _fused_yarn_freqs_kernel(
    out_ptr,
    inv_freq_ptr,
    offset,
    max_seq_len,
    half_dim: tl.constexpr,
    full_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused kernel: arange + offset + outer + cat.

    Each program handles 1 position x full_dim output columns.
    Grid size = max_seq_len.
    """
    pid_s = tl.program_id(0)

    if pid_s >= max_seq_len:
        return

    # Compute position: (index + offset) as float32
    pos = (pid_s + offset).to(tl.float32)

    # Load inv_freq [half_dim]
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < half_dim

    inv_freq = tl.load(inv_freq_ptr + d_offs, mask=d_mask, other=0.0)

    # freqs[d] = pos * inv_freq[d]
    freqs = pos * inv_freq

    # Write twice (cat((freqs, freqs), axis=-1)):
    row_base = pid_s * full_dim
    tl.store(out_ptr + row_base + d_offs, freqs, mask=d_mask)
    tl.store(out_ptr + row_base + half_dim + d_offs, freqs, mask=d_mask)


def fused_yarn_rope_freqs(
    inv_freq: Tensor,
    max_seq_len: int,
    offset: int = 0,
) -> Tensor:
    """Fused generation of YaRN RoPE frequency embeddings.

    Replaces:
        seq = paddle.arange(max_seq_len).astype(dtype) + offset
        freqs = paddle.outer(seq, inv_freq)
        emb = paddle.cat((freqs, freqs), axis=-1)
        emb = emb[None, :, None, :]

    Args:
        inv_freq: [half_dim] pre-computed inverse frequency tensor.
        max_seq_len: sequence length.
        offset: position offset (default 0).

    Returns:
        emb: [1, max_seq_len, 1, dim] where dim = 2 * half_dim.
    """
    half_dim = inv_freq.shape[0]
    full_dim = half_dim * 2

    # Allocate output: [max_seq_len, full_dim]
    out = paddle.empty([max_seq_len, full_dim], dtype=paddle.float32)

    # Triton grid: one program per position
    BLOCK_D = triton.next_power_of_2(half_dim)
    grid = (max_seq_len,)

    _fused_yarn_freqs_kernel[grid](
        out,
        inv_freq,
        offset,
        max_seq_len,
        half_dim=half_dim,
        full_dim=full_dim,
        BLOCK_D=BLOCK_D,
    )

    # Reshape to [1, max_seq_len, 1, full_dim]
    return out.reshape([1, max_seq_len, 1, full_dim])
