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
Fused MoE TopK 操作的 Triton Kernel 实现。

该模块实现了 MoE (Mixture of Experts) 模型中的 TopK 专家选择操作，使用 Triton 进行 GPU 加速。

主要功能：
- TopK 专家选择：从所有专家中选择得分最高的 k 个专家
- 节点限制 (Node Limit)：支持将专家分组，从每组中选择 topk_group 个组
- 概率归一化：对选中的专家概率进行归一化处理
- Routing Map 生成：根据选中的专家索引生成路由映射和分发掩码
- 支持 padding mask 和纯文本 mask

"""

import paddle
import triton
import triton.language as tl

from paddlefleet.ops.triton_ops.utils import enable_compat_on_triton_kernel


@enable_compat_on_triton_kernel
@triton.jit
def _fwd_kernel(
    ptr_gate,
    ptr_choice,
    ptr_out_probs,
    ptr_out_idx,
    ptr_out_sum,
    stride_gate_s,
    stride_gate_e,
    stride_choice_s,
    stride_choice_e,
    stride_out_s,
    stride_out_k,
    n_experts,
    moe_k: tl.constexpr,
    use_node_limit: tl.constexpr,
    n_group: tl.constexpr,
    topk_group: tl.constexpr,
    norm_gate_logits: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused MoE TopK 前向计算 kernel。

    对每个序列位置，从专家中选择 topk 个专家，支持节点限制和归一化。
    """
    pid = tl.program_id(0)

    # Calculate offset for this sequence row
    row_choice_ptr = ptr_choice + pid * stride_choice_s
    row_gate_ptr = ptr_gate + pid * stride_gate_s

    # Offsets for loading the expert row
    off_e = tl.arange(0, BLOCK_SIZE)
    mask_e = off_e < n_experts

    # Load choice probs into registers; init out of bounds with -inf
    choice_vals = tl.load(
        row_choice_ptr + off_e * stride_choice_e,
        mask=mask_e,
        other=float("-inf"),
    )

    # --- Node Limit Logic ---
    if use_node_limit:
        epg = n_experts // n_group
        selected_groups_mask = 0

        # Iteratively select topk groups using a simplistic approach suitable for small n_group
        for _ in range(topk_group):
            best_g_score = float("-inf")
            best_g_idx = -1

            # Evaluate all groups
            for g in range(n_group):
                # Check if this group index 'g' is already in the mask
                is_set = (selected_groups_mask >> g) & 1
                if is_set == 0:
                    g_start = g * epg

                    # Find top 2 sum in this group
                    m1 = float("-inf")
                    m2 = float("-inf")

                    # Iterate experts in group.
                    # Re-loading from memory here is necessary as Triton doesn't support
                    # dynamic indexing into register tensors (choice_vals).
                    # L1 cache should handle the bandwidth well given the small problem size per row.
                    for i in range(epg):
                        idx = g_start + i
                        if idx < n_experts:
                            val = tl.load(
                                row_choice_ptr + idx * stride_choice_e
                            )
                            if val > m1:
                                m2 = m1
                                m1 = val
                            elif val > m2:
                                m2 = val

                    score = m1 + m2
                    if score > best_g_score:
                        best_g_score = score
                        best_g_idx = g

            # Mark selected
            if best_g_idx != -1:
                selected_groups_mask = selected_groups_mask | (1 << best_g_idx)

        # Apply mask to choice_vals
        choice_group_idx = off_e // epg
        is_selected = (selected_groups_mask >> choice_group_idx) & 1
        choice_vals = tl.where(
            (is_selected & mask_e), choice_vals, float("-inf")
        )

    # --- Choice TopK Logic ---
    row_out_probs = ptr_out_probs + pid * stride_out_s
    row_out_idx = ptr_out_idx + pid * stride_out_s

    for k_i in range(moe_k):
        # Find max across the block
        k_val = tl.max(choice_vals, axis=0)

        # Identify elements equal to max
        is_max = choice_vals == k_val

        # Determine index (prefer smallest index if tie)
        k_idx_candidates = tl.where(is_max, off_e, n_experts + 1)
        k_idx = tl.min(k_idx_candidates, axis=0)

        # Store index and value
        tl.store(row_out_idx + k_i * stride_out_k, k_idx)

        # Load gate probability for this index (fetching from global as needed)
        gate_val = tl.load(row_gate_ptr + k_idx * stride_gate_e)
        tl.store(row_out_probs + k_i * stride_out_k, gate_val)

        # Mask out this index so we don't pick it again
        choice_vals = tl.where(off_e != k_idx, choice_vals, float("-inf"))

    # --- Normalization ---
    if norm_gate_logits:
        # Sum the collected probs
        total_sum = 0.0
        for k_i in range(moe_k):
            total_sum += tl.load(row_out_probs + k_i * stride_out_k)

        tl.store(ptr_out_sum + pid, total_sum)

        denom = total_sum
        if denom < 1e-12:
            denom = 1e-12

        for k_i in range(moe_k):
            val_ptr = row_out_probs + k_i * stride_out_k
            val = tl.load(val_ptr)
            tl.store(val_ptr, val / denom)


@enable_compat_on_triton_kernel
@triton.jit
def _bwd_kernel(
    grad_out_probs_ptr,
    ind_ptr,
    normed_probs_ptr,
    sum_ptr,
    grad_gate_ptr,
    stride_grad_out_s,
    stride_grad_out_k,
    stride_ind_s,
    stride_ind_k,
    stride_normed_s,
    stride_normed_k,
    stride_grad_gate_s,
    stride_grad_gate_e,
    moe_k: tl.constexpr,
    norm_gate_logits: tl.constexpr,
    K_BLOCK_SIZE: tl.constexpr,
):
    """
    Fused MoE TopK 反向计算 kernel。

    计算对 gate_probs 的梯度，支持归一化梯度。
    """
    pid = tl.program_id(0)

    row_grad_out = grad_out_probs_ptr + pid * stride_grad_out_s
    row_ind = ind_ptr + pid * stride_ind_s
    row_normed = normed_probs_ptr + pid * stride_normed_s
    row_grad_gate_base = grad_gate_ptr + pid * stride_grad_gate_s

    # 优化1: 使用 mask 方式一次性加载所有 k 个值
    offs_k = tl.arange(0, K_BLOCK_SIZE)
    mask_k = offs_k < moe_k

    grad_out_vals = tl.load(
        row_grad_out + offs_k * stride_grad_out_k, mask=mask_k
    )
    indices = tl.load(row_ind + offs_k * stride_ind_k, mask=mask_k)

    # 优化2: 根据 norm_gate_logits 分支优化 - 避免冗余计算
    if norm_gate_logits:
        # norm_gate_logits=True 路径：需要归一化
        normed_vals = tl.load(
            row_normed + offs_k * stride_normed_k, mask=mask_k
        )
        sigma = tl.load(sum_ptr + pid)

        # 预计算 inv_denom_masked = inv_denom * grad_sigma_mask
        denom = tl.maximum(sigma, 1e-12)
        inv_denom = 1.0 / denom
        inv_denom_masked = tl.where(sigma > 1e-12, inv_denom, 0.0)

        # 向量化计算 dot_prod 和梯度
        dot_prod = tl.sum(grad_out_vals * normed_vals)
        grad_vals = grad_out_vals * inv_denom - dot_prod * inv_denom_masked
    else:
        # norm_gate_logits=False 路径：直接使用 grad_out_vals，无需额外计算
        grad_vals = grad_out_vals

    tl.store(
        row_grad_gate_base + indices * stride_grad_gate_e,
        grad_vals,
        mask=mask_k,
    )


class FusedMoETopk(paddle.autograd.PyLayer):
    """
    Fused MoE TopK 操作，使用 Triton 加速。

    支持节点限制、分组选择和概率归一化。
    """

    @staticmethod
    def forward(
        ctx,
        gate_probs,
        probs_for_choice,
        moe_k,
        use_node_limit,
        n_group,
        topk_group,
        norm_gate_logits,
    ):
        """
        前向计算：选择 topk 专家。

        Args:
            gate_probs: 原始 gate 概率，shape [seq_len, n_experts]
            probs_for_choice: 用于选择专家的概率（可能包含 correction bias），shape [seq_len, n_experts]
            moe_k: 每个 token 选择的专家数量
            use_node_limit: 是否使用节点限制
            n_group: 专家分组数量
            topk_group: 选择的 topk 分组数量
            norm_gate_logits: 是否对 gate logits 进行归一化

        Returns:
            topk_probs: 归一化后的 topk 概率，shape [seq_len, moe_k]
            topk_indices: topk 专家索引，shape [seq_len, moe_k]
        """
        seq_len, n_experts = gate_probs.shape

        topk_indices = paddle.empty((seq_len, moe_k), dtype="int32")
        topk_probs = paddle.empty((seq_len, moe_k), dtype=gate_probs.dtype)
        topk_sum = (
            paddle.empty((seq_len,), dtype="float32")
            if norm_gate_logits
            else None
        )

        # Block size must cover n_experts for the single-block reduction logic
        BLOCK_SIZE = triton.next_power_of_2(n_experts)
        if BLOCK_SIZE < 32:
            BLOCK_SIZE = 32

        # Use topk_probs as dummy pointer for sum if not needed, as it is writable
        ptr_sum_arg = topk_sum if norm_gate_logits else topk_probs

        _fwd_kernel[(seq_len,)](
            gate_probs,
            probs_for_choice,
            topk_probs,
            topk_indices,
            ptr_sum_arg,
            int(gate_probs.stride(0)),
            int(gate_probs.stride(1)),
            int(probs_for_choice.stride(0)),
            int(probs_for_choice.stride(1)),
            int(topk_probs.stride(0)),
            int(topk_probs.stride(1)),
            n_experts,
            moe_k,
            use_node_limit,
            n_group if use_node_limit else 1,
            topk_group if use_node_limit else 1,
            norm_gate_logits,
            BLOCK_SIZE,
        )

        ctx.save_for_backward(topk_indices, topk_probs, topk_sum)
        ctx.input_shape = gate_probs.shape
        ctx.norm_gate_logits = norm_gate_logits
        ctx.moe_k = moe_k

        return topk_probs, topk_indices.to(paddle.int64)

    @staticmethod
    def backward(ctx, grad_output_probs, grad_output_indices):
        """
        反向计算：计算 gate_probs 的梯度。
        """
        topk_indices, topk_normed_probs, topk_sum = ctx.saved_tensor()

        grad_gate_probs = paddle.zeros(
            ctx.input_shape, dtype=grad_output_probs.dtype
        )

        # Dummy ptr for sum if not used
        ptr_sum_arg = topk_sum if ctx.norm_gate_logits else grad_output_probs

        K_BLOCK_SIZE = triton.next_power_of_2(ctx.moe_k)

        _bwd_kernel[(ctx.input_shape[0],)](
            grad_output_probs,
            topk_indices,
            topk_normed_probs,
            ptr_sum_arg,
            grad_gate_probs,
            int(grad_output_probs.stride(0)),
            int(grad_output_probs.stride(1)),
            int(topk_indices.stride(0)),
            int(topk_indices.stride(1)),
            int(topk_normed_probs.stride(0)),
            int(topk_normed_probs.stride(1)),
            int(grad_gate_probs.stride(0)),
            int(grad_gate_probs.stride(1)),
            ctx.moe_k,
            ctx.norm_gate_logits,
            K_BLOCK_SIZE,
        )

        return grad_gate_probs, None


@enable_compat_on_triton_kernel
@triton.jit
def _routing_map_fwd_kernel(
    topk_indices_ptr,
    input_ids_ptr,
    is_pure_text_line_ptr,
    routing_map_ptr,
    topk_indices_out_ptr,
    dispatch_mask_ptr,
    stride_topk_s,
    stride_topk_k,
    stride_routing_s,
    stride_routing_e,
    n_experts,
    seq_len,  # 新增：显式传入 seq_len 以便在 Block 处理时做边界检查
    moe_k,  # 运行时参数：实际的 moe_k 值
    has_input_ids: tl.constexpr,
    has_pure_text_mask: tl.constexpr,
    BLOCK_M: tl.constexpr,  # Sequence 维度的分块大小 (e.g., 32, 64)
    BLOCK_N: tl.constexpr,  # Expert 维度的分块大小 (e.g., 64, 128)
    BLOCK_K: tl.constexpr,  # MoE_K 维度的分块大小 (必须是 2 的幂次方)
):
    """
    Routing Map 前向计算 kernel。

    根据 topk 索引生成路由映射和分发掩码，支持 padding mask 和纯文本 mask。
    """
    # -----------------------------------------------------------
    # 1. 坐标与 Mask 设置
    # -----------------------------------------------------------
    # pid_m 处理 Sequence 维度，pid_n 处理 Expert 维度
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Sequence 维度的偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # 边界 Mask：防止处理超过 seq_len 的数据
    mask_m = offs_m < seq_len

    # -----------------------------------------------------------
    # 2. 加载数据 (利用合并访问)
    # -----------------------------------------------------------
    # 加载 TopK Indices: [BLOCK_M, BLOCK_K]
    # 使用 BLOCK_K 作为编译时常量，通过 mask 处理实际的 moe_k
    offs_k = tl.arange(0, BLOCK_K)
    # moe_k 维度的 mask：只处理有效的 k 值
    mask_k = offs_k < moe_k
    # 计算加载地址：基址 + 行偏移 + 列偏移
    indices_ptrs = (
        topk_indices_ptr
        + (offs_m[:, None] * stride_topk_s)
        + (offs_k[None, :] * stride_topk_k)
    )
    # 加载索引，越界处填充 -1，同时考虑 moe_k 边界
    indices = tl.load(
        indices_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=-1
    )

    # 计算有效性 Mask (is_valid): [BLOCK_M]
    is_valid = tl.full((BLOCK_M,), 1, dtype=tl.int1)

    if has_input_ids:
        in_ids = tl.load(input_ids_ptr + offs_m, mask=mask_m, other=0)
        is_valid = is_valid & (in_ids != 0)

    if has_pure_text_mask:
        p_mask = tl.load(is_pure_text_line_ptr + offs_m, mask=mask_m, other=0)
        is_valid = is_valid & (p_mask > 0)

    # -----------------------------------------------------------
    # 3. 输出 TopK Indices (带 Mask 处理)
    # -----------------------------------------------------------
    # 只需要在 Expert 维度的第一个 Block (pid_n == 0) 进行写入，避免重复写入
    if pid_n == 0:
        # 将无效行的索引置为 -1
        masked_indices = tl.where(is_valid[:, None], indices, -1)
        out_indices_ptrs = (
            topk_indices_out_ptr
            + (offs_m[:, None] * stride_topk_s)
            + (offs_k[None, :] * stride_topk_k)
        )
        # 写入时需要同时考虑 seq_len 和 moe_k 的边界
        tl.store(
            out_indices_ptrs,
            masked_indices,
            mask=mask_m[:, None] & mask_k[None, :],
        )

    # -----------------------------------------------------------
    # 4. 生成 Routing Map (核心优化点)
    # -----------------------------------------------------------
    # Expert 维度的偏移 [BLOCK_N]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_experts

    # 利用 Broadcasting 进行并行比对，消除内部循环
    # indices: [BLOCK_M, moe_k] -> [BLOCK_M, moe_k, 1]
    # offs_n : [BLOCK_N]        -> [1,       1,     BLOCK_N]
    # 结果    : [BLOCK_M, moe_k, BLOCK_N] (Boolean)

    # 判断当前 Block 内的 Experts 是否被选中
    matches = indices[:, :, None] == offs_n[None, None, :]

    # 聚合 moe_k 维度：只要任意一个 k 选中了该 expert，则置为 1
    # 使用 max 实现 "any" 逻辑 (max of [0,0,0,1] = 1, max of [0,0,0,0] = 0)
    routing_block = tl.max(matches.to(tl.float32), axis=1)

    # 应用有效性 Mask：无效行的 Routing Map 全为 0
    routing_block = tl.where(is_valid[:, None], routing_block, 0.0)

    # -----------------------------------------------------------
    # 5. 写入 Routing Map
    # -----------------------------------------------------------
    # 计算写入地址：[BLOCK_M, BLOCK_N]
    routing_out_ptrs = (
        routing_map_ptr
        + (offs_m[:, None] * stride_routing_s)
        + (offs_n[None, :] * stride_routing_e)
    )

    # 组合 Mask：同时考虑 Sequence 边界和 Expert 边界
    full_store_mask = mask_m[:, None] & mask_n[None, :]

    tl.store(routing_out_ptrs, routing_block, mask=full_store_mask)

    # -----------------------------------------------------------
    # 6. 计算 Dispatch Mask (沿 sequence 维度求和)
    # -----------------------------------------------------------
    # 对当前 block 处理的 expert 维度，累加 routing_block 的值
    # 使用原子加法来处理多个 block 同时写入同一个 expert 的情况
    dispatch_block = tl.sum(routing_block, axis=0)  # [BLOCK_N]
    # 使用 tl.where 将越界的 dispatch_block 值置为 0
    dispatch_block = tl.where(mask_n, dispatch_block, 0.0)
    # 转换为 int64 类型
    dispatch_block = dispatch_block.to(tl.int64)
    # 计算目标地址
    dispatch_ptrs = dispatch_mask_ptr + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # 使用 tl.atomic_add 进行向量化的原子加法
    # 注意：tl.atomic_add 支持向量化的指针和值
    tl.atomic_add(dispatch_ptrs, dispatch_block, mask=mask_n)


# -----------------------------------------------------------
# Python Wrapper 示例
# -----------------------------------------------------------


def routing_map_forward(
    gate_probs, topk_indices, input_ids=None, is_pure_text_line=None
):
    """
    Get routing_map using Triton kernel.

    Args:
        gate_probs: Gate probabilities [seq_len, n_experts]
        topk_indices: Topk expert indices [seq_len, moe_k]
        input_ids: Input token IDs [seq_len] (optional, for padding mask)
        is_pure_text_line: Pure text line mask [seq_len] (optional)

    Returns:
        routing_map: Routing map [seq_len, n_experts]
        topk_indices_out: Topk indices with masking [seq_len, moe_k]
        dispatch_mask: Dispatch mask [n_experts]
    """
    seq_len, moe_k = topk_indices.shape
    n_experts = gate_probs.shape[1]

    # 准备输出 Tensor
    routing_map = paddle.zeros((seq_len, n_experts), dtype=paddle.float32)
    topk_indices_out = paddle.empty_like(topk_indices)
    # 初始化 dispatch_mask 为 0，kernel 会使用 atomic_add 累加
    dispatch_mask = paddle.zeros((n_experts,), dtype=paddle.int64)

    # 调优的 Block Size
    # BLOCK_M: 一次处理多少行。推荐 32 或 64，越大显存带宽利用越好，但寄存器压力也大。
    # BLOCK_N: 一次处理多少 Expert。推荐 64 或 128。
    # BLOCK_K: 一次处理多少 MoE_K。必须是 2 的幂次方，使用 next_power_of_2 向上取整
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = triton.next_power_of_2(moe_k)

    grid = (triton.cdiv(seq_len, BLOCK_M), triton.cdiv(n_experts, BLOCK_N))

    # 准备指针参数 - Paddle Tensor 可以直接传递给 Triton
    _routing_map_fwd_kernel[grid](
        topk_indices_ptr=topk_indices,
        input_ids_ptr=input_ids
        if input_ids is not None
        else topk_indices,  # 占位
        is_pure_text_line_ptr=is_pure_text_line
        if is_pure_text_line is not None
        else topk_indices,  # 占位
        routing_map_ptr=routing_map,
        topk_indices_out_ptr=topk_indices_out,
        dispatch_mask_ptr=dispatch_mask,
        stride_topk_s=int(topk_indices.stride(0)),
        stride_topk_k=int(topk_indices.stride(1)),
        stride_routing_s=int(routing_map.stride(0)),
        stride_routing_e=int(routing_map.stride(1)),
        n_experts=n_experts,
        seq_len=seq_len,
        moe_k=moe_k,
        has_input_ids=input_ids is not None,
        has_pure_text_mask=is_pure_text_line is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return routing_map, topk_indices_out, dispatch_mask
