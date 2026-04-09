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

import paddle

from paddlefleet.transformer.moe.fp8_utils import ExpertsGroupGemmContiguousNode

from .fp8_utils import FP8_ALIGN


class UnZipNode:
    """
    UnZipNode 类用于对输入的token 矩阵根据分发索引进行解压操作,得到专家需要处理的 token。
    """

    def __init__(self, token_dispatcher, name="unzip"):
        self.token_dispatcher = token_dispatcher
        self.name = name
        self.unzipped_probs = None
        self.zipped_expertwise_rowmap = None

    def reset_state(self):
        """
        重置模型的状态。

        Args:
            无

        Returns:
            无

        """
        self.unzipped_probs = None
        self.zipped_expertwise_rowmap = None

    def cached_tensors(self):
        """
        cached_tensors
        """
        return [self.unzipped_probs, self.zipped_expertwise_rowmap]

    def set_cached_tensors(self, tensors):
        """
        set_cached_tensors
        """
        self.unzipped_probs, self.zipped_expertwise_rowmap = tensors

    def clear_cached_tensors(self):
        """
        clear_cached_tensors
        """
        self.set_cached_tensors([None] * len(self.cached_tensors()))

    @paddle.no_grad()
    def forward(
        self,
        hs_2d_dispatched,
        dispatched_indices,
        dispatched_probs,
        topk,
        num_experts,
        tokens_per_expert,
        fill_output=True,
    ):
        """
        前向传播函数，用于解压输入的张量。

        Args:
            hs_2d_dispatched: 原始输入的token。
            dispatched_indices: 分发索引。
            dispatched_probs: 分发概率。

        Returns:
            tuple: 返回解压后的令牌、压缩后的专家行映射、解压后的概率。
        """
        if isinstance(hs_2d_dispatched, tuple):
            assert len(hs_2d_dispatched) == 2, (
                f"hs_2d_dispatched should has at most 2 tensors, but bot {len(hs_2d_dispatched)}"
            )
            hidden_states, scale = hs_2d_dispatched
        else:
            hidden_states, scale = hs_2d_dispatched, None

        with paddle.amp.auto_cast(False):
            (
                unzipped_tokens,
                zipped_expertwise_rowmap,
                unzipped_probs,
                unzipped_scale,
            ) = paddle.nn.functional.moe_permute(
                hidden_states,
                scale,
                dispatched_indices,
                dispatched_probs,
                num_experts=num_experts,
                tokens_per_expert=tokens_per_expert,
                padding_alignment=FP8_ALIGN,
            )

        if scale is None:
            # NOTE: 由于自定义算子不能返回None, 所以scale为None时
            # unzipped_scale会返回一个0shape的fake ouutput
            assert unzipped_scale.shape[0] == 0
            unzipped_scale = None

        self.unzipped_probs = unzipped_probs
        self.zipped_expertwise_rowmap = zipped_expertwise_rowmap
        return (
            unzipped_tokens,
            zipped_expertwise_rowmap,
            unzipped_probs,
            unzipped_scale,
        )

    @paddle.no_grad()
    def backward(
        self,
        dx,
        hidden_states_out_grad_shape,
        probs_grad,
        dispatched_indices,
        num_experts,
    ):
        with paddle.amp.auto_cast(False):
            weighted_zipped_tokens, probs_grad_zipped = (
                paddle.nn.functional.moe_unpermute(
                    dx,
                    self.zipped_expertwise_rowmap,
                    dispatched_indices,
                    probs_grad,
                    total_zipped_tokens=hidden_states_out_grad_shape[0],
                    num_experts=num_experts,
                )
            )
        self.reset_state()
        return weighted_zipped_tokens, probs_grad_zipped


class ZipNode:
    """
    与 UnzipNode 相反，类用将解压后的 token 张量压缩回原始状态。
    """

    def __init__(self, token_dispatcher, name="zip"):
        self.token_dispatcher = token_dispatcher
        self.name = name

    def cached_tensors(self):
        """
        cached_tensors
        """
        return []

    def set_cached_tensors(self, tensors):
        """
        set_cached_tensors
        """
        assert len(tensors) == 0

    def clear_cached_tensors(self):
        """
        clear_cached_tensors
        """
        pass

    @paddle.no_grad()
    def forward(
        self,
        expert_out,
        zipped_expertwise_rowmap,
        routemap_topk,
        unzipped_probs,
        total_zipped_tokens,
        num_experts,
    ):
        with paddle.amp.auto_cast(False):
            expert_out_zipped, zipped_probs_topk = (
                paddle.nn.functional.moe_unpermute(
                    expert_out,
                    zipped_expertwise_rowmap,
                    routemap_topk,
                    unzipped_probs,
                    total_zipped_tokens,
                    num_experts,
                )
            )
        return expert_out_zipped

    @paddle.no_grad()
    def backward(
        self,
        grad_output,
        dispatched_indices,
        dispatched_probs,
        top_k,
        num_experts,
        tokens_per_expert,
        fill_output=True,
    ):
        with paddle.amp.auto_cast(False):
            (
                unzipped_grad,
                zipped_expertwise_rowmap_grad,
                unzipped_probs_grad,
                unzipped_scale_grad,
            ) = paddle.nn.functional.moe_permute(
                grad_output,
                None,
                dispatched_indices,
                dispatched_probs,
                num_experts,
                tokens_per_expert,
                padding_alignment=FP8_ALIGN,
            )
        return unzipped_grad


class MlpNode:
    """
    The FusedMoeLayer class includes operations for unzipping, expert computation, and zipping.
    """

    def __init__(
        self,
        custom_map,
        num_experts_per_tok,
        recompute_moe_gate_up=False,
        dequant_input=False,
        moe_expert_fusion=True,
        recompute_moe_premute=False,
        moe_subbatch_token_num_after_dispatch=None,
        use_bf16_gemm_weight_grad=False,
        use_fp8_mlp=True,
        moe_deep_gemm=True,
        moe_grouped_gemm=False,
    ):
        """
        Constructor
        """
        self.token_dispatcher = custom_map.token_dispatcher
        self.moe_expert_fusion = moe_expert_fusion
        if recompute_moe_premute:
            assert not moe_expert_fusion, (
                "moe_expert_fusion must be disabled when recompute_unzipped = True"
            )
            assert recompute_moe_gate_up, (
                "recompute_moe_gate_up must be enabled when recompute_moe_premute = True"
            )
            assert dequant_input, (
                "dequant_input must be enabled with recompute_moe_premute = True"
            )
        self.recompute_moe_premute = recompute_moe_premute

        self.moe_subbatch_token_num_after_dispatch = (
            moe_subbatch_token_num_after_dispatch
        )

        if self.moe_subbatch_token_num_after_dispatch is not None:
            assert (
                self.moe_subbatch_token_num_after_dispatch > 0
                and self.moe_subbatch_token_num_after_dispatch % FP8_ALIGN == 0
            ), self.moe_subbatch_token_num_after_dispatch
            assert not moe_expert_fusion, (
                "moe_expert_fusion must be disabled when moe_subbatch_token_num_after_dispatch > 0"
            )
            assert recompute_moe_gate_up, (
                "recompute_moe_gate_up must be enabled when moe_subbatch_token_num_after_dispatch > 0"
            )
            assert dequant_input, (
                "dequant_input must be enabled when moe_subbatch_token_num_after_dispatch > 0"
            )

        if not self.moe_expert_fusion:
            raise NotImplementedError(
                "moe_expert_fusion = False is not supported currently"
            )
        else:
            self.experts_group_gemm_node = ExpertsGroupGemmContiguousNode(
                custom_map,
                recompute_moe_gate_up=recompute_moe_gate_up,
                dequant_input=dequant_input,
                moe_subbatch_token_num_after_dispatch=moe_subbatch_token_num_after_dispatch,
                use_bf16_gemm_weight_grad=use_bf16_gemm_weight_grad,
                use_fp8_mlp=use_fp8_mlp,
                moe_deep_gemm=moe_deep_gemm,
                moe_grouped_gemm=moe_grouped_gemm,
            )
        self.unzip_node = UnZipNode(self.token_dispatcher)
        self.zip_node = ZipNode(self.token_dispatcher)
        self.hs_2d_dispatched_fp8 = None
        self.hs_2d_dispatched_scale = None
        self.dispatched_indices = None
        self.dispatched_probs = None
        self.unzipped_probs = None
        self.tokens_per_expert = (
            self.token_dispatcher._comm_manager.tokens_per_expert
        )
        self.padding_token_per_experts = [
            (x + FP8_ALIGN - 1) // FP8_ALIGN * FP8_ALIGN
            for x in self.tokens_per_expert
        ]
        self.token_offsets = [0]
        for padding_token in self.padding_token_per_experts:
            self.token_offsets.append(self.token_offsets[-1] + padding_token)
        self.router_topk = num_experts_per_tok
        self.use_fp8_mlp = use_fp8_mlp

    def cached_tensors(self):
        """
        cached tensors
        """
        if self.experts_group_gemm_node is not None:
            if not self.moe_expert_fusion:
                gemm_node_tensors = []
                for gemm_node in self.experts_group_gemm_node:
                    gemm_node_tensors.extend(gemm_node.cached_tensors())
            else:
                gemm_node_tensors = (
                    self.experts_group_gemm_node.cached_tensors()
                )
        else:
            gemm_node_tensors = []

        return (
            gemm_node_tensors
            + self.unzip_node.cached_tensors()
            + self.zip_node.cached_tensors()
            + [
                self.hs_2d_dispatched_fp8,
                self.hs_2d_dispatched_scale,
                self.dispatched_indices,
                self.dispatched_probs,
                self.unzipped_probs,
                self.tokens_per_expert,
                self.router_topk,
            ]
        )

    def set_cached_tensors(self, tensors):
        """
        set_cached_tensors
        """
        idx = 0
        if self.experts_group_gemm_node is not None:
            if not self.moe_expert_fusion:
                for expert_id, gemm_node in enumerate(
                    self.experts_group_gemm_node
                ):
                    num = len(gemm_node.cached_tensors())
                    gemm_node.set_cached_tensors(tensors[idx : idx + num])
                    idx += num
            else:
                num = len(self.experts_group_gemm_node.cached_tensors())
                self.experts_group_gemm_node.set_cached_tensors(
                    tensors[idx : idx + num]
                )
                idx += num

        num = len(self.unzip_node.cached_tensors())
        self.unzip_node.set_cached_tensors(tensors[idx : idx + num])
        idx += num

        num = len(self.zip_node.cached_tensors())
        self.zip_node.set_cached_tensors(tensors[idx : idx + num])
        idx += num

        (
            self.hs_2d_dispatched_fp8,
            self.hs_2d_dispatched_scale,
            self.dispatched_indices,
            self.dispatched_probs,
            self.unzipped_probs,
            self.tokens_per_expert,
            self.router_topk,
        ) = tensors[idx:]

    def clear_cached_tensors(self):
        """
        clear_cached_tensors
        """
        self.set_cached_tensors([None] * len(self.cached_tensors()))

    def reset_state(self):
        """
        重置所有状态变量。

        Args:
            无。

        Returns:
            无。

        """
        self.dispatched_indices = None
        self.dispatched_probs = None
        self.unzipped_probs = None
        self.tokens_per_expert = None
        self.padding_token_per_experts = None
        self.router_topk = None
        self.release_mem()

    def release_mem(self):
        """
            释放内存，将变量置为None。
        这个函数应该在程序结束时调用，以便释放不再需要的资源。

        Args:
            无参数。

        Returns:
            无返回值，直接修改了类实例中的变量。
        """
        if not self.moe_expert_fusion:
            for node in self.experts_group_gemm_node:
                node.reset_state()
        else:
            self.experts_group_gemm_node.reset_state()
        self.experts_group_gemm_node = None

    @paddle.no_grad()
    def forward(self, hs_2d_dispatched, dispatched_indices, dispatched_probs):
        """
        对输入数据进行前向传播计算。

        Args:
            hs_2d_dispatched (Tensor): 表示被分派到各个专家的输入数据。
            dispatched_indices (Tensor):表示输入数据被分派到的专家索引。
            dispatched_probs (Tensor): 表示输入数据被分派到各个专家的概率。

        Returns:
            Tensor: 经过前向传播计算后的输出数据。

        """
        use_fp8_dispatch_a2a = isinstance(hs_2d_dispatched, tuple)

        num_experts = len(self.tokens_per_expert)
        # 1 unzip
        self.dispatched_indices = dispatched_indices.to(paddle.int32)
        (
            unzipped_tokens,
            zipped_expertwise_rowmap,
            unzipped_probs,
            unzipped_scale,
        ) = self.unzip_node.forward(
            hs_2d_dispatched,
            self.dispatched_indices,
            dispatched_probs,
            topk=self.router_topk,
            num_experts=num_experts,
            tokens_per_expert=self.tokens_per_expert,
            fill_output=self.moe_expert_fusion,
        )
        self.unzipped_probs = unzipped_probs
        if not self.moe_expert_fusion:
            unzipped_tokens = None

        if use_fp8_dispatch_a2a:
            total_zipped_tokens = hs_2d_dispatched[0].shape[0]
            hidden_size = hs_2d_dispatched[0].shape[-1]
            hs_2d_dispatched[0]._record_stream()
            hs_2d_dispatched[1]._record_stream()
        else:
            total_zipped_tokens = hs_2d_dispatched.shape[0]
            hidden_size = hs_2d_dispatched.shape[-1]
            hs_2d_dispatched._record_stream()
        dispatched_indices._record_stream()
        dispatched_probs._record_stream()
        if self.dispatched_indices is not dispatched_indices:
            dispatched_indices._clear_to_zero_allocation()

        if not self.moe_expert_fusion:
            raise NotImplementedError(
                "moe_expert_fusion = False is not supported currently"
            )
        else:
            if not use_fp8_dispatch_a2a:
                hs_2d_dispatched._clear_to_zero_allocation()
            # 2 experts
            expert_out = self.experts_group_gemm_node.forward(
                unzipped_tokens,
                unzipped_probs,
                self.padding_token_per_experts,
                self.tokens_per_expert,
                output=unzipped_tokens,
                scale=unzipped_scale,  # maybe None
            )

            # 3 zip
            expert_out = expert_out.reshape([-1, expert_out.shape[-1]])

            expert_out = self.zip_node.forward(
                expert_out,
                zipped_expertwise_rowmap,
                self.dispatched_indices,
                unzipped_probs,
                total_zipped_tokens=total_zipped_tokens,
                num_experts=num_experts,
            )

        self.dispatched_probs = dispatched_probs
        expert_out.stop_gradient = False

        return expert_out

    @paddle.no_grad()
    def backward(self, hidden_states_out_grad):
        """
        反向传播函数。

        Args:
            hidden_states_out_grad (Tensor): 隐藏状态梯度。

        Returns:
            Tuple[Tensor, Tensor]: 包含两个元素，分别为hs_fp8_dispatched_grad和dispatched_probs_grad。
                - hs_fp8_dispatched_grad (Tensor): 解压后的隐藏状态梯度。
                - dispatched_probs_grad (Tensor): 分发概率梯度。

        """
        # zip_grad
        hidden_states_out_grad_shape = hidden_states_out_grad.shape
        unzipped_grad = self.zip_node.backward(
            hidden_states_out_grad,
            self.dispatched_indices,
            self.dispatched_probs,
            top_k=self.router_topk,
            num_experts=len(self.tokens_per_expert),
            tokens_per_expert=self.tokens_per_expert,
            fill_output=self.moe_expert_fusion,
        )
        hidden_states_out_grad._record_stream()

        if not self.moe_expert_fusion:
            raise NotImplementedError(
                "moe_expert_fusion = False is not supported currently"
            )
        else:
            hidden_states_out_grad._clear_to_zero_allocation()

            # expert_grad
            expert_out, probs_grad = self.experts_group_gemm_node.backward(
                unzipped_grad, self.unzipped_probs
            )
            del unzipped_grad

            hs_fp8_dispatched_grad, dispatched_probs_grad = (
                self.unzip_node.backward(
                    expert_out,
                    hidden_states_out_grad_shape,
                    probs_grad,
                    self.dispatched_indices,
                    num_experts=len(self.tokens_per_expert),
                )
            )
        self.reset_state()
        return hs_fp8_dispatched_grad, dispatched_probs_grad


class FusionMoePyLayer(paddle.autograd.PyLayer):
    """
    The Fp8FusedMoeFunc class includes operations for unzipping, expert computation, and zipping.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        dispatched_probs,
        dispatched_indices,
        custom_map,
        num_experts_per_tok,
        use_fp8_mlp=True,
        moe_deep_gemm=True,
        moe_grouped_gemm=False,
        recompute_moe_gate_up=False,
        dequant_input=True,
        moe_expert_fusion=True,
        recompute_moe_premute=False,
        moe_subbatch_token_num_after_dispatch=None,
        use_bf16_gemm_weight_grad=False,
        is_first_fwd=False,
        fp8_dispatched_handle=None,
    ):
        """
        根据给定的参数执行前向传播操作。

        Args:
            hidden_states (tensor): 输入的隐藏状态张量。
            dispatched_probs (tensor): 分派概率张量。
            dispatched_indices (tensor): 分派索引张量。
            num_experts_per_tok (int): topk。

        Returns:
            tensor: 前向传播的结果张量。
        """
        ctx.node = MlpNode(
            custom_map,
            num_experts_per_tok,
            recompute_moe_gate_up=recompute_moe_gate_up,
            dequant_input=dequant_input,
            moe_expert_fusion=moe_expert_fusion,
            recompute_moe_premute=recompute_moe_premute,
            moe_subbatch_token_num_after_dispatch=moe_subbatch_token_num_after_dispatch,
            use_bf16_gemm_weight_grad=use_bf16_gemm_weight_grad,
            use_fp8_mlp=use_fp8_mlp,
            moe_deep_gemm=moe_deep_gemm,
            moe_grouped_gemm=moe_grouped_gemm,
        )

        if fp8_dispatched_handle is not None:
            assert hidden_states.dtype == paddle.float8_e4m3fn
            scale = fp8_dispatched_handle["scale"]
            hidden_states = (hidden_states, scale)

        out = ctx.node.forward(
            hidden_states, dispatched_indices, dispatched_probs
        )

        if is_first_fwd:
            ctx.node.release_mem()

        cached_tensors = ctx.node.cached_tensors()
        ctx.save_for_backward(cached_tensors)
        ctx.node.clear_cached_tensors()
        return out

    @staticmethod
    def backward(ctx, output_grad):
        """
        计算反向传播梯度。

        Args:
            output_grad (Tensor): 输出梯度张量。

        Returns:
            Tuple[Tensor, Tensor, None]: 返回三个梯度张量，前两个分别是隐藏状态和派发概率的梯度，
                                            第三个为None，表示没有需要传递给更前向节点的梯度。

        """
        (cached_tensors,) = ctx.saved_tensor()
        ctx.node.set_cached_tensors(cached_tensors)
        hidden_states_grad, dispatched_probs_grad = ctx.node.backward(
            output_grad
        )
        return hidden_states_grad, dispatched_probs_grad, None
