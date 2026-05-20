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

import random
import sys
import unittest

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle.distributed.fleet.utils import mix_precision_utils

paddle.compat.enable_torch_proxy(
    scope={"sonicmoe", "paddlefleet_ops.sonicmoe", "quack", "triton"},
    silent=True,
)
import paddlefleet_ops
from paddle.distributed import fleet

from paddlefleet.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
)
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.training.global_vars import unset_global_variables
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.moe.moe_layer import MoELayer
from paddlefleet.transformer.transformer_config import TransformerConfig

if paddlefleet_ops.is_sonic_moe_available():
    from paddlefleet_ops.sonicmoe.functional import clear_all_fp8_weight_caches

for _key in list(sys.modules):
    if _key.startswith("paddlefleet_ops.sonicmoe"):
        _alias = _key.replace("paddlefleet_ops.sonicmoe", "sonicmoe", 1)
        sys.modules.setdefault(_alias, sys.modules[_key])


@unittest.skipUnless(
    paddlefleet_ops.is_sonic_moe_available(),
    "Sonic-MoE not available (requires Python>=3.12, CUDA>=12.9, SM>=90)",
)
class TestSonicMoEExpertParallelPrecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": 8,
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": 8,
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
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    @classmethod
    def tearDownClass(cls):
        unset_global_variables()

    def setUp(self):
        self.seed = 123
        self.hidden_size = 2048
        self.n_routed_experts = 64

        random.seed(self.seed)
        np.random.seed(self.seed)
        paddle.seed(self.seed)
        paddle.manual_seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        self.pg_collection = self.__class__.pg_collection

    @staticmethod
    def calc_diff(x: paddle.Tensor, y: paddle.Tensor):
        x, y = x.double(), y.double()
        denominator = (x * x + y * y).sum()
        if denominator.item() == 0:
            return 0.0
        sim = 2 * (x * y).sum() / denominator
        return (1 - sim).item()

    def _build_transformer_config(
        self,
        using_sonic_moe=False,
        fp8=None,
        moe_deep_gemm=False,
        fp8_wgrad=True,
        expert_model_parallel_size=4,
    ):
        return TransformerConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            n_routed_experts=self.n_routed_experts,
            use_cpu_initialization=False,
            num_experts_per_tok=2,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=expert_model_parallel_size,
            sequence_parallel=False,
            bf16=True,
            params_dtype=paddle.bfloat16,
            moe_intermediate_size=1024,
            gated_linear_unit=True,
            n_shared_experts=0,
            hidden_act=F.silu,
            moe_grouped_gemm=True,
            moe_deep_gemm=moe_deep_gemm,
            bias_activation_fusion=True,
            moe_token_dispatcher_type="deepep",
            moe_use_fusion_node=True,
            using_sonic_moe=using_sonic_moe,
            fp8=fp8,
            fp8_wgrad=fp8_wgrad,
        )

    def _build_moe_layer(
        self,
        using_sonic_moe=False,
        fp8=None,
        moe_deep_gemm=False,
        fp8_wgrad=True,
        expert_model_parallel_size=4,
        pg_collection=None,
    ):
        transformer_config = self._build_transformer_config(
            using_sonic_moe=using_sonic_moe,
            fp8=fp8,
            moe_deep_gemm=moe_deep_gemm,
            fp8_wgrad=fp8_wgrad,
            expert_model_parallel_size=expert_model_parallel_size,
        )
        transformer_layer_spec = get_gpt_layer_local_spec(
            transformer_config,
            num_experts=self.n_routed_experts,
        )
        return MoELayer(
            transformer_config,
            transformer_layer_spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
            pg_collection or self.pg_collection,
        )

    @staticmethod
    def _split_to_sonic_interleaved(weight):
        gate, up = paddle.chunk(weight, 2, axis=-1)
        gate = gate.transpose([0, 2, 1])
        up = up.transpose([0, 2, 1])
        return paddle.stack([gate, up], axis=2).reshape(
            [weight.shape[0], -1, weight.shape[1]]
        )

    @staticmethod
    def _sonic_interleaved_to_split_grad(grad):
        grad = grad.reshape([grad.shape[0], -1, 2, grad.shape[2]])
        gate = grad[:, :, 0, :].transpose([0, 2, 1])
        up = grad[:, :, 1, :].transpose([0, 2, 1])
        return paddle.concat([gate, up], axis=-1)

    @classmethod
    def _aligned_grad_for_compare(
        cls, name, grad, transpose_grouped_gemm=False
    ):
        if not transpose_grouped_gemm:
            return grad
        if "grouped_gemm_experts.weight1" in name:
            return cls._sonic_interleaved_to_split_grad(grad)
        if "grouped_gemm_experts.weight2" in name:
            return grad.transpose([0, 2, 1])
        return grad

    @classmethod
    def _copy_weights(cls, src_layer, dst_layer, transpose_grouped_gemm=False):
        src_params = dict(src_layer.named_parameters())
        for name, dst_param in dst_layer.named_parameters():
            src_param = src_params[name]
            if (
                transpose_grouped_gemm
                and "grouped_gemm_experts.weight1" in name
            ):
                dst_param.set_value(cls._split_to_sonic_interleaved(src_param))
            elif (
                transpose_grouped_gemm
                and "grouped_gemm_experts.weight2" in name
            ):
                dst_param.set_value(src_param.transpose([0, 2, 1]))
            else:
                dst_param.set_value(src_param.clone())

    @staticmethod
    def _expert_slice_for_rank(tensor, ep_rank, ep_size):
        chunk_size = tensor.shape[0] // ep_size
        return tensor[ep_rank * chunk_size : (ep_rank + 1) * chunk_size]

    @classmethod
    def _copy_single_card_weights_to_ep(
        cls, src_layer, dst_layer, ep_rank, ep_size
    ):
        src_params = dict(src_layer.named_parameters())
        for name, dst_param in dst_layer.named_parameters():
            src_param = src_params[name]
            if "grouped_gemm_experts.weight" in name:
                src_param = cls._expert_slice_for_rank(
                    src_param, ep_rank, ep_size
                )
            dst_param.set_value(src_param.clone())

    @classmethod
    def _gather_ep_expert_grads(cls, grads, ep_group, ep_size):
        gathered_grads = {}
        for name, grad in grads.items():
            if "grouped_gemm_experts.weight" not in name:
                gathered_grads[name] = grad
                continue
            parts = []
            dist.all_gather(parts, grad, group=ep_group)
            gathered_grads[name] = paddle.concat(parts, axis=0) / ep_size
        return gathered_grads

    @staticmethod
    def _collect_grads(layer):
        grads = {}
        for name, param in layer.named_parameters():
            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
            if grad is not None:
                grads[name] = grad.detach().clone()
        return grads

    @staticmethod
    def _clear_grads(layer):
        for _, param in layer.named_parameters():
            if hasattr(param, "main_grad") and param.main_grad is not None:
                param.main_grad.zero_()
            if param.grad is not None:
                param.grad.zero_()

    def _run_forward_backward(self, moe_layer, input_data):
        moe_layer = paddle.amp.decorate(
            models=moe_layer,
            level="O2",
            dtype="bfloat16",
            master_grad=True,
            master_weight=True,
        )
        mix_precision_utils.MixPrecisionLayer(moe_layer, dtype="bfloat16")
        hidden_states = input_data.detach().clone()
        hidden_states.stop_gradient = False
        with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
            output = moe_layer(hidden_states)[0]
            loss = output.sum()
            # loss = paddle.mean(paddle.square(output.cast("float32")))
        loss.backward()
        return (
            output.detach().clone(),
            loss.item(),
            hidden_states.grad.detach().clone(),
            self._collect_grads(moe_layer),
        )

    def _run_accumulated_forward_backward(self, moe_layer, input_data_list):
        self._clear_grads(moe_layer)
        outputs = []
        for input_data in input_data_list:
            hidden_states = input_data.detach().clone()
            hidden_states.stop_gradient = False
            with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
                output = moe_layer(hidden_states)[0]
                loss = paddle.mean(paddle.square(output.cast("float32")))
            loss.backward()
            outputs.append(output.detach().clone())
        return outputs[-1], self._collect_grads(moe_layer)

    def _assert_loss_close(self, lhs, rhs, tol, title):
        loss_rdiff = abs(lhs - rhs) / max(abs(rhs), 1e-12)
        print(f"{title}: loss relative diff = {loss_rdiff:.6e}")
        self.assertLess(
            loss_rdiff,
            tol,
            f"{title} loss deviates too much: lhs={lhs}, rhs={rhs}",
        )

    def _assert_tensor_diff_less(self, lhs, rhs, tol, title):
        diff = self.calc_diff(lhs, rhs)
        print(f"{title}: diff = {diff:.6e}")
        self.assertLess(diff, tol, f"{title} diff too large: diff={diff:.6e}")

    def _assert_grad_diff_less(
        self,
        lhs_grads,
        rhs_grads,
        tol,
        title,
        transpose_grouped_gemm=False,
    ):
        lhs_names = set(lhs_grads)
        rhs_names = set(rhs_grads)
        self.assertEqual(
            lhs_names,
            rhs_names,
            (
                f"Gradient tensors mismatch for {title}: "
                f"lhs_only={sorted(lhs_names - rhs_names)}, "
                f"rhs_only={sorted(rhs_names - lhs_names)}"
            ),
        )
        self.assertTrue(lhs_names, f"No grad tensors found for {title}")
        for name in sorted(lhs_names):
            lhs_grad = self._aligned_grad_for_compare(
                name,
                lhs_grads[name],
                transpose_grouped_gemm=transpose_grouped_gemm,
            )
            grad_tol = tol[name] if isinstance(tol, dict) else tol
            self._assert_tensor_diff_less(
                lhs_grad,
                rhs_grads[name],
                tol=grad_tol,
                title=f"{title} grad {name}",
            )

    def run_test_sonic_moe_ep_grad_accumulation(self):
        acc_steps = 1

        paddle.seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        moe_layer_base = self._build_moe_layer(using_sonic_moe=False)
        paddle.seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        moe_layer_sonic_bf16 = self._build_moe_layer(using_sonic_moe=True)
        paddle.seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        moe_layer_sonic_fp8 = self._build_moe_layer(
            using_sonic_moe=True,
            fp8="e4m3",
        )

        input_data_list = []
        for step_idx in range(acc_steps):
            paddle.seed(self.seed + step_idx)
            input_data_list.append(
                paddle.randn(
                    [4, 256, self.hidden_size],
                    dtype=paddle.bfloat16,
                )
            )

        output_base, grads_base = self._run_accumulated_forward_backward(
            moe_layer_base, input_data_list
        )
        output_bf16, grads_bf16 = self._run_accumulated_forward_backward(
            moe_layer_sonic_bf16, input_data_list
        )
        output_fp8, grads_fp8 = self._run_accumulated_forward_backward(
            moe_layer_sonic_fp8, input_data_list
        )
        clear_all_fp8_weight_caches()

        self._assert_tensor_diff_less(
            output_bf16,
            output_base,
            tol=1e-2,
            title="Sonic-MoE BF16 vs Baseline final output",
        )
        self._assert_grad_diff_less(
            grads_bf16,
            grads_base,
            tol=1e-2,
            title="Sonic-MoE BF16 vs Baseline accumulated grad",
            transpose_grouped_gemm=True,
        )

        fp8_tol = 5e-3
        self._assert_tensor_diff_less(
            output_fp8,
            output_bf16,
            tol=fp8_tol,
            title="Sonic-MoE FP8 vs BF16 final output",
        )
        self._assert_grad_diff_less(
            grads_fp8,
            grads_bf16,
            tol=fp8_tol,
            title="Sonic-MoE FP8 vs BF16 accumulated grad",
        )

        print("Final output and parameter gradient precision checks passed!")

    def run_test_ep_precision(self):
        ep_size = self.pg_collection.ep.nranks
        ep_rank = dist.get_rank(self.pg_collection.ep)
        single_rank_group = dist.new_group([dist.get_rank()])
        single_pg_collection = ProcessGroupCollection(
            ep=single_rank_group,
            expt_dp=single_rank_group,
        )

        moe_layer_single = self._build_moe_layer(
            using_sonic_moe=True,
            fp8="e4m3",
            expert_model_parallel_size=1,
            pg_collection=single_pg_collection,
        )
        moe_layer_ep = self._build_moe_layer(
            using_sonic_moe=True,
            fp8="e4m3",
            expert_model_parallel_size=ep_size,
            pg_collection=self.pg_collection,
        )
        self._copy_single_card_weights_to_ep(
            moe_layer_single,
            moe_layer_ep,
            ep_rank,
            ep_size,
        )

        paddle.seed(self.seed + 2024)
        input_data = paddle.randn(
            [4, 256, self.hidden_size],
            dtype=paddle.bfloat16,
        )

        output_single, loss_single, input_grad_single, grads_single = (
            self._run_forward_backward(moe_layer_single, input_data)
        )
        output_ep, loss_ep, input_grad_ep, grads_ep = (
            self._run_forward_backward(moe_layer_ep, input_data)
        )
        grads_ep = self._gather_ep_expert_grads(
            grads_ep,
            self.pg_collection.ep,
            ep_size,
        )
        clear_all_fp8_weight_caches()

        self._assert_loss_close(
            loss_ep,
            loss_single,
            tol=1e-2,
            title="Sonic-MoE FP8 EP vs single-card",
        )
        self._assert_tensor_diff_less(
            output_ep,
            output_single,
            tol=5e-3,
            title="Sonic-MoE FP8 EP vs single-card output",
        )
        self._assert_tensor_diff_less(
            input_grad_ep,
            input_grad_single,
            tol=5e-3,
            title="Sonic-MoE FP8 EP vs single-card input grad",
        )
        self._assert_grad_diff_less(
            grads_ep,
            grads_single,
            tol=5e-3,
            title="Sonic-MoE FP8 EP vs single-card param grad",
        )

        print("Expert-parallel FP8 precision checks passed!")

    def test_sonic_moe_all(self):
        self.run_test_sonic_moe_ep_grad_accumulation()
        self.run_test_ep_precision()


if __name__ == "__main__":
    unittest.main()
