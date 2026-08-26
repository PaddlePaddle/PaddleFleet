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
"""Single-card tests for fine-grained activation offloading on a real GPTModel.

Offloading must be numerically invisible: it only changes *where* an activation
lives between forward and backward. So the bar is bit-exact loss and gradients
versus a run with the feature off -- not "close enough". Anything looser cannot
tell a harmless floating-point reordering from a real stream/event race, and
those races are exactly what this machinery can get wrong.

Run with the repository copy of PaddleFleet, not the one in site-packages::

    PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=0 python -m pytest \
        tests/single_card_tests/model/test_gpt_model_activation_offload.py -v
"""

from __future__ import annotations

import functools
import random
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import NoPipelineParallel

import paddlefleet.parallel_state as ps
from paddlefleet.activation_offload import (
    manager_from_config,
    reset_offload_manager,
)
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig

_HAS_GPU = (
    paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
)
_REQUIRE_GPU = unittest.skipUnless(_HAS_GPU, "Requires a CUDA device")

ATTENTION_MODULES = ["attn_norm", "qkv_linear", "core_attn", "attn_proj"]


@_REQUIRE_GPU
class TestGPTModelActivationOffload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": 1,
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": 1,
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
        ps.initialize_model_parallel(fleet.get_hybrid_communicate_group())

    def setUp(self):
        seed = 46
        random.seed(seed)
        np.random.seed(seed)
        paddle.manual_seed(seed)
        self.strategy = fleet.fleet._user_defined_strategy

    # ---------------- fixtures ----------------

    @staticmethod
    def _base_config(**overrides):
        kwargs = {
            "num_hidden_layers": 2,
            "hidden_size": 512,
            "vocab_size": 100,
            "max_sequence_length": 64,
            "num_attention_heads": 4,
            "moe_expert_fusion": False,
            "intermediate_size": 1024,
            "normalization": "RMSNorm",
            "hidden_dropout_prob": 0.0,
            "first_k_dense_replace": 1,
            "attention_dropout": 0.0,
            "n_routed_experts": 8,
            "use_bias": False,
            "rotary_percent": 1.0,
            "rotary_base": 10000,
            "rope_scaling": 1.0,
            "moe_intermediate_size": 1024,
            "moe_token_dispatcher_type": "alltoall",
            "n_shared_experts": 1,
            "init_method": functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            "output_layer_init_method": functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            "tie_word_embeddings": True,
            "use_qk_norm": True,
            "recompute_granularity": None,
            "recompute_modules": [],
            # Binding NUMA would change the CPU affinity of the whole test
            # process, which a test must not do as a side effect.
            "activation_offload_numa_bind": False,
        }
        kwargs.update(overrides)  # overrides may replace any of the above
        return GPTConfig(**kwargs)

    @classmethod
    def _offload_config(cls, modules, **kw):
        # 卸载字段必须作为构造参数传进去:GPTConfig 是 dataclass,构造完再改字段
        # 就得重跑 __post_init__ 才能过校验,而 __post_init__ 不是幂等的
        # (它会把 moe_layer_freq 推导成非 int,第二次跑就撞上
        # "Cannot specify both first_k_dense_replace and moe_layer_freq")。
        return cls._base_config(
            fine_grained_activation_offloading=True,
            offload_modules=list(modules),
            # 这个测试形状的激活只有几百 KB;默认 2MB 门限会把它们全滤掉,测不到东西。
            min_offloaded_tensor_bytes=1,
            **kw,
        )

    def _prepare_input_data(self, config):
        seq = config.max_sequence_length
        data = list(range(seq))
        ids = paddle.to_tensor(data, dtype=paddle.int64).repeat((1, 1))
        labels = paddle.to_tensor(
            list(range(1, seq + 1)), dtype=paddle.int64
        ).repeat((1, 1))
        mask = paddle.ones((1, 1, seq, seq), dtype=bool)
        return (
            {
                "input_ids": [ids],
                "position_ids": [ids],
                "attention_mask": [mask],
            },
            [labels],
        )

    # ---------------- driver ----------------

    @staticmethod
    def _manager(config, offload):
        """按 config 建 manager。

        必须在建模型之前调:``TransformerLayer.__init__`` 会自己去取这个单例,
        它拿到的必须是这里建的这一个。``reset_offload_manager()`` 是因为单例只认
        第一次调用的 kwargs —— 同一个进程里跑多套配置,不重置就全沿用第一套。
        """
        reset_offload_manager()
        mgr = manager_from_config(config)
        mgr.enabled = offload
        return mgr

    def _run(self, config, offload, mgr=None):
        """One forward+backward. Returns (loss, {param: grad}, mgr.stats)."""
        if mgr is None:
            mgr = self._manager(config, offload)
        mgr.reset_stats()
        # 单卡没有 fleet 的 micro-step 回调,手动开组;组名 key 任意但要一致。
        if offload:
            mgr.begin_forward_group(0)

        model = gpt_builder(config, num_stages=1)
        data = self._prepare_input_data(config)
        # 不在这里手工进 scope:scope 由 TransformerLayer.forward 自己按配置开,
        # 测的就是那条生产路径。
        loss = NoPipelineParallel(
            model, self.strategy
        ).forward_backward_pipeline(data)

        grads = {
            name: param.grad.detach().clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        stats = dict(mgr.stats)
        if offload:
            mgr.clear_current_group()
            mgr.end_iteration()
        return loss.item(), grads, stats

    def _assert_bit_exact(self, ref, got, tag):
        ref_loss, ref_grads, _ = ref
        got_loss, got_grads, _ = got
        self.assertEqual(
            ref_loss, got_loss, f"{tag}: loss差异 {ref_loss} vs {got_loss}"
        )
        self.assertEqual(
            set(ref_grads), set(got_grads), f"{tag}: 梯度参数集合不同"
        )
        for name, ref_grad in ref_grads.items():
            np.testing.assert_array_equal(
                ref_grad.astype("float32").numpy(),
                got_grads[name].astype("float32").numpy(),
                err_msg=f"{tag}: {name} 梯度不是 bit 级一致",
            )

    # ---------------- tests ----------------

    def test_attention_modules_are_bit_exact(self):
        """全部注意力 module 打开,loss 与逐参数梯度必须 bit 级一致。"""
        self.setUp()
        ref = self._run(self._base_config(), offload=False)
        self.setUp()
        got = self._run(self._offload_config(ATTENTION_MODULES), offload=True)
        self.assertGreater(
            got[2]["packed"], 0, "一张都没卸载,说明区间没进或门限把张量全滤了"
        )
        self._assert_bit_exact(ref, got, "全部注意力 module")

    def test_single_module_offloads_less_than_all_modules(self):
        """只开一个 module 必须比全开卸得少 —— 区间的选择真的在起作用。"""
        self.setUp()
        one = self._run(self._offload_config(["core_attn"]), offload=True)
        self.setUp()
        every = self._run(self._offload_config(ATTENTION_MODULES), offload=True)
        self.assertGreater(one[2]["packed"], 0, "core_attn 一张都没卸载")
        self.assertLess(
            one[2]["packed"],
            every[2]["packed"],
            "只开一个 module 却和全开卸了同样多的张量,区间划分没起作用",
        )

    def test_region_offloads_whole_module_not_one_tensor(self):
        """区间语义的核心断言:一个 module 一次卸的是**多张**,不是一张。

        `core_attn` 的反向要用 q/k/v(以及 softmax 中间量),所以 2 层模型上
        `packed` 必须显著多于 2。这一条正是"对齐 Megatron 的粒度"的回归防线 ——
        退回"只标 module 输出那一张"的写法时,这里会立刻掉到 2。
        """
        self.setUp()
        got = self._run(self._offload_config(["core_attn"]), offload=True)
        n_layers = self._base_config().num_hidden_layers
        self.assertGreater(
            got[2]["packed"],
            n_layers,
            f"{n_layers} 层只卸了 {got[2]['packed']} 张,说明退回了"
            "'一个名字一张张量'的旧语义",
        )

    def test_expert_fc1_is_bit_exact(self):
        """MoE 专家的 fc1 输出(最大的一张 MoE 激活)也要能卸且 bit 级一致。

        ``moe_expert_fusion=True`` 是必须的:这个边界在 ``GroupedMLPExpert`` 里,
        而默认的 ``False`` 会走每专家一个 ``StandardMLPExpert``,压根没有这个点。
        """
        self.setUp()
        # moe_deep_gemm=False:DeepGEMM 只吃 bf16。而融合专家的权重本身就是 bf16,
        # 所以整个模型也得跑 bf16,否则 grouped GEMM 的两个入参 dtype 不一致。
        # bf16 不影响"bit 级一致"这条判据:卸载不改变任何一次计算。
        fused = {
            "moe_expert_fusion": True,
            "moe_deep_gemm": False,
            "bf16": True,
            "params_dtype": paddle.bfloat16,
        }
        ref = self._run(self._base_config(**fused), offload=False)
        self.setUp()
        got = self._run(
            self._offload_config(["expert_fc1"], **fused), offload=True
        )
        self.assertGreater(
            got[2]["packed"],
            0,
            "expert_fc1 一张都没卸载:要么没走 GroupedMLPExpert(融合的 "
            "SonicMoE 路径在 Python 侧没有这个边界),要么打标点没生效",
        )
        self._assert_bit_exact(ref, got, "expert_fc1")

    def test_moe_act_is_bit_exact(self):
        """``moe_act`` 区间卸的是 fc1 的输出,与 ``expert_fc1`` 是两张不同的张量。

        gated_linear_unit 下 fc1 输出是 gate+up 拼接,宽度两倍
        ``moe_intermediate_size``,是 MoE 里最大的一张激活。同样只在
        ``GroupedMLPExpert`` 上存在。
        """
        self.setUp()
        fused = {
            "moe_expert_fusion": True,
            "moe_deep_gemm": False,
            "bf16": True,
            "params_dtype": paddle.bfloat16,
        }
        ref = self._run(self._base_config(**fused), offload=False)
        self.setUp()
        got = self._run(
            self._offload_config(["moe_act"], **fused), offload=True
        )
        self.assertGreater(got[2]["packed"], 0, "moe_act 一张都没卸载")
        self._assert_bit_exact(ref, got, "moe_act")

    def test_moe_act_and_expert_fc1_offload_different_tensors(self):
        """两个 MoE 边界一起开,卸的张量数应当多于任一单独开。"""
        self.setUp()
        fused = {
            "moe_expert_fusion": True,
            "moe_deep_gemm": False,
            "bf16": True,
            "params_dtype": paddle.bfloat16,
        }
        ref = self._run(self._base_config(**fused), offload=False)
        self.setUp()
        fc1 = self._run(self._offload_config(["expert_fc1"], **fused), True)
        self.setUp()
        act = self._run(self._offload_config(["moe_act"], **fused), True)
        self.setUp()
        both = self._run(
            self._offload_config(["expert_fc1", "moe_act"], **fused), True
        )
        self.assertEqual(
            both[2]["packed"],
            fc1[2]["packed"] + act[2]["packed"],
            "两个边界卸的应该是不同的张量,一起开的数量该是各自之和",
        )
        self._assert_bit_exact(ref, both, "expert_fc1+moe_act")

    def test_fraction_zero_offloads_nothing(self):
        """fraction=0 是"一个边界都不卸",且仍要 bit 级一致。"""
        self.setUp()
        ref = self._run(self._base_config(), offload=False)
        config = self._offload_config(
            ATTENTION_MODULES, activation_offload_fraction=0.0
        )
        # 两个迭代必须共用同一个 manager:学习期的观察结果(边界序号 → 字节数)
        # 存在 manager 上,换一个实例就等于重新学一遍。
        mgr = self._manager(config, offload=True)
        self.setUp()
        first = self._run(config, offload=True, mgr=mgr)  # 学习期:全卸
        self.setUp()
        second = self._run(config, offload=True, mgr=mgr)  # 定档后:全不卸
        self.assertGreater(first[2]["packed"], 0)
        self.assertEqual(
            second[2]["packed"], 0, "fraction=0 之后不该再卸载任何边界"
        )
        self._assert_bit_exact(ref, second, "fraction=0")


class TestOffloadConfigValidation(unittest.TestCase):
    """``_validate_activation_offloading`` 的规则。纯配置,不需要 GPU。

    重算与卸载**不是**互斥的:卸载搬走的是区间的输入,正好是重算必须留在显存里
    的那一张,两者叠加显存更低(实测 4 层 GPTModel 激活峰值 713.5 → 542.5 →
    507.2MB,且 bit 级一致)。唯一要拦的是"区间嵌在 checkpoint 内部"——那种情况
    前向什么都不存,区间改在反向重放时被进入,每次卸载都变成一次白跑的
    D2H+H2D 往返。逐组合的实测见
    ``probes/probe_region_inside_recompute.py``。
    """

    @staticmethod
    def _config(offload, recompute):
        return TestGPTModelActivationOffload._offload_config(
            offload,
            recompute_granularity="selective" if recompute else None,
            recompute_modules=list(recompute),
            recompute_num_layers=None,
        )

    def test_same_module_in_recompute_and_offload_is_allowed(self):
        for name in ("core_attn", "attn_norm", "mlp_norm"):
            with self.subTest(name=name):
                # 名字对得上的那几组(core_attn / norm)在两边同时出现是合法的。
                rc = "norm" if name.endswith("norm") else name
                self._config([name] if name != "attn_proj" else [], [rc])

    def test_moe_gate_up_recompute_allowed_with_expert_fc1(self):
        # moe_gate_up 重算的是 fc1 的输出,与 expert_fc1 区间的输入不是同一张。
        self._config(["expert_fc1"], ["moe_gate_up"])

    def test_mlp_recompute_rejects_moe_internal_offload(self):
        for name in ("expert_fc1", "moe_act", "fused_group_mlp"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "backward replay"),
            ):
                self._config([name], ["mlp"])

    def test_mlp_recompute_allows_attention_offload(self):
        # mlp 重算不覆盖注意力段的任何区间。
        self._config(ATTENTION_MODULES, ["mlp"])

    def test_full_recompute_is_rejected(self):
        # full 把整层包进 checkpoint,所有区间都只在反向重放时被进入 —— 实测
        # 前向 pack 0 张、反向 pack 全部,纯往返浪费。K3 的 GB200 配置就是 full,
        # 所以这条必须硬报错而不是静默降级。
        with self.assertRaisesRegex(ValueError, "recompute_granularity"):
            TestGPTModelActivationOffload._offload_config(
                ["core_attn"],
                recompute_granularity="full",
                recompute_method="uniform",
                recompute_num_layers=1,
            )

    def test_fused_group_mlp_excludes_finer_moe_boundaries(self):
        # fused_group_mlp 已经覆盖整个融合节点的 cached_tensors,与两个细粒度
        # MoE 边界互斥(Megatron 同规则)。
        for name in ("expert_fc1", "moe_act"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "fused_group_mlp"),
            ):
                self._config(["fused_group_mlp", name], [])
        # 单独用是合法的
        self._config(["fused_group_mlp"], [])

    def test_none_knobs_fall_back_to_defaults(self):
        """外层配置层把没写的旋钮写成 None 时，必须还原成 dataclass 默认。

        PaddleFormers 的 LlmMetaConfig 白名单会把**每一个**白名单 key 都写到模型
        config 上，包括 yaml 里根本没提的那些。所以这些字段会以 None 到达这里，
        而它们的默认值不是 None —— 不还原就会在范围检查里炸
        （GB200 上真实踩到过：`'<' not supported between NoneType and int`）。
        """
        config = TestGPTModelActivationOffload._base_config(
            fine_grained_activation_offloading=True,
            offload_modules=["core_attn", "attn_proj"],
            min_offloaded_tensor_bytes=None,
            activation_offload_fraction=None,
            delta_offload_bytes_across_pp_ranks=None,
            activation_offload_numa_bind=None,
            activation_offload_prefetch_budget_bytes=None,
            activation_offload_pool_capacity_bytes=None,
        )
        self.assertEqual(config.min_offloaded_tensor_bytes, 2 * 1024 * 1024)
        self.assertEqual(config.activation_offload_fraction, 1.0)
        self.assertEqual(config.delta_offload_bytes_across_pp_ranks, 0)
        self.assertIs(config.activation_offload_numa_bind, True)
        # 这两个的默认值本来就是 None，不该被动
        self.assertIsNone(config.activation_offload_prefetch_budget_bytes)
        self.assertIsNone(config.activation_offload_pool_capacity_bytes)

    def test_attn_proj_requires_core_attn(self):
        with self.assertRaisesRegex(ValueError, "requires 'core_attn'"):
            self._config(["attn_proj"], [])

    def test_offload_modules_without_master_switch(self):
        with self.assertRaisesRegex(ValueError, "master switch"):
            TestGPTModelActivationOffload._base_config(
                offload_modules=["core_attn"]
            )

    def test_unknown_offload_module(self):
        with self.assertRaisesRegex(ValueError, "Invalid offload_modules"):
            self._config(["not_a_module"], [])


if __name__ == "__main__":
    unittest.main()
