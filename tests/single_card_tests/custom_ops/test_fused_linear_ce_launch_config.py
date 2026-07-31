#!/usr/bin/env python3

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

"""单测：fused_linear_cross_entropy 的 CE kernel launch-config 调优。

被测优化：`_select_ce_launch_config` 从「BLOCK_SIZE=next_pow2(V) 上限 32768 +
num_warps=32 固定」改为「BLOCK_SIZE 上限 2048 + 按每线程 ~16 列的预算推导
num_warps」。CE kernel 在 `for i in range(0, n_cols, BLOCK_SIZE)` 中流式扫描
整行 vocab，因此 BLOCK_SIZE 只决定每个 program 的寄存器 tile 大小，不影响覆盖的
列数 —— 换言之这是纯 launch 调优，数值结果必须与旧配置完全一致。

覆盖两部分：
  1. TestCELaunchConfigSelection：纯 Python，校验 `_select_ce_launch_config`
     的返回值满足 2 的幂、上限、num_warps 边界、每线程列预算等不变量（无需 GPU）。
  2. TestCELaunchConfigInvariance：GPU，直接调用
     `fused_linear_cross_entropy_forward`，对比默认（小 tile）配置与被 monkeypatch
     成旧式（大 tile）配置下的 loss / grad_input / grad_weight / grad_bias，
     并与朴素 F.linear + cross_entropy 参考实现对齐，证明调优不改变数值。
"""

import unittest

import numpy as np
import paddle

# 注意：不要在这里 `import triton`。当环境没有真 torch 时（CI 即如此），Triton 的
# nvidia backend 靠 `import torch` 探活 driver，只有在 paddle 的 torch-compat
# proxy 已启用后导入 triton，其模块才会被标记为 compat 作用域内，`import torch`
# 才拿得到 proxy；否则 driver 探活失败，kernel 启动时报
# "0 active drivers"。paddlefleet 的导入会先启用 compat，所以 triton 必须晚于它
# 被导入 —— 需要 triton 时统一走 `flce_module.triton`。
from paddlefleet.triton_ops.fused_linear_cross_entropy import (
    fused_linear_cross_entropy as flce_module,
)
from paddlefleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy import (
    _select_ce_launch_config,
)


class TestCELaunchConfigSelection(unittest.TestCase):
    """纯 Python 校验 `_select_ce_launch_config` 的不变量（无需 GPU）。"""

    # 覆盖从极小到真实 lm_head 规模的 vocab。
    VOCABS = [1, 32, 100, 128, 511, 512, 1024, 2048, 4096, 32000, 201216]

    def test_block_size_is_power_of_two(self):
        for v in self.VOCABS:
            block_size, _ = _select_ce_launch_config(v)
            self.assertEqual(
                block_size & (block_size - 1),
                0,
                msg=f"block_size={block_size} 应为 2 的幂 (V={v})",
            )
            self.assertGreaterEqual(block_size, 1)

    def test_block_size_within_caps(self):
        for v in self.VOCABS:
            block_size, _ = _select_ce_launch_config(v)
            expected = min(
                flce_module.MAX_FUSED_SIZE,
                flce_module.triton.next_power_of_2(v),
                flce_module.CE_BLOCK_SIZE_CAP,
            )
            self.assertEqual(block_size, expected, msg=f"V={v}")
            # 关键调优点：block_size 永远不超过 2048 上限。
            self.assertLessEqual(block_size, flce_module.CE_BLOCK_SIZE_CAP)

    def test_num_warps_bounds(self):
        for v in self.VOCABS:
            _, num_warps = _select_ce_launch_config(v)
            self.assertGreaterEqual(num_warps, 1, msg=f"V={v}")
            self.assertLessEqual(num_warps, 32, msg=f"V={v}")

    def test_num_warps_matches_per_thread_budget(self):
        # num_warps 应由「每线程 ~CE_ELEMENTS_PER_THREAD 列」推导并夹在 [1,32]。
        for v in self.VOCABS:
            block_size, num_warps = _select_ce_launch_config(v)
            expected = block_size // (32 * flce_module.CE_ELEMENTS_PER_THREAD)
            expected = max(1, min(32, expected))
            self.assertEqual(num_warps, expected, msg=f"V={v}")

    def test_large_vocab_hits_tuned_target(self):
        # 大 vocab（如 201216）应命中调优目标 (2048 tile / 4 warps)，
        # 即每线程 2048/(4*32)=16 列，恰为 CE_ELEMENTS_PER_THREAD。
        block_size, num_warps = _select_ce_launch_config(201216)
        self.assertEqual(block_size, 2048)
        self.assertEqual(num_warps, 4)
        cols_per_thread = block_size // (num_warps * 32)
        self.assertEqual(cols_per_thread, flce_module.CE_ELEMENTS_PER_THREAD)

    def test_small_vocab_single_warp(self):
        # V 很小时 block_size<512，num_warps 被夹到 1（不会退化为 0）。
        for v in [1, 32, 128, 256, 512]:
            _, num_warps = _select_ce_launch_config(v)
            self.assertEqual(num_warps, 1, msg=f"V={v}")


@unittest.skipUnless(
    paddle.device.is_compiled_with_cuda()
    and paddle.device.cuda.device_count() > 0,
    "launch-config invariance 测试需要 GPU",
)
class TestCELaunchConfigInvariance(unittest.TestCase):
    """GPU：证明 launch-config 调优不改变前向 loss / 梯度数值。

    对同一组输入分别用「默认（新）小 tile 配置」与「monkeypatch 成旧式大 tile
    配置」调用 `fused_linear_cross_entropy_forward`，二者结果必须一致；再与朴素
    F.linear + cross_entropy 参考实现对齐。
    """

    ATOL = 1e-4
    RTOL = 1e-4

    def _run_forward(self, hidden, weight, bias, labels, reduction):
        _input = hidden.clone()
        _input.stop_gradient = False
        w = weight.clone()
        w.stop_gradient = False
        b = bias.clone()
        b.stop_gradient = False
        loss, gi, gw, gb = flce_module.fused_linear_cross_entropy_forward(
            _input=_input,
            weight=w,
            target=labels,
            bias=b,
            ignore_index=-100,
            reduction=reduction,
            num_chunks=1,
        )
        return (
            loss.numpy(),
            gi.numpy(),
            gw.numpy(),
            gb.numpy(),
        )

    def _reference(self, hidden, weight, bias, labels, reduction):
        logits = paddle.compat.nn.functional.linear(hidden, weight, bias)
        loss = paddle.nn.functional.cross_entropy(
            logits.cast("float32"),
            labels,
            ignore_index=-100,
            reduction="none",
        )
        # kernel 的 reduction="none" 返回每 token loss（忽略位置为 0）。
        loss = loss.reshape([-1])
        mask = (labels.reshape([-1]) != -100).cast("float32")
        loss = loss * mask
        return loss.numpy()

    def _make_case(self, V, seed=0):
        paddle.seed(seed)
        BT, H = 24, 48
        hidden = paddle.randn([BT, H], dtype="float32")
        weight = paddle.randn([V, H], dtype="float32") * 0.02
        bias = paddle.randn([V], dtype="float32") * 0.02
        labels = paddle.randint(0, V, [BT], dtype="int64")
        # 混入若干 ignore_index，覆盖清零分支。
        labels_np = labels.numpy()
        labels_np[::7] = -100
        labels = paddle.to_tensor(labels_np)
        return hidden, weight, bias, labels

    def test_forward_invariant_to_launch_config(self):
        # V=8192：默认配置 -> (2048, 4)；旧式配置 -> (8192, 32)。两者 tile 大小与
        # warp 数均不同，但数学完全一致，结果必须逐元素吻合。
        V = 8192
        hidden, weight, bias, labels = self._make_case(V)

        self.assertEqual(_select_ce_launch_config(V), (2048, 4))

        loss_new, gi_new, gw_new, gb_new = self._run_forward(
            hidden, weight, bias, labels, "none"
        )

        # Monkeypatch 成旧式：cap=32768 且每线程 1 列 -> num_warps 饱和到 32。
        orig_cap = flce_module.CE_BLOCK_SIZE_CAP
        orig_ept = flce_module.CE_ELEMENTS_PER_THREAD
        try:
            flce_module.CE_BLOCK_SIZE_CAP = flce_module.MAX_FUSED_SIZE
            flce_module.CE_ELEMENTS_PER_THREAD = 1
            self.assertEqual(_select_ce_launch_config(V), (8192, 32))
            loss_old, gi_old, gw_old, gb_old = self._run_forward(
                hidden, weight, bias, labels, "none"
            )
        finally:
            flce_module.CE_BLOCK_SIZE_CAP = orig_cap
            flce_module.CE_ELEMENTS_PER_THREAD = orig_ept

        np.testing.assert_allclose(
            loss_new,
            loss_old,
            atol=self.ATOL,
            rtol=self.RTOL,
            err_msg="loss 应与 launch-config 无关",
        )
        np.testing.assert_allclose(
            gi_new,
            gi_old,
            atol=self.ATOL,
            rtol=self.RTOL,
            err_msg="grad_input 应与 launch-config 无关",
        )
        np.testing.assert_allclose(
            gw_new,
            gw_old,
            atol=self.ATOL,
            rtol=self.RTOL,
            err_msg="grad_weight 应与 launch-config 无关",
        )
        np.testing.assert_allclose(
            gb_new,
            gb_old,
            atol=self.ATOL,
            rtol=self.RTOL,
            err_msg="grad_bias 应与 launch-config 无关",
        )

    def test_matches_reference(self):
        # 新配置下的 loss 与朴素参考实现对齐（含 ignore 分支）。
        V = 4096
        hidden, weight, bias, labels = self._make_case(V, seed=1)
        loss_new, _, _, _ = self._run_forward(
            hidden, weight, bias, labels, "none"
        )
        loss_ref = self._reference(hidden, weight, bias, labels, "none")
        np.testing.assert_allclose(
            loss_new,
            loss_ref,
            atol=1e-3,
            rtol=1e-3,
            err_msg="fused loss 应与 F.linear + cross_entropy 对齐",
        )


if __name__ == "__main__":
    unittest.main()
