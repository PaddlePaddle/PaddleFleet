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

"""Behavior tests for paddlefleet.peft.lora.loraga_utils.

Production behaviors verified here:

- set_hook_enable / get_hook_enable global flag, and the effect of
  GradientOffloadHookContext.__enter__ / __exit__ on that flag, observed
  through the real context manager.
- GradientOffloadHookContext gradient recording: the registered backward
  hook stores ``param.grad / loraga_init_iters`` under a ``local_rank``
  suffixed key, accumulates across iterations, consumes ``local_rank``, and
  records nothing while the flag is disabled.
- loraga_svd_module: SVD-based LoRA-GA re-init writes A/B into both the live
  module parameters and the init dict under prefix-stripped keys, applies the
  correct ``stable_gamma`` vs. ``scaling`` branch factor, and re-initializes
  the base weight to ``W - scaling * (A @ B)``.

No-card (CPU) test. paddle is required for the real forward/backward and the
linalg SVD path; none of the code under test is mocked.
"""

import unittest

import numpy as np
import paddle
from paddle import nn

from paddlefleet.peft.lora import LoRALinear
from paddlefleet.peft.lora.loraga_utils import (
    GradientOffloadHookContext,
    get_hook_enable,
    loraga_svd_module,
    set_hook_enable,
)


class TestHookEnableFlag(unittest.TestCase):
    """The context manager drives the global ENABLE_HOOK flag."""

    def setUp(self):
        set_hook_enable(False)
        self.addCleanup(set_hook_enable, False)

    def test_context_toggles_flag_via_real_enter_exit(self):
        model = nn.Linear(3, 2)
        ctx = GradientOffloadHookContext(
            model=model,
            gradient_dict={},
            local_rank=0,
            loraga_init_iters=2,
        )
        self.assertFalse(get_hook_enable())
        with ctx:
            self.assertTrue(get_hook_enable())
        self.assertFalse(get_hook_enable())


class TestGradientRecordHook(unittest.TestCase):
    """The backward hook records scaled, rank-keyed gradients.

    For ``y = x @ W + b`` with ``loss = y.sum()`` the gradients are
    independent of the parameter values:
        dL/dW[i, j] = sum_b x[b, i]        (identical across output j)
        dL/db[j]    = batch_size
    so the expected values below are hand-derived from ``x`` alone.
    """

    def setUp(self):
        set_hook_enable(False)
        self.addCleanup(set_hook_enable, False)
        self.x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        # column sums of x over the batch, broadcast across the 2 outputs.
        self.expected_dw = np.array(
            [[5.0, 5.0], [7.0, 7.0], [9.0, 9.0]], dtype=np.float32
        )
        self.expected_db = np.array([2.0, 2.0], dtype=np.float32)

    def test_single_pass_records_scaled_gradient(self):
        model = nn.Linear(3, 2)
        grad_dict = {}
        with GradientOffloadHookContext(
            model=model,
            gradient_dict=grad_dict,
            local_rank=0,
            loraga_init_iters=2,
            gradient_offload=False,
        ):
            model(self.x).sum().backward()
        self.assertEqual(set(grad_dict), {"weight_0", "bias_0"})
        np.testing.assert_allclose(
            grad_dict["weight_0"].numpy(), self.expected_dw / 2, atol=1e-5
        )
        np.testing.assert_allclose(
            grad_dict["bias_0"].numpy(), self.expected_db / 2, atol=1e-5
        )

    def test_accumulates_across_iterations(self):
        model = nn.Linear(3, 2)
        grad_dict = {}
        with GradientOffloadHookContext(
            model=model,
            gradient_dict=grad_dict,
            local_rank=0,
            loraga_init_iters=2,
            gradient_offload=False,
        ):
            # two passes each contributing grad/2 sum to the full gradient.
            for _ in range(2):
                model(self.x).sum().backward()
        np.testing.assert_allclose(
            grad_dict["weight_0"].numpy(), self.expected_dw, atol=1e-5
        )
        np.testing.assert_allclose(
            grad_dict["bias_0"].numpy(), self.expected_db, atol=1e-5
        )

    def test_local_rank_appended_to_key(self):
        model = nn.Linear(3, 2)
        grad_dict = {}
        with GradientOffloadHookContext(
            model=model,
            gradient_dict=grad_dict,
            local_rank=3,
            loraga_init_iters=1,
            gradient_offload=False,
        ):
            model(self.x).sum().backward()
        self.assertEqual(set(grad_dict), {"weight_3", "bias_3"})
        np.testing.assert_allclose(
            grad_dict["weight_3"].numpy(), self.expected_dw, atol=1e-5
        )

    def test_disabled_flag_records_nothing(self):
        model = nn.Linear(3, 2)
        grad_dict = {}
        ctx = GradientOffloadHookContext(
            model=model,
            gradient_dict=grad_dict,
            local_rank=0,
            loraga_init_iters=1,
        )
        # __enter__ registers the hooks; __exit__ turns the flag off. The
        # hooks stay attached, so a later backward exercises the real
        # get_hook_enable() gate, which must suppress all recording.
        with ctx:
            pass
        self.assertFalse(get_hook_enable())
        model(self.x).sum().backward()
        self.assertEqual(grad_dict, {})


class TestLoragaSvdModule(unittest.TestCase):
    """SVD-based LoRA-GA re-initialization of adapters and base weight."""

    IN, OUT, R, ALPHA = 16, 16, 2, 8

    def _make_module(self):
        return LoRALinear(
            in_features=self.IN,
            out_features=self.OUT,
            r=self.R,
            lora_alpha=self.ALPHA,
        )

    def _fixed_grads(self):
        # Deterministic, full-rank, content-distinguishable weight gradient
        # of shape [in_features, out_features].
        rng = np.random.RandomState(0)
        base = rng.standard_normal((self.IN, self.OUT)).astype(np.float32)
        return paddle.to_tensor(base, dtype="float32")

    def test_reinit_writes_keys_params_and_base_weight(self):
        module = self._make_module()
        self.assertEqual(module.scaling, self.ALPHA / self.R)
        grads = self._fixed_grads()
        original_weight = module.weight.numpy().copy()

        init_dict = {}
        loraga_svd_module(
            name="base_model.layer0.q_proj",
            module=module,
            grads=grads,
            stable_gamma=8,
            loraga_init_dict=init_dict,
        )

        # The leading name segment is stripped when building the keys.
        self.assertEqual(
            set(init_dict),
            {"layer0.q_proj.lora_A", "layer0.q_proj.lora_B"},
        )
        a = init_dict["layer0.q_proj.lora_A"]
        b = init_dict["layer0.q_proj.lora_B"]
        self.assertEqual(list(a.shape), [self.IN, self.R])
        self.assertEqual(list(b.shape), [self.R, self.OUT])

        # The dict values are exactly what was written into the live params.
        np.testing.assert_allclose(a.numpy(), module.lora_A.numpy(), atol=1e-6)
        np.testing.assert_allclose(b.numpy(), module.lora_B.numpy(), atol=1e-6)

        # Base weight is re-initialized to W - scaling * (A @ B). The
        # reference is derived independently (numpy matmul) from the returned
        # adapters, not by re-running the code under test.
        expected_weight = original_weight - module.scaling * (
            a.numpy() @ b.numpy()
        )
        np.testing.assert_allclose(
            module.weight.numpy(), expected_weight, atol=1e-4
        )

    def test_stable_gamma_and_scaling_branches_differ_by_known_factor(self):
        # Both calls run the same SVD (fixed grads + identical paddle.seed),
        # so the adapters differ only by the per-branch scalar factor:
        #   stable_gamma != -1:  A = A_raw * m**0.25 / gamma**0.5
        #   stable_gamma == -1:  A = A_raw / scaling
        # => A_stable == A_scaled * scaling * m**0.25 / gamma**0.5.
        grads = self._fixed_grads()
        gamma = 8
        m = self.IN

        mod_stable = self._make_module()
        paddle.seed(20240101)
        loraga_svd_module(
            name="p.q_proj",
            module=mod_stable,
            grads=grads,
            stable_gamma=gamma,
            loraga_init_dict={},
        )
        a_stable = mod_stable.lora_A.numpy()
        b_stable = mod_stable.lora_B.numpy()

        mod_scaled = self._make_module()
        paddle.seed(20240101)
        loraga_svd_module(
            name="p.q_proj",
            module=mod_scaled,
            grads=grads,
            stable_gamma=-1,
            loraga_init_dict={},
        )
        a_scaled = mod_scaled.lora_A.numpy()
        b_scaled = mod_scaled.lora_B.numpy()

        factor = mod_scaled.scaling * (m**0.25) / (gamma**0.5)
        # A real, non-trivial difference between the two branches.
        self.assertNotAlmostEqual(factor, 1.0, places=3)
        self.assertFalse(np.allclose(a_stable, a_scaled))

        np.testing.assert_allclose(
            a_stable, a_scaled * factor, rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            b_stable, b_scaled * factor, rtol=1e-4, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
