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

"""Behaviour tests for ``paddlefleet.utils.optimizer`` (无卡 / CPU, fp32).

Scope and evidence boundary:
* ``AdamWMini.adamw_python`` and ``AdamWCustom.adamw_custom`` (non-HF branch)
  are pure AdamW update kernels that run on CPU in fp32. Every numeric check
  below derives the expected parameter, first/second moment and beta-power
  accumulators from an INDEPENDENT text-book AdamW step (``_ref_adamw_step``);
  none of them calls the function under test to build its own expectation.
* ``AdamWMini`` keeps a single shared scalar second moment (mean of g*g), while
  ``AdamWCustom`` keeps an element-wise second moment -- the fixtures use
  distinguishable per-element grads so the two are not interchangeable.
* State restore is checked end-to-end through the real ``optimizer.step``:
  a checkpoint is loaded into a *new* optimizer instance and training continues,
  and a negative control (weights restored but optimizer state fresh) is shown
  to diverge -- comparing only the weights is therefore not enough.
* The fp16/bf16 master-weight numerics are a GPU concern; the control logic
  (master-weight consumption, ``skip_update_param``, ``multi_precision`` gate)
  is verified on CPU in fp32, and the device-dtype numeric case is guarded by
  ``skipUnless(is_compiled_with_cuda)``.
"""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import paddle
from paddle import nn

from paddlefleet.utils.optimizer import AdamWCustom, AdamWMini


def setUpModule():
    # Keep everything on CPU: the fp32 AdamW kernels are CPU-runnable and this
    # removes any dependence on a visible accelerator for the numeric checks.
    paddle.set_device("cpu")


def _ref_adamw_step(
    param,
    grad,
    moment1,
    moment2,
    *,
    lr,
    beta1,
    beta2,
    epsilon,
    coeff,
    beta1_pow,
    beta2_pow,
    shared_second_moment,
):
    """Independent, text-book decoupled-AdamW step in float64.

    Derived from the AdamW definition, not from the production source:
      * decoupled weight decay shrinks the parameter first,
      * the bias-corrected moments are ``m / (1 - beta1_pow)`` and
        ``v / (1 - beta2_pow)`` with ``beta*_pow`` supplied explicitly,
      * ``shared_second_moment`` selects the AdamW-mini scalar variance
        (mean of ``g*g``) versus the element-wise AdamWCustom variance.
    Returns updated (param, moment1, moment2, beta1_pow, beta2_pow).
    """
    param = np.asarray(param, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    moment1 = np.asarray(moment1, dtype=np.float64)
    moment2 = np.asarray(moment2, dtype=np.float64)

    p = param * (1.0 - lr * coeff)
    m_new = beta1 * moment1 + (1.0 - beta1) * grad
    if shared_second_moment:
        g2 = np.mean(grad * grad)
    else:
        g2 = grad * grad
    v_new = beta2 * moment2 + (1.0 - beta2) * g2

    m_hat = m_new / (1.0 - beta1_pow)
    v_hat = v_new / (1.0 - beta2_pow)
    p = p - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return p, m_new, v_new, beta1 * beta1_pow, beta2 * beta2_pow


class TestAdamWMiniAdamwPython(unittest.TestCase):
    """``AdamWMini.adamw_python`` update kernel (real instance method)."""

    def setUp(self):
        self.linear = nn.Linear(4, 2)
        self.opt = AdamWMini(
            parameters=self.linear.parameters(),
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=0.01,
        )

    def test_single_step_matches_independent_reference(self):
        param = paddle.to_tensor([1.0, 2.0, 3.0, 4.0], dtype="float32")
        grad = paddle.to_tensor([0.1, 0.2, 0.3, 0.4], dtype="float32")
        moment1 = paddle.zeros([4], dtype="float32")
        moment2 = paddle.zeros([1], dtype="float32")  # mini: shared scalar
        beta1_pow = paddle.to_tensor([0.9], dtype="float32")
        beta2_pow = paddle.to_tensor([0.999], dtype="float32")

        self.opt.adamw_python(
            param,
            grad,
            0.001,
            moment1,
            moment2,
            beta1_pow,
            beta2_pow,
            None,
            False,
            0.9,
            0.999,
            1e-8,
            1.0,
            0.01,
            True,
            False,
        )

        exp_p, exp_m, exp_v, exp_b1, exp_b2 = _ref_adamw_step(
            [1.0, 2.0, 3.0, 4.0],
            [0.1, 0.2, 0.3, 0.4],
            [0, 0, 0, 0],
            [0.0],
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            coeff=0.01,
            beta1_pow=0.9,
            beta2_pow=0.999,
            shared_second_moment=True,
        )
        np.testing.assert_allclose(param.numpy(), exp_p, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(moment1.numpy(), exp_m, rtol=1e-5, atol=1e-7)
        # AdamW-mini contract: one shared scalar variance, not per-element.
        self.assertEqual(list(moment2.shape), [1])
        np.testing.assert_allclose(moment2.numpy(), exp_v, rtol=1e-5, atol=1e-9)
        self.assertAlmostEqual(float(moment2), 7.5e-5, places=9)
        np.testing.assert_allclose(beta1_pow.numpy(), [exp_b1], rtol=1e-6)
        np.testing.assert_allclose(beta2_pow.numpy(), [exp_b2], rtol=1e-6)

    def test_skip_update_leaves_everything_untouched(self):
        param = paddle.to_tensor([1.0, 2.0, 3.0, 4.0], dtype="float32")
        grad = paddle.to_tensor([0.5, 0.5, 0.5, 0.5], dtype="float32")
        # Non-trivial initial state so a spurious write would be observable.
        moment1 = paddle.to_tensor([0.3, 0.3, 0.3, 0.3], dtype="float32")
        moment2 = paddle.to_tensor([0.7], dtype="float32")
        beta1_pow = paddle.to_tensor([0.81], dtype="float32")
        beta2_pow = paddle.to_tensor([0.998], dtype="float32")
        snap = [
            t.numpy().copy()
            for t in (param, moment1, moment2, beta1_pow, beta2_pow)
        ]

        self.opt.adamw_python(
            param,
            grad,
            0.001,
            moment1,
            moment2,
            beta1_pow,
            beta2_pow,
            None,
            True,
            0.9,
            0.999,
            1e-8,
            1.0,
            0.01,
            True,
            False,
        )
        for t, before in zip(
            (param, moment1, moment2, beta1_pow, beta2_pow), snap
        ):
            np.testing.assert_array_equal(t.numpy(), before)

    def test_weight_decay_flag_selects_coeff(self):
        base = ([1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4])

        def run(with_decay):
            param = paddle.to_tensor(base[0], dtype="float32")
            grad = paddle.to_tensor(base[1], dtype="float32")
            m1 = paddle.zeros([4], dtype="float32")
            m2 = paddle.zeros([1], dtype="float32")
            b1 = paddle.to_tensor([0.9], dtype="float32")
            b2 = paddle.to_tensor([0.999], dtype="float32")
            self.opt.adamw_python(
                param,
                grad,
                0.001,
                m1,
                m2,
                b1,
                b2,
                None,
                False,
                0.9,
                0.999,
                1e-8,
                1.0,
                0.01,
                with_decay,
                False,
            )
            return param.numpy()

        got_decay, got_plain = run(True), run(False)
        exp_decay = _ref_adamw_step(
            base[0],
            base[1],
            [0, 0, 0, 0],
            [0.0],
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            coeff=0.01,
            beta1_pow=0.9,
            beta2_pow=0.999,
            shared_second_moment=True,
        )[0]
        exp_plain = _ref_adamw_step(
            base[0],
            base[1],
            [0, 0, 0, 0],
            [0.0],
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            coeff=0.0,
            beta1_pow=0.9,
            beta2_pow=0.999,
            shared_second_moment=True,
        )[0]
        np.testing.assert_allclose(got_decay, exp_decay, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(got_plain, exp_plain, rtol=1e-5, atol=1e-6)
        # Decoupled decay must actually shrink positive params relative to none.
        self.assertTrue(np.all(got_decay < got_plain))


def _make_custom_optimizer(
    parameters, *, multi_precision=False, weight_decay=0.01, learning_rate=0.001
):
    """Build a real ``AdamWCustom`` (non-HF branch).

    ``quantization_config`` is a plain namespace: for ordinary (non
    ``quantization_linear``) parameters the constructor never consults it, so no
    mock of the code-under-test is involved. ``get_world_size`` is pinned to 1
    to isolate the single-process path (the HCG/Fleet branch needs a real
    process group and is not exercised here).
    """
    with patch("paddle.distributed.get_world_size", return_value=1):
        return AdamWCustom(
            quantization_config=SimpleNamespace(
                weight_quantize_algo="a8w8linear", apply_hadamard=False
            ),
            tensorwise_offload_optimizer=False,
            parameters=parameters,
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=weight_decay,
            multi_precision=multi_precision,
        )


class TestAdamWCustomAdamwCustom(unittest.TestCase):
    """``AdamWCustom.adamw_custom`` update kernel (non-HF, CPU/fp32)."""

    def setUp(self):
        self.linear = nn.Linear(4, 2)
        self.opt = _make_custom_optimizer(
            self.linear.parameters(), multi_precision=True
        )
        self.assertFalse(self.opt.hf_bitexact)  # guard: non-HF numeric branch

    def test_single_step_matches_independent_reference(self):
        param = paddle.to_tensor([1.0, 2.0, 3.0, 4.0], dtype="float32")
        grad = paddle.to_tensor([0.1, 0.2, 0.3, 0.4], dtype="float32")
        moment1 = paddle.zeros([4], dtype="float32")
        moment2 = paddle.zeros([4], dtype="float32")  # element-wise variance
        beta1_pow = paddle.to_tensor([0.9], dtype="float32")
        beta2_pow = paddle.to_tensor([0.999], dtype="float32")

        self.opt.adamw_custom(
            param,
            grad,
            0.001,
            moment1,
            moment2,
            beta1_pow,
            beta2_pow,
            None,
            False,
            0.9,
            0.999,
            1e-8,
            1.0,
            0.01,
            True,
            False,
            False,
        )

        exp_p, exp_m, exp_v, exp_b1, exp_b2 = _ref_adamw_step(
            [1.0, 2.0, 3.0, 4.0],
            [0.1, 0.2, 0.3, 0.4],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            coeff=0.01,
            beta1_pow=0.9,
            beta2_pow=0.999,
            shared_second_moment=False,
        )
        np.testing.assert_allclose(param.numpy(), exp_p, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(moment1.numpy(), exp_m, rtol=1e-5, atol=1e-7)
        # Full Adam contract: per-element second moment, distinct values.
        self.assertEqual(list(moment2.shape), [4])
        np.testing.assert_allclose(moment2.numpy(), exp_v, rtol=1e-5, atol=1e-9)
        self.assertGreater(
            float(moment2[3]), float(moment2[0])
        )  # larger grad -> larger variance, not a shared scalar
        np.testing.assert_allclose(beta1_pow.numpy(), [exp_b1], rtol=1e-6)
        np.testing.assert_allclose(beta2_pow.numpy(), [exp_b2], rtol=1e-6)

    def test_skip_update_leaves_everything_untouched(self):
        param = paddle.to_tensor([1.0, 2.0, 3.0, 4.0], dtype="float32")
        grad = paddle.to_tensor([0.5, 0.6, 0.7, 0.8], dtype="float32")
        moment1 = paddle.to_tensor([0.3, 0.3, 0.3, 0.3], dtype="float32")
        moment2 = paddle.to_tensor([0.7, 0.7, 0.7, 0.7], dtype="float32")
        beta1_pow = paddle.to_tensor([0.81], dtype="float32")
        beta2_pow = paddle.to_tensor([0.998], dtype="float32")
        snap = [
            t.numpy().copy()
            for t in (param, moment1, moment2, beta1_pow, beta2_pow)
        ]

        self.opt.adamw_custom(
            param,
            grad,
            0.001,
            moment1,
            moment2,
            beta1_pow,
            beta2_pow,
            None,
            True,
            0.9,
            0.999,
            1e-8,
            1.0,
            0.01,
            True,
            False,
            False,
        )
        for t, before in zip(
            (param, moment1, moment2, beta1_pow, beta2_pow), snap
        ):
            np.testing.assert_array_equal(t.numpy(), before)

    def test_master_weight_is_update_base_and_skip_update_param(self):
        # master weight starts far from param; the update must operate on the
        # master weight, and skip_update_param must leave ``param`` untouched.
        param = paddle.to_tensor([1.0, 2.0], dtype="float32")
        master = paddle.to_tensor([10.0, 20.0], dtype="float32")
        grad = paddle.to_tensor([0.1, 0.2], dtype="float32")
        moment1 = paddle.zeros([2], dtype="float32")
        moment2 = paddle.zeros([2], dtype="float32")
        beta1_pow = paddle.to_tensor([0.9], dtype="float32")
        beta2_pow = paddle.to_tensor([0.999], dtype="float32")

        self.opt.adamw_custom(
            param,
            grad,
            0.001,
            moment1,
            moment2,
            beta1_pow,
            beta2_pow,
            master,
            False,
            0.9,
            0.999,
            1e-8,
            1.0,
            0.01,
            True,
            True,
            True,
        )

        exp_master = _ref_adamw_step(
            [10.0, 20.0],
            [0.1, 0.2],
            [0, 0],
            [0, 0],
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            coeff=0.01,
            beta1_pow=0.9,
            beta2_pow=0.999,
            shared_second_moment=False,
        )[0]
        np.testing.assert_allclose(
            master.numpy(), exp_master, rtol=1e-5, atol=1e-6
        )
        # skip_update_param=True: param must be byte-for-byte unchanged.
        np.testing.assert_array_equal(param.numpy(), [1.0, 2.0])
        np.testing.assert_allclose(
            moment1.numpy(), [0.01, 0.02], rtol=1e-5, atol=1e-7
        )

    def test_master_weight_written_back_to_param_when_not_skipped(self):
        param = paddle.to_tensor([1.0, 2.0], dtype="float32")
        master = paddle.to_tensor([10.0, 20.0], dtype="float32")
        grad = paddle.to_tensor([0.1, 0.2], dtype="float32")
        moment1 = paddle.zeros([2], dtype="float32")
        moment2 = paddle.zeros([2], dtype="float32")
        beta1_pow = paddle.to_tensor([0.9], dtype="float32")
        beta2_pow = paddle.to_tensor([0.999], dtype="float32")

        self.opt.adamw_custom(
            param,
            grad,
            0.001,
            moment1,
            moment2,
            beta1_pow,
            beta2_pow,
            master,
            False,
            0.9,
            0.999,
            1e-8,
            1.0,
            0.01,
            True,
            True,
            False,
        )
        exp_master = _ref_adamw_step(
            [10.0, 20.0],
            [0.1, 0.2],
            [0, 0],
            [0, 0],
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            coeff=0.01,
            beta1_pow=0.9,
            beta2_pow=0.999,
            shared_second_moment=False,
        )[0]
        # skip_update_param=False: param receives the (master) update, so it
        # jumps to ~10/20, proving the write-back path used the master weight.
        np.testing.assert_allclose(
            param.numpy(), exp_master, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            master.numpy(), exp_master, rtol=1e-5, atol=1e-6
        )

    def test_multi_precision_false_ignores_master_weight(self):
        # ``multi_precision=False`` drops the master weight: the fp32 param is
        # updated in place and the passed master weight stays put.
        param = paddle.to_tensor([1.0, 2.0], dtype="float32")
        master = paddle.to_tensor([10.0, 20.0], dtype="float32")
        grad = paddle.to_tensor([0.1, 0.2], dtype="float32")
        moment1 = paddle.zeros([2], dtype="float32")
        moment2 = paddle.zeros([2], dtype="float32")
        beta1_pow = paddle.to_tensor([0.9], dtype="float32")
        beta2_pow = paddle.to_tensor([0.999], dtype="float32")

        self.opt.adamw_custom(
            param,
            grad,
            0.001,
            moment1,
            moment2,
            beta1_pow,
            beta2_pow,
            master,
            False,
            0.9,
            0.999,
            1e-8,
            1.0,
            0.01,
            True,
            False,
            False,
        )
        exp_param = _ref_adamw_step(
            [1.0, 2.0],
            [0.1, 0.2],
            [0, 0],
            [0, 0],
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            coeff=0.01,
            beta1_pow=0.9,
            beta2_pow=0.999,
            shared_second_moment=False,
        )[0]
        np.testing.assert_allclose(
            param.numpy(), exp_param, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_array_equal(master.numpy(), [10.0, 20.0])


def _clone_state(state):
    """Deep-copy an optimizer ``state_dict`` so later steps cannot mutate it."""
    if isinstance(state, paddle.Tensor):
        return state.detach().clone()
    if isinstance(state, dict):
        return {k: _clone_state(v) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return type(state)(_clone_state(v) for v in state)
    return copy.deepcopy(state)


class TestAdamWCustomStateRestore(unittest.TestCase):
    """End-to-end resume through the real ``optimizer.step`` (CPU/fp32).

    The gradient is held constant and independent of the parameters, so the
    trajectory is driven purely by the optimizer state (moments / beta powers).
    A restored *new* optimizer must reproduce the continuous trajectory, while a
    fresh optimizer that only inherits the weights must diverge -- i.e. checking
    the weights alone would not prove the state was restored.
    """

    def _fixed_grad_step(self, linear, opt):
        loss = (linear.weight * self.cw).sum() + (linear.bias * self.cb).sum()
        loss.backward()
        opt.step()
        opt.clear_grad()

    def setUp(self):
        paddle.seed(20240101)
        self.cw = paddle.to_tensor(
            [[0.1, -0.2], [0.3, 0.05], [-0.15, 0.25]], dtype="float32"
        )
        self.cb = paddle.to_tensor([0.2, -0.1], dtype="float32")

    def test_new_optimizer_resumes_continuous_trajectory(self):
        linear = nn.Linear(3, 2)
        init_w = linear.weight.detach().clone()
        init_b = linear.bias.detach().clone()

        # Reference: three continuous steps with one optimizer instance.
        ref_opt = _make_custom_optimizer(linear.parameters())
        for _ in range(3):
            self._fixed_grad_step(linear, ref_opt)
        ref_w = linear.weight.numpy().copy()
        ref_b = linear.bias.numpy().copy()

        # Interrupted: reset params, run 2 steps, snapshot optimizer state.
        linear.weight.set_value(init_w)
        linear.bias.set_value(init_b)
        opt2 = _make_custom_optimizer(linear.parameters())
        for _ in range(2):
            self._fixed_grad_step(linear, opt2)
        saved_state = _clone_state(opt2.state_dict())
        w_at2 = linear.weight.detach().clone()
        b_at2 = linear.bias.detach().clone()

        # Resume into a brand new optimizer instance, then take the 3rd step.
        opt3 = _make_custom_optimizer(linear.parameters())
        opt3.set_state_dict(saved_state)
        self._fixed_grad_step(linear, opt3)
        np.testing.assert_allclose(
            linear.weight.numpy(), ref_w, rtol=1e-6, atol=1e-7
        )
        np.testing.assert_allclose(
            linear.bias.numpy(), ref_b, rtol=1e-6, atol=1e-7
        )

        # Negative control: same weights, but a fresh (un-restored) optimizer.
        # Its zeroed moments / initial beta powers must give a different update,
        # proving the moment/step state genuinely participated in the resume.
        linear.weight.set_value(w_at2)
        linear.bias.set_value(b_at2)
        fresh_opt = _make_custom_optimizer(linear.parameters())
        self._fixed_grad_step(linear, fresh_opt)
        self.assertFalse(
            np.allclose(linear.weight.numpy(), ref_w, rtol=1e-4, atol=1e-5),
            "a fresh optimizer must not reproduce the resumed trajectory",
        )


class TestAdamWCustomDtypePredicate(unittest.TestCase):
    """``AdamWCustom._is_dtype_fp16_or_bf16`` low-precision classification."""

    def setUp(self):
        self.linear = nn.Linear(4, 2)
        self.opt = _make_custom_optimizer(
            self.linear.parameters(), multi_precision=True
        )

    def test_low_precision_dtypes_are_true(self):
        self.assertTrue(self.opt._is_dtype_fp16_or_bf16(paddle.float16))
        self.assertTrue(self.opt._is_dtype_fp16_or_bf16(paddle.bfloat16))
        # int8 (quantized) is treated as low precision by an early return.
        self.assertTrue(self.opt._is_dtype_fp16_or_bf16(paddle.int8))
        fp8 = getattr(paddle, "float8_e4m3fn", None)
        if fp8 is not None:
            self.assertTrue(self.opt._is_dtype_fp16_or_bf16(fp8))

    def test_fp32_is_false(self):
        self.assertFalse(self.opt._is_dtype_fp16_or_bf16(paddle.float32))

    def test_non_dtype_argument_raises(self):
        with self.assertRaises(AssertionError):
            self.opt._is_dtype_fp16_or_bf16("not-a-dtype")


class TestAdamWCustomLowPrecisionMasterWeight(unittest.TestCase):
    """fp16 master-weight numerics are a GPU concern (device-only branch)."""

    @unittest.skipUnless(
        paddle.is_compiled_with_cuda(),
        "fp16 elementwise math is not reliably runnable on CPU; the "
        "master-weight numeric path is a single-card (GPU) concern.",
    )
    def test_fp16_param_skip_update_param_keeps_param(self):
        paddle.set_device("gpu")
        try:
            opt = _make_custom_optimizer(
                nn.Linear(2, 2).parameters(), multi_precision=True
            )
            param = paddle.to_tensor([1.0, 2.0], dtype="float16")
            master = paddle.to_tensor([1.0, 2.0], dtype="float32")
            grad = paddle.to_tensor([0.5, 0.25], dtype="float16")
            moment1 = paddle.zeros([2], dtype="float32")
            moment2 = paddle.zeros([2], dtype="float32")
            beta1_pow = paddle.to_tensor([0.9], dtype="float32")
            beta2_pow = paddle.to_tensor([0.999], dtype="float32")
            param_before = param.numpy().copy()

            opt.adamw_custom(
                param,
                grad,
                0.001,
                moment1,
                moment2,
                beta1_pow,
                beta2_pow,
                master,
                False,
                0.9,
                0.999,
                1e-8,
                1.0,
                0.01,
                True,
                True,
                True,
            )
            exp_master = _ref_adamw_step(
                [1.0, 2.0],
                [0.5, 0.25],
                [0, 0],
                [0, 0],
                lr=0.001,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
                coeff=0.01,
                beta1_pow=0.9,
                beta2_pow=0.999,
                shared_second_moment=False,
            )[0]
            # param stays put (re-quantized elsewhere); master takes the update.
            np.testing.assert_array_equal(param.numpy(), param_before)
            np.testing.assert_allclose(
                master.numpy(), exp_master, rtol=2e-3, atol=2e-3
            )
        finally:
            paddle.set_device("cpu")


if __name__ == "__main__":
    unittest.main()
