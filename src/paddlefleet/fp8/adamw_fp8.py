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
from paddle import _C_ops
from paddle.base import framework
from paddle.base.framework import Variable, in_dynamic_or_pir_mode, in_pir_mode
from paddle.optimizer.adamw import AdamW


class AdamWForFp8(AdamW):
    """
    AdamW optimizer with FP8 parameter support.

    For parameters with ``is_fp8=True``, the optimizer performs the AdamW
    update directly on the fp32 master weight (treating it as the ``param``
    argument) and does **not** pass ``master_weight`` to the underlying
    ``adamw_`` operator.  After the update the original low-precision
    parameter is refreshed from the updated master weight.

    For all other parameters the behaviour is identical to
    :class:`paddle.optimizer.AdamW`.
    """

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, (framework.Block,))
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)
        param, grad = param_and_grad

        # Check if this parameter is an FP8 parameter.
        is_fp8 = getattr(param, "is_fp8", False)

        if not is_fp8:
            # Non-FP8 path: delegate to the standard AdamW implementation.
            return super()._append_optimize_op(block, param_and_grad)

        # ---- FP8 path ----
        # For FP8 params we require multi_precision to be enabled so that a
        # fp32 master weight exists.  We will hand the master weight to the
        # adamw_ kernel as ``param`` and set ``master_weight=None`` /
        # ``find_master=False`` so the kernel updates the master weight
        # in-place as if it were the primary parameter.

        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(
            param.dtype
        )
        assert find_master, (
            f"AdamWForFp8 requires multi_precision=True and the FP8 param "
            f"'{param.name}' to be fp16/bf16, but got dtype={param.dtype}"
        )

        master_weight = self._master_weights[param.name]

        # Whether we should do weight decay for the parameter.
        with_decay = True
        if (
            self._apply_decay_param_fun is not None
            and not self._apply_decay_param_fun(param.name)
        ):
            with_decay = False

        # Fetch accumulators (keyed by master weight name).
        moment1 = self._get_accumulator_master(
            self._moment1_acc_str, param
        )
        moment2 = self._get_accumulator_master(
            self._moment2_acc_str, param
        )
        moment2_max = (
            self._get_accumulator_master(
                self._moment2_acc_max_str, param
            )
            if self._amsgrad
            else None
        )
        beta1_pow_acc = self._get_accumulator_master(
            self._beta1_pow_acc_str, param
        )
        beta2_pow_acc = self._get_accumulator_master(
            self._beta2_pow_acc_str, param
        )

        lr = self._create_param_lr(param_and_grad)

        if in_dynamic_or_pir_mode():
            lr_ratio_ = (
                1.0
                if self._lr_ratio is None
                else self._lr_ratio(param)
            )

            _beta1 = (
                self._beta1
                if not isinstance(self._beta1, Variable)
                else self._beta1.item(0)
            )
            _beta2 = (
                self._beta2
                if not isinstance(self._beta2, Variable)
                else self._beta2.item(0)
            )

            found_inf = (
                self._get_auxiliary_var('found_inf') if in_pir_mode() else None
            )

            # Call adamw_ with master_weight as the param, and no
            # master_weight argument (find_master=False).
            _, _, _, _, _, _, _ = _C_ops.adamw_(
                master_weight,          # param  <-- use master weight here
                grad,                   # grad
                lr,                     # learning_rate
                moment1,                # moment1
                moment2,                # moment2
                moment2_max,            # moment2_max
                beta1_pow_acc,          # beta1_pow
                beta2_pow_acc,          # beta2_pow
                None,                   # master_param  <-- no master weight
                found_inf,              # skip_update
                _beta1,                 # beta1
                _beta2,                 # beta2
                self._epsilon,          # epsilon
                lr_ratio_,              # lr_ratio
                self._weight_decay,     # coeff
                with_decay,             # with_decay
                self._lazy_mode,        # lazy_mode
                1000,                   # min_row_size_to_use_multithread
                False,                  # multi_precision  <-- False
                False,                  # use_global_beta_pow
                self._amsgrad,          # amsgrad
            )

            # NOTE: We do NOT copy master_weight back to param here.
            # In the distributed (sharding) scenario the full param storage
            # for FP8 colors is cleared by
            # DygraphShardingOptimizerV2ForFp8, so there is no buffer to
            # write to.  The master_weight is the single source of truth
            # and will be used directly at the next forward.

            return None
        else:
            raise NotImplementedError(
                "AdamWForFp8 only supports dynamic graph mode."
            )
