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

"""
Distributed sharding optimizer with per-color control over parameter storage
clearing and all_gather skipping.

Usage::

    FP8_COLOR = 99
    for param in model.parameters():
        if <is_fp8_param>(param):
            param.color = FP8_COLOR

    inner_opt = AdamWForFp8(
        learning_rate=1e-3,
        parameters=model.parameters(),
        multi_precision=True,
    )
    sharding_opt = DygraphShardingOptimizerV2ForFp8(
        optimizer=inner_opt,
        hcg=hcg,
        skip_colors={FP8_COLOR},
    )
    # skip_colors 中的 param 会被自动标记 is_fp8=True，
    # AdamWForFp8 据此走 master-weight-as-param 路径。
"""

import logging

from paddle.base import framework
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
    DygraphShardingOptimizerV2,
)

logger = logging.getLogger(__name__)


class DygraphShardingOptimizerV2ForFp8(DygraphShardingOptimizerV2):
    """
    Extends :class:`DygraphShardingOptimizerV2` to support skipping specific
    *colors* during the all_gather phase and proactively clearing parameter
    storage for those colors.

    For the specified ``skip_colors``:

    1. Parameters belonging to those colors are automatically marked with
       ``is_fp8 = True`` (on both the original param **and** the slice_param),
       so that :class:`AdamWForFp8` uses master_weight as the update target.
    2. ``clear_param_storage`` is called for each skip-color after the
       optimizer step, freeing the full-param GPU memory of that color group.
    3. The ``all_gather`` communication is **skipped** for those colors'
       comm-buffers, since the full parameters are not needed.
    """

    def __init__(self, optimizer, hcg, skip_colors=None):
        """
        Args:
            optimizer: The inner (unwrapped) optimizer, e.g. ``AdamWForFp8``.
            hcg: The hybrid communication group.
            skip_colors: An iterable of color values (int) whose parameters
                should have their storage cleared and all_gather skipped.
        """
        super().__init__(optimizer, hcg)
        self._skip_colors = set(skip_colors) if skip_colors else set()

        # ---- Linkage: mark skip-color params with is_fp8=True ----
        # After super().__init__(), _color_to_comm_buffer_list and
        # _slice_params are fully built.  Walk every skip-color's buffers
        # and stamp is_fp8 on both the original param and its slice_param
        # so that AdamWForFp8._append_optimize_op sees the flag.
        for color in self._skip_colors:
            for comm_buffer in self._color_to_comm_buffer_list.get(color, []):
                for param in comm_buffer.params:
                    param.is_fp8 = True
                    if param.name in self._slice_params:
                        self._slice_params[param.name].is_fp8 = True

    # ------------------------------------------------------------------
    # Override: _create_slice_param — propagate is_fp8 to slice_param
    # ------------------------------------------------------------------

    def _create_slice_param(self, param):
        slice_param = super()._create_slice_param(param)
        if getattr(param, "is_fp8", False):
            slice_param.is_fp8 = True
        return slice_param

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_skip_buffer(self, comm_buffer):
        """Return True if *comm_buffer* belongs to a skip-color."""
        for color, buffers in self._color_to_comm_buffer_list.items():
            if comm_buffer in buffers and color in self._skip_colors:
                return True
        return False

    # ------------------------------------------------------------------
    # Override: step
    # ------------------------------------------------------------------

    def step(self):
        """
        Same as ``DygraphShardingOptimizerV2.step()`` except:

        * After the optimizer update, ``clear_param_storage`` is called for
          every skip-color so that the full-param memory is released.
        * The all_gather phase skips comm-buffers that belong to a
          skip-color.
        """

        if self._all_gather_overlap_forward:
            for hook_remove in self._forward_pre_hook_remove_helper:
                hook_remove.remove()
            self._forward_pre_hook_remove_helper = []

        self._collect_comm_buffers()
        self._assign_slice_grad()

        # ---- build params_grads & run inner optimizer ----
        if not isinstance(self._parameter_list[0], dict):
            params_grads = []
            for param in self._parameter_list:
                if (
                    hasattr(param, "regularizer")
                    and param.regularizer is not None
                ):
                    raise ValueError(
                        f"param {param.name} should not has the regularizer attribute"
                    )
                if param.stop_gradient:
                    continue
                assert param.name in self._slice_params
                slice_param = self._slice_params[param.name]
                grad_var = slice_param._grad_ivar()
                if (
                    hasattr(slice_param, "main_grad")
                    and slice_param.main_grad is not None
                ):
                    grad_var = slice_param.main_grad
                if grad_var is not None:
                    params_grads.append((slice_param, grad_var))

            if self._enable_timer:
                self.timers("apply-optimize").start()

            self._apply_optimize(
                loss=None,
                startup_program=None,
                params_grads=params_grads,
            )

            if self._enable_timer:
                self.timers("apply-optimize").stop()

        # ---- clear param storage for skip-colors ----
        for color in self._skip_colors:
            if color in self._color_to_comm_buffer_list:
                self.clear_param_storage(color)

        # ---- sync parameters (all_gather), skipping skip-colors ----
        if not self._all_gather_overlap_forward:
            self._sharding_sync_parameters()
        else:
            from paddle.distributed.fleet.utils.tensor_fusion_helper import (
                FusedCommBuffer,
            )

            for comm_buffer in self._comm_buffer_list:
                comm_buffer.status = FusedCommBuffer.Status.SHARDED
                comm_buffer.sync_param_task = None

            # Mark skip-color buffers as READY directly so they won't be
            # synced by _try_start_bucket_param_sync or forward hooks.
            for comm_buffer in self._comm_buffer_list:
                if self._is_skip_buffer(comm_buffer):
                    comm_buffer.status = FusedCommBuffer.Status.READY

            self._try_start_bucket_param_sync()
            if not self.has_register_forward_hook:
                self._register_pre_forward_hooks()
                self.has_register_forward_hook = True

    # ------------------------------------------------------------------
    # Override: _sharding_sync_parameters  (non-overlap path)
    # ------------------------------------------------------------------

    def _sharding_sync_parameters(self):
        """
        Sync parameters via all_gather, but **skip** comm-buffers whose
        color is in ``self._skip_colors``.
        """
        if self._enable_timer:
            self.timers("sync-parameters").start()

        logger.debug("sharding start sync parameters (fp8-aware)")
        with framework.no_grad():
            if self._all_gather_overlap_forward:
                param2task = {}
                for comm_buffer in self._comm_buffer_list:
                    if self._is_skip_buffer(comm_buffer):
                        continue
                    comm_buffer.sync_params(sync=False, param2task=param2task)

                for layer in self._layers.sublayers():
                    if len(layer.sublayers()) == 0:
                        tasks = []
                        for param in layer.parameters():
                            if param.trainable and param.name in param2task:
                                tasks.append(param2task[param.name])
                        self._forward_pre_hook_remove_helper.append(
                            layer.register_forward_pre_hook(
                                self._forward_pre_hook_function(tasks)
                            )
                        )
            else:
                for comm_buffer in self._comm_buffer_list:
                    if self._is_skip_buffer(comm_buffer):
                        continue
                    comm_buffer.sync_params()

        if self._enable_timer:
            self.timers("sync-parameters").stop()
