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

"""Thin facade that centralises all ``flash_mask.cute.interface`` imports.

Import sites across the codebase should use this module instead of
reaching into ``flash_mask.cute.interface`` or
``paddlefleet.ops.flash_mask.cute.interface`` directly.  The facade
resolves the correct backend once at import time:

* **Blackwell (capability == 10)** – uses ``paddlefleet.ops.flash_mask``
  (the cute/CUTLASS implementation).
* **Other GPUs** – falls back to ``paddle.nn.functional.flash_attention``
  for ``flashmask_attention`` / ``flash_attention``.  Symbols that only
  exist in the cute implementation (``_flash_attn_fwd``,
  ``_flash_attn_bwd``, ``FlashMaskInfoPaddle``) are set to ``None``.
"""

from paddlefleet.ops import is_flash_mask_available

# -- Always-available paddle built-ins ------------------------------------
from paddle.nn.functional.flash_attention import (
    flash_attention as paddle_flash_attention,
    flashmask_attention as paddle_flashmask_attention,
)

# -- Resolve Blackwell-only symbols --------------------------------------
if is_flash_mask_available():
    from paddlefleet.ops.flash_mask.cute.interface import (
        _flash_attn_bwd,
        _flash_attn_fwd,
        flash_attention,
        flashmask_attention,
    )
    from paddlefleet.ops.flash_mask.cute.flashmask_utils import (
        FlashMaskInfoPaddle,
    )
else:
    _flash_attn_fwd = None
    _flash_attn_bwd = None
    FlashMaskInfoPaddle = None
    # Fall back to paddle built-ins for the attention callables.
    flashmask_attention = paddle_flashmask_attention
    flash_attention = paddle_flash_attention

flashmask_attention_fwd = flashmask_attention

def get_fa_version(
    query,
    key,
    value,
    startend_row_indices,
    causal,
    config,
):
    device = paddle.get_device()

    if "xpu" in device:
        return 2

    if "iluvatar_gpu" in device:
        return 2

    if "gpu" in device:

def flashmask_attention_bwd(
    query,
    key,
    value,
    startend_row_indices,
    causal,
    output,
    log_sum_exp,
    output_grad,
):
    fa_version = config.fa_version
    if "block_mask" in inspect.signature(flashmask_attention).parameters:
        if config.deterministic_mode and query.shape[-1] > 128:
            fa_version = 2
    elif config.deterministic_mode:
        fa_version = 2
    if fa_version == 2:
        # Create seed offset tensor (required for gradient computation)
        seed_offset = paddle.zeros(
            shape=[query.shape[1], query.shape[2]], dtype=paddle.int64
        )

        # Compute gradients using flashmask attention backward pass
        query_grad, key_grad_gathered, value_grad_gathered = (
            paddle._C_ops.flashmask_attention_grad(
                query,
                key_gathered,
                value_gathered,
                startend_row_indices,
                output,
                log_sum_exp,
                seed_offset,
                output_grad,
                0.0,  # dropout probability
                causal,
            )
        )
    elif fa_version == 3:
        sig_params = inspect.signature(flashmask_attention).parameters
        if "group" in sig_params:
            query_grad, key_grad_gathered, value_grad_gathered = (
                paddle._C_ops.flashmask_attention_v2_grad(
                    query,
                    key_gathered,
                    value_gathered,
                    output,
                    log_sum_exp,
                    startend_row_indices,
                    None,  # block_mask
                    output_grad,
                    query.shape[-1] ** (-0.5),
                    False,
                    0,  # rank
                    1,  # nranks
                )
            )
        elif "block_mask" in sig_params:
            query_grad, key_grad_gathered, value_grad_gathered = (
                paddle._C_ops.flashmask_attention_v2_grad(
                    query,
                    key_gathered,
                    value_gathered,
                    output,
                    log_sum_exp,
                    startend_row_indices,
                    None,  # block_mask
                    output_grad,
                    query.shape[-1] ** (-0.5),
                    False,
                )
            )
        else:
            query_grad, key_grad_gathered, value_grad_gathered = (
                paddle._C_ops.flashmask_attention_v2_grad(
                    query,
                    key_gathered,
                    value_gathered,
                    output,
                    log_sum_exp,
                    startend_row_indices,
                    output_grad,
                    query.shape[-1] ** (-0.5),
                    False,
                )
            )
    elif fa_version == 4:
        if startend_row_indices is not None:
            flashmask_info = FlashMaskInfoPaddle(
                startend_row_indices=startend_row_indices,
                is_causal=causal,
            )
        else:
            flashmask_info = None
        query_grad, key_grad_gathered, value_grad_gathered = _flash_attn_bwd(
            query,
            key_gathered,
            value_gathered,
            output,
            output_grad,
            log_sum_exp,
            flashmask_info,
            causal=causal,
            deterministic=paddle.get_flags(["FLAGS_cudnn_deterministic"])[
                "FLAGS_cudnn_deterministic"
            ],
        )
    else:
        raise ValueError(
            f"FlashAttention version {fa_version} is not supported."
        )

__all__ = [
    "flashmask_attention",
    "flash_attention",
    "flashmask_attention_fwd",
    "flashmask_attention_bwd",
    "flash_attention_fwd",
    "flash_attention_bwd",
]
