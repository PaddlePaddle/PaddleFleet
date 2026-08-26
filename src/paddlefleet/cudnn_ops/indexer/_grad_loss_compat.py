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

"""Sync-free ``grad_loss`` validation for the cuDNN-frontend indexer backward.

The frontend guards its ``grad_loss`` argument with

    if grad_loss.numel() != 1 or grad_loss.dtype != torch.float32 or ...

(``indexer_backward/api.py:38``), which is free under real torch because
``Tensor.numel()`` returns a Python ``int``. ``paddlefleet_ops`` enables
``paddle.enable_compat(scope={"cudnn"})`` so the frontend's ``torch.*`` calls
resolve to Paddle, and Paddle's ``numel()`` is an *op* returning a 0-D Tensor.
``!= 1`` therefore builds a device bool and the Python ``or`` forces
``Tensor.__bool__``, which lowers to a blocking ``GpuMemcpySync`` on the CUDA
*legacy default* stream.

Paddle creates its compute and communication streams as blocking streams, so a
legacy-default-stream copy is a bidirectional full-device barrier: it cannot
retire until everything already in flight drains. Under
``dsa_indexer_loss_bwd_p2p_overlap`` the launching thread hits this barrier
between the deferred KL forward and ``compute_csa_indexer_grads``, and cannot
resume until the pipeline's ``ncclDevKernel_SendRecv`` finishes -- so the whole
indexer backward is enqueued only *after* the p2p window it was supposed to hide
behind has closed. Measured on a 64k pp4/vpp3 profile: a 256-byte device-to-host
copy stalling the launching thread for 35 ms and leaving the compute stream idle
for 24.7 ms of a 28 ms send/recv.

Same failure mode, and the same reason, as the ``d_index_k`` note in
``csa_indexer_bwd``.

The replacement below keeps the validation semantics byte-for-byte and only
changes *how* the element count is obtained: ``shape`` is host metadata under
both torch and Paddle, and the ``dtype`` / ``device`` comparisons already return
Python bools. This shim can be dropped once the wheel picks up the equivalent
fix in ``PFCCLab/cudnn-frontend``.
"""

from __future__ import annotations

import sys

import paddle

_API = "paddlefleet_ops.cudnn.deepseek_sparse_attention.indexer_backward.api"

_PATCHED = False


def _validate_grad_loss_tensor(grad_loss, device):
    """Drop-in replacement for the frontend validator, without the host sync."""
    if not paddle.is_tensor(grad_loss):
        raise TypeError("grad_loss must be a torch.Tensor")
    numel = 1
    for dim in grad_loss.shape:
        numel *= int(dim)
    if (
        numel != 1
        or grad_loss.dtype != paddle.float32
        or grad_loss.device != device
    ):
        raise ValueError(
            f"grad_loss must be a single-element float32 tensor on {device}"
        )
    return grad_loss.detach().reshape([1])


def patch_indexer_backward_api() -> None:
    """Install the sync-free validator on the frontend module, once.

    Call this right after importing from the frontend api module, so the module
    is guaranteed to be in ``sys.modules``. The wrappers resolve
    ``_validate_grad_loss_tensor`` as a module global at call time, so replacing
    the attribute after the import is enough.

    A no-op on wheels that predate the guard -- they expose
    ``_as_grad_loss_tensor``, which never reads a scalar back to the host -- and
    on a mocked api module, so this is safe across ``paddlefleet_ops`` versions.
    """
    global _PATCHED
    if _PATCHED:
        return
    api_module = sys.modules.get(_API)
    if api_module is not None and hasattr(
        api_module, "_validate_grad_loss_tensor"
    ):
        api_module._validate_grad_loss_tensor = _validate_grad_loss_tensor
        _PATCHED = True
