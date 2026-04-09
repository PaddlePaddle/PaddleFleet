# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import paddle
import inspect

if not (hasattr(paddle.library.CustomOpDef, "__call__") and inspect.isfunction(paddle.library.CustomOpDef.__call__)):
    def __call__(self, *args, **kwargs):
        return getattr(getattr(paddle.ops, self._namespace), self._name)(*args, **kwargs)

    paddle.library.CustomOpDef.__call__ = __call__

def torch_compat_empty(*args, **kwargs):
    if "device" in  kwargs and kwargs["device"] == "cuda":
        del kwargs["device"]
    return paddle.empty(*args, **kwargs)

paddle.compat.proxy._extend_torch_proxy_overrides(
    {
        "torch.empty": paddle.compat.proxy.RawOverriddenAttribute(torch_compat_empty),
    }
)

from .count_cumsum import count_cumsum
from .enums import KernelBackendMoE
from .functional import enable_quack_gemm, moe_TC_softmax_topk_layer
from .moe import MoE
