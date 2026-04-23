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

"""
Triton 操作工具函数模块。

该模块提供了 Triton kernel 的辅助工具函数，用于支持 PaddlePaddle 与 Triton 的集成。

主要功能：
- Torch 兼容性检查：检测 PaddlePaddle 是否支持 torch 兼容模式
- 条件分发装饰器：根据条件选择高性能 Triton 实现或后备实现

"""

import paddle


def is_torch_compat_available() -> bool:
    """
    判断是否支持 Torch 兼容性
    Returns:
        bool: 如果存在 enable_compat 方法，则返回 True；否则返回 False。
    """
    return hasattr(paddle, "enable_compat")


def dispatch_to(dispatch_fn, *, cond=None):
    """
    创建条件分发装饰器，根据 cond 条件选择高性能内核或后备实现

    Args:
        dispatch_fn: 高性能实现函数，当启用高性能内核时调用
        cond: 条件函数，返回布尔值，决定是否使用高性能实现

    Returns:
        decorator: 装饰器函数，用于包装目标函数
    """
    if cond is None:
        cond = lambda self, *args, **kwargs: True

    def decorator(fn):
        def wrapper(*args, **kwargs):
            if cond(*args, **kwargs) and is_torch_compat_available():
                return dispatch_fn(*args, **kwargs)
            return fn(*args, **kwargs)

        wrapper.__original_fn__ = fn
        return wrapper

    return decorator
