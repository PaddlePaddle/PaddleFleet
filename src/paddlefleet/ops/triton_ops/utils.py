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

"""Utilities for Triton ops: torch compat check and conditional dispatch."""

import paddle


def is_torch_compat_available() -> bool:
    """Return True if paddle provides torch-compat mode."""
    return hasattr(paddle, "enable_compat")


def dispatch_to(dispatch_fn, *, cond=None):
    """Decorator: call dispatch_fn when cond is True, else fall back to fn.

    Args:
        dispatch_fn: high-performance implementation.
        cond: predicate deciding whether to use dispatch_fn.
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
