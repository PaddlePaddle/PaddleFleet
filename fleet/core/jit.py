# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import paddle

jit_fuser = paddle.jit.to_static(backend="CINN")
# try:
#     if is_torch_min_version("2.2.0a0"):
#         jit_fuser = paddle.compile
# except ImportError:

#     def noop_decorator(func):
#         return func

#     jit_fuser = noop_decorator
