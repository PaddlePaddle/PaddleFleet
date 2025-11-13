# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import paddle

jit_fuser = paddle.jit.to_static(backend="CINN")