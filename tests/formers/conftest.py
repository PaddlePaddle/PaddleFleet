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

import logging
import os


def pytest_configure(config):
    os.environ.setdefault("DOWNLOAD_SOURCE", "aistudio")
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Pin the default Paddle device to GPU when one is available. Under
    # pytest-xdist every worker starts with the default device at
    # Place(cpu); GPU-only code paths (triton RoPE kernels, bf16 AdamW,
    # paddle.device.get_device_capability()) then raise "The device type
    # Place(cpu) is not expected" unless some earlier test on the same
    # worker happened to call paddle.set_device("gpu") first. Pinning it
    # here removes that ordering dependency so scheduling changes cannot
    # flip these tests between pass and fail. CPU-only builds/machines are
    # left untouched.
    try:
        import paddle
    except ImportError:
        paddle = None
    if (
        paddle is not None
        and paddle.is_compiled_with_cuda()
        and paddle.device.cuda.device_count() > 0
    ):
        paddle.set_device("gpu")
