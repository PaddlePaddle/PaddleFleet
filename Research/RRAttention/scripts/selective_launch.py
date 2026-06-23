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
Selective launch script.

Usage: python scripts/selective_launch.py <port>
"""

import os
import subprocess
import sys


def _visible_cuda_device_count():
    """
    Count visible CUDA devices without importing Paddle.
    """
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        visible_devices = visible_devices.strip()
        if not visible_devices or visible_devices in {"-1", "NoDevFiles"}:
            return 0
        return len(
            [device for device in visible_devices.split(",") if device.strip()]
        )

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return 0

    if result.returncode != 0:
        return 0
    return sum(
        1 for line in result.stdout.splitlines() if line.startswith("GPU ")
    )


def main(port):
    """
    main
    """
    nproc_per_node = max(_visible_cuda_device_count(), 1)
    print(
        f"--master_addr 127.0.0.1 --master_port {port} --node_rank 0 --nnodes 1 --nproc_per_node {nproc_per_node}"
    )


if __name__ == "__main__":
    main(int(sys.argv[1]))
