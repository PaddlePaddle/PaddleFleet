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

"""Backend detection and dispatch."""

import logging

logger = logging.getLogger(__name__)

AVAILABLE_MONITORS = {
    "megatron": [
        "qk_stats",
        "moe_health",
        "ple_health",
        "massive_act",
        "mhc_health",
    ],
    "paddlefleet": [
        "ape_health",
        "qk_stats",
        "moe_health",
        "massive_act",
        "mhc_health",
        "vha_health",
        "attn_update",
        "mlp_update",
        "kda_health",
        "dsa_health",
    ],
}


def detect_backend() -> str:
    try:
        import megatron.core  # noqa: F401

        return "megatron"
    except ImportError:
        pass
    try:
        import paddle  # noqa: F401

        return "paddlefleet"
    except ImportError:
        pass
    raise RuntimeError(
        "No supported backend found. Install megatron-core (torch) or paddlepaddle."
    )


def get_backend_setup_fn(backend: str | None = None):
    """Return the setup_monitors function for the given backend."""
    backend = backend or detect_backend()
    if backend == "megatron":
        from ..backends.megatron import setup_monitors

        return setup_monitors
    elif backend == "paddlefleet":
        from ..backends.paddlefleet import setup_monitors

        return setup_monitors
    else:
        raise ValueError(f"Unknown backend: {backend}")
