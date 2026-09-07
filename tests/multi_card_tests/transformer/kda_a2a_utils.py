#  Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Shared helpers for the two ``linear_attn_cp_mode="headwise"`` tests.

``test_kda_a2a_core_bitwise.py`` (bitwise, KDA core only) and
``test_kda_head_a2a_layer.py`` (whole layer, rel-L2) need the same two things,
and the first is subtle enough that a second copy of it would eventually drift:

* every loaded fla autotuner pinned to one config.  The autotune keys contain
  the head counts (``ops/kda/chunk_bwd.py``: ``key=['H','HV','K','V',...]``),
  and both tests run the KDA core at *different* head counts on their two arms
  by construction -- that is what head-sharded CP is.  Unpinned, the two arms
  legitimately pick different triton configs, which changes the reduction order,
  which changes the bits.  An unpinned bitwise test measures the autotuner, not
  the all-to-all.
* a relative L2 that does not divide by zero.
"""

from __future__ import annotations

import sys

import paddle

# A neighbour's sequence shard -- or a gradient that was never sharded at all --
# must be at least this far away in relative L2.  Same constant and same purpose
# as ``test_flash_mask_cp_a2a.py``: without a *lower* bound of this kind, an
# all-zero output satisfies every upper-bound assertion in these files.
WRONG_SHARD_MIN = 0.1


def pin_fla_autotune(*prefixes: str) -> list[str]:
    """Pin every already-loaded fla autotuner to its first config.

    Call this **before the first kernel launch** -- an autotuner that has
    already run has cached its choice under a key that includes the head count.

    Walks ``sys.modules`` instead of importing anything: the ``@autotune`` /
    ``@heuristics`` decorators run at import time, so which kernels exist
    depends on what the caller imported.  Both wrappers keep the wrapped object
    in ``.fn``, hence the short peel loop looking for whoever owns ``configs``.

    Returns the names it pinned, so a caller can assert it was not a no-op.
    """
    prefixes = prefixes or ("paddlefleet_ops.fla", "fla")
    pinned = []
    for modname, mod in list(sys.modules.items()):
        if not any(
            modname == prefix or modname.startswith(prefix + ".")
            for prefix in prefixes
        ):
            continue
        for attr in dir(mod):
            try:
                node = getattr(mod, attr)
            except Exception:
                # Module attributes can be properties that raise on an
                # unsupported build; none of them is worth failing a test over.
                continue
            for _ in range(6):
                if node is None:
                    break
                configs = getattr(node, "configs", None)
                if isinstance(configs, list):
                    if len(configs) > 1:
                        node.configs = configs[:1]
                        # A choice already cached under a previous key would
                        # otherwise survive the pinning.
                        if hasattr(node, "cache"):
                            node.cache = {}
                        pinned.append(f"{modname}.{attr}")
                    break
                node = getattr(node, "fn", None)
    return pinned


def rel_l2(got, ref) -> float:
    """``||got - ref|| / ||ref||``, computed in fp32.

    A zero reference falls back to the absolute norm rather than dividing by
    zero: the callers use this on gradients that are legitimately zero outside
    one rank's head block.
    """
    got, ref = got.astype("float32"), ref.astype("float32")
    denom = float(paddle.linalg.norm(ref))
    return float(paddle.linalg.norm(got - ref)) / (denom if denom else 1.0)
