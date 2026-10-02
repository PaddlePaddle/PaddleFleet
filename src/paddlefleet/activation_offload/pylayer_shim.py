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

"""Flatten container arguments passed to ``PyLayerContext.save_for_backward``.

Framework workaround. While ``saved_tensors_hooks`` is installed, saving a list
or tuple through ``save_for_backward`` faults during the backward unpack, and a
container holding non-Tensor elements can raise on save. Only the top-level
varargs form ``save_for_backward(t1, t2, None)`` behaves correctly.

Layers that save lists -- fused MoE layers save one that mixes tensors with
``None`` and ints -- therefore need this shim installed before activation
offloading is enabled. On save it flattens containers into top-level elements
and keeps non-Tensor elements aside on the context; ``saved_tensor`` rebuilds
the original structure from the recorded layout. PyLayers that do not use hooks
are unaffected, since top-level varargs keep their native semantics.

Remove this module once the framework handles containers directly; callers need
no change.
"""

from __future__ import annotations

import paddle
from paddle.autograd.py_layer import PyLayerContext

_installed = False


def install():
    """Monkey-patch ``PyLayerContext`` once per process (idempotent)."""
    global _installed
    if _installed:
        return
    _installed = True

    orig_save = PyLayerContext.save_for_backward
    orig_saved = PyLayerContext.saved_tensor

    def save_for_backward(self, *args):
        # Layout entry: None for an argument that was already a top-level
        # element, (is_list, n) for one that was a container.
        layout = []
        flat = []
        for a in args:
            if type(a) in (list, tuple):
                layout.append((type(a) is list, len(a)))
                flat.extend(a)
            else:
                layout.append(None)
                flat.append(a)
        # Non-Tensor elements are kept aside: with hooks installed the container
        # setter rejects them.
        side = {}
        for i, x in enumerate(flat):
            if x is not None and not isinstance(x, paddle.Tensor):
                side[i] = x
                flat[i] = None
        self._shim_layout = (layout, side)
        orig_save(self, *flat)

    def saved_tensor(self):
        got = orig_saved(self)
        shim = getattr(self, "_shim_layout", None)
        if shim is None:
            return got
        layout, side = shim
        flat = list(got)
        for i, x in side.items():
            flat[i] = x
        out, idx = [], 0
        for ent in layout:
            if ent is None:
                out.append(flat[idx])
                idx += 1
            else:
                is_list, n = ent
                seg = flat[idx : idx + n]
                idx += n
                out.append(seg if is_list else tuple(seg))
        return tuple(out)

    PyLayerContext.save_for_backward = save_for_backward
    PyLayerContext.saved_tensor = saved_tensor
