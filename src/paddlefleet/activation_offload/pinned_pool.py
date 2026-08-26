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

"""Pinned host memory pool, bucketed by (power-of-two byte size, dtype).

Why a pool is needed rather than relying on the framework's pinned allocator:
with the default place set to GPU, ``paddle.empty(shape, dtype)`` builds a
*device* tensor, so ``.pin_memory()`` always pays a full-size D2H copy on top of
the allocation. Caching inside the allocator cannot remove that copy -- only
reusing an already existing pinned buffer can.

Why buckets are keyed by rounded-up byte size instead of ``(shape, dtype)``:
activation row counts vary from step to step (MoE routing being the usual
cause), so an exact-shape key misses on nearly every allocation. A miss is
expensive twice over -- it costs an extra full-size copy, and since ``free``
returns buffers to a bucket rather than releasing them, every new key adds
permanently to the pool's footprint. Rounding to a power of two bounds internal
fragmentation at 2x and bounds the number of distinct buffers to roughly
log2(largest tensor) per dtype.

Getting an exactly-shaped pinned tensor out of a rounded-up buffer needs care:
any ordinary view operation on a pinned tensor (``reshape``, slicing,
``paddle.view``, ``_slice``) moves it to the GPU. The only way to keep the
pinned place is to share the underlying holder and then rewrite the dims::

    v = paddle.Tensor()
    base._share_buffer_to(v)                  # place stays gpu_pinned
    v.value().get_tensor()._set_dims(shape)   # metadata only, no copy

Capacity and degradation:

- ``capacity_bytes=None`` places no limit on the pool.
- ``capacity_bytes=N`` limits the total bytes of buffers the pool holds,
  counting idle buckets. ``alloc()`` returns ``None`` past that limit and the
  caller degrades (leaves the activation on the device), which is a controlled
  fallback rather than an out-of-memory failure on the host.
- Counters for diagnosis: ``n_alloc`` (real pinned allocations), ``n_reuse``,
  ``n_oob``, ``total_bytes`` (pinned memory held), ``in_use_bytes`` (handed out
  and not yet returned) and ``waste_bytes`` (accumulated internal
  fragmentation).
"""

from __future__ import annotations

import logging

import paddle

logger = logging.getLogger(__name__)

# Lower bound on bucket size: power-of-two bucketing is pointless for tensors of
# a few KB and would only create a crowd of tiny buckets. In production the
# min_offloaded_tensor_bytes threshold keeps allocations well above this.
_BUCKET_FLOOR_BYTES = 1 << 12  # 4KB

_ITEMSIZE_CACHE: dict = {}


def _norm_dtype(dtype):
    """Normalize a string dtype such as ``'float32'`` to a paddle dtype object.

    The bucket key uses ``str(dtype)``, so ``alloc('float32')`` and the matching
    ``free(tensor)`` -- which sees a dtype object -- would compute two different
    keys and the pool would silently never hit.
    """
    if isinstance(dtype, str):
        d = getattr(paddle, dtype, None)
        if isinstance(d, paddle.dtype):
            return d
        try:
            return paddle.empty([0], dtype=dtype).dtype
        except Exception:
            return dtype
    return dtype


def _itemsize(dtype) -> int:
    """Bytes per element, cached."""
    hit = _ITEMSIZE_CACHE.get(dtype)
    if hit is not None:
        return hit
    size = getattr(dtype, "itemsize", None)
    if not isinstance(size, int) or size <= 0:
        try:
            size = paddle.empty([1], dtype=dtype).itemsize
        except Exception:  # very old runtimes: assume 2 bytes
            size = 2
    _ITEMSIZE_CACHE[dtype] = size
    return size


def _nbytes(shape, dtype) -> int:
    """Byte size of a (shape, dtype) pair."""
    n = 1
    for s in shape:
        n *= int(s)
    return n * _itemsize(_norm_dtype(dtype))


def _round_pow2(n: int) -> int:
    """Round up to a power of two."""
    if n <= 1:
        return 1
    return 1 << (int(n) - 1).bit_length()


class PinnedPool:
    """Bucketed pool of pinned host buffers.

    Args:
        capacity_bytes: upper bound on the pinned bytes the pool holds, idle
            buckets included. ``None`` means no limit.
        bucket_floor_bytes: smallest bucket size.
    """

    def __init__(
        self,
        capacity_bytes: int | None = None,
        bucket_floor_bytes: int = _BUCKET_FLOOR_BYTES,
    ):
        # (bucket_bytes, str(dtype)) -> [(base_tensor, reload_event)]
        self._free: dict = {}
        # data_ptr -> (base_tensor, key, bucket_bytes, handle); never pruned,
        # the pool keeps owning its buffers
        self._bases: dict = {}
        # data_ptrs currently sitting in an idle bucket, used to reject a double
        # free -- handing the same buffer to two consumers is a silent data error
        self._idle: set = set()
        self.capacity_bytes = capacity_bytes
        self.bucket_floor_bytes = int(bucket_floor_bytes)
        self.n_alloc = 0  # real pinned allocations
        self.n_reuse = 0
        self.n_oob = 0  # requests refused for exceeding capacity
        self.n_free_unknown = 0  # freed a tensor this pool never handed out
        self.n_free_dup = 0  # same buffer freed twice (ignored)
        self.total_bytes = 0  # pinned memory held, counted per bucket size
        self.in_use_bytes = 0  # handed out and not yet returned
        self.waste_bytes = 0  # accumulated internal fragmentation

    # ---------------- keys and views ----------------

    def _bucket_bytes(self, nbytes: int, itemsize: int) -> int:
        return max(_round_pow2(nbytes), self.bucket_floor_bytes, itemsize)

    @staticmethod
    def _key(bucket_bytes: int, dtype):
        return (int(bucket_bytes), str(_norm_dtype(dtype)))

    @staticmethod
    def _dense(t: paddle.Tensor):
        """Reach the DenseTensor under an eager Tensor, across versions."""
        if hasattr(t, "value"):
            return t.value().get_tensor()
        return t.get_tensor()

    def _view(self, handle, shape, nbytes: int, cap: int):
        """Retarget a buffer's resident handle to ``shape`` and return it.

        Reshaping or slicing is not an option: those move a pinned tensor to the
        GPU (see the module docstring). ``_set_dims`` writes metadata only and
        does not bounds-check -- dims larger than the holder are accepted and
        turn into silent out-of-bounds writes into host memory -- so the size is
        asserted here.

        The handle object is reused across allocations rather than rebuilt.
        Creating a ``paddle.Tensor`` and calling ``_share_buffer_to`` on every
        allocation dominates the hit path; reusing the handle leaves only
        ``_set_dims``. The identity semantics are unchanged: a buffer is handed
        to one consumer at a time, so a record table keyed by ``id()`` of the
        returned tensor cannot collide.
        """
        assert nbytes <= cap, (
            f"pinned view of {nbytes}B exceeds its {cap}B bucket; _set_dims "
            "does not bounds-check, so this would corrupt host memory"
        )
        self._dense(handle)._set_dims([int(s) for s in shape])
        return handle

    # ---------------- allocate / release ----------------

    def alloc(self, shape, dtype) -> paddle.Tensor | None:
        """Take a pinned tensor of exactly ``shape``; None if over capacity."""
        dtype = _norm_dtype(dtype)
        itemsize = _itemsize(dtype)
        nbytes = _nbytes(shape, dtype)
        cap = self._bucket_bytes(nbytes, itemsize)
        key = self._key(cap, dtype)

        bucket = self._free.get(key)
        if bucket:
            for i, (base, ev) in enumerate(bucket):
                # A returned buffer may still be read by the H2D stream if its
                # reload has not finished; it can only be reused once that event
                # is complete. query() does not block.
                if ev is None or ev.query():
                    bucket.pop(i)
                    ptr = base.data_ptr()
                    self._idle.discard(ptr)
                    self.n_reuse += 1
                    self.in_use_bytes += cap
                    self.waste_bytes += cap - nbytes
                    handle = self._bases[ptr][3]
                    return self._view(handle, shape, nbytes, cap)

        if (
            self.capacity_bytes is not None
            and self.total_bytes + cap > self.capacity_bytes
        ):
            self.n_oob += 1
            return None

        base = paddle.empty([cap // itemsize], dtype=dtype).pin_memory()
        # Resident handle sharing the holder; later allocations only rewrite dims
        handle = paddle.Tensor()
        base._share_buffer_to(handle)
        self.n_alloc += 1
        self.total_bytes += cap
        self.in_use_bytes += cap
        self.waste_bytes += cap - nbytes
        self._bases[base.data_ptr()] = (base, key, cap, handle)
        return self._view(handle, shape, nbytes, cap)

    def free(self, t: paddle.Tensor, reload_event=None):
        """Return a pinned buffer, optionally with the event of its last read.

        The buffer is found by ``data_ptr()``: what comes back is a view built by
        ``_view()``, whose shape says nothing about the bucket it came from.
        """
        ptr = t.data_ptr()
        ent = self._bases.get(ptr)
        if ent is None:
            # Not ours (a separate large-tensor path, or another pool). Ignore it
            # rather than guessing a bucket.
            self.n_free_unknown += 1
            return
        if ptr in self._idle:  # a double free would hand it out twice
            self.n_free_dup += 1
            return
        base, key, cap, _handle = ent
        self._free.setdefault(key, []).append((base, reload_event))
        self._idle.add(ptr)
        self.in_use_bytes -= cap

    def prewarm(self, specs) -> int:
        """Preallocate at startup so training does not pay for allocations.

        Every buffer must be taken first and returned afterwards. Interleaving
        them as alloc/free/alloc would hit the buffer just returned, so ``count``
        would only ever allocate one.

        Args:
            specs: iterable of ``(shape, dtype, count)``.

        Returns:
            Number of buffers actually preallocated, which may be fewer than
            requested if ``capacity_bytes`` is reached.
        """
        made = 0
        for shape, dtype, count in specs:
            held = []
            for _ in range(int(count)):
                t = self.alloc(shape, dtype)
                if t is None:  # hit the capacity limit; stop this spec
                    break
                held.append(t)
            for t in held:
                self.free(t)
            made += len(held)
        return made

    # ---------------- diagnostics / teardown ----------------

    def stats_line(self) -> str:
        mb = 1048576.0
        hit = self.n_reuse / max(1, self.n_reuse + self.n_alloc)
        return (
            f"pinned={self.total_bytes / mb:.0f}MB "
            f"in_use={self.in_use_bytes / mb:.0f}MB "
            f"buckets={len(self._free)} hit={hit:.3f} "
            f"alloc={self.n_alloc} reuse={self.n_reuse} oob={self.n_oob} "
            f"waste={self.waste_bytes / mb:.0f}MB"
        )

    def release_all(self):
        """Drop every idle bucket and give the pinned memory back to the host.

        Only idle buffers are dropped. Those still handed out stay alive through
        their holder, but the pool forgets them, so a later ``free()`` of one
        counts as ``n_free_unknown``. Call this at an iteration boundary or at
        teardown.
        """
        for key, bucket in self._free.items():
            for base, _ev in bucket:
                self._bases.pop(base.data_ptr(), None)
        self._free.clear()
        self._idle.clear()
        self.total_bytes = self.in_use_bytes

    def clear(self):
        self._free.clear()
        self._idle.clear()
        self._bases.clear()
        self.total_bytes = 0
        self.in_use_bytes = 0
