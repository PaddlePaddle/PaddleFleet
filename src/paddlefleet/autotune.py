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

"""Launch-parameter autotuning with a bit-exactness gate.

Why this exists instead of tuned constants
------------------------------------------
A launch configuration swept on one machine at one shape is not portable. The
same six hardcoded configurations that made a kernel family faster on the
machine they were swept on were "effective but harmful" on another generation
(the 256x256 tile spilled 3940 B where 128x128 spilled 312 B, and resident CTAs
per SM went 2 -> 1), and had to be reverted wholesale. So no launch parameter
below is written down: candidate *bounds* come from
``cuDeviceGetAttribute``, and which candidate wins is measured here.

Two rules that the order of operations encodes
----------------------------------------------
1. **The bit-exactness gate runs before timing, never after.** A candidate is
   compared byte-for-byte (not with a tolerance) against the factory
   configuration's output, and a single differing byte drops it however fast it
   is. This is what makes "autotuned" compatible with "numerically lossless":
   correctness is not something a human re-verifies per machine.

   It also catches cases that cannot be reasoned about. Reduction order is
   usually predictable from the source -- a loop-carried accumulator survives
   any tiling, a lane-shuffle tree (``ct.sum``) does not -- but library calls
   are not: a cuBLASLt algorithm choice is not a pure function of its operands
   (merely having another GEMM live in the process changes it), so "this
   rewrite cannot reorder the reduction" has been measured false. Only a
   measurement can decide those.

2. **Selection is measured, never proxied by a static metric.** Static
   occupancy and register/spill counts have both pointed at the wrong winner:
   a configuration with 100% static occupancy and zero spill ran 3.7x slower
   than the winner (which sat at 25%), and a cleaner one (37 regs, 0 spill)
   lost by 4% to a dirtier one. Issue rate can even be inverted -- the faster
   variant issued *more*. Static numbers are used here only to bound the
   candidate set -- an infeasible candidate is not offered at all, rather than
   offered and then detected after the fact.

Three entry points
------------------
==============================  =========================================
``<PREFIX>_AUTOTUNE=0``         always the factory config; no cache, no scan
``<PREFIX>_PIN[_<NAME>]=k=v``   force one config; no cache, no scan
default                         cache lookup; on a miss, scan
==============================  =========================================

``_PIN`` is not a debugging nicety. Autotuning and "the A/B differs in exactly
one variable" are in direct conflict: if a configuration is allowed to drift
per machine and per shape, an end-to-end comparison silently contains more
than the change under test. Pinning both sides is the only way to attribute a
difference. (A tile change that landed *between* the two sides of one A/B once
produced a published-then-retracted conclusion.)

Who scans
---------
Whoever takes the lock. There is no leader and no rank arithmetic.

Processes that share a cache directory serialise on one lock per key: the first
one scans, the others block, and when they wake the entry is already there, so
they read it and return. Wall time is one scan either way, but only one process
per directory does the compiling and the timing -- which matters, because it
keeps a sibling's compilation off the CPU while a measurement is in flight.
Processes that do not share a directory never meet and do not need to: they own
different GPUs, the timing runs on CUDA events, so there is no shared resource
to contend for. If the lock cannot be taken, the process scans anyway on its
own device; that costs duplicated work, not a corrupt reading.

The cache is therefore an optimisation, never a rendezvous. Nothing has to be
visible to anyone else for the tuning to be correct, which is what lets the
directory be node local -- see :func:`cache_dir`.

Each entry is one JSON file, written to a temporary name and ``os.replace``d
into place, so a reader can never catch a half-written file.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import warnings

CACHE_LAYOUT = "v1"

_FALSEY = ("", "0", "false", "off", "no")


def _env_on(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSEY


# ---------------------------------------------------------------------------
# Machine facts -- the only source. No caller may write one of these down.
# ---------------------------------------------------------------------------
# Every one of these numbers changes with the GPU, and a stale one does not
# raise: it silently sizes a tile or a register budget for the wrong machine.
# So they are read from the driver on the device that is actually in use, and
# they go into the cache key (below) so that a cache built elsewhere cannot be
# applied here.
_MACHINE_CACHE: dict = {}

_ATTRS = {
    "multiProcessorCount": "MULTIPROCESSOR_COUNT",
    "regsPerMultiprocessor": "MAX_REGISTERS_PER_MULTIPROCESSOR",
    "regsPerBlock": "MAX_REGISTERS_PER_BLOCK",
    "maxThreadsPerMultiProcessor": "MAX_THREADS_PER_MULTIPROCESSOR",
    "maxThreadsPerBlock": "MAX_THREADS_PER_BLOCK",
    "sharedMemPerMultiprocessor": "MAX_SHARED_MEMORY_PER_MULTIPROCESSOR",
    "sharedMemPerBlockOptin": "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN",
    "maxBlocksPerMultiProcessor": "MAX_BLOCKS_PER_MULTIPROCESSOR",
    "warpSize": "WARP_SIZE",
    "major": "COMPUTE_CAPABILITY_MAJOR",
    "minor": "COMPUTE_CAPABILITY_MINOR",
    # For the lower bound that prunes candidates before they are compiled.
    # Theoretical peak, not a measured one, and that is the safe direction: a
    # bandwidth that is too high makes every bound too small, so the pruning is
    # looser than it could be but never discards a candidate that could win.
    "memClockRate": "MEMORY_CLOCK_RATE",           # kHz
    "memBusWidth": "GLOBAL_MEMORY_BUS_WIDTH",      # bits
}


def _current_device() -> int:
    try:
        import paddle

        place = paddle.framework._current_expected_place()
        return int(getattr(place, "gpu_device_id", lambda: 0)())
    except Exception:
        return 0


def machine_facts(device: int | None = None) -> dict:
    """``cuDeviceGetAttribute`` for every quantity the tuner may use.

    Includes ``uuid`` for the log only -- it is deliberately *not* part of the
    cache key, or two identical GPUs in one job could not share a cache entry.
    """
    dev = _current_device() if device is None else device
    hit = _MACHINE_CACHE.get(dev)
    if hit is not None:
        return hit
    facts: dict = {}
    try:
        from cuda.bindings import driver as drv

        (err,) = (drv.cuInit(0),)
        (err, handle) = drv.cuDeviceGet(dev)
        for name, attr in _ATTRS.items():
            code = getattr(drv.CUdevice_attribute,
                           "CU_DEVICE_ATTRIBUTE_" + attr)
            err, val = drv.cuDeviceGetAttribute(code, handle)
            if int(err) == 0:
                facts[name] = int(val)
        err, uuid = drv.cuDeviceGetUuid(handle)
        if int(err) == 0:
            facts["uuid"] = bytes(uuid.bytes).hex()
    except Exception as exc:  # pragma: no cover - environment dependent
        facts["error"] = f"{type(exc).__name__}: {exc}"
    if "major" in facts:
        facts["arch"] = f"{facts['major']}.{facts['minor']}"
    _MACHINE_CACHE[dev] = facts
    return facts


def _arch(facts: dict) -> str:
    return facts.get("arch", "unknown")


# ---------------------------------------------------------------------------
# Toolchain facts -- in the key because the *behaviour being tuned around* is
# partly the toolchain's.
# ---------------------------------------------------------------------------
# Concretely: cuTile emits ``.reqntid`` + ``.minnctapersm 1`` and passes ptxas
# no register budget, so ptxas sizes registers for one CTA per SM and takes the
# whole file (exactly 255 at ntid=256, exactly 168 at ntid=384 = 65536/384
# rounded down to the 8-register granularity). Every register-cap decision in
# this repository exists because of that, and a toolchain upgrade can remove
# it. A cache entry from before the upgrade would then be tuned for a machine
# that no longer exists.
_TOOLCHAIN_CACHE: dict = {}


def toolchain_facts() -> dict:
    if _TOOLCHAIN_CACHE:
        return _TOOLCHAIN_CACHE
    out: dict = {"python": "%d.%d" % sys.version_info[:2]}
    for mod, key in (("cuda.tile", "cuda_tile"), ("triton", "triton"),
                     ("paddle", "paddle")):
        try:
            m = __import__(mod, fromlist=["__version__"])
            out[key] = str(getattr(m, "__version__", "unknown"))
        except Exception:
            pass
    try:
        import shutil

        ptxas = shutil.which("ptxas") or "/usr/local/cuda/bin/ptxas"
        txt = subprocess.run([ptxas, "--version"], capture_output=True,
                             text=True, timeout=60).stdout
        for line in txt.splitlines():
            if "release" in line:
                out["ptxas"] = line.strip()
                break
    except Exception:
        pass
    _TOOLCHAIN_CACHE.update(out)
    return out




def ctas_per_sm(regs: int, ntid: int, smem: int,
                facts: dict | None = None) -> dict:
    """Resident CTAs per SM implied by a compiled kernel's resource usage.

    Two uses, both of them bounding the candidate set before anything is
    compiled. First, an occupancy hint is silently ignored by cuTile whenever
    ``k * smem_per_block`` exceeds the shared memory an SM has, so such a
    candidate would compile to the same code as the unhinted one and be timed
    as though it were distinct -- it is simply not offered. Second, capping
    registers only buys occupancy when registers, and not shared memory, are
    what limits residency; ``by_reg`` and ``by_smem`` say which.
    """
    facts = machine_facts() if facts is None else facts
    out = {}
    if regs and ntid:
        out["by_reg"] = facts["regsPerMultiprocessor"] // (regs * ntid)
    if smem:
        out["by_smem"] = facts["sharedMemPerMultiprocessor"] // smem
    if ntid:
        out["by_thread"] = facts["maxThreadsPerMultiProcessor"] // ntid
    out["by_hw"] = facts.get("maxBlocksPerMultiProcessor", 32)
    out["blocks"] = min(v for v in out.values() if v)
    return out


def pow2_upto(limit: int, divides: int, count: int) -> list:
    """``count`` largest powers of two that are <= ``limit`` and divide.

    A candidate generator, not a policy: the bound comes from the caller's
    machine-derived ``limit`` and the shape's divisibility, so the list shrinks
    on a smaller machine instead of staying at what one machine liked.
    """
    out = []
    v = 1 << int(math.floor(math.log2(max(limit, 1))))
    while v >= 1 and len(out) < count:
        if divides % v == 0:
            out.append(v)
        v //= 2
    return out


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------
# Dropping any component of this key does not produce an error; it produces a
# configuration tuned for something else, applied silently. ``launch_shape`` is
# stored verbatim and never bucketed, because two shapes in one bucket can have
# different optima and bucketing hides that. If the number of entries ever
# needs bounding, evict by age -- do not widen the key.
def cache_key(name: str, launch_shape, dtypes, src_hash: str,
              facts: dict | None = None, toolchain: dict | None = None,
              extra=None) -> tuple:
    facts = machine_facts() if facts is None else facts
    toolchain = toolchain_facts() if toolchain is None else toolchain
    machine = {k: v for k, v in sorted(facts.items())
               if k not in ("uuid", "error", "arch")}
    plain = {
        "layout": CACHE_LAYOUT,
        "kernel": name,
        "arch": _arch(facts),
        "machine": machine,
        "toolchain": dict(sorted(toolchain.items())),
        "src_sha1": src_hash,
        "dtypes": [str(d) for d in dtypes],
        "launch_shape": [int(v) for v in launch_shape],
    }
    if extra:
        plain["extra"] = extra
    blob = json.dumps(plain, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:16], plain


def source_hash(*objs) -> str:
    """sha1 of the source of the functions/strings that define a kernel.

    In the key so that editing a kernel invalidates its tuning instead of
    running new code under an old configuration.
    """
    import inspect

    h = hashlib.sha1()
    for o in objs:
        if isinstance(o, (bytes, bytearray)):
            h.update(bytes(o))
            continue
        if isinstance(o, str):
            h.update(o.encode())
            continue
        try:
            fn = getattr(o, "_pyfunc", o)
            h.update(inspect.getsource(fn).encode())
        except (OSError, TypeError):
            h.update(repr(o).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Cache directory: one file per key, atomic publish, advisory lock
# ---------------------------------------------------------------------------
#: How long to block on another process's scan before giving up and scanning
#: too. Sized against the scans themselves, not guessed: the slowest entry
#: measured on a production shape took 22.3 s, and the cost of overshooting is
#: only duplicated work on a GPU this process owns anyway.
_LOCK_WAIT_SECONDS = 120.0

#: Cache root when the host framework has one. Reading it means the tuning
#: results live under the same tree as every other JIT cache, and are carried
#: between nodes by whatever already distributes that tree.
_HOST_CACHE_ROOT_ENV = "KERNEL_WARMUP_CACHE_ROOT"


def cache_dir() -> str:
    """Where tuning results are kept. **Node local is fine and is the default.**

    This used to default to ``~/.cache/paddlefleet/autotune`` while the rest of
    the module assumed every rank could see what one designated rank published.
    On a multi-node job ``$HOME`` is per container, so the publisher wrote where
    nobody else could read, every other rank waited out its timeout, and the
    whole job silently ran on the factory configuration. Startup went past
    twenty minutes and looked like a hang rather than a misconfiguration.

    The lookup is no longer a rendezvous -- any process that misses simply scans
    on its own GPU (see the module docstring) -- so a node-local directory is
    correct, not a compromise. Two consequences worth stating:

    * Nothing breaks when the directory is not shared. That is the property the
      old default silently lacked.
    * Riding the host framework's cache root (``KERNEL_WARMUP_CACHE_ROOT``, if
      it is set) means the results sit alongside the Triton / cuTile / other JIT
      caches and are carried between nodes by whatever already ships that tree,
      so the scan is paid once per cluster rather than once per node -- without
      this module having to know how that distribution works, or requiring it.

    Precedence: ``PADDLEFLEET_AUTOTUNE_CACHE_DIR`` (explicit, wins), then the
    host cache root, then a per-user directory.
    """
    d = os.environ.get("PADDLEFLEET_AUTOTUNE_CACHE_DIR")
    if not d:
        root = os.environ.get(_HOST_CACHE_ROOT_ENV)
        if root and root.strip():
            d = os.path.join(root.strip().rstrip("/"), "autotune")
        else:
            d = os.path.join(
                os.environ.get("XDG_CACHE_HOME",
                               os.path.expanduser("~/.cache")),
                "paddlefleet", "autotune",
            )
    d = os.path.join(d, CACHE_LAYOUT)
    os.makedirs(d, exist_ok=True)
    return d


def _entry_path(name: str, key: str) -> str:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    return os.path.join(cache_dir(), f"{safe}__{key}.json")


def _read_entry(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        # A truncated read is what the atomic publish below prevents for
        # writers; a reader that still sees one treats it as a miss.
        return None


def _publish(path: str, record: dict) -> None:
    """Write then rename. A reader on another node cannot tell a partial file
    from a complete one, so it must never see one."""
    tmp = f"{path}.tmp.{os.getpid()}.{int(time.time() * 1e6)}"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class _Lock:
    """``flock`` on a sidecar file; ``acquired`` is False if someone has it."""

    def __init__(self, path: str, blocking: bool = True, timeout: float = 0.0):
        self.path = path + ".lock"
        self.blocking = blocking
        self.timeout = timeout
        self.fh = None
        self.acquired = False

    def __enter__(self):
        try:
            self.fh = open(self.path, "a+")
        except OSError:
            return self
        deadline = time.time() + self.timeout
        while True:
            try:
                fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.acquired = True
                return self
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    return self
                if not self.blocking or time.time() > deadline:
                    return self
                time.sleep(0.2)

    def __exit__(self, *_):
        if self.fh is not None:
            if self.acquired:
                fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()


# ---------------------------------------------------------------------------
# The bit-exactness gate
# ---------------------------------------------------------------------------
def raw_bytes(tensor) -> bytes:
    """The tensor's storage, as bytes, with no dtype conversion.

    ``Tensor.numpy()`` hands bfloat16 back as ``uint16`` and float16 as
    ``float16``, both of which are the stored bit pattern, so this is a
    byte-level view and not a comparison with a tolerance. A widening cast
    here would be the difference between "bit-exact" and "close", which is the
    whole point of the gate.
    """
    arr = tensor.numpy()
    return arr.dtype.str.encode() + arr.shape.__repr__().encode() \
        + arr.tobytes()


def outputs_signature(tensors) -> str:
    h = hashlib.sha1()
    for t in tensors:
        h.update(raw_bytes(t))
    return h.hexdigest()
def _first_diff(a, b) -> dict:
    """Deprecated placeholder kept out of the public surface."""
    raise NotImplementedError


def explain_miss(name: str, key: str, plain: dict) -> str:
    """Why an existing cache entry for this kernel was not applied.

    A miss must never be silent: the failure mode being guarded against is a
    configuration swept elsewhere being applied here, and its mirror image is
    a cache that quietly stops being used. This walks the entries stored for
    the same kernel and names the key components that differ.
    """
    others = []
    try:
        prefix = "".join(c if c.isalnum() or c in "._-" else "_"
                         for c in name) + "__"
        for fn in sorted(os.listdir(cache_dir())):
            if fn.startswith(prefix) and fn.endswith(".json"):
                others.append(os.path.join(cache_dir(), fn))
    except OSError:
        pass
    if not others:
        return (f"no cache entry for key {key} and none stored for {name} "
                f"at all (first run on this machine/shape)")
    parts = []
    for path in others[:4]:
        rec = _read_entry(path)
        if not rec or "key_plain" not in rec:
            continue
        diff = []
        for field in ("arch", "machine", "toolchain", "src_sha1", "dtypes",
                      "launch_shape", "layout", "extra"):
            mine, theirs = plain.get(field), rec["key_plain"].get(field)
            if mine != theirs:
                diff.append(field)
        parts.append(f"{os.path.basename(path)} differs in {diff or ['?']}")
    return (f"no cache entry for key {key}; {len(others)} entry(ies) exist "
            f"for {name} but " + "; ".join(parts))
# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
MIN_WARMUP = 5
MIN_ITERS = 20


def time_median_us(fn, warmup: int = MIN_WARMUP, iters: int = MIN_ITERS
                   ) -> dict:
    """Median per-call microseconds over CUDA events.

    The floors are not negotiable downwards: the first launches of a cuTile
    kernel include its compilation, and a single sample lands anywhere in a
    distribution whose spread is a few per cent.
    """
    import paddle

    warmup = max(warmup, MIN_WARMUP)
    iters = max(iters, MIN_ITERS)
    for _ in range(warmup):
        fn()
    paddle.device.synchronize()
    samples = []
    for _ in range(iters):
        e0 = paddle.device.cuda.Event(enable_timing=True)
        e1 = paddle.device.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        e1.synchronize()
        samples.append(e0.elapsed_time(e1) * 1000.0)
    med = statistics.median(samples)
    return {
        "us": med,
        "lo": min(samples),
        "hi": max(samples),
        "mad": statistics.median([abs(v - med) for v in samples]),
        "n": len(samples),
    }


def _cfg_to_str(cfg: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(cfg.items()))


def _framework_rank():
    """Rank for the provenance record only. ``None`` when it cannot be had.

    Deliberately asks the framework instead of reading launcher environment
    variables. An earlier version of this module walked a list of five variable
    names (``PADDLE_TRAINER_ID``, ``RANK``, ``OMPI_COMM_WORLD_RANK``, ...) and
    returned 0 when none matched. That is a guess whose ordering is load
    bearing: under ``mpirun`` plus a launcher, ``OMPI_COMM_WORLD_RANK`` is
    inherited by every process on a node and equals the node index, so if the
    first name in the list ever stops being set, every process on a node reports
    the same rank. It never raised, it just answered wrongly.

    Nothing depends on the answer any more -- who scans is decided by the lock --
    so this is pure provenance and is allowed to be unknown.
    """
    try:
        import paddle.distributed as dist

        return dist.get_rank()
    except Exception:  # noqa: BLE001 - provenance must never break a scan
        return None


def parse_config(text: str) -> dict:
    """``TILE_C=128,TILE_SIZE=2`` -> dict of ints (or strings if not ints)."""
    out: dict = {}
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        v = v.strip()
        try:
            out[k.strip()] = int(v)
        except ValueError:
            out[k.strip()] = v
    return out


MAX_CANDIDATES = 8


def _warn(msg: str) -> None:
    warnings.warn(f"[paddlefleet.autotune] {msg}", stacklevel=3)
    print(f"[paddlefleet.autotune] {msg}", file=sys.stderr, flush=True)


_MEMO: dict = {}


def tune(
    name: str,
    factory: dict,
    candidates,
    launch_shape,
    dtypes,
    src_hash: str,
    make_outputs=None,
    launch=None,
    env_prefix: str = "PADDLEFLEET_AUTOTUNE",
    warmup: int = MIN_WARMUP,
    iters: int = MIN_ITERS,
    input_sets: int = 1,
    reseed=None,
    bound=None,
    key_extra=None,
) -> dict:
    """Return the launch configuration to use for one kernel at one shape.

    ``launch(cfg, outputs)`` performs exactly one launch; ``make_outputs()``
    returns freshly allocated output tensors. Both are only called by the
    process that ends up scanning, and only on a cache miss.

    Candidates are expected to be feasible: a configuration the toolchain will
    silently ignore compiles to the same code as another candidate and would be
    timed as though it were distinct, so the caller keeps it out of the list
    (see ``ctas_per_sm``) instead of the tuner detecting it afterwards.

    ``bound(cfg)`` may return a lower bound in microseconds on what ``cfg`` can
    achieve. It is used only to drop candidates that cannot beat one already
    measured, which saves their compilation -- the expensive part. It must be a
    genuine lower bound: too small only weakens the pruning, while too large
    would discard a winner.

    Never raises: every failure path returns ``factory``.
    """
    if not _env_on(f"{env_prefix}_AUTOTUNE", True):
        return dict(factory)

    pinned = os.environ.get(f"{env_prefix}_PIN_{name.upper().replace('.', '_')}")
    if pinned is None:
        whole = os.environ.get(f"{env_prefix}_PIN")
        if whole:
            for chunk in whole.split(";"):
                head, _, rest = chunk.partition(":")
                if head.strip() == name:
                    pinned = rest
                    break
    if pinned:
        cfg = dict(factory)
        cfg.update(parse_config(pinned))
        return cfg

    key, plain = cache_key(name, launch_shape, dtypes, src_hash,
                           extra=key_extra)
    memo = _MEMO.get(key)
    if memo is not None:
        return dict(memo)

    path = _entry_path(name, key)
    entry = _read_entry(path)
    if entry is not None and entry.get("key") == key:
        _MEMO[key] = entry["config"]
        return dict(entry["config"])

    if make_outputs is None or launch is None:
        _warn(f"{name}: {explain_miss(name, key, plain)}, and this call site "
              f"gave the tuner nothing to measure; using the factory "
              f"configuration {_cfg_to_str(factory)}.")
        return dict(factory)

    # ---- who scans: whoever gets the lock, and nobody waits on a rank -------
    # There used to be a "global rank 0 scans, every other rank waits for it to
    # publish" branch here. It was wrong twice over.
    #
    # The premise was that concurrent scans measure each other, which is only
    # true when several processes share one GPU. In a job with one rank per GPU
    # they do not share the resource being measured, and the timing below runs
    # on CUDA events, i.e. on the GPU timeline. So there is nothing to serialise
    # across GPUs.
    #
    # And "global rank" was the wrong unit anyway: it made the ranks that own
    # the other GPUs on this node sit idle, while two jobs sharing a node would
    # both have a "rank 0" scanning at the same time without either noticing.
    # Determining it also meant guessing at launcher environment variables, and
    # guessing wrong did not raise -- a process wrongly deciding it was not rank
    # 0 waited out the timeout and then used the factory configuration for the
    # rest of the run.
    #
    # The lock alone does the whole job, on the right unit. Ranks that share a
    # cache directory (i.e. that are on one node, since the directory is node
    # local) serialise on it: the first one scans, the rest block, and when they
    # wake the entry is already there, so they read it and return. Wall time is
    # one scan either way, but only one GPU per node does the work and only one
    # process per node is compiling -- which also keeps sibling compilation off
    # the CPU while a measurement is running. Different nodes never meet, and do
    # not need to.
    with _Lock(path, blocking=True, timeout=_LOCK_WAIT_SECONDS) as lock:
        entry = _read_entry(path)
        if entry is not None and entry.get("key") == key:
            _MEMO[key] = entry["config"]
            return dict(entry["config"])
        if not lock.acquired:
            # Not a reason to give up on tuning: this process owns its own GPU,
            # so scanning without the lock costs duplicated work, not a corrupt
            # reading, and ``_publish`` renames into place atomically. Falling
            # back to the factory configuration here is what used to make a
            # contended start silently permanent.
            _warn(f"{name}: could not take the scan lock for {path}.lock "
                  f"within {_LOCK_WAIT_SECONDS:.0f}s; scanning anyway on this "
                  f"process's own device.")
        try:
            record = _scan(name, factory, candidates, plain, key,
                           make_outputs, launch, warmup, iters,
                           input_sets, reseed, bound)
        except Exception as exc:
            _warn(f"{name}: scan failed ({type(exc).__name__}: {exc}); "
                  f"using the factory configuration "
                  f"{_cfg_to_str(factory)}.")
            return dict(factory)
        _publish(path, record)
    _MEMO[key] = record["config"]
    return dict(record["config"])


def peak_bytes_per_us(facts: dict | None = None) -> float:
    """Theoretical peak DRAM bytes per microsecond, or 0 when unknown.

    Theoretical, not measured, and that is the safe direction for a lower bound
    on time: too high a bandwidth makes every bound too small, so pruning is
    looser than it could be and never discards a candidate that could have won.
    """
    facts = machine_facts() if facts is None else facts
    khz, bits = facts.get("memClockRate"), facts.get("memBusWidth")
    if not khz or not bits:
        return 0.0
    return 2.0 * khz * 1e3 * (bits / 8.0) / 1e6


def _scan(name, factory, candidates, plain, key, make_outputs, launch,
          warmup, iters, input_sets, reseed, bound=None) -> dict:
    """Measure, pruning by a lower bound first; verify the winner's bytes last.

    ## Why this order

    Cost is dominated by compilation: each candidate's first launch compiles it,
    and measured on production shapes the pass that first launches every
    candidate was 98% of the scan (13.0 s of 13.2, 22.1 of 22.3). So the way to
    make a scan cheap is to compile fewer candidates, not to time them less.

    Two changes follow from that:

    * **A lower bound prunes before compiling.** ``bound(cfg)`` returns
      microseconds that ``cfg`` cannot beat. Once some candidate has been
      measured at ``t_best``, any candidate whose bound is already >= ``t_best``
      cannot win and is dropped without being built. This is sound in one
      direction only, which is the direction used: a lower bound can be
      exceeded, never undercut, so pruning by it cannot discard the winner.
      Candidates are tried in increasing bound order so that the cheap
      information arrives first and prunes the most.
    * **The bit-exactness gate runs on the winner, not on everybody.** It used
      to launch every candidate and copy every output tensor back to the host to
      compare bytes -- on a production shape that is hundreds of megabytes per
      candidate per input set, plus a fresh output allocation each time, inside
      a training step. Verifying in speed order and stopping at the first
      candidate that passes gives the same answer: a candidate that is not
      bit-exact is rejected before it can be returned, and if the fastest one
      fails, the next fastest is checked. The only case that costs more than
      before is one where several of the fastest candidates change the output,
      which has not happened on this family.

    What is deliberately *not* claimed: the bound does not rank candidates. It
    could not -- two configurations with identical bounds have measured 1.7-2.6%
    apart, and static predictions of speed have been wrong here by 3.7x. It only
    says who cannot win.
    """
    import paddle

    uniq = [dict(factory)]
    for cfg in candidates:
        merged = dict(factory)
        merged.update(cfg)
        if merged not in uniq:
            uniq.append(merged)
    if len(uniq) > MAX_CANDIDATES:
        # A budget, not a preference: this runs inside the first training step.
        uniq = uniq[:MAX_CANDIDATES]

    t0 = time.time()
    rejected = []

    bound_broken = []

    def bound_of(cfg):
        """The caller's lower bound, or None. A broken bound is loud.

        It must be loud: a bound that raises used to be swallowed into None,
        which silently turned the pruning off. That is exactly how this function
        was first landed against the wrong call site -- the lambda referred to
        names that did not exist there, every candidate came back unbounded, and
        the only symptom was that nothing got pruned.
        """
        if bound is None:
            return None
        try:
            v = bound(cfg)
        except Exception as exc:
            if not bound_broken:
                bound_broken.append(f"{type(exc).__name__}: {exc}")
                _warn(f"{name}: the lower bound raised "
                      f"({bound_broken[0]}); pruning is off for this scan, so "
                      f"every candidate will be compiled.")
            return None
        return v if v and v > 0 else None

    # The factory config is measured first and unconditionally: it is the
    # bit-exactness reference, it is what we fall back to, and its time is what
    # gives the pruning something to prune against.
    def measure(cfg):
        outs = make_outputs()
        launch(cfg, outs)                      # compiles on the first launch
        paddle.device.synchronize()
        t = time_median_us(lambda: launch(cfg, outs), warmup, iters)
        del outs
        return t

    timings = [{"config": uniq[0], **measure(uniq[0])}]
    t_best = timings[0]["us"]

    rest = sorted(uniq[1:], key=lambda c: (bound_of(c) or 0.0))
    for cfg in rest:
        lb = bound_of(cfg)
        if lb is not None and lb >= t_best:
            rejected.append({"config": cfg, "reason": "bounded_out",
                             "lower_bound_us": lb, "best_us": t_best})
            continue
        try:
            t = measure(cfg)
        except Exception as exc:
            rejected.append({"config": cfg, "reason": "failed",
                             "detail": {"error": f"{type(exc).__name__}: {exc}"}})
            continue
        timings.append({"config": cfg, "lower_bound_us": lb, **t})
        t_best = min(t_best, t["us"])

    timings.sort(key=lambda r: r["us"])

    # -- verify bytes, fastest first, stop at the first that matches ----------
    references = None
    winner = None
    gate_checked = 0
    for r in timings:
        if r["config"] == uniq[0]:
            winner = uniq[0]           # the reference cannot differ from itself
            break
        if references is None:
            references = []
            for s in range(max(input_sets, 1)):
                if s and reseed is not None:
                    reseed(s)
                outs = make_outputs()
                launch(uniq[0], outs)
                paddle.device.synchronize()
                references.append([raw_bytes(t) for t in outs])
                del outs
        bad = None
        for s in range(max(input_sets, 1)):
            if reseed is not None:
                reseed(s)
            outs = make_outputs()
            launch(r["config"], outs)
            paddle.device.synchronize()
            got = [raw_bytes(t) for t in outs]
            del outs
            if got != references[s]:
                bad = {"input_set": s, "differing_outputs": [
                    i for i, (a, b) in enumerate(zip(got, references[s]))
                    if a != b]}
                break
        gate_checked += 1
        if bad is None:
            winner = r["config"]
            break
        rejected.append({"config": r["config"], "reason": "not_bit_exact",
                         "detail": bad, "us": r["us"]})
    if reseed is not None:
        reseed(0)
    if winner is None:
        winner = uniq[0]

    base = next(r for r in timings if r["config"] == uniq[0])
    win_t = next(r for r in timings if r["config"] == winner)
    for r in timings:
        if r["config"] != winner and r["config"] != uniq[0] \
                and not any(x["config"] == r["config"] for x in rejected):
            rejected.append({"config": r["config"], "reason": "slower",
                             "us": r["us"],
                             "vs_winner": r["us"] / win_t["us"]})

    return {
        "key": key,
        "config": winner,
        "factory": dict(factory),
        "speedup_vs_factory": base["us"] / win_t["us"],
        "bit_exact_vs_factory": True,
        "bit_exact_input_sets": max(input_sets, 1),
        "bit_exact_candidates_checked": gate_checked,
        "timings": timings,
        "rejected": rejected,
        "candidates_considered": len(uniq),
        "candidates_compiled": len(timings),
        "peak_bytes_per_us": peak_bytes_per_us(),
        "bound_error": bound_broken[0] if bound_broken else None,
        "scan_seconds": time.time() - t0,
        "measured": {"warmup": max(warmup, MIN_WARMUP),
                     "iters": max(iters, MIN_ITERS)},
        "key_plain": plain,
        "written_by": {
            "rank": _framework_rank(),
            "pid": os.getpid(),
            "host": os.uname().nodename,
            "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "device_uuid": machine_facts().get("uuid"),
        },
    }
