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

"""Fused cuTile kernels for mHC (Manifold-Constrained Hyper-Connections).

Requires cuda.tile (cuTile) for optimal performance on supported GPUs
(compute capability 10.x+).  Reference (non-fused) implementations live in
``paddlefleet.transformer.hyper_connection`` and are used when cuTile is
unavailable or when the ``use_fused_mhc`` config flag is False.

Four fused operations:
  - sinkhorn:     Sinkhorn-Knopp projection to doubly stochastic matrix
  - h_aggregate:  weighted n-stream -> 1-stream aggregation
  - h_post_bda:   fused H_res @ residual + H_post * (x + bias)
  - proj_rms:     fused projection + RMS normalization
  - compute_h:    fused r * proj * alpha + bias, plus the two sigmoid heads
"""

import math
import os

import paddle
from paddle import Tensor

# ---------------------------------------------------------------------------
# Tile shapes: autotuned, not written down (paddlefleet/autotune.py)
# ---------------------------------------------------------------------------
# Every launcher below used to pick its cuTile tile shape from a hardcoded
# expression, and a previous round replaced four of those with a *second* set
# of hardcoded expressions swept on one machine at one shape. Both are the
# same mistake: the earlier instance of it -- six launch configurations
# written down from one sweep, with no architecture test -- was "effective but
# harmful" on another generation (3940 B of spill against 312 B, resident CTAs
# per SM 2 -> 1) and had to be reverted in full.
#
# So the tile shapes are now decided by ``paddlefleet.autotune.tune``: the
# candidate list is bounded by what the shape divides and by what an SM
# actually has, every candidate must reproduce the shipped configuration's
# output byte for byte before it is timed at all, and the winner is the
# measured-fastest survivor. Nothing here needs a per-machine re-verification
# of numerics; that is the property the bit-exactness gate buys.
#
# WHICH TILE VALUES ARE LEGAL IS STILL NOT A FREE CHOICE -- the gate is what
# enforces it now instead of a comment. Measured per kernel over 20 random
# inputs at the production shapes (opt/mhc_0908/logs/bitexact_prod.txt):
#
#   * a reduction carried in a loop-local accumulator -- the unrolled ``j``
#     sums in the hpb kernels, ``ct.mma(acc=...)`` across m-tiles in
#     proj_rms_bwd -- keeps its summation order under any tile shape, because
#     chunking a sequential accumulation differently does not reorder it.
#     Tile changes there are bit-exact.
#   * a reduction written as ``ct.sum`` is a lane-shuffle tree whose shape
#     follows the *whole* tile shape, not just the length of the reduced axis.
#     Almost any tile change reassociates it (hpb_bwd's g_hp/g_hr, h_agg_bwd's
#     gh, proj_rms_fwd's norm, every sinkhorn reduction). ``_ct_hpb_fwd_kernel``
#     is the only member of the family with no ``ct.sum`` at all.
#
# The candidate lists below therefore still only offer values that were
# measured legal; the gate is the backstop that makes a wrong entry harmless
# instead of a silent numerics change.
_MHC_TUNE_PREFIX = "PADDLEFLEET_MHC_TILE"

# One ``tune`` call per (kernel, shape) per process. ``tune`` memoises on the
# full cache key, but building that key means hashing the machine and toolchain
# facts, and these launchers run once per layer per step.
_MHC_TUNED: dict = {}


def _mhc_tune(name: str, shape_key, build):
    """``build()`` -> config, called at most once per (kernel, shape).

    ``build`` is a thunk so the closures a scan needs are not constructed on
    the hot path. Any failure is reported once and falls back to the shipped
    tiles: this must never be able to break a launch.
    """
    memo_key = (name, shape_key)
    if memo_key in _MHC_TUNED:
        return _MHC_TUNED[memo_key] or None
    try:
        got = build()
    except Exception as exc:  # pragma: no cover - environment dependent
        print(
            f"[paddlefleet] mHC autotune for {name} failed "
            f"({type(exc).__name__}: {exc}); using the shipped tiles.",
            flush=True,
        )
        got = None
    _MHC_TUNED[memo_key] = got if got else False
    return got or None


def _mhc_autotune_call(name, factory, candidates, launch_shape, dtypes,
                       kernels, make_outputs, launch, **kw):
    """Thin wrapper so a missing ``paddlefleet.autotune`` is not fatal."""
    from paddlefleet.autotune import source_hash, tune

    return tune(
        name=name,
        factory=factory,
        candidates=candidates,
        launch_shape=launch_shape,
        dtypes=dtypes,
        src_hash=source_hash(*kernels),
        make_outputs=make_outputs,
        launch=launch,
        env_prefix=_MHC_TUNE_PREFIX,
        **kw,
    )


_MHC_TILE_OPT_ENV = "PADDLEFLEET_MHC_TILE_OPT"
_MHC_TILE_OPT_WARNED: list = []


def _mhc_live_budget() -> int:
    """Bytes of live tile a single block may hold, read from the driver.

    Both halves are machine facts: ``regsPerBlock`` words are what ptxas may
    allocate to one block, and ``sharedMemPerBlockOptin`` is the most shared
    memory one block may address. A tile shape whose live footprint exceeds the
    smaller of the two cannot be resident, so it is not offered as a candidate.
    This is a bound on the candidate set, not a prediction of which candidate
    wins -- that is measured, because static occupancy has pointed at the wrong
    winner by 3.7x.
    """
    from paddlefleet.autotune import machine_facts

    facts = machine_facts()
    return min(facts.get("regsPerBlock", 0) * 4,
               facts.get("sharedMemPerBlockOptin", 0)) or (1 << 20)


# A tile row shorter than one 128-byte sector cannot use the transaction it
# pays for; measured on a sibling kernel, a transpose whose tile row fell below
# 128 B lost 2.4x. 128 is an invariant of the memory system, in the same
# category as a 32-lane ballot mask, so it is written down -- unlike anything
# that scales with the part.
_SECTOR_BYTES = 128


def _mhc_dtype_bytes(dtype) -> int:
    """Bytes per element of a paddle dtype, without allocating a tensor."""
    import paddle

    try:
        return paddle.empty([0], dtype=dtype).element_size()
    except Exception:
        return 2


def _mhc_tile_bound(tiled_bytes: int, rows: int, tile_rows: int,
                    tile_row_bytes: int):
    """Microseconds a tile shape cannot beat, or ``None`` when not computable.

    A lower bound, used only to drop candidates that cannot beat one already
    measured so that their compilation is never paid -- and compilation is 98%
    of a scan. It deliberately does not rank: two shapes with equal bounds have
    measured 1.7-2.6% apart, and predicting speed from static numbers has been
    wrong here by 3.7x.

    Two terms, both of them true floors:

    * **Sector waste.** A tile row of R bytes with R < 128 still costs a full
      128 B transaction, so the traffic is at least ``128 / R`` times the useful
      bytes. This is what makes the very narrow tiles hopeless -- measured 2.4x
      to 32x slower than the winner -- and it prunes them by arithmetic instead
      of by the shape of a sampling heuristic.
    * **Idle multiprocessors.** With fewer blocks than multiprocessors, at most
      ``grid`` of them can be busy, so the time is at least ``SMs / grid`` times
      the ideal.

    ``tiled_bytes`` must count only traffic that the tile actually governs, and
    counting less than the truth is the safe direction: every term here makes
    the bound smaller than the real floor, never larger, so pruning by it cannot
    discard a candidate that would have won.
    """
    from paddlefleet.autotune import machine_facts, peak_bytes_per_us

    bw = peak_bytes_per_us()
    if bw <= 0 or tiled_bytes <= 0 or tile_rows <= 0:
        return None
    ideal = tiled_bytes / bw
    inflate = max(1.0, _SECTOR_BYTES / max(tile_row_bytes, 1))
    sm = machine_facts().get("multiProcessorCount") or 0
    grid = math.ceil(rows / tile_rows)
    starve = max(1.0, sm / grid) if (sm and grid) else 1.0
    return ideal * max(inflate, starve)


def _mhc_pair_candidates(factory, rows, cols, per_elem_bytes, elem_bytes,
                         count=8):
    """``(rows, cols)`` tile candidates around the shipped pair.

    Three rules, each of them there because a simpler generator was measured
    to miss:

    * walk the row count up in powers of two (x1, x2, x4). Beyond that nothing
      won on any of these kernels.
    * at each row count offer three columns: the *widest that still fits*, and
      two narrow ones (the shipped width over 32 and over 64). Sampling the
      whole fitting set evenly instead spends slots on tiles 2 and 16 columns
      wide, measured 2.4x to 32x slower than the winner; offering only the
      over-64 narrow branch missed one kernel's best tile by 1.7-2.6%
      (measured: 419.6 us at (2, 128) against 426.8 us at (4, 64), 200
      launches each, repeated, exclusively-owned card). The landscape is not
      monotonic in either axis, so the coverage has to be geometric in both.
    * never go below one sector per tile row.

    ``factory`` is first, so it is the bit-exactness reference and cannot be
    pushed out by the candidate budget.
    """
    budget = _mhc_live_budget()
    fr, fc = factory
    floor_c = max(_SECTOR_BYTES // max(elem_bytes, 1), 1)
    out = [(fr, fc)]

    def widest(r, start):
        c = start
        while c >= 1:
            if cols % c == 0 and r * c * per_elem_bytes <= budget:
                return c
            c //= 2
        return 0

    for mult in (1, 2, 4):
        r = fr * mult
        if r > rows or rows % r:
            continue
        for start in (fc, max(fc // 32, floor_c), max(fc // 64, floor_c)):
            c = widest(r, start)
            if c >= floor_c and (r, c) not in out and len(out) < count:
                out.append((r, c))
    return out


def _mhc_size_candidates(factory, extent, per_row_bytes, count=8):
    """One-dimensional tile extents: powers of two around the shipped one.

    Up while the live footprint still fits, and two steps down. Enumerating
    every divisor instead offers extents of 1 and 2, which are not close to
    optimal for anything in this family and cost a timing pass each.
    """
    budget = _mhc_live_budget()
    out = [factory]
    v = factory * 2
    while v <= extent and extent % v == 0 and v * per_row_bytes <= budget:
        out.append(v)
        v *= 2
    v = factory // 2
    for _ in range(2):
        if v >= 1 and extent % v == 0 and v not in out:
            out.append(v)
        v //= 2
    return out[:count]


_MHC_OCC_ENV = "PADDLEFLEET_MHC_TILE_OCC"


def _mhc_occ_candidates(factory_pair, allowed: bool, per_elem_bytes: int = 0):
    """Occupancy-hint candidates, off unless asked for, and only feasible ones.

    Starting at 2 rather than at 4 is deliberate: the values that were tried by
    hand (4/8/12/16) are the ones whose shared-memory demand this family cannot
    meet, so they were silently dropped, and 2 and 3 -- never tried -- are the
    only ones with room. They are off by default because the candidate budget
    is eight per kernel and the tile shapes are worth more (the best hinted
    configuration measured 1.2% ahead of the best unhinted one, against a 1.9x
    spread across tiles).

    ``PADDLEFLEET_MHC_TILE_OCC=2,3`` spends two of the eight slots on them.

    **Feasibility is computed here, not observed afterwards.** cuTile drops
    ``occupancy=k`` whenever k CTAs' worth of shared memory does not fit, and
    the result compiles to the same code as the unhinted kernel -- so an
    infeasible hint is not a slow candidate, it is the *same* candidate timed
    twice, which is how a knob that does nothing gets believed.

    This used to be handled by launching the candidate and then reading the
    compiled cubin back out of cuTile's on-disk sqlite cache to see what it had
    actually done. That was wrong three ways: it took the most recently inserted
    blob with no key match (so anything else compiling in between was reported
    as this kernel's), it opened a database whose owner deletes it on any sqlite
    error, and it shelled out to ``cuobjdump`` twice per candidate from inside a
    training step. The same conclusion follows from
    ``k * smem_per_cta > smem_per_sm``, both of which are already known here --
    so the infeasible hints are simply never offered, which also saves the
    compile that observing them would have cost.
    """
    raw = os.environ.get(_MHC_OCC_ENV, "")
    if not raw.strip() or not allowed or not factory_pair:
        return []
    per_cta = (factory_pair[0] or 0) * (factory_pair[1] or 0) * per_elem_bytes
    per_sm = 0
    if per_cta > 0:
        from paddlefleet.autotune import machine_facts

        per_sm = machine_facts().get("sharedMemPerMultiprocessor") or 0
    occs = []
    for part in raw.replace(";", ",").split(","):
        try:
            v = int(part)
        except ValueError:
            continue
        if v < 2:
            continue
        if per_sm and per_cta and v * per_cta > per_sm:
            # Would be dropped by the compiler; offering it costs a compile and
            # buys a duplicate of the unhinted candidate.
            continue
        occs.append(v)
    return [{"TILE_SIZE": factory_pair[0], "TILE_C": factory_pair[1],
             "OCC": v} for v in occs]


def _mhc_tile_opt_retired() -> None:
    """The previous round's switch no longer selects anything -- say so.

    It installed the four tile shapes that are now candidates for the tuner.
    Leaving it silently inert would let a launch script keep setting it and
    believe it still does something.
    """
    if _MHC_TILE_OPT_WARNED:
        return
    raw = os.environ.get(_MHC_TILE_OPT_ENV)
    if raw is None or raw.strip().lower() in ("", "0", "false", "off", "no"):
        return
    _MHC_TILE_OPT_WARNED.append(True)
    print(
        f"[paddlefleet] {_MHC_TILE_OPT_ENV}={raw} is retired and ignored: the "
        f"four tile shapes it used to install are now candidates that "
        f"{_MHC_TUNE_PREFIX}_* picks by measurement. To hold a configuration "
        f"fixed (for an A/B) use "
        f"{_MHC_TUNE_PREFIX}_PIN_<KERNEL>='TILE_SIZE=2,TILE_C=128'; to keep "
        f"the shipped tiles use {_MHC_TUNE_PREFIX}_AUTOTUNE=0.",
        flush=True,
    )


_MHC_HINTED_KERNELS: dict = {}


def _mhc_with_occupancy(kernel, occ: int):
    """``kernel`` recompiled for a target of ``occ`` resident blocks per SM.

    cuTile's ``occupancy`` hint caps registers per thread, which is the only
    published knob that moves this family off 8 resident warps out of 64. It
    also rewrites the shared-memory pipeline, which changes instruction order
    and can therefore reassociate a lane-shuffle reduction -- so it is offered
    as a *candidate* only where that was measured harmless
    (``_ct_hpb_fwd_kernel``, 20 + 20 random inputs at occupancy 4/8/12/16:
    opt/mhc_0908/logs/hints_hpb_fwd.txt, logs/bitexact_extra.txt), and the
    bit-exactness gate rejects it anywhere it is not. It is silently dropped
    whenever ``occ * smem_per_block`` exceeds the shared memory an SM has,
    which is why the candidate set starts at 2 rather than at 4 and why a
    dropped hint is detected instead of being timed as a distinct
    configuration.

    Cached: ``replace_hints`` builds a new kernel object, and this is on the
    per-launch path.
    """
    if occ <= 0:
        return kernel
    key = (id(kernel), occ)
    hinted = _MHC_HINTED_KERNELS.get(key)
    if hinted is None:
        hinted = kernel.replace_hints(occupancy=occ)
        _MHC_HINTED_KERNELS[key] = hinted
    return hinted

# ---------------------------------------------------------------------------
# Check cuTile availability
# ---------------------------------------------------------------------------
_CUTILE_AVAILABLE = False
try:
    import cuda.tile as ct

    _CUTILE_AVAILABLE = True
except ImportError:
    pass


def is_cutile_available() -> bool:
    """Return True if cuTile fused kernels are available."""
    return _CUTILE_AVAILABLE


def _get_cuda_stream():
    """Get current CUDA stream for cuTile launch."""
    return paddle.device.current_stream().stream_base.cuda_stream






















# ============================================================================
# CuTile implementations (only defined when cuda.tile is available)
# ============================================================================

if _CUTILE_AVAILABLE:
    ConstInt = ct.Constant[int]
    ConstBool = ct.Constant[bool]
    PAD_ZERO = ct.PaddingMode.ZERO
    LOG2E = 1.4426950408889634
    _INT32_MAX = 2**31 - 1

    # -- Sinkhorn kernels ----------------------------------------------------

    @ct.kernel
    def _ct_sinkhorn_fwd_kernel(
        inp,
        out,
        M_init_out,
        eps,
        HC: ConstInt,
        NUM_ITERS: ConstInt,
        TILE_SIZE: ConstInt,
    ):
        pid = ct.bid(0)
        logits = ct.load(
            inp, index=(pid, 0, 0), shape=(TILE_SIZE, HC, HC)
        ).astype(ct.float32)
        row_max = ct.max(logits, axis=2, keepdims=True)
        M = ct.exp2((logits - row_max) * LOG2E)
        ct.store(
            M_init_out,
            index=(pid, 0, 0),
            tile=ct.reshape(M.astype(M_init_out.dtype), (TILE_SIZE, HC, HC)),
        )
        row_sum = ct.sum(M, axis=2, keepdims=True)
        M = M / row_sum + eps
        col_sum = ct.sum(M, axis=1, keepdims=True)
        M = M / (col_sum + eps)
        for _ in range(NUM_ITERS - 1):
            row_sum = ct.sum(M, axis=2, keepdims=True)
            M = M / (row_sum + eps)
            col_sum = ct.sum(M, axis=1, keepdims=True)
            M = M / (col_sum + eps)
        ct.store(
            out,
            index=(pid, 0, 0),
            tile=ct.reshape(M.astype(out.dtype), (TILE_SIZE, HC, HC)),
        )

    @ct.kernel
    def _ct_sinkhorn_bwd_kernel(
        grad_out,
        M_init,
        grad_inp,
        ws_M,
        ws_rs,
        ws_cs,
        eps,
        HC: ConstInt,
        NUM_ITERS: ConstInt,
        TILE_SIZE: ConstInt,
    ):
        pid = ct.bid(0)
        M_base = pid * (2 * NUM_ITERS)
        v_base = pid * NUM_ITERS

        M = ct.load(
            M_init, index=(pid, 0, 0), shape=(TILE_SIZE, HC, HC)
        ).astype(ct.float32)
        for t in range(NUM_ITERS):
            ct.store(ws_M, index=(M_base + 2 * t, 0, 0), tile=M)
            row_sum = ct.sum(M, axis=2, keepdims=True)
            ct.store(ws_rs, index=(v_base + t, 0, 0), tile=row_sum)
            if t == 0:
                M = M / row_sum + eps
            else:
                M = M / (row_sum + eps)
            ct.store(ws_M, index=(M_base + 2 * t + 1, 0, 0), tile=M)
            col_sum = ct.sum(M, axis=1, keepdims=True)
            ct.store(ws_cs, index=(v_base + t, 0, 0), tile=col_sum)
            M = M / (col_sum + eps)

        grad = ct.load(
            grad_out, index=(pid, 0, 0), shape=(TILE_SIZE, HC, HC)
        ).astype(ct.float32)
        for t_rev in range(NUM_ITERS):
            t = NUM_ITERS - 1 - t_rev
            col_s = ct.load(
                ws_cs, index=(v_base + t, 0, 0), shape=(TILE_SIZE, 1, HC)
            )
            grad = grad / (col_s + eps)
            col_corr = ct.sum(grad * M, axis=1, keepdims=True)
            grad = grad - col_corr
            M = ct.load(
                ws_M,
                index=(M_base + 2 * t + 1, 0, 0),
                shape=(TILE_SIZE, HC, HC),
            )
            row_s = ct.load(
                ws_rs, index=(v_base + t, 0, 0), shape=(TILE_SIZE, HC, 1)
            )
            if t == 0:
                grad = grad / row_s
                row_corr = ct.sum(grad * (M - eps), axis=2, keepdims=True)
            else:
                grad = grad / (row_s + eps)
                row_corr = ct.sum(grad * M, axis=2, keepdims=True)
            grad = grad - row_corr
            M = ct.load(
                ws_M, index=(M_base + 2 * t, 0, 0), shape=(TILE_SIZE, HC, HC)
            )
        grad = grad * M
        ct.store(grad_inp, index=(pid, 0, 0), tile=grad.astype(grad_inp.dtype))

    def _cutile_sinkhorn_fwd(
        input_logits: Tensor, num_iterations: int, eps: float = 1e-8
    ) -> tuple[Tensor, Tensor]:
        original_shape = input_logits.shape
        hc = original_shape[-1]
        N_batch = input_logits.size // (hc * hc)
        # Optimized: TILE_SIZE=64 is fastest for fwd (0.061ms vs 0.071ms@128)
        TILE_SIZE = math.gcd(N_batch, 64)
        out = paddle.empty(shape=[N_batch, hc, hc], dtype=input_logits.dtype)
        M_init = paddle.empty(shape=[N_batch, hc, hc], dtype=input_logits.dtype)
        ct.launch(
            _get_cuda_stream(),
            (math.ceil(N_batch / TILE_SIZE), 1, 1),
            _ct_sinkhorn_fwd_kernel,
            (
                input_logits.reshape([N_batch, hc, hc]),
                out,
                M_init,
                eps,
                hc,
                num_iterations,
                TILE_SIZE,
            ),
        )
        return out.reshape(original_shape), M_init.reshape(original_shape)

    def _cutile_sinkhorn_bwd(
        grad_output: Tensor,
        M_init: Tensor,
        num_iterations: int,
        eps: float = 1e-8,
    ) -> Tensor:
        original_shape = grad_output.shape
        hc = original_shape[-1]
        N_batch = grad_output.size // (hc * hc)
        SHIPPED_TS = math.gcd(N_batch, 64)
        ws_M = paddle.empty(
            shape=[N_batch * 2 * num_iterations, hc, hc], dtype="float32"
        )
        ws_rs = paddle.empty(
            shape=[N_batch * num_iterations, hc, 1], dtype="float32"
        )
        ws_cs = paddle.empty(
            shape=[N_batch * num_iterations, 1, hc], dtype="float32"
        )
        # opt/mhc_0908: every reduction in this kernel is a ``ct.sum`` and they
        # are chained 40 deep, so TILE_SIZE is a numerics knob almost
        # everywhere: measured exact only at 128 (16 and 32 differ, up to 2e4
        # fp32 ULP on 82% of the elements -- logs/bitexact_prod.txt). That was
        # a hand-measured whitelist of one value on one machine; the gate now
        # rediscovers it, so the candidate set is simply every extent that
        # divides and fits and everything that reassociates is thrown out on
        # bytes.
        _mhc_tile_opt_retired()

        def _tune_sinkhorn_bwd():
            def make_outputs():
                return (paddle.empty(shape=[N_batch, hc, hc],
                                     dtype=grad_output.dtype),)

            def do_launch(cfg, outs):
                ts = cfg["TILE_SIZE"]
                ct.launch(
                    _get_cuda_stream(),
                    (math.ceil(N_batch / ts), 1, 1),
                    _ct_sinkhorn_bwd_kernel,
                    (
                        grad_output.reshape([N_batch, hc, hc]),
                        M_init.reshape([N_batch, hc, hc]),
                        outs[0], ws_M, ws_rs, ws_cs, eps, hc,
                        num_iterations, ts,
                    ),
                )

            return _mhc_autotune_call(
                name="mhc.sinkhorn_bwd",
                factory={"TILE_SIZE": SHIPPED_TS},
                candidates=[
                    {"TILE_SIZE": v} for v in _mhc_size_candidates(
                        SHIPPED_TS, N_batch, hc * hc * 4 * 4)
                ],
                launch_shape=(N_batch, hc, num_iterations),
                dtypes=(grad_output.dtype, M_init.dtype),
                kernels=(_ct_sinkhorn_bwd_kernel,),
                make_outputs=make_outputs,
                launch=do_launch,
            )

        cfg = _mhc_tune("mhc.sinkhorn_bwd",
                        (N_batch, hc, num_iterations, str(grad_output.dtype)),
                        _tune_sinkhorn_bwd)
        TILE_SIZE = cfg["TILE_SIZE"] if cfg else SHIPPED_TS
        grad_input = paddle.empty(
            shape=[N_batch, hc, hc], dtype=grad_output.dtype
        )
        ct.launch(
            _get_cuda_stream(),
            (math.ceil(N_batch / TILE_SIZE), 1, 1),
            _ct_sinkhorn_bwd_kernel,
            (
                grad_output.reshape([N_batch, hc, hc]),
                M_init.reshape([N_batch, hc, hc]),
                grad_input,
                ws_M,
                ws_rs,
                ws_cs,
                eps,
                hc,
                num_iterations,
                TILE_SIZE,
            ),
        )
        return grad_input.reshape(original_shape)

    # -- compute_h kernels ---------------------------------------------------
    #
    # Column layout of proj / bias / g_proj, all of width P = N*N + 2*N:
    #
    #     [ pre : N ][ post : N ][ res : N*N ]
    #
    # Everything is addressed in tiles of width N. ``ct.load``/``ct.store``
    # index in whole tiles: tile 0 is the pre segment, tile 1 the post, and
    # tiles 2 .. N+1 the res segment (N*N is exactly N of them). Slicing the res
    # segment out in one piece is not expressible -- a width-N*N tile can only
    # start at a multiple of N*N, and it starts at 2*N -- which is why the res
    # work below is an unrolled loop over N sub-tiles rather than one operation.
    #
    # On dtypes: alpha and bias arrive in params_dtype and the global default
    # dtype respectively, i.e. bf16 in bf16 training, while proj and r are fp32
    # under high_precision_mhc. Nothing below widens them, and nothing needs to:
    # every expression pairs them with ``r * proj``, which is fp32 already, so
    # the promotion lifts the whole expression -- the same thing Paddle's
    # promotion does in ``native_compute_h``.
    #
    # The rule, established by mutating every widening in this file and watching
    # which removals the tests catch: an expression only degrades to the narrow
    # dtype when *all* of its narrow operands stay narrow. The one place that
    # happens is ``_ct_proj_rms_fwd_kernel``'s ``a_tile * a_tile``, which is
    # widened explicitly.

    @ct.kernel
    def _ct_compute_h_fwd_kernel(
        proj,
        r,
        a_pre,
        a_post,
        a_res,
        bias,
        h_pre,
        h_post,
        h_res,
        N: ConstInt,
        TILE_M: ConstInt,
        eps: float,
    ):
        """h = r * proj * alpha + bias, then the two sigmoid heads.

        Replaces ``expand x3 -> concat -> mul -> mul -> add -> sigmoid x2`` with
        one launch. The alpha vector the reference builds by concatenating three
        broadcast scalars never materializes: which scalar a column belongs to
        is a compile-time fact here, so each segment gets its own scalar.

        The sigmoid is ``1 / (1 + exp(-u))`` because cuTile has no sigmoid. That
        form is bitwise equal to ``paddle.sigmoid`` on fp32 via ``ct.exp`` --
        but not via ``ct.exp2``, which drifts ~1e-6 relative, so do not rewrite
        into the exp2 form the softmax above uses.
        """
        pid = ct.bid(0)
        r_tile = ct.load(
            r, index=(pid, 0), shape=(TILE_M, 1), padding_mode=PAD_ZERO
        )
        ap = ct.reshape(ct.load(a_pre, index=(0,), shape=(1,)), (1, 1))
        aq = ct.reshape(ct.load(a_post, index=(0,), shape=(1,)), (1, 1))
        ar = ct.reshape(ct.load(a_res, index=(0,), shape=(1,)), (1, 1))

        u_pre = r_tile * ct.load(
            proj, index=(pid, 0), shape=(TILE_M, N), padding_mode=PAD_ZERO
        ) * ap + ct.reshape(
            ct.load(bias, index=(0,), shape=(N,), padding_mode=PAD_ZERO), (1, N)
        )
        u_post = r_tile * ct.load(
            proj, index=(pid, 1), shape=(TILE_M, N), padding_mode=PAD_ZERO
        ) * aq + ct.reshape(
            ct.load(bias, index=(1,), shape=(N,), padding_mode=PAD_ZERO), (1, N)
        )

        s_pre = 1.0 / (1.0 + ct.exp(-u_pre))
        s_post = 1.0 / (1.0 + ct.exp(-u_post))
        ct.store(h_pre, index=(pid, 0), tile=(s_pre + eps).astype(h_pre.dtype))
        ct.store(
            h_post, index=(pid, 0), tile=(s_post * 2.0).astype(h_post.dtype)
        )

        for k in range(N):
            u_k = r_tile * ct.load(
                proj,
                index=(pid, 2 + k),
                shape=(TILE_M, N),
                padding_mode=PAD_ZERO,
            ) * ar + ct.reshape(
                ct.load(
                    bias, index=(2 + k,), shape=(N,), padding_mode=PAD_ZERO
                ),
                (1, N),
            )
            ct.store(h_res, index=(pid, k), tile=u_k.astype(h_res.dtype))

    @ct.kernel
    def _ct_compute_h_bwd_kernel(
        g_h_pre,
        g_h_post,
        g_h_res,
        proj,
        r,
        h_pre,
        h_post,
        a_pre,
        a_post,
        a_res,
        g_proj,
        g_r,
        ga_part,
        gb_part,
        N: ConstInt,
        TILE_M: ConstInt,
        eps: float,
    ):
        """Backward of ``_ct_compute_h_fwd_kernel``.

        ``g_proj`` and ``g_r`` are per-token and final here. The alpha and bias
        gradients reduce over every token, which one block cannot finish: each
        block writes its own partial into ``ga_part``/``gb_part`` and the
        launcher sums those. That costs ``num_blocks * (3 + P)`` elements and is
        deterministic, unlike an atomic accumulation.

        The sigmoid derivatives come from the forward outputs
        (``s_pre = h_pre - eps``, ``s_post = h_post / 2``) rather than from a
        recomputed ``u``, which would need the bias and another pass over proj.

        Multiplication order follows the chain rule the reference's autograd
        actually walks -- ``t = r*proj``, ``u = t*alpha``, ``h = u+bias`` -- so
        ``d(proj) = (du*alpha)*r`` and ``d(alpha) = du*(r*proj)``, not the
        left-to-right grouping. Float multiplication is commutative but not
        associative, and the wrong grouping costs a ULP for no reason.
        """
        pid = ct.bid(0)
        r_tile = ct.load(
            r, index=(pid, 0), shape=(TILE_M, 1), padding_mode=PAD_ZERO
        )
        ap = ct.reshape(ct.load(a_pre, index=(0,), shape=(1,)), (1, 1))
        aq = ct.reshape(ct.load(a_post, index=(0,), shape=(1,)), (1, 1))
        ar = ct.reshape(ct.load(a_res, index=(0,), shape=(1,)), (1, 1))

        # du = dL/d(r * proj * alpha + bias), one segment at a time
        s_pre = (
            ct.load(
                h_pre, index=(pid, 0), shape=(TILE_M, N), padding_mode=PAD_ZERO
            )
            - eps
        )
        du_pre = (
            ct.load(
                g_h_pre,
                index=(pid, 0),
                shape=(TILE_M, N),
                padding_mode=PAD_ZERO,
            )
            * s_pre
            * (1.0 - s_pre)
        )
        s_post = (
            ct.load(
                h_post, index=(pid, 0), shape=(TILE_M, N), padding_mode=PAD_ZERO
            )
            / 2.0
        )
        du_post = (
            ct.load(
                g_h_post,
                index=(pid, 0),
                shape=(TILE_M, N),
                padding_mode=PAD_ZERO,
            )
            * 2.0
            * s_post
            * (1.0 - s_post)
        )

        p_pre = ct.load(
            proj, index=(pid, 0), shape=(TILE_M, N), padding_mode=PAD_ZERO
        )
        p_post = ct.load(
            proj, index=(pid, 1), shape=(TILE_M, N), padding_mode=PAD_ZERO
        )

        # d(proj) = du * r * alpha ; d(bias) = du summed over tokens
        ct.store(
            g_proj,
            index=(pid, 0),
            tile=(du_pre * ap * r_tile).astype(g_proj.dtype),
        )
        ct.store(
            g_proj,
            index=(pid, 1),
            tile=(du_post * aq * r_tile).astype(g_proj.dtype),
        )
        ct.store(
            gb_part,
            index=(pid, 0),
            tile=ct.sum(du_pre, axis=0, keepdims=True).astype(gb_part.dtype),
        )
        ct.store(
            gb_part,
            index=(pid, 1),
            tile=ct.sum(du_post, axis=0, keepdims=True).astype(gb_part.dtype),
        )

        # d(r) sums over every column; d(alpha) over every column and token
        dr = ct.sum(du_pre * ap * p_pre, axis=1, keepdims=True) + ct.sum(
            du_post * aq * p_post, axis=1, keepdims=True
        )
        da_pre = ct.sum(
            ct.sum(du_pre * (r_tile * p_pre), axis=1, keepdims=True),
            axis=0,
            keepdims=True,
        )
        da_post = ct.sum(
            ct.sum(du_post * (r_tile * p_post), axis=1, keepdims=True),
            axis=0,
            keepdims=True,
        )
        da_res = ct.full((1, 1), 0.0, dtype=ct.float32)
        for k in range(N):
            du_k = ct.load(
                g_h_res,
                index=(pid, k),
                shape=(TILE_M, N),
                padding_mode=PAD_ZERO,
            )
            p_k = ct.load(
                proj,
                index=(pid, 2 + k),
                shape=(TILE_M, N),
                padding_mode=PAD_ZERO,
            )
            ct.store(
                g_proj,
                index=(pid, 2 + k),
                tile=(du_k * ar * r_tile).astype(g_proj.dtype),
            )
            ct.store(
                gb_part,
                index=(pid, 2 + k),
                tile=ct.sum(du_k, axis=0, keepdims=True).astype(gb_part.dtype),
            )
            dr = dr + ct.sum(du_k * ar * p_k, axis=1, keepdims=True)
            da_res = da_res + ct.sum(
                ct.sum(du_k * (r_tile * p_k), axis=1, keepdims=True),
                axis=0,
                keepdims=True,
            )
        ct.store(g_r, index=(pid, 0), tile=dr.astype(g_r.dtype))
        ct.store(ga_part, index=(pid, 0), tile=da_pre.astype(ga_part.dtype))
        ct.store(ga_part, index=(pid, 1), tile=da_post.astype(ga_part.dtype))
        ct.store(ga_part, index=(pid, 2), tile=da_res.astype(ga_part.dtype))

    # -- H_aggregate kernels -------------------------------------------------

    @ct.kernel
    def _ct_h_agg_fwd_kernel(
        x,
        h_pre,
        out,
        N: ConstInt,
        TILE_M: ConstInt,
        TILE_C: ConstInt,
        UPCAST_INPUTS: ConstBool,
    ):
        """n-stream -> 1-stream weighted aggregation.

        ``UPCAST_INPUTS`` widens ``x`` in-register when it arrives narrower than
        ``h_pre``; without it the product and its reduction over ``N`` would run
        in the narrow dtype.
        """
        pid = ct.bid(0)
        num_tiles = ct.num_tiles(x, axis=2, shape=(TILE_M, N, TILE_C))
        h_tile = ct.load(
            h_pre, index=(pid, 0), shape=(TILE_M, N), padding_mode=PAD_ZERO
        )
        h_tile = ct.expand_dims(h_tile, axis=2)
        for j in range(num_tiles):
            x_tile = ct.load(
                x,
                index=(pid, 0, j),
                shape=(TILE_M, N, TILE_C),
                padding_mode=PAD_ZERO,
            )
            if UPCAST_INPUTS:
                x_tile = x_tile.astype(ct.float32)
            acc = ct.sum(x_tile * h_tile, axis=1).astype(ct.float32)
            ct.store(out, index=(pid, j), tile=acc.astype(out.dtype))

    @ct.kernel
    def _ct_h_agg_bwd_kernel(
        go,
        x,
        h_pre,
        gx,
        gh,
        N: ConstInt,
        TILE_M: ConstInt,
        TILE_C: ConstInt,
        UPCAST_INPUTS: ConstBool,
    ):
        """Backward of ``_ct_h_agg_fwd_kernel``.

        ``UPCAST_INPUTS`` widens ``x`` before the ``gh`` reduction. ``gx`` does
        not involve ``x`` at all, so it is unaffected.
        """
        pid = ct.bid(0)
        num_c_tiles = ct.num_tiles(go, axis=1, shape=(TILE_M, TILE_C))
        h_tile = ct.load(
            h_pre, index=(pid, 0), shape=(TILE_M, N), padding_mode=PAD_ZERO
        )
        h_expanded = ct.expand_dims(h_tile, axis=2)
        gh_acc = ct.full((TILE_M, N), 0, dtype=ct.float32)
        for ct_idx in range(num_c_tiles):
            go_tile = ct.load(
                go,
                index=(pid, ct_idx),
                shape=(TILE_M, TILE_C),
                padding_mode=PAD_ZERO,
            )
            if UPCAST_INPUTS:
                # ``go`` is the gradient of ``aggregated``, which the fusion
                # narrows too, so it needs widening as well -- otherwise both
                # products below would fall back to cuTile's implicit promotion
                # instead of the fp32 arithmetic the un-fused path did.
                go_tile = go_tile.astype(ct.float32)
            go_expanded = ct.expand_dims(go_tile, axis=1)
            x_tile = ct.load(
                x,
                index=(pid, 0, ct_idx),
                shape=(TILE_M, N, TILE_C),
                padding_mode=PAD_ZERO,
            )
            if UPCAST_INPUTS:
                x_tile = x_tile.astype(ct.float32)
            gx_tile = go_expanded * h_expanded
            ct.store(gx, index=(pid, 0, ct_idx), tile=gx_tile.astype(gx.dtype))
            gh_acc += ct.sum(go_expanded * x_tile, axis=2)
        ct.store(gh, index=(pid, 0), tile=gh_acc.astype(gh.dtype))

    def _cutile_compute_h_fwd(
        proj: Tensor,
        r: Tensor,
        alpha_pre: Tensor,
        alpha_post: Tensor,
        alpha_res: Tensor,
        bias: Tensor,
        n: int,
        eps: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        leading = list(proj.shape[:-1])
        P = proj.shape[-1]
        M = math.prod(leading)
        TILE_M = 64
        dt = proj.dtype
        h_pre = paddle.empty(shape=[M, n], dtype=dt)
        h_post = paddle.empty(shape=[M, n], dtype=dt)
        h_res = paddle.empty(shape=[M, n * n], dtype=dt)
        ct.launch(
            _get_cuda_stream(),
            (math.ceil(M / TILE_M),),
            _ct_compute_h_fwd_kernel,
            (
                proj.detach().reshape([M, P]),
                r.detach().reshape([M, 1]),
                alpha_pre.detach(),
                alpha_post.detach(),
                alpha_res.detach(),
                bias.detach(),
                h_pre,
                h_post,
                h_res,
                n,
                TILE_M,
                eps,
            ),
        )
        return (
            h_pre.reshape([*leading, n]),
            h_post.reshape([*leading, n]),
            h_res.reshape([*leading, n * n]),
        )

    def _cutile_compute_h_bwd(
        g_h_pre: Tensor,
        g_h_post: Tensor,
        g_h_res: Tensor,
        proj: Tensor,
        r: Tensor,
        h_pre: Tensor,
        h_post: Tensor,
        alpha_pre: Tensor,
        alpha_post: Tensor,
        alpha_res: Tensor,
        bias: Tensor,
        n: int,
        eps: float,
    ):
        leading = list(proj.shape[:-1])
        P = proj.shape[-1]
        M = math.prod(leading)
        TILE_M = 64
        blocks = math.ceil(M / TILE_M)
        g_proj = paddle.empty(shape=[M, P], dtype=proj.dtype)
        g_r = paddle.empty(shape=[M, 1], dtype=r.dtype)
        # One partial row per block; the cross-token sums finish below in fp32.
        ga_part = paddle.empty(shape=[blocks, 3], dtype="float32")
        gb_part = paddle.empty(shape=[blocks, P], dtype="float32")
        ct.launch(
            _get_cuda_stream(),
            (blocks,),
            _ct_compute_h_bwd_kernel,
            (
                g_h_pre.detach().reshape([M, n]),
                g_h_post.detach().reshape([M, n]),
                g_h_res.detach().reshape([M, n * n]),
                proj.detach().reshape([M, P]),
                r.detach().reshape([M, 1]),
                h_pre.detach().reshape([M, n]),
                h_post.detach().reshape([M, n]),
                alpha_pre.detach(),
                alpha_post.detach(),
                alpha_res.detach(),
                g_proj,
                g_r,
                ga_part,
                gb_part,
                n,
                TILE_M,
                eps,
            ),
        )
        g_alpha = ga_part.sum(axis=0)
        return (
            g_proj.reshape([*leading, P]),
            g_r.reshape([*leading, 1]),
            g_alpha[0:1].astype(alpha_pre.dtype),
            g_alpha[1:2].astype(alpha_post.dtype),
            g_alpha[2:3].astype(alpha_res.dtype),
            gb_part.sum(axis=0).astype(bias.dtype),
        )

    def _cutile_h_aggregate_fwd(
        x: Tensor, h_pre: Tensor, fuse_cast: bool = False
    ) -> Tensor:
        s, b, n, C = x.shape
        sb = s * b
        SHIPPED = (math.gcd(sb, 4), math.gcd(C, 1024))
        # opt/mhc_0908: TILE_SIZE is free here (the reduction is over ``n``,
        # which always sits whole inside the tile), and so is TILE_C except for
        # 128, which reassociates the ``ct.sum`` over ``n``: measured exact for
        # TILE_C in {4096, 2048, 1024, 512, 256} and for TILE_SIZE in
        # {1, 2, 4, 8, 16, 32}, and DIFF (2-4 bf16 ULP on 3e-5 of the elements)
        # at TILE_C=128 -- logs/bitexact_prod.txt. That whitelist is no longer
        # what protects the numerics; the gate is, and it re-derives the same
        # exclusion on whatever machine it runs on.
        _mhc_tile_opt_retired()
        # ``out`` follows x either way: under fuse_cast x is already the dtype
        # the downstream layernorm wants, so no override is needed.

        def _tune_h_agg_fwd():
            def make_outputs():
                return (paddle.empty(shape=[sb, C], dtype=x.dtype),)

            def do_launch(cfg, outs):
                tm, tc = cfg["TILE_SIZE"], cfg["TILE_C"]
                ct.launch(
                    _get_cuda_stream(),
                    (math.ceil(sb / tm),),
                    _ct_h_agg_fwd_kernel,
                    (x.reshape([sb, n, C]), h_pre.reshape([sb, n]), outs[0],
                     n, tm, tc, fuse_cast),
                )

            pairs = _mhc_pair_candidates(
                SHIPPED, sb, C, n * 4, x.element_size())
            return _mhc_autotune_call(
                name="mhc.h_agg_fwd",
                factory={"TILE_SIZE": SHIPPED[0], "TILE_C": SHIPPED[1]},
                candidates=[{"TILE_SIZE": r, "TILE_C": c} for r, c in pairs],
                launch_shape=(sb, n, C),
                dtypes=(x.dtype, h_pre.dtype, bool(fuse_cast)),
                kernels=(_ct_h_agg_fwd_kernel,),
                make_outputs=make_outputs,
                launch=do_launch,
            )

        cfg = _mhc_tune(
            "mhc.h_agg_fwd",
            (sb, n, C, str(x.dtype), bool(fuse_cast)), _tune_h_agg_fwd)
        TILE_SIZE = cfg["TILE_SIZE"] if cfg else SHIPPED[0]
        TILE_C = cfg["TILE_C"] if cfg else SHIPPED[1]
        out = paddle.empty(shape=[sb, C], dtype=x.dtype)
        ct.launch(
            _get_cuda_stream(),
            (math.ceil(sb / TILE_SIZE),),
            _ct_h_agg_fwd_kernel,
            (
                x.reshape([sb, n, C]),
                h_pre.reshape([sb, n]),
                out,
                n,
                TILE_SIZE,
                TILE_C,
                fuse_cast,
            ),
        )
        return out.reshape([s, b, C])

    def _cutile_h_aggregate_bwd(
        grad_output: Tensor,
        x: Tensor,
        h_pre: Tensor,
        fuse_cast: bool = False,
    ) -> tuple[Tensor, Tensor]:
        s, b, n, C = x.shape
        sb = s * b
        # opt/mhc_0908: ``gh`` reduces over C with ``ct.sum``, so TILE_C is
        # pinned -- every smaller value reassociates it. TILE_M is exact only
        # for {1, 2, 4}; 8 and 16 change the tile shape enough to reassociate
        # the same sum (logs/bitexact_prod.txt). TILE_C is deliberately not a
        # candidate: it is not that the gate would let it through, it is that
        # there is nothing to gain by asking.
        TILE_C = min(C, 4096) if C <= 4096 else math.gcd(C, 1024)
        SHIPPED_TM = math.gcd(sb, 2)
        _mhc_tile_opt_retired()

        def _tune_h_agg_bwd():
            def make_outputs():
                return (
                    paddle.empty(shape=[sb, n, C], dtype=x.dtype),
                    paddle.empty(shape=[sb, n],
                                 dtype=h_pre.dtype if fuse_cast else x.dtype),
                )

            def do_launch(cfg, outs):
                tm = cfg["TILE_M"]
                ct.launch(
                    _get_cuda_stream(),
                    (math.ceil(sb / tm),),
                    _ct_h_agg_bwd_kernel,
                    (grad_output.reshape([sb, C]), x.reshape([sb, n, C]),
                     h_pre.reshape([sb, n]), outs[0], outs[1], n, tm, TILE_C,
                     fuse_cast),
                )

            return _mhc_autotune_call(
                name="mhc.h_agg_bwd",
                factory={"TILE_M": SHIPPED_TM},
                candidates=[
                    {"TILE_M": v} for v in _mhc_size_candidates(
                        SHIPPED_TM, sb, n * TILE_C * 2)
                ],
                launch_shape=(sb, n, C, TILE_C),
                dtypes=(grad_output.dtype, x.dtype, h_pre.dtype,
                        bool(fuse_cast)),
                kernels=(_ct_h_agg_bwd_kernel,),
                make_outputs=make_outputs,
                launch=do_launch,
            )

        cfg = _mhc_tune(
            "mhc.h_agg_bwd",
            (sb, n, C, str(x.dtype), bool(fuse_cast)), _tune_h_agg_bwd)
        TILE_M = cfg["TILE_M"] if cfg else SHIPPED_TM
        gx = paddle.empty(shape=[sb, n, C], dtype=x.dtype)

        # gh is the gradient of h_pre, which stays fp32 under fuse_cast even
        # though x does not, so it cannot follow x's dtype there.
        gh = paddle.empty(
            shape=[sb, n], dtype=h_pre.dtype if fuse_cast else x.dtype
        )
        ct.launch(
            _get_cuda_stream(),
            (math.ceil(sb / TILE_M),),
            _ct_h_agg_bwd_kernel,
            (
                grad_output.reshape([sb, C]),
                x.reshape([sb, n, C]),
                h_pre.reshape([sb, n]),
                gx,
                gh,
                n,
                TILE_M,
                TILE_C,
                fuse_cast,
            ),
        )
        return gx.reshape([s, b, n, C]), gh.reshape([s, b, n])

    # -- H_post BDA kernels --------------------------------------------------

    @ct.kernel
    def _ct_hpb_fwd_kernel(
        hr,
        orig,
        hp,
        x,
        out,
        N: ConstInt,
        TILE_C: ConstInt,
        TILE_SIZE: ConstInt,
        UPCAST_INPUTS: ConstBool,
    ):
        """Fused H_res @ residual + H_post * x.

        ``UPCAST_INPUTS`` is a compile-time constant: set it when ``orig``/``x``
        arrive narrower than the fp32 the arithmetic below runs in, so they
        get widened in-register instead of the caller materializing fp32
        copies. When false the guard folds away and this is the original
        kernel, including its bf16-in/bf16-out behaviour -- ``TestConstGuard``
        pins that down.
        """
        pid = ct.bid(0)
        num_c_tiles = ct.num_tiles(x, axis=1, shape=(TILE_SIZE, TILE_C))
        hp_tile = ct.load(
            hp, index=(pid, 0), shape=(TILE_SIZE, N), padding_mode=PAD_ZERO
        )
        hp_exp = ct.expand_dims(hp_tile, axis=2)  # (TILE_SIZE, N, 1)
        hr_tile = ct.load(
            hr,
            index=(pid, 0, 0),
            shape=(TILE_SIZE, N, N),
            padding_mode=PAD_ZERO,
        )
        for ct_idx in range(num_c_tiles):
            orig_tile = ct.load(
                orig,
                index=(pid, 0, ct_idx),
                shape=(TILE_SIZE, N, TILE_C),
                padding_mode=PAD_ZERO,
            )
            x_tile = ct.load(
                x,
                index=(pid, ct_idx),
                shape=(TILE_SIZE, TILE_C),
                padding_mode=PAD_ZERO,
            )
            if UPCAST_INPUTS:
                orig_tile = orig_tile.astype(ct.float32)
                x_tile = x_tile.astype(ct.float32)
            x_exp = ct.expand_dims(x_tile, axis=1)  # (TILE_SIZE, 1, TILE_C)
            out_tile = hp_exp * x_exp  # (TILE_SIZE, N, TILE_C)
            for j in range(N):
                hr_row = ct.extract(hr_tile, (0, j, 0), shape=(TILE_SIZE, 1, N))
                hr_col = ct.reshape(hr_row, (TILE_SIZE, N, 1))
                orig_row = ct.extract(
                    orig_tile, (0, j, 0), shape=(TILE_SIZE, 1, TILE_C)
                )
                out_tile = out_tile + hr_col * orig_row
            ct.store(
                out, index=(pid, 0, ct_idx), tile=out_tile.astype(out.dtype)
            )

    @ct.kernel
    def _ct_hpb_fwd_bias_kernel(
        hr,
        orig,
        hp,
        x,
        bias,
        out,
        N: ConstInt,
        TILE_C: ConstInt,
        TILE_SIZE: ConstInt,
    ):
        """As ``_ct_hpb_fwd_kernel``, plus ``+ bias`` on the layer output.

        No ``UPCAST_INPUTS``: the fusion is vetoed whenever a bias is present,
        see ``_cutile_h_post_bda_bwd`` for why, so this kernel only ever sees
        pre-widened operands.
        """
        pid = ct.bid(0)
        num_c_tiles = ct.num_tiles(x, axis=1, shape=(TILE_SIZE, TILE_C))
        hp_tile = ct.load(
            hp, index=(pid, 0), shape=(TILE_SIZE, N), padding_mode=PAD_ZERO
        )
        hp_exp = ct.expand_dims(hp_tile, axis=2)  # (TILE_SIZE, N, 1)
        hr_tile = ct.load(
            hr,
            index=(pid, 0, 0),
            shape=(TILE_SIZE, N, N),
            padding_mode=PAD_ZERO,
        )
        for ct_idx in range(num_c_tiles):
            orig_tile = ct.load(
                orig,
                index=(pid, 0, ct_idx),
                shape=(TILE_SIZE, N, TILE_C),
                padding_mode=PAD_ZERO,
            )
            x_tile = ct.load(
                x,
                index=(pid, ct_idx),
                shape=(TILE_SIZE, TILE_C),
                padding_mode=PAD_ZERO,
            )
            bias_tile = ct.load(
                bias, index=(ct_idx,), shape=(TILE_C,), padding_mode=PAD_ZERO
            )
            xb_exp = ct.expand_dims(
                x_tile + bias_tile, axis=1
            )  # (TILE_SIZE, 1, TILE_C)
            out_tile = hp_exp * xb_exp  # (TILE_SIZE, N, TILE_C)
            for j in range(N):
                hr_row = ct.extract(hr_tile, (0, j, 0), shape=(TILE_SIZE, 1, N))
                hr_col = ct.reshape(hr_row, (TILE_SIZE, N, 1))
                orig_row = ct.extract(
                    orig_tile, (0, j, 0), shape=(TILE_SIZE, 1, TILE_C)
                )
                out_tile = out_tile + hr_col * orig_row
            ct.store(
                out, index=(pid, 0, ct_idx), tile=out_tile.astype(out.dtype)
            )

    @ct.kernel
    def _ct_hpb_bwd_kernel(
        go,
        hr,
        orig,
        hp,
        x,
        g_hr,
        g_orig,
        g_hp,
        g_x,
        N: ConstInt,
        TILE_C: ConstInt,
        TILE_SIZE: ConstInt,
        UPCAST_INPUTS: ConstBool,
    ):
        """Backward of ``_ct_hpb_fwd_kernel``.

        ``UPCAST_INPUTS`` widens ``go``/``x``/``orig`` in-register, same
        contract as the forward. Note this is where the fused path stops being
        bit-identical to the un-fused one: narrowing those three halves the tile
        footprint, so cuTile schedules the two ``ct.sum`` reductions differently
        and the fp32 additions group differently. Only ``g_hp``/``g_hr`` are
        affected -- they are the ones reduced over ``TILE_C``. The summands are
        unchanged, so this is a reassociation, not a precision loss.
        """
        pid = ct.bid(0)
        num_c_tiles = ct.cdiv(go.shape[2], TILE_C)
        hp_tile = ct.load(hp, index=(pid, 0), shape=(TILE_SIZE, N))
        hp_2d = ct.reshape(hp_tile, (1, N))
        hr_tile = ct.load(
            hr,
            index=(pid, 0, 0),
            shape=(TILE_SIZE, N, N),
            padding_mode=PAD_ZERO,
        )
        hr_2d = ct.reshape(hr_tile, (N, N))
        acc_g_hp_2d = ct.full((N, 1), 0, dtype=ct.float32)
        acc_g_hr_2d = ct.full((N, N), 0, dtype=ct.float32)
        for ct_idx in range(num_c_tiles):
            x_tile = ct.load(
                x,
                index=(pid, ct_idx),
                shape=(TILE_SIZE, TILE_C),
                padding_mode=PAD_ZERO,
            )
            if UPCAST_INPUTS:
                x_tile = x_tile.astype(ct.float32)
            x_2d = ct.reshape(x_tile, (1, TILE_C))
            go_tile = ct.load(
                go,
                index=(pid, 0, ct_idx),
                shape=(TILE_SIZE, N, TILE_C),
                padding_mode=PAD_ZERO,
            )
            if UPCAST_INPUTS:
                go_tile = go_tile.astype(ct.float32)
            go_2d = ct.reshape(go_tile, (N, TILE_C))
            orig_tile = ct.load(
                orig,
                index=(pid, 0, ct_idx),
                shape=(TILE_SIZE, N, TILE_C),
                padding_mode=PAD_ZERO,
            )
            if UPCAST_INPUTS:
                orig_tile = orig_tile.astype(ct.float32)
            orig_2d = ct.reshape(orig_tile, (N, TILE_C))
            g_x_2d = ct.full((1, TILE_C), 0, dtype=hp.dtype)
            g_orig_2d = ct.full((N, TILE_C), 0, dtype=hp.dtype)
            for j in range(N):
                g_x_2d += ct.extract(
                    hp_2d, (0, j), shape=(1, 1)
                ).item() * ct.extract(go_2d, (j, 0), shape=(1, TILE_C))
                g_orig_2d += ct.extract(
                    hr_2d, (0, j), shape=(N, 1)
                ) * ct.extract(go_2d, (j, 0), shape=(1, TILE_C))
            acc_g_hp_2d += ct.sum(go_2d * x_2d, axis=1, keepdims=True)
            acc_g_hr_2d += ct.sum(
                ct.expand_dims(go_2d, axis=0) * ct.expand_dims(orig_2d, axis=1),
                axis=2,
            )
            ct.store(
                g_x,
                index=(pid, ct_idx),
                tile=ct.reshape(g_x_2d, (TILE_SIZE, TILE_C)).astype(g_x.dtype),
            )
            ct.store(
                g_orig,
                index=(pid, 0, ct_idx),
                tile=ct.reshape(g_orig_2d, (TILE_SIZE, N, TILE_C)).astype(
                    g_orig.dtype
                ),
            )
        ct.store(
            g_hp,
            index=(pid, 0),
            tile=ct.reshape(acc_g_hp_2d, (TILE_SIZE, N)).astype(g_hp.dtype),
        )
        ct.store(
            g_hr,
            index=(pid, 0, 0),
            tile=ct.reshape(acc_g_hr_2d, (TILE_SIZE, N, N)).astype(g_hr.dtype),
        )

    @ct.kernel
    def _ct_hpb_bwd_bias_kernel(
        go,
        hr,
        orig,
        hp,
        x,
        bias,
        g_hr,
        g_orig,
        g_hp,
        g_x,
        N: ConstInt,
        TILE_C: ConstInt,
        TILE_SIZE: ConstInt,
    ):
        """As ``_ct_hpb_bwd_kernel``, plus the ``bias`` gradient path.

        No ``UPCAST_INPUTS`` -- see ``_cutile_h_post_bda_bwd``.
        """
        pid = ct.bid(0)
        num_c_tiles = ct.cdiv(go.shape[2], TILE_C)
        hp_tile = ct.load(hp, index=(pid, 0), shape=(TILE_SIZE, N))
        hp_2d = ct.reshape(hp_tile, (1, N))
        hr_tile = ct.load(
            hr,
            index=(pid, 0, 0),
            shape=(TILE_SIZE, N, N),
            padding_mode=PAD_ZERO,
        )
        hr_2d = ct.reshape(hr_tile, (N, N))
        acc_g_hp_2d = ct.full((N, 1), 0, dtype=ct.float32)
        acc_g_hr_2d = ct.full((N, N), 0, dtype=ct.float32)
        for ct_idx in range(num_c_tiles):
            x_tile = ct.load(
                x,
                index=(pid, ct_idx),
                shape=(TILE_SIZE, TILE_C),
                padding_mode=PAD_ZERO,
            )
            bias_tile = ct.load(
                bias, index=(ct_idx,), shape=(TILE_C,), padding_mode=PAD_ZERO
            )
            xb_2d = ct.reshape(x_tile, (1, TILE_C)) + ct.reshape(
                bias_tile, (1, TILE_C)
            )
            go_tile = ct.load(
                go,
                index=(pid, 0, ct_idx),
                shape=(TILE_SIZE, N, TILE_C),
                padding_mode=PAD_ZERO,
            )
            go_2d = ct.reshape(go_tile, (N, TILE_C))
            orig_tile = ct.load(
                orig,
                index=(pid, 0, ct_idx),
                shape=(TILE_SIZE, N, TILE_C),
                padding_mode=PAD_ZERO,
            )
            orig_2d = ct.reshape(orig_tile, (N, TILE_C))
            g_x_2d = ct.full((1, TILE_C), 0, dtype=hp.dtype)
            g_orig_2d = ct.full((N, TILE_C), 0, dtype=hp.dtype)
            for j in range(N):
                g_x_2d += ct.extract(
                    hp_2d, (0, j), shape=(1, 1)
                ).item() * ct.extract(go_2d, (j, 0), shape=(1, TILE_C))
                g_orig_2d += ct.extract(
                    hr_2d, (0, j), shape=(N, 1)
                ) * ct.extract(go_2d, (j, 0), shape=(1, TILE_C))
            acc_g_hp_2d += ct.sum(go_2d * xb_2d, axis=1, keepdims=True)
            acc_g_hr_2d += ct.sum(
                ct.expand_dims(go_2d, axis=0) * ct.expand_dims(orig_2d, axis=1),
                axis=2,
            )
            ct.store(
                g_x,
                index=(pid, ct_idx),
                tile=ct.reshape(g_x_2d, (TILE_SIZE, TILE_C)).astype(g_x.dtype),
            )
            ct.store(
                g_orig,
                index=(pid, 0, ct_idx),
                tile=ct.reshape(g_orig_2d, (TILE_SIZE, N, TILE_C)).astype(
                    g_orig.dtype
                ),
            )
        ct.store(
            g_hp,
            index=(pid, 0),
            tile=ct.reshape(acc_g_hp_2d, (TILE_SIZE, N)).astype(g_hp.dtype),
        )
        ct.store(
            g_hr,
            index=(pid, 0, 0),
            tile=ct.reshape(acc_g_hr_2d, (TILE_SIZE, N, N)).astype(g_hr.dtype),
        )

    def _cutile_h_post_bda_fwd(
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        x: Tensor,
        bias: Tensor | None,
        fuse_cast: bool = False,
    ) -> Tensor:
        s, b, n, C = original_residual.shape
        sb = s * b
        # Shipped: the largest power-of-2 tile that divides C, up to 4096
        # (cuTile requires TILE_C to be a power of 2), one row at a time.
        SHIPPED = (
            math.gcd(sb, 1),
            math.gcd(C, 4096)
            if C % 4096 == 0
            else math.gcd(C, 2048)
            if C % 2048 == 0
            else math.gcd(C, 1024),
        )
        # opt/mhc_0908: this kernel holds two (TILE_SIZE, n, TILE_C) fp32 tiles
        # at once -- the accumulator and the in-register widening of ``orig``.
        # At the shipped (1, 4096) that is 2 x 16384 fp32 over 128 threads =
        # 254 registers/thread, i.e. 2 blocks and 8 warps per SM (12.5%), which
        # is why the kernel only reaches ~1.7 TB/s while moving exactly its
        # compulsory bytes. Smaller tiles lift that ceiling, but NOT
        # MONOTONICALLY, which is the reason this is measured rather than
        # derived. Measured on an exclusively-owned card, median of 100
        # launches at one production shape (16384 x 4 x 4096):
        #
        #   (1, 4096) shipped  740.7 us  1633 GB/s  23.5% of peak  254 reg
        #   (2,  128)          383.7 us  3152 GB/s  45.4%          109 reg
        #   (4,   64)          394.9 us  3062 GB/s  44.1%          109 reg
        #   (1, 1024)          431.6 us  2802 GB/s  40.4%          255 reg
        #   (1,  256)          448.0 us  2699 GB/s  38.9%          114 reg
        #   (1,  128)          883.1 us  1369 GB/s  19.7%          141 reg
        #   (1,   64)          761.7 us  1587 GB/s  22.9%           74 reg
        #                      ^ 37.5% occupancy and still slower than shipped
        #
        # Those numbers are here as the record of *why* this is a tuned
        # quantity, not as values to install: they are one machine and one
        # shape. The tuner re-measures.
        #
        # Bit-exact for every tile shape tested, including under catastrophic
        # cancellation: the only reduction here is the unrolled ``for j`` sum,
        # which is a loop-local accumulator and therefore tile-invariant.
        # 20 + 20 random inputs x 11 configs, 0 differing elements
        # (opt/mhc_0908/logs/bitexact_prod.txt, logs/bitexact_extra.txt). This
        # is the one kernel in the family with no ``ct.sum`` at all, which is
        # also why it is the only one allowed to offer occupancy candidates.
        _mhc_tile_opt_retired()
        # ``fuse_cast`` means the caller skipped the fp32 widening it would
        # otherwise have done, so the kernel widens in-register and the result
        # comes back in the residual dtype. This cannot be sniffed
        # from dtypes -- h_res is fp32 either way -- so the caller states it.
        # UPCAST_INPUTS is a compile-time constant: fuse_cast=False compiles
        # to exactly the pre-fusion kernel.
        #
        # Declined when a bias is present: ``g_bias`` is computed outside the
        # kernel as ``g_x.sum(axis=0)``, i.e. a [sb, C] -> [C] reduction with no
        # fp32 accumulator of its own. A narrow g_x would therefore be summed in
        # bf16 and lose ~5e-3 relative, which is real precision loss rather than
        # the ULP-scale reassociation the other gradients see. Fixing it means
        # accumulating g_bias in-kernel from the un-rounded registers, but the
        # grid is one block per token so that needs a deterministic cross-block
        # reduction -- worth revisiting only for a bias-carrying config.
        fuse_cast = fuse_cast and bias is None
        out_dtype = original_residual.dtype if fuse_cast else h_res.dtype

        def _tune_hpb_fwd():
            def make_outputs():
                return (paddle.empty(shape=[sb, n, C], dtype=out_dtype),)

            def kernel_for(cfg):
                return _mhc_with_occupancy(_ct_hpb_fwd_kernel,
                                           cfg.get("OCC", 0))

            def do_launch(cfg, outs):
                ts, tc = cfg["TILE_SIZE"], cfg["TILE_C"]
                if bias is not None:
                    ct.launch(
                        _get_cuda_stream(), (math.ceil(sb / ts),),
                        _ct_hpb_fwd_bias_kernel,
                        (h_res.reshape([sb, n, n]),
                         original_residual.reshape([sb, n, C]),
                         h_post.reshape([sb, n]), x.reshape([sb, C]),
                         bias.detach(), outs[0], n, tc, ts),
                    )
                else:
                    ct.launch(
                        _get_cuda_stream(), (math.ceil(sb / ts),),
                        kernel_for(cfg),
                        (h_res.reshape([sb, n, n]),
                         original_residual.reshape([sb, n, C]),
                         h_post.reshape([sb, n]), x.reshape([sb, C]),
                         outs[0], n, tc, ts, fuse_cast),
                    )

            # The occupancy candidates, if any were asked for, take their slots
            # out of the eight rather than being appended past the budget --
            # otherwise they would be silently dropped by the truncation and
            # look as though they had been tried.
            occ = _mhc_occ_candidates(SHIPPED, bias is None, n * 4 * 2)
            pairs = _mhc_pair_candidates(
                SHIPPED, sb, C, n * 4 * 2,
                original_residual.element_size(), count=8 - len(occ))
            cands = [{"TILE_SIZE": r, "TILE_C": c, "OCC": 0} for r, c in pairs]
            cands += occ
            return _mhc_autotune_call(
                name="mhc.hpb_fwd",
                factory={"TILE_SIZE": SHIPPED[0], "TILE_C": SHIPPED[1],
                         "OCC": 0},
                candidates=cands,
                launch_shape=(sb, n, C),
                dtypes=(h_res.dtype, original_residual.dtype, h_post.dtype,
                        x.dtype, out_dtype, bias is not None,
                        bool(fuse_cast)),
                kernels=(_ct_hpb_fwd_kernel, _ct_hpb_fwd_bias_kernel),
                make_outputs=make_outputs,
                launch=do_launch,
                # Only the traffic the tile actually governs is counted: the
                # [sb, n, C] residual read and the [sb, n, C] output write.
                # Undercounting keeps the bound below the real floor, which is
                # the safe direction -- pruning by too small a bound is merely
                # weaker, pruning by too large a one could discard the winner.
                bound=lambda cfg, _res=original_residual,
                _ob=_mhc_dtype_bytes(out_dtype): _mhc_tile_bound(
                    sb * n * C * (_res.element_size() + _ob),
                    sb, cfg["TILE_SIZE"],
                    cfg["TILE_C"] * _res.element_size()),
            )

        cfg = _mhc_tune(
            "mhc.hpb_fwd",
            (sb, n, C, str(out_dtype), bias is not None, bool(fuse_cast)),
            _tune_hpb_fwd,
        )
        if cfg is None:
            cfg = {"TILE_SIZE": SHIPPED[0], "TILE_C": SHIPPED[1], "OCC": 0}
        TILE_SIZE, TILE_C = cfg["TILE_SIZE"], cfg["TILE_C"]
        fwd_kernel = _mhc_with_occupancy(_ct_hpb_fwd_kernel, cfg.get("OCC", 0))
        out = paddle.empty(shape=[sb, n, C], dtype=out_dtype)
        grid = (math.ceil(sb / TILE_SIZE),)
        if bias is not None:
            ct.launch(
                _get_cuda_stream(),
                grid,
                _ct_hpb_fwd_bias_kernel,
                (
                    h_res.reshape([sb, n, n]),
                    original_residual.reshape([sb, n, C]),
                    h_post.reshape([sb, n]),
                    x.reshape([sb, C]),
                    bias.detach(),
                    out,
                    n,
                    TILE_C,
                    TILE_SIZE,
                ),
            )
        else:
            ct.launch(
                _get_cuda_stream(),
                grid,
                fwd_kernel,
                (
                    h_res.reshape([sb, n, n]),
                    original_residual.reshape([sb, n, C]),
                    h_post.reshape([sb, n]),
                    x.reshape([sb, C]),
                    out,
                    n,
                    TILE_C,
                    TILE_SIZE,
                    fuse_cast,
                ),
            )
        return out.reshape([s, b, n, C])

    def _cutile_h_post_bda_bwd(
        grad_output: Tensor,
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        x: Tensor,
        bias: Tensor | None,
        fuse_cast: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        s, b, n, C = original_residual.shape
        sb = s * b
        # Optimized: use 2048 when possible (fastest for bwd), fall back to gcd for non-power-of-2 C
        TILE_C = math.gcd(C, 2048) if C % 2048 == 0 else math.gcd(C, 1024)
        TILE_SIZE = math.gcd(sb, 1)
        # Under ``fuse_cast`` residual/x arrived narrow, so their grads must
        # come back narrow too: returning fp32 would only make Paddle insert
        # the cast this path exists to remove. h_res/h_post stay fp32. Same
        # bias veto as the forward -- g_bias reduces over g_x.
        fuse_cast = fuse_cast and bias is None
        g_hr = paddle.empty(shape=[sb, n, n], dtype=h_res.dtype)
        g_res = paddle.empty(
            shape=[sb, n, C],
            dtype=original_residual.dtype if fuse_cast else h_res.dtype,
        )
        g_hp = paddle.empty(shape=[sb, n], dtype=h_res.dtype)
        g_x = paddle.empty(
            shape=[sb, C], dtype=x.dtype if fuse_cast else h_res.dtype
        )
        grid = (sb,)
        if bias is not None:
            ct.launch(
                _get_cuda_stream(),
                grid,
                _ct_hpb_bwd_bias_kernel,
                (
                    grad_output.reshape([sb, n, C]),
                    h_res.reshape([sb, n, n]),
                    original_residual.reshape([sb, n, C]),
                    h_post.reshape([sb, n]),
                    x.reshape([sb, C]),
                    bias.detach(),
                    g_hr,
                    g_res,
                    g_hp,
                    g_x,
                    n,
                    TILE_C,
                    TILE_SIZE,
                ),
            )
        else:
            ct.launch(
                _get_cuda_stream(),
                grid,
                _ct_hpb_bwd_kernel,
                (
                    grad_output.reshape([sb, n, C]),
                    h_res.reshape([sb, n, n]),
                    original_residual.reshape([sb, n, C]),
                    h_post.reshape([sb, n]),
                    x.reshape([sb, C]),
                    g_hr,
                    g_res,
                    g_hp,
                    g_x,
                    n,
                    TILE_C,
                    TILE_SIZE,
                    fuse_cast,
                ),
            )
        g_bias = g_x.sum(axis=0) if bias is not None else None
        return (
            g_hr.reshape([s, b, n, n]),
            g_res.reshape([s, b, n, C]),
            g_hp.reshape([s, b, n]),
            g_x.reshape([s, b, C]),
            g_bias,
        )

    # -- Proj RMS kernels ----------------------------------------------------

    @ct.function
    def _ct_rms_dnorm(a_tile, norm_tile, dr_tile, K, eps):
        inv_norm = ct.where(norm_tile > 0, 1.0 / norm_tile, 0.0)
        inv_sqrt_k = 1.0 / ct.sqrt(K)
        u = norm_tile * inv_sqrt_k + eps
        coeff = -(1.0 / (u * u)) * inv_sqrt_k
        return dr_tile * coeff * a_tile * inv_norm

    @ct.kernel
    def _ct_proj_rms_fwd_kernel(
        A,
        B,
        PROJ,
        NORM,
        R,
        M: int,
        N: int,
        K: int,
        eps: float,
        TILE_M: ConstInt,
        TILE_N: ConstInt,
        TILE_K: ConstInt,
        UPCAST_INPUTS: ConstBool,
    ):
        """Fused projection + RMS norm.

        ``UPCAST_INPUTS`` widens ``A`` in-register when it arrives narrower than
        the fp32 the norm reduction runs in. The ``ct.mma`` below is unaffected
        either way -- it truncates both operands to tfloat32, and a bf16-valued
        tile survives that exactly -- but ``sum_sq`` would otherwise square and
        accumulate in the narrow dtype and lose real precision.
        """
        tile_m_id = ct.bid(0)
        num_k_tiles = ct.cdiv(K, TILE_K)
        acc = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        sum_sq = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
        for tile_k_id in range(num_k_tiles):
            a_tile = ct.load(
                A,
                index=(tile_m_id, tile_k_id),
                shape=(TILE_M, TILE_K),
                padding_mode=PAD_ZERO,
            )
            b_tile = ct.load(
                B,
                index=(0, tile_k_id),
                shape=(TILE_N, TILE_K),
                padding_mode=PAD_ZERO,
            )
            if UPCAST_INPUTS:
                a_tile = a_tile.astype(ct.float32)
            acc = ct.mma(
                a_tile.astype(ct.tfloat32),
                b_tile.transpose().astype(ct.tfloat32),
                acc=acc,
            )
            sum_sq += ct.sum(a_tile * a_tile, axis=1, keepdims=True)
        norm_tile = ct.sqrt(sum_sq)
        v = norm_tile / ct.sqrt(K) + eps
        r_tile = 1.0 / v
        ct.store(PROJ, index=(tile_m_id, 0), tile=acc.astype(PROJ.dtype))
        ct.store(NORM, index=(tile_m_id, 0), tile=norm_tile.astype(NORM.dtype))
        ct.store(R, index=(tile_m_id, 0), tile=r_tile.astype(R.dtype))

    @ct.kernel
    def _ct_proj_rms_bwd_kernel(
        A,
        B,
        NORM,
        DD,
        DR,
        DA,
        DB,
        M: int,
        N: int,
        K: int,
        eps: float,
        TILE_SIZE_M: ConstInt,
        TILE_SIZE_N: ConstInt,
        TILE_SIZE_K: ConstInt,
        UPCAST_INPUTS: ConstBool,
    ):
        """Backward of ``_ct_proj_rms_fwd_kernel`` for the large-K path.

        ``UPCAST_INPUTS`` widens ``A`` before ``_ct_rms_dnorm``; the mma below
        truncates to tfloat32 regardless. ``_ct_proj_rms_bwd_small_k_kernel``
        needs no such flag -- it already widens unconditionally.
        """
        zero_pad = ct.PaddingMode.ZERO
        tile_k_id = ct.bid(0)
        NUM_M_TILES = ct.cdiv(M, TILE_SIZE_M)
        accumulator_db = ct.full(
            (TILE_SIZE_K, TILE_SIZE_N), 0.0, dtype=ct.float32
        )
        for tile_m_id in range(NUM_M_TILES):
            accumulator_da = ct.full(
                (TILE_SIZE_M, TILE_SIZE_K), 0.0, dtype=ct.float32
            )
            a_tile = ct.load(
                A,
                index=(tile_m_id, tile_k_id),
                shape=(TILE_SIZE_M, TILE_SIZE_K),
                padding_mode=zero_pad,
            )
            norm_tile = ct.load(
                NORM,
                index=(tile_m_id, 0),
                shape=(TILE_SIZE_M, 1),
                padding_mode=zero_pad,
            )
            dr_tile = ct.load(
                DR,
                index=(tile_m_id, 0),
                shape=(TILE_SIZE_M, 1),
                padding_mode=zero_pad,
            )
            if UPCAST_INPUTS:
                a_tile = a_tile.astype(ct.float32)
            accumulator_da = accumulator_da + _ct_rms_dnorm(
                a_tile, norm_tile, dr_tile, K, eps
            )
            b_tile = ct.load(
                B,
                index=(0, tile_k_id),
                shape=(TILE_SIZE_N, TILE_SIZE_K),
                padding_mode=zero_pad,
            )
            dd_tile = ct.load(
                DD,
                index=(tile_m_id, 0),
                shape=(TILE_SIZE_M, TILE_SIZE_N),
                padding_mode=zero_pad,
            )
            dd_tile = ct.astype(dd_tile, ct.tfloat32)
            accumulator_da = ct.mma(
                dd_tile, b_tile.astype(ct.tfloat32), acc=accumulator_da
            )
            ct.store(
                DA,
                index=(tile_m_id, tile_k_id),
                tile=accumulator_da.astype(DA.dtype),
            )
            accumulator_db = ct.mma(
                a_tile.transpose().astype(ct.tfloat32),
                dd_tile,
                acc=accumulator_db,
            )
        ct.store(
            DB,
            index=(0, tile_k_id),
            tile=accumulator_db.transpose().astype(DB.dtype),
        )

    @ct.kernel
    def _ct_proj_rms_bwd_small_k_kernel(
        A,
        B,
        NORM,
        DD,
        DR,
        DA,
        DB,
        M: int,
        N: int,
        K: int,
        eps: float,
        TILE_N_SIZE: ConstInt,
    ):
        zero_pad = ct.PaddingMode.ZERO
        TILE_DB_SIZE_M = 128
        TILE_DB_SIZE_K = 64
        NUM_M_TILES = ct.cdiv(M, TILE_DB_SIZE_M)
        NUM_K_TILES = ct.cdiv(K, TILE_DB_SIZE_K)
        if ct.bid(1) == 0:
            for tile_id in range(ct.bid(0), NUM_K_TILES, ct.num_blocks(0)):
                accumulator_db = ct.full(
                    (TILE_DB_SIZE_K, TILE_N_SIZE), 0.0, dtype=ct.float32
                )
                for m_tile in range(NUM_M_TILES):
                    a_tile = ct.load(
                        A,
                        index=(m_tile, tile_id),
                        shape=(TILE_DB_SIZE_M, TILE_DB_SIZE_K),
                        padding_mode=zero_pad,
                    )
                    dd_tile = ct.load(
                        DD,
                        index=(m_tile, 0),
                        shape=(TILE_DB_SIZE_M, TILE_N_SIZE),
                        padding_mode=zero_pad,
                    )
                    accumulator_db = ct.mma(
                        a_tile.transpose().astype(ct.tfloat32),
                        dd_tile.astype(ct.tfloat32),
                        acc=accumulator_db,
                    )
                ct.store(
                    DB,
                    index=(0, tile_id),
                    tile=accumulator_db.transpose().astype(DB.dtype),
                    allow_tma=False,
                )
        TILE_DA_SIZE_M = 128
        TILE_DA_SIZE_K = 256
        NUM_DA_TILES = ct.cdiv(M, TILE_DA_SIZE_M) * ct.cdiv(K, TILE_DA_SIZE_K)
        NUM_DA_K_TILES = ct.cdiv(K, TILE_DA_SIZE_K)
        if ct.bid(1) == 1:
            for tile_id in range(ct.bid(0), NUM_DA_TILES, ct.num_blocks(0)):
                b_tile_idx = tile_id % NUM_DA_K_TILES
                dd_tile_idx = tile_id // NUM_DA_K_TILES
                accumulator_da = ct.full(
                    (TILE_DA_SIZE_M, TILE_DA_SIZE_K), 0.0, dtype=ct.float32
                )
                a_tile = ct.load(
                    A,
                    index=(dd_tile_idx, b_tile_idx),
                    shape=(TILE_DA_SIZE_M, TILE_DA_SIZE_K),
                    padding_mode=zero_pad,
                )
                norm_tile = ct.load(
                    NORM,
                    index=(dd_tile_idx, 0),
                    shape=(TILE_DA_SIZE_M, 1),
                    padding_mode=zero_pad,
                )
                dr_tile = ct.load(
                    DR,
                    index=(dd_tile_idx, 0),
                    shape=(TILE_DA_SIZE_M, 1),
                    padding_mode=zero_pad,
                )
                accumulator_da = accumulator_da + _ct_rms_dnorm(
                    a_tile.astype(ct.float32), norm_tile, dr_tile, K, eps
                )
                b_tile = ct.load(
                    B,
                    index=(0, b_tile_idx),
                    shape=(TILE_N_SIZE, TILE_DA_SIZE_K),
                    padding_mode=zero_pad,
                )
                dd_tile = ct.load(
                    DD,
                    index=(dd_tile_idx, 0),
                    shape=(TILE_DA_SIZE_M, TILE_N_SIZE),
                    padding_mode=zero_pad,
                )
                accumulator_da = ct.mma(
                    dd_tile.astype(ct.tfloat32),
                    b_tile.astype(ct.tfloat32),
                    acc=accumulator_da,
                )
                ct.store(
                    DA,
                    index=(dd_tile_idx, b_tile_idx),
                    tile=accumulator_da.astype(DA.dtype),
                )

    def _next_power_of_2(n: int) -> int:
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        n |= n >> 32
        n += 1
        return n

    def _cutile_proj_rms_fwd(
        x: Tensor, weight: Tensor, eps: float = 1e-8, fuse_cast: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        M, K = x.shape
        N = weight.shape[0]
        # Optimized: TILE_M=64, TILE_K=128 is fastest (0.113ms vs 0.145ms@128/128)
        TILE_M = 64
        TILE_N = _next_power_of_2(N)
        TILE_K = 128
        num_tiles_m = math.ceil(M / TILE_M)
        # Under fuse_cast x is narrow but the three outputs must stay fp32: proj
        # and r feed _compute_h, and narrowing them there would drag h_res /
        # h_post down with them, i.e. silently undo high_precision_mhc. norm is
        # kept for backward and consumed at fp32 by _ct_rms_dnorm.
        out_dtype = "float32" if fuse_cast else x.dtype
        proj = paddle.empty(shape=[M, N], dtype=out_dtype)
        norm = paddle.empty(shape=[M, 1], dtype=out_dtype)
        r = paddle.empty(shape=[M, 1], dtype=out_dtype)
        ct.launch(
            _get_cuda_stream(),
            (num_tiles_m,),
            _ct_proj_rms_fwd_kernel,
            (
                x.detach(),
                weight.detach(),
                proj,
                norm,
                r,
                M,
                N,
                K,
                eps,
                TILE_M,
                TILE_N,
                TILE_K,
                fuse_cast,
            ),
        )
        return proj, norm, r

    def _cutile_proj_rms_bwd(
        grad_proj: Tensor,
        grad_r: Tensor,
        x: Tensor,
        weight: Tensor,
        norm: Tensor,
        eps: float = 1e-8,
        fuse_cast: bool = False,
    ) -> tuple[Tensor, Tensor]:
        M, K = x.shape
        N = weight.shape[0]
        da = paddle.empty(shape=x.shape, dtype=x.dtype)
        db = paddle.empty(shape=weight.shape, dtype=weight.dtype)
        TILE_SIZE_N = _next_power_of_2(N)
        assert TILE_SIZE_N <= 256, f"TILE_SIZE_N too large: {TILE_SIZE_N}"
        num_sms = (
            paddle.device.cuda.get_device_properties().multi_processor_count
        )
        if K >= 8192:
            # Optimized: TM=64, TK=128 is fastest (0.186ms vs 0.208ms@128/128)
            TILE_SIZE_M, TILE_SIZE_K = 64, 128
            # opt/mhc_0908: the grid here is one block per K tile, so
            # TILE_SIZE_K=128 with K=n*hidden_size=16384 launches 128 blocks,
            # fewer than the multiprocessor count of the part this was measured
            # on -- some multiprocessors stay empty for the whole kernel, and
            # each block then walks all M/TILE_SIZE_M m-tiles serially. Filling
            # the part with TK=64 looked like the obvious fix and is 1.54x
            # SLOWER (432.4 us vs 281.6 at TM=64), as is every other legal
            # combination: the shipped (64, 128) is the best of the eight
            # measured. So the defaults here are the shipped values and this is
            # a knob only.
            #
            # Bit-exact for every (TM, TK) tested: ``DB`` accumulates across
            # m-tiles in a single ``ct.mma(acc=...)`` carried by the loop, and
            # re-chunking a sequential accumulation does not reorder it. 20
            # random inputs at M=256 plus 20 at M=4096 with operands spread
            # over 12 decades, 0 differing elements (logs/bitexact_prod.txt,
            # logs/bitexact_extra.txt).
            # logs/bitexact_extra.txt). Not routed through the tuner: the
            # shipped pair already won its sweep and there is no second
            # candidate worth eight timing passes in the first step. The knob
            # that used to sit here defaulted to the shipped values, so
            # removing it changes nothing.
            grid = (math.ceil(K / TILE_SIZE_K), 1)
            ct.launch(
                _get_cuda_stream(),
                grid,
                _ct_proj_rms_bwd_kernel,
                (
                    x.detach(),
                    weight.detach(),
                    norm.detach(),
                    grad_proj.detach(),
                    grad_r.detach(),
                    da,
                    db,
                    M,
                    N,
                    K,
                    eps,
                    TILE_SIZE_M,
                    TILE_SIZE_N,
                    TILE_SIZE_K,
                    fuse_cast,
                ),
            )
        else:
            grid = (num_sms, 2, 1)
            ct.launch(
                _get_cuda_stream(),
                grid,
                _ct_proj_rms_bwd_small_k_kernel,
                (
                    x.detach(),
                    weight.detach(),
                    norm.detach(),
                    grad_proj.detach(),
                    grad_r.detach(),
                    da,
                    db,
                    M,
                    N,
                    K,
                    eps,
                    TILE_SIZE_N,
                ),
            )
        return da, db


# ============================================================================
# Autograd Functions (cuTile only – guarded by _CUTILE_AVAILABLE)
# ============================================================================

if not _CUTILE_AVAILABLE:

    def _no_cutile_error(*_args, **_kwargs):
        raise RuntimeError(
            "Fused mHC kernels require cuda.tile (cuTile) which is not installed. "
            "Either install cuTile or set use_fused_mhc=False to use reference "
            "implementations."
        )

    fused_sinkhorn = _no_cutile_error
    fused_h_aggregate = _no_cutile_error
    fused_h_post_bda = _no_cutile_error
    fused_proj_rms = _no_cutile_error
    fused_compute_h = _no_cutile_error

else:

    class FusedSinkhornKnopp(paddle.autograd.PyLayer):
        """Fused Sinkhorn-Knopp projection to doubly stochastic matrix (cuTile).

        ``backward`` must return ``None`` at every position whose forward input
        had ``stop_gradient=True``, or Paddle raises

            InvalidArgumentError: GradNodePyLayer_FusedSinkhornKnopp's backward
            function should return None at N position, ...

        That happens with a frozen backbone (``train_indexer_only``): the mHC
        block runs on detached inputs and frozen parameters, yet the segment
        stays differentiable because the Indexer loss is attached downstream.
        ``stop_gradient`` is only trustworthy on a PyLayer's *forward* inputs, so
        it is recorded here and read in backward.
        """

        @staticmethod
        def forward(
            ctx, input_logits: Tensor, num_iterations: int, eps: float = 1e-6
        ):
            """cuTile fused Sinkhorn forward."""
            output, M_init = _cutile_sinkhorn_fwd(
                input_logits, num_iterations, eps
            )
            ctx.save_for_backward(M_init)
            ctx.num_iterations = num_iterations
            ctx.eps = eps
            ctx.input_logits_stop_gradient = input_logits.stop_gradient
            return output

        @staticmethod
        def backward(ctx, grad_output):
            """cuTile fused Sinkhorn backward."""
            if ctx.input_logits_stop_gradient:
                return None
            (M_init,) = ctx.saved_tensor()
            grad_input = _cutile_sinkhorn_bwd(
                grad_output, M_init, ctx.num_iterations, ctx.eps
            )
            return grad_input

    class FusedComputeH(paddle.autograd.PyLayer):
        """Fused ``h = r * proj * alpha + bias`` plus the two sigmoid heads.

        See ``FusedSinkhornKnopp`` for why the ``stop_gradient`` flags are
        recorded in forward and honored in backward: the mHC parameters are
        frozen under ``train_indexer_only``, and Paddle demands ``None`` at
        every position whose forward input was detached.
        """

        @staticmethod
        def forward(
            ctx,
            proj: Tensor,
            r: Tensor,
            alpha_pre: Tensor,
            alpha_post: Tensor,
            alpha_res: Tensor,
            bias: Tensor,
            n: int,
            eps: float,
        ):
            h_pre, h_post, h_res = _cutile_compute_h_fwd(
                proj, r, alpha_pre, alpha_post, alpha_res, bias, n, eps
            )
            ctx.save_for_backward(
                proj, r, h_pre, h_post, alpha_pre, alpha_post, alpha_res, bias
            )
            ctx.n = n
            ctx.eps = eps
            ctx.stop = (
                proj.stop_gradient,
                r.stop_gradient,
                alpha_pre.stop_gradient,
                alpha_post.stop_gradient,
                alpha_res.stop_gradient,
                bias.stop_gradient,
            )
            return h_pre, h_post, h_res

        @staticmethod
        def backward(ctx, g_h_pre, g_h_post, g_h_res):
            (
                proj,
                r,
                h_pre,
                h_post,
                alpha_pre,
                alpha_post,
                alpha_res,
                bias,
            ) = ctx.saved_tensor()
            grads = _cutile_compute_h_bwd(
                g_h_pre,
                g_h_post,
                g_h_res,
                proj,
                r,
                h_pre,
                h_post,
                alpha_pre,
                alpha_post,
                alpha_res,
                bias,
                ctx.n,
                ctx.eps,
            )
            return tuple(
                None if frozen else g for g, frozen in zip(grads, ctx.stop)
            )

    class FusedHAggregate(paddle.autograd.PyLayer):
        """Fused n-stream weighted aggregation (cuTile).

        See ``FusedSinkhornKnopp`` for why the ``stop_gradient`` flags are
        recorded in forward and honored in backward.
        """

        @staticmethod
        def forward(ctx, x: Tensor, h_pre: Tensor, fuse_cast: bool = False):
            """cuTile fused h_aggregate forward."""
            output = _cutile_h_aggregate_fwd(x, h_pre, fuse_cast)
            ctx.save_for_backward(x, h_pre)
            ctx.x_stop_gradient = x.stop_gradient
            ctx.h_pre_stop_gradient = h_pre.stop_gradient
            ctx.fuse_cast = fuse_cast
            return output

        @staticmethod
        def backward(ctx, grad_output):
            """cuTile fused h_aggregate backward."""
            x, h_pre = ctx.saved_tensor()
            g_x, g_h_pre = _cutile_h_aggregate_bwd(
                grad_output, x, h_pre, ctx.fuse_cast
            )
            if ctx.x_stop_gradient:
                g_x = None
            if ctx.h_pre_stop_gradient:
                g_h_pre = None
            return g_x, g_h_pre

    class FusedHPostBDA(paddle.autograd.PyLayer):
        """Fused: output = H_res @ orig_res + H_post * (x [+ bias]) (cuTile).

        See ``FusedSinkhornKnopp`` for why the ``stop_gradient`` flags are
        recorded in forward and honored in backward.
        """

        @staticmethod
        def forward(
            ctx,
            h_res: Tensor,
            original_residual: Tensor,
            h_post: Tensor,
            x: Tensor,
            bias: Tensor | None,
            fuse_cast: bool = False,
        ):
            """cuTile fused h_post_bda forward."""
            output = _cutile_h_post_bda_fwd(
                h_res, original_residual, h_post, x, bias, fuse_cast
            )
            if bias is not None:
                ctx.save_for_backward(h_res, original_residual, h_post, x, bias)
                ctx.has_bias = True
            else:
                ctx.save_for_backward(h_res, original_residual, h_post, x)
                ctx.has_bias = False
            ctx.fuse_cast = fuse_cast
            ctx.h_res_stop_gradient = h_res.stop_gradient
            ctx.original_residual_stop_gradient = (
                original_residual.stop_gradient
            )
            ctx.h_post_stop_gradient = h_post.stop_gradient
            ctx.x_stop_gradient = x.stop_gradient
            ctx.bias_stop_gradient = (
                bias.stop_gradient if bias is not None else True
            )
            return output

        @staticmethod
        def backward(ctx, grad_output):
            """cuTile fused h_post_bda backward."""
            if ctx.has_bias:
                h_res, orig_res, h_post, x, bias = ctx.saved_tensor()
                g_hr, g_res, g_hp, g_x, g_bias = _cutile_h_post_bda_bwd(
                    grad_output, h_res, orig_res, h_post, x, bias, ctx.fuse_cast
                )
            else:
                h_res, orig_res, h_post, x = ctx.saved_tensor()
                g_hr, g_res, g_hp, g_x, _ = _cutile_h_post_bda_bwd(
                    grad_output, h_res, orig_res, h_post, x, None, ctx.fuse_cast
                )
                g_bias = None
            if ctx.h_res_stop_gradient:
                g_hr = None
            if ctx.original_residual_stop_gradient:
                g_res = None
            if ctx.h_post_stop_gradient:
                g_hp = None
            if ctx.x_stop_gradient:
                g_x = None
            if ctx.bias_stop_gradient:
                g_bias = None
            if ctx.has_bias:
                return g_hr, g_res, g_hp, g_x, g_bias
            return g_hr, g_res, g_hp, g_x

    class FusedProjRms(paddle.autograd.PyLayer):
        """Fused projection + RMS normalization (cuTile).

        See ``FusedSinkhornKnopp`` for why the ``stop_gradient`` flags are
        recorded in forward and honored in backward. ``weight`` is the one that
        bites in practice: it is a frozen backbone parameter of the mHC block
        under ``train_indexer_only``, which used to abort backward with
        ``... should return None at 1 position``.
        """

        @staticmethod
        def forward(
            ctx,
            x: Tensor,
            weight: Tensor,
            eps: float = 1e-6,
            fuse_cast: bool = False,
        ):
            """cuTile fused proj_rms forward."""
            original_shape = x.shape
            K = original_shape[-1]
            x_2d = x.reshape([-1, K])
            proj, norm, r = _cutile_proj_rms_fwd(x_2d, weight, eps, fuse_cast)
            ctx.save_for_backward(x_2d, weight, norm)
            ctx.eps = eps
            ctx.fuse_cast = fuse_cast
            ctx.original_shape = original_shape
            ctx.x_stop_gradient = x.stop_gradient
            ctx.weight_stop_gradient = weight.stop_gradient
            N = weight.shape[0]
            batch_shape = list(original_shape[:-1])
            return proj.reshape([*batch_shape, N]), r.reshape([*batch_shape, 1])

        @staticmethod
        def backward(ctx, grad_proj, grad_r):
            """cuTile fused proj_rms backward."""
            x_2d, weight, norm = ctx.saved_tensor()
            original_shape = ctx.original_shape
            grad_proj_2d = grad_proj.reshape([-1, grad_proj.shape[-1]])
            grad_r_2d = grad_r.reshape([-1, 1])
            grad_x, grad_weight = _cutile_proj_rms_bwd(
                grad_proj_2d,
                grad_r_2d,
                x_2d,
                weight,
                norm,
                ctx.eps,
                ctx.fuse_cast,
            )
            return (
                None if ctx.x_stop_gradient else grad_x.reshape(original_shape),
                None if ctx.weight_stop_gradient else grad_weight,
            )

    # ========================================================================
    # Public API (only available when cuTile is installed)
    # ========================================================================

    def fused_sinkhorn(
        input_logits: Tensor, num_iterations: int, eps: float = 1e-6
    ) -> Tensor:
        """Project logits to doubly stochastic matrix via Sinkhorn-Knopp.

        Args:
            input_logits: [..., n, n] raw logits
            num_iterations: Sinkhorn iterations
            eps: numerical stability

        Returns:
            [..., n, n] doubly stochastic matrix
        """
        assert input_logits.ndim >= 2, (
            f"fused_sinkhorn: input must be at least 2D, got shape {list(input_logits.shape)}"
        )
        assert input_logits.shape[-1] == input_logits.shape[-2], (
            f"fused_sinkhorn: last two dims must be equal (square matrix), "
            f"got shape {list(input_logits.shape)}"
        )
        hc = input_logits.shape[-1]
        N_batch = input_logits.size // (hc * hc)
        assert N_batch <= _INT32_MAX, (
            f"fused_sinkhorn: N_batch={N_batch} exceeds int32 max ({_INT32_MAX})"
        )
        return FusedSinkhornKnopp.apply(input_logits, num_iterations, eps)

    def fused_compute_h(
        proj: Tensor,
        r: Tensor,
        alpha_pre: Tensor,
        alpha_post: Tensor,
        alpha_res: Tensor,
        bias: Tensor,
        n: int,
        eps: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Fused mHC mapping head.

        Replaces ``expand x3 -> concat -> mul -> mul -> add -> sigmoid x2`` with
        one launch, and drops the ``alpha`` vector entirely: ``n`` is a
        compile-time constant in the kernel, so each output segment applies its
        own scalar directly.

        Args:
            proj: [..., n*n + 2*n] projection of the n-stream hidden states
            r: [..., 1] inverse RMS scale
            alpha_pre / alpha_post / alpha_res: [1] learnable gates
            bias: [n*n + 2*n] static bias
            n: number of residual streams
            eps: the constant added to h_pre

        Returns:
            h_pre: [..., n] sigmoid(u_pre) + eps
            h_post: [..., n] 2 * sigmoid(u_post)
            h_res: [..., n*n] u_res, unactivated
        """
        P = n * n + 2 * n
        # Raised, not asserted: ``python -O`` strips asserts, and a wrong P or
        # bias length then reaches a kernel that addresses the three output
        # segments at fixed offsets, i.e. reads out of bounds.
        if proj.shape[-1] != P:
            raise ValueError(
                f"fused_compute_h: proj last dim must be n*n+2*n={P}, "
                f"got {proj.shape[-1]} (proj.shape={list(proj.shape)})"
            )
        if r.shape[-1] != 1:
            raise ValueError(
                f"fused_compute_h: r last dim must be 1, got {list(r.shape)}"
            )
        if list(r.shape[:-1]) != list(proj.shape[:-1]):
            raise ValueError(
                f"fused_compute_h: r shape {list(r.shape)} and proj shape "
                f"{list(proj.shape)} must agree on the leading dims"
            )
        if bias.shape != [P]:
            raise ValueError(
                f"fused_compute_h: bias must be [{P}], got {list(bias.shape)}"
            )
        return FusedComputeH.apply(
            proj, r, alpha_pre, alpha_post, alpha_res, bias, n, eps
        )

    def fused_h_aggregate(
        x: Tensor, h_pre: Tensor, fuse_cast: bool = False
    ) -> Tensor:
        """Weighted n-stream to 1-stream aggregation.

        Args:
            x: [s, b, n, C] n-stream hidden states
            h_pre: [s, b, n] aggregation weights
            fuse_cast: when True, ``x`` may be narrower than ``h_pre``; the
                kernel widens it in-register. Only the caller knows whether it
                skipped the widening, so it cannot be inferred from dtypes.

        Returns:
            [s, b, C] aggregated hidden states
        """
        assert x.ndim == 4, (
            f"fused_h_aggregate: x must be 4D [s,b,n,C], got shape {list(x.shape)}"
        )
        assert h_pre.ndim == 3, (
            f"fused_h_aggregate: h_pre must be 3D [s,b,n], got shape {list(h_pre.shape)}"
        )
        assert x.shape[:3] == h_pre.shape[:3], (
            f"fused_h_aggregate: x shape {list(x.shape)} and h_pre shape {list(h_pre.shape)} "
            f"must match on first 3 dims [s,b,n]"
        )
        s, b, n, C = x.shape
        assert s * b <= _INT32_MAX, (
            f"fused_h_aggregate: s*b={s * b} exceeds int32 max ({_INT32_MAX})"
        )
        assert C <= _INT32_MAX, (
            f"fused_h_aggregate: C={C} exceeds int32 max ({_INT32_MAX})"
        )
        return FusedHAggregate.apply(x, h_pre, fuse_cast)

    def fused_h_post_bda(
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        x: Tensor,
        bias: Tensor | None,
        fuse_cast: bool = False,
    ) -> Tensor:
        """Fused H_res @ residual + H_post * (x + bias).

        Args:
            h_res: [s, b, n, n] residual mixing matrix
            original_residual: [s, b, n, C] n-stream residual
            h_post: [s, b, n] expansion weights
            x: [s, b, C] layer output
            bias: [C] or None
            fuse_cast: when True, ``original_residual``/``x`` may be narrower
                than the fp32 the kernel computes in; it widens them in-register
                and returns its result in ``original_residual``'s dtype. Only
                the caller knows whether it skipped the widening, so this cannot
                be inferred from dtypes -- ``h_res`` is fp32 either way.
                Declined when ``bias`` is not None, since ``g_bias`` reduces
                over ``g_x`` and would lose precision if ``g_x`` were narrow.

        Returns:
            [s, b, n, C] fused output
        """
        assert h_res.ndim == 4 and h_res.shape[-1] == h_res.shape[-2], (
            f"fused_h_post_bda: h_res must be 4D [s,b,n,n], got shape {list(h_res.shape)}"
        )
        assert original_residual.ndim == 4, (
            f"fused_h_post_bda: original_residual must be 4D [s,b,n,C], got shape {list(original_residual.shape)}"
        )
        n = h_res.shape[-1]
        assert original_residual.shape[2] == n, (
            f"fused_h_post_bda: original_residual dim2={original_residual.shape[2]} != n={n}"
        )
        assert h_post.ndim == 3 and h_post.shape[-1] == n, (
            f"fused_h_post_bda: h_post must be 3D [s,b,n], got shape {list(h_post.shape)}"
        )
        assert x.ndim == 3 and x.shape[-1] == original_residual.shape[-1], (
            f"fused_h_post_bda: x must be 3D [s,b,C] with C={original_residual.shape[-1]}, got shape {list(x.shape)}"
        )
        s, b = original_residual.shape[:2]
        C = original_residual.shape[-1]
        assert s * b <= _INT32_MAX, (
            f"fused_h_post_bda: s*b={s * b} exceeds int32 max ({_INT32_MAX})"
        )
        assert C <= _INT32_MAX, (
            f"fused_h_post_bda: C={C} exceeds int32 max ({_INT32_MAX})"
        )
        return FusedHPostBDA.apply(
            h_res, original_residual, h_post, x, bias, fuse_cast
        )

    def fused_proj_rms(
        x: Tensor,
        weight: Tensor,
        eps: float = 1e-6,
        fuse_cast: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Fused projection + RMS normalization.

        Args:
            x: [..., K] input (last dim is K)
            weight: [K, N] projection weight
            eps: stability epsilon
            fuse_cast: when True, ``x``/``weight`` may be narrow; the kernel
                widens ``x`` in-register and returns ``proj``/``r`` in fp32 so
                the mappings built from them keep their precision.

        Returns:
            proj: [..., N] = x @ weight^T
            r: [..., 1] = 1 / (||x|| / sqrt(K) + eps)
        """
        # [K, N] --> [N, K]
        weight = weight.t()
        assert weight.ndim == 2, (
            f"fused_proj_rms: weight must be 2D [N, K], got shape {list(weight.shape)}"
        )
        K = x.shape[-1]
        N, K_w = weight.shape
        assert K == K_w, (
            f"fused_proj_rms: x last dim (K={K}) must match weight dim1 (K={K_w}). "
            f"x.shape={list(x.shape)}, weight.shape={list(weight.shape)}. "
            f"If weight is [K, N], you need to transpose it: fused_proj_rms(x, weight.t())"
        )
        assert N <= 256, (
            f"fused_proj_rms: N={N} exceeds max supported tile size 256. "
            f"weight.shape={list(weight.shape)}. Check if weight needs transposing."
        )
        M = x.size // K
        assert M <= _INT32_MAX, (
            f"fused_proj_rms: M={M} (x reshaped to [M, K]) exceeds int32 max ({_INT32_MAX})"
        )
        assert K <= _INT32_MAX, (
            f"fused_proj_rms: K={K} exceeds int32 max ({_INT32_MAX})"
        )
        return FusedProjRms.apply(x, weight, eps, fuse_cast)
