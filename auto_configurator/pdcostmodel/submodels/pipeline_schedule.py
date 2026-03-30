#!/usr/bin/env python3
"""Shared helpers for stage-aware 1F1B pipeline timing."""

from typing import Optional, Sequence, Tuple


def simulate_1f1b_makespan(
    forward_stage_ms: Sequence[float],
    backward_stage_ms: Sequence[float],
    num_micro_batches: int,
    forward_boundary_ms: Optional[Sequence[float]] = None,
    backward_boundary_ms: Optional[Sequence[float]] = None,
) -> float:
    """
    Simulate a simple 1F1B schedule and return step makespan in milliseconds.

    The model is intentionally stage-aware:
    - forward/backward durations are per physical stage and per micro-batch
    - boundary latencies can differ across PP edges
    - backward is prioritized over forward on a stage once gradients are ready

    This is still a simplified scheduler:
    - no VPP interleave
    - no dual-pipe
    - no explicit activation memory constraints
    """
    stage_count = len(forward_stage_ms)
    if stage_count == 0:
        return 0.0

    micro_batches = max(1, int(num_micro_batches))
    if stage_count == 1:
        return micro_batches * max(
            0.0,
            float(forward_stage_ms[0]) + float(backward_stage_ms[0]),
        )

    fwd_comm = [0.0] * (stage_count - 1)
    bwd_comm = [0.0] * (stage_count - 1)
    if forward_boundary_ms is not None:
        for idx, value in enumerate(forward_boundary_ms[: stage_count - 1]):
            fwd_comm[idx] = max(0.0, float(value))
    if backward_boundary_ms is not None:
        for idx, value in enumerate(backward_boundary_ms[: stage_count - 1]):
            bwd_comm[idx] = max(0.0, float(value))

    stage_free = [0.0] * stage_count
    forward_done = [[None] * micro_batches for _ in range(stage_count)]
    backward_done = [[None] * micro_batches for _ in range(stage_count)]
    next_forward = [0] * stage_count
    next_backward = [0] * stage_count

    scheduled = 0
    total_ops = 2 * stage_count * micro_batches

    while scheduled < total_ops:
        best: Optional[Tuple[float, int, int, int, str]] = None

        for stage_id in range(stage_count):
            backward_mb = next_backward[stage_id]
            if (
                backward_mb < micro_batches
                and forward_done[stage_id][backward_mb] is not None
            ):
                backward_ready = float(forward_done[stage_id][backward_mb])
                if stage_id < stage_count - 1:
                    next_stage_done = backward_done[stage_id + 1][backward_mb]
                    if next_stage_done is not None:
                        backward_ready = max(
                            backward_ready,
                            float(next_stage_done) + bwd_comm[stage_id],
                        )
                    else:
                        backward_ready = -1.0
                if backward_ready >= 0.0:
                    backward_start = max(stage_free[stage_id], backward_ready)
                    candidate = (
                        backward_start,
                        0,  # backward has higher priority
                        stage_id,
                        backward_mb,
                        "b",
                    )
                    if best is None or candidate < best:
                        best = candidate

            forward_mb = next_forward[stage_id]
            if forward_mb < micro_batches:
                if stage_id == 0:
                    forward_ready = 0.0
                else:
                    prev_stage_done = forward_done[stage_id - 1][forward_mb]
                    if prev_stage_done is None:
                        forward_ready = -1.0
                    else:
                        forward_ready = float(prev_stage_done) + fwd_comm[stage_id - 1]
                if forward_ready >= 0.0:
                    forward_start = max(stage_free[stage_id], forward_ready)
                    candidate = (
                        forward_start,
                        1,
                        stage_id,
                        forward_mb,
                        "f",
                    )
                    if best is None or candidate < best:
                        best = candidate

        if best is None:
            raise RuntimeError("failed to construct a valid 1F1B schedule")

        start_time, _, stage_id, micro_batch, kind = best
        if kind == "f":
            end_time = start_time + max(0.0, float(forward_stage_ms[stage_id]))
            forward_done[stage_id][micro_batch] = end_time
            next_forward[stage_id] += 1
        else:
            end_time = start_time + max(0.0, float(backward_stage_ms[stage_id]))
            backward_done[stage_id][micro_batch] = end_time
            next_backward[stage_id] += 1
        stage_free[stage_id] = end_time
        scheduled += 1

    return max(stage_free)
