---
name: monitor-hook-perf-rules
description: >
  Mandatory discipline for monitor (probe) hot paths in internal_medicine.
  Hooks can enqueue CUDA/NCCL work that contends with EP a2a / DDP / TP
  communication, so they must not D2H-sync
  (.item/.cpu/.tolist/.numpy/python-compare on tensors), should avoid hook-time
  collectives unless correctness requires them, and must declare their full
  metric schema at registration before allocate_buffers.
  Use when adding or modifying any monitor, touching any _make_*_hook /
  register_hooks / _compute_* function, reviewing a PR that changes hot-path
  code or adds a metric, or diagnosing a perf regression that appears after
  enabling internal_medicine_monitors.
---

# Monitor Hook Performance Rules

> Mandatory discipline when adding or modifying any monitor (probe) in
> `internal_medicine`. The monitor hot path runs **inside forward hooks**, on
> training processes that also run expert-parallel a2a, DDP grad reduce, and TP
> communication. A single careless `.item()` can wipe out a 4x EP-overlap
> speedup. This skill exists because that already happened.

## Why This Exists

In a 2026-06 rollout we shipped a redesigned monitor that achieved a 4.8×
speedup on a 1.5B MoE training run (305s → 63s/iter). Re-enabling
`internal_medicine_monitors` regressed the run to 261s/iter. Root cause: every
hook was forcing GPU-CPU sync (`.item()` on scalar tensors) and queuing
`dist.all_reduce` / `dist.all_gather` in the hook itself, which serialized
parts of the hot path and contended with the training communication workload.

The fix was a "GPU-buffer" API: hooks record GPU 0-dim tensors into
pre-allocated accumulators, and a batched D2H happens during monitor `step()`
after the monitored forward. This skill encodes the rules that keep that
property intact.

## The Three Rules

### Rule 1 — No D2H sync on the hot path

The hot path is anything called from a hook (forward, pre-forward, backward).
Forbidden inside a hook:

- `.item()` (any tensor)
- `.cpu()` / `.tolist()` / `.numpy()`
- `int(tensor)`, `float(tensor)`, `bool(tensor)`
- `tensor.numel()` is fine; `tensor.unique().numel()` is **not** if it forces
  a host-side count
- Python comparisons that read a tensor value (`if tensor > 0:` triggers
  `.item()` implicitly)
- `tensor.tolist()` to feed something else
- `print(tensor)` and `f"{tensor}"` — both sync

What to do instead:

- Keep the value as a 0-dim GPU tensor and pass it to `record_layer_metric`
  / `record_mean` / `record_max` / `record_min`.
- The Megatron `TorchProbe` batches declared metrics into a single
  `torch.stack(...).cpu()` inside monitor `step()`. Hook code should not add
  another D2H site.

### Rule 2 — Avoid collectives on the hot path

Avoid inside a hook unless correctness requires it:

- `dist.all_reduce` / `dist.all_gather` / `dist.reduce_scatter`
- Anything that calls into NCCL on the default or DDP stream

The QK monitor previously had `dist.all_reduce` on 4 scalars + a per-head
`all_gather` per layer per microbatch. With EP=8 / 24 layers / 16 microbatches
that's ~3000 NCCL launches per step contending with a2a. We deleted them all.

What to do instead:

- For min/max metrics: `gather_and_aggregate()` at flush time produces a
  global min/max correctly without any per-hook collective.
- For mean metrics: `gather_and_aggregate()` averages the per-rank means.
  This is mathematically correct **only if every rank observes the same
  number of samples** (standard TP head-split is fine; EREC / unequal sharding
  is not). Document the assumption when you add a metric.
- If a collective is genuinely required for correctness (e.g. TP shards the
  channel dimension and the per-channel max needs to be across all ranks),
  keep it but justify it in a comment, and make it the smallest possible
  reduction (a local hidden-shard vector is OK; a per-token reduction is not).

The one collective we kept is in
`massive_activation_monitor._aggregate_per_channel_max` — TP shards the
channel dim, so a per-channel max reduction over the local hidden-shard vector
is required for correctness. Comment in code explains exactly why.

### Rule 3 — Schema is fixed at registration, hot path is checkless

The base API is:

1. `declare_layer_metric(layer_idx, name)` at hook-registration time —
   declares a metric and picks mean/max/min from class-level
   `MAX_AGGREGATED` / `MIN_AGGREGATED` sets.
2. `allocate_buffers(device)` — materializes 0-dim GPU tensors for every
   declared key. Locks the schema.
3. `record_layer_metric(layer_idx, name, val)` inside the hook — `val` is a
   **GPU 0-dim tensor**, never a Python float. Keep schema decisions out of
   the hook; the recording API is the only schema gate.

Forbidden in hooks:

- `if some_condition: declare_*(...)` — declares are not allowed after
  `allocate_buffers`.
- Lazy / per-batch key registration.
- Any `try / except` that catches and **silently** falls back to a CPU path.

What to do instead:

- Enumerate every key your metric will ever emit at `register_hooks` time.
- For metrics with dynamic suffixes (e.g. `channel_count_gt_{threshold}`),
  build the suffix list from `__init__` arguments and declare them all
  upfront. Add them to `self.MAX_AGGREGATED` so the right aggregation is
  picked.
- If you need static metadata (e.g. `router.num_experts` /
  `router.num_moe_experts`), read it as Python metadata from the module or pass
  it down from the calling site rather than running a tensor reduction plus
  `.item()` inside the hook to discover it.

## Multi-Chunk / VPP / Interleaved PP

`setup_*` functions get called with a list of model chunks (VPP /
interleaved 1F1B). The schema must be declared across **all** chunks before
`allocate_buffers` runs, otherwise `record_layer_metric` will hit an
undeclared key and KeyError at runtime.

Pattern: split `register_hooks` into three phases.

```python
def register_hooks(self, model):
    self._init_parallel_state()
    targets = self._prepare_layers(model)  # discover + declare; no allocate
    if not targets:
        return
    self.allocate_buffers(next(model.parameters()).device)
    self._attach_hooks(targets)

def _prepare_layers(self, model):
    layers = self._find_layers(model)
    for idx, _ in layers:
        for name in self._metric_names():
            self.declare_layer_metric(idx, name)
    return layers
```

And in `setup_<monitor>`:

```python
monitor._init_parallel_state()
chunk_targets = [(m, monitor._prepare_layers(m)) for m in models]
if any(targets for _, targets in chunk_targets):
    device = next((p.device for m in models for p in m.parameters()), None)
    assert device is not None, "no parameters across model chunks; cannot pick a device"
    monitor.allocate_buffers(device)
    for _, targets in chunk_targets:
        monitor._attach_hooks(targets)
```

This way every chunk's keys are declared before the schema is locked. The
single-model path (`register_hooks(model)`) still works because the helper
methods compose into the original behavior.

## Globals Are Derived At Flush Time, Not Per-Record

Don't write the same metric twice (per-layer + global) on the hot path. The
base monitor stores a `_layer_metric_groups: {global_key: (agg, [layer_keys])}`
map at declare time, and at flush time reduces across the per-layer
accumulators to produce the global value. That halves the kernel-launch rate
on the hot path (24 layers × 13 metrics × 16 microbatches = ~5k launches/step
becomes ~2.5k).

If you need a custom global aggregation that doesn't fit the built-in
count-weighted mean / max-of-maxes / min-of-mins shape, declare an extra
global key explicitly via `declare_max(global_key)` / `declare_mean(global_key)`
and write to it from the hook, but understand you've doubled the launches for
that metric.

## Verification Checklist

Before merging a new monitor or a hook change:

- [ ] `rg '\.item\(\)|\.cpu\(\)|\.tolist\(\)|\.numpy\(\)' src/paddlefleet/internal_medicine/backends/<backend>`
      returns nothing inside `_make_*_hook` / `_compute_*` paths, including
      helpers in `_metrics.py` that are called from hooks.
- [ ] `rg 'dist\.(all_reduce|all_gather|reduce_scatter)' src/paddlefleet/internal_medicine/backends/<backend>/<monitor>.py`
      returns either nothing, or a hit with a justification comment.
- [ ] Every key recorded by `record_layer_metric` was declared by
      `declare_layer_metric` at registration. Run with `verbose=True` and
      look for the GPU buffer budget log line — it reports declared key counts
      by aggregation type.
- [ ] `setup_<monitor>` correctly handles `models = [chunk0, chunk1]`. If you
      call `register_hooks` per chunk, you have a bug — split into prepare /
      allocate / attach phases.
- [ ] Run an end-to-end perf check with one monitor on at a time. Compare
      step time against a baseline with all monitors off. The delta must be
      <5% for a real-size MoE workload.

## Performance Budget

Empirically, a healthy single monitor on a 1.5B-MoE / EP=8 / GBS=4096 /
24-layer config costs ~1-2% of step time. The four-monitor stack
(qk_stats + moe_health + massive_act + ple_health) should stay under 5%. If you blow
that budget, bisect by enabling monitors one at a time and look for hidden
sync points.

## Code Anchors

- Backend-agnostic lifecycle: `src/paddlefleet/internal_medicine/core/base_monitor.py`
- Megatron GPU-buffer API: `src/paddlefleet/internal_medicine/backends/megatron/base.py`
- Reference monitor (clean hot path): `src/paddlefleet/internal_medicine/backends/megatron/qk_monitor.py`
- Justified collective: `src/paddlefleet/internal_medicine/backends/megatron/massive_activation_monitor.py:_aggregate_per_channel_max`
- Caller passes static metadata into a metric to avoid hot-path sync:
  `src/paddlefleet/internal_medicine/backends/megatron/moe_monitor.py:_compute_router_metrics`
  calling `compute_bias_affinity_jaccard(..., num_experts=...)`

## When This Skill Applies

Apply when:

- Adding a new monitor to any backend (megatron / paddlefleet / future).
- Touching any `_make_*_hook` / `register_hooks` / `_compute_*` function.
- Reviewing a PR that changes a hot path or adds a metric.
- A perf regression appears after enabling internal_medicine_monitors.

Do not apply when:

- The change is purely in the flush / aggregation / training_logs layer
  (cold path).
- You are adding tests or docs only.
