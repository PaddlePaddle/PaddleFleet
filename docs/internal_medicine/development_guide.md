# internal_medicine — development guide

> Probes / monitors for in-flight LLM training (Megatron + PaddleFleet
> backends). Megatron monitor work often runs from forward hooks and can enqueue
> CUDA/NCCL work that contends with expert-parallel a2a, DDP grad reduce, and TP
> communication.
>
> Code lives in `src/paddlefleet/internal_medicine/`, tests in
> `tests/single_card_tests/internal_medicine/`. The megatron-backend tests need
> torch and are listed in `tests/single_card_tests/disable_single_card_uts.txt`;
> run them in a megatron env with
> `INTERNAL_MEDICINE_TEST_BACKEND=megatron`.

## Skills

The `.agents/skills/` directory contains structured guides for tasks that
have non-obvious correctness or perf constraints. **Read the relevant
SKILL.md before touching code it covers.**

- **`monitor-hook-perf-rules/`** — Mandatory before adding or modifying
  any monitor (probe). Encodes the rules that prevent hooks from breaking
  overlap via hidden D2H syncs or unnecessary hook-time collectives. A 2026-06
  regression (63s → 261s/iter) is the reason this skill exists.

Commit gating is the repo-wide `pre-commit` suite (ruff at 80 columns, copyright
header, typos) plus the unit tests above.

## Boundaries

**ALWAYS** in monitor hot paths:
- Pass GPU 0-dim tensors to `record_*`. Never `.item()` / `.cpu()` /
  `.tolist()` inside a hook.
- Declare the full metric schema at `register_hooks` time before
  `allocate_buffers`.
- For multi-chunk models (VPP / interleaved 1F1B), use the three-phase
  setup pattern (prepare across all chunks → allocate → attach hooks).

**AVOID** in monitor hot paths unless explicitly justified:
- `dist.all_reduce` / `dist.all_gather` / `dist.reduce_scatter` from inside
  a hook unless correctness requires it. Defer cross-rank aggregation to flush
  time via `gather_and_aggregate` whenever possible. Exception: if a collective
  is genuinely required for correctness (e.g. TP shards a tensor dim), keep it
  but justify it in a comment and minimize its size.
- Lazy / per-batch declare. The schema is locked at `allocate_buffers`.
