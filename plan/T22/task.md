---
status: todo
priority: P2
complexity: 2
readiness: 40%
agent:
---

# T22. Hot-core residency for looped models

For looped models, spend spare budget on the repeated core first (Inkling layers 6-17), then the
suffix, then the prefix; shrink releases in reverse. Generic over the loop description in the
file, no Inkling special case in the cache code.

Decision (creator, 24 Nov 2026): Inkling 2 Mini is the Atomic Engine comparison branch, not the first MoE target. Start after [T13](../T13/task.md) and [T14](../T14/task.md) work on Qwen3.5-35B-A3B; reuse the store, cache and budget of T15 to T19.

On: Zig 0.17 and Metal (Apple Silicon, measured on the M3 Max 64 GB), no new runtime dependencies.

Components: `src/expert.zig` (residency classes), `src/budget.zig`. Zig 0.17.

Done means: Inkling at 14 GiB shows zero expert misses after warm-up (Atomic's result); misses and tok/s at 6, 8, 10, 14 GiB with and without the priority.

Check: `tern stats` after a 14 GiB run reports 0 misses in decode; tokens unchanged.

Depends on: [T16](../T16/task.md), [T18](../T18/task.md), [T21](../T21/task.md).

Design: [plan/moe-dspark.md](../moe-dspark.md). Token gate: see "Correctness gates" in the design; every speed claim has an interleaved before/after measurement.
