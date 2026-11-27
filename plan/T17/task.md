---
status: todo
priority: P2
complexity: 3
readiness: 30%
agent:
---

# T17. Next-layer expert prefetch

Hide SSD latency: while layer L runs, apply layer L+1's router to the current hidden state on the
GPU to predict its experts and issue reads for predicted misses, bounded (at most 8 per layer,
same admission as demand, never evicting experts predicted for L). Measure prediction accuracy
from the traces first. For Inkling, after layer 17 in passes 1 and 2 the next layer is 6.

On: Zig 0.17 and Metal (Apple Silicon, measured on the M3 Max 64 GB), no new runtime dependencies.

Components: `src/kernels.metal` (preview router GEMV), `src/expert.zig` (prefetch queue), `src/engine.zig`. Zig 0.17 + Metal.

Done means: prediction hit rate per prompt; decode tok/s at 4 and 8 GiB with and without prefetch; no change when fully resident; tokens unchanged.

Check: `tern gen --budget-mib 4096 --prefetch` faster than without in an interleaved run, same token ids.

Depends on: [T15](../T15/task.md), [T16](../T16/task.md).

Design: [plan/moe-dspark.md](../moe-dspark.md). Token gate: see "Correctness gates" in the design; every speed claim has an interleaved before/after measurement.
