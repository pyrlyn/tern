---
status: todo
priority: P0
complexity: 2
readiness: 10%
agent:
---

# T27. Gate: 2x Atomic Engine

The final measurement. On the M3 Max, interleaved with Atomic Engine (T10's harness, quiet
machine): Inkling 2 Mini `AD-IQ2_M` + DSpark at a 14 GiB budget, and Ternary Bonsai 27B + DSpark
(if Atomic Engine runs pair 2). tern must be at least 2x Atomic Engine's decode tok/s on the
archive fixture and on the three bench prompts, with greedy tokens equal to the reference and
default flags. Report the published 105.71 tok/s (M4 Max) and its 2x (211 tok/s) alongside.

On: Zig 0.17 and Metal (Apple Silicon, measured on the M3 Max 64 GB), no new runtime dependencies.

Components: `bench/compare.py`, `bench/results/`, REPORT.md, README.md goal line. No engine code.

Done means: `>= 2x` Atomic Engine on decode for every prompt of both pairs, tokens equal; otherwise the measured gap and the next lever are written in REPORT.md.

Check: `bench/results/atomic_gate.json` with `tern_tps / atomic_tps >= 2.0` per prompt and token equality true.

Depends on: [T10](../T10/task.md), [T22](../T22/task.md), [T23](../T23/task.md), [T24](../T24/task.md), [T25](../T25/task.md).

Design: [plan/moe-dspark.md](../moe-dspark.md). Token gate: see "Correctness gates" in the design; every speed claim has an interleaved before/after measurement.
