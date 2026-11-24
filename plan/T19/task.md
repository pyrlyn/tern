---
status: todo
priority: P2
complexity: 3
readiness: 30%
agent:
---

# T19. Live budget shrink and grow

Change the cap during a session without losing it. At a token boundary, shrink drops expert slots
in policy order (Inkling hot core last), never fixed weights or session state; a cap below the
floor is refused and the session goes on. Grow refills lazily. A control path (stdin command or
signal in `tern gen`/server) sets the cap.

On: Zig 0.17 and Metal (Apple Silicon, measured on the M3 Max 64 GB), no new runtime dependencies.

Components: `src/budget.zig`, `src/expert.zig`, `src/main.zig`. Zig 0.17.

Done means: a run that shrinks from 16 to 4 GiB and back mid-generation gives the same tokens as a run without changes; memory returned is visible in the ledger and in `vm_stat`/`footprint`.

Check: scripted test in `bench/` that changes the cap twice during `tern gen` and compares token ids.

Depends on: [T16](../T16/task.md), [T18](../T18/task.md).

Design: [plan/moe-dspark.md](../moe-dspark.md). Token gate: see "Correctness gates" in the design; every speed claim has an interleaved before/after measurement.
