---
status: todo
priority: P1
complexity: 3
readiness: 50%
agent:
---

# T18. One memory budget (ledger and admission)

One cap for everything: fixed weights, expert slots, KV cache, GDN states, scratch, the draft model
(weights, its cache, its scratch), I/O staging and a fixed envelope. Start-up admits only if the
floor (fixed + envelope + one slab per MoE layer + staging) fits, and otherwise refuses with the
numbers. The expert pool gets what is left. Works for Bonsai too (cap only, no streaming).

On: Zig 0.17 and Metal (Apple Silicon, measured on the M3 Max 64 GB), no new runtime dependencies.

Components: new `src/budget.zig`, `src/engine.zig` and `src/expert.zig` (charge every allocation), `src/main.zig` (`--budget-mib`, `tern stats`). Zig 0.17.

Done means: `tern stats` prints the ledger and the sum matches Metal's `currentAllocatedSize` within the envelope; an impossible cap is refused at start with a clear message.

Check: unit tests for admission arithmetic in `zig build test`; ledger vs `currentAllocatedSize` printed for Bonsai and Qwen3.5-35B-A3B.

Depends on: none to start; integrates with [T15](../T15/task.md).

Design: [plan/moe-dspark.md](../moe-dspark.md). Token gate: see "Correctness gates" in the design; every speed claim has an interleaved before/after measurement.
