---
status: todo
priority: P1
complexity: 5
readiness: 40%
agent:
---

# T15. Expert store and SSD streaming

Run an MoE whose experts do not fit the budget. New `src/expert.zig`: a fixed pool of 16 KiB-
aligned slots wrapped once with `newBufferWithBytesNoCopy`, a (layer, expert) -> slot table, a
reader thread pool. Implement both read paths and keep the faster on the M3 Max SSD: `pread` on a
separate fd with `F_NOCACHE` (+ `F_RDADVISE`), and `mmap` + `madvise(MADV_WILLNEED)` + copy. With
misses possible, split each MoE layer's command buffer after the router: GPU writes ids and
signals an `MTLSharedEvent`, CPU resolves hits and reads misses, signals a second event, the
expert GEMV waits. Fully resident runs must not start threads, reads or splits.

On: Zig 0.17 and Metal (Apple Silicon, measured on the M3 Max 64 GB), no new runtime dependencies.

Components: new `src/expert.zig` (SSD streaming layer), `src/metal.zig` (shared events, no-copy buffers), `src/engine.zig` (split points), `src/main.zig` (`--budget-mib`). Zig 0.17 + Metal; POSIX calls through `extern "c"` like `model.zig`.

Done means: Qwen3.5-35B-A3B runs at a 4 GiB budget with the same tokens as fully resident; a table of decode tok/s and bytes read at 4, 8, 12 GiB and resident for both read paths, with the chosen one justified.

Check: `tern gen --budget-mib 4096` token ids equal the resident run on the three prompts; `tern stats` prints hits, misses, bytes read, read latency percentiles.

Depends on: [T13](../T13/task.md); the budget numbers come from [T18](../T18/task.md) (a fixed slot count is enough to start).

Design: [plan/moe-dspark.md](../moe-dspark.md). Token gate: see "Correctness gates" in the design; every speed claim has an interleaved before/after measurement.
