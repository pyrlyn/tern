---
status: todo
priority: P1
complexity: 3
readiness: 40%
agent:
---

# T16. Expert cache policy: LRU and ClassLRU, with trace replay

Give the store a replacement policy. LRU as default; ClassLRU (decayed use counts, victims from the
cold class first, LRU inside a class) as an option. A trace writer records (token, layer, ids);
a replay tool computes hit rate and bytes read for any policy and slot count without running the
model. A policy becomes default only if it lowers wall time, not just bytes (Atomic kept LRU
default after ClassLRU saved 3.55% of bytes on one trace).

On: Zig 0.17 and Metal (Apple Silicon, measured on the M3 Max 64 GB), no new runtime dependencies.

Components: `src/expert.zig` (LRU cache, policies), `tern trace` output, new `bench/expert_replay.py`. Zig 0.17.

Done means: traces for the three prompts on Qwen3.5-35B-A3B; a replay table (policy x 4/8/12 GiB) and wall-time runs for the two best; tokens unchanged.

Check: `python bench/expert_replay.py bench/results/traces/*.bin` prints the table; `tern gen --cache classlru` tokens equal `--cache lru`.

Depends on: [T15](../T15/task.md).

Design: [plan/moe-dspark.md](../moe-dspark.md). Token gate: see "Correctness gates" in the design; every speed claim has an interleaved before/after measurement.
