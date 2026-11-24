---
status: todo
priority: P1
complexity: 3
readiness: 30%
agent:
---

# T25. DSpark drafts for MoE targets

Reuse T24's draft graph for (a) Qwen3.5-35B-A3B with `jinzex/Qwen3.5-35B-A3B-DSpark` @
`9aede6f56fa546dbf1f62fe9974d596620fae201` (block 8, taps 1, 8, 15, 23, 30, 37, confidence head) and (b) Inkling 2 Mini with `dspark-Inkling-2-Mini-Q8_0.gguf`
(`dflash` with Markov context, block 8, taps over unrolled visits 5, 23, 41, 48, softcap 30). Verify
uses the batched MoE pass of T14.

License (checked 24 November 2026): the jinzex head has **no license** (no card field, no README
text, no LICENSE file at `main`, `v1.0` or `v1.1`, no safetensors metadata). It is **not usable until
the author confirms a license**: local measurement only, not shipped, not published in converted
form, not a gate dependency. Fallback for (a): train a Qwen3.5-35B-A3B head with the same recipe in
T26's setup. The Inkling draft (b) is Apache-2.0 (same repo as the model).

On: Zig 0.17 and Metal (Apple Silicon, measured on the M3 Max 64 GB), no new runtime dependencies.

Components: `src/dspark.zig` (variants: confidence head on/off, Markov kinds, visit-indexed taps), `tools/convert.py`, `src/chain.zig`. Zig 0.17 + Metal.

Done means: tokens with the draft equal tokens without it on both models; accepted length and tok/s per prompt; Inkling at 14 GiB on the archive fixture compared with T10's Atomic run.

Check: `tern_bench.py --dspark` on both models reports equal tokens.

Depends on: [T14](../T14/task.md), [T24](../T24/task.md); (b) also [T21](../T21/task.md), [T22](../T22/task.md).

Design: [plan/moe-dspark.md](../moe-dspark.md). Token gate: see "Correctness gates" in the design; every speed claim has an interleaved before/after measurement.
