---
status: todo
priority: P0
complexity: 4
readiness: 40%
agent:
---

# T23. Block verify for 4 to 8 token passes (Bonsai)

Make a 4-8 token verification pass cheap enough that a block draft pays on Apple Silicon (the prism
card says it does not yet). Today a pass of 4 is 2.1 and a pass of 8 is 3.5 plain tokens (quiet
machine). Target: 4 at <= 1.4, 8 at <= 2.0. A dedicated n = 4/8 ternary kernel that unpacks each
group once and reuses it across tokens, half activations with f32 sums, simdgroup matrix
intrinsics on the unpacked tile; the agent chooses and measures. Same accept/rollback as T3.

On: Zig 0.17 and Metal (Apple Silicon, measured on the M3 Max 64 GB), no new runtime dependencies.

Components: `src/kernels.metal` (new `gemm_t2_n8`-class kernel), `src/engine.zig` (`encodeVerify` selection), `src/draft.zig` (pass prices). Zig 0.17 + Metal.

Done means: measured pass prices for 2, 4, 8 in plain tokens before/after (quiet and loaded); tokens unchanged; rejected variants added to REPORT.md.

Check: `tern prof --draft` prints pass prices; `tern_bench.py` `same as MLX: True`.

Depends on: none (extends [T3](../T3/task.md)).

Design: [plan/moe-dspark.md](../moe-dspark.md). Token gate: see "Correctness gates" in the design; every speed claim has an interleaved before/after measurement.
