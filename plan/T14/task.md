---
status: todo
priority: P1
complexity: 4
readiness: 40%
agent:
---

# T14. Batched MoE pass for prefill and verify

For a pass of n tokens, bucket the n x 8 (token, expert) pairs by expert on the GPU (count,
prefix sum, scatter) and run one tiled GEMM per used expert over its tokens, then gather back.
Each used expert is read once per pass. This is what makes a 7-8 token verify affordable on an
MoE target, and it is the MoE prefill path.

On: Zig 0.17 and Metal (Apple Silicon, measured on the M3 Max 64 GB), no new runtime dependencies.

Components: `src/kernels.metal` (bucket/scatter/gather, `gemm_a4_expert` tile), `src/engine.zig` (`encodeBlock`/`encodeVerify` for MoE). Zig 0.17 + Metal.

Done means: tokens of prefill and of speculative passes equal the token-by-token run; a verify pass of 8 on Qwen3.5-35B-A3B costs at most 2.5 plain tokens (measured, quiet machine), and prefill is reported against MLX-LM.

Check: `tern_bench.py` `same as MLX: True` with passes on; `tern prof --pass 8` prints the price in plain tokens.

Depends on: [T13](../T13/task.md).

Design: [plan/moe-dspark.md](../moe-dspark.md). Token gate: see "Correctness gates" in the design; every speed claim has an interleaved before/after measurement.
