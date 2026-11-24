---
status: todo
priority: P0
complexity: 4
readiness: 50%
agent:
---

# T13. Resident MoE decode on Metal: router, top-k, expert GEMV

Run Qwen3.5-35B-A3B fully resident. Router GEMV, a top-k kernel (softmax, top 8 of 256 with
ties to the lower id, renormalise; sigmoid + bias variant for Inkling later) writing `ids` and
`weights` to a GPU buffer, an expert-indexed 4-bit affine GEMV that reads `ids` on the GPU and
indexes `base + id * stride` (no CPU readback, one command buffer per token kept), gate/up with
SwiGLU, down, weighted sum into the residual, shared expert with its sigmoid gate. The engine
gets a run-time-shaped MoE path; Bonsai keeps its comptime path.

On: Zig 0.17 and Metal (Apple Silicon, measured on the M3 Max 64 GB), no new runtime dependencies.

Components: `src/kernels.metal` (router/top-k, `gemv_a4_expert`), `src/engine.zig` (`encodeMoe`, layer kinds), `src/model.zig` (run-time shape), `src/metal.zig` if a new binding is needed, `src/main.zig` (model selection). Zig 0.17 + Metal.

Done means: greedy tokens equal MLX-LM (same 4-bit files) on the three bench prompts; `tern prof` shows the per-token time split (router, experts, rest) and the bytes read per token against the memory floor.

Check: `tern_bench.py --model models/qwen35-35b-a3b.tern` reports `same as MLX: True`; Bonsai tokens and speed unchanged.

Depends on: [T11](../T11/task.md), [T12](../T12/task.md).

Design: [plan/moe-dspark.md](../moe-dspark.md). Token gate: see "Correctness gates" in the design; every speed claim has an interleaved before/after measurement.
