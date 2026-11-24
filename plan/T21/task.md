---
status: todo
priority: P1
complexity: 5
readiness: 20%
agent:
---

# T21. Inkling 2 Mini: the looped architecture

Implement `inkling2_loop` in the engine: 24 physical layers with the core 6-17 run 3 times
(per-visit KV and conv state, shared weights, loop control `loop_*_ctrl_{down,forget,gain}` rank
64), sliding window 1024 on 5 of 6 layers, relative-position attention (`attn_rel*`, extent
1024, clips), short convolutions on q/k/v and after attention/FFN, attention-residual blocks of 6,
attention sinks, router with gate bias and scale 8, gate/up softcaps 4/25, final logit softcap
30. Follow llama.cpp's `inkling2-loop` graph as the reference.

Decision (creator, 24 Nov 2026): Inkling 2 Mini is the Atomic Engine comparison branch, not the first MoE target. Start after [T13](../T13/task.md) and [T14](../T14/task.md) work on Qwen3.5-35B-A3B; reuse the store, cache and budget of T15 to T19.

On: Zig 0.17 and Metal (Apple Silicon, measured on the M3 Max 64 GB), no new runtime dependencies.

Components: `src/engine.zig` (new layer kinds and visit schedule), `src/kernels.metal` (short conv, relative attention, softcaps), `src/model.zig` (description fields). Zig 0.17 + Metal.

Done means: greedy tokens equal Atomic Engine / llama.cpp without a draft on the three prompts and the archive fixture (reference ids from T10), fully resident.

Check: `tern_bench.py --model models/inkling2-mini-iq2m.tern --ref atomic` reports equal tokens.

Depends on: [T10](../T10/task.md) (reference ids), [T20](../T20/task.md).

Design: [plan/moe-dspark.md](../moe-dspark.md). Token gate: see "Correctness gates" in the design; every speed claim has an interleaved before/after measurement.
