---
status: todo
priority: P0
complexity: 4
readiness: 50%
agent:
---

# T24. DSpark block draft for Bonsai (prism head)

Run the pretrained DSpark head from `prism-ml/Ternary-Bonsai-27B-gguf` @
`86e89f34c93201c3dfd5e5880fedb0022fc7e34d` (`Ternary-Bonsai-27B-dspark-Q4_1.gguf`; bf16 as the
reference). Part a: convert to `.tern` (Q4_1 as 4-bit affine group 32), share embeddings and output
head with the target. Part b: the draft forward: taps from target layers 1, 16, 31, 46, 61 (written
by the target pass, per-source norm), `dspark.fc`, 6 layers over a block of mask tokens (248319),
log-SNR conditioning, Markov head (rank 256, chained in the block), confidence head to cut the pass,
draft cache rollback; wired into `src/chain.zig` and gated by `draft.Policy`/`Gate`. Block size 4 is
what the head was trained for; 7-8 needs [T26](../T26/task.md) or the z-lab DFlash head (measure it
as an alternative). First confirm the head matches the MLX 2-bit weights tern uses (same model,
different files).

On: Zig 0.17 and Metal (Apple Silicon, measured on the M3 Max 64 GB), no new runtime dependencies.

Components: `tools/convert.py`, new `src/dspark.zig`, `src/kernels.metal` (Q4_1 GEMV/GEMM, Markov head), `src/engine.zig` (hidden-state taps), `src/chain.zig`, `src/draft.zig`, `src/model.zig`. Zig 0.17 + Metal.

Done means: draft logits match a reference forward (bf16 GGUF in llama.cpp PrismML fork or Atomic Engine) within tolerance; accepted length per prompt vs the card's tau ~3.7 at k = 4; greedy tokens with the draft equal tokens without it and MLX-LM; decode tok/s on the three prompts vs plain and vs `--draft` (0.8B).

Check: `tern_bench.py --dspark` `same as MLX: True`; `tern prof --dspark` prints accepted length and pass price.

Depends on: [T23](../T23/task.md) (to pay off), [T11](../T11/task.md) for the new dtype; uses the [T3](../T3/task.md) machinery.

Design: [plan/moe-dspark.md](../moe-dspark.md). Token gate: see "Correctness gates" in the design; every speed claim has an interleaved before/after measurement.
