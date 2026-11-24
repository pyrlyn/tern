---
status: todo
priority: P2
complexity: 4
readiness: 10%
agent:
---

# T26. A block-8 DSpark head for Bonsai (training, needs the creator's go)

The brief asks for 7-8 tokens per draft pass; the shipped Bonsai head proposes 4. Train a block-8
head only after the creator approves the cost and data. Plan: Model Optimizer DSpark recipe (PR
#2340) or z-lab `dflash`, initialised from the prism head, target taps from
`prism-ml/Ternary-Bonsai-27B-unpacked` (bf16, ~54 GB), self-distilled data from Ternary Bonsai 27B
on an open instruction set, on rented 8 x H100-class GPUs; a pilot first to fix the GPU-hour
estimate. Before training: measure the z-lab Qwen3.5-27B DFlash head (block 16) on Bonsai; if its
acceptance at 8 is good enough, training is not needed. Nothing here runs without approval.

On: Zig 0.17 and Metal (Apple Silicon, measured on the M3 Max 64 GB), no new runtime dependencies.

Components: training scripts outside the engine (`tools/train_dspark/`, Python), then T24's loader. No engine code unless the head format differs.

Done means: a written proposal with pilot numbers (GPU-hours, cost, data) approved by the creator; then a head whose accepted length at block 8 beats the block-4 head in tern's tok/s on the three prompts.

Check: the approval is recorded in this task; `tern_bench.py --dspark --block 8` faster than block 4, tokens equal.

Depends on: [T24](../T24/task.md); the creator's go.

Design: [plan/moe-dspark.md](../moe-dspark.md). Token gate: see "Correctness gates" in the design; every speed claim has an interleaved before/after measurement.
