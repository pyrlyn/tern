---
status: todo
priority: P2
complexity: 4
readiness: 10%
agent:
---

# T26. A 7-8 token block draft for Bonsai: z-lab DFlash first, then a block-8 head (approved)

Approved by the creator on 24 November 2026, as planned. The brief asks for 7-8 tokens per draft
pass; the shipped Bonsai head proposes 4. Step 1: measure the pretrained `z-lab/Qwen3.5-27B-DFlash`
@ `25ee0025ff950496a634e100b75c2db4515e9824` (block 16, trained on bf16 Qwen3.5-27B) on Bonsai in
tern with T24's draft graph: acceptance and tok/s at blocks 8 and 16 on the three prompts. Step 2,
only if step 1 is insufficient (it does not beat the prism block-4 head in tok/s): train our own
block-8 head. Model Optimizer DSpark recipe (PR #2340) or z-lab `dflash`, initialised from the prism
head, target taps from `prism-ml/Ternary-Bonsai-27B-unpacked` (bf16, ~54 GB), self-distilled data
from Ternary Bonsai 27B on an open instruction set, on rented 8 x H100-class GPUs. A pilot run comes
first and fixes the GPU-hour and cost estimate before the full run.

On: Zig 0.17 and Metal (Apple Silicon, measured on the M3 Max 64 GB), no new runtime dependencies.

Components: training scripts outside the engine (`tools/train_dspark/`, Python), then T24's loader. No engine code unless the head format differs.

Done means: the z-lab measurement is in REPORT.md; if training was needed, the pilot numbers (GPU-hours, cost, data) are recorded, then a head whose accepted length at block 8 beats the block-4 head in tern's tok/s on the three prompts.

Check: `tern_bench.py --dspark --head zlab` numbers recorded; `tern_bench.py --dspark --block 8` faster than block 4, tokens equal.

Depends on: [T24](../T24/task.md).

Design: [plan/moe-dspark.md](../moe-dspark.md). Token gate: see "Correctness gates" in the design; every speed claim has an interleaved before/after measurement.
