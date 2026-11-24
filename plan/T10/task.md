---
status: todo
priority: P0
complexity: 2
readiness: 70%
agent:
---

# T10. Atomic Engine baselines on the M3 Max

Add Atomic Engine as a third engine in the interleaved harness and measure the two pairs the 2x
goal is gated on, on the same M3 Max 64 GB as every other tern number. Build Atomic Engine
(`AtomicBot-ai/atomic-engine`, branch `engine-development`, pin the commit) with the cmake line from
its README. Pair 1: Inkling 2 Mini `AD-IQ2_M` + `dspark-Inkling-2-Mini-Q8_0.gguf`, the README flags
(14436 MiB budget, `-c 4096`, q4_0 KV, `--spec-type draft-dspark`), on Atomic's archive fixture
(`tools/expert-source/bench-inkling-metal.py`) and on tern's three prompts. Pair 2: check whether
Atomic Engine loads `Ternary-Bonsai-27B-Q2_0.gguf` with `Ternary-Bonsai-27B-dspark-Q4_1.gguf`
(`--spec-type draft-dspark`); if it does, measure it on the three prompts; if not, record why.
Also record Atomic's greedy tokens without the draft as the Inkling token reference.

On: Zig 0.17 and Metal (Apple Silicon, measured on the M3 Max 64 GB), no new runtime dependencies.

Components: `bench/compare.py`, a new `bench/atomic_bench.py`, `bench/results/`, REPORT.md. No engine code.

Done means: `bench/results/atomic.json` with decode tok/s, prefill tok/s, TTFT and output token ids per prompt for both pairs, load average recorded (T1 rule), and a REPORT.md section quoting it next to the published 105.71 tok/s (M4 Max).

Check: `python bench/compare.py --engines tern,atomic --rounds 3` runs end to end; the JSON has both pairs or a written reason pair 2 is unsupported.

Depends on: none (quiet-machine rule from [T1](../T1/task.md)).

Design: [plan/moe-dspark.md](../moe-dspark.md). Token gate: see "Correctness gates" in the design; every speed claim has an interleaved before/after measurement.
