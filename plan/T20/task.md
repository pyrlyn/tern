---
status: todo
priority: P1
complexity: 4
readiness: 30%
agent:
---

# T20. Inkling 2 Mini: GGUF conversion and its weight formats

Convert `AtomicChat/Inkling-2-Mini-GGUF` @ `309509848b3151419aa26d069eb03db57fd29a00`
`Inkling-2-Mini-AD-IQ2_M.gguf` (and its `dspark-Inkling-2-Mini-Q8_0.gguf`) to `.tern` v2 without
requantization: split `ffn_gate_up_exps` (`IQ2_XXS`, `[2304, 2304, 128]`) and `ffn_down_exps`
(`Q2_0` g64, type 42) into per-expert slabs, keep `Q8_0`, `Q4_K`, `Q6_K` blocks byte-for-byte. Add
Metal GEMV kernels for `IQ2_XXS`, `Q2_0` g64, `Q8_0`, `Q4_K`, `Q6_K` (expert-indexed where they hold
experts). Porting ggml's dequantisation tables/kernels is allowed with the MIT notice.

Decision (creator, 24 Nov 2026): Inkling 2 Mini is the Atomic Engine comparison branch, not the first MoE target. Start after [T13](../T13/task.md) and [T14](../T14/task.md) work on Qwen3.5-35B-A3B; reuse the store, cache and budget of T15 to T19.

On: Zig 0.17 and Metal (Apple Silicon, measured on the M3 Max 64 GB), no new runtime dependencies.

Components: `tools/convert.py` (GGUF reader), `src/kernels.metal` (5 formats), `src/model.zig` (dtypes). Zig 0.17 + Metal.

Done means: each kernel matches a reference dequantise-and-multiply (numpy or ggml) to the bit on random rows; the converter verifies slabs against GGUF bytes.

Check: `zig build test` runs the kernel checks on a small fixture; `python tools/convert.py --gguf ... --verify` passes.

Depends on: [T11](../T11/task.md), [T13](../T13/task.md).

Design: [plan/moe-dspark.md](../moe-dspark.md). Token gate: see "Correctness gates" in the design; every speed claim has an interleaved before/after measurement.
