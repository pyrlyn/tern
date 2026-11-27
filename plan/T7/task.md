---
status: todo
priority: P0
complexity: 5
readiness: 40%
agent:
---

# T7. macOS Metal at >= 2x the fastest baseline, every benchmark

Raise the existing Metal backend so that, on Apple Silicon, tern is **`>= 2x`** as fast as the **fastest macOS baseline** on **every** harness metric and prompt, with greedy tokens unchanged.

This is the macOS instance of the global bar in [docs/cross-platform-plan.md](../../docs/cross-platform-plan.md). It builds on [T3](../T3/task.md) (speculation and the draft model), [T4](../T4/task.md) (prefill) and [T5](../T5/task.md) (fused small kernels), and it needs a quiet-machine table from [T1](../T1/task.md).

**Baseline.** Composite of llama.cpp Metal (PrismML fork, same PQ2_0 GGUF as REPORT.md) and MLX-LM (same MLX 2-bit weights). For each of `decode_tps`, `prefill_tps` and `ttft_s`, on each of `free_text`, `code_edit` and `doc_qa`, take the **better** of the two engines in the same interleaved `bench/compare.py` run. tern must be `>= 2x` that value: at least twice the tok/s for decode and prefill, at most half the seconds for TTFT. Current snapshot (not a quiet run): the fourth-run table in [REPORT.md](../../REPORT.md). Re-measure after T1 and quote the new file.

Today only code_edit decode clears `>= 2x`. Long-prompt prefill and TTFT are behind llama.cpp; free_text and doc_qa decode are not at `>= 2x`.

**Atomic Engine.** A second macOS goal is to be twice as fast as [Atomic Engine](https://github.com/AtomicBot-ai/atomic-engine), a llama.cpp fork with SSD expert streaming. Their published figure is 105.71 tokens/s on a 48 GB M4 Max (Inkling 2 Mini IQ2_M with a DSpark draft model, 14 GiB budget). That model is not Ternary Bonsai 27B, so this is a named throughput target alongside the composite baseline above, not a replacement for the harness check.

Done means: a quiet interleaved `bench/compare.py --rounds 5` (load average below 4, T1 rule) shows tern `>= 2x` the composite baseline on decode, prefill and TTFT for every prompt, and `matches_mlx` is true in every run, with and without `--draft` as shipped by default.

Check: `bench/results/compare.json` (or the quiet-run successor named in REPORT.md) has tern at `>= 2x` vs `max(llama.cpp, MLX-LM)` on `decode_tps` and `prefill_tps`, and at `<= 1/2` of the faster engine's `ttft_s`, for `free_text`, `code_edit` and `doc_qa`, with `matches_mlx` true.
