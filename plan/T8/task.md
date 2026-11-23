---
status: todo
priority: P0
complexity: 5
readiness: 0%
agent:
---

# T8. Linux at >= 2x llama.cpp, every benchmark

Ship tern on Linux (CUDA, Vulkan, CPU) and make it **`>= 2x`** as fast as the Linux baseline on **every** harness metric and prompt, with greedy tokens equal to the reference.

**Baseline.** llama.cpp, PrismML fork, on the fastest backend the machine has (CUDA if NVIDIA, else HIP/ROCm if that is the fast path, else Vulkan, else CPU). See [docs/cross-platform-plan.md](../../docs/cross-platform-plan.md) for why no faster realistic runtime for this 2-bit model was adopted, and what remains unverified. There are no tern Linux numbers yet.

Depends on the T6 backend interface (design and implementation PRs). Token check first, then speed. Default shipped flags must satisfy the bar, not a hidden profile.

Done means: on a named Linux machine, an interleaved `bench/compare.py` run against that llama.cpp build shows tern `>= 2x` on `decode_tps` and `prefill_tps`, and `<= 1/2` the baseline `ttft_s`, for `free_text`, `code_edit` and `doc_qa`, with the same greedy tokens as the MLX-LM reference (`matches_mlx` true). Load is recorded. CUDA, Vulkan and CPU each at least pass the token check on hardware that has them; the `>= 2x` bar applies to the fastest tern backend vs the fastest llama.cpp backend on that machine.

Check: a results JSON committed under `bench/results/` (Linux, machine and backend named) and a REPORT.md section that quotes it, with every prompt/metric meeting `>= 2x` and `matches_mlx` true.
