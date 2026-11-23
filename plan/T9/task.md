---
status: todo
priority: P0
complexity: 5
readiness: 0%
agent:
---

# T9. Windows at >= 2x llama.cpp, every benchmark

Ship tern as a native Windows binary (CUDA, Vulkan, CPU; no WSL-only release) and make it **`>= 2x`** as fast as the Windows baseline on **every** harness metric and prompt, with greedy tokens equal to the reference.

**Baseline.** llama.cpp, PrismML fork, on the fastest backend the machine has (CUDA if NVIDIA, else HIP, else Vulkan, else CPU). Same rationale as [T8](../T8/task.md); details and rejected alternatives in [docs/cross-platform-plan.md](../../docs/cross-platform-plan.md). There are no tern Windows numbers yet.

Depends on T6. File mapping and timing must work with the Windows APIs (or Zig's wrappers), not POSIX-only `mmap` / `clock_gettime_nsec_np`.

Done means: on a named Windows machine, an interleaved `bench/compare.py` run against that llama.cpp build shows tern `>= 2x` on `decode_tps` and `prefill_tps`, and `<= 1/2` the baseline `ttft_s`, for `free_text`, `code_edit` and `doc_qa`, tokens matching the MLX-LM reference. CUDA, Vulkan and CPU each at least pass the token check where hardware exists; the `>= 2x` bar is fastest tern vs fastest llama.cpp on that machine.

Check: a results JSON committed under `bench/results/` (Windows, machine and backend named) and a REPORT.md section that quotes it, with every prompt/metric meeting `>= 2x` and `matches_mlx` true.
