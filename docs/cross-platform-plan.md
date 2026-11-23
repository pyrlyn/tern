# Cross-platform plan

Make tern a realistic alternative to the fastest local runtime on **macOS, Linux and Windows** for single-user greedy generation of Ternary Bonsai 27B (and the Qwen3.5-0.8B draft). This document is the product and measurement plan: platforms, baselines, the `>= 2x` bar, task breakdown, phases and the PR-only workflow.

The compile-time backend interface, memory model, Zig signature sketch and reuse / rewrite analysis are **not** in this document. They live in a separate PR, [pyrlyn/tern#8](https://github.com/pyrlyn/tern/pull/8) (design file `docs/backend-abstraction.md`, task [T6](../plan/T6/task.md)).

## Target

On **each** of macOS, Linux and Windows, on equivalent hardware, tern must be **`>= 2x`** as fast as that platform's **fastest realistic baseline** on **every** benchmark in the existing harness: decode tok/s, prefill tok/s, and time to first token, on every prompt class (`free_text`, `code_edit`, `doc_qa`). Greedy tokens stay equal to MLX-LM (and to llama.cpp's text) on those prompts.

Wording for this repository: write **`>= 2x`** in every PR description, overview document and task acceptance criterion that talks about the speed bar. Do not write "+40%", "40% faster" or "2x or better" as the official phrasing.

For throughput (`decode_tps`, `prefill_tps`), `>= 2x` means at least twice the baseline tok/s. For TTFT, `>= 2x` means at most **half** the baseline seconds (twice as fast). `matches_mlx` remains true.

The previous goal in [README.md](../README.md) / [REPORT.md](../REPORT.md) was +40% vs llama.cpp and MLX-LM, and only some prompts cleared it. This plan replaces that bar. Until T1 produces a quiet-machine run, reported gaps use the existing interleaved figures and must be re-measured.

This is a hard bar. On Apple Silicon, plain decode is already at the memory floor ([REPORT.md](../REPORT.md)): kernels alone cannot deliver `>= 2x` decode. Prefill on the current Metal path is still behind llama.cpp on the long prompts. On discrete NVIDIA GPUs, llama.cpp's CUDA path is mature. Treat `>= 2x` on all metrics as the acceptance criterion, and record honestly if a metric cannot be reached with a measured reason.

## Workflow

- All work goes through pull requests against `main`. No direct commits to `main`.
- Do not merge a PR unless the creator has explicitly asked to merge it **and** CI is green.
- Do not merge on green CI alone.
- Docs, plan tasks and code follow the same rule.
- User-facing files and docs stay in English.

## Platform baselines

A baseline has to (1) actually run this model (Ternary Bonsai 27B, 2-bit ternary, Qwen3.5 hybrid attention / gated delta net; the same weights tern uses), (2) be realistic for **single-user** local decode and prefill, (3) be the **fastest** such runtime on that OS, on equivalent hardware. Server engines measured at high concurrency are not a substitute. Ollama is llama.cpp underneath and is not a separate baseline.

tern's own measurements use the PrismML llama.cpp fork (PQ2_0 GGUF) and MLX-LM on the MLX 2-bit file, converted to `.tern` without changing weights ([README.md](../README.md), [REPORT.md](../REPORT.md)). Vendor material about the later **Bonsai 2** family (Hadamard rotation, PTQ1_0, etc.) is cited only for ecosystem facts (who ships binaries, which GGUF types exist). Acceptance runs stay on the model and files this repo already benches, unless the creator later switches the product model.

### macOS (Apple Silicon, Metal)

**Chosen baseline (strictest reading of "the faster"):** a **composite** of llama.cpp Metal (PrismML fork, PQ2_0) and MLX-LM. For each metric and each prompt, the baseline number is the **better** of the two engines on the same interleaved harness. tern must be `>= 2x` that number.

Neither engine wins every column in the in-repo fourth run (`bench/results/compare_pass2.json`, [REPORT.md](../REPORT.md), 3 rounds, medians, load 15.7–88.6, GPU not quiet — T1 still open):

| Prompt | Metric | llama.cpp Metal | MLX-LM | Faster baseline |
| --- | --- | --- | --- | --- |
| free_text | decode, tok/s | 22.0 | **24.9** | MLX-LM |
| | prefill, tok/s | **104.1** | 77.1 | llama.cpp |
| | TTFT, s (lower is better) | **0.37** | 0.49 | llama.cpp |
| code_edit | decode, tok/s | 22.3 | **23.5** | MLX-LM |
| | prefill, tok/s | **169.2** | 132.0 | llama.cpp |
| | TTFT, s | **2.52** | 3.23 | llama.cpp |
| doc_qa | decode, tok/s | 22.6 | **24.8** | MLX-LM |
| | prefill, tok/s | **167.9** | 147.0 | llama.cpp |
| | TTFT, s | **2.52** | 2.88 | llama.cpp |

So today tern is already `>= 2x` only on **decode** for **code_edit** (62.3 tok/s vs 23.5). It is not `>= 2x` on decode for free_text or doc_qa, and it is **behind** the faster baseline on long-prompt prefill and TTFT (117.2 / 116.1 tok/s and 3.64 s vs llama.cpp 169.2 / 167.9 tok/s and 2.52 s).

**Why this pair, and not someone else.** They are the only two engines this repo has actually run on this model with a token check. MLX is the native Apple stack; llama.cpp Metal is the portable GGUF stack. PrismML's product docs say decode on an M4 Pro was slightly faster in llama.cpp than MLX for **Bonsai 2**; that is a different machine and a different generation, so it does not override REPORT.md for this model ([Formats & Runtime Support](https://docs.prismml.com/download/formats)).

**Why not a single named engine.** Picking only MLX-LM would weaken the prefill/TTFT bar; picking only llama.cpp would weaken the decode bar. The composite is the fastest baseline on each benchmark.

**Unverified / to re-measure:** all REPORT.md figures were taken with load averages far above 4; T1 must replace the table. Absolute "fast-state" decode around 38–39 tok/s for tern and MLX-LM is a single-run indication, not a median. No comparison with MLC LLM or other Metal-adjacent runtimes on this checkpoint.

### Linux

**Chosen baseline:** llama.cpp, **PrismML fork**, on the **fastest backend the machine has** (CUDA on NVIDIA, otherwise HIP/ROCm on supported AMD, otherwise Vulkan, otherwise CPU). Compare against that backend, not against a CPU build when a GPU is present. Official wording: `>= 2x` vs this llama.cpp.

**Why.** It is the only widely shipped runtime that is documented to load this class of ternary Bonsai GGUF (PQ2_0 / related fork types on the PrismML binaries; upstream llama.cpp `b11555`+ can load official `Q2_0` with the Hadamard/Bonsai 2 path, which is a related but not automatically identical file to the one REPORT.md used). PrismML ships Linux x64 builds for CPU, CUDA 12.4/12.8/13.3, Vulkan, ROCm 7.2, and SYCL ([formats doc](https://docs.prismml.com/download/formats), [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp/), [Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo)). llama.cpp's own backend set is CPU (AVX2/AVX-512/NEON), CUDA, HIP, Vulkan, SYCL, Metal ([build.md](https://github.com/ggml-org/llama.cpp/blob/HEAD/docs/build.md), [feature matrix](https://github.com/ggml-org/llama.cpp/wiki/Feature-matrix)). CUDA and ROCm are generally faster than Vulkan on the same GPU; Vulkan is the cross-vendor fallback.

**Alternatives checked, not used as the bar:**

- **vLLM, SGLang, TensorRT-LLM** — strong Qwen3.5 / GDN support on NVIDIA (and some ROCm) at **serving** concurrency; no evidence they load Ternary Bonsai 27B 2-bit / PQ2_0. Throughput numbers at hundreds of concurrent sequences are the wrong workload. Linux-first; not a single-user baseline for this checkpoint.
- **ExLlamaV2/V3** — EXL2/EXL3 (and similar) on NVIDIA, not this GGUF packing.
- **MLC LLM** — no evidence of this 2-bit packing.
- **llamafile** — llama.cpp packaged; not a faster engine.
- **Ollama** — llama.cpp.
- **ik_llama.cpp** — sometimes faster than mainline on other models (especially some CPU / MoE prefill cases), sometimes slower; no evidence it reads PQ2_0 / this Bonsai file. Not a valid baseline until it runs the same weights and wins a measured head-to-head.
- **bitnet.cpp** — Microsoft BitNet b1.58 and a small set of similar ternary models, not Qwen3.5-hybrid Bonsai 27B.

**Unverified:** tern has **no** Linux numbers. Whether PrismML CUDA on a discrete GPU is far ahead of today's Metal-relative gaps is unknown. CUDA 13.3 PrismML builds have reported crashes; 12.8 is the documented Linux workaround in Bonsai 2 known issues (may or may not apply to this repo's older 27B file). Vulkan/SYCL kernel selection for PQ2_0 is device-dependent per vendor docs.

### Windows

**Chosen baseline:** the same as Linux — llama.cpp, **PrismML fork**, fastest available backend (CUDA on NVIDIA, otherwise HIP, otherwise Vulkan, otherwise CPU). Official wording: `>= 2x` vs this llama.cpp.

**Why.** PrismML ships Windows x64 CPU, CUDA 12.4, CUDA 13.3, Vulkan, HIP/ROCm, and SYCL binaries (and Windows arm64 CPU / CUDA 13.4). That is the practical way to run this model natively. Stock llama.cpp / Ollama / LM Studio still need a new enough build and the `Q2_0` file for Bonsai 2; they do not replace the fork for PQ2_0.

**Alternatives checked, not used as the bar:**

- **ONNX Runtime GenAI, DirectML, Windows ML** — ONNX / D3D12 stacks; DirectML is in [sustained engineering](https://learn.microsoft.com/en-us/windows/ai/directml/dml) with new work aimed at Windows ML. No evidence they run this 2-bit ternary checkpoint. Even if they did, they would be a different numeric path (token parity would have to be re-proven). Not the baseline.
- **vLLM / SGLang / TensorRT-LLM** — same model-support and workload problems as on Linux; Windows support is not the primary path.
- **ExLlamaV3** — Windows is possible with extra Triton tooling; still the wrong quant.
- **bitnet.cpp** — wrong model family; native Windows builds are painful in public reports.
- **llamafile / Ollama** — llama.cpp again.

**Unverified:** no tern Windows numbers. PrismML known issues (as of 2026-09-23, written for Bonsai 2 packages) include Windows CUDA 13.3 crashes (workaround: CUDA 12.4), some CPUs where Windows CUDA builds will not start, and garbled HIP output on at least one APU. Re-check on the exact file this repo benches. TTFT and prefill on discrete GPUs include PCIe and page-wiring effects that REPORT.md never measured.

## What each OS has to ship

Shared product: the same `.tern` models, the same CLI (`gen` / `prof` / `bench`), the same speculation and optional `--draft`, the same three-prompt token check. Backend implementation is T6 and later code PRs.

**macOS.** Keep Metal as the production backend. Task [T7](../plan/T7/task.md) raises the **current** Metal engine from today's mixed results to `>= 2x` vs the composite baseline on every benchmark, building on [T3](../plan/T3/task.md), [T4](../plan/T4/task.md) and [T5](../plan/T5/task.md), with [T1](../plan/T1/task.md) for a quiet-machine table. This work can start before Linux/Windows backends exist.

**Linux.** After the T6 interface exists: CUDA (NVIDIA), Vulkan (AMD/Intel and NVIDIA fallback), CPU fallback. Build with Zig 0.17, Linux libc, and the backend's toolchain (CUDA toolkit or `glslang`/`glslc` + Vulkan loader). CI on a GitHub-hosted Linux runner can format, compile the CPU backend, and compile shaders / CUDA sources; it generally **cannot** run GPU benches (no GPU). Full `>= 2x` numbers need a real Linux box (see [T8](../plan/T8/task.md)).

**Windows.** Same backends as Linux (CUDA, Vulkan, CPU). HIP is not required in v1 if Vulkan covers AMD. File mapping and clocks must work natively (no WSL-only story). CI on `windows-latest` (or `windows-2022` / `windows-2025`) compiles; GPU benches need a real machine ([T9](../plan/T9/task.md)). A DirectML/D3D12 backend is not planned: it would not help the custom kernels and is not the fastest baseline.

## Task breakdown

Existing: [T1](../plan/T1/task.md) quiet Mac baselines; [T2](../plan/T2/task.md) Q8_0/f16 (not required for the 27B `>= 2x` bar); [T3](../plan/T3/task.md) speculation; [T4](../plan/T4/task.md) prefill; [T5](../plan/T5/task.md) fused kernels; [T6](../plan/T6/task.md) backend interface (design PR separate).

New:

- [T7](../plan/T7/task.md) — macOS Metal: `>= 2x` vs the composite llama.cpp / MLX-LM baseline on every harness metric and prompt.
- [T8](../plan/T8/task.md) — Linux: ship CUDA + Vulkan + CPU, token parity, then `>= 2x` vs PrismML llama.cpp on that machine's fastest backend, every metric and prompt.
- [T9](../plan/T9/task.md) — Windows: the same acceptance as T8 on native Windows.

T3–T5 remain the technical levers for T7 (and will likely need analogues on CUDA/Vulkan for T8/T9): pipelined verification / draft, cheaper short passes, a draft that helps free text and document QA, a faster GEMM for prefill and for speculation passes, fused small kernels.

## Suggested order and effort

Rough, calendar time for one focused engineer, not a promise.

1. **T6 interface, Metal still green** — about 1–2 weeks. Blocks Linux/Windows code, not T7 kernel work on Metal.
2. **T1 + T7 (macOS `>= 2x`)** — several weeks, high risk on prefill/TTFT and on free-text decode. Can overlap T6.
3. **CPU backend** — about 2–3 weeks to pass the token check and run in CI; not expected to hit `>= 2x` vs a GPU llama.cpp.
4. **Vulkan** (Linux and Windows) — about 1–2 months for the kernel set, then tuning.
5. **CUDA** (Linux and Windows) — about 1–2 months, overlapping Vulkan where the algorithms are shared. This is the tightest `>= 2x` fight.
6. **CI matrix** — format + compile on macOS / Linux / Windows, shader/PTX compile without GPUs; benches stay on real hardware.
7. **T8 / T9 measurement loops** — until every column is `>= 2x` or a metric is explicitly failed with measurements.

## Risks

- **Physics of decode.** Reading 6.8 GB of weights is ~18.3 ms on the M3 Max; a 26 ms token cannot become a 13 ms token by kernel tweaks. `>= 2x` decode on every prompt needs speculation that works on free text and document QA, with passes cheap enough under load (today `--draft` does not gain; accept is decided on the CPU).
- **Prefill gap.** T4 is already behind llama.cpp and MLX-LM on long prompts on Metal. `>= 2x` prefill vs the *faster* of those (llama.cpp) is a much larger jump, and llama.cpp CUDA on a discrete GPU will be a harder mark still.
- **CUDA llama.cpp is strong.** llama.cpp has years of CUDA (and Vulkan coopmat) work. A new CUDA/Vulkan port can lose token parity or speed in the first iterations. Budget time for both.
- **No numbers yet on Linux/Windows.** The bar is defined; the distance is unknown.
- **CI cannot prove `>= 2x`.** GitHub-hosted runners will not replace a bench machine.
- **Discrete memory and Windows mapping** are new failure modes (upload time, BAR size, path encodings).
- **Kernel toolchain risk** if a unified shader language is adopted too early (see [pyrlyn/tern#8](https://github.com/pyrlyn/tern/pull/8)).
