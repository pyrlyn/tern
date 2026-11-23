# Research: 37 techniques for fast LLM inference

Scope: single-stream greedy generation of Ternary Bonsai 27B (Qwen3.5 27B hybrid, 2-bit ternary
weights) on an M3 Max (40-core GPU, 64 GB, no SME: `hw.optional.arm.FEAT_SME: 0`), plus the
server-GPU case as theory. The earlier survey was lost; this one is rebuilt from the sources and
from measurements made in this repository (see [REPORT.md](REPORT.md)).

Each entry: what it is, whether it applies to the M3 Max, whether hand-written asm/C helps, and
a source. Links were checked on 2026-11-22: arXiv ids by fetching the abstract page and comparing
the title, other URLs by HTTP status 200. "Measured here" means the number comes from this
repository, not from the paper.

Two facts frame everything below:

- **Decode is memory-bound.** One token reads 6.4 GB of weights and 0.42 GB of scales. Reading
  them and doing nothing else takes 18.3 ms on this GPU (about 350 GB/s); the production GEMV
  takes 19.7 ms. Kernels alone cannot go below that floor, so a large gain needs either fewer
  bytes per token (denser packing) or fewer weight passes per token (speculation).
- **Prefill and verification are compute-bound.** A batched pass decodes each weight word once
  for several tokens; the limit becomes arithmetic throughput (2.2 to 2.7 T weight-token
  products per second here; MLX-LM's 160 tok/s prefill implies about 4.4 T).

## A. Reading the weights less than once per token

| # | Technique | Essence | M3 Max, one stream | Hand asm / C | Source |
| --- | --- | --- | --- | --- | --- |
| 1 | Draft-model speculative decoding | A small model proposes k tokens, the big model checks all of them in one batched pass; greedy output is unchanged | Yes, the main lever: a batched pass costs about half a sequential one per token (13.3 vs 26.5 ms, measured here at batch 32). Needs a 0.5B-class draft of the same tokenizer | No; the gain is in the pass structure | [2211.17192](https://arxiv.org/abs/2211.17192), [2302.01318](https://arxiv.org/abs/2302.01318) |
| 2 | Prompt lookup decoding | Draft = continuation of the longest n-gram match in the prompt; no draft model | Yes; costs nothing, strong on code edit and document QA, useless on free text | No | [prompt-lookup-decoding](https://github.com/apoorvumang/prompt-lookup-decoding) |
| 3 | Retrieval speculation (REST) | Draft from a datastore of text instead of the prompt | Maybe; a datastore is a product decision, the mechanism is the same as 2 | No | [2311.08252](https://arxiv.org/abs/2311.08252) |
| 4 | Self-speculative decoding | The model drafts with some of its own layers skipped | Weak for a ternary model: skipped layers still leave most of the weight traffic, and layer choice needs tuning per model | No | [2309.08168](https://arxiv.org/abs/2309.08168) |
| 5 | LayerSkip | Early-exit heads trained for self-speculation | No: needs a model trained with the early-exit loss | No | [2404.16710](https://arxiv.org/abs/2404.16710) |
| 6 | Medusa heads | Extra heads predict tokens t+2..t+k from the last hidden state | Needs head training on the frozen model; the heads are cheap at run time | No | [2401.10774](https://arxiv.org/abs/2401.10774) |
| 7 | EAGLE-3 | A one-layer drafter on hidden features with a dynamic draft tree; the best published acceptance rates | Needs drafter training; server case first. Published speedups assume a GPU where a wide batch is nearly free, which is not true here | No | [2503.01840](https://arxiv.org/abs/2503.01840), [2406.16858](https://arxiv.org/abs/2406.16858), [2401.15077](https://arxiv.org/abs/2401.15077) |
| 8 | Lookahead decoding | Jacobi-style parallel n-gram guesses verified in the same pass, no draft model | Low: trades a much wider batch for few accepted tokens; a wide batch is not free on this GPU | No | [2402.02057](https://arxiv.org/abs/2402.02057) |
| 9 | Token-tree verification | Verify a tree of candidate continuations in one pass with a tree attention mask | Later: a tree multiplies the batch, and batch width is the cost here. Worth it only with a faster GEMM | No | [2305.09781](https://arxiv.org/abs/2305.09781) |
| 10 | Adaptive draft depth and a speculation gate | Choose k from recent acceptance; turn speculation off when it stops paying | Yes, required: a rejected draft costs a full batched pass. This is what turns the -3% on free text into no loss | No | [2405.19715](https://arxiv.org/abs/2405.19715) |
| 11 | Multi-token verify kernel | One read of a weight word serves k positions (GEMM with a small token dimension) | Yes, built: `GEMM_T2` in [src/kernels.metal](src/kernels.metal); 4 tokens per thread is the best shape so far, 8 and 16 are slower | MSL only; no asm on the GPU | Measured here |

## B. Weight format and the memory path

| # | Technique | Essence | M3 Max, one stream | Hand asm / C | Source |
| --- | --- | --- | --- | --- | --- |
| 12 | Ternary weights | Weights in {-1, 0, +1} with a scale per group; 2 bits per weight as stored | The model under test; 6.4 GB per token instead of 13.6 GB at 4 bits | n/a | [2402.17764](https://arxiv.org/abs/2402.17764) |
| 13 | Lookup-table ternary kernels | Index a table of partial sums by a group of weights (TL1/TL2 in bitnet.cpp) | No on the GPU: a 256-entry table per 4 columns was 4.1 to 5x slower than arithmetic unpacking (gathers from a 1.3 MB table, measured here). Good on a CPU with `tbl` | Yes on CPU (NEON `tbl`), no on GPU | [2502.11880](https://arxiv.org/abs/2502.11880), [2410.16144](https://arxiv.org/abs/2410.16144) |
| 14 | Dense trit packing | 5 trits in 8 bits (1.6 bits per weight) instead of 2 bits | Yes, the only kernel-level way under the floor: 20% less traffic. Decode cost on the GPU is unknown and must stay below the saving | MSL only | [2502.11880](https://arxiv.org/abs/2502.11880) |
| 15 | Mask-in-place unpacking with a lane split | Decode 2-bit slots with AND and an integer subtract at their bit position, fold the power of four into the input; one SIMD group per row block, 32 lanes over the columns | Yes, accepted: the GEMV went from compute-bound to 93% of the memory floor; bit-identical to the shift version | MSL only | Measured here |
| 16 | Multi-row GEMV | Several rows per thread share the loads of the input vector | Yes, accepted together with 15 (4 rows, 16 for wide matrices). Without the lane split it was slower | MSL only; on a CPU backend this is where NEON asm pays | Measured here; idea from colibri |
| 17 | Q8_0 / f16 GEMV with repacked weights | Store rows in the order the kernel walks them, interleaved for the lanes | Yes for the other weight formats runa will load; same kernel shape as 15 and 16 with a different decode | MSL only | [llama.cpp](https://github.com/ggml-org/llama.cpp) |
| 18 | 4-bit quantization (AWQ, GPTQ) | Calibrated low-bit weights | A different model, not this benchmark; relevant for the server kernels | No | [2306.00978](https://arxiv.org/abs/2306.00978), [2210.17323](https://arxiv.org/abs/2210.17323) |
| 19 | Entropy-coded weights | Huffman-coded weights decoded during inference | No on the GPU: serial bit-stream decode does not fit a SIMD lane. Ternary weights with many zeros would compress, but 14 gets most of it | Yes on CPU or FPGA only | [2502.00922](https://arxiv.org/abs/2502.00922) |

## C. Kernels and GPU execution

| # | Technique | Essence | M3 Max, one stream | Hand asm / C | Source |
| --- | --- | --- | --- | --- | --- |
| 20 | One command buffer per token, fused kernels | Encode the whole token once; merge neighbouring small kernels | Partly done: one command buffer per token, GPU busy 93% of wall time, CPU encode 0.13 to 0.19 ms per token. Fusing the small kernels is worth about 1 to 2 ms of 26.5 | No | [MTLCommandBuffer](https://developer.apple.com/documentation/metal/mtlcommandbuffer) |
| 21 | Event-driven GPU wait | `MTLSharedEvent` or a completion handler instead of a spinning CPU thread | Yes; the decode thread sleeps in `waitUntilCompleted` today. Matters for power and for the cores the product needs elsewhere, not for tokens per second | No | [MTLSharedEvent](https://developer.apple.com/documentation/metal/mtlsharedevent) |
| 22 | Sampling on the GPU with encode lookahead | Argmax writes the next token id into a GPU buffer, so the CPU can encode token t+1 before t finishes | Done; lookahead 1, 2 and 4 give the same speed, so the CPU is not the limit | No | Measured here |
| 23 | Tiled, batched GEMM | Keep the input tile in threadgroup memory, decode each weight word once for many tokens | Yes, the open prefill problem: 2.2 to 2.7 T products/s here against about 4.4 T/s implied by MLX-LM's prefill speed | MSL only | [MLX](https://github.com/ml-explore/mlx) |
| 24 | Compiler-generated fused kernels | A tensor compiler emits one kernel per subgraph, or one for a whole small model | For the 0.5B draft model: yes, one kernel per layer is realistic. For 27B the weights dominate and fusion buys little | No | [1802.04799](https://arxiv.org/abs/1802.04799), [MLC LLM](https://github.com/mlc-ai/mlc-llm) |
| 25 | Half-precision arithmetic | Compute in `half` where the reference does | Risky: activations are stored as half, but sums must be float to reproduce the reference tokens (near-ties in the logits) | No | [Metal Shading Language](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) |
| 26 | CPU SIMD kernels | NEON / dotprod / i8mm kernels, SME where present, Accelerate for f32 | For the CPU backend only, which is the fallback and not the fast path. No SME on this M3 Max. CPU bandwidth for this workload is not measured yet | Yes: this is the one place hand asm pays | [Instruction set sysctls](https://developer.apple.com/documentation/kernel/1387446-sysctlbyname/determining_instruction_set_characteristics), [Accelerate BLAS](https://developer.apple.com/documentation/accelerate/blas) |
| 27 | Spin-wait thread pool | Worker threads spin briefly instead of sleeping between tokens | Rejected earlier (under the 10% threshold); irrelevant on the GPU path | C | colibri idea; earlier measurement |
| 37 | Lane split of latency-bound kernels | Small kernels with a serial inner loop (delta-rule step, softmax, value sum) are bound by loop latency, not work; split the loop over the 32 lanes | Accepted for the delta-rule step and the softmax: 1.93 to 1.17 ms and 0.70 to 0.08 ms per token. Changes the summation order, so every combination is checked against the reference tokens | MSL only | Measured here |

## D. Attention and the KV cache

| # | Technique | Essence | M3 Max, one stream | Hand asm / C | Source |
| --- | --- | --- | --- | --- | --- |
| 28 | GQA-aware decode attention | Read each KV head once for the whole group of query heads | Yes: 24 query heads share 4 KV heads here. The value sum grows linearly with context (0.55 ms at 200 tokens, 2.6 ms at 600) | MSL only | [2305.13245](https://arxiv.org/abs/2305.13245) |
| 29 | FlashAttention-3 | Tiled attention with asynchronous warp specialisation and FP8 on Hopper GPUs | Server only; the tiling idea carries over to prefill attention | No (CUDA) | [2407.08608](https://arxiv.org/abs/2407.08608) |
| 30 | FlashInfer | Block-sparse KV storage with JIT-compiled attention kernels for serving | Server only | No (CUDA) | [2501.01005](https://arxiv.org/abs/2501.01005), [flashinfer](https://github.com/flashinfer-ai/flashinfer) |
| 31 | PagedAttention | KV cache in fixed blocks behind a page table, shared between requests | Server; for one stream a flat buffer is faster | No | [2309.06180](https://arxiv.org/abs/2309.06180) |
| 32 | Gated delta networks | Linear attention with a fixed-size state instead of a KV cache | The model already has it in 48 of 64 layers: no growth with context there, but the state update is the largest non-GEMV cost | MSL only | [2412.06464](https://arxiv.org/abs/2412.06464) |
| 33 | KV cache quantization | Store keys and values in 8 bits or fewer | Not needed at these context lengths; the cache is half floats for 16 layers only | No | [2401.18079](https://arxiv.org/abs/2401.18079) |

## E. Serving, prefill and loading

| # | Technique | Essence | M3 Max, one stream | Hand asm / C | Source |
| --- | --- | --- | --- | --- | --- |
| 34 | Continuous batching and chunked prefill | Add and remove requests every step; split long prompts into chunks that share steps with decodes | Chunked prefill: yes, already the shape of the batched pass (32 tokens per chunk). Continuous batching: server only | No | [2308.16369](https://arxiv.org/abs/2308.16369), [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) |
| 35 | Prefix cache (RadixAttention) | Reuse the KV cache and state of a shared prompt prefix across requests | Yes for the product (system prompt, repeated document); it removes prefill instead of speeding it up. Needs a snapshot of the delta-rule states too | No | [2312.07104](https://arxiv.org/abs/2312.07104) |
| 36 | Weight loading | `mmap` with no copy, pages wired at load, parallel block reads in layer order for a cold start | Partly done: zero-copy `mmap`, pages touched by a GPU kernel at load. The 4-thread 4 MiB reader and the `--nvme` path are not ported yet | C / Zig, no asm | Measured here |

## Top 5 for the M3 Max, one stream

1. Speculative decoding: prompt lookup plus a 0.5B draft model, adaptive draft depth with a gate,
   verified by the multi-token kernel (1, 2, 10, 11). The only route to +40%.
2. A faster batched GEMM (23). It sets both the prefill speed and the price of every
   verification pass, so it multiplies the gain of 1.
3. Dense trit packing (14): the only way to lower the memory floor of plain decode.
4. GQA-aware attention and fused small kernels (28, 20, 37): a few percent now, more at long
   context.
5. Q8_0 / f16 kernels with repacked weights (17) so that the other runa models get the same GEMV.

Sampling on the GPU (22) is done and is no longer a candidate.

## Top 5 for a server GPU (theory, not measured)

1. Continuous batching with chunked prefill and a paged KV cache (34, 31).
2. FlashAttention-3 / FlashInfer attention with FP8 (29, 30).
3. Prefix cache, RadixAttention (35).
4. EAGLE-3 speculation (7).
5. Kernels for 4-bit and FP8 weights, and one fused kernel per small model (18, 24).

## Draft sources for Ternary Bonsai 27B

For [T3](plan/T3/task.md). Checked on Hugging Face on 2026-11-22 from the config files and file
lists; no model was downloaded. Every candidate has the tokenizer of the 27B model (vocabulary
248320) and the Qwen3.5 hybrid layout. Costs per draft token are estimates, not measurements:
weights read at the rate of the 27B decode (6.8 GB in 18.3 ms) plus 0.1 ms per layer for the
small kernels, against 26 ms for a plain token.

| Candidate | What it is | Size | What tern needs to run it | Cost per draft token |
| --- | --- | --- | --- | --- |
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), also as [MLX 4-bit](https://huggingface.co/mlx-community/Qwen3.5-0.8B-4bit) | 24 layers, hidden size 1024, tied embeddings, Apache-2.0 | 0.87B parameters; 1.75 GB bf16, 0.63 GB 4-bit | The model shape at run time instead of compile time; a product kernel for 4-bit affine or f16 weights ([T2](plan/T2/task.md)) | about 3.5 ms, 0.13 plain tokens |
| MTP head of [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) | One more decoder layer and a projection (`mtp.*`, 15 tensors) that predict the token after next from the main model's last hidden state. The Ternary Bonsai 27B files do not carry it | about 0.4B parameters, 0.8 GB bf16 | An f16 or 4-bit kernel for one layer; the head and the embeddings are those of the main model | 1.7 to 3.3 ms, 0.07 to 0.13 plain tokens |
| [Ternary-Bonsai-2-2B MLX 2-bit](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-mlx-2bit) | 24 layers, hidden size 2048, 2-bit ternary with groups of 128 like the 27B file, but stored behind a Walsh-Hadamard rotation of the input (`hadamard.json`); stock mlx-lm does not load it | 1.2 GB | The model shape at run time; a Hadamard transform in front of every rotated product | about 4 ms, 0.15 plain tokens |

Ruled out: the first Bonsai generation (1.7B, 4B, 8B) is built on Qwen3 with a vocabulary of
151669; the MTP heads shipped with Ternary-Bonsai-2-4B and 9B belong to hidden sizes 2560 and
4096.

Risks. The MTP head was trained on the hidden states of the bf16 model, not of the ternary one,
so its acceptance on our model is unknown. The vendor's note for its own MTP drafters
([MTP.md](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-mlx-2bit/blob/main/MTP.md)) says
they help on copy-heavy work and are level with plain decoding or slightly slower on general
chat.

**The price of a pass decides before the draft does.** With a draft token accepted with
probability p, independently, a draft cost d and the measured pass prices (2.9 plain tokens for
4 tokens, 4.0 for 8, 8.2 for 32), the best pass size gives:

| | p = 0.70 | 0.80 | 0.85 | 0.90 | 0.95 | break-even | +40% at |
| --- | --- | --- | --- | --- | --- | --- | --- |
| measured prices, d = 0.10 | 0% | 0% | +3% | +21% | +43% | p = 0.84 | p = 0.94 |
| measured prices, d = 0.15 | 0% | 0% | 0% | +13% | +33% | p = 0.86 | p = 0.96 |
| short passes at 1.2 / 1.5 / 2.2 for 2 / 4 / 8 tokens, d = 0.10 | +41% | +64% | +77% | +96% | +132% | p = 0.50 | p = 0.70 |
| the same, d = 0.15 | +30% | +51% | +63% | +76% | +107% | p = 0.50 | p = 0.75 |

A small model does not agree with a 27B model on 94% of free-text tokens, so with today's pass
prices no candidate can reach the goal, and below 84% it loses. The short-pass prices in the
lower rows are an assumption: they are what a 4-token pass would cost if its matrix product ran
at the throughput of the 32-token tile (4.3 T products per second; the lane kernel does 1.9 to
2.2). With them prompt lookup alone would have an oracle bound of +88% on doc_qa instead of +36%.

Order of work that follows:

1. Measure p before building anything: run the draft candidate over the prompt and the 27B
   reference tokens once (teacher forcing, in Python with mlx-lm) and count how often its
   greedy token equals the reference. The accepted run lengths are exact for a greedy draft, and
   the policy simulation turns them into a speed. Needs the 0.63 GB download of the 4-bit 0.8B
   model.
2. Bring the price of a 4-token and an 8-token pass down; this helps prompt lookup at once and
   is the condition for any draft model.
3. Only then run the draft model inside tern (run-time model shape, the kernel for its format).

## Not measured

vLLM and SGLang need an NVIDIA GPU; MLC LLM and ExecuTorch targets need a phone. Neither is
available here, so there are no numbers for them and none are estimated.
Sources for a later run: [vLLM](https://github.com/vllm-project/vllm),
[SGLang](https://github.com/sgl-project/sglang), [MLC LLM](https://github.com/mlc-ai/mlc-llm),
[ExecuTorch](https://github.com/pytorch/executorch).
