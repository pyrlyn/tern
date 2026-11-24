# MoE with SSD expert streaming, and DSpark block drafts

Design for tasks [T10](T10/task.md) to [T27](T27/task.md). Written 24 November 2026. No code in
this PR: every item below is a task for an implementing agent.

Sources read: the Atomic Engine study (`~/Grok/atomic-engine-study/concepts.md`, Atomic Engine
`engine-development` at `fb69fff`, MIT), this repo's README, REPORT.md, RESEARCH.md, `plan/`,
`ideas.md` and `src/`. Model facts below come from Hugging Face API metadata, `config.json`
files and GGUF headers read with HTTP range requests (first 25 MB). No model was downloaded.

Atomic Engine is MIT. Studying its design is fine. If a task ports code from it (or from
llama.cpp/ggml, also MIT), the ported file keeps the MIT copyright notice and the commit is
named in the PR.

## Goal and how it is measured

Atomic Engine's published figure: **105.71 tok/s**, Inkling 2 Mini `AD-IQ2_M` plus the Q8_0
DSpark draft, 14 GiB engine budget plus 100 MiB KV, 48 GB M4 Max, archive prompt, 64 generated
tokens. Stock llama.cpp on the same fixture: 33.71 tok/s. Almost all of the gap is the block
draft at full residency (zero expert misses at 14 GiB); at 4 GiB without the draft they quote
18.26 tok/s. See the study for the caveats (55/55 accepted draft tokens on that fixture).

The goal is **2x Atomic Engine**: about **211 tok/s** against the published figure. Because the
machines differ (tern is measured on an M3 Max 64 GB), the gate is a same-machine one: T10 runs
Atomic Engine on the M3 Max, and T27 requires tern `>= 2x` that run, interleaved, with greedy
tokens equal to the reference. Two pairs are gated:

| Pair | tern | Atomic Engine | Reference for greedy tokens |
| --- | --- | --- | --- |
| Inkling | Inkling 2 Mini `AD-IQ2_M` + its DSpark draft, 14 GiB budget | same files, `--spec-type draft-dspark`, 14436 MiB budget, flags from Atomic README | Atomic Engine / llama.cpp `inkling2-loop`, greedy, no draft |
| Bonsai | Ternary Bonsai 27B + prism DSpark head | `Ternary-Bonsai-27B-Q2_0.gguf` + `Ternary-Bonsai-27B-dspark-Q4_1.gguf`, if Atomic Engine loads them (T10 checks) | MLX-LM, as today |

The existing T7 bar (`>= 2x` max(llama.cpp, MLX-LM) on Bonsai) stays; this is an additional bar.

Rough memory floor for Inkling 2 Mini `AD-IQ2_M` (estimate from the GGUF header, not measured):
per layer visit 8 experts x (1.37 MB gate_up `IQ2_XXS` + 0.75 MB down `Q2_0`) = 17 MB, plus
about 28 MB of `Q8_0` attention and loop-control weights; 48 visits per token (6 prefix + 12 core
x 3 passes + 6 suffix) and a 380 MB `Q6_K` output head make about 2.5 GB read per plain token.
At the M3 Max's ~400 GB/s that is a floor of roughly 160 tok/s without speculation, so 211 tok/s
needs the block draft and a verify pass that reads the union of experts once (T14, T23).

## Models (exact revisions)

| Role | Repo @ revision | Files and size | Notes |
| --- | --- | --- | --- |
| Dense target (today) | `prism-ml/Ternary-Bonsai-27B-mlx-2bit` @ `70f75f3ad081ab840a42f3304c02c27e7f89bfb7` | 8.49 GB | Already used by tern |
| Bonsai DSpark head | `prism-ml/Ternary-Bonsai-27B-gguf` @ `86e89f34c93201c3dfd5e5880fedb0022fc7e34d` | `Ternary-Bonsai-27B-dspark-Q4_1.gguf` 1.95 GB, `...-dspark-bf16.gguf` 7.29 GB | **Pretrained, Apache-2.0.** See below |
| MoE bring-up target | `mlx-community/Qwen3.5-35B-A3B-4bit` @ `1e20fd8d42056f870933bf98ca6211024744f7ec` | 20.4 GB | Same GDN/attention blocks as Bonsai, MLX safetensors, MLX-LM reference |
| Its DSpark head | `jinzex/Qwen3.5-35B-A3B-DSpark` @ `9aede6f56fa546dbf1f62fe9974d596620fae201` | 1.01 GB bf16 | Block 8, Model Optimizer recipe; no license field on the card |
| Atomic comparison target | `AtomicChat/Inkling-2-Mini-GGUF` @ `309509848b3151419aa26d069eb03db57fd29a00` | `Inkling-2-Mini-AD-IQ2_M.gguf` 8.67 GB, `dspark-Inkling-2-Mini-Q8_0.gguf` 0.50 GB | Apache-2.0, base `thinkingmachines/Inkling-2-Mini` |
| Alternative Bonsai draft | `z-lab/Qwen3.5-27B-DFlash` @ `25ee0025ff950496a634e100b75c2db4515e9824` | 4.26 GB bf16 | DFlash, block 16, trained on bf16 Qwen3.5-27B, not on the ternary model |

Why two MoE targets. Qwen3.5-35B-A3B reuses everything tern already has (GDN and attention
kernels, the MLX 4-bit affine product of the draft model, the MLX-LM token check), so the MoE
machinery (router, expert GEMV, cache, streaming, budget) can be built and checked without a new
architecture. Inkling 2 Mini is the model Atomic Engine is measured on, and the only one where
hot-core residency applies, but it needs a new architecture and five new weight formats (T20,
T21). Both fit an M3 Max 64 GB fully resident (20.4 GB and 8.67 GB).

### Model shapes

- **Qwen3.5-35B-A3B**: 40 layers (every 4th full attention), hidden 2048, 256 routed experts,
  top 8, expert intermediate 512, one shared expert (intermediate 512) with a sigmoid gate,
  vocabulary 248320. MLX stores routed experts stacked: `mlp.switch_mlp.{gate,up,down}_proj`
  with a leading expert axis; router `mlp.gate`; group 64, 4-bit affine.
- **Inkling 2 Mini** (`inkling2_loop`): 24 physical layers, hidden 2304, 128 experts, top 8,
  expert intermediate 1152, no shared expert, gate bias (`exp_probs_b`), `expert_weights_scale`
  8.0, gate/up softcaps 4.0/25.0. Loop core is layers 6 to 17 run 3 times (`loop.start` 6,
  `loop.end` 18, `loop.count` 3, gate rank 64, not varying). Sliding window 1024 on 5 of every 6
  layers, relative-position attention (`attn_rel*`, extent 1024), short convolutions on q/k/v
  and after attention/FFN (`*_conv`, 4 taps), attention-residual blocks of 6, final logit softcap
  30, vocabulary 201024. In `AD-IQ2_M` the expert tensors are `blk.N.ffn_gate_up_exps.weight`
  `[2304, 2304, 128]` `IQ2_XXS` and `blk.N.ffn_down_exps.weight` `[1152, 2304, 128]` `Q2_0`
  (ggml type 42: blocks of 64 2-bit codes with one f16 scale, the PrismML type); attention and
  loop-control matrices `Q8_0`, `token_embd` `Q4_K`, `output` `Q6_K`. 704 tensors.

### DSpark heads that already exist

The 22 November draft-source survey in RESEARCH.md missed these. **No training is needed to
start**:

- **Bonsai** (`general.architecture = dspark`, 3.6B with shared embeddings/head): 6 layers,
  hidden 5120, 40 heads, 4 KV heads, head dim 128, FFN 5120, **block size 4**, mask token
  248319, target layers `[1, 16, 31, 46, 61]` (hidden states tapped from the 27B), fusion
  `dspark.fc` `[25600 -> 5120]` (5 taps x 5120), Markov head rank 256 (`markov_head_a` bf16,
  `markov_head_b` Q4_1, each `[256, 248320]`), confidence head (with Markov input), log-SNR
  conditioning (`log_snr_fc1/fc2`, range -9..9). The model card: trained against the ternary
  target, block size chosen from a verify-cost model, tau ~3.7 at k = 4, 1.34x on H100, and
  "on Apple Silicon the batch-1 verification pass does not yet amortize, so the drafter is not
  enabled by default on-device". That last sentence is exactly what T23 must change.
- **Inkling 2 Mini** (`general.architecture = dflash` with Markov context): 3 layers, hidden
  2304, FFN 9216, **block size 8**, target layers `[5, 23, 41, 48]` (indices over the 48 unrolled
  layer visits), no confidence head. The base model also ships it as `mtp.safetensors`
  (`mtp_config.block_size` 8, `markov_rank` 256, `markov_head_type` `rnn_hidden`).
- **Qwen3.5-35B-A3B** (jinzex): 6 layers, hidden 2048, block 8, target layers `[1, 8, 15, 23,
  30, 37]`, Markov rank 256 (vanilla), confidence head; SPEED-Bench acceptance length 3.97 at
  T = 0.

The brief asks for 7-8 token blocks on Bonsai. The shipped Bonsai head proposes 4. Options, in
order: (1) ship the block-4 head first (T24), (2) measure the z-lab DFlash block-16 head on
Bonsai (trained on the bf16 model, acceptance unknown), (3) train a block-8 head (T26), only
with the creator's go.

### Training a block-8 head (T26, not run without approval)

Recipe: NVIDIA Model Optimizer DSpark example (Model-Optimizer PR #2340, the recipe behind the
jinzex head) or z-lab's `dflash` trainer, initialised from the prism block-4 head. Target in
bf16 is needed for hidden-state taps: `prism-ml/Ternary-Bonsai-27B-unpacked` (about 54 GB).
Data: responses generated by Ternary Bonsai 27B itself (self-distillation) on an open
instruction set (for example Nemotron post-training v2, as jinzex used). Where: rented GPUs
(8 x H100 80 GB class), not the Mac. Order-of-magnitude estimate, to be confirmed by a short
pilot: tens to a few hundred GPU-hours. Cost and data choice are the creator's decision.

## The `.tern` format, version 2 (T11)

Today: magic `TERN0001`, count, page size 16384, 64-byte entries (name[48], offset, nbytes,
rows, cols, dtype), every tensor 16 KiB-aligned in execution order, mapped once.

Version 2 keeps everything above and adds:

1. Magic `TERN0002`; the reader still accepts `TERN0001`.
2. A JSON model description after the entry table (architecture, layer list with kinds, expert
   count, top k, expert intermediate size, loop start/end/count, sliding windows, softcaps,
   token ids). The engine stops being compile-time shaped for new models (Bonsai's `cfg` stays
   as a comptime fast path).
3. Entries grow by `experts: u32` and `stride: u64`. An expert tensor is one entry; expert `e`
   lives at `offset + e * stride`.
4. **Expert slabs**: for each MoE layer, expert `e`'s gate_up weights, gate_up scales (and
   biases), down weights, down scales are stored **contiguously** in one slab, and the slab is
   padded to 16 KiB. One `pread` per (layer, expert) fetches everything one expert needs, and the
   resident path addresses the same bytes through the mapping. Slabs of one layer are adjacent,
   so a layer is one range for prefetch-all.
5. New dtypes: 4-bit affine with group 32/64 (Qwen MLX and Q4_1 heads), `Q8_0`, `Q4_K`, `Q6_K`,
   `IQ2_XXS`, `Q2_0` g64. GGUF block layouts are kept byte-for-byte where the kernels read them
   directly (no requantization: greedy tokens must equal the reference engine's).
6. Fixed (non-expert) weights first, expert slabs after, so the fixed part is one contiguous
   range that is always resident.

## Runtime design

### Router and top-k (T13)

Router GEMV `[hidden -> experts]` in f32/f16 (small), then one Metal kernel per token: softmax
(or sigmoid plus bias for Inkling), top-8 selection by a simdgroup bitonic or per-lane partial
sort over 128/256 scores, renormalisation and scale, and writes `ids[8]` and `weights[8]` into a
GPU buffer. Ties break by lower expert id, matching the reference (checked against MLX-LM /
llama.cpp on the bench prompts). Resident path: the expert GEMV reads `ids` on the GPU and
indexes the slab (`base + id * stride`), so there is **no CPU readback** and the one command
buffer per token survives. Gate/up GEMV with SwiGLU fused, down GEMV, weighted sum into the
residual, shared expert in the same dispatch where present.

### Batched MoE for prefill and verify (T14)

For a pass of n tokens the n x 8 (token, expert) pairs are bucketed by expert on the GPU (count,
prefix sum, scatter), then one tiled GEMM per used expert over its token list. A verify pass of
8 tokens then reads each used expert once instead of up to 8 times. This is the MoE side of
"verify is cheap"; T23 is the dense side.

### Expert store and SSD streaming (T15)

New `src/expert.zig`. Resident vs streamed is decided per run by the budget (T18):

- **Fully resident** (the expected 2x configuration): experts are read through the existing
  mapping, touched at load like today. No store, no thread, no readback.
- **Streamed**: a fixed pool of 16 KiB-aligned slots (one slot = one slab of the largest layer)
  allocated once and wrapped with `newBufferWithBytesNoCopy` (one Metal buffer per pool chunk,
  never per expert). A miss table maps (layer, expert) to slot.

Reads: implement both and measure on the M3 Max SSD, keep the faster:
  (a) `pread` from a separate `fd` with `F_NOCACHE` (no double caching in the page cache;
      `F_RDADVISE` for the next layer), a small reader thread pool (4 threads, queue depth per
      layer), and (b) `mmap` plus `madvise(MADV_WILLNEED)` and copy into the slot. Atomic Engine
      uses `pread`/`preadv` with `F_RDADVISE` and leaves the page cache on; do not assume that is
      best here.

Synchronisation: with misses possible the CPU must know the ids. Per MoE layer the command buffer
is split after the router: the GPU writes `ids` to shared memory and signals an `MTLSharedEvent`;
the CPU resolves hits, issues reads for misses, and signals a second event when the slots are
filled; the expert GEMV waits on it. Hit-only layers skip the wait. The engine must not start
extra threads, I/O or splits when the run is fully resident (Atomic's disabled-path rule).

### Expert cache policy (T16)

Slots are charged to the budget. Policies: **LRU** (default) and **ClassLRU** (frequency-aware:
keep a decayed use count; victims come from the cold class first, LRU inside a class; Atomic
measured -3.55% read bytes on one trace and did not make it default). A trace writer records
`(token, layer, ids)`; a replay tool computes hit rate and bytes for each policy and slot count
without running the model, so policy changes are judged on traces first, then on wall time.

### Prefetch (T17)

While layer L runs, apply layer L+1's router to the current hidden state (one small GEMV on the
GPU) to predict its experts; issue reads for the predicted misses, bounded (at most 8 per layer,
same admission rules as demand reads, never evicting an expert already predicted for L). For
Inkling the next physical layer after 17 in passes 1 and 2 is 6.

### One memory budget (T18) and live shrink (T19)

`src/budget.zig`, one cap (`--budget-mib`) with named charges: fixed weights, expert slots, KV
cache, GDN states, scratch, draft model (weights, its KV, its scratch), I/O staging, and a fixed
envelope. Start-up admits only if fixed + envelope + one slab per layer + staging fits; otherwise
it refuses with the numbers. `tern stats` prints the ledger. Shrink at a token boundary: drop
slots from the cache (policy order, Inkling hot-core last), never touch fixed weights or session
state, refuse a cap below the floor and keep running. Grow refills lazily. Optional automatic
policy from macOS memory-pressure notifications (`dispatch_source` memory pressure) is an idea,
not a task.

### Hot-core priority (T22)

Inkling layers 6 to 17 are run 3 times per token on the same physical experts. Spare budget goes
to the core's slots first, then suffix, then prefix; shrink releases in reverse. This is why
Atomic's 14 GiB run has zero misses.

### DSpark block draft (T24, T25)

Per pass: the target's hidden states at the tapped layers for the last accepted tokens are
normalised per source and concatenated (5 x 5120 for Bonsai), projected by `dspark.fc`, and
form the draft's context. The draft runs **one** forward over a block of `k` mask tokens (non-
causal within the block, causal to the context) with log-SNR conditioning, then the Markov head
adds a rank-256 term keyed on the previous token, chained left to right inside the block (cheap:
two `[256 x vocab]` products per position), and the confidence head scores each position so the
pass length can be cut where confidence drops. The block is verified by the target in one pass
(T23); the accepted prefix plus the target's next token is committed, GDN states and KV are
rolled back as in T3. The draft's own context cache is rolled back the same way. Embeddings and
the output head are shared with the target (no second copy, budget charges only the draft-unique
weights, ~0.5 GB at Q4_1 per the card).

### Block verify (T23)

Today (REPORT.md): a pass of 2 is 1.15-1.3 plain tokens, 4 is 2.1, 8 is 3.5, under load worse;
from 4 tokens on the ternary unpacking makes the product compute-bound. Target: a pass of 4 at
<= 1.4 and a pass of 8 at <= 2.0 plain tokens on a quiet machine, tokens unchanged. Directions
(the implementing agent picks and measures): unpack each 128-weight group once into registers or
threadgroup memory and reuse it across the 4-8 tokens; half-precision activations with f32
accumulation; 8 x 8 simdgroup matrix intrinsics on the unpacked tile; a dedicated n = 4/8 kernel
rather than the 32-token tile. Rejected variants in REPORT.md are not retried without a new
reason.

## Correctness gates (every implementation PR)

- Bonsai: `tern_bench.py` `same as MLX: True` on the three bench prompts, with and without the
  draft, as today.
- Qwen3.5-35B-A3B: greedy tokens equal MLX-LM (`mlx-lm`, same 4-bit files) on the three prompts.
- Inkling 2 Mini: greedy tokens equal Atomic Engine / llama.cpp `inkling2-loop` without a draft
  on the three prompts and on the Atomic archive fixture.
- Speculation changes nothing: tokens with the draft equal tokens without it.
- Streaming changes nothing: tokens at a 4 GiB budget equal tokens fully resident.
- Every speed change has a before/after interleaved measurement (plan rule).

## Order of implementation PRs

Each PR is one task or one part of one, small enough to review, with its own token check.

| # | Task | PR | Expected effect |
| --- | --- | --- | --- |
| 1 | T10 | Atomic Engine in `bench/compare.py`; M3 Max baselines for both pairs | The number the 2x gate is against |
| 2 | T23 | Block verify kernel for n = 4 and 8 (Bonsai) | Pass of 8 from 3.5 to <= 2 plain tokens; multiplies every draft |
| 3 | T24a | Convert the prism DSpark head (GGUF -> `.tern`), Q4_1 product | Loads, matches a reference forward |
| 4 | T24b | DSpark draft forward, Markov and confidence heads, wired into `chain.zig` | Bonsai decode up by the accepted length (card: tau ~3.7 at k = 4) |
| 5 | T11 | `.tern` v2 reader, writer, format doc | Experts addressable per expert |
| 6 | T12 | Qwen3.5-35B-A3B converter | First MoE `.tern` |
| 7 | T13 | Router, top-k, expert GEMV, resident MoE decode | First MoE tokens, equal to MLX-LM |
| 8 | T14 | Batched MoE for prefill and verify | Prefill and verify read each expert once |
| 9 | T18 | Budget ledger and admission | One auditable cap |
| 10 | T15 | Expert store, `pread`/`F_NOCACHE` vs `mmap`, events | Models above the cap run |
| 11 | T16 | LRU, ClassLRU, trace and replay | Fewer bytes read at tight caps |
| 12 | T17 | Next-layer prefetch | Hides SSD latency at tight caps |
| 13 | T19 | Live shrink and grow | RAM given back mid-session |
| 14 | T25a | DSpark for Qwen3.5-35B-A3B | MoE decode x accepted length |
| 15 | T20 | Inkling GGUF converter and its 5 weight formats | Inkling `.tern` |
| 16 | T21 | `inkling2_loop` architecture | Inkling tokens equal the reference |
| 17 | T22 | Hot-core residency | Zero misses at 14 GiB |
| 18 | T25b | Inkling DSpark draft | The Atomic configuration in tern |
| 19 | T27 | The 2x gate run and REPORT.md section | Done or the measured gap |
| - | T26 | Block-8 Bonsai head (training) | Only with the creator's go |

## New dependencies and downloads

- No new build dependencies: still Zig 0.17 and Metal through the Objective-C runtime. Python
  tools keep numpy; reading GGUF in `convert.py` uses a small own reader (or the `gguf` package
  from llama.cpp, MIT, as a dev dependency; the agent decides and says so).
- Downloads when the tasks start (none in this PR): Bonsai DSpark Q4_1 1.95 GB (and bf16 7.29 GB
  for a reference forward), Qwen3.5-35B-A3B MLX 4-bit 20.4 GB plus jinzex head 1.0 GB, Inkling
  `AD-IQ2_M` 8.67 GB plus its draft 0.50 GB, `Ternary-Bonsai-27B-Q2_0.gguf` 7.17 GB for the Atomic
  Engine baseline, optionally z-lab DFlash 4.26 GB.
- A built Atomic Engine (`engine-development`, pinned commit) on the M3 Max for T10/T27 and as the
  Inkling token reference; `mlx-lm` with Qwen3.5 MoE support for the Qwen reference.
