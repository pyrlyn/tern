# Report

State on 2026-11-23, after the draft model went into the engine (`--draft`, off by default)
([T3](plan/T3/task.md)). Machine: MacBook Pro M3 Max (16 CPU cores, 40-core GPU, 64 GB), macOS 27.0.1,
on AC power. Model: Ternary Bonsai 27B, 2-bit ternary weights, the same weights for every engine
(MLX safetensors for MLX-LM, the PQ2_0 GGUF for llama.cpp, `.tern` converted from the MLX file).
Greedy generation of 200 tokens after prompts of 38, 427 and 423 tokens.

Engines: llama.cpp PrismML fork d12ca13 (Metal, `-ngl 99`), MLX-LM 0.32.2 on MLX 0.32.4, tern
(this repository, Zig 0.17.1).

## Conditions, read first

The rule is a load average below 4. **It was not met.** Other jobs on the machine kept the load
between 4 and 110 for the whole session, and the GPU ran throttled for minutes at a time: every
engine then drops to about half of its speed (MLX-LM decode went from 38.7 to 17.5 tok/s between
two runs with no change on our side). Numbers taken minutes apart are therefore not comparable.

What is comparable: `bench/compare.py` runs the three engines one after another for every prompt,
in a rotating order, and takes the median over the rounds. The table below is from that run, with
the load average recorded per measurement. The absolute numbers are not peak numbers; the run on a
quiet machine is task [T1](plan/T1/task.md).

## Goals

Interleaved run, 3 rounds, medians; load average at the start of the single measurements between
7.7 and 68.3 (all values in `bench/results/compare.json`). The GPU was in its slow state for the
whole run, so every engine is at about half of its best speed.

| Prompt | Metric | llama.cpp | MLX-LM | tern | tern vs llama.cpp | tern vs MLX-LM |
| --- | --- | --- | --- | --- | --- | --- |
| free_text (38 tokens) | decode, tok/s | 20.6 | 20.8 | 24.4 | +19% | +17% |
| | prefill, tok/s | 103.3 | 64.9 | 99.8 | -3% | +54% |
| | first token, s | 0.37 | 0.59 | 0.38 | +4% | -35% |
| code_edit (427 tokens) | decode, tok/s | 19.9 | 19.0 | 20.3 | +2% | +7% |
| | prefill, tok/s | 132.5 | 114.8 | 99.4 | -25% | -13% |
| | first token, s | 3.22 | 3.72 | 4.29 | +33% | +15% |
| doc_qa (423 tokens) | decode, tok/s | 20.2 | 19.1 | 20.1 | 0% | +5% |
| | prefill, tok/s | 134.0 | 116.9 | 92.4 | -31% | -21% |
| | first token, s | 3.16 | 3.62 | 4.58 | +45% | +27% |

Single runs of decode, tok/s: llama.cpp 20.5 to 20.7, 18.0 to 21.5, 17.2 to 21.6; MLX-LM 19.6 to
24.0, 18.4 to 19.8, 17.2 to 19.2; tern 20.8 to 25.1, 18.9 to 23.6, 18.9 to 20.1. The spread of
one engine is as large as the differences between engines, so decode is a tie, with tern at or
slightly above the other two.

### Second run: the fast state, for a few minutes

A second interleaved run (`bench/results/compare_fast.json`, same build apart from formatting)
was started when the machine had left its slow state. That lasted for the first prompt of the
first round; then the load rose from 11 to 78 and the machine fell back.

The one set of measurements taken in the fast state, back to back, load 11 to 13, one run per
engine after its warm-up (free_text, 38 prompt tokens):

| Metric | llama.cpp | MLX-LM | tern | tern vs llama.cpp | tern vs MLX-LM |
| --- | --- | --- | --- | --- | --- |
| decode, tok/s | 30.2 | 38.2 | 38.9 | +29% | +2% |
| prefill, tok/s | 120.1 | 104.3 | 118.2 | -2% | +13% |
| first token, s | 0.32 | 0.36 | 0.32 | 0% | -11% |

This is one run per engine, not a median, so it is an indication and not a result. It does say
two things the slow-state table cannot: tern and MLX-LM are level at about 38 to 39 tok/s when
the machine allows it, and llama.cpp was slower there. Its two best runs in the whole session are
30.2 and 30.4 tok/s, on two different prompts.

The rest of that run mixes both states, and its medians mean little. All single runs, in the
order of the rounds:

| Prompt | Engine | decode, tok/s | prefill, tok/s | load at start |
| --- | --- | --- | --- | --- |
| free_text | llama.cpp | 30.2, 14.2, 13.1 | 120.1, 54.5, 47.7 | 11.0, 26.3, 9.6 |
| | MLX-LM | 38.2, 14.8, 12.9 | 104.3, 51.9, 48.0 | 13.4, 30.1, 23.2 |
| | tern | 38.9, 15.5, 11.3 | 118.2, 61.4, 35.8 | 13.3, 27.3, 10.0 |
| code_edit | llama.cpp | 9.2, 30.4, 12.3 | 46.6, 211.1, 88.8 | 32.0, 29.6, 21.5 |
| | MLX-LM | 26.9, 24.7, 13.5 | 185.4, 147.2, 86.7 | 11.9, 25.2, 44.7 |
| | tern | 21.3, 28.7, 14.9 | 123.0, 153.7, 64.2 | 10.9, 42.0, 29.8 |
| doc_qa | llama.cpp | 13.3, 21.1, 12.6 | 108.9, 162.3, 114.8 | 10.8, 20.4, 77.8 |
| | MLX-LM | 15.6, 15.7, 12.8 | 89.2, 106.8, 78.3 | 22.1, 18.0, 22.8 |
| | tern | 13.6, 11.1, 17.9 | 62.3, 52.6, 70.2 | 18.7, 14.3, 41.2 |

Best prefill of each engine on code_edit in this run, not at the same moment: llama.cpp 211
tok/s, MLX-LM 185, tern 154. That is the same order as in the slow-state table, so the prefill
gap is real in both states. Tokens: tern equal to MLX-LM in every run, llama.cpp text equal to
MLX-LM, MLX-LM equal to its own reference.

Earlier best values, also not simultaneous (load 5 to 12): MLX-LM 38.7 / 36.6 / 29.5 tok/s decode
and 109 / 156 / 164 tok/s prefill; tern 38.5 tok/s decode on free_text before the prefill work.

### Third run: speculative decoding

Same harness after speculative decoding went in (`bench/results/compare_spec.json`), 3 rounds,
medians, load at the start of the single measurements between 13.7 and 113.5. The machine was
slower than in the first run: llama.cpp and MLX-LM decode at 14 to 15 tok/s here against 19 to 21
there.

| Prompt | Metric | llama.cpp | MLX-LM | tern | tern vs llama.cpp | tern vs MLX-LM |
| --- | --- | --- | --- | --- | --- | --- |
| free_text (38 tokens) | decode, tok/s | 14.8 | 14.7 | 17.5 | +19% | +19% |
| | prefill, tok/s | 67.9 | 55.9 | 53.2 | -22% | -5% |
| | first token, s | 0.56 | 0.68 | 0.71 | +27% | +5% |
| code_edit (427 tokens) | decode, tok/s | 14.6 | 15.5 | 35.6 | +145% | +130% |
| | prefill, tok/s | 104.4 | 98.7 | 70.8 | -32% | -28% |
| | first token, s | 4.09 | 4.33 | 6.03 | +47% | +39% |
| doc_qa (423 tokens) | decode, tok/s | 14.2 | 15.4 | 17.1 | +21% | +12% |
| | prefill, tok/s | 97.7 | 94.6 | 72.6 | -26% | -23% |
| | first token, s | 4.33 | 4.47 | 5.83 | +35% | +30% |

Single runs of decode, tok/s: llama.cpp 14.4 to 15.7, 13.5 to 15.0, 14.0 to 15.2; MLX-LM 14.0 to
15.5, 15.4 to 17.6, 14.9 to 16.6; tern 16.6 to 20.9, 34.3 to 39.5, 16.6 to 20.8. On code_edit the
slowest tern run is twice the fastest run of the other two. On free_text no verification pass
runs, so the +19% there is plain decode in this state of the machine, as in the first run. On
doc_qa the spread of tern (16.6 to 20.8) is larger than its lead over MLX-LM.

Prefill is not touched by this work: the build before it and the build after it, run in turn
(4 rounds, speculation off), give 58.9 and 64.7 tok/s on free_text and 79.1 and 84.3 on
code_edit, with decode at 20.4 and 19.0, 18.8 and 19.7 tok/s. The gap to the other two engines
is the one of the first run.

### Fourth run: the 2-token pass

Same harness with the 2-token pass as the first pass size (`bench/results/compare_pass2.json`),
3 rounds, medians, load at the start of the single measurements between 15.7 and 88.6. The
machine was faster than in the third run: llama.cpp and MLX-LM decode at 22 to 25 tok/s.

| Prompt | Metric | llama.cpp | MLX-LM | tern | tern vs llama.cpp | tern vs MLX-LM |
| --- | --- | --- | --- | --- | --- | --- |
| free_text (38 tokens) | decode, tok/s | 22.0 | 24.9 | 26.7 | +21% | +7% |
| | prefill, tok/s | 104.1 | 77.1 | 102.4 | -2% | +33% |
| | first token, s | 0.37 | 0.49 | 0.37 | +2% | -25% |
| code_edit (427 tokens) | decode, tok/s | 22.3 | 23.5 | 62.3 | +180% | +165% |
| | prefill, tok/s | 169.2 | 132.0 | 117.2 | -31% | -11% |
| | first token, s | 2.52 | 3.23 | 3.64 | +44% | +13% |
| doc_qa (423 tokens) | decode, tok/s | 22.6 | 24.8 | 30.2 | +33% | +22% |
| | prefill, tok/s | 167.9 | 147.0 | 116.1 | -31% | -21% |
| | first token, s | 2.52 | 2.88 | 3.64 | +45% | +27% |

Single runs of decode, tok/s, in the order of the rounds: llama.cpp 26.0, 21.8, 22.0 on
free_text, 25.4, 22.2, 22.3 on code_edit, 21.8, 22.6, 23.3 on doc_qa; MLX-LM 15.4, 26.1, 24.9,
then 13.4, 23.5, 26.3, then 26.0, 24.8, 22.6; tern 25.6, 29.0, 26.6, then 64.8, 62.3, 61.9, then
31.0, 30.2, 24.9. On doc_qa two of the three tern runs are above every run of the other two
engines and the third (24.9) is inside their range, so the lead there is real but smaller than
+40%. On free_text no pass runs. Prefill and first token are not touched by this work.

| Goal | Baseline | Ours | Difference | Reached | Where we lose |
| --- | --- | --- | --- | --- | --- |
| llama.cpp (Metal): decode +40% | 22.0 / 22.3 / 22.6 tok/s | 26.7 / 62.3 / 30.2 tok/s (fourth run) | +21% / +180% / +33% | code_edit only | Plain decode is bound by reading the weights. Speculation with prompt lookup gains where the output repeats the prompt: fully on code edit, partly on document QA. Free text needs a draft model |
| MLX-LM: decode +40% | 24.9 / 23.5 / 24.8 tok/s | 26.7 / 62.3 / 30.2 tok/s (fourth run) | +7% / +165% / +22% | code_edit only | Same |
| MLX-LM: prefill at 160 to 210 tok/s, then faster | 77.1 / 132.0 / 147.0 tok/s | 102.4 / 117.2 / 116.1 tok/s (fourth run) | +33% / -11% / -21% | No | Batched GEMM at 4.2 to 4.4 T products/s, tail tokens of a prompt, per-pass overhead. llama.cpp reached 168 to 169 tok/s on the long prompts in this run |
| First token within 1.8 s | MLX-LM 0.49 / 3.23 / 2.88 s; llama.cpp 0.37 / 2.52 / 2.52 s | 0.37 / 3.64 / 3.64 s (fourth run) | Long prompts: 13 to 45% slower | No | Follows from prefill |
| vLLM, SGLang: throughput, TTFT, p50/p99 on a server GPU | not measured | not measured | n/a | not measured | No NVIDIA GPU here; these engines do not run on a Mac |
| MLC LLM, ExecuTorch: +40% tok/s on a phone | not measured | not measured | n/a | not measured | No phone-class device here; no estimates given |

Correctness: tern produces the same 200 greedy tokens as MLX-LM on all three prompts, in every
run of every comparison, with and without speculation. llama.cpp produces the same text as MLX-LM, and the same number of prompt
tokens.

## Where the time of a token goes (decode)

Per token at position 200, `tern prof`, GPU time, taken while the GPU was not throttled.

| Part | ms | Share |
| --- | --- | --- |
| Reading the weights and scales (6.8 GB, the memory floor measured with `probe_mem_sg`) | 18.3 | 73% |
| GEMV arithmetic above the floor | 1.4 | 6% |
| Delta-rule layers: conv, q/k norm, state update, output norm | 2.0 | 8% |
| Full attention: prep, scores, softmax, value sum | 1.0 | 4% |
| Add + RMSNorm (129 dispatches) | 0.9 | 4% |
| SwiGLU, argmax, embedding | 0.3 | 1% |
| Between kernels (whole token minus the sum of its parts) | 1.3 | 5% |
| **GPU, whole token** | **about 25** | |
| CPU: encoding the token's commands | 0.13 to 0.19 | |
| Wall time not covered by GPU execution | about 7% | |

GEMV by matrix: MLP up/gate 8.3 ms, MLP down 4.9, layer input projections 4.0, output
projections 1.7, head 0.9; floor for the same matrices 7.5 to 8.0, 4.0 to 4.2, 3.5 to 3.7,
1.5 to 1.6, 0.8.

So the split asked for in the brief, as shares of wall time: memory 68%, GPU arithmetic and
dispatch 25%, CPU below 1%, waiting and scheduling about 7%. The same split for llama.cpp and MLX-LM was not taken
(no Metal System Trace in this session); for them only end-to-end speeds are measured.

Batched pass (prefill), per token at batch 32: GEMM 6.6 ms of 7.2 ms (MLP 3.1, MLP down 1.6,
input projections 1.3, output projections 0.5), delta-rule update 0.3, everything else 0.3. It is
bound by arithmetic: 4.2 to 4.4 T weight-token products per second.

## Candidates

Accepted, each with tokens equal to MLX-LM on the three prompts:

| Candidate | Before | After | Conditions |
| --- | --- | --- | --- |
| GEMV: mask-in-place unpacking, 32 lanes over the columns, 4 rows per thread (16 for wide matrices) | Token 36 to 45 ms; MLP 13 to 20 ms, MLP down 10 to 11 ms | Token 26.5 ms; MLP 8.3 ms, MLP down 4.9 ms; GEMV at 93% of the memory floor | `tern prof`, best of 5, same hour, GPU not throttled |
| Batched prefill, lane GEMM (2 rows x 4 tokens per thread), 32 tokens per pass | 35.0 tok/s, 12.2 s to first token | 62.4 tok/s, 6.8 s | code_edit, back to back |
| Tiled GEMM: 64 rows x 32 tokens per threadgroup, 8 x 8 matrix intrinsics | 50.2 tok/s, 8.51 s | 107.0 tok/s, 3.99 s | code_edit, 3 interleaved rounds, medians, load 4 to 71 |
| Column split of the tiled GEMM for matrices with few rows | 107.0 tok/s, 3.99 s | 117.1 tok/s, 3.65 s (+9%) | same run |
| Lane split of the delta-rule state update and of the softmax | Update 3.09 ms, softmax 1.20 ms per token (2.55 ms at position 600) | 1.17 ms and 0.08 ms (0.13 ms at position 600); decode +3% on free_text, +1% on doc_qa | `tern prof` in the same minute; decode from 3 interleaved rounds on a throttled GPU, load 6 to 18 |
| Token id kept on the GPU, commands encoded ahead | n/a | Lookahead 1, 2 and 4 give the same speed: the CPU is not the limit | Kept as the design, no gain claimed |
| Weight pages touched by a GPU kernel at load | First tokens paid for page wiring | Paid in load time (0.85 s) | Not measured separately |
| Speculative decoding: prompt-lookup drafts, passes of 4, 8 and 32 tokens, states not written by the pass and replayed for the accepted tokens | Decode 20.4 / 19.0 / 19.1 tok/s (free_text / code_edit / doc_qa) | 20.8 / 41.0 / 18.8 tok/s: 0% (no pass ran), +116%, -2% | `--spec 0` against `--spec 32 --gemm-tile8 0`, 3 interleaved rounds, medians, load 54 to 151, GPU in its slow state |
| 8-token tile for passes of 8 to 15 tokens | 20.8 / 41.0 / 18.8 tok/s with the lane GEMM for 8 tokens | 20.4 / 43.7 / 20.6 tok/s: +7% on code_edit, +10% on doc_qa; against no speculation 0% / +131% / +8% | same run |

| Verification pass of 2 tokens as the first pass size (lane GEMM with 2 rows x 2 tokens per thread; sizes 2, 4, 8, 32 instead of 4, 8, 32) | doc_qa 27.2 tok/s, code_edit 63.3 tok/s; without speculation 25.1 and 28.8 | doc_qa 33.0 tok/s (+21%; +31% over no speculation), code_edit 64.2 (+1%; +123%); free_text unchanged, no pass runs | Build of `main` against this build against `--spec 0`, 3 interleaved rounds, medians, load 12 to 39. Repeated against `--spec 0` on doc_qa: +10% (load 18 to 176) and +14% (5 rounds, load 9 to 14), so +31% was the high end |

The lane split is below the 10% bar on the bench prompts; it is kept because its gain grows with
the context length and the tokens are unchanged. Say so if the bar should apply to it as well:
`--gdn-sg 0 --attn-sg 0` restores the serial kernels.

Rejected:

| Candidate | Result |
| --- | --- |
| Several rows per thread without the lane split | Slower than one row per thread |
| Table lookup GEMV (256 entries per 4 columns) | 4.1 to 5 times slower: gathers from a 1.3 MB table |
| MLX-style unpacking, sum(q * x) minus sum(x) | Fast, but the tokens differ from MLX-LM at token 110 of free_text |
| Lane GEMM with 8 or 16 tokens per thread, or 4 rows | 1.1 to 1.7 T products/s against 1.9 to 2.2 T for 2 rows x 4 tokens |
| Lane GEMM with named accumulators, token-interleaved input | No repeatable gain |
| Lane split of the attention value sum together with the delta-rule split | Each alone keeps the tokens, both together flip token 110 of free_text |
| Tiled GEMM below 16 tokens | A pass costs the same from 1 to 32 tokens (about 8 sequential tokens); the lane GEMM is cheaper up to 8 |
| Prefill passes of 64 or 128 tokens | +5% prefill (112 against 107 tok/s, 3 interleaved rounds); below the bar, default stays 32 |
| Lane GEMM, weights unpacked through a table of 256 entries (constant array or buffer) | 0.130 to 0.149 ms per token for one MLP matrix against 0.083: 1.5 to 1.8 times slower |
| Lane GEMM, inputs converted to float and divided by the slot weight ahead of the product | 0.16 to 0.24 ms against 0.085: 2 to 3 times slower |
| Lane GEMM over tables of partial sums of the inputs, one lookup per byte of weights | 0.29 ms against 0.083: 3.5 times slower, gathers from a 5 MB table |
| Lane GEMM with weights and inputs held as half, or with the rows as the outer loop | Within the noise of 10% |
| Unpacking with one multiply-add per weight in the lane and tile kernels; the sum of the inputs subtracted once in the GEMV | Lane -2 to -10%, 8-token tile -3%, 32-token tile 0%, GEMV -1 to -7% (`tern bench`, best of 3 interleaved), with single results of +28% and +43% on the small matrix: inside the noise, not adopted |
| 2-token lane GEMM with 1 or 4 rows per thread | Pass of 31.6 and 30.3 ms against 28.9 to 29.7 ms with 2 rows |
| A pass of 3 tokens | 45 to 52 ms, 1.7 to 1.9 plain tokens |
| Pass sizes 2, 8, 32 without the size 4 | doc_qa 31.0 against 33.0 tok/s, code_edit 52.2 against 64.2 (same run as the accepted 2-token pass) |
| Draft model Qwen3.5-0.8B in tern (`--draft`): where prompt lookup has no match, a pass of 2 with one token of the draft model | Decode -4% / -1% / 0% on free_text / code_edit / doc_qa against the same build without it, prefill -8 to -18%; tokens unchanged. 3 interleaved rounds, medians, load 42 to 108. Kept as an option, off by default; details under "Draft model in tern" |
| Shorter prompt match before a pass starts (3, 2 or 1 tokens instead of 4) | 3: doc_qa 29.6 against 28.8 tok/s, code_edit 59.9 against 59.7, free_text 26.7 against 26.9, inside the noise. 2: code_edit 53.7. 1: code_edit 44.6, free_text 24.3 against 26.9 without speculation. 3 interleaved rounds, medians, load 18 to 176 |

## The gap to +40%

**Decode.** In the best state both MLX-LM and tern ran at about 38 tok/s, 26 ms per token. +40%
is 53 to 54 tok/s, 18.6 ms per token. Reading the weights alone takes 18.3 ms. So +40% without
speculation would need every other part of the token to cost nothing; it is not reachable with
kernels. What kernels can still give:

- Fusing the small kernels and the GQA-aware value sum: about 1.5 to 2.5 ms of the 25, up to +10%.
- Dense trit packing (1.6 bits per weight): the floor drops from 18.3 to about 14.6 ms if the
  decode stays cheap, about +15%. Not built, not measured.

Together that is about 45 to 50 tok/s against 38, +20 to +30% on paper, and each part still has
to survive the token check.

**Speculation** is the only route to +40%. It is built for one draft source, prompt lookup:
when the last 4 or more tokens of the context occurred earlier, the tokens that followed there
are verified in one batched pass of 2, 4, 8 or 32 tokens. The pass does not write the states of the linear layers; the
accepted tokens are replayed through the state kernels alone (144 small dispatches), and the KV
cache needs nothing. The output equals plain greedy decoding on all three prompts.

Price of a pass. GPU time from `tern prof` at load 8 to 25, with a plain token at 26.1 to
27.4 ms:

| Tokens in the pass | 2 | 4 | 8 | 32 |
| --- | --- | --- | --- | --- |
| GPU time of the pass, ms | 28.9 to 29.7 | 53.6 | 88 | 203 |
| Matrix products per token, ms | 12.0 | 11.9 | 10.0 | 5.6 |
| Wall time inside a generation run, ms | 40 to 52 (plain token at 30 to 40) | 58 | 97 | 220 |
| In plain tokens | 1.1 to 1.3 | 2.1 | 3.5 | 8.0 |

The wall time adds the head, the argmax and the replay of the accepted tokens; nothing is lost
on the CPU side. In the slow state of the machine the passes of the third run cost 2.8 to 3.0,
3.8 to 4.1 and 7.6 to 8.9 plain tokens: the batched products are bound by arithmetic and slow
down more than the plain token, which is bound by memory.

The 2-token pass is nearly free because its products are still close to the memory bound: two
tokens take 24 ms of matrix products against 19.7 ms for one. From 4 tokens on the unpacking of
the 2-bit weights sets the time. In the 4-token lane kernel, leaving out the loads of the inputs
or the unpacking alone changes nothing (0.077 and 0.086 ms per token for one matrix against
0.083 to 0.085), leaving out both halves it (0.043 ms, 4.2 T products per second, the rate of
the 32-token tile). The variants of the unpacking that were tried are in the table of rejected
candidates; none was faster, so the 4-token and 8-token prices stand.

A pass pays off when it yields more tokens than its price, so the depth starts at 2, moves one
size up after a fully accepted pass, falls back to what was accepted otherwise, and a pass that
yields at most two tokens without being full is followed by a pause.

- code_edit: 14 passes produce 175 of the 200 tokens, five of them 32-token passes with 135
  tokens. Decode +123% against the same build without speculation (load 12 to 39), +141% in a
  later run at load 18 to 176.
- doc_qa: 22 passes produce 79 tokens: 13 from seven 2-token passes, 29 from eight 4-token
  passes, 32 from six 8-token passes. Decode +31% against no speculation and +21% against the
  passes of 4, 8 and 32 in the first run (load 12 to 39). That run was at the high end: a
  later one gave +10% (load 18 to 176), and a third, 5 rounds at load 9 to 14 during which the
  machine went from its fast to its slow state, gave per-round ratios of 1.14, 1.51, 1.06, 1.24
  and 0.99, median +14%. The pass prices of the table predict +10%
  (`bench/draft_accept.py`). So the 2-token pass is worth about +10 to +15% on doc_qa, not 31%.
  A gate that knew the outcome of every pass in advance would reach +31% with these drafts.
- free_text: the context never repeats 4 tokens, no pass runs, 0%. Starting passes at shorter
  matches loses there (table of rejected candidates).

**Draft model, measured offline.** Qwen3.5-0.8B in MLX 4-bit
(`mlx-community/Qwen3.5-0.8B-4bit`, 0.63 GB), run with mlx-lm over each prompt and the 200
reference tokens of the 27B model in one forward pass (`bench/draft_accept.py`,
`bench/results/draft_accept.json`). For a greedy draft the accepted runs this gives are exact.
Same tokenizer: the rendered prompts encode to the same ids. Load 8 to 18.

| | free_text | code_edit | doc_qa |
| --- | --- | --- | --- |
| First draft token equal to the reference | 53% | 98% | 88% |
| The same where prompt lookup has no match | 53% | 95% | 81% |
| Mean accepted run | 0.8 | 25.0 | 10.3 |

The draft model costs 2.75 to 3.04 ms per token in mlx-lm, d = 0.10 to 0.11 plain tokens. Its
runs turned into decode speed with the pass prices above, against plain decoding (1.00):

| Draft source and gate | free_text | code_edit | doc_qa |
| --- | --- | --- | --- |
| Prompt lookup, ladder 2, 4, 8, 32 (today) | 1.00 | 2.39 | 1.10 |
| Draft model, one draft token and a pass of 2 at every position | 1.22 | 1.58 | **1.49** |
| Prompt lookup with the ladder where it matches, else draft model with a pass of 2 | **1.22** | **2.57** | 1.34 |
| Draft model with the ladder of today | 1.01 | 2.01 | 1.24 |
| Draft model, draft as long as the product of its probabilities stays above 0.7 | 1.06 | 2.24 | 1.39 |
| Best possible gate over prompt lookup and draft model (oracle) | 1.36 | 3.13 | 1.88 |

With d = 0.07 and 0.15 instead of the measured value, the rows move by 0.03 to 0.05 (all values
in the JSON file). What this says:

- The draft model is good enough for document QA: its 2-token passes alone give +49% over plain
  decoding, more than prompt lookup (+10%) and more than lookup followed by the model (+34%),
  because a wrong lookup draft in a 4-token or 8-token pass costs more than a model draft in a
  pass of 2.
- On free text the small model agrees with the 27B model on only 53% of the tokens. The best
  simple gate gives +22%, the oracle +36%: free text stays below +40% over plain decoding even
  with a perfect gate.
- The ladder of today does not suit a draft model: it climbs to passes of 4 and 8 after two
  accepted tokens and loses there. The gap to the oracle (1.49 against 1.88 on doc_qa) is in the
  gate: a pass size chosen from the draft model's own confidence did not close it.
- Not covered: the draft model inside tern (its cost there and its state after a rejected draft
  are estimates), numerics of the batched forward pass against step-by-step decoding, more
  prompts than these three.

**Draft model in tern.** `--draft models/qwen35-08b.tern` runs the 0.8B model next to the 27B
one, on the same device and in the same command buffers (`tools/convert.py` repacks it: 4-bit
affine weights with a scale and a bias per 64, the head shares the embedding table, 0.43 GB).
The engine is generic over the model shape; the 4-bit products have kernels of their own
(`gemm_a4_*`: one, two or 8 tokens per thread, any batch size). The draft model reads the
token ids of the main model and only ever sees confirmed tokens, so it needs no rollback: before
each pass of 2 it catches up on the tokens accepted since its last step, in one batch, and its
pick for the next token becomes the draft. Prompt lookup keeps its ladder where it matches; a
draft-model pass counts against the pause like plain tokens. Tokens: equal to MLX-LM on all three
prompts in every run.

Acceptance inside tern is the offline one: 68 of 131 drafts on free_text (52%), 52 of 63 on
doc_qa (83%), 11 of 12 on code_edit. Speed, interleaved against the same build without the draft
model, 3 rounds, medians, load 42 to 108:

| Prompt | Decode without / with, tok/s | Prefill without / with, tok/s | First token without / with, s |
| --- | --- | --- | --- |
| free_text | 21.2 / 20.4 (-4%) | 59.2 / 48.8 | 0.64 / 0.78 |
| code_edit | 46.9 / 46.4 (-1%) | 87.3 / 80.4 | 4.89 / 5.31 |
| doc_qa | 20.9 / 20.9 (0%) | 82.1 / 71.5 | 5.15 / 5.91 |

Single runs of decode: free_text 24.9, 21.2, 17.7 without and 23.5, 20.4, 19.7 with; doc_qa
23.3, 20.9, 17.6 and 23.7, 19.3, 20.9. No gain at this load. Where the time goes (free_text,
load 41 to 191):

- The draft model is cheap: 3.5 ms of GPU time for a token with its head, 1.3 ms per token in a
  batch of 2 (`tern prof models/qwen35-08b.tern 200 1`), 0.07 to 0.12 plain tokens.
- The pass of 2 of the main model is not: inside a generation run a draft-model pass took 81 to
  100 ms of wall time and 74 to 94 ms of GPU time against 48 to 50 ms for a plain token, 1.7 to
  2.1 plain tokens, where the simulation assumed 1.2 to 1.4. `tern prof` in the same minutes gives
  1.1 to 1.5 plain tokens for the pass of 2 alone: in the loaded state the batched products slow
  down more than the memory-bound single token, as noted for the third run.
- 6 to 9 ms per pass is idle time between command buffers: a pass is not pipelined, the CPU
  learns the outcome before it can encode the next step, while plain tokens keep two command
  buffers queued. Under load the GPU goes to other work in that gap.
- With 1.52 tokens per pass on free_text, a pass has to cost less than 1.5 plain tokens; it cost
  1.7 to 2.1. On doc_qa the passes produce 1.83 tokens each and break even.
- The draft model's prefill costs 0.9 ms per prompt token against 9.6 ms for the main model:
  +9% on the prefill, -8 to -18% in the table above, where the load was higher in the draft runs.

At the pass price of a quiet machine (1.2 to 1.3 plain tokens plus the draft) the same
acceptance would give about +10% on free_text and +20 to +30% on doc_qa, the simulation's
values. That is an estimate; the measurement on a quiet machine is open ([T1](plan/T1/task.md)).
Two things would make the pass cheaper in any state: deciding the acceptance on the GPU so that
the next step can be queued before the CPU sees the outcome, and a pass of 2 closer to the
memory bound.

**Prefill.** From 35 tok/s at the start of the session to 92 to 128 tok/s depending on the state
of the machine; under identical conditions still 25 to 31% behind llama.cpp and 13 to 21% behind
MLX-LM on the long prompts, and ahead of MLX-LM on the short one. Known losses: the last tokens
of a prompt that do not fill a tile (10 of 427 tokens take about 6% of the time), 32 tokens per
pass (64 gives +5%, measured, not adopted yet), and the GEMM itself. One observation without an
explanation yet: tern's prefill fell from about 120 to about 95 tok/s when the machine went from
its medium to its slow state, more than llama.cpp's did.

**Honest summary.** Decode against llama.cpp and MLX-LM in the fourth run: +180% and +165% on
code edit, where the output repeats the prompt; the goal is reached there. On document QA +33%
and +22%: speculation with prompt lookup is worth +10 to +15% there (one run gave +31%, two
later ones +10% and +14%), the rest of the lead is plain decode; the goal is not reached. On free
text +21% and +7%, all of it plain decode, since no pass runs. The missing points need a draft
model. Qwen3.5-0.8B, measured offline, is good enough for document QA (+49% over plain decoding
in simulation with 2-token passes) and not for free text (+22%, at most +36% with a perfect
gate). It now runs inside tern (`--draft`) with unchanged tokens and the offline acceptance, but
at the load of this session (42 to 108) it gains nothing: the pass of 2 cost 1.7 to 2.1 plain
tokens instead of 1.2 to 1.4, mostly in the main model's pass, 6 to 9 ms of it idle time
between command buffers. It stays off by
default. Prefill: 31% behind llama.cpp and 11 to 21% behind MLX-LM on the long
prompts. First token within 1.8 s: not reached on the long prompts.

## What was not done

- A run with load below 4 ([T1](plan/T1/task.md)).
- `--draft` on a quiet machine ([T1](plan/T1/task.md)): its gain is estimated from pass prices
  measured earlier, not measured.
- Acceptance decided on the GPU, so that a pass can be pipelined like plain tokens
  ([T3](plan/T3/task.md)).
- A faster prefill of the draft model (0.9 ms per prompt token, +9% on the prefill).
- A cheaper 4-token or 8-token pass: every kernel variant tried was slower or inside the noise.
- Kernels for Q8_0 and f16 ([T2](plan/T2/task.md)), fused small kernels ([T5](plan/T5/task.md)),
  the backend interface ([T6](plan/T6/task.md)).
- The time split of llama.cpp and MLX-LM themselves (Metal System Trace).
- The cold-start (`--nvme`) path of the earlier prototype.
