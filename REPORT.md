# Report

State on 2026-11-22, after speculative decoding ([T3](plan/T3/task.md)). Machine: MacBook Pro M3 Max (16 CPU cores, 40-core GPU, 64 GB), macOS 27.0.1,
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

| Goal | Baseline | Ours | Difference | Reached | Where we lose |
| --- | --- | --- | --- | --- | --- |
| llama.cpp (Metal): decode +40% | 14.8 / 14.6 / 14.2 tok/s | 17.5 / 35.6 / 17.1 tok/s (third run) | +19% / +145% / +21% | code_edit only | Plain decode is bound by reading the weights. Speculation with prompt lookup gains where the output repeats the prompt; free text and document QA need a draft model |
| MLX-LM: decode +40% | 14.7 / 15.5 / 15.4 tok/s | 17.5 / 35.6 / 17.1 tok/s (third run) | +19% / +130% / +12% | code_edit only | Same |
| MLX-LM: prefill at 160 to 210 tok/s, then faster | 64.9 / 114.8 / 116.9 tok/s | 99.8 / 99.4 / 92.4 tok/s | +54% / -13% / -21% | No | Batched GEMM at 4.2 to 4.4 T products/s, tail tokens of a prompt, per-pass overhead. Nobody reached 160 tok/s in this run |
| First token within 1.8 s | MLX-LM 0.59 / 3.72 / 3.62 s; llama.cpp 0.37 / 3.22 / 3.16 s | 0.38 / 4.29 / 4.58 s | Long prompts: 15 to 45% slower | No | Follows from prefill |
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
are verified in one batched pass. The pass does not write the states of the linear layers; the
accepted tokens are replayed through the state kernels alone (144 small dispatches), and the KV
cache needs nothing. The output equals plain greedy decoding on all three prompts.

Price of a pass, measured inside the generation runs (slow state, a plain token at about 50 ms):
138 to 150 ms for 4 tokens (2.8 to 3.0 plain tokens), 188 to 204 ms for 8 (3.8 to 4.1), 382 to
443 ms for 32 (7.6 to 8.9). A pass pays off when it yields more tokens than that, so the depth
starts at 4, moves up after a fully accepted pass, and a pass that yields at most two tokens is
followed by a pause.

- code_edit: 12 passes produce 175 of the 200 tokens, six of them 32-token passes with 142
  tokens. Decode +116 to +131% against the same build without speculation.
- doc_qa: 18 passes produce 68 tokens, 3.4 per 4-token pass and 4.4 per 8-token pass, which is
  about their price: +8%. With these drafts and these prices even a gate that knew the outcome
  of every pass in advance would reach +36% (computed on the reference tokens). The answer
  paraphrases the document, so +40% here needs a better draft, not a better gate.
- free_text: the context never repeats 4 tokens, no pass runs, 0%.

What is missing for the other two prompts is a draft model: a small model of the same tokenizer
that proposes tokens where the prompt does not. Its cost comes on top of the pass, so on free
text it has to get about three of four drafts accepted to gain anything at a 4-token pass.

**Prefill.** From 35 tok/s at the start of the session to 92 to 128 tok/s depending on the state
of the machine; under identical conditions still 25 to 31% behind llama.cpp and 13 to 21% behind
MLX-LM on the long prompts, and ahead of MLX-LM on the short one. Known losses: the last tokens
of a prompt that do not fill a tile (10 of 427 tokens take about 6% of the time), 32 tokens per
pass (64 gives +5%, measured, not adopted yet), and the GEMM itself. One observation without an
explanation yet: tern's prefill fell from about 120 to about 95 tok/s when the machine went from
its medium to its slow state, more than llama.cpp's did.

**Honest summary.** Decode: +130 to +145% over both engines on code edit, where the output
repeats the prompt; that goal is reached there. On document QA +12 to +21% and on free text +19%,
both in a slow state of the machine in which tern's plain decode is ahead by about that much, so
speculation itself adds +8% and 0% there. In the fast state plain decode was level with MLX-LM,
and that measurement has not been repeated. The 40 points on those two prompts need a draft
model. Prefill: 2.3 times faster than the first batched kernel, still 22 to 32% behind llama.cpp
and 5 to 28% behind MLX-LM in the third run. First token within 1.8 s: not reached on the long
prompts.

## What was not done

- A run with load below 4 ([T1](plan/T1/task.md)).
- Speculative decoding ([T3](plan/T3/task.md)) beyond prompt lookup: no draft model, so nothing
  is gained where the output does not repeat the prompt.
- Kernels for Q8_0 and f16 ([T2](plan/T2/task.md)), fused small kernels ([T5](plan/T5/task.md)),
  the backend interface ([T6](plan/T6/task.md)).
- The time split of llama.cpp and MLX-LM themselves (Metal System Trace).
- The cold-start (`--nvme`) path of the earlier prototype.
