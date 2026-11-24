# tern

Inference runtime for ternary LLMs on Apple Silicon, written in Zig.
[pyrlyn/tern](https://github.com/pyrlyn/tern); the working copy lives in `~/apps/tern`.

Tasks are the folders next to this file. Numbers and the gap to the goals are in
[REPORT.md](../REPORT.md), the technique survey in [RESEARCH.md](../RESEARCH.md).
The cross-platform product plan, per-platform baselines and the **`>= 2x`**
speed bar are in [docs/cross-platform-plan.md](../docs/cross-platform-plan.md)
(T7 macOS, T8 Linux, T9 Windows). The compile-time backend interface (T6) is
specified in a separate PR, [pyrlyn/tern#8](https://github.com/pyrlyn/tern/pull/8).

## Speed bar

On macOS, Linux and Windows, tern must be **`>= 2x`** as fast as that
platform's fastest realistic baseline on every harness benchmark (decode,
prefill, TTFT) and every prompt, with greedy tokens unchanged. Details,
baseline research and the PR-only workflow are in the document above. The
older +40% wording in README.md and REPORT.md is superseded for new work.


## MoE, SSD expert streaming and DSpark block drafts (T10 to T27)

Goal for this track: **2x Atomic Engine** (published 105.71 tok/s, Inkling 2 Mini IQ2_M + DSpark,
14 GiB, M4 Max 48 GB; gated same-machine on the M3 Max). The design, model revisions, `.tern` v2
format, streaming/cache/budget design, DSpark facts and the ordered list of implementation PRs are
in [moe-dspark.md](moe-dspark.md). Tasks: T10 Atomic baselines, T11 `.tern` v2, T12 Qwen3.5-35B-A3B
converter, T13 resident MoE decode, T14 batched MoE pass, T15 SSD streaming, T16 LRU/ClassLRU,
T17 prefetch, T18 memory budget, T19 live shrink, T20 Inkling formats, T21 Inkling architecture,
T22 hot-core residency, T23 block verify, T24 Bonsai DSpark, T25 MoE DSpark, T26 block-8 head
training (needs the creator's go), T27 the 2x gate.

## Decisions

- The project is in Zig and lives in `~/apps/tern` (the creator's word; it replaces the
  `~/Grok/<task>` rule of the brief). Converted models are in `models/`, which is not tracked.
  The original model in LM Studio is only read.
- Correctness rule: greedy tokens equal to MLX-LM on the three bench prompts. llama.cpp
  (PrismML fork, the only one that reads the PQ2_0 GGUF) produces the same text as MLX-LM.
- Activations are stored as half floats and every sum is computed in float, rounded to half
  where MLX rounds. Changing the order of a sum is allowed only after the token check.
- A candidate is kept only with a before/after measurement; rejected ones are listed in the
  report with their numbers.

- Speed bar for new work: **`>= 2x`** the fastest baseline on each of macOS,
  Linux and Windows, on every harness metric and prompt (see
  [docs/cross-platform-plan.md](../docs/cross-platform-plan.md)).
- All changes go through pull requests. Merge only on an explicit request, and
  only with green CI.

## Findings

- Token 110 of the free-text prompt is a near-tie in the logits. Two summation-order changes that
  each keep the reference tokens (lane-split delta-rule step, lane-split attention value sum)
  flip it when combined. The shipped combination is delta-rule step plus softmax. More prompts
  are needed before any further order change is trusted.
- Decode is at the memory floor (GEMV 19.7 ms against 18.3 ms for reading the weights alone), so
  kernels cannot give +40%. The batched pass is compute-bound instead.
- The machine was never quiet during the measurements. The GPU throttles for minutes and all
  three engines slow down together, so only interleaved runs are comparable.
- Speculation with prompt-lookup drafts keeps the reference tokens on the three prompts although
  the batched pass sums in another order than the single-token pass. It more than doubles decode
  where the output repeats the prompt (code edit) and gives nothing where it does not: an oracle
  gate over the same drafts stays below +40% on document QA. The next gain there is a draft
  model, not a better gate.
- A verification pass of 2 tokens costs about as much as one plain token (1.06 to 1.09 times by
  GPU time): its products are still bound by reading the weights. From 4 tokens on the
  arithmetic sets the time, and no variant of the weight unpacking tried so far made that
  cheaper. One draft token that is right three times out of four is therefore enough for +40%.
- The price of a pass in plain tokens depends on the state of the machine: the batched products
  are compute-bound and slow down more under load than the memory-bound plain token (a 4-token
  pass was 2.1 plain tokens at load 8 to 25 and 2.9 in the slow state).
- Qwen3.5-0.8B as a draft for Ternary Bonsai 27B, measured offline: its greedy token equals the
  27B reference on 53% of free-text positions, 88% on document QA, 98% on code edit. The right
  gate for it is one draft token in a pass of 2, not the ladder of prompt lookup. Free text
  stays below +40% over plain decoding with this draft even under an oracle gate (+36%).
- Inside tern the 0.8B draft costs 0.07 to 0.12 plain tokens and keeps its offline acceptance,
  but under load a pass of 2 cost 1.7 to 2.1 plain tokens and the draft model gained nothing.
  Chaining the passes with the acceptance on the GPU removed the CPU readback but not the price:
  the pass is GPU-bound at 1.45 to 1.6 plain tokens. Its value depends on that price, so on the
  state of the machine.
- The pass of 2 has no cheap part left: profiled by piece on a real prompt it is 1.50 to 1.55
  plain tokens once the GPU has slowed down, and the block of two tokens itself is 1.33 to 1.40
  (1.16 to 1.29 on a rested GPU). Commit and draft model are worth 3% each at most.
- GPU timings depend on what ran just before: a plain token takes 28 to 30 ms after a pause and
  48 to 52 ms right after two passes. Pieces are comparable only after the same warm-up and in
  a shuffled order (`tern prof --draft` does both).
- The draft model's gate needs a margin and few trials. Timed inside one run, a free-text token
  costs from 4% less to 13% more through a pass than plain, and without a margin the gate
  followed that noise into the passes. Each trial of the slower way costs: on doc_qa 16 plain
  tokens per 48 took 3% off the run.
