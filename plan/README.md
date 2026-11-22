# tern

Inference runtime for ternary LLMs on Apple Silicon, written in Zig. No GitHub repository yet;
the code lives in `~/apps/tern`.

Tasks are the folders next to this file. Numbers and the gap to the goals are in
[REPORT.md](../REPORT.md), the technique survey in [RESEARCH.md](../RESEARCH.md).

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

## Findings

- Token 110 of the free-text prompt is a near-tie in the logits. Two summation-order changes that
  each keep the reference tokens (lane-split delta-rule step, lane-split attention value sum)
  flip it when combined. The shipped combination is delta-rule step plus softmax. More prompts
  are needed before any further order change is trusted.
- Decode is at the memory floor (GEMV 19.7 ms against 18.3 ms for reading the weights alone), so
  kernels cannot give +40%. The batched pass is compute-bound instead.
- The machine was never quiet during the measurements. The GPU throttles for minutes and all
  three engines slow down together, so only interleaved runs are comparable.
