---
status: in progress
priority: P0
complexity: 5
readiness: 60%
agent: Claude Code / claude-fable-5-2
---

# T3. Speculative decoding that never loses

Plain decode is at the memory floor, so the +40% over llama.cpp and MLX-LM has to come from
reading the weights less than once per token. The batched pass exists (`Engine.encodeBlock`) and
its price is measured: a pass costs about as much as 2.8 sequential tokens for 4 tokens (lane
kernel), 4.5 for 8, and about 8 for anything from 16 to 32 (tiled kernel). Missing: the head and
argmax for every token of a pass, rollback of the delta-rule states and the KV cache after a
rejected draft, the draft sources (prompt lookup, a 0.5B draft model), adaptive depth and a gate
that turns speculation off when acceptance is low (free text lost 3% in the earlier prototype).

Done means: greedy output identical to the non-speculative run on the three bench prompts;
decode speed at least +40% over llama.cpp and MLX-LM on code edit and document QA, and not below
the non-speculative speed on free text.

## Execution plan

1. State rollback: `gdn_conv` and `gdn_step_sg` get a no-write mode; the input projections of
   the linear layers are kept per layer (`lin_proj`), and `Engine.encodeCommit` replays the
   accepted tokens with the same kernels. The KV cache needs nothing, rejected positions are
   overwritten.
2. Head and argmax for every token of a pass (`Engine.encodeVerify`, `pred` buffer).
3. Draft source and depth policy in `src/draft.zig` (prompt lookup; sizes 4, 8, 32; pause after
   a loss), wired into `tern gen` behind `--spec n`.
4. Verify: tokens equal to the reference on the three prompts with `--spec 32`; unit tests of
   `draft.zig` in CI.
5. Measure plain against speculative decode interleaved (medians, load given), record in
   REPORT.md.
6. Open: a draft that works where the prompt is not repeated (document QA, free text). With
   the measured pass prices the oracle bound of prompt lookup on doc_qa is +36%, so the goal
   there needs a draft model. Which model, and its download, are the creator's call.

State: steps 1 to 5 are done. `bench/compare.py`, 3 rounds: decode against llama.cpp and MLX-LM
+145% and +130% on code_edit, +21% and +12% on doc_qa, +19% and +19% on free_text, `matches_mlx`
true everywhere. The check below holds for code_edit and free_text, not for doc_qa.

Check: `bench/compare.py` shows the tern decode column at +40% or more on code_edit and doc_qa, at 0% or more on free_text, with `matches_mlx` true.
