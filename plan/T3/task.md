---
status: in progress
priority: P0
complexity: 5
readiness: 75%
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
6. Lower the price of a short pass: a 2-token pass with a lane kernel of its own as the first
   pass size (`gemm_t2_r2b2`, sizes 2, 4, 8, 32 in `src/draft.zig`). Kernel variants for a
   cheaper 4-token and 8-token pass were measured and rejected (REPORT.md).
7. Open: a draft that works where the prompt is not repeated (free text, the rest of document
   QA). The candidates and the acceptance they need are in
   [RESEARCH.md](../../RESEARCH.md#draft-sources-for-ternary-bonsai-27b): Qwen3.5-0.8B, the MTP
   head of Qwen3.5-27B, Ternary-Bonsai-2-2B. With the 2-token pass a draft model breaks even
   at 25 to 30% of first draft tokens accepted and reaches +40% at 71 to 82%, so:
   1. Done: acceptance of Qwen3.5-0.8B measured offline (`bench/draft_accept.py`, REPORT.md):
      first draft token right in 53% / 98% / 88% of the positions on free text / code edit /
      document QA. Simulated with the measured prices: one draft token in a pass of 2 where
      lookup has no match gives 1.22 / 2.57 / 1.34 times plain decoding, the model alone 1.49
      on document QA.
   2. Next: run the draft model in the engine (`--draft <model>`): the model shape at run time,
      a product kernel for 4-bit affine weights ([T2](../T2/task.md)), the draft's own state
      kept in step with the accepted tokens, the pass-of-2 gate. Measured against the
      simulation, interleaved.

State: steps 1 to 6 are done. `bench/compare.py`, 3 rounds, with the 2-token pass: decode against
llama.cpp and MLX-LM +180% and +165% on code_edit, +33% and +22% on doc_qa, +21% and +7% on
free_text, `matches_mlx` true everywhere. The check below holds for code_edit and free_text, not
for doc_qa.

Check: `bench/compare.py` shows the tern decode column at +40% or more on code_edit and doc_qa, at 0% or more on free_text, with `matches_mlx` true.
