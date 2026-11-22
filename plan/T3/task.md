---
status: todo
priority: P0
complexity: 5
readiness: 10%
agent:
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

Check: `bench/compare.py` shows the tern decode column at +40% or more on code_edit and doc_qa, at 0% or more on free_text, with `matches_mlx` true.
