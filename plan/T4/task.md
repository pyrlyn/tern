---
status: todo
priority: P0
complexity: 4
readiness: 60%
agent:
---

# T4. Prefill at MLX-LM level and below

Target from the brief: at least 160 to 210 tok/s and a time to first token of 1.8 s or less on
the long prompts, then faster than MLX-LM. The tiled GEMM (64 rows x 32 tokens per threadgroup,
8 x 8 matrix intrinsics, column split for matrices with few rows) took prefill from about 50 to
92 to 128 tok/s depending on the state of the machine; in the interleaved run that is still 25
to 31% behind llama.cpp and 13 to 21% behind MLX-LM on the long prompts. The GEMM runs at 4.2 to 4.4 T weight-token products per second, close to the GPU's
arithmetic limit, so the next step is less arithmetic or less overhead: a larger prefill chunk,
half-precision operands with float accumulation, fewer barriers per column block, first-run
costs moved to load time.

Done means: prefill and time to first token on code_edit and doc_qa are better than MLX-LM and
llama.cpp in an interleaved run, tokens unchanged.

Check: `bench/compare.py` shows tern ahead on `prefill_tps` and `ttft_s` for code_edit and doc_qa with `matches_mlx` true.
