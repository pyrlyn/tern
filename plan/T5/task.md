---
status: todo
priority: P2
complexity: 3
readiness: 40%
agent:
---

# T5. Fused Metal kernels

One command buffer per token is done, the token id stays on the GPU and the CPU encodes ahead;
the GPU is busy 93% of the wall time and encoding costs 0.13 to 0.19 ms per token. What is left
is the small kernels between the GEMVs: add + RMSNorm (129 dispatches, 0.89 ms per token), the
delta-rule chain (conv, q/k norm, step, output norm) and the attention chain. Each dispatch has a
fixed latency, so merging neighbours should recover 1 to 2 ms of the 26.5 ms token.

Done means: fewer dispatches per token with a measured gain in `tern prof` and the same tokens.

Check: `tern prof <model> 200 1` shows the non-GEMV sum at least 1 ms lower than before and `tern_bench.py` reports `same as MLX: True` on all prompts.
