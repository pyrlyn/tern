---
status: todo
priority: P1
complexity: 3
readiness: 50%
agent:
---

# T2. Multi-row GEMV for every weight format

The ternary GEMV is finished: mask-in-place unpacking, 32 lanes over the columns, 4 rows per
thread (16 for wide matrices), at 93% of the memory floor. runa also loads Q8_0 and f16 models,
and they still have no kernel here. The verify kernel of speculation should be the same code
path as the batched GEMM, not a second implementation.

Done means: GEMV kernels for Q8_0 and f16 with weights repacked in kernel order, each with a
before/after measurement against the thread-per-row reference and against the memory floor
probe, and the same greedy tokens as the reference engine for that format.

Check: `tern bench` on a Q8_0 and an f16 matrix reaches at least 85% of `probe_mem_sg`, and `tern_bench.py` reports the reference tokens.
