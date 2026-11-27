---
status: todo
priority: P0
complexity: 2
readiness: 60%
agent:
---

# T12. Convert Qwen3.5-35B-A3B (MLX 4-bit) to .tern v2

Teach the converter the Qwen3.5 MoE layout: `mlp.gate` (router), `mlp.switch_mlp.{gate,up,down}_proj`
stacked on a leading expert axis (256 experts, intermediate 512, group 64 4-bit affine),
`mlp.shared_expert.*` and `mlp.shared_expert_gate`. Split the stacked tensors into per-expert
slabs, concatenate gate and up per expert like the dense MLP, keep the GDN/attention handling that
exists. Source: `mlx-community/Qwen3.5-35B-A3B-4bit` @ `1e20fd8d42056f870933bf98ca6211024744f7ec`,
read only.

Decision (creator, 24 Nov 2026): Qwen3.5-35B-A3B is the first MoE target, checked against MLX-LM; the MoE machinery is built and gated on it before the Inkling branch starts.

On: Zig 0.17 and Metal (Apple Silicon, measured on the M3 Max 64 GB), no new runtime dependencies.

Components: `tools/convert.py` (Python, numpy). No Zig.

Done means: a `models/qwen35-35b-a3b.tern` file whose description says 40 layers, 256 experts, top 8; a converter self-check compares every slab back to the source bytes.

Check: `python tools/convert.py --src <mlx dir> --out models/qwen35-35b-a3b.tern --verify` passes.

Depends on: [T11](../T11/task.md).

Design: [plan/moe-dspark.md](../moe-dspark.md). Token gate: see "Correctness gates" in the design; every speed claim has an interleaved before/after measurement.
