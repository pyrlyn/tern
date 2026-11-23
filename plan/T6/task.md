---
status: todo
priority: P3
complexity: 4
readiness: 0%
agent:
---

# T6. Backend interface and non-Metal GPU backends

The runtime is meant to have interchangeable CPU and GPU backends behind one interface resolved
at compile time (no indirect calls on the hot path). Today `engine.zig` talks to Metal directly.
CUDA and ROCm backends, and the comparison with vLLM and SGLang, need NVIDIA or AMD hardware that
is not available; the comparison with MLC LLM and ExecuTorch needs a phone.

Done means: the engine is written against a backend interface with Metal as the first
implementation, and a second backend exists where hardware is available.

Check: `zig build -Dbackend=metal` produces the same tokens and speed as before the split; a second backend passes the same token check on its hardware.

## Design

The compile-time interface, memory model, backend set, Zig signature sketch, and
reuse / port / rewrite analysis are in [docs/backend-abstraction.md](../../docs/backend-abstraction.md).
That document is the specification for this task. Do not re-introduce a vtable on
the encode path.

