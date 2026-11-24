---
status: todo
priority: P0
complexity: 3
readiness: 60%
agent:
---

# T11. .tern format version 2: per-expert slabs and a model description

Extend the `.tern` file so MoE experts are addressable one by one and stream with one read each,
and so new architectures are described in the file instead of a comptime `cfg`. Magic `TERN0002`,
a JSON model description after the entry table, entries with `experts` and `stride`, expert slabs
(gate_up weights + scales, down weights + scales of one expert contiguous, padded to 16 KiB),
fixed weights first and slabs after, and the new dtypes listed in the design. `TERN0001` files
keep loading unchanged.

On: Zig 0.17 and Metal (Apple Silicon, measured on the M3 Max 64 GB), no new runtime dependencies.

Components: `src/model.zig` (reader, `Tensor` gains expert addressing, run-time shape struct), `tools/convert.py` (writer), a format section in the design doc or `docs/tern-format.md`. Zig 0.17; no kernels.

Done means: the reader opens both versions; a v2 file written from the Bonsai MLX source gives the same tokens as the v1 file; unit tests in `zig build test` cover entry parsing, expert offset arithmetic and slab alignment.

Check: `zig build test` passes; `tern gen` on Bonsai v1 and v2 files gives identical token ids on the three prompts (`tern_bench.py` `same as MLX: True`).

Depends on: none.

Design: [plan/moe-dspark.md](../moe-dspark.md). Token gate: see "Correctness gates" in the design; every speed claim has an interleaved before/after measurement.
