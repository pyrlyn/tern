# Ideas

Not approved. Nothing here is a task until the creator says so.

Approved speed goals live in [README.md](README.md) and [plan/README.md](plan/README.md).
One of them is to be twice as fast as
[Atomic Engine](https://github.com/AtomicBot-ai/atomic-engine) (a llama.cpp fork
with SSD expert streaming). Their published figure is 105.71 tokens/s on a
48 GB M4 Max (Inkling 2 Mini IQ2_M with a DSpark draft model, 14 GiB budget).

- Dense trit packing: 5 trits per byte (1.6 bits per weight) lowers the memory floor of decode
  by 20%. The decode cost on the GPU has to stay below that; the 256-entry table tried for the
  2-bit format was 4 to 5 times slower than arithmetic, so the decode has to be arithmetic too.
- A 64 rows x 16 tokens tile for speculation: a pass of 16 tokens would cost about 4 sequential
  tokens instead of 8.
- GQA-aware attention value sum: one thread per (dimension, KV head) with 6 accumulators reads
  the value cache once for the 6 query heads; same order of summation, so bit-identical.
- More bench prompts (20 or more, different lengths) so that the token check is not decided by
  one near-tie.
- C ABI (`tern.h`) over the engine for runa's Kotlin/Native cinterop, next to llama.cpp.
- Prefix cache: snapshot of the KV cache and the delta-rule states after the system prompt.
- Port the cold-start path of the earlier prototype (4 threads, 4 MiB blocks in layer order,
  `--nvme`).
- CPU bandwidth measurement for the CPU backend (not measured yet).
- Long-context bench (2k to 4k tokens): the attention value sum grows linearly with context.
