# tern

Inference runtime for ternary LLMs on Apple Silicon, written in Zig, with Metal compute kernels.
It runs Ternary Bonsai 27B (Qwen3.5 27B hybrid: 48 gated-delta-net layers, 16 full-attention
layers, 2-bit ternary weights) and produces the same greedy tokens as MLX-LM.

The goal is to be 40% faster than llama.cpp (Metal) and MLX-LM on an M3 Max. Where it stands is
in [REPORT.md](REPORT.md); the survey of techniques is in [RESEARCH.md](RESEARCH.md); open work
is in [plan/](plan/README.md).

## Build

Needs Zig 0.17 and macOS with Metal. No other dependencies: Metal is called through the
Objective-C runtime from Zig.

```bash
zig build --release=fast
```

```bash
zig build test
```

## Model

The engine reads its own `.tern` file: every tensor page-aligned, mapped once and handed to Metal
without a copy. Convert the MLX 2-bit model (the source is only read):

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python mlx-lm
```

```bash
.venv/bin/python tools/convert.py --src ~/.lmstudio/models/prism-ml/Ternary-Bonsai-27B-mlx-2bit --out models/bonsai27b.tern
```

## Run

```bash
./zig-out/bin/tern gen models/bonsai27b.tern bench/rendered/code_edit.ids 200 --eos 248046,248044
```

`gen` takes a file of token ids (tokenization is not part of the engine yet) and prints one JSON
line: tokens, load time, time to first token, prefill and decode speed, GPU time, and the
verification passes of speculative decoding per pass size.

Speculative decoding is on by default: when the end of the context repeats something seen
earlier, the tokens that followed there are verified in one batched pass. The output is the same
as without it; `--spec 0` turns it off, `--spec n` caps a pass at n tokens.
A pass starts at 2 tokens, which costs little more than a plain token, and grows to 4, 8 and n
while its drafts are accepted.

`--draft <draft.tern>` adds a draft model, Qwen3.5-0.8B in MLX 4-bit, for the places where the
context repeats nothing: it proposes one token and the main model checks it in a pass of 2. The
GPU accepts each pass and sets up the next, so two passes are in flight like plain tokens. Off
by default; at the load measured so far it gains 4 to 8% on document QA and nothing on free text
(REPORT.md). A gate times the passes against plain tokens and runs the faster way, so the draft
model does not slow free text down; `--gate 0` turns the gate off. The draft file comes from the
same converter:

```sh
.venv/bin/python tools/convert.py --src models/qwen3.5-0.8b-4bit --out models/qwen35-08b.tern
```

| Command | Purpose |
| --- | --- |
| `tern gen <model> <ids> <n>` | Greedy generation with timings |
| `tern prof <model> <position> <batch>` | GPU time per token for every kernel class, best of 5 (main or draft model) |
| `tern prof <model> <ids> 2 --draft <draft>` | GPU time of the pieces of a chained pass of two after a real prompt, in plain tokens |
| `tern bench <model> <matrix> <iters> <batch>` | One matrix product: ms per token, weights per second, checksum |

Options are listed by `tern` without arguments.

## Benchmarks

```bash
.venv/bin/python bench/compare.py --rounds 3
```

`compare.py` runs llama.cpp, MLX-LM and tern in turn for every prompt and reports medians; the
single-engine scripts are `llama_bench.py`, `mlx_bench.py` and `tern_bench.py`. All of them use
the prompts rendered once by `mlx_bench.py` (`bench/rendered/`), so every engine gets the same
tokens. `mlx_bench.py` without `--out` rewrites the reference `bench/results/mlx.json`.

llama.cpp has to be the [PrismML fork](https://github.com/PrismML-Eng/llama.cpp) (the only one
that reads the PQ2_0 GGUF), built in `third_party/llama.cpp-prism`, with the GGUF from
[prism-ml/Ternary-Bonsai-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf) in
`models/`.

## Layout

| Path | Holds |
| --- | --- |
| `src/kernels.metal` | Every compute kernel, compiled at start |
| `src/engine.zig` | Buffers, pipelines and the encoding of a token or a batch of tokens, generic over the model shape |
| `src/metal.zig` | Metal bindings through `objc_msgSend` |
| `src/model.zig` | Shapes of the main and the draft model, the `.tern` reader |
| `src/draft.zig` | Prompt lookup and the depth policy of speculative decoding |
| `src/chain.zig` | Draft-model passes chained on the GPU: acceptance and the next pass's position |
| `src/main.zig` | The command line |
| `tools/convert.py` | MLX safetensors (2-bit main model, 4-bit draft model) to `.tern` |
| `bench/` | Prompts, rendered token ids, bench scripts, results |

## Design

- **One pass, any batch.** Every kernel takes a batch of tokens. A batch of 1 is decode, a
  batch of 32 is prefill, and a batch of draft tokens is the verification pass of speculative
  decoding. The matrix product picks its kernel by batch size: lane-split GEMV for one token,
  lane GEMM for a few, an 8-token tile from 8 tokens and a 32-token tile from 16 tokens up.
- **Speculation without copies.** A verification pass reads the recurrent states of the linear
  layers and does not write them. The accepted tokens are replayed through the state kernels
  alone, from inputs the pass left behind; the KV cache needs no rollback because rejected
  positions are overwritten.
- **The token stays on the GPU.** Argmax writes the next token id into a GPU buffer, the
  embedding kernel of the next step reads it, and the CPU encodes steps ahead of the GPU.
- **Hot path.** The engine allocates nothing per token and calls its own code directly
  (`inline`, no function pointers). What remains is what Metal requires: one command buffer and
  one encoder per token, and the `objc_msgSend` calls that encode it (0.13 to 0.19 ms per token).
- **Exact arithmetic where it is free.** Ternary weights are unpacked with integer operations
  and multiplied as exact -1, 0, +1; sums are float, activations are stored as half. The order
  of every sum is part of the contract with the reference tokens.
- **Backends.** Today the engine talks to Metal directly. The intended shape is a backend
  interface resolved at compile time, with CPU and other GPU backends behind it ([T6](plan/T6/task.md)).
