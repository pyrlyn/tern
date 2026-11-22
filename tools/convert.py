"""Repack an MLX 2-bit Qwen3.5 model (Ternary Bonsai) into tern's flat weight file.

The source is opened read-only and never modified. The output keeps every tensor page-aligned
(16 KiB) and in execution order, so the runtime can map the file once, hand it to the GPU without
a copy and read it front to back.

What changes relative to the MLX layout:
  * projections that read the same input vector are concatenated row-wise into one matrix
    (linear attention: qkv, z, b, a; full attention: q, k, v; MLP: gate, up), so each becomes a
    single kernel launch;
  * the per-group biases are dropped for matrices: the weights are ternary and the file stores
    bias == -scale exactly (checked here), so w = scale * (q - 1);
  * exp(A_log) is precomputed in float32.

Usage: python tools/convert.py [--src <mlx model dir>] [--out models/bonsai27b.tern]
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np

PAGE = 16384
MAGIC = b"TERN0001"
ENTRY = struct.Struct("<48sQQIII4x")  # name, offset, nbytes, rows, cols, dtype
DT_U32, DT_F16, DT_F32 = 0, 1, 2
DEFAULT_SRC = Path.home() / ".lmstudio/models/prism-ml/Ternary-Bonsai-27B-mlx-2bit"


class Source:
    def __init__(self, path: Path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(n))
        self.base = 8 + n
        self.map = np.memmap(path, dtype=np.uint8, mode="r")

    def get(self, name: str) -> np.ndarray:
        v = self.header[name]
        a, b = v["data_offsets"]
        dt = {"F16": np.float16, "U32": np.uint32}[v["dtype"]]
        return np.frombuffer(self.map[self.base + a : self.base + b], dt).reshape(v["shape"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default="models/bonsai27b.tern")
    args = ap.parse_args()

    src_dir = Path(args.src)
    cfg = json.loads((src_dir / "config.json").read_text())["text_config"]
    assert cfg["quantization"] == {"group_size": 128, "bits": 2}
    src = Source(src_dir / "model.safetensors")
    lm = "language_model."
    tensors: list[tuple[str, int, np.ndarray]] = []

    def add(name: str, dtype: int, arr: np.ndarray) -> None:
        tensors.append((name, dtype, np.ascontiguousarray(arr)))

    def ternary(prefix: str, parts: list[str]) -> None:
        ws, ss = [], []
        for p in parts:
            w = src.get(f"{lm}{p}.weight")
            s = src.get(f"{lm}{p}.scales")
            b = src.get(f"{lm}{p}.biases")
            assert np.array_equal(b, -s), f"{p}: bias != -scale, weights are not plain ternary"
            ws.append(w)
            ss.append(s)
        add(f"{prefix}.w", DT_U32, np.concatenate(ws) if len(ws) > 1 else ws[0])
        add(f"{prefix}.s", DT_F16, np.concatenate(ss) if len(ss) > 1 else ss[0])

    def f16(name: str, key: str) -> None:
        add(name, DT_F16, src.get(lm + key))

    e = "model.embed_tokens"
    add("embed.w", DT_U32, src.get(f"{lm}{e}.weight"))
    add("embed.s", DT_F16, src.get(f"{lm}{e}.scales"))
    add("embed.b", DT_F16, src.get(f"{lm}{e}.biases"))

    for i, kind in enumerate(cfg["layer_types"]):
        L = f"model.layers.{i}"
        f16(f"L{i}.n1", f"{L}.input_layernorm.weight")
        if kind == "linear_attention":
            a = f"{L}.linear_attn"
            ternary(f"L{i}.in", [f"{a}.in_proj_qkv", f"{a}.in_proj_z", f"{a}.in_proj_b", f"{a}.in_proj_a"])
            conv = src.get(f"{lm}{a}.conv1d.weight")
            add(f"L{i}.conv", DT_F16, conv.reshape(conv.shape[0], conv.shape[1]))
            A = np.exp(src.get(f"{lm}{a}.A_log").astype(np.float32))
            dt = src.get(f"{lm}{a}.dt_bias").astype(np.float32)
            add(f"L{i}.adt", DT_F32, np.stack([A, dt]))
            f16(f"L{i}.gnorm", f"{a}.norm.weight")
            ternary(f"L{i}.out", [f"{a}.out_proj"])
        else:
            a = f"{L}.self_attn"
            ternary(f"L{i}.in", [f"{a}.q_proj", f"{a}.k_proj", f"{a}.v_proj"])
            f16(f"L{i}.qn", f"{a}.q_norm.weight")
            f16(f"L{i}.kn", f"{a}.k_norm.weight")
            ternary(f"L{i}.out", [f"{a}.o_proj"])
        f16(f"L{i}.n2", f"{L}.post_attention_layernorm.weight")
        ternary(f"L{i}.mlp", [f"{L}.mlp.gate_proj", f"{L}.mlp.up_proj"])
        ternary(f"L{i}.down", [f"{L}.mlp.down_proj"])

    f16("norm", "model.norm.weight")
    ternary("head", ["lm_head"])

    header_len = len(MAGIC) + 8 + ENTRY.size * len(tensors)
    offset = -(-header_len // PAGE) * PAGE
    entries = []
    for name, dtype, arr in tensors:
        rows = arr.shape[0]
        cols = arr.size // rows
        entries.append(ENTRY.pack(name.encode(), offset, arr.nbytes, rows, cols, dtype))
        offset += -(-arr.nbytes // PAGE) * PAGE

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(MAGIC + struct.pack("<II", len(tensors), PAGE) + b"".join(entries))
        for (name, dtype, arr), entry in zip(tensors, entries):
            f.seek(ENTRY.unpack(entry)[1])
            f.write(arr.tobytes())
        f.truncate(offset)
    print(f"{out}: {len(tensors)} tensors, {offset / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
