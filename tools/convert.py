"""Repack an MLX Qwen3.5 model into tern's flat weight file: the 2-bit Ternary Bonsai model, or
a 4-bit affine model used as a draft (Qwen3.5-0.8B in MLX 4-bit).

The source is opened read-only and never modified. The output keeps every tensor page-aligned
(16 KiB) and in execution order, so the runtime can map the file once, hand it to the GPU without
a copy and read it front to back.

What changes relative to the MLX layout:
  * projections that read the same input vector are concatenated row-wise into one matrix
    (linear attention: qkv, z, b, a; full attention: q, k, v; MLP: gate, up), so each becomes a
    single kernel launch;
  * 2-bit: the per-group biases are dropped for matrices: the weights are ternary and the file
    stores bias == -scale exactly (checked here), so w = scale * (q - 1);
  * 4-bit: matrices keep their biases (`.b`), w = scale * q + bias; bfloat16 scales, biases and
    norm weights are stored as float16 (checked to stay finite);
  * tied embeddings: the head entries point at the embedding table, nothing is copied;
  * exp(A_log) is precomputed in float32.

Usage: python tools/convert.py [--src <mlx model dir>] [--out models/bonsai27b.tern]
       python tools/convert.py --src models/qwen3.5-0.8b-4bit --out models/qwen35-08b.tern
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
        raw = self.map[self.base + a : self.base + b]
        if v["dtype"] == "BF16":
            bits = np.frombuffer(raw, np.uint16).astype(np.uint32) << 16
            return bits.view(np.float32).reshape(v["shape"])
        dt = {"F16": np.float16, "F32": np.float32, "U32": np.uint32}[v["dtype"]]
        return np.frombuffer(raw, dt).reshape(v["shape"])


def half(a: np.ndarray, what: str) -> np.ndarray:
    out = a.astype(np.float16)
    assert np.isfinite(out).all(), f"{what}: does not fit in float16"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default="models/bonsai27b.tern")
    args = ap.parse_args()

    src_dir = Path(args.src)
    top = json.loads((src_dir / "config.json").read_text())
    cfg = top["text_config"]
    quant = cfg.get("quantization") or top["quantization"]
    bits = quant["bits"]
    assert (bits, quant["group_size"]) in ((2, 128), (4, 64)), f"unsupported quantization {quant}"
    assert quant.get("mode", "affine") == "affine"
    src = Source(src_dir / "model.safetensors")
    lm = "language_model."
    tensors: list[tuple[str, int, np.ndarray]] = []
    aliases: list[tuple[str, str]] = []

    def add(name: str, dtype: int, arr: np.ndarray) -> None:
        tensors.append((name, dtype, np.ascontiguousarray(arr)))

    def matrix(prefix: str, parts: list[str]) -> None:
        ws, ss, bs = [], [], []
        for p in parts:
            w = src.get(f"{lm}{p}.weight")
            s = src.get(f"{lm}{p}.scales")
            b = src.get(f"{lm}{p}.biases")
            if bits == 2:
                assert np.array_equal(b, -s), f"{p}: bias != -scale, weights are not plain ternary"
            ws.append(w)
            ss.append(half(s, p))
            bs.append(half(b, p))
        add(f"{prefix}.w", DT_U32, np.concatenate(ws))
        add(f"{prefix}.s", DT_F16, np.concatenate(ss))
        if bits == 4:
            add(f"{prefix}.b", DT_F16, np.concatenate(bs))

    def f16(name: str, key: str) -> None:
        add(name, DT_F16, half(src.get(lm + key), key))

    e = "model.embed_tokens"
    add("embed.w", DT_U32, src.get(f"{lm}{e}.weight"))
    add("embed.s", DT_F16, half(src.get(f"{lm}{e}.scales"), e))
    add("embed.b", DT_F16, half(src.get(f"{lm}{e}.biases"), e))

    for i, kind in enumerate(cfg["layer_types"]):
        L = f"model.layers.{i}"
        f16(f"L{i}.n1", f"{L}.input_layernorm.weight")
        if kind == "linear_attention":
            a = f"{L}.linear_attn"
            matrix(f"L{i}.in", [f"{a}.in_proj_qkv", f"{a}.in_proj_z", f"{a}.in_proj_b", f"{a}.in_proj_a"])
            conv = src.get(f"{lm}{a}.conv1d.weight")
            add(f"L{i}.conv", DT_F16, half(conv.reshape(conv.shape[0], conv.shape[1]), "conv"))
            A = np.exp(src.get(f"{lm}{a}.A_log").astype(np.float32))
            dt = src.get(f"{lm}{a}.dt_bias").astype(np.float32)
            add(f"L{i}.adt", DT_F32, np.stack([A, dt]))
            f16(f"L{i}.gnorm", f"{a}.norm.weight")
            matrix(f"L{i}.out", [f"{a}.out_proj"])
        else:
            a = f"{L}.self_attn"
            matrix(f"L{i}.in", [f"{a}.q_proj", f"{a}.k_proj", f"{a}.v_proj"])
            f16(f"L{i}.qn", f"{a}.q_norm.weight")
            f16(f"L{i}.kn", f"{a}.k_norm.weight")
            matrix(f"L{i}.out", [f"{a}.o_proj"])
        f16(f"L{i}.n2", f"{L}.post_attention_layernorm.weight")
        matrix(f"L{i}.mlp", [f"{L}.mlp.gate_proj", f"{L}.mlp.up_proj"])
        matrix(f"L{i}.down", [f"{L}.mlp.down_proj"])

    f16("norm", "model.norm.weight")
    if f"{lm}lm_head.weight" in src.header:
        matrix("head", ["lm_head"])
    else:
        assert cfg.get("tie_word_embeddings", top.get("tie_word_embeddings")), "no lm_head"
        aliases += [("head.w", "embed.w"), ("head.s", "embed.s"), ("head.b", "embed.b")]

    header_len = len(MAGIC) + 8 + ENTRY.size * (len(tensors) + len(aliases))
    offset = -(-header_len // PAGE) * PAGE
    entries = []
    for name, dtype, arr in tensors:
        rows = arr.shape[0]
        cols = arr.size // rows
        entries.append(ENTRY.pack(name.encode(), offset, arr.nbytes, rows, cols, dtype))
        offset += -(-arr.nbytes // PAGE) * PAGE
    names = [t[0] for t in tensors]
    for name, target in aliases:
        target_entry = ENTRY.unpack(entries[names.index(target)])
        entries.append(ENTRY.pack(name.encode(), *target_entry[1:]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(MAGIC + struct.pack("<II", len(entries), PAGE) + b"".join(entries))
        for (name, dtype, arr), entry in zip(tensors, entries):
            f.seek(ENTRY.unpack(entry)[1])
            f.write(arr.tobytes())
        f.truncate(offset)
    print(f"{out}: {len(entries)} tensors, {offset / 1e9:.2f} GB, {bits}-bit")


if __name__ == "__main__":
    main()
