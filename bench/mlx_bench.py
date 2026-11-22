"""MLX-LM baseline: greedy generation, time to first token, prefill and decode speed.

Renders each prompt once through the model's chat template (thinking off) and stores the
rendered text and token ids in bench/rendered/, so every other engine is fed the same tokens.

Usage: python bench/mlx_bench.py [--runs 5] [--max-tokens 200] [--prompts free_text,code_edit,doc_qa]
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = Path.home() / ".lmstudio/models/prism-ml/Ternary-Bonsai-27B-mlx-2bit"


def render(tokenizer, name: str) -> list[int]:
    text = (ROOT / "prompts" / f"{name}.txt").read_text().strip()
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    out = ROOT / "rendered"
    out.mkdir(exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps({"text": rendered, "ids": ids}))
    return ids


def run_once(model, ids: list[int], max_tokens: int, eos: set[int]) -> dict:
    prompt = mx.array(ids)
    cache = make_prompt_cache(model)
    tokens: list[int] = []
    load_before = os.getloadavg()[0]
    t0 = time.perf_counter()
    t_first = None
    # Greedy: generate_step's default sampler is argmax.
    for token, _ in generate_step(prompt, model, max_tokens=max_tokens, prompt_cache=cache):
        if t_first is None:
            t_first = time.perf_counter()
        token = int(token)
        if token in eos:
            break
        tokens.append(token)
    t1 = time.perf_counter()
    ttft = t_first - t0
    n = len(tokens)
    return {
        "ttft_s": ttft,
        "prefill_tps": len(ids) / ttft,
        "decode_tps": (n - 1) / (t1 - t_first) if n > 1 else 0.0,
        "n_prompt": len(ids),
        "n_gen": n,
        "tokens": tokens,
        "load1": load_before,
        "peak_gb": mx.get_peak_memory() / 1e9,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--prompts", default="free_text,code_edit,doc_qa")
    ap.add_argument("--out", default=str(ROOT / "results" / "mlx.json"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())
    load_s = time.perf_counter() - t0
    eos = set(tokenizer.eos_token_ids)

    report = {"engine": f"mlx-lm (mlx {mx.__version__})", "load_s": load_s, "prompts": {}}
    for name in args.prompts.split(","):
        ids = render(tokenizer, name)
        run_once(model, ids, 8, eos)  # warm-up: shader compilation, page-in
        runs = [run_once(model, ids, args.max_tokens, eos) for _ in range(args.runs)]
        same = all(r["tokens"] == runs[0]["tokens"] for r in runs)
        med = lambda k: statistics.median(r[k] for r in runs)  # noqa: E731
        report["prompts"][name] = {
            "n_prompt": len(ids),
            "n_gen": runs[0]["n_gen"],
            "decode_tps": med("decode_tps"),
            "decode_tps_all": [round(r["decode_tps"], 2) for r in runs],
            "prefill_tps": med("prefill_tps"),
            "ttft_s": med("ttft_s"),
            "load1": [round(r["load1"], 1) for r in runs],
            "deterministic": same,
            "tokens": runs[0]["tokens"],
            "text": tokenizer.decode(runs[0]["tokens"]),
        }
        p = report["prompts"][name]
        print(
            f"{name}: prompt {p['n_prompt']} tok, gen {p['n_gen']} tok, "
            f"decode {p['decode_tps']:.1f} tok/s {p['decode_tps_all']}, "
            f"prefill {p['prefill_tps']:.1f} tok/s, ttft {p['ttft_s']:.2f} s, "
            f"load {p['load1']}, deterministic {same}",
            flush=True,
        )
    report["peak_gb"] = mx.get_peak_memory() / 1e9
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"load {load_s:.1f} s, peak {report['peak_gb']:.1f} GB -> {args.out}")


if __name__ == "__main__":
    main()
