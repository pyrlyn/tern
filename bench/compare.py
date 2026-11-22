"""Interleaved comparison of llama.cpp, MLX-LM and tern on the same prompts.

The machine's state (other jobs, GPU throttling) drifts over minutes, so the engines are not
measured in three long blocks. Every round runs each engine once per prompt, back to back, in a
rotating order; the result is the median over the rounds. Each single measurement is one run of
the engine's own bench script (which does its own warm-up first).

Usage: python bench/compare.py [--rounds 3] [--max-tokens 200] [--out bench/results/compare.json]
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINES = {"llama.cpp": "llama_bench.py", "mlx-lm": "mlx_bench.py", "tern": "tern_bench.py"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--prompts", default="free_text,code_edit,doc_qa")
    ap.add_argument("--out", default=str(ROOT / "results" / "compare.json"))
    args = ap.parse_args()

    raw = ROOT / "results" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    prompts = args.prompts.split(",")
    names = list(ENGINES)
    runs: dict[str, dict[str, list[dict]]] = {p: {e: [] for e in names} for p in prompts}
    for rnd in range(args.rounds):
        for pi, prompt in enumerate(prompts):
            shift = (rnd + pi) % len(names)
            for engine in names[shift:] + names[:shift]:
                out = raw / f"cmp_{engine}_{prompt}_{rnd}.json"
                load1 = os.getloadavg()[0]
                subprocess.run(
                    [sys.executable, str(ROOT / ENGINES[engine]), "--runs", "1", "--max-tokens",
                     str(args.max_tokens), "--prompts", prompt, "--out", str(out)],
                    check=True, capture_output=True, text=True,
                )
                r = json.loads(out.read_text())["prompts"][prompt]
                r["load1_start"] = load1
                runs[prompt][engine].append(r)
                print(f"round {rnd} {prompt} {engine}: decode {r['decode_tps']:.1f} tok/s, "
                      f"prefill {r['prefill_tps']:.1f} tok/s, ttft {r['ttft_s']:.2f} s, load {load1:.1f}",
                      flush=True)

    report = {"rounds": args.rounds, "max_tokens": args.max_tokens, "prompts": {}}
    for prompt in prompts:
        row = {}
        for engine in names:
            rs = runs[prompt][engine]
            row[engine] = {
                "decode_tps": statistics.median(r["decode_tps"] for r in rs),
                "decode_tps_all": [round(r["decode_tps"], 2) for r in rs],
                "prefill_tps": statistics.median(r["prefill_tps"] for r in rs),
                "prefill_tps_all": [round(r["prefill_tps"], 1) for r in rs],
                "ttft_s": statistics.median(r["ttft_s"] for r in rs),
                "load1": [round(r["load1_start"], 1) for r in rs],
            }
        row["tern"]["matches_mlx"] = all(r["matches_mlx"] for r in runs[prompt]["tern"])
        row["llama.cpp"]["text_equals_mlx"] = all(
            r["same_chars_as_mlx"] == r["mlx_chars"] for r in runs[prompt]["llama.cpp"])
        # The reference itself: every MLX-LM run must reproduce the stored reference tokens.
        ref = json.loads((ROOT / "results" / "mlx.json").read_text())["prompts"][prompt]["tokens"]
        row["mlx-lm"]["matches_reference"] = all(r["tokens"] == ref for r in runs[prompt]["mlx-lm"])
        report["prompts"][prompt] = row
    Path(args.out).write_text(json.dumps(report, indent=1))

    print("\n| prompt | metric | llama.cpp | MLX-LM | tern | tern vs llama.cpp | tern vs MLX-LM |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for prompt, row in report["prompts"].items():
        for key, unit in (("decode_tps", "tok/s"), ("prefill_tps", "tok/s"), ("ttft_s", "s")):
            a, b, c = (row[e][key] for e in names)
            print(f"| {prompt} | {key} ({unit}) | {a:.2f} | {b:.2f} | {c:.2f} | "
                  f"{(c / a - 1) * 100:+.0f}% | {(c / b - 1) * 100:+.0f}% |")


if __name__ == "__main__":
    main()
