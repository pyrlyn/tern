"""tern benchmark: same rendered prompts and the same metrics as mlx_bench.py.

Compares the generated ids with the MLX-LM reference (bench/results/mlx.json) token by token.

Usage: python bench/tern_bench.py [--runs 5] [--max-tokens 200] [--out bench/results/tern.json] [-- extra tern args]
"""

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EOS = "248046,248044"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=str(ROOT.parent / "zig-out/bin/tern"))
    ap.add_argument("--model", default=str(ROOT.parent / "models/bonsai27b.tern"))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--prompts", default="free_text,code_edit,doc_qa")
    ap.add_argument("--out", default=str(ROOT / "results" / "tern.json"))
    ap.add_argument("--label", default="tern")
    ap.add_argument("extra", nargs="*")
    args = ap.parse_args()

    ref = json.loads((ROOT / "results" / "mlx.json").read_text())["prompts"]
    report = {"engine": args.label, "prompts": {}}
    for name in args.prompts.split(","):
        runs = []
        for _ in range(args.runs + 1):  # the first run is a warm-up (page-in, shader cache)
            load1 = os.getloadavg()[0]
            cmd = [args.bin, "gen", args.model, str(ROOT / "rendered" / f"{name}.ids"),
                   str(args.max_tokens), "--eos", EOS, *args.extra]
            r = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
            r["load1"] = load1
            runs.append(r)
        runs = runs[1:]
        want = ref[name]["tokens"]
        got = runs[0]["tokens"]
        n_same = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), min(len(want), len(got)))
        med = lambda k: statistics.median(r[k] for r in runs)  # noqa: E731
        p = {
            "n_prompt": runs[0]["n_prompt"],
            "n_gen": runs[0]["n_gen"],
            "decode_tps": med("decode_tps"),
            "decode_tps_all": [r["decode_tps"] for r in runs],
            "prefill_tps": med("prefill_tps"),
            "ttft_s": med("ttft_s"),
            "load_s": med("load_s"),
            "load1": [round(r["load1"], 1) for r in runs],
            "deterministic": all(r["tokens"] == got for r in runs),
            "matches_mlx": got == want,
            "first_mismatch": None if got == want else n_same,
            "tokens": got,
        }
        report["prompts"][name] = p
        print(
            f"{name}: prompt {p['n_prompt']} tok, gen {p['n_gen']} tok, decode {p['decode_tps']:.1f} tok/s "
            f"{p['decode_tps_all']}, prefill {p['prefill_tps']:.1f} tok/s, ttft {p['ttft_s']:.2f} s, "
            f"load {p['load1']}, same as MLX: {p['matches_mlx']}"
            + ("" if p["matches_mlx"] else f" (first mismatch at {n_same})"),
            flush=True,
        )
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
