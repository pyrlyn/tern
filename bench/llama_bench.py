"""llama.cpp (Metal) baseline on the same rendered prompts as mlx_bench.py.

Runs llama-completion from the PrismML fork (the only llama.cpp that reads the PQ2_0 ternary
GGUF), greedy, and compares the generated text with the MLX-LM reference.

Usage: python bench/llama_bench.py [--runs 5] [--max-tokens 200] [--out bench/results/llama.json]
"""

import argparse
import json
import os
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROMPT_EVAL = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
EVAL = re.compile(r"\beval time =\s*([\d.]+) ms /\s*(\d+) runs")
LOAD = re.compile(r"load time =\s*([\d.]+) ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=str(ROOT.parent / "third_party/llama.cpp-prism/build/bin/llama-completion"))
    ap.add_argument("--model", default=str(ROOT.parent / "models/Ternary-Bonsai-27B-PQ2_0.gguf"))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--prompts", default="free_text,code_edit,doc_qa")
    ap.add_argument("--out", default=str(ROOT / "results" / "llama.json"))
    args = ap.parse_args()

    ref = json.loads((ROOT / "results" / "mlx.json").read_text())["prompts"]
    raw = ROOT / "results" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    report = {"engine": "llama.cpp (PrismML fork, Metal)", "prompts": {}}
    for name in args.prompts.split(","):
        rendered = json.loads((ROOT / "rendered" / f"{name}.json").read_text())
        prompt_file = raw / f"{name}.prompt.txt"
        # llama.cpp drops one trailing newline of a prompt file; add one so the tokens stay the same.
        prompt_file.write_text(rendered["text"] + "\n")
        runs = []
        for _ in range(args.runs + 1):  # the first run is a warm-up (page-in, shader cache)
            load1 = os.getloadavg()[0]
            cmd = [args.bin, "-m", args.model, "-f", str(prompt_file), "-n", str(args.max_tokens),
                   "--temp", "0", "-ngl", "99", "-c", "4096", "-no-cnv", "--no-display-prompt"]
            r = subprocess.run(cmd, check=True, capture_output=True, text=True)
            pe, ev = PROMPT_EVAL.search(r.stderr), EVAL.search(r.stderr)
            if not pe or not ev:
                raise SystemExit(f"cannot parse llama.cpp timings:\n{r.stderr[-2000:]}")
            runs.append({
                "n_prompt": int(pe.group(2)),
                "ttft_s": float(pe.group(1)) / 1e3,
                "prefill_tps": int(pe.group(2)) / (float(pe.group(1)) / 1e3),
                "decode_tps": int(ev.group(2)) / (float(ev.group(1)) / 1e3),
                "load_s": float(LOAD.search(r.stderr).group(1)) / 1e3,
                "load1": load1,
                "text": r.stdout,
            })
        runs = runs[1:]
        want, got = ref[name]["text"], runs[0]["text"]
        n_same = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), min(len(want), len(got)))
        med = lambda k: statistics.median(r[k] for r in runs)  # noqa: E731
        p = {
            "n_prompt": runs[0]["n_prompt"],
            "n_prompt_matches": runs[0]["n_prompt"] == len(rendered["ids"]),
            "decode_tps": med("decode_tps"),
            "decode_tps_all": [round(r["decode_tps"], 2) for r in runs],
            "prefill_tps": med("prefill_tps"),
            "ttft_s": med("ttft_s"),
            "load_s": med("load_s"),
            "load1": [round(r["load1"], 1) for r in runs],
            "deterministic": all(r["text"] == got for r in runs),
            "same_chars_as_mlx": n_same,
            "mlx_chars": len(want),
            "text": got,
        }
        report["prompts"][name] = p
        print(
            f"{name}: prompt {p['n_prompt']} tok (same as rendered: {p['n_prompt_matches']}), "
            f"decode {p['decode_tps']:.1f} tok/s {p['decode_tps_all']}, prefill {p['prefill_tps']:.1f} tok/s, "
            f"ttft {p['ttft_s']:.2f} s, load {p['load1']}, text equal to MLX for {n_same}/{len(want)} chars",
            flush=True,
        )
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
