"""Acceptance of a draft model against the reference tokens of the 27B model, offline.

For every generated position of the three bench prompts, the draft model sees the prompt and the
reference tokens before that position (teacher forcing, one forward pass per prompt) and its
greedy token and that token's probability are compared with the reference. For a greedy draft this is exact: a draft of k
tokens started at position i is accepted up to the first position from i on where the draft's
greedy token differs from the reference.

The accepted runs are then turned into a decode speed in plain-token units with the pass prices
of tern (REPORT.md) and the depth policy of `src/draft.zig`, for three draft sources: prompt
lookup alone (what tern does today), the draft model alone, and prompt lookup with the draft
model where lookup has no match. For the draft model two more gates are simulated: a pass of 2
tokens at every position, and a draft that goes on while the product of the draft model's
probabilities of its tokens stays above a threshold (the pass is then the largest size that the
draft fills). The probabilities past a wrong draft token are taken in the reference context;
that only changes how long a rejected draft runs, not what is accepted.

Usage: python bench/draft_accept.py --model models/qwen3.5-0.8b-4bit [--plain-ms 27]
           [--out bench/results/draft_accept.json]
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate

ROOT = Path(__file__).resolve().parent
PROMPTS = ("free_text", "code_edit", "doc_qa")

# Pass prices in plain tokens, measured on the machine in its faster state (REPORT.md).
PRICES = {2: 1.15, 4: 2.1, 8: 3.5, 32: 8.0}
SIZES = [2, 4, 8, 32]
MIN_MATCH = 4
PAUSE = 4
MAX_MATCH = 16


def prompt_ids(name: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", (ROOT / "rendered" / f"{name}.ids").read_text())]


def greedy_teacher_forced(model, full: list[int], n_prompt: int) -> tuple[list[int], list[float]]:
    """Greedy token of the draft model and its probability for every generated position."""
    logits = model(mx.array([full[:-1]]))[0, n_prompt - 1 :].astype(mx.float32)
    pred = mx.argmax(logits, axis=-1)
    prob = mx.max(mx.softmax(logits, axis=-1), axis=-1)
    mx.eval(pred, prob)
    return pred.tolist(), prob.tolist()


def lookup(ctx: list[int]) -> tuple[int, int]:
    """Prompt lookup as in `src/draft.zig`: (index after the match, match length)."""
    n = len(ctx)
    last = ctx[-1]
    best = (0, 0)
    for e in range(n - 2, -1, -1):
        if ctx[e] != last:
            continue
        length = 1
        while length < MAX_MATCH and length <= e and ctx[e - length] == ctx[n - 1 - length]:
            length += 1
        if length > best[1]:
            best = (e + 1, length)
            if length == MAX_MATCH:
                break
    return best


def runs(full: list[int], n_prompt: int, pred: list[int]) -> dict[int, tuple[int, int]]:
    """Per confirmed position `done` (index into `full`): accepted draft tokens of prompt
    lookup (0 when its match is shorter than MIN_MATCH, -1 when there is no match) and of the
    draft model, each capped at 31."""
    end = len(full) - 1
    right = [pred[i] == full[n_prompt + i] for i in range(len(pred))]
    out = {}
    for done in range(n_prompt - 1, end):
        at, m = lookup(full[: done + 1])
        look = -1
        if m >= MIN_MATCH:
            buf = full[: done + 1]
            for i in range(31):
                buf.append(buf[at + i])
            look = 0
            while look < 31 and done + 1 + look <= end and buf[done + 1 + look] == full[done + 1 + look]:
                look += 1
        model = 0
        i = done + 1 - n_prompt
        while model < 31 and i + model < len(right) and right[i + model]:
            model += 1
        out[done] = (look, model)
    return out


def simulate(r, start: int, end: int, source: str, d: float, pause_model: bool, model_pair: bool = False) -> dict:
    """Decode time of the generated tokens in plain-token units under the depth policy of
    `src/draft.zig`. `source` is "lookup", "model" or "both"; a pass drafted by the model costs
    (size - 1) * d on top of its price. With `model_pair` (source "both"), a position without a
    lookup match gets one draft-model token in a pass of 2, outside the ladder and the pause."""
    done, t, level, cool, losses = start, 0.0, 0, 0, 0
    passes = 0
    while done < end:
        look, model = r[done]
        if model_pair and look < 0:
            if end - done < 2:
                t += 1
                done += 1
                continue
            got = min(model, 1) + 1
            t += PRICES[2] + d
            done += got
            cool = max(0, cool - got)
            passes += 1
            continue
        if source == "lookup" or (source == "both" and look >= 0):
            acc, by_model = (look if look >= 0 else None), False
        else:
            acc, by_model = model, True
        room = end - done
        k = 0
        if acc is not None and cool == 0:
            lv = level
            while lv > 0 and SIZES[lv] > room:
                lv -= 1
            if SIZES[lv] <= room:
                k = SIZES[lv]
        if k == 0:
            t += 1
            done += 1
            cool = max(0, cool - 1)
            continue
        got = min(acc, k - 1) + 1
        t += PRICES[k] + (k - 1) * d * by_model
        done += got
        passes += 1
        if got == k:
            losses = 0
            if level + 1 < len(SIZES) and SIZES[level] == k:
                level += 1
            continue
        while level > 0 and SIZES[level] > got:
            level -= 1
        if got <= 2 and (pause_model or not by_model):
            losses += 1
            cool = PAUSE * losses
        else:
            losses = 0
    return {"speedup": (end - start) / t, "passes": passes}


def simulate_gate(r, prob, n_prompt: int, start: int, end: int, d: float, tau: float, lookup_first: bool) -> float:
    """Decode speedup with the draft model gated by its confidence: draft while the product of
    the probabilities stays at or above `tau` (tau = 0: one draft token and a pass of 2 at every
    position); the pass is the largest size whose draft tokens the draft fills, and the draft
    model pays for every token it generated, including the one that stopped it. With
    `lookup_first`, a position where prompt lookup has a match takes the lookup draft in a pass
    of 8 tokens (the ladder of `simulate` is not modelled here)."""
    done, t = start, 0.0
    while done < end:
        look, model = r[done]
        room = end - done
        if lookup_first and look >= 0:
            k = max([s for s in SIZES if s <= min(room, 8)] or [0])
            if k == 0:
                t += 1
                done += 1
                continue
            t += PRICES[k]
            done += min(look, k - 1) + 1
            continue
        i = done + 1 - n_prompt
        if tau == 0:
            n_draft, generated = 1, 1
        else:
            c, n_draft = 1.0, 0
            while n_draft < 31 and i + n_draft < len(prob):
                c *= prob[i + n_draft]
                if c < tau:
                    break
                n_draft += 1
            generated = min(n_draft + 1, 31)
        k = max([s for s in SIZES if s - 1 <= n_draft and s <= room] or [0])
        t += generated * d
        if k == 0:
            t += 1
            done += 1
            continue
        t += PRICES[k]
        done += min(model, k - 1) + 1
    return (end - start) / t


def oracle(r, start: int, end: int, source: str, d: float) -> float:
    """Best speedup any gate could reach with the same drafts and prices."""
    best = [0.0] * (end + 1)
    for done in range(end - 1, start - 1, -1):
        b = 1 + best[done + 1]
        look, model = r[done]
        if source == "lookup" or (source == "both" and look >= 0):
            acc, extra = (look if look >= 0 else None), 0.0
        else:
            acc, extra = model, d
        if acc is not None:
            for k, c in PRICES.items():
                got = min(acc, k - 1) + 1
                if done + got <= end:
                    b = min(b, c + (k - 1) * extra + best[done + got])
        best[done] = b
    return (end - start) / best[start]


def draft_ms(model, tokenizer) -> float:
    """Median time per generated token of the draft model in mlx-lm, 3 runs of 64 tokens."""
    ids = prompt_ids("free_text")
    tps = []
    for _ in range(4):
        last = None
        for last in stream_generate(model, tokenizer, ids, max_tokens=64):
            pass
        tps.append(last.generation_tps)
    tps = sorted(tps[1:])
    return 1000.0 / tps[len(tps) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--plain-ms", type=float, default=27.0, help="time of a plain tern token")
    ap.add_argument("--out", default=str(ROOT / "results" / "draft_accept.json"))
    args = ap.parse_args()

    ref = json.loads((ROOT / "results" / "mlx.json").read_text())["prompts"]
    model, tokenizer = load(args.model)

    ms = draft_ms(model, tokenizer)
    costs = {"measured": ms / args.plain_ms, "low": 0.07, "high": 0.15}
    result = {
        "model": os.path.relpath(args.model, ROOT.parent),
        "draft_ms": ms,
        "plain_ms": args.plain_ms,
        "d": costs,
        "prompts": {},
    }
    print(f"draft model: {ms:.2f} ms per token in mlx-lm, d = {costs['measured']:.3f} plain tokens")

    for name in PROMPTS:
        prompt = prompt_ids(name)
        text = json.loads((ROOT / "rendered" / f"{name}.json").read_text())["text"]
        same_tokens = tokenizer.encode(text, add_special_tokens=False) == prompt
        gen = ref[name]["tokens"]
        full = prompt + gen
        t0 = time.perf_counter()
        pred, prob = greedy_teacher_forced(model, full, len(prompt))
        fwd = time.perf_counter() - t0
        r = runs(full, len(prompt), pred)
        start, end = len(prompt) - 1, len(full) - 1
        right = [p == g for p, g in zip(pred, gen)]
        run_len = [r[done][1] for done in range(start, end)]
        lookup_pos = [done for done in range(start, end) if r[done][0] >= 0]
        no_lookup = [done for done in range(start, end) if r[done][0] < 0]
        entry = {
            "tokenizer_matches": same_tokens,
            "forward_s": fwd,
            "first_token_accepted": sum(right) / len(right),
            "first_token_accepted_without_lookup": (
                sum(r[d][1] >= 1 for d in no_lookup) / len(no_lookup) if no_lookup else None
            ),
            "positions_with_lookup": len(lookup_pos),
            "mean_run": sum(run_len) / len(run_len),
            "run_at_least": {k: sum(x >= k for x in run_len) / len(run_len) for k in (1, 2, 3, 4, 7)},
            "speedup": {},
        }
        for source in ("lookup", "model", "both"):
            for label, d in costs.items():
                if source == "lookup" and label != "measured":
                    continue
                key = source if source == "lookup" else f"{source} d={label}"
                entry["speedup"][key] = {
                    "lookup_ladder_then_model_pair": (
                        simulate(r, start, end, source, d, pause_model=True, model_pair=True)["speedup"]
                        if source == "both"
                        else None
                    ),
                    "policy": simulate(r, start, end, source, d, pause_model=True)["speedup"],
                    "policy_no_pause_for_model": simulate(r, start, end, source, d, pause_model=False)["speedup"],
                    "oracle": oracle(r, start, end, source, d),
                }
        entry["gated"] = {}
        for label, d in costs.items():
            for tau in (0, 0.3, 0.5, 0.7, 0.8, 0.9):
                entry["gated"][f"d={label} tau={tau}"] = {
                    "model": simulate_gate(r, prob, len(prompt), start, end, d, tau, False),
                    "lookup_then_model": simulate_gate(r, prob, len(prompt), start, end, d, tau, True),
                }
        result["prompts"][name] = entry
        print(
            f"{name}: tokenizer {same_tokens}, first draft token right {entry['first_token_accepted']:.2f}"
            f" (where lookup has no match {entry['first_token_accepted_without_lookup']}), mean run"
            f" {entry['mean_run']:.2f}"
        )
        for key, v in entry["speedup"].items():
            print(
                f"  {key:16s} policy {v['policy']:.2f}  no pause {v['policy_no_pause_for_model']:.2f}"
                f"  oracle {v['oracle']:.2f}"
                + (f"  lookup ladder, else pair {v['lookup_ladder_then_model_pair']:.2f}" if key.startswith("both") else "")
            )

    print("gated by draft confidence (tau 0: a pass of 2 everywhere): model / lookup, then model")
    for key in result["prompts"]["doc_qa"]["gated"]:
        print(
            f"  {key:20s}",
            "  ".join(
                f"{n} {e['gated'][key]['model']:.2f}/{e['gated'][key]['lookup_then_model']:.2f}"
                for n, e in result["prompts"].items()
            ),
        )
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
