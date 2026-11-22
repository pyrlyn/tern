---
status: todo
priority: P0
complexity: 2
readiness: 70%
agent:
---

# T1. Baselines on a quiet Mac

The goals are stated against llama.cpp (Metal) and MLX-LM on Ternary Bonsai 27B, and the rules
ask for a load average below 4. So far every run was made while other jobs kept the load between
4 and 100 and the GPU throttled for minutes at a time; `bench/compare.py` interleaves the three
engines so the comparison stays fair, but the absolute numbers are not peak numbers.

Done means: `bench/compare.py --rounds 5` finished with every recorded load average below 4, and
the tables in REPORT.md are replaced with that run, including the per-kernel breakdown from
`tern prof` at batch 1 and batch 32.

Check: `bench/results/compare.json` has `load1` below 4 for every run and REPORT.md quotes it.
