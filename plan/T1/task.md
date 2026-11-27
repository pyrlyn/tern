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

A second, separate goal is to be twice as fast as
[Atomic Engine](https://github.com/AtomicBot-ai/atomic-engine). Their published
figure is 105.71 tokens/s on a 48 GB M4 Max (Inkling 2 Mini IQ2_M with a DSpark
draft model, 14 GiB budget). That is a named throughput target, not part of
this quiet `compare.py` run; see [plan/README.md](../README.md).

Done means: `bench/compare.py --rounds 5` finished with every recorded load average below 4, and
the tables in REPORT.md are replaced with that run, including the per-kernel breakdown from
`tern prof` at batch 1 and batch 32.

Check: `bench/results/compare.json` has `load1` below 4 for every run and REPORT.md quotes it.
