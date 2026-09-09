# Omarchy Inference Fleet

This repository holds a small, reproducible MLX inference benchmark for vendor operating systems. It records single-node baselines now so later Apple and Linux nodes can run the same pinned models and prompts.

The JSON config is the benchmark contract. It pins model revisions, quantization, prompt token counts, generation settings, concurrency, repetitions, warmup, runtime versions, the memory guard, and the single-node network layout. `benchmark.py` contains no benchmark parameter defaults.

## Results

Published run JSON and the generated median table live under [`results/`](results/). Hostnames and local paths are not recorded. Each run records:

- time to first token from the generation call to the first streamed token;
- MLX-reported prompt prefill tokens per second;
- MLX-reported generation tokens per second;
- total wall time;
- MLX peak memory;
- chip, operating system, Python, MLX, mlx-lm, and harness commit.

## Run

See [RUNBOOK.md](RUNBOOK.md). The harness uses Python's standard library plus `mlx-lm` and its MLX runtime.

## Test

```sh
python3 -m unittest test_benchmark.py
```

## License

MIT
