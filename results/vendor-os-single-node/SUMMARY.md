# Benchmark summary

Values are medians across consistent measured runs. The harness measures prefill and decode wall time. MLX reports peak memory.

| Model | Prompt tokens | Runs | TTFT (s) | Prefill tok/s | Decode tok/s | Total (s) | Peak memory (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5-0.5b-instruct-4bit | 30 | 3 | 0.133 | 226.33 | 146.10 | 1.000 | 0.313 |
| qwen2.5-0.5b-instruct-4bit | 262 | 3 | 0.186 | 1407.70 | 139.18 | 1.090 | 0.523 |
| qwen2.5-0.5b-instruct-4bit | 1053 | 3 | 0.314 | 3353.64 | 125.93 | 1.328 | 1.276 |
