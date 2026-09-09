# Benchmark summary

Values are medians across measured runs. MLX reports prefill, decode, and peak memory.

| Model | Prompt tokens | Runs | TTFT (s) | Prefill tok/s | Decode tok/s | Total (s) | Peak memory (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5-0.5b-instruct-4bit | 30 | 3 | 0.227 | 218.26 | 46.41 | 2.993 | 0.316 |
| qwen2.5-0.5b-instruct-4bit | 262 | 3 | 0.146 | 4268.67 | 217.37 | 0.742 | 0.523 |
| qwen2.5-0.5b-instruct-4bit | 1053 | 3 | 0.551 | 2261.95 | 85.74 | 2.057 | 1.276 |
