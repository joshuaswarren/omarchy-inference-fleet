# Vendor OS single-node results

These schema 2 results were measured on an Apple M1 Max MacBook Pro with 64 GiB unified memory. The machine ran macOS 26.6.2 build 25G83. The environment used Python 3.9.6, MLX 0.29.3, and mlx-lm 0.29.1.

The model was `mlx-community/Qwen2.5-0.5B-Instruct-4bit` at revision `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`. Each prompt length had one warmup and three measured runs. Each run generated 128 tokens at concurrency one.

The pre-load guard observed 18.598 GiB free against an 8 GiB requirement. It observed a 1-minute load average of 1.705. The derived ceiling was 2.0 for 10 logical cores. All nine files passed the schema 2 self-check.

| Prompt tokens | Runs | TTFT (s) | Prefill tok/s | Decode tok/s | Total (s) | Peak memory (GiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 3 | 0.133 | 226.33 | 146.10 | 1.000 | 0.313 |
| 262 | 3 | 0.186 | 1407.70 | 139.18 | 1.090 | 0.523 |
| 1053 | 3 | 0.314 | 3353.64 | 125.93 | 1.328 | 1.276 |

The values are medians across the three measured runs. See `SUMMARY.md` and the nine JSON files for the complete measurements. Each JSON file records the harness commit and measured conditions.

The superseded schema 1 set came from an M1 Ultra and used incompatible timing intervals. Do not compare these M1 Max numbers to that withdrawn set.
