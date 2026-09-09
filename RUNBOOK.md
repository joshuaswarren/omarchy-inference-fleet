# Benchmark runbook

## Preconditions

Use an isolated shell on the node. Do not stop or reconfigure inference services. Confirm the Python, MLX, and mlx-lm versions match `config.json`:

```sh
python3 -c 'import importlib.metadata, platform; print(platform.python_version()); print(importlib.metadata.version("mlx")); print(importlib.metadata.version("mlx-lm"))'
```

Check disk, load, and memory before a run:

```sh
df -h /
top -l 1 -n 0 | head -n 15
vm_stat
```

On Linux, replace the macOS checks with `df -h /`, `top -b -n 1`, and `/proc/meminfo`. The harness reads `MemAvailable` on Linux. On macOS it uses only `vm_stat` free pages. It reads the 1-minute load average from the operating system on both platforms.

The checked-in config requires at least 40 GiB free and a 1-minute load average no greater than 4.0. The harness checks both conditions before model load. A failed guard exits without creating a result.

## Run the pinned points

Run from a clean clone so each JSON receipt records the exact harness commit:

```sh
python3 benchmark.py \
  --config config.json \
  --model qwen2.5-0.5b-instruct-4bit \
  --output results/vendor-os-single-node \
  --host-label mac-m1-ultra

python3 benchmark.py \
  --config config.json \
  --model qwen3-14b-4bit \
  --output results/vendor-os-single-node \
  --host-label mac-m1-ultra
```

The first run may download the pinned Hugging Face revision. Review free disk before allowing it. The output directory receives one JSON file per measured run and a regenerated `SUMMARY.md` with medians across consistent runs. Inspect any file with `status: "inconsistent"` before using or publishing its measurements.

## Add a Linux node later

Install a compatible MLX and mlx-lm build in an isolated Python environment on the Linux node. Update only the version pins if that platform requires different package versions, then run the same commands. Keep the model revisions, prompts, generation settings, concurrency, repetitions, warmup, measurement limits, and network layout unchanged when comparing nodes.

## Result handling

Use a generic `--host-label`. Do not put hostnames or local model paths in public output. Preserve full command output and local paths only in the private operations repository.
