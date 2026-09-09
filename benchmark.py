#!/usr/bin/env python3
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def validate_config(config):
    benchmark = config["benchmark"]
    generation = config["generation"]
    if benchmark["concurrency"] != 1:
        raise ValueError("concurrency must be 1")
    if config["network_layout"]["concurrency"] != benchmark["concurrency"]:
        raise ValueError("network and benchmark concurrency must match")
    for key in ("runs_per_point", "warmup_runs"):
        if not isinstance(benchmark[key], int) or benchmark[key] < (0 if key == "warmup_runs" else 1):
            raise ValueError(f"{key} has an invalid value")
    if not isinstance(generation["max_tokens"], int) or generation["max_tokens"] < 1:
        raise ValueError("max_tokens must be a positive integer")
    prompt_ids = {prompt["id"] for prompt in config["prompts"]}
    model_keys = {model["key"] for model in config["models"]}
    if len(prompt_ids) != len(config["prompts"]):
        raise ValueError("prompt ids must be unique")
    if len(model_keys) != len(config["models"]):
        raise ValueError("model keys must be unique")
    if any(prompt["target_tokens"] < 1 or not prompt["seed_text"] for prompt in config["prompts"]):
        raise ValueError("each prompt needs target_tokens and seed_text")
    return config


def load_config(path):
    return validate_config(json.loads(Path(path).read_text()))


def command_output(*command):
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_state(root):
    commit = command_output("git", "-C", str(root), "rev-parse", "HEAD")
    dirty = command_output("git", "-C", str(root), "status", "--porcelain")
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None}


def macos_free_memory_bytes(output):
    first_line, *lines = output.splitlines()
    page_size = int(first_line.split("page size of ", 1)[1].split(" bytes", 1)[0])
    pages = {}
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            pages[key] = int(value.strip().rstrip("."))
    return pages.get("Pages free", 0) * page_size


def available_memory_bytes():
    if platform.system() == "Darwin":
        output = command_output("vm_stat")
        if not output:
            return None
        return macos_free_memory_bytes(output)
    if platform.system() == "Linux":
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return None


def environment(host_label, root):
    system = platform.system()
    chip = command_output("sysctl", "-n", "machdep.cpu.brand_string") if system == "Darwin" else platform.processor()
    os_version = command_output("sw_vers", "-productVersion") if system == "Darwin" else platform.release()
    os_build = command_output("sw_vers", "-buildVersion") if system == "Darwin" else None
    available = available_memory_bytes()
    return {
        "host": host_label,
        "chip": chip,
        "os": system,
        "os_version": os_version,
        "os_build": os_build,
        "python": platform.python_version(),
        "mlx": importlib.metadata.version("mlx"),
        "mlx-lm": importlib.metadata.version("mlx-lm"),
        "git": git_state(root),
        "available_memory_gib_before_load": round(available / (1024 ** 3), 3) if available is not None else None,
    }


def check_versions(expected, actual):
    mismatches = []
    for name, version in expected.items():
        if actual.get(name) != version:
            mismatches.append(f"{name}: expected {version}, found {actual.get(name)}")
    if mismatches:
        raise RuntimeError("version mismatch: " + "; ".join(mismatches))


def materialize_prompt(tokenizer, prompt):
    seed_tokens = tokenizer.encode(prompt["seed_text"], add_special_tokens=False)
    if not seed_tokens:
        raise ValueError(f"prompt {prompt['id']} seed encoded to zero tokens")
    target = prompt["target_tokens"]
    return (seed_tokens * math.ceil(target / len(seed_tokens)))[:target]


def benchmark_once(model, tokenizer, prompt_tokens, generation):
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    mx.reset_peak_memory()
    mx.synchronize()
    started = time.perf_counter()
    first_token_seconds = None
    final = None
    text_hash = hashlib.sha256()
    sampler = make_sampler(temp=generation["temperature"])
    for response in stream_generate(
        model,
        tokenizer,
        prompt_tokens,
        max_tokens=generation["max_tokens"],
        sampler=sampler,
    ):
        if first_token_seconds is None:
            first_token_seconds = time.perf_counter() - started
        final = response
        text_hash.update(response.text.encode("utf-8"))
    mx.synchronize()
    total_seconds = time.perf_counter() - started
    if final is None:
        raise RuntimeError("generation returned no tokens")
    if final.prompt_tokens != len(prompt_tokens):
        raise RuntimeError(f"prompt token mismatch: expected {len(prompt_tokens)}, measured {final.prompt_tokens}")
    return {
        "time_to_first_token_seconds": first_token_seconds,
        "prefill_tokens_per_second": final.prompt_tps,
        "decode_tokens_per_second": final.generation_tps,
        "total_time_seconds": total_seconds,
        "peak_memory_gib": final.peak_memory,
        "generated_tokens": final.generation_tokens,
        "finish_reason": final.finish_reason,
        "output_sha256": text_hash.hexdigest(),
    }


def write_run(output, model_config, prompt, run_number, metrics, env, network_layout):
    result = {
        "schema_version": 1,
        "status": "complete",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "environment": env,
        "network_layout": network_layout,
        "model": {
            "key": model_config["key"],
            "model_id": model_config["model_id"],
            "revision": model_config["revision"],
            "quantization": model_config["quantization"],
            "present_before_baseline": model_config["present_before_baseline"],
        },
        "prompt": {"id": prompt["id"], "tokens": prompt["target_tokens"]},
        "generation": env["generation"],
        "run": run_number,
        "metrics": metrics,
    }
    path = output / f"{model_config['key']}__{prompt['id']}__run-{run_number}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


def build_summary(output):
    groups = defaultdict(list)
    for path in sorted(output.glob("*.json")):
        result = json.loads(path.read_text())
        if result.get("status") != "complete" or "metrics" not in result:
            continue
        key = (result["model"]["key"], result["prompt"]["tokens"])
        groups[key].append(result["metrics"])
    lines = [
        "# Benchmark summary",
        "",
        "Values are medians across measured runs. MLX reports prefill, decode, and peak memory.",
        "",
        "| Model | Prompt tokens | Runs | TTFT (s) | Prefill tok/s | Decode tok/s | Total (s) | Peak memory (GiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (model, prompt_tokens), metrics in sorted(groups.items()):
        median = lambda name: statistics.median(point[name] for point in metrics)
        lines.append(
            f"| {model} | {prompt_tokens} | {len(metrics)} | {median('time_to_first_token_seconds'):.3f} | "
            f"{median('prefill_tokens_per_second'):.2f} | {median('decode_tokens_per_second'):.2f} | "
            f"{median('total_time_seconds'):.3f} | {median('peak_memory_gib'):.3f} |"
        )
    return "\n".join(lines) + "\n"


def run(config, model_key, output, host_label, root):
    from mlx_lm import load

    model_config = next((model for model in config["models"] if model["key"] == model_key), None)
    if model_config is None:
        raise ValueError(f"unknown model key: {model_key}")
    env = environment(host_label, root)
    env["generation"] = config["generation"]
    check_versions(config["versions"], env)
    available = available_memory_bytes()
    if model_config["size_class"] == "large" and available is not None:
        minimum = config["benchmark"]["minimum_available_memory_gib_for_large_models"] * (1024 ** 3)
        if available < minimum:
            raise RuntimeError(
                f"large-model memory guard: {available / (1024 ** 3):.2f} GiB available, "
                f"{config['benchmark']['minimum_available_memory_gib_for_large_models']} GiB required"
            )

    output.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "load", "model": model_config["model_id"], "revision": model_config["revision"]}), flush=True)
    model, tokenizer = load(model_config["model_id"], revision=model_config["revision"])
    prompt_tokens = [(prompt, materialize_prompt(tokenizer, prompt)) for prompt in config["prompts"]]

    for prompt, tokens in prompt_tokens:
        for warmup in range(config["benchmark"]["warmup_runs"]):
            benchmark_once(model, tokenizer, tokens, config["generation"])
            print(json.dumps({"event": "warmup", "prompt": prompt["id"], "run": warmup + 1}), flush=True)
        for run_number in range(1, config["benchmark"]["runs_per_point"] + 1):
            metrics = benchmark_once(model, tokenizer, tokens, config["generation"])
            path = write_run(output, model_config, prompt, run_number, metrics, env, config["network_layout"])
            print(json.dumps({"event": "run", "path": path.name, "metrics": metrics}), flush=True)
    (output / "SUMMARY.md").write_text(build_summary(output))


def main():
    parser = argparse.ArgumentParser(description="Run a pinned MLX single-node inference baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--host-label", default="redacted-host")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    run(load_config(args.config), args.model, Path(args.output), args.host_label, root)


if __name__ == "__main__":
    main()
