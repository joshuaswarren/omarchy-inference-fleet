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
    if config["network_layout"]["concurrency"] != benchmark["concurrency"]:
        raise ValueError("network and benchmark concurrency must match")
    for key in ("runs_per_point", "warmup_runs"):
        if not isinstance(benchmark[key], int) or benchmark[key] < (0 if key == "warmup_runs" else 1):
            raise ValueError(f"{key} has an invalid value")
    for key in ("minimum_available_memory_gib", "maximum_load_average_1m"):
        value = benchmark.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
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
    try:
        load_average = os.getloadavg()[0]
    except OSError:
        load_average = None
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
        "load_average_1m_before_load": round(load_average, 3) if load_average is not None else None,
    }


def measurement_guard_reason(available_gib, load_average, minimum_memory_gib, maximum_load_average):
    if available_gib is None:
        return "measurement conditions guard: available memory could not be measured"
    if available_gib < minimum_memory_gib:
        return f"measurement conditions guard: {available_gib:.2f} GiB available, {minimum_memory_gib} GiB required"
    if load_average is None:
        return "measurement conditions guard: 1-minute load average could not be measured"
    if load_average > maximum_load_average:
        return (
            f"measurement conditions guard: 1-minute load average {load_average:.2f}, "
            f"maximum {maximum_load_average}"
        )
    return None


def relative_error(actual, expected):
    if expected == 0:
        return 0.0 if actual == 0 else math.inf
    return abs(actual - expected) / abs(expected)


def check_measurement_consistency(metrics):
    failures = []
    expected_prefill_rate = metrics["prompt_tokens"] / metrics["prefill_seconds"]
    if relative_error(metrics["prefill_tokens_per_second"], expected_prefill_rate) > 0.01:
        failures.append("prefill rate does not equal prompt tokens divided by prefill seconds within 1 percent")
    expected_ttft = metrics["tokenize_seconds"] + metrics["prefill_seconds"]
    if relative_error(metrics["time_to_first_token_seconds"], expected_ttft) > 0.01:
        failures.append("time to first token does not equal tokenize plus prefill intervals within 1 percent")
    expected_total = expected_ttft + metrics["decode_seconds"]
    if relative_error(metrics["total_time_seconds"], expected_total) > 0.01:
        failures.append("tokenize, prefill, and decode intervals do not sum to total time within 1 percent")
    return "; ".join(failures) or None


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


def benchmark_once(model, tokenizer, prompt, generation, model_load_seconds):
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=generation["temperature"])
    mx.reset_peak_memory()
    mx.synchronize()
    started = time.perf_counter()
    prompt_tokens = materialize_prompt(tokenizer, prompt)
    tokenized = time.perf_counter()
    first_token_at = None
    final = None
    text_hash = hashlib.sha256()
    for response in stream_generate(
        model,
        tokenizer,
        prompt_tokens,
        max_tokens=generation["max_tokens"],
        sampler=sampler,
    ):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        final = response
        text_hash.update(response.text.encode("utf-8"))
    mx.synchronize()
    finished = time.perf_counter()
    if final is None:
        raise RuntimeError("generation returned no tokens")
    if final.prompt_tokens != len(prompt_tokens):
        raise RuntimeError(f"prompt token mismatch: expected {len(prompt_tokens)}, measured {final.prompt_tokens}")

    tokenize_seconds = tokenized - started
    prefill_seconds = first_token_at - tokenized
    decode_seconds = finished - first_token_at
    return {
        "prompt_tokens": len(prompt_tokens),
        "model_load_seconds": model_load_seconds,
        "tokenize_seconds": tokenize_seconds,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "time_to_first_token_seconds": first_token_at - started,
        "prefill_tokens_per_second": len(prompt_tokens) / prefill_seconds,
        "decode_tokens_per_second": (final.generation_tokens - 1) / decode_seconds if final.generation_tokens > 1 else 0.0,
        "mlx_lm_reported_prompt_tok_s": final.prompt_tps,
        "mlx_lm_reported_generation_tok_s": final.generation_tps,
        "total_time_seconds": finished - started,
        "peak_memory_gib": final.peak_memory,
        "generated_tokens": final.generation_tokens,
        "finish_reason": final.finish_reason,
        "output_sha256": text_hash.hexdigest(),
    }


def write_run(output, model_config, prompt, run_number, metrics, env, config):
    inconsistency = check_measurement_consistency(metrics)
    result = {
        "schema_version": config["schema_version"],
        "status": "inconsistent" if inconsistency else "complete",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "environment": env,
        "network_layout": config["network_layout"],
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
    if inconsistency:
        result["inconsistency_reason"] = inconsistency
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
        "Values are medians across consistent measured runs. The harness measures prefill and decode wall time. MLX reports peak memory.",
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
    benchmark_config = config["benchmark"]
    env["minimum_available_memory_gib"] = benchmark_config["minimum_available_memory_gib"]
    env["maximum_load_average_1m"] = benchmark_config["maximum_load_average_1m"]
    guard_reason = measurement_guard_reason(
        env["available_memory_gib_before_load"],
        env["load_average_1m_before_load"],
        benchmark_config["minimum_available_memory_gib"],
        benchmark_config["maximum_load_average_1m"],
    )
    if guard_reason:
        raise RuntimeError(guard_reason)

    output.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "load", "model": model_config["model_id"], "revision": model_config["revision"]}), flush=True)
    load_started = time.perf_counter()
    model, tokenizer = load(model_config["model_id"], revision=model_config["revision"])
    model_load_seconds = time.perf_counter() - load_started

    for prompt in config["prompts"]:
        for warmup in range(benchmark_config["warmup_runs"]):
            benchmark_once(model, tokenizer, prompt, config["generation"], model_load_seconds)
            print(json.dumps({"event": "warmup", "prompt": prompt["id"], "run": warmup + 1}), flush=True)
        for run_number in range(1, benchmark_config["runs_per_point"] + 1):
            metrics = benchmark_once(model, tokenizer, prompt, config["generation"], model_load_seconds)
            path = write_run(output, model_config, prompt, run_number, metrics, env, config)
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
