# Omarchy Inference Fleet

This repository holds a small, reproducible MLX inference benchmark for vendor operating systems. It records single-node baselines so Apple and Linux nodes can run the same pinned models and prompts.

The JSON config is the benchmark contract. It pins model revisions, quantization, prompt token counts, generation settings, concurrency, repetitions, warmup, runtime versions, measurement conditions, and the single-node network layout. `benchmark.py` contains no benchmark parameter defaults. Schema version 2 separates wall-clock intervals and checks each result before writing it.

## Results

Published run JSON and the generated median table live under [`results/`](results/). Hostnames and local paths are not recorded.

Each result uses these metric intervals:

- `model_load_seconds` starts immediately before `mlx_lm.load` and ends when it returns the model and tokenizer. The harness loads once, then records that same startup interval with each measured run. It is not part of `total_time_seconds`.
- `tokenize_seconds` starts immediately before the tokenizer encodes the prompt seed and ends when the exact target-length token list is ready.
- `prefill_seconds` starts when the token list is ready and ends when `stream_generate` returns the first `GenerationResponse` to the harness. It includes stream setup, conversion of the token list to an MLX array, model prompt evaluation, first-token selection, detokenization, and the Python handoff to the harness.
- `time_to_first_token_seconds` starts immediately before tokenization and ends when the first `GenerationResponse` reaches the harness. It equals `tokenize_seconds + prefill_seconds` within 1 percent.
- `prefill_tokens_per_second` is `prompt_tokens / prefill_seconds`.
- `decode_seconds` starts when the first `GenerationResponse` reaches the harness and ends after the response stream is exhausted and the final `mx.synchronize()` returns.
- `decode_tokens_per_second` is the number of generated tokens after the first token divided by `decode_seconds`.
- `mlx_lm_reported_prompt_tok_s` is mlx-lm's internal `prompt_tps`. In mlx-lm 0.29.1 its timer starts inside `stream_generate` immediately before advancing `generate_step` and stops when the first token reaches that function. It excludes the outer stream setup and Python handoff included by `prefill_seconds`.
- `mlx_lm_reported_generation_tok_s` is mlx-lm's internal final `generation_tps`. It uses mlx-lm's timer and token count rather than the harness decode interval. The median table does not use it.
- `total_time_seconds` starts immediately before tokenization and ends after final GPU synchronization. It equals `tokenize_seconds + prefill_seconds + decode_seconds` within 1 percent.
- `peak_memory_gib` is mlx-lm's peak MLX memory reading after the harness resets the counter for the run.
- `prompt_tokens` and `generated_tokens` record the token counts used by the rate formulas.
- `finish_reason` records why mlx-lm stopped generation.
- `output_sha256` hashes the streamed text segments in order without storing generated text.

Before model load, the harness records `available_memory_gib_before_load` and `load_average_1m_before_load`. It refuses to load the model unless both meet the configured limits. The checked-in config requires at least 40 GiB free and a 1-minute load average no greater than 4.0.

Before writing a run, the harness checks the prefill rate, TTFT decomposition, and total-time decomposition. A failed check writes `status: "inconsistent"` and `inconsistency_reason`. The generated summary excludes inconsistent runs.

## Run

See [RUNBOOK.md](RUNBOOK.md). The harness uses Python's standard library plus `mlx-lm` and its MLX runtime.

## Test

```sh
python3 -m unittest test_benchmark.py
```

## License

MIT
