import json
import tempfile
import unittest
from pathlib import Path

from benchmark import build_summary, validate_config


def valid_config():
    return {
        "network_layout": {"concurrency": 1},
        "benchmark": {"concurrency": 1, "runs_per_point": 3, "warmup_runs": 1},
        "generation": {"max_tokens": 32, "temperature": 0.0},
        "prompts": [{"id": "short", "target_tokens": 30, "seed_text": "test"}],
        "models": [{"key": "model", "model_id": "org/model", "revision": "abc", "quantization": "4bit"}],
        "versions": {"mlx": "1", "mlx-lm": "1"},
    }


class BenchmarkConfigTests(unittest.TestCase):
    def test_rejects_concurrency_above_one(self):
        config = valid_config()
        config["benchmark"]["concurrency"] = 2

        with self.assertRaisesRegex(ValueError, "concurrency must be 1"):
            validate_config(config)

    def test_rejects_network_concurrency_drift(self):
        config = valid_config()
        config["network_layout"]["concurrency"] = 2

        with self.assertRaisesRegex(ValueError, "network and benchmark concurrency must match"):
            validate_config(config)


class SummaryTests(unittest.TestCase):
    def test_builds_median_table_from_run_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for run, ttft in enumerate((1.0, 3.0, 2.0), start=1):
                result = {
                    "status": "complete",
                    "model": {"key": "tiny", "model_id": "org/tiny", "quantization": "4bit"},
                    "prompt": {"id": "p30", "tokens": 30},
                    "run": run,
                    "metrics": {
                        "time_to_first_token_seconds": ttft,
                        "prefill_tokens_per_second": 10.0 + run,
                        "decode_tokens_per_second": 20.0 + run,
                        "total_time_seconds": 4.0 + run,
                        "peak_memory_gib": 1.0 + run,
                    },
                }
                (output / f"run-{run}.json").write_text(json.dumps(result))

            summary = build_summary(output)

        self.assertIn("| tiny | 30 | 3 | 2.000 | 12.00 | 22.00 | 6.000 | 3.000 |", summary)


if __name__ == "__main__":
    unittest.main()
