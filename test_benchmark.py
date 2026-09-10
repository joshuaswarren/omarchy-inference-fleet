import json
import tempfile
import unittest
from pathlib import Path

from benchmark import (
    build_summary,
    check_measurement_consistency,
    macos_free_memory_bytes,
    measurement_guard_reason,
    measurement_requirements,
    validate_config,
    write_run,
)


def valid_config():
    return {
        "schema_version": 2,
        "network_layout": {"concurrency": 1},
        "benchmark": {
            "concurrency": 1,
            "runs_per_point": 3,
            "warmup_runs": 1,
            "minimum_available_memory_gib": 40,
            "maximum_load_average_1m_per_core": 0.2,
        },
        "generation": {"max_tokens": 32, "temperature": 0.0},
        "prompts": [{"id": "short", "target_tokens": 30, "seed_text": "test"}],
        "models": [{"key": "model", "model_id": "org/model", "revision": "abc", "quantization": "4bit"}],
        "versions": {"mlx": "1", "mlx-lm": "1"},
    }


class BenchmarkConfigTests(unittest.TestCase):

    def test_rejects_network_concurrency_drift(self):
        config = valid_config()
        config["network_layout"]["concurrency"] = 2

        with self.assertRaisesRegex(ValueError, "network and benchmark concurrency must match"):
            validate_config(config)

    def test_rejects_invalid_per_model_memory_floor(self):
        config = valid_config()
        config["models"][0]["min_free_memory_gib"] = 0

        with self.assertRaisesRegex(ValueError, "min_free_memory_gib"):
            validate_config(config)


class MemoryTests(unittest.TestCase):
    def test_macos_guard_uses_free_pages_only(self):
        vm_stat = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 100.
Pages inactive: 900.
Pages speculative: 500.
Pages purgeable: 200.
"""

        self.assertEqual(100 * 16384, macos_free_memory_bytes(vm_stat))


class MeasurementGuardTests(unittest.TestCase):
    def test_model_floor_overrides_global_default(self):
        config = valid_config()
        config["models"][0]["min_free_memory_gib"] = 8

        required, _ = measurement_requirements(config["benchmark"], config["models"][0], 10)

        self.assertEqual(8, required)

    def test_model_without_floor_uses_global_default(self):
        config = valid_config()

        required, _ = measurement_requirements(config["benchmark"], config["models"][0], 10)

        self.assertEqual(40, required)

    def test_load_ceiling_scales_with_core_count(self):
        config = valid_config()

        _, ten_core_ceiling = measurement_requirements(config["benchmark"], config["models"][0], 10)
        _, twenty_core_ceiling = measurement_requirements(config["benchmark"], config["models"][0], 20)

        self.assertEqual(2.0, ten_core_ceiling)
        self.assertEqual(4.0, twenty_core_ceiling)

    def test_rejects_low_memory_before_model_load(self):
        reason = measurement_guard_reason(39.5, 1.0, 40, 4.0)

        self.assertEqual("measurement conditions guard: 39.50 GiB available, 40 GiB required", reason)

    def test_rejects_high_load_before_model_load(self):
        reason = measurement_guard_reason(64.0, 4.5, 40, 4.0)

        self.assertEqual("measurement conditions guard: 1-minute load average 4.50, maximum 4.0", reason)

    def test_accepts_idle_host_with_enough_memory(self):
        self.assertIsNone(measurement_guard_reason(64.0, 2.0, 40, 4.0))


class ConsistencyTests(unittest.TestCase):
    def test_accepts_reconciled_intervals(self):
        metrics = {
            "prompt_tokens": 100,
            "tokenize_seconds": 0.01,
            "prefill_seconds": 0.04,
            "prefill_tokens_per_second": 2500.0,
            "time_to_first_token_seconds": 0.05,
            "decode_seconds": 0.45,
            "total_time_seconds": 0.50,
        }

        self.assertIsNone(check_measurement_consistency(metrics))

    def test_inconsistent_result_is_written_with_reason(self):
        metrics = {
            "prompt_tokens": 100,
            "tokenize_seconds": 0.01,
            "prefill_seconds": 0.04,
            "prefill_tokens_per_second": 1000.0,
            "time_to_first_token_seconds": 0.05,
            "decode_seconds": 0.45,
            "total_time_seconds": 0.50,
        }
        config = valid_config()
        config["schema_version"] = 2
        model = config["models"][0]
        model.update({"size_class": "comparison", "present_before_baseline": False})
        env = {"generation": config["generation"]}

        with tempfile.TemporaryDirectory() as directory:
            path = write_run(Path(directory), model, config["prompts"][0], 1, metrics, env, config)
            result = json.loads(path.read_text())

        self.assertEqual("inconsistent", result["status"])
        self.assertIn("prefill rate", result["inconsistency_reason"])


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
