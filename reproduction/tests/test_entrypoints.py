"""The documented paper entry points must start before any experiment is run."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("module,dataset,filename", [
    ("run_hparam_dev", "longmemeval", "longmemeval_s_cleaned.json"),
    ("run_multihop_scan", "locomo", "locomo10.json"),
])
def test_experiment_data_paths_match_documented_runner(module, dataset, filename):
    import importlib
    from reproduction.run import parse_cli

    experiment = importlib.import_module(f"reproduction.experiments.main.{module}")
    expected = REPO / "reproduction" / "data" / filename
    assert experiment.DATA == expected
    assert parse_cli(["--method", "sag-mem", "--dataset", dataset]).data_path == expected


@pytest.mark.parametrize("entrypoint", [
    "reproduction/run.py",
    "reproduction/experiments/ablations/prepare.py",
    "reproduction/experiments/ablations/run.py",
    "reproduction/experiments/ablations/validate_three.py",
    "reproduction/experiments/latency/run.py",
    "reproduction/experiments/lite/replay.py",
    "reproduction/experiments/lite/verify.py",
    "reproduction/experiments/lite/canonical_summary.py",
    "reproduction/experiments/ablations/verify.py",
    "reproduction/analysis/aggregate.py",
])
def test_help_starts_without_running_an_experiment(entrypoint):
    # Use module invocation where documented; run.py is also a standalone script.
    command = ([entrypoint] if entrypoint == "reproduction/run.py" else
               ["-m", entrypoint.removesuffix(".py").replace("/", ".")])
    result = subprocess.run([sys.executable, *command, "--help"], cwd=REPO,
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_documented_direct_aggregate_entrypoint():
    result = subprocess.run([sys.executable, "reproduction/analysis/aggregate.py", "--help"],
                            cwd=REPO, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("method", ["sag-mem", "sag-mem-lite"])
def test_preflight_needs_no_credentials(tmp_path, method):
    data = tmp_path / "data.json"
    data.write_text(json.dumps([{"sample_id": "offline", "conversation": {
        "session_1_date_time": "2026/11/05", "session_1": [
            {"dia_id": "m1", "speaker": "user", "text": "I moved to Paris."}]},
        "qa": [{"question": "Where did I move?", "answer": "Paris", "category": 4}]}]))
    env = {key: value for key, value in os.environ.items() if not key.endswith("API_KEY")}
    env["PYTHON_DOTENV_DISABLED"] = "1"
    result = subprocess.run([sys.executable, "reproduction/run.py", "--method", method,
                             "--dataset", "locomo", "--data-path", str(data), "--preflight"],
                            cwd=REPO, env=env, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["model_calls"] == 0
