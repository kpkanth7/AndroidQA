import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from qa_agents.benchmark import _summarize


def test_deterministic_benchmark_writes_summary(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "benchmark"
    env = {**os.environ, "PYTHONPATH": str(project_root / "src")}
    command = [
        sys.executable, "-m", "qa_agents.benchmark", "--repeats", "1", "--output-dir", str(output_dir),
    ]

    assert subprocess.call(command, env=env) == 0
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["configuration"] == "deterministic"
    assert summary["suite"] == "canonical-v1"
    assert summary["run_count"] == 4
    assert summary["success_rate"] == 1
    assert summary["llm_plan_acceptance_rate"] is None
    roles = {"planner", "executor", "verifier", "supervisor"}
    for run in summary["runs"]:
        assert run["finish_reason"] is None
        assert run["role_call_counts"].keys() == run["role_processing_ms"].keys() == roles
        assert all(count >= 1 for count in run["role_call_counts"].values())
    with (output_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        csv_run = next(csv.DictReader(handle))
    assert json.loads(csv_run["mock_initial_state"]) == summary["runs"][0]["mock_initial_state"]
    assert "role_processing_ms" not in csv_run

    # Reusing an output directory must not overwrite or silently reuse traces.
    original = (output_dir / "summary.json").read_text(encoding="utf-8")
    second_run = subprocess.run(command, env=env, capture_output=True, text=True)
    assert second_run.returncode != 0
    assert "already contains benchmark results" in second_run.stderr
    assert (output_dir / "summary.json").read_text(encoding="utf-8") == original


def test_extended_benchmark_records_each_actual_starting_state(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "extended"
    env = {**os.environ, "PYTHONPATH": str(project_root / "src")}
    command = [
        sys.executable, "-m", "qa_agents.benchmark", "--suite", "extended-v1",
        "--repeats", "1", "--output-dir", str(output_dir),
    ]

    assert subprocess.call(command, env=env) == 0
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["suite"] == "extended-v1"
    assert summary["run_count"] == summary["scenario_count"] == 8
    assert summary["success_rate"] == 1
    starting_states = {row["scenario"]: row["mock_initial_state"] for row in summary["runs"]}
    default = {
        "screen": "home", "wifi_on": True, "bluetooth_on": False,
        "dark_mode": False, "camera_permission": False,
    }
    assert starting_states == {
        "wifi": default,
        "bluetooth": default,
        "dark_mode": default,
        "camera_permission": default,
        "wifi_initially_off": {**default, "screen": "network", "wifi_on": False},
        "bluetooth_off": {**default, "screen": "network", "bluetooth_on": True},
        "dark_mode_off": {**default, "screen": "display", "dark_mode": True},
        "camera_permission_denied": {**default, "screen": "apps", "camera_permission": True},
    }


def test_summary_distinguishes_success_fallback_and_missing_metrics():
    def row(success, source, latency, messages, replans, calls, failed):
        return {"success": success, "plan_source": source, "end_to_end_latency_ms": latency,
                "agent_message_count": messages, "replan_count": replans,
                "llm_call_count": calls, "llm_failed_call_count": failed}

    results = [row(True, "llm", 10, 4, 0, 3, 0),
               row(True, "llm_with_rule_fallback", 20, 6, 1, 2, 1),
               row(False, "llm", 30, 5, 0, 4, 0),
               row(False, None, None, None, None, 0, 0)]
    summary = _summarize(results, repeats=1, llm_planner=True)

    assert summary["success_rate"] == .5
    assert summary["llm_plan_acceptance_rate"] == .5
    assert summary["llm_only_success_rate"] == .25
    assert summary["fallback_rate"] == .25
    assert summary["median_end_to_end_latency_ms"] == 20
    assert summary["latency_sample_count"] == 3
    assert summary["average_agent_message_count"] == 5
    assert summary["replan_rate"] == 1 / 3
    assert summary["llm_call_count"] == 9
    assert summary["llm_failed_call_count"] == 1


def test_even_sample_p50_uses_conventional_median():
    rows = [{"success": True, "plan_source": "llm", "end_to_end_latency_ms": latency,
             "agent_message_count": 1, "replan_count": 0, "llm_call_count": 1,
             "llm_failed_call_count": 0} for latency in (10, 20, 30, 40)]
    summary = _summarize(rows, repeats=1, llm_planner=True)
    assert summary["p50_end_to_end_latency_ms"] == 25
    assert summary["p95_end_to_end_latency_ms"] == 40
