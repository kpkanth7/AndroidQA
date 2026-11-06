from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCENARIOS = {
    "wifi": "Toggle Wi-Fi off and on",
    "bluetooth": "Turn Bluetooth on",
    "dark_mode": "Enable dark mode",
    "camera_permission": "Allow camera permission",
}
DEFAULT_SUITE = "canonical-v1"
_CANONICAL_CASES = {scenario: (goal, {}) for scenario, goal in SCENARIOS.items()}
SUITES = {
    DEFAULT_SUITE: _CANONICAL_CASES,
    "extended-v1": {
        **_CANONICAL_CASES,
        "wifi_initially_off": ("Toggle Wi-Fi off and on", {"screen": "network", "wifi_on": False}),
        "bluetooth_off": ("Turn Bluetooth off", {"screen": "network", "bluetooth_on": True}),
        "dark_mode_off": ("Disable dark mode", {"screen": "display", "dark_mode": True}),
        "camera_permission_denied": ("Deny camera permission", {"screen": "apps", "camera_permission": True}),
    },
}


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _event(records: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((record for record in records if record.get("event") == name), None)


def _summarize(results: list[dict[str, Any]], repeats: int, llm_planner: bool, *, suite: str = DEFAULT_SUITE) -> dict[str, Any]:
    count = len(results)
    latencies = [row["end_to_end_latency_ms"] for row in results if row["end_to_end_latency_ms"] is not None]
    messages = [row["agent_message_count"] for row in results if row["agent_message_count"] is not None]
    observed_replans = [row["replan_count"] for row in results if row["replan_count"] is not None]
    fallbacks = [row for row in results if row["plan_source"] in {
        "llm_with_rule_fallback", "rule_fallback_invalid_or_unavailable", "rule_replan"
    }]
    return {
        "configuration": "llm_planner" if llm_planner else "deterministic",
        "suite": suite,
        "repeats_per_scenario": repeats,
        "scenario_count": len(SUITES[suite]),
        "run_count": count,
        "success_rate": sum(row["success"] for row in results) / count,
        "median_end_to_end_latency_ms": statistics.median(latencies) if latencies else None,
        "p50_end_to_end_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_end_to_end_latency_ms": _percentile(latencies, 0.95),
        "p95_method": "nearest_rank",
        "latency_sample_count": len(latencies),
        "average_agent_message_count": statistics.mean(messages) if messages else None,
        "message_sample_count": len(messages),
        "replan_rate": sum(value > 0 for value in observed_replans) / len(observed_replans) if observed_replans else None,
        "replan_sample_count": len(observed_replans),
        # Keep the previous field for compatibility: it measures no-fallback
        # source, including unsuccessful runs; success is reported separately.
        "llm_plan_acceptance_rate": sum(row["plan_source"] == "llm" for row in results) / count if llm_planner else None,
        "llm_only_success_rate": sum(row["success"] and row["plan_source"] == "llm" for row in results) / count if llm_planner else None,
        "fallback_rate": len(fallbacks) / count if llm_planner else None,
        "llm_call_count": sum(row["llm_call_count"] for row in results),
        "llm_failed_call_count": sum(row["llm_failed_call_count"] for row in results),
        "runs": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark mock QA scenarios.")
    parser.add_argument("--repeats", type=int, default=3, help="Runs per scenario (default: 3).")
    parser.add_argument("--suite", choices=SUITES, default=DEFAULT_SUITE,
                        help="Scenario suite (default: canonical-v1; extended-v1 adds varied starts and goals).")
    parser.add_argument("--output-dir", default="./artifacts/benchmark", help="Directory for raw run data and summaries.")
    parser.add_argument("--llm-planner", action="store_true", help="Opt in to Groq planning; consumes provider quota.")
    parser.add_argument("--model", help="Optional provider model override used with --llm-planner.")
    parser.add_argument("--pause-between-runs", type=float, default=0, metavar="SECONDS",
                        help="Wait between runs to give provider limits time to recover.")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if not math.isfinite(args.pause_between_runs) or args.pause_between_runs < 0:
        parser.error("--pause-between-runs must be a finite, non-negative number")

    output_dir = Path(args.output_dir)
    raw_root = output_dir / "runs"
    if (output_dir / "summary.json").exists() or (output_dir / "summary.csv").exists() or raw_root.exists():
        parser.error("output directory already contains benchmark results; choose a new --output-dir")
    try:
        raw_root.mkdir(parents=True)
    except FileExistsError:
        parser.error("output directory already contains benchmark results; choose a new --output-dir")
    results: list[dict[str, Any]] = []
    package_dir = Path(__file__).resolve().parent
    source_digest = hashlib.sha256()
    for source in sorted(package_dir.rglob("*.py")):
        source_digest.update(source.relative_to(package_dir).as_posix().encode() + b"\0")
        source_digest.update(source.read_bytes())

    for scenario, (goal, initial_state) in SUITES[args.suite].items():
        for repeat in range(1, args.repeats + 1):
            if results and args.pause_between_runs:
                time.sleep(args.pause_between_runs)
            run_dir = raw_root / scenario / str(repeat)
            command = [
                sys.executable, "-m", "qa_agents.run_test", "--env", "mock",
                "--goal", goal, "--logs_dir", str(run_dir),
            ]
            if args.llm_planner:
                command.append("--llm_planner")
            if args.model:
                command.extend(["--model", args.model])
            if initial_state:
                command.extend(["--mock-initial-state", json.dumps(initial_state)])

            started_at = time.perf_counter()
            try:
                completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=180)
                exit_code = completed.returncode
            except subprocess.TimeoutExpired:
                exit_code = None
            process_wall_latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
            run_path = run_dir / "qa_run.json"
            records: list[dict[str, Any]] = []
            trace_error = None
            if run_path.exists():
                try:
                    records = json.loads(run_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    trace_error = "Run trace could not be read"
            finish = (_event(records, "finish") or {}).get("payload", {})
            metrics = (_event(records, "run_metrics") or {}).get("payload", {})
            plan = _event(records, "plan")
            llm_calls = [record["payload"] for record in records if record.get("event") == "llm"]
            metadata = (_event(records, "run_metadata") or {}).get("payload", {})
            finish_reason = finish.get("reason")
            if exit_code is None:
                finish_reason = "Run exceeded the benchmark's 180-second time limit"
            elif not finish:
                finish_reason = trace_error or f"Process exited with code {exit_code} before recording a finish event"
            results.append({
                "scenario": scenario,
                "goal": goal,
                "repeat": repeat,
                "success": exit_code == 0 and finish.get("status") == "passed",
                "exit_code": exit_code,
                "finish_reason": finish_reason,
                "process_wall_latency_ms": process_wall_latency_ms,
                "end_to_end_latency_ms": metrics.get("end_to_end_latency_ms"),
                "replan_count": metrics.get("replan_count"),
                "agent_message_count": metrics.get("agent_message_count"),
                "role_processing_ms": metrics.get("role_processing_ms"),
                "role_call_counts": metrics.get("role_call_counts"),
                "llm_latency_ms": sum(call["latency_ms"] for call in llm_calls) if llm_calls else None,
                "llm_call_count": len(llm_calls),
                "llm_failed_call_count": sum(not call.get("success") for call in llm_calls),
                "plan_source": plan["payload"].get("source") if plan else None,
                "provider_clients": metadata.get("clients", []),
                "planner_prompt_version": metadata.get("planner_prompt_version"),
                "mock_initial_state": metadata.get("mock_initial_state"),
                "run_dir": str(run_dir),
            })

    summary = _summarize(results, args.repeats, args.llm_planner, suite=args.suite)
    summary["source_sha256"] = source_digest.hexdigest()
    summary["pause_between_runs_seconds"] = args.pause_between_runs
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        json_only_fields = {"provider_clients", "role_processing_ms", "role_call_counts"}
        csv_fields = [key for key in results[0] if key not in json_only_fields]
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in results:
            csv_row = {key: row[key] for key in csv_fields}
            if csv_row["mock_initial_state"] is not None:
                csv_row["mock_initial_state"] = json.dumps(csv_row["mock_initial_state"])
            writer.writerow(csv_row)

    print(json.dumps({key: value for key, value in summary.items() if key != "runs"}, indent=2))
    return 0 if summary["success_rate"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
