"""Run a small, separately labeled LLM pilot against an already booted emulator.

Use the AndroidWorld Python environment. Bluetooth preparation runs use the
rule planner to establish the opposite state; they are not pilot observations.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import qa_agents


CASES = (
    ("wifi_cycle", "settings_wifi", "Toggle Wi-Fi off and on", None),
    ("bluetooth_on", "settings_bluetooth", "Turn Bluetooth on", "Turn Bluetooth off"),
    ("bluetooth_off", "settings_bluetooth", "Turn Bluetooth off", "Turn Bluetooth on"),
)


def summarize_run(directory, exit_code):
    trace = directory / "qa_run.json"
    records = json.loads(trace.read_text()) if trace.exists() else []
    payload = lambda name: next((r["payload"] for r in records if r["event"] == name), {})
    metadata, finish, plan = payload("run_metadata"), payload("finish"), payload("plan")
    calls = [r["payload"] for r in records if r["event"] == "llm"]
    errors = Counter(call["error_type"] for call in calls if not call["success"])
    fallback = plan.get("source") in {"llm_with_rule_fallback", "rule_fallback_invalid_or_unavailable", "rule_replan"}
    verified = exit_code == 0 and finish.get("status") == "passed" and metadata.get("environment") == "android"
    return {
        "verified_pass": verified,
        "llm_only_pass": verified and plan.get("source") == "llm" and not fallback,
        "fallback": fallback,
        "exit_code": exit_code,
        "finish_reason": finish.get("reason") if finish else "No complete run trace (timeout or startup failure)",
        "plan_source": plan.get("source"),
        "llm_calls": len(calls),
        "provider_errors": sum(count for name, count in errors.items() if name != "MalformedPlanError"),
        "malformed_responses": errors.get("MalformedPlanError", 0),
        "errors_by_type": dict(errors),
        "metadata": metadata,
        "metrics": payload("run_metrics"),
        "trace": str(trace),
    }


def run_case(args, task, goal, directory, *, llm):
    directory.mkdir(parents=True)
    command = [sys.executable, "-m", "qa_agents.run_test", "--env", "android", "--task", task,
               "--goal", goal, "--adb-path", args.adb_path, "--logs_dir", str(directory)]
    if llm:
        command.append("--llm_planner")
    if args.model:
        command.extend(["--model", args.model])
    if args.save_frames:
        command.append("--save_frames")
    # A finite subprocess deadline also bounds startup and device stalls.
    with (directory / "process.log").open("w") as log:
        try:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600)
            code = completed.returncode
        except subprocess.TimeoutExpired:
            code = None
    return summarize_run(directory, code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--pause-between-runs", type=float, default=20)
    parser.add_argument("--save-frames", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not math.isfinite(args.pause_between_runs) or args.pause_between_runs < 0:
        parser.error("--pause-between-runs must be finite and non-negative")
    output = Path(args.output_dir).resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("output directory exists; choose a fresh directory")
    source_root = Path(qa_agents.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root)
        digest.update(relative.as_posix().encode() + b"\0" + path.read_bytes())
        destination = output / "source" / "qa_agents" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    shutil.copyfile(__file__, output / "source" / "emulator_pilot.py")
    runs = []
    for repeat in range(1, args.repeats + 1):
        for case, task, goal, preparation in CASES:
            if runs:
                time.sleep(args.pause_between_runs)
            directory = output / "runs" / f"{case}-{repeat}"
            if preparation:
                prepared = run_case(args, task, preparation, output / "preparation" / f"{case}-{repeat}", llm=False)
                if not prepared["verified_pass"]:
                    raise RuntimeError(f"Could not prepare {case}; inspect {prepared['trace']}")
            row = {"case": case, "repeat": repeat, "goal": goal, **run_case(args, task, goal, directory, llm=True)}
            runs.append(row)
            summary = {
                "environment": "android", "suite": "pixel6-settings-pilot-v1", "model": args.model,
                "source_sha256": digest.hexdigest(), "repeats": args.repeats,
                "pause_between_runs_seconds": args.pause_between_runs,
                "run_count": len(runs), "planned_run_count": len(CASES) * args.repeats,
                "verified_passes": sum(r["verified_pass"] for r in runs),
                "llm_only_passes": sum(r["llm_only_pass"] for r in runs),
                "fallback_runs": sum(r["fallback"] for r in runs),
                "provider_errors": sum(r["provider_errors"] for r in runs),
                "malformed_responses": sum(r["malformed_responses"] for r in runs),
                "llm_calls": sum(r["llm_calls"] for r in runs),
                "limitation": "Small emulator pilot; separate from mock benchmarks and not a reliability estimate.",
                "runs": runs,
            }
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps({key: value for key, value in row.items() if key not in {"metadata", "metrics"}}), flush=True)
    return 0 if all(row["verified_pass"] for row in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
