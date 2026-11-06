from __future__ import annotations
import argparse, json, os, time
from contextlib import ExitStack
from functools import partial
from .agents.planner import Planner
from .agents.executor import Executor
from .agents.verifier import Verifier
from .agents.supervisor import Supervisor
from .llm.openai_client import OBSERVATION_PROMPT_VERSION
from .mock_contracts import mock_goal
from .utils.schemas import LogRecord, Plan
from .envs.mock_android import MockAndroidEnv

MAX_LLM_STEPS = 15

def save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", type=str, default="Toggle Wi‑Fi off and on")
    parser.add_argument("--env", type=str, choices=["mock","android"], default="mock")
    parser.add_argument("--task", type=str, default="settings_wifi", help="Android task name (for android env)")
    parser.add_argument("--adb-path", help="Path to the ADB executable for --env android.")
    parser.add_argument("--logs_dir", type=str, default="./artifacts/latest")
    parser.add_argument("--save_frames", action="store_true", help="Save rendered frames if env supports it")
    parser.add_argument("--llm_planner", action="store_true", help="Use bounded observation-driven LLM planning in the selected environment.")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model name (e.g., openai/gpt-oss-20b)")
    parser.add_argument("--mock-initial-state", help="Mock-only JSON object overriding the starting screen or setting values.")
    parser.add_argument("--inject-failure-once", choices=["toggle"], help="Mock-only: fail this action once to demonstrate recovery.")
    parser.add_argument("--max-replans", type=int, default=1, help="Maximum deterministic recovery plans after a failed step.")
    args = parser.parse_args()
    if args.env != "mock" and args.mock_initial_state is not None:
        parser.error("--mock-initial-state is available only with --env mock")
    if args.env != "mock" and args.inject_failure_once:
        parser.error("--inject-failure-once is available only with --env mock")
    if args.env == "mock" and args.adb_path is not None:
        parser.error("--adb-path is available only with --env android")
    if args.env == "android":
        task = mock_goal(args.goal)
        supported = task is not None and (
            (args.task == "settings_wifi" and task.wifi_cycle)
            or (args.task == "settings_bluetooth" and task.setting == "bluetooth_on")
        )
        if not supported:
            parser.error("--env android supports settings_wifi with 'Toggle Wi-Fi off and on' "
                         "or settings_bluetooth with 'Turn Bluetooth on' / 'Turn Bluetooth off'")
    if args.max_replans < 0:
        parser.error("--max-replans must be non-negative")
    with ExitStack() as resources:
        return _run(args, parser, resources)


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser, resources: ExitStack) -> int:
    run_started_at = time.perf_counter()
    role_seconds = dict.fromkeys(("planner", "executor", "verifier", "supervisor"), 0.0)
    role_calls = dict.fromkeys(role_seconds, 0)

    def call_role(role, method, *args):
        started_at = time.perf_counter()
        try:
            return method(*args)
        finally:
            role_seconds[role] += time.perf_counter() - started_at
            role_calls[role] += 1

    # 1) Create agents
    planner = Planner(
        use_llm=args.llm_planner,
        model=args.model,
        cache_path=os.path.join(args.logs_dir, "plan_cache.json"),
    )
    verifier = Verifier(goal=args.goal)
    # The report is deterministic so it never makes unsupported claims about
    # action failures or recovery. Groq is used for planning only in this phase.
    supervisor = Supervisor(model=args.model, use_llm=False)

    # 2) Environment
    plan_from_observation = planner.recovery_plan
    if args.env == "mock":
        try:
            initial_state = json.loads(args.mock_initial_state) if args.mock_initial_state is not None else {}
            if not isinstance(initial_state, dict):
                raise ValueError("Mock initial state must be a JSON object")
            env = MockAndroidEnv(fail_once_action=args.inject_failure_once, initial_state=initial_state)
        except ValueError as error:
            parser.error(f"--mock-initial-state: {error}")
        executor = Executor(env)
    else:
        from .envs.android_world_env import AndroidWorldEnv, WIFI_ROUTE, BLUETOOTH_ROUTE
        try:
            env = AndroidWorldEnv(task_name=args.task, enable_render=args.save_frames, adb_path=args.adb_path)
            resources.callback(env.close)
            executor = Executor(env)
        except Exception as error:
            parser.exit(1, f"Could not start AndroidWorld: {error}\n")
        route = WIFI_ROUTE if args.task == "settings_wifi" else BLUETOOTH_ROUTE
        plan_from_observation = partial(planner.recovery_plan, route=route)
        planner.last_plan_source = "android_rule"

    # 3) Plan
    # Live planning is observation-driven and deliberately bypasses the
    # goal-only cache in both environments.
    interactive = args.llm_planner
    unsupported_goal = args.env == "mock" and mock_goal(args.goal) is None
    if interactive or unsupported_goal:
        plan = Plan(goal=args.goal, steps=[])
    elif args.env == "android" or args.mock_initial_state is not None:
        plan = call_role("planner", plan_from_observation, args.goal, executor.obs, [])
    else:
        plan = call_role("planner", planner.plan, args.goal)
    if unsupported_goal:
        planner.last_plan_source = "unsupported_goal"
    elif interactive:
        planner.last_plan_source = "llm"

    # 4) Loop
    log_records = []
    llm_clients = [client for client in (planner.llm, supervisor.llm) if client]
    metadata = {
        "environment": args.env,
        "llm_enabled": args.llm_planner,
        "clients": [client.metadata() for client in llm_clients],
    }
    if args.env == "mock":
        metadata["mock_initial_state"] = {"screen": executor.obs.screen, **executor.obs.info}
    else:
        metadata["android_task"] = args.task
        metadata["android_initial_state"] = executor.obs.model_dump()
    if interactive:
        metadata["planner_prompt_version"] = OBSERVATION_PROMPT_VERSION
        metadata["max_llm_steps"] = MAX_LLM_STEPS
    log_records.append(LogRecord(event="run_metadata", payload=metadata).model_dump())
    plan_record = LogRecord(event="plan", payload={
        **plan.model_dump(),
        "source": planner.last_plan_source,
    }).model_dump()
    log_records.append(plan_record)
    for client in llm_clients:
        for call in client.calls:
            log_records.append(LogRecord(event="llm", payload=call).model_dump())
        client.calls.clear()
    finish_reason = "Goal is not supported by the mock environment" if unsupported_goal else None
    final_verify_pass = None
    replan_count = 0

    # frames dir (optional)
    frames_dir = os.path.join(args.logs_dir, "frames")
    imageio = None
    if args.save_frames:
        os.makedirs(frames_dir, exist_ok=True)
        try:
            import imageio.v2 as imageio  # for imwrite
        except ImportError:
            pass

    steps_summary = []
    history = []
    action_count = 0
    fallback_used = False
    llm_actions = 0

    active_plan = None if unsupported_goal else plan
    while active_plan is not None:
        if interactive and not fallback_used:
            if action_count >= MAX_LLM_STEPS:
                finish_reason = f"LLM step budget exhausted after {MAX_LLM_STEPS} steps"
                break
            next_step = call_role("planner", planner.plan_next, args.goal, executor.obs, history)
            if next_step is None and planner.last_action_rejection.startswith("Action unavailable"):
                reason = planner.last_action_rejection
                log_records.append(LogRecord(event="plan_rejection", payload={"reason": reason}).model_dump())
                next_step = call_role("planner", planner.plan_next, args.goal, executor.obs, history + [{"feedback": reason}])
                if next_step is None and planner.last_action_rejection.startswith("Action unavailable"):
                    log_records.append(LogRecord(event="plan_rejection", payload={
                        "reason": planner.last_action_rejection,
                    }).model_dump())
            for client in llm_clients:
                for call in client.calls:
                    log_records.append(LogRecord(event="llm", payload=call).model_dump())
                client.calls.clear()
            if next_step is None:
                fallback_used = True
                planner.last_plan_source = "rule_fallback_invalid_or_unavailable"
                plan_record["payload"]["source"] = "llm_with_rule_fallback" if llm_actions else planner.last_plan_source
                log_records.append(LogRecord(event="plan_fallback", payload={"reason": planner.last_action_rejection or "Model action invalid or unavailable; using rule fallback"}).model_dump())
                active_plan = call_role("planner", plan_from_observation, args.goal, executor.obs, history)
                for index, step in enumerate(active_plan.steps, start=action_count + 1):
                    step.id = index
                plan_record["payload"]["steps"].extend(step.model_dump() for step in active_plan.steps)
            else:
                llm_actions += 1
                next_step.id = action_count + 1
                plan_record["payload"]["steps"].append(next_step.model_dump())
                active_plan = Plan(goal=args.goal, steps=[next_step])
        log_records.append(LogRecord(event="agent_message", payload={
            "from": "planner", "to": "executor", "kind": "plan", "plan_source": planner.last_plan_source,
        }).model_dump())
        should_replan = False

        for step in active_plan.steps:
            action_count += 1
            # Execute
            previous_info = executor.obs.info
            previous_toggles = executor.obs.ui_tree.get("toggles", {})
            res = call_role("executor", executor.execute, step.intent, step.target, step.params)
            history.append({"intent": step.intent, "target": step.target, "params": step.params,
                            "ok": res.ok, "message": res.message, "observation": res.observation.model_dump(),
                            "previous_info": previous_info, "previous_toggles": previous_toggles})
            log_records.append(LogRecord(event="action", payload={
                "step_id": step.id,
                "intent": step.intent,
                "target": step.target,
                "params": step.params,
                "ok": res.ok,
                "message": res.message,
                "observation": res.observation.model_dump()
            }).model_dump())
            log_records.append(LogRecord(event="agent_message", payload={
                "from": "executor", "to": "verifier", "kind": "action_result", "step_id": step.id,
            }).model_dump())

            # Optional frame capture
            if imageio is not None and hasattr(env, "render"):
                try:
                    frame = env.render()  # numpy array (H,W,3)
                    if frame is not None:
                        fname = os.path.join(frames_dir, f"step_{step.id:02d}_attempt_{replan_count}.png")
                        imageio.imwrite(fname, frame)
                except Exception:
                    pass  # non-fatal

            ver = call_role("verifier", verifier.verify_result, step, res, history)
            log_records.append(LogRecord(event="verify", payload={
                "step_id": step.id,
                "passed": ver.passed,
                "reason": ver.reason,
                "need_replan": ver.need_replan,
            }).model_dump())
            log_records.append(LogRecord(event="agent_message", payload={
                "from": "verifier", "to": "supervisor", "kind": "verification", "step_id": step.id,
            }).model_dump())

            steps_summary.append({
                "id": step.id,
                "intent": step.intent,
                "target": step.target or "",
                "action_ok": res.ok,
                "action_msg": res.message,
                "verify_passed": ver.passed,
                "verify_reason": ver.reason
            })

            if step.intent == "verify":
                final_verify_pass = ver.passed

            decision = call_role("supervisor", supervisor.recovery_decision, ver, args.max_replans - replan_count)
            if decision == "fail":
                finish_reason = f"Recovery budget exhausted. {ver.reason}" if ver.need_replan else ver.reason
                break
            if decision == "replan":
                reason = ver.reason
                replan_count += 1
                log_records.append(LogRecord(event="replan", payload={"attempt": replan_count, "reason": reason}).model_dump())
                log_records.append(LogRecord(event="agent_message", payload={
                    "from": "supervisor", "to": "planner", "kind": "replan_request", "reason": reason,
                }).model_dump())
                active_plan = call_role("planner", plan_from_observation, args.goal, executor.obs, history)
                for index, pending_step in enumerate(active_plan.steps, start=action_count + 1):
                    pending_step.id = index
                # A new plan replaces all steps after the failed action.
                plan_record["payload"]["steps"][action_count:] = [
                    pending_step.model_dump() for pending_step in active_plan.steps
                ]
                planner.last_plan_source = "rule_replan"
                if interactive:
                    fallback_used = True
                    plan_record["payload"]["source"] = "llm_with_rule_fallback" if llm_actions else "rule_replan"
                should_replan = True
                break

        if finish_reason is not None:
            break
        if should_replan:
            continue
        if interactive and not fallback_used and final_verify_pass is None:
            continue
        active_plan = None

    if finish_reason is None and final_verify_pass is not True:
        finish_reason = "Run finished without successful goal verification"
    status = "failed" if finish_reason is not None else "passed"
    log_records.append(LogRecord(event="finish", payload={"status": status, "reason": finish_reason}).model_dump())

    # 6) Supervisor report (+ step table)
    report = call_role("supervisor", supervisor.summarize, args.goal, log_records)
    if args.env == "android":
        report = "Run environment — AndroidWorld emulator\n\n" + report
    header = "| Step | Intent | Target | Action OK | Verify | Reason |\n|---:|---|---|:---:|:---:|---|\n"
    rows = "\n".join([
        f"| {s['id']} | {s['intent']} | {s['target']} | {'✅' if s['action_ok'] else '❌'} | {'✅' if s['verify_passed'] else '❌'} | {s['verify_reason']} |"
        for s in steps_summary
    ])
    report_full = report + "\n\n### Step Summary\n\n" + header + rows + f"\n\n**Final status:** {status}\n"

    with open(os.path.join(args.logs_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report_full)

    for client in llm_clients:
        for call in client.calls:
            log_records.append(LogRecord(event="llm", payload=call).model_dump())

    # 7) Persist the complete trace, including LLM telemetry.
    log_records.append(LogRecord(event="run_metrics", payload={
        "end_to_end_latency_ms": round((time.perf_counter() - run_started_at) * 1000, 2),
        "role_processing_ms": {role: round(seconds * 1000, 2) for role, seconds in role_seconds.items()},
        "role_call_counts": role_calls,
        "replan_count": replan_count,
        "agent_message_count": sum(record["event"] == "agent_message" for record in log_records),
    }).model_dump())
    run_path = os.path.join(args.logs_dir, "qa_run.json")
    save_json(run_path, log_records)

    print(f"Run status: {status}")
    print(f"Logs: {run_path}")
    print(f"Report: {os.path.join(args.logs_dir, 'report.md')}")
    return 0 if status == "passed" else 1

if __name__ == "__main__":
    raise SystemExit(main())
