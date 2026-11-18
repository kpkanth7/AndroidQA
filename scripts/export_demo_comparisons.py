"""Export public, scenario-matched measurements from two recorded mock suites."""
import argparse
import json
from pathlib import Path

from qa_agents.benchmark import SCENARIOS
from qa_agents.envs.mock_android import MockAndroidEnv


def extract(path, scenario, planner, fault):
    events = json.loads(path.read_text())
    def first(name):
        return next(e["payload"] for e in events if e["event"] == name)
    meta, plan, metrics = (first(name) for name in ("run_metadata", "plan", "run_metrics"))
    assert plan["goal"] == SCENARIOS[scenario]
    assert meta["mock_initial_state"] == MockAndroidEnv.DEFAULT_STATE
    assert meta.get("environment", "mock") == "mock"
    if planner != "rules":
        assert meta["clients"][0]["model"] == "openai/" + planner
    else:
        assert not meta["llm_enabled"]
    actions = []
    calls = []
    notes = []
    for e in events:
        p = e["payload"]
        if e["event"] == "action":
            obs = p["observation"]
            actions.append(dict(intent=p["intent"], target=p["target"], params=p["params"],
                                ok=p["ok"], screen=obs["screen"], toggles=obs["ui_tree"].get("toggles", {}),
                                message=p["message"]))
        elif e["event"] == "verify":
            actions[-1]["check"] = p["reason"]
        elif e["event"] == "llm":
            calls.append({k:p.get(k) for k in ("operation", "latency_ms", "success", "error_type")})
        elif e["event"] in ("replan", "plan_rejection", "plan_fallback"):
            notes.append(dict(after_actions=len(actions), kind=e["event"], reason=p["reason"]))
    assert any("Injected transient failure" in a["message"] for a in actions) == fault
    return dict(planner=planner, repeat=int(path.parent.name), status=first("finish")["status"],
                source=plan["source"], ms=metrics["end_to_end_latency_ms"], handoffs=metrics["agent_message_count"],
                replans=metrics["replan_count"], calls=calls, actions=actions, notes=notes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("faults", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = {"initial_state": MockAndroidEnv.DEFAULT_STATE, "scenarios": {}}
    for scenario in SCENARIOS:
        groups = {}
        for name, root, repeats in (("normal", args.baseline, 3), ("fault", args.faults, 1)):
            groups[name] = [
                extract(root/planner/"runs"/scenario/str(i)/"qa_run.json", scenario, planner, name == "fault")
                for planner in ("rules", "gpt-oss-20b", "gpt-oss-120b") for i in range(1, repeats+1)
            ]
        data["scenarios"][scenario] = groups
    args.output.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"))+"\n")


if __name__ == "__main__":
    main()
