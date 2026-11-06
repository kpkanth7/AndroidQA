import json
import sys
from itertools import repeat

import pytest

from qa_agents.envs.mock_android import MockAndroidEnv
from qa_agents.llm.openai_client import LLMClient
from qa_agents.run_test import main


def model_action(intent, target=None, **params):
    return {"id": 1, "description": intent, "intent": intent, "target": target, "params": params}


def run_with_actions(monkeypatch, tmp_path, goal, actions, initial_state, max_replans=1):
    choices = iter(actions)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(LLMClient, "plan_next", lambda *_: next(choices))
    monkeypatch.setattr(sys, "argv", [
        "android-qa", "--env", "mock", "--llm_planner", "--goal", goal,
        "--mock-initial-state", json.dumps(initial_state), "--max-replans", str(max_replans),
        "--logs_dir", str(tmp_path / "run"),
    ])
    exit_code = main()
    records = json.loads((tmp_path / "run" / "qa_run.json").read_text())
    return exit_code, records


def events(records, name):
    return [record["payload"] for record in records if record["event"] == name]


@pytest.mark.parametrize(("start_screen", "failed_action", "expected_actions"), [
    ("home", "open_settings", ["open_settings", "open_settings", "tap", "toggle", "verify"]),
    ("display", "toggle", ["toggle", "toggle", "verify"]),
])
def test_recovers_from_navigation_or_setting_failure(
    monkeypatch, tmp_path, start_screen, failed_action, expected_actions,
):
    original_step = MockAndroidEnv.step
    failed = False

    def fail_once(env, action):
        nonlocal failed
        if not failed and action.name == failed_action:
            failed = True
            return False, env._observe(), "Temporary device error"
        return original_step(env, action)

    monkeypatch.setattr(MockAndroidEnv, "step", fail_once)
    first_action = (
        model_action("open_settings") if start_screen == "home"
        else model_action("toggle", "Dark mode", state="on")
    )
    exit_code, records = run_with_actions(
        monkeypatch, tmp_path, "Enable dark mode", [first_action], {"screen": start_screen},
    )

    actions = events(records, "action")
    assert exit_code == 0
    assert [action["intent"] for action in actions] == expected_actions
    assert actions[0]["ok"] is False
    assert actions[-1]["observation"]["info"]["dark_mode"] is True
    assert [event["attempt"] for event in events(records, "replan")] == [1]
    assert events(records, "finish")[0]["status"] == "passed"


def test_wifi_recovery_after_off_transition_retries_only_turning_on(monkeypatch, tmp_path):
    original_step = MockAndroidEnv.step
    failed = False

    def fail_first_wifi_on(env, action):
        nonlocal failed
        if action.name == "toggle" and action.params.get("state") == "on" and not failed:
            failed = True
            return False, env._observe(), "Interrupted while restoring Wi-Fi"
        return original_step(env, action)

    monkeypatch.setattr(MockAndroidEnv, "step", fail_first_wifi_on)
    exit_code, records = run_with_actions(
        monkeypatch, tmp_path, "Toggle Wi-Fi off and on", [
            model_action("toggle", "Wi-Fi", state="off"),
            model_action("toggle", "Wi-Fi", state="on"),
        ], {"screen": "network", "wifi_on": True},
    )

    actions = events(records, "action")
    assert exit_code == 0
    assert [(action["intent"], action["ok"], action["observation"]["info"]["wifi_on"])
            for action in actions] == [
        ("toggle", True, False), ("toggle", False, False),
        ("toggle", True, True), ("verify", True, True),
    ]
    assert [event["passed"] for event in events(records, "verify")] == [True, False, True, True]
    assert len(events(records, "replan")) == 1
    assert events(records, "plan")[0]["source"] == "llm_with_rule_fallback"


def test_initially_off_wifi_requires_an_actual_off_transition(monkeypatch, tmp_path):
    exit_code, records = run_with_actions(
        monkeypatch, tmp_path, "Toggle Wi-Fi off and on", [
            model_action("toggle", "Wi-Fi", state="off"),
            model_action("toggle", "Wi-Fi", state="on"),
            model_action("verify", "Wi-Fi", expected_state="on"),
        ], {"screen": "network", "wifi_on": False},
    )

    actions = events(records, "action")
    assert exit_code == 0
    assert [event["passed"] for event in events(records, "verify")] == [True, True, False, True, True, True]
    assert [action["observation"]["info"]["wifi_on"] for action in actions] == [
        False, True, True, False, True, True,
    ]
    assert [action["step_id"] for action in actions] == list(range(1, 7))
    assert [step["id"] for step in events(records, "plan")[0]["steps"]] == list(range(1, 7))
    assert events(records, "run_metrics")[0]["role_call_counts"] == {
        "planner": 4, "executor": 6, "verifier": 6, "supervisor": 7,
    }


def test_unsupported_goal_finishes_without_calling_model(monkeypatch, tmp_path):
    exit_code, records = run_with_actions(
        monkeypatch, tmp_path, "Turn Bluetooth on and then turn it off", [], {"screen": "network"},
    )

    assert exit_code == 1
    assert events(records, "finish")[0]["reason"] == "Goal is not supported by the mock environment"
    assert not events(records, "action")
    assert not events(records, "replan")
    assert "Finish reason: Goal is not supported" in (tmp_path / "run" / "report.md").read_text()


@pytest.mark.parametrize("action_succeeded", [False, True], ids=["execution-failed", "unchanged-state"])
def test_repeated_failures_exhaust_recovery_budget(monkeypatch, tmp_path, action_succeeded):
    def leave_setting_unchanged(env, action):
        assert action.name == "toggle"
        return action_succeeded, env._observe(), "Device setting stayed unchanged"

    monkeypatch.setattr(MockAndroidEnv, "step", leave_setting_unchanged)
    exit_code, records = run_with_actions(
        monkeypatch, tmp_path, "Enable dark mode", [model_action("toggle", "Dark mode", state="on")],
        {"screen": "display"}, max_replans=2,
    )

    assert exit_code == 1
    assert [action["ok"] for action in events(records, "action")] == [action_succeeded] * 3
    assert [event["passed"] for event in events(records, "verify")] == [False] * 3
    assert [event["attempt"] for event in events(records, "replan")] == [1, 2]
    assert events(records, "run_metrics")[0]["replan_count"] == 2
    assert events(records, "finish")[0]["reason"].startswith("Recovery budget exhausted.")


def test_model_waiting_without_progress_exhausts_step_budget(monkeypatch, tmp_path):
    exit_code, records = run_with_actions(
        monkeypatch, tmp_path, "Enable dark mode", repeat(model_action("wait")), {"screen": "display"},
    )

    assert exit_code == 1
    assert [action["intent"] for action in events(records, "action")] == ["wait"] * 15
    assert not events(records, "replan")
    assert events(records, "finish")[0]["status"] == "failed"
    assert "step budget exhausted" in events(records, "finish")[0]["reason"]
