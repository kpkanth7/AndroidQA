"""Runner contract tests with a device stub; these are not emulator evidence."""

import json
import sys

import pytest

from qa_agents import run_test
from qa_agents.agents.planner import Planner
from qa_agents.envs import android_world_env
from qa_agents.utils.schemas import Observation, PlanStep


@pytest.fixture
def device(monkeypatch):
    route = (("connected_devices", "Connected devices"), ("bluetooth", "Bluetooth"))
    monkeypatch.setattr(android_world_env, "BLUETOOTH_ROUTE", route, raising=False)

    class Device:
        closed = False

        def __init__(self, **kwargs):
            self.screen = "other"
            self.enabled = False

        def observe(self):
            labels = {"settings": ["Connected devices"], "connected_devices": ["Bluetooth"]}
            ui = {"toggles": {"Bluetooth": self.enabled}} if self.screen == "bluetooth" else {"list": labels.get(self.screen, [])}
            return Observation(screen=self.screen, ui_tree=ui, info={"bluetooth_on": self.enabled})

        def reset(self):
            return self.observe()

        def step(self, action):
            if action.name == "open_settings":
                self.screen = "settings"
            elif action.name == "tap":
                assert action.target in self.observe().ui_tree["list"]
                self.screen = next(screen for screen, label in route if label == action.target)
            elif action.name == "toggle":
                assert self.screen == "bluetooth"
                self.enabled = action.params["state"] == "on"
            elif action.name != "wait":
                raise AssertionError(action)
            return True, self.observe(), "Device stub action"

        def close(self):
            Device.closed = True

    monkeypatch.setattr(android_world_env, "AndroidWorldEnv", Device)
    return Device


def step(intent, target=None, **params):
    return PlanStep(id=1, description=intent, intent=intent, target=target, params=params)


def run(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "android-qa", "--env", "android", "--task", "settings_bluetooth",
        "--goal", "Turn Bluetooth on", "--llm_planner", "--logs_dir", str(tmp_path),
    ])
    code = run_test.main()
    return code, json.loads((tmp_path / "qa_run.json").read_text())


def test_android_llm_uses_observations_and_independent_verification(monkeypatch, tmp_path, device):
    seen = []

    def choose(self, goal, obs, history):
        seen.append(obs.screen)
        if obs.screen == "other":
            return step("open_settings")
        if obs.ui_tree.get("list"):
            return step("tap", obs.ui_tree["list"][0])
        return step("verify", "Bluetooth", expected_state="on") if obs.info["bluetooth_on"] else step("toggle", "Bluetooth", state="on")

    monkeypatch.setattr(Planner, "plan_next", choose)
    code, records = run(monkeypatch, tmp_path)
    assert code == 0
    assert seen == ["other", "settings", "connected_devices", "bluetooth", "bluetooth"]
    assert next(r["payload"] for r in records if r["event"] == "run_metadata")["environment"] == "android"
    assert next(r["payload"] for r in records if r["event"] == "plan")["source"] == "llm"
    assert device.closed


@pytest.mark.parametrize("early_verify", [False, True])
def test_android_fallback_uses_device_route_and_is_reported(monkeypatch, tmp_path, device, early_verify):
    monkeypatch.setattr(Planner, "plan_next", lambda *_: step("verify", "Bluetooth", expected_state="on") if early_verify else None)
    code, records = run(monkeypatch, tmp_path)
    assert code == 0
    actions = [r["payload"] for r in records if r["event"] == "action"]
    assert [a["target"] for a in actions if a["intent"] == "tap"] == ["Connected devices", "Bluetooth"]
    assert actions[-1]["observation"]["info"]["bluetooth_on"] is True
    assert next(r["payload"] for r in records if r["event"] == "plan")["source"] == (
        "llm_with_rule_fallback" if early_verify else "rule_fallback_invalid_or_unavailable"
    )
    assert any(r["event"] == ("replan" if early_verify else "plan_fallback") for r in records)


def test_android_llm_budget_exhaustion_fails_explicitly(monkeypatch, tmp_path, device):
    monkeypatch.setattr(Planner, "plan_next", lambda *_: step("wait", seconds=0))
    code, records = run(monkeypatch, tmp_path)
    assert code == 1
    assert sum(r["event"] == "action" for r in records) == run_test.MAX_LLM_STEPS
    finish = next(r["payload"] for r in records if r["event"] == "finish")
    assert "step budget exhausted" in finish["reason"]
    assert device.closed


@pytest.mark.parametrize(("task", "goal"), [
    ("settings_wifi", "Turn Bluetooth on"),
    ("settings_bluetooth", "Toggle Wi-Fi off and on"),
    ("settings_bluetooth", "Enable dark mode"),
    ("unknown", "Turn Bluetooth off"),
])
def test_android_rejects_mismatched_task_and_goal(monkeypatch, task, goal):
    monkeypatch.setattr(sys, "argv", ["android-qa", "--env", "android", "--task", task, "--goal", goal])
    with pytest.raises(SystemExit) as error:
        run_test.main()
    assert error.value.code == 2
