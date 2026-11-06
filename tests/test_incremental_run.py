import json
import sys

import pytest

from qa_agents.agents.planner import Planner
from qa_agents.envs.android_world_env import WIFI_ROUTE
from qa_agents.llm.openai_client import LLMClient
from qa_agents.run_test import main
from qa_agents.utils.schemas import Observation, PlanStep


def make_step(intent, target=None, **params):
    return PlanStep(id=1, description=intent, intent=intent, target=target, params=params)


def run_with_planner(monkeypatch, tmp_path, choose):
    observations = []

    def fake_next(self, goal, observation, history):
        observations.append(observation.screen)
        return choose(observation, history)

    monkeypatch.setattr(Planner, "plan_next", fake_next)
    monkeypatch.setattr(sys, "argv", ["android-qa", "--env", "mock", "--goal", "Toggle Wi-Fi off and on",
                                      "--llm_planner", "--logs_dir", str(tmp_path)])
    result = main()
    records = json.loads((tmp_path / "qa_run.json").read_text())
    return result, records, observations


def test_incremental_planner_uses_each_new_observation(monkeypatch, tmp_path):
    def choose(obs, history):
        if obs.screen == "home":
            return make_step("open_settings")
        if obs.screen == "settings":
            return make_step("tap", "Network & Internet")
        if not any(item["intent"] == "toggle" for item in history):
            return make_step("toggle", "Wi-Fi", state="off")
        if obs.info["wifi_on"] is False:
            return make_step("toggle", "Wi-Fi", state="on")
        return make_step("verify", "Wi-Fi", expected_state="on")

    result, records, observations = run_with_planner(monkeypatch, tmp_path, choose)
    assert result == 0
    assert observations == ["home", "settings", "network", "network", "network"]
    assert next(r for r in records if r["event"] == "plan")["payload"]["source"] == "llm"
    assert not any(r["event"] == "plan_fallback" for r in records)


def test_early_verify_cannot_pass_without_wifi_off_transition(monkeypatch, tmp_path):
    def choose(obs, history):
        if obs.screen == "home":
            return make_step("open_settings")
        if obs.screen == "settings":
            return make_step("tap", "Network & Internet")
        return make_step("verify", "Wi-Fi", expected_state="on")

    result, records, _ = run_with_planner(monkeypatch, tmp_path, choose)
    assert result == 0  # deterministic recovery still completes the goal
    assert next(r for r in records if r["event"] == "plan")["payload"]["source"] == "llm_with_rule_fallback"
    assert any(r["event"] == "replan" and "off transition" in r["payload"]["reason"] for r in records)


def test_invalid_model_action_is_labeled_as_fallback(monkeypatch, tmp_path):
    result, records, _ = run_with_planner(monkeypatch, tmp_path, lambda obs, history: None)
    assert result == 0
    assert next(r for r in records if r["event"] == "plan")["payload"]["source"] == "rule_fallback_invalid_or_unavailable"
    assert any(r["event"] == "plan_fallback" for r in records)


def test_recovery_continues_from_current_screen_and_observed_wifi_off(tmp_path):
    planner = Planner(cache_path=str(tmp_path / "plans.json"))
    observation = Observation(screen="network", ui_tree={"toggles": {"Wi-Fi": False}}, info={"wifi_on": False})
    history = [{
        "intent": "toggle", "target": "Wi-Fi", "ok": True,
        "previous_info": {"wifi_on": True}, "observation": {"info": {"wifi_on": False}},
    }]

    plan = planner.recovery_plan("Toggle Wi-Fi off and on", observation, history)

    assert [(step.intent, step.params) for step in plan.steps] == [
        ("toggle", {"state": "on"}),
        ("verify", {"expected_state": "on"}),
    ]


def test_recovery_navigates_from_a_different_screen(tmp_path):
    planner = Planner(cache_path=str(tmp_path / "plans.json"))
    observation = Observation(screen="display", ui_tree={}, info={"camera_permission": False})
    plan = planner.recovery_plan("Allow camera permission", observation, [])
    assert [(step.intent, step.target) for step in plan.steps] == [
        ("open_settings", None), ("tap", "Apps"), ("tap", "Camera"),
        ("toggle", "Camera permission"), ("verify", "Camera permission"),
    ]


def test_android_wifi_plan_uses_live_route_and_cycles_wifi_when_initially_off(tmp_path):
    planner = Planner(cache_path=str(tmp_path / "plans.json"))
    observation = Observation(screen="network_settings", ui_tree={"list": ["Internet"]}, info={"wifi_on": False})

    plan = planner.recovery_plan("Toggle Wi-Fi off and on", observation, [], route=WIFI_ROUTE)

    assert [(step.intent, step.target, step.params) for step in plan.steps] == [
        ("tap", "Internet", {}),
        ("toggle", "Wi‑Fi", {"state": "on"}),
        ("toggle", "Wi‑Fi", {"state": "off"}),
        ("toggle", "Wi‑Fi", {"state": "on"}),
        ("verify", "Wi‑Fi", {"expected_state": "on"}),
    ]


@pytest.mark.parametrize(("goal", "initial_state", "state_key", "turns"), [
    ("Toggle Wi-Fi off and on", {"screen": "network"}, "wifi_on", [
        ("network", True, make_step("toggle", "Wi-Fi", state="off")),
        ("network", False, make_step("toggle", "Wi-Fi", state="on")),
        ("network", True, make_step("verify", "Wi-Fi", expected_state="on")),
    ]),
    ("Turn Bluetooth on", {"screen": "network", "wifi_on": False, "bluetooth_on": True}, "bluetooth_on", [
        ("network", True, make_step("verify", "Bluetooth", expected_state="on")),
    ]),
    ("Enable dark mode", {"screen": "display"}, "dark_mode", [
        ("display", False, make_step("toggle", "Dark mode", state="on")),
        ("display", True, make_step("verify", "Dark mode", expected_state="on")),
    ]),
    ("Allow camera permission", {"screen": "apps", "camera_permission": True}, "camera_permission", [
        ("apps", None, make_step("tap", "Camera")),
        ("camera", True, make_step("verify", "Camera permission", expected_state="on")),
    ]),
], ids=["wifi-in-network", "bluetooth-already-on", "dark-mode-in-display", "camera-already-allowed"])
def test_mock_start_state_reaches_observation_planner(monkeypatch, tmp_path, goal, initial_state, state_key, turns):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    seen = []

    def respond(self, requested_goal, observation, history):
        index = len(seen)
        screen, value, step = turns[index]
        assert requested_goal == goal
        assert len(history) == index
        assert observation["screen"] == screen
        if value is not None:
            assert observation["ui_tree"]["toggles"][step.target.replace("-", "‑")] is value
        seen.append(step.intent)
        return step.model_dump()

    monkeypatch.setattr(LLMClient, "plan_next", respond)
    monkeypatch.setattr(sys, "argv", [
        "android-qa", "--env", "mock", "--goal", goal, "--llm_planner",
        "--mock-initial-state", json.dumps(initial_state), "--logs_dir", str(tmp_path),
    ])

    assert main() == 0
    assert len(seen) == len(turns)
    records = json.loads((tmp_path / "qa_run.json").read_text())
    recorded_state = next(r["payload"]["mock_initial_state"] for r in records if r["event"] == "run_metadata")
    assert recorded_state == {
        "screen": "home", "wifi_on": True, "bluetooth_on": False, "dark_mode": False,
        "camera_permission": False, **initial_state,
    }
    assert next(r["payload"]["source"] for r in records if r["event"] == "plan") == "llm"
    final_action = [r["payload"] for r in records if r["event"] == "action"][-1]
    assert final_action["observation"]["info"][state_key] is True


@pytest.mark.parametrize(("raw_state", "error"), [
    ("null", "must be a JSON object"),
    ('{"screen":"battery"}', "Mock screen must be one of"),
    ('{"wifi_on":"off"}', "must be true or false"),
    ('{"typo":true}', "Unknown mock state field"),
])
def test_invalid_mock_start_state_is_a_cli_error(monkeypatch, capsys, raw_state, error):
    monkeypatch.setattr(sys, "argv", ["android-qa", "--mock-initial-state", raw_state])

    with pytest.raises(SystemExit) as caught:
        main()

    assert caught.value.code == 2
    assert error in capsys.readouterr().err
