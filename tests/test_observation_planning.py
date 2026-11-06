import json
from types import SimpleNamespace

import pytest

from qa_agents.agents.planner import Planner
from qa_agents.llm.openai_client import LLMClient
from qa_agents.utils.schemas import Observation


def observation(screen, ui_tree):
    return Observation(screen=screen, ui_tree=ui_tree, info={})


def step(intent, target=None, **params):
    return {"id": 1, "description": intent, "intent": intent, "target": target, "params": params}


def test_next_action_changes_with_the_visible_screen(monkeypatch, tmp_path):
    planner = Planner(use_llm=True, cache_path=str(tmp_path / "plans.json"))
    seen = []

    def respond(goal, obs, history):
        seen.append(obs["screen"])
        if obs["screen"] == "settings":
            return step("tap", "Network & Internet")
        return step("toggle", "Bluetooth", state="on")

    monkeypatch.setattr(planner.llm, "plan_next", respond)
    first = planner.plan_next("Turn Bluetooth on", observation("settings", {"list": ["Network & Internet", "Display"]}), [])
    second = planner.plan_next("Turn Bluetooth on", observation("network", {"toggles": {"Bluetooth": False}}), [])

    assert (first.intent, first.target) == ("tap", "Network & Internet")
    assert (second.intent, second.target) == ("toggle", "Bluetooth")
    assert seen == ["settings", "network"]


def test_open_settings_accepts_visible_app_label(monkeypatch, tmp_path):
    planner = Planner(use_llm=True, cache_path=str(tmp_path / "plans.json"))
    monkeypatch.setattr(planner.llm, "plan_next", lambda *_: step("open_settings", "Settings"))
    selected = planner.plan_next("Turn Bluetooth on", observation("home", {"buttons": ["Settings"]}), [])
    assert selected is not None
    assert selected.intent == "open_settings"
    assert selected.target is None


@pytest.mark.parametrize("candidate", [
    step("tap", "Camera"),
    step("toggle", "Bluetooth", state="on"),
    step("delete_everything"),
    {"intent": "tap", "target": ["Network & Internet"], "params": {}},
])
def test_model_cannot_select_an_unavailable_action(monkeypatch, tmp_path, candidate):
    planner = Planner(use_llm=True, cache_path=str(tmp_path / "plans.json"))
    monkeypatch.setattr(planner.llm, "plan_next", lambda *_: candidate)
    obs = observation("settings", {"list": ["Network & Internet", "Display", "Apps"]})

    result = planner.plan_next("Turn Bluetooth on", obs, [])

    # A fallback is permitted, but it must never execute the rejected candidate.
    assert result is None or (result.intent, result.target) != (candidate.get("intent"), candidate.get("target"))


def test_next_action_prompt_includes_observation_without_prescribed_sequence(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    client = LLMClient()
    requests = []

    def create(**request):
        requests.append(request)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "steps": [step("tap", "Network & Internet")]
        })))])

    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    obs = observation("settings", {"list": ["Network & Internet", "Display"]})
    obs.info["bluetooth_on"] = False
    history = [{
        "intent": "open_settings", "target": None, "params": {}, "ok": True,
        "previous_info": {"bluetooth_on": False}, "observation": obs.model_dump(),
    }]
    result = client.plan_next("Turn Bluetooth on", obs.model_dump(), history)

    assert result is not None
    prompt = "\n".join(message["content"] for message in requests[0]["messages"])
    assert "settings" in prompt.lower()
    assert "Network & Internet" in prompt
    assert "bluetooth_on" not in prompt
    assert "exact device sequence" not in prompt.lower()
    assert "mock contract" not in prompt.lower()
    assert "simulated" not in prompt.lower()


def test_next_action_resolves_normalized_target_to_observed_label(monkeypatch, tmp_path):
    planner = Planner(use_llm=True, cache_path=str(tmp_path / "plans.json"))
    monkeypatch.setattr(planner.llm, "plan_next", lambda *_: step("tap", "network & internet"))
    selected = planner.plan_next("Toggle Wi-Fi off and on", observation("settings", {"list": ["Network & internet"]}), [])
    assert selected.target == "Network & internet"


def test_malformed_next_action_response_does_not_escape_to_the_caller(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    client = LLMClient()
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"steps": ["bad"]}'))])
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response)))

    assert client.plan_next("Turn Bluetooth on", {"screen": "settings", "ui_tree": {}, "info": {}}, []) is None
