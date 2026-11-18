import json

import pytest

from qa_agents import demo_decision as demo


@pytest.fixture
def model(monkeypatch):
    calls = []

    class Client:
        provider = "groq"
        client = True

        def __init__(self, model):
            self.calls = []

        def plan_next(self, goal, obs, history):
            assert self.single_decision
            calls.append((goal, obs, list(history)))
            self.calls.append({"latency_ms": 1.5})
            return dict(id=1, intent="toggle", target="Wi‑Fi", params={"state":"off"}, description="Disable Wi-Fi")

    monkeypatch.setattr(demo, "LLMClient", Client)
    return calls


def test_decision_from_failed_switch(model, monkeypatch):
    monkeypatch.setattr(demo, "reserve_attempt", lambda: True)
    status, data = demo.decision("Toggle Wi-Fi off and on", True, 3)
    assert status == 200 and data["accepted"]
    assert not model[0][2][-1]["ok"]
    assert data["before"]["info"]["wifi_on"] and not data["after"]["info"]["wifi_on"]
    assert data["check"]["passed"] and not data["goal_check"]["passed"]


def test_decision_stops_before_provider(model, monkeypatch):
    monkeypatch.setattr(demo, "reserve_attempt", lambda: False)
    assert demo.decision("Turn Bluetooth on", False, 0)[0] == 429
    def offline():
        raise OSError("unavailable")
    monkeypatch.setattr(demo, "reserve_attempt", offline)
    assert demo.decision("Turn Bluetooth on", False, 0)[0] == 503
    assert demo.decision("Turn Bluetooth on", False, 20)[0] == 400
    assert not model


def test_daily_limit_checks_atomic_redis_result(monkeypatch):
    from io import BytesIO
    counts = iter([24, 25])
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://example.invalid")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "test-token")
    def respond(request, timeout):
        commands = json.loads(request.data)
        assert request.full_url.endswith("/multi-exec") and timeout > 0
        assert commands[0][0] == "INCR" and commands[0][1] == commands[1][1]
        assert commands[1][0] == "EXPIRE" and commands[1][2] >= 86400
        return BytesIO(json.dumps([{"result": next(counts)}, {"result": 1}]).encode())
    monkeypatch.setattr(demo, "urlopen", respond)
    assert demo.reserve_attempt()
    assert not demo.reserve_attempt()


def test_single_action_menu_exposes_both_states_without_goal():
    from qa_agents.llm.openai_client import LLMClient
    choices = LLMClient._visible_choices({"toggles": {"Bluetooth": False}})
    assert choices["Switch Bluetooth off"]["params"] == {"state": "off"}
    assert choices["Verify Bluetooth on"]["params"] == {"expected_state": "on"}
    assert "Switch Wi-Fi on" not in choices
