from qa_agents.agents.planner import Planner


def test_invalid_llm_wifi_plan_falls_back_to_mock_safe_rule_plan(monkeypatch, tmp_path):
    planner = Planner(use_llm=True, cache_path=str(tmp_path / "plans.json"))
    assert planner.llm is not None
    monkeypatch.setattr(planner.llm, "plan", lambda _: [
        {"id": 1, "description": "Open Settings", "intent": "open_settings", "target": None, "params": {}},
        {"id": 2, "description": "Tap Wi-Fi", "intent": "tap", "target": "Wi-Fi", "params": {}},
        {"id": 3, "description": "Verify Wi-Fi", "intent": "verify", "target": "Wi-Fi", "params": {}},
    ])

    plan = planner.plan("Toggle Wi-Fi off and on")

    assert [step.intent for step in plan.steps] == ["open_settings", "tap", "toggle", "wait", "toggle", "verify"]


def test_invalid_cached_wifi_plan_is_discarded(tmp_path):
    planner = Planner(use_llm=False, cache_path=str(tmp_path / "plans.json"))
    key = planner.cache.key_from("plan", "Toggle Wi-Fi off and on")
    planner.cache.set(key, {"steps": [
        {"id": 1, "description": "Tap Wi-Fi", "intent": "tap", "target": "Wi-Fi", "params": {}},
    ]})

    plan = planner.plan("Toggle Wi-Fi off and on")

    assert plan.steps[0].intent == "open_settings"


def test_wifi_mock_compatibility_requires_off_then_on_sequence():
    valid = [
        {"id": 1, "intent": "open_settings", "target": None, "params": {}},
        {"id": 2, "intent": "tap", "target": "Network & Internet", "params": {}},
        {"id": 3, "intent": "toggle", "target": "Wi-Fi", "params": {"state": "off"}},
        {"id": 4, "intent": "wait", "target": None, "params": {"seconds": 1}},
        {"id": 5, "intent": "toggle", "target": "Wi-Fi", "params": {"state": "on"}},
        {"id": 6, "intent": "verify", "target": "Wi-Fi", "params": {"expected_state": "on"}},
    ]

    assert Planner._is_mock_compatible("Toggle Wi-Fi off and on", valid)
    valid[4]["params"]["state"] = "off"
    assert not Planner._is_mock_compatible("Toggle Wi-Fi off and on", valid)


def test_known_mock_tasks_reject_incompatible_llm_plans():
    cases = [
        ("Turn Bluetooth on", "Bluetooth"),
        ("Enable dark mode", "Dark mode"),
        ("Allow camera permission", "Camera permission"),
    ]
    for goal, target in cases:
        plan = Planner(use_llm=False)._rule_plan(goal)
        steps = [step.model_dump() for step in plan.steps]
        assert Planner._is_mock_compatible(goal, steps)
        steps[-1]["target"] = "Wrong target"
        assert not Planner._is_mock_compatible(goal, steps), target
