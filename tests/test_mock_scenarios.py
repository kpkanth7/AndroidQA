import pytest

from qa_agents.agents.executor import Executor
from qa_agents.agents.planner import Planner
from qa_agents.agents.verifier import Verifier
from qa_agents.envs.mock_android import MockAndroidEnv


@pytest.mark.parametrize(
    ("goal", "expected_key"),
    [
        ("Toggle Wi-Fi off and on", "wifi_on"),
        ("Toggle WiFi off then back on", "wifi_on"),
        ("Turn Bluetooth on", "bluetooth_on"),
        ("Enable dark mode", "dark_mode"),
        ("Allow camera permission", "camera_permission"),
    ],
)
def test_deterministic_mock_scenario_reaches_expected_state(goal, expected_key, tmp_path):
    plan = Planner(cache_path=str(tmp_path / "plans.json")).plan(goal)
    executor = Executor(MockAndroidEnv())
    verifier = Verifier(goal)

    final_verification = None
    for step in plan.steps:
        result = executor.execute(step.intent, step.target, step.params)
        assert result.ok, result.message
        final_verification = verifier.verify_step(step.intent, step.target, step.params, result.observation)

    assert final_verification is not None
    assert final_verification.passed
    assert executor.obs.info[expected_key] is True
