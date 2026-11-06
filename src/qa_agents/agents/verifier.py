from __future__ import annotations
from .base import Agent
from ..mock_contracts import mock_goal, wifi_off_transition_seen
from ..utils.schemas import ActionResult, Observation, PlanStep, Verification

class Verifier(Agent):
    def __init__(self, goal: str) -> None:
        self.task = mock_goal(goal)

    def verify_goal(self, obs: Observation, history: list) -> Verification:
        """Independently check the supported goal, not the model's assertion."""
        if self.task is None:
            return Verification(passed=False, reason="Goal is not supported by the independent mock verifier")
        actual = obs.info.get(self.task.setting)
        if self.task.wifi_cycle:
            passed = wifi_off_transition_seen(history) and actual is True
            return Verification(passed=passed, reason="Wi-Fi was turned off and is now on" if passed else "Wi-Fi goal requires an observed off transition and final on state")
        return Verification(passed=actual is self.task.enabled, reason=f"Goal state {self.task.setting}={actual}")

    def verify_result(self, step: PlanStep, result: ActionResult, history: list) -> Verification:
        if not result.ok:
            return Verification(passed=False, reason=f"Action failed: {result.message}", need_replan=True)
        verification = self.verify_step(step.intent, step.target, step.params, result.observation)
        if step.intent == "verify":
            goal_verification = self.verify_goal(result.observation, history)
            if not goal_verification.passed:
                return Verification(passed=False, reason=goal_verification.reason, need_replan=True)
        return verification

    def verify_step(self, intent: str, target: str|None, params: dict, obs: Observation) -> Verification:
        normalized_target = (target or "").lower().replace("‑", "-")
        settings = {
            "wifi": ("wifi_on", "Wi‑Fi"),
            "wi-fi": ("wifi_on", "Wi‑Fi"),
            "bluetooth": ("bluetooth_on", "Bluetooth"),
            "dark mode": ("dark_mode", "Dark mode"),
            "camera permission": ("camera_permission", "Camera permission"),
        }
        setting = settings.get(normalized_target)
        if intent not in {"toggle", "verify"} or not setting:
            return Verification(passed=True, reason="No specific rule")

        state_key, label = setting
        expected_value = params.get("expected_state", params.get("state")) if isinstance(params, dict) else None
        expected = True if expected_value is None else str(expected_value).strip().lower() in {"on", "true", "1", "enabled", "allow"}
        actual = bool(obs.info.get(state_key))
        passed = actual == expected
        action = "Toggled" if intent == "toggle" else "Verified"
        return Verification(passed=passed, reason=f"{action} {label}: desired={expected} actual={actual}", need_replan=not passed)
