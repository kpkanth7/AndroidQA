"""Executable plan contracts for the bundled mock Android environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _normalize(text: str) -> str:
    return text.lower().replace("‑", "-").replace("–", "-").replace("—", "-")


@dataclass(frozen=True)
class MockGoal:
    setting: str
    target: str
    route: tuple[tuple[str, str], ...]
    enabled: bool
    wifi_cycle: bool = False


_NETWORK = (("network", "Network & Internet"),)
_DISPLAY = (("display", "Display"),)
_CAMERA = (("apps", "Apps"), ("camera", "Camera"))
_WIFI_CYCLE = MockGoal("wifi_on", "Wi‑Fi", _NETWORK, True, wifi_cycle=True)
_GOALS = {
    "togglewifioffandon": _WIFI_CYCLE,
    "togglewifioffthenbackon": _WIFI_CYCLE,
    "turnbluetoothon": MockGoal("bluetooth_on", "Bluetooth", _NETWORK, True),
    "turnbluetoothoff": MockGoal("bluetooth_on", "Bluetooth", _NETWORK, False),
    "enabledarkmode": MockGoal("dark_mode", "Dark mode", _DISPLAY, True),
    "disabledarkmode": MockGoal("dark_mode", "Dark mode", _DISPLAY, False),
    "allowcamerapermission": MockGoal("camera_permission", "Camera permission", _CAMERA, True),
    "denycamerapermission": MockGoal("camera_permission", "Camera permission", _CAMERA, False),
}


def mock_goal(goal: str) -> MockGoal | None:
    return _GOALS.get("".join(_normalize(goal).replace("-", "").split()))


def wifi_off_transition_seen(history: list[dict[str, Any]]) -> bool:
    return any(
        item.get("ok") and item.get("intent") == "toggle"
        and _normalize(item.get("target") or "").replace("-", "") == "wifi"
        and item.get("previous_info", {}).get("wifi_on") is True
        and item.get("observation", {}).get("info", {}).get("wifi_on") is False
        for item in history
    )


def mock_contract_for_goal(goal: str) -> list[dict[str, Any]] | None:
    """Return the legacy plan contract from the default mock starting state."""
    task = mock_goal(goal)
    if task is None:
        return None
    steps = [{"intent": "open_settings", "target": None, "params": {}}]
    steps.extend({"intent": "tap", "target": label, "params": {}} for _, label in task.route)
    if task.wifi_cycle:
        steps.extend([
            {"intent": "toggle", "target": task.target, "params": {"state": "off"}},
            {"intent": "wait", "target": None, "params": {}},
        ])
    state = "on" if task.enabled else "off"
    steps.extend([
        {"intent": "toggle", "target": task.target, "params": {"state": state}},
        {"intent": "verify", "target": task.target, "params": {"expected_state": state}},
    ])
    return steps


def is_mock_compatible(goal: str, steps: list[dict[str, Any]]) -> bool:
    """Check whether steps conform to the supported mock task contract."""
    contract = mock_contract_for_goal(goal)
    if contract is None:
        return True
    if len(steps) != len(contract):
        return False

    for step, expected in zip(steps, contract):
        if step.get("intent") != expected["intent"]:
            return False
        if _normalize(str(step.get("target") or "")) != _normalize(str(expected["target"] or "")):
            return False
        actual_params = step.get("params") or {}
        if any(str(actual_params.get(key, "")).lower() != str(value).lower()
               for key, value in expected["params"].items()):
            return False
    return True


def mock_contract_instruction(goal: str) -> str:
    """Return an LLM instruction that describes a supported mock plan."""
    contract = mock_contract_for_goal(goal)
    if contract is None:
        return "- End with a 'verify' step when applicable.\n"

    lines = []
    for step in contract:
        target = step["target"]
        params = step["params"]
        detail = f" target '{target}'" if target else ""
        if params:
            detail += " with " + ", ".join(f"{key} '{value}'" for key, value in params.items())
        lines.append(f"{step['intent']}{detail}")
    return (
        "- This goal runs in the bundled mock environment. Use exactly this sequence, "
        "with no extra steps: " + "; ".join(lines) + ".\n"
    )
