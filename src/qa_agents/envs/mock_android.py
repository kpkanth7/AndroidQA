from __future__ import annotations
from typing import Dict, Any
from .types import EnvLike
from ..utils.schemas import Observation, Action

class MockAndroidEnv(EnvLike):
    """Deterministic Android Settings simulation for local QA scenarios."""
    DEFAULT_STATE = {
        "screen": "home",
        "wifi_on": True,
        "bluetooth_on": False,
        "dark_mode": False,
        "camera_permission": False,
    }
    SCREENS = ("home", "settings", "network", "display", "apps", "camera")

    def __init__(self, fail_once_action: str | None = None, *, initial_state: Dict[str, Any] | None = None) -> None:
        self.fail_once_action = fail_once_action
        self._failure_injected = False
        self._initial_state = self.DEFAULT_STATE.copy()
        if initial_state is not None:
            if not isinstance(initial_state, dict):
                raise ValueError("Mock initial state must be a JSON object")
            for key, value in initial_state.items():
                if key not in self.DEFAULT_STATE:
                    raise ValueError(f"Unknown mock state field {key!r}")
                if key == "screen":
                    if value not in self.SCREENS:
                        raise ValueError(f"Mock screen must be one of {', '.join(self.SCREENS)}")
                elif not isinstance(value, bool):
                    raise ValueError(f"Mock state field {key!r} must be true or false")
                self._initial_state[key] = value
        self.reset()

    def reset(self) -> Observation:
        self.state = self._initial_state.copy()
        return self._observe()

    def _observe(self) -> Observation:
        screen = self.state["screen"]
        ui = {}
        if screen == "home":
            ui = {"buttons": ["Settings"]}
        elif screen == "settings":
            ui = {"list": ["Network & Internet", "Display", "Apps", "Battery"]}
        elif screen == "network":
            ui = {"toggles": {"Wi‑Fi": self.state["wifi_on"], "Bluetooth": self.state["bluetooth_on"]}}
        elif screen == "display":
            ui = {"toggles": {"Dark mode": self.state["dark_mode"]}}
        elif screen == "apps":
            ui = {"list": ["Camera"]}
        elif screen == "camera":
            ui = {"toggles": {"Camera permission": self.state["camera_permission"]}}
        return Observation(screen=screen, ui_tree=ui, info={key: value for key, value in self.state.items() if key != "screen"})

    @staticmethod
    def _normalized(value: str | None) -> str:
        return (value or "").lower().replace("‑", "-")

    @staticmethod
    def _desired_state(params: Dict[str, Any], current: bool) -> bool:
        state = str(params.get("state", "")).strip().lower()
        if state in {"on", "true", "1", "enabled", "allow"}:
            return True
        if state in {"off", "false", "0", "disabled", "deny"}:
            return False
        return not current

    def step(self, action: Action):
        if self.fail_once_action == action.name and not self._failure_injected:
            self._failure_injected = True
            return False, self._observe(), f"Injected transient failure for action={action.name}"

        ok = True
        msg = ""
        if action.name == "open_settings":
            self.state["screen"] = "settings"
            msg = "Opened Settings"
        elif action.name == "tap":
            target = self._normalized(action.target)
            if self.state["screen"] == "home" and target == "settings":
                self.state["screen"] = "settings"
                msg = "Tapped Settings"
            elif self.state["screen"] == "settings" and target in ["network & internet","network","internet"]:
                self.state["screen"] = "network"
                msg = "Opened Network & Internet"
            elif self.state["screen"] == "settings" and target == "display":
                self.state["screen"] = "display"
                msg = "Opened Display"
            elif self.state["screen"] == "settings" and target == "apps":
                self.state["screen"] = "apps"
                msg = "Opened Apps"
            elif self.state["screen"] == "apps" and target == "camera":
                self.state["screen"] = "camera"
                msg = "Opened Camera settings"
            else:
                ok = False
                msg = f"Tap failed on screen={self.state['screen']} target={action.target}"
        elif action.name == "toggle":
            target = self._normalized(action.target)
            toggle_targets = {
                ("network", "wifi"): ("wifi_on", "Wi‑Fi"),
                ("network", "wi-fi"): ("wifi_on", "Wi‑Fi"),
                ("network", "bluetooth"): ("bluetooth_on", "Bluetooth"),
                ("display", "dark mode"): ("dark_mode", "Dark mode"),
                ("camera", "camera permission"): ("camera_permission", "Camera permission"),
            }
            setting = toggle_targets.get((self.state["screen"], target))
            if setting:
                state_key, label = setting
                self.state[state_key] = self._desired_state(action.params, self.state[state_key])
                msg = f"{label} set to {self.state[state_key]}"
            else:
                ok, msg = False, "Toggle failed"
        elif action.name == "type":
            msg = "Type has no effect in mock"
        elif action.name == "wait":
            msg = "Waited"
        else:
            ok = False
            msg = f"Unknown action {action.name}"

        return ok, self._observe(), msg
