from types import SimpleNamespace
import subprocess

import pytest

from qa_agents.envs import android_world_env as adapter
from qa_agents.utils.schemas import Action, Observation


def node(**kwargs):
    return SimpleNamespace(**{
        "package_name": "com.android.settings", "bbox_pixels": SimpleNamespace(y_min=0, y_max=100),
        "text": None, "content_description": None, "resource_id": "", "is_checkable": False,
        "is_checked": False, "class_name": "android.widget.Switch", **kwargs,
    })


def bluetooth_device(state="OFF", checked=False):
    env = object.__new__(adapter.AndroidWorldEnv)
    nodes = [node(resource_id="com.android.settings:id/collapsing_toolbar", content_description="Bluetooth"),
             node(resource_id="android:id/switch_widget", is_checkable=True, is_checked=checked)]
    env._parse_ui = lambda _: nodes

    def adb(*args):
        if args[:3] == ("shell", "uiautomator", "dump"):
            return "UI hierarchy dumped to: /data/local/tmp/androidqa-window.xml"
        if args[:2] == ("exec-out", "cat"):
            return "xml"
        if args[:3] == ("shell", "settings", "get"):
            return "1"
        if args == ("shell", "dumpsys", "bluetooth_manager"):
            return f"Bluetooth Status\n  state: {state}\n"
        raise AssertionError(args)

    env._adb = adb
    env._click = lambda _: pytest.fail("An inconsistent or unrelated switch must not be clicked")
    return env


@pytest.mark.parametrize(("state", "expected"), [("ON", True), ("OFF", False), ("BLE_ON", False),
                                                    ("TURNING_ON", None), ("TURNING_OFF", None)])
def test_bluetooth_state_comes_from_manager_not_visible_switch(state, expected):
    env = bluetooth_device(state, checked=True)
    obs = env._observe()
    assert obs.info["bluetooth_on"] is expected
    assert obs.info["bluetooth_manager_state"] == state
    assert obs.ui_tree["toggles"]["Bluetooth"] is True


@pytest.mark.parametrize("state", ["UNKNOWN", "", "12"])
def test_unknown_bluetooth_state_is_not_accepted(state):
    with pytest.raises(RuntimeError, match="unknown Bluetooth manager state"):
        bluetooth_device(state)._observe()


@pytest.mark.parametrize(("state", "checked", "desired"), [("OFF", True, "off"), ("ON", False, "on"),
                                                            ("TURNING_OFF", False, "off")])
def test_bluetooth_mismatch_or_transition_cannot_pass_toggle(state, checked, desired):
    ok, obs, message = bluetooth_device(state, checked).step(Action(name="toggle", target="Bluetooth", params={"state": desired}))
    assert not ok
    assert "disagrees" in message


def test_wifi_action_cannot_click_bluetooth_switch():
    ok, _, message = bluetooth_device().step(Action(name="toggle", target="Wi-Fi", params={"state": "off"}))
    assert not ok
    assert "Wi-Fi switch is unavailable" in message


@pytest.mark.parametrize("error", [RuntimeError("blank UI dump"), subprocess.TimeoutExpired("adb", 15)])
def test_wait_retries_transient_observation_errors(monkeypatch, error):
    env = object.__new__(adapter.AndroidWorldEnv)
    ready = Observation(screen="network_settings", ui_tree={"list": ["Internet"]}, info={})
    attempts = iter([error, ready])

    def observe():
        value = next(attempts)
        if isinstance(value, Exception):
            raise value
        return value

    env._observe = observe
    monkeypatch.setattr(adapter.time, "sleep", lambda _: None)
    assert env._wait_for(lambda obs: obs.screen == "network_settings", "Navigation failed") is ready


def test_wait_still_fails_at_deadline_and_retains_observation_error(monkeypatch):
    env = object.__new__(adapter.AndroidWorldEnv)
    env._observe = lambda: (_ for _ in ()).throw(RuntimeError("blank UI dump"))
    clock = iter([0, 16])
    monkeypatch.setattr(adapter.time, "monotonic", lambda: next(clock))
    with pytest.raises(RuntimeError, match="Navigation failed; last observation error: blank UI dump"):
        env._wait_for(lambda _: False, "Navigation failed")


def test_failed_navigation_refreshes_screen_for_recovery(monkeypatch):
    env = object.__new__(adapter.AndroidWorldEnv)
    before = Observation(screen="settings", ui_tree={"list": [adapter.NETWORK_LABEL]}, info={})
    after = Observation(screen="network_settings", ui_tree={"list": [adapter.INTERNET_LABEL]}, info={})
    env._last_obs = before
    env._labels = {adapter.NETWORK_LABEL: object()}
    observations = iter([before, after])
    env._observe = lambda: next(observations)
    env._click = lambda _: None
    env._wait_for = lambda *_: (_ for _ in ()).throw(RuntimeError("Timed out reading page"))
    ok, obs, message = env.step(Action(name="tap", target=adapter.NETWORK_LABEL))
    assert not ok
    assert obs is after
    assert message == "Timed out reading page"


def test_failed_refresh_preserves_original_action_error():
    env = object.__new__(adapter.AndroidWorldEnv)
    env._last_obs = Observation(screen="other", ui_tree={}, info={})
    env._observe = lambda: (_ for _ in ()).throw(RuntimeError("Observation failed"))
    ok, obs, message = env.step(Action(name="type"))
    assert not ok
    assert obs is env._last_obs
    assert message == "Unsupported AndroidWorld action: type"
