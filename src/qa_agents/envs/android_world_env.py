"""AndroidWorld adapter for Pixel 6 Android 13 Wi-Fi and Bluetooth Settings."""

from __future__ import annotations

from pathlib import Path
import shutil
import re
import subprocess
import tempfile
import time

from ..utils.schemas import Action, Observation
from .types import EnvLike

NETWORK_LABEL = "Network & internet"
INTERNET_LABEL = "Internet"
WIFI_LABEL = "Wi-Fi"
WIFI_ROUTE = (("network_settings", NETWORK_LABEL), ("internet", INTERNET_LABEL))
CONNECTED_LABEL = "Connected devices"
PREFERENCES_LABEL = "Connection preferences"
BLUETOOTH_LABEL = "Bluetooth"
BLUETOOTH_ROUTE = (("connected_devices", CONNECTED_LABEL),
                   ("connection_preferences", PREFERENCES_LABEL), ("bluetooth", BLUETOOTH_LABEL))
_SETTINGS = "com.android.settings"
_UI_TIMEOUT_SECONDS = 15
_DEVICE = "emulator-5554"


class AndroidWorldEnv(EnvLike):
    def __init__(
        self,
        task_name: str = "settings_wifi",
        enable_render: bool = True,
        adb_path: str | None = None,
    ) -> None:
        if task_name not in {"settings_wifi", "settings_bluetooth"}:
            raise ValueError("AndroidWorld supports settings_wifi and settings_bluetooth")
        try:
            from android_env import loader
            from android_env.components import config_classes
            from android_world.env import android_world_controller, interface, json_action, representation_utils
        except ImportError as error:
            raise RuntimeError("Run with the Python environment where AndroidWorld is installed") from error

        if adb_path is None:
            adb_path = shutil.which("adb")
            if adb_path is None:
                sdk_paths = (Path.home() / "Library/Android/sdk", Path.home() / "Android/Sdk")
                adb_path = next((str(sdk / "platform-tools/adb") for sdk in sdk_paths
                                 if (sdk / "platform-tools/adb").is_file()), None)
        if adb_path is None or not Path(adb_path).expanduser().is_file():
            raise ValueError("ADB executable not found. Pass --adb-path with its filesystem path.")

        self.enable_render = enable_render
        self._adb_path = str(Path(adb_path).expanduser().resolve())
        self._actions = json_action
        self._parse_ui = representation_utils.xml_dump_to_ui_elements
        # The upstream accessibility forwarder enables Wi-Fi during startup
        # and on reconnect. UIAutomator can observe the device while Wi-Fi is off.
        with tempfile.TemporaryDirectory(prefix="androidqa-task-") as directory:
            task_path = Path(directory) / "settings.textproto"
            task_path.write_text(f'id: "{task_name}"\nmax_episode_sec: 7200\n', encoding="utf-8")
            config = config_classes.AndroidEnvConfig(
                task=config_classes.FilesystemTaskConfig(path=str(task_path)),
                simulator=config_classes.EmulatorConfig(
                    emulator_launcher=config_classes.EmulatorLauncherConfig(
                        emulator_console_port=5554, adb_port=5555, grpc_port=8554,
                    ),
                    adb_controller=config_classes.AdbControllerConfig(adb_path=self._adb_path),
                ),
            )
            device = loader.load(config)
        controller = android_world_controller.AndroidWorldController(
            device, a11y_method=android_world_controller.A11yMethod.NONE
        )
        self._env = interface.AsyncAndroidEnv(controller)

    def close(self) -> None:
        self._env.close()

    def reset(self) -> Observation:
        self._env.reset()
        self._env.hide_automation_ui()
        return self._observe()

    def _adb(self, *args: str) -> str:
        result = subprocess.run(
            [self._adb_path, "-s", _DEVICE, *args], capture_output=True, text=True, timeout=15,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "ADB command failed")
        return result.stdout.strip()

    def _observe(self) -> Observation:
        dump_path = "/data/local/tmp/androidqa-window.xml"
        dump_status = self._adb("shell", "uiautomator", "dump", dump_path)
        if "dumped to" not in dump_status:
            raise RuntimeError(f"Could not read Android's UI hierarchy: {dump_status}")
        nodes = [node for node in self._parse_ui(self._adb("exec-out", "cat", dump_path))
                 if node.package_name == _SETTINGS and node.bbox_pixels]
        toolbar = next(
            (node.content_description for node in nodes
             if node.resource_id == f"{_SETTINGS}:id/collapsing_toolbar"),
            None,
        )
        self._labels = {node.text: node for node in nodes
                        if node.text in (NETWORK_LABEL, INTERNET_LABEL, CONNECTED_LABEL,
                                         PREFERENCES_LABEL, BLUETOOTH_LABEL)}
        self._switch = None
        ui_tree = {}
        if toolbar == INTERNET_LABEL:
            screen = "internet"
            wifi_row = next((node for node in nodes if node.text == WIFI_LABEL), None)
            if wifi_row:
                switches = [
                    node for node in nodes if node.is_checkable and node.class_name == "android.widget.Switch"
                    and node.bbox_pixels.y_min <= wifi_row.bbox_pixels.y_max
                    and node.bbox_pixels.y_max >= wifi_row.bbox_pixels.y_min
                ]
                if len(switches) == 1:
                    self._switch = switches[0]
                    ui_tree["toggles"] = {WIFI_LABEL: self._switch.is_checked}
        elif toolbar == BLUETOOTH_LABEL:
            screen = "bluetooth"
            switches = [node for node in nodes if node.is_checkable
                        and node.resource_id == "android:id/switch_widget"]
            if len(switches) == 1:
                self._switch = switches[0]
                ui_tree["toggles"] = {BLUETOOTH_LABEL: self._switch.is_checked}
        elif toolbar in {CONNECTED_LABEL, PREFERENCES_LABEL}:
            screen, child = (("connected_devices", PREFERENCES_LABEL) if toolbar == CONNECTED_LABEL
                             else ("connection_preferences", BLUETOOTH_LABEL))
            ui_tree["list"] = [child] if child in self._labels else []
        elif toolbar == NETWORK_LABEL:
            screen = "network_settings"
            ui_tree["list"] = [label for label in (INTERNET_LABEL,) if label in self._labels]
        elif any(node.resource_id == f"{_SETTINGS}:id/homepage_title" for node in nodes):
            screen = "settings"
            ui_tree["list"] = [label for label in (NETWORK_LABEL, CONNECTED_LABEL) if label in self._labels]
        else:
            screen = "other"

        wifi_value = self._adb("shell", "settings", "get", "global", "wifi_on")
        if wifi_value not in {"0", "1", "2"}:
            raise RuntimeError(f"Android returned an unknown Wi-Fi state: {wifi_value!r}")
        bluetooth_dump = self._adb("shell", "dumpsys", "bluetooth_manager")
        bluetooth_state = re.search(r"^\s*state: (\w+)\s*$", bluetooth_dump, re.MULTILINE)
        if bluetooth_state is None or bluetooth_state[1] not in {
            "ON", "OFF", "TURNING_ON", "TURNING_OFF", "BLE_ON", "BLE_TURNING_ON", "BLE_TURNING_OFF",
        }:
            raise RuntimeError("Android returned an unknown Bluetooth manager state")
        state = bluetooth_state[1]
        # Transitional states are unknown, never evidence of a completed toggle.
        bluetooth_on = True if state == "ON" else False if state in {"OFF", "BLE_ON"} else None
        self._last_obs = Observation(screen=screen, ui_tree=ui_tree, info={
            "wifi_on": wifi_value != "0", "bluetooth_on": bluetooth_on,
            "bluetooth_manager_state": state,
        })
        return self._last_obs

    def _wait_for(self, condition, reason: str) -> Observation:
        deadline = time.monotonic() + _UI_TIMEOUT_SECONDS
        while True:
            observation = self._observe()
            if condition(observation):
                return observation
            if time.monotonic() >= deadline:
                raise RuntimeError(reason)
            time.sleep(0.5)

    def _click(self, element) -> None:
        x, y = element.bbox_pixels.center
        self._env.execute_action(self._actions.JSONAction(action_type=self._actions.CLICK, x=int(x), y=int(y)))

    def step(self, action: Action) -> tuple[bool, Observation, str]:
        try:
            if action.name == "open_settings":
                self._env.execute_action(self._actions.JSONAction(
                    action_type=self._actions.OPEN_APP, app_name="settings"
                ))
                observation = self._wait_for(
                    lambda obs: NETWORK_LABEL in obs.ui_tree.get("list", ()), "Settings did not open"
                )
                return True, observation, "Opened Android Settings"

            if action.name == "tap":
                observation = self._observe()
                if action.target not in observation.ui_tree.get("list", ()):
                    raise ValueError(f"{action.target!r} is unavailable on screen {observation.screen}")
                self._click(self._labels[action.target])
                destinations = {
                    NETWORK_LABEL: ("list", INTERNET_LABEL), INTERNET_LABEL: ("toggles", WIFI_LABEL),
                    CONNECTED_LABEL: ("list", PREFERENCES_LABEL),
                    PREFERENCES_LABEL: ("list", BLUETOOTH_LABEL),
                    BLUETOOTH_LABEL: ("toggles", BLUETOOTH_LABEL),
                }
                collection, label = destinations[action.target]
                ready = lambda obs: label in obs.ui_tree.get(collection, ())
                observation = self._wait_for(ready, f"Timed out opening {action.target}")
                return True, observation, f"Opened {action.target}"

            if action.name == "toggle":
                target = (action.target or "").casefold().replace("‑", "-")
                desired = action.params.get("state")
                if target not in {"wifi", "wi-fi", "bluetooth"} or desired not in {"on", "off"}:
                    raise ValueError("AndroidWorld supports Wi-Fi and Bluetooth toggles with state on or off")
                label, key = ((BLUETOOTH_LABEL, "bluetooth_on") if target == "bluetooth"
                              else (WIFI_LABEL, "wifi_on"))
                observation = self._observe()
                if self._switch is None or label not in observation.ui_tree.get("toggles", {}):
                    raise ValueError(f"The {label} switch is unavailable on this screen")
                enabled = desired == "on"
                if self._switch.is_checked is not observation.info[key]:
                    raise RuntimeError(f"The visible {label} switch disagrees with Android's reported state")
                if observation.info[key] is enabled:
                    return True, observation, f"{label} is already {desired}"
                self._click(self._switch)
                observation = self._wait_for(
                    lambda obs: obs.info[key] is enabled
                    and obs.ui_tree.get("toggles", {}).get(label) is enabled,
                    f"{label} did not turn {desired}",
                )
                return True, observation, f"{label} turned {desired}"

            if action.name == "wait" and action.params.get("seconds", 1) in (0, 1):
                time.sleep(action.params.get("seconds", 1))
                return True, self._observe(), "Waited for Android"

            raise ValueError(f"Unsupported AndroidWorld action: {action.name}")
        except Exception as error:
            return False, self._last_obs, str(error)

    def render(self):
        if not self.enable_render:
            raise RuntimeError("render() disabled")
        return self._env.get_state().pixels
