"""Exercise one interrupted Bluetooth-off action on an already running emulator.

Run with the AndroidWorld interpreter and the usual android-qa CLI arguments.
The emulator must start with Bluetooth on, so recovery proves a real transition.
"""

from qa_agents.envs.android_world_env import AndroidWorldEnv
from qa_agents.run_test import main


def run():
    original_step = AndroidWorldEnv.step
    interrupted = False

    def interrupt_once(env, action):
        nonlocal interrupted
        if action.name == "toggle" and action.target == "Bluetooth" and not interrupted:
            observation = env._observe()
            if observation.info["bluetooth_on"] is not True or action.params.get("state") != "off":
                raise RuntimeError("Recovery harness requires Bluetooth on and an off goal")
            interrupted = True
            return False, observation, "Test harness interrupted Bluetooth toggle before delivery"
        return original_step(env, action)

    AndroidWorldEnv.step = interrupt_once
    try:
        status = main()
        if not interrupted:
            raise RuntimeError("Recovery harness did not interrupt an action")
        return status
    finally:
        AndroidWorldEnv.step = original_step


if __name__ == "__main__":
    raise SystemExit(run())
