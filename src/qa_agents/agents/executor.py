from __future__ import annotations
from .base import Agent
from ..utils.schemas import Action, ActionResult, Observation

class Executor(Agent):
    def __init__(self, env) -> None:
        self.env = env
        self.obs: Observation = self.env.reset()

    def execute(self, intent: str, target: str|None, params: dict) -> ActionResult:
        # Verification is evaluated by Verifier against the most recent
        # observation; it does not require a second device action.
        if intent == "verify":
            return ActionResult(ok=True, observation=self.obs, message="Ready for verification")

        mapping = {
            "open_app":      lambda: Action(name="open_settings"),
            "open_settings": lambda: Action(name="open_settings"),
            "tap":           lambda: Action(name="tap",   target=target),
            "type":          lambda: Action(name="type",  target=target, params=params),
            "toggle":        lambda: Action(name="toggle", target=target, params=params),
            "wait":          lambda: Action(name="wait",  params=params),
        }
        make = mapping.get(intent)
        if not make:
            return ActionResult(ok=False, observation=self.obs, message=f"Unknown intent {intent}")

        action = make()
        ok, obs, msg = self.env.step(action)
        self.obs = obs
        return ActionResult(ok=ok, observation=obs, message=msg)
