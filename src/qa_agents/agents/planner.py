from __future__ import annotations
from typing import List, Optional
import os

from .base import Agent
from ..utils.schemas import Observation, Plan, PlanStep
from ..utils.cache import JsonCache
from ..llm.openai_client import LLMClient
from ..mock_contracts import is_mock_compatible, mock_contract_for_goal, mock_goal, wifi_off_transition_seen

class Planner(Agent):
    """Planner with LLM-first strategy and rule-based fallback."""

    def __init__(self, use_llm: bool = False, model: Optional[str] = None, cache_path: Optional[str] = None) -> None:
        self.use_llm = use_llm
        self.llm = LLMClient(model=model) if use_llm else None
        self.cache = JsonCache(cache_path or os.path.join(".", "artifacts", "latest", "plan_cache.json"))
        self.last_plan_source = "rule"
        self.last_action_rejection = ""

    def _rule_plan(self, goal: str) -> Plan:
        contract = mock_contract_for_goal(goal) or [
            {"intent": "open_settings", "target": None, "params": {}},
            {"intent": "verify", "target": None, "params": {}},
        ]
        steps = []
        for index, step in enumerate(contract, start=1):
            intent = step["intent"]
            target = step["target"]
            params = {"seconds": 1} if intent == "wait" else step["params"]
            steps.append(PlanStep(id=index, description=f"{intent} {target or ''}".strip(),
                                  intent=intent, target=target, params=params))
        plan = Plan(goal=goal, steps=steps)
        self.log("Created rule-based plan with", len(steps), "steps")
        return plan

    def recovery_plan(
        self, goal: str, observation: Observation, history: list,
        route: tuple[tuple[str, str], ...] | None = None,
    ) -> Plan:
        """Plan the remaining actions from the observed state and navigation route."""
        task = mock_goal(goal)
        if task is None:
            return self._rule_plan(goal)

        steps: List[PlanStep] = []

        def add(intent: str, action_target=None, **params):
            steps.append(PlanStep(id=len(steps) + 1, description=f"{intent} {action_target or ''}".strip(),
                                  intent=intent, target=action_target, params=params))

        route = task.route if route is None else route
        route_screens = ["settings", *(screen for screen, _ in route)]
        if observation.screen in route_screens:
            remaining = route[route_screens.index(observation.screen):]
        else:
            add("open_settings")
            remaining = route
        for _, label in remaining:
            add("tap", label)

        current = observation.info.get(task.setting)
        desired = "on" if task.enabled else "off"
        if task.wifi_cycle and not wifi_off_transition_seen(history):
            if current is False:
                add("toggle", task.target, state="on")
            add("toggle", task.target, state="off")
            current = False
        if current is not task.enabled:
            add("toggle", task.target, state=desired)
        add("verify", task.target, expected_state=desired)
        return Plan(goal=goal, steps=steps)

    @staticmethod
    def _is_mock_compatible(goal: str, steps: List[dict]) -> bool:
        """Reject LLM plans the bundled mock environment cannot execute."""
        return is_mock_compatible(goal, steps)

    def plan_next(self, goal: str, observation: Observation, history: list) -> Optional[PlanStep]:
        """Get one model action and check it against the current visible screen."""
        if not self.llm:
            self.last_action_rejection = "LLM planner unavailable"
            return None
        candidate = self.llm.plan_next(goal, observation.model_dump(), history)
        if not isinstance(candidate, dict):
            last_call = self.llm.calls[-1] if self.llm.calls else {}
            error_type = last_call.get("error_type")
            self.last_action_rejection = (f"Provider request failed: {error_type}" if error_type
                                          else "Provider did not return a valid action")
            return None
        try:
            step = PlanStep(**candidate)
        except (ValueError, TypeError):
            self.last_action_rejection = "Provider returned an invalid action schema"
            return None
        normalize = lambda value: str(value or "").lower().replace("‑", "-").replace("-", "").strip()
        target = normalize(step.target)
        ui = observation.ui_tree
        visible_items = {normalize(item): item for item in ui.get("list", []) + ui.get("buttons", [])}
        visible_toggles = {normalize(item): item for item in ui.get("toggles", {})}
        if step.intent == "tap" and target in visible_items:
            step.target = visible_items[target]
            return step
        if step.intent == "toggle" and target in visible_toggles and step.params.get("state") in ("on", "off"):
            step.target = visible_toggles[target]
            return step
        if step.intent == "verify" and target in visible_toggles and step.params.get("expected_state") in ("on", "off"):
            step.target = visible_toggles[target]
            return step
        if step.intent == "open_settings" and target in ("", "settings") and not step.params:
            step.target = None
            return step
        if step.intent == "wait" and not step.target and step.params.get("seconds", 1) in (0, 1):
            return step
        self.last_action_rejection = f"Action unavailable or invalid on current screen: {step.intent} {step.target or ''}".strip()
        return None

    def plan(self, goal: str) -> Plan:
        # Try cache first
        key = self.cache.key_from("plan", goal)
        cached = self.cache.get(key)
        if cached:
            steps = [PlanStep(**s) for s in cached.get("steps", [])]
            serialized_steps = [step.model_dump() for step in steps]
            if self._is_mock_compatible(goal, serialized_steps):
                plan = Plan(goal=goal, steps=steps)
                self.last_plan_source = "cache"
                self.log("Loaded plan from cache with", len(steps), "steps")
                return plan
            self.log("Discarded a cached plan incompatible with the mock environment.")

        # Try LLM
        if self.use_llm and self.llm:
            steps = self.llm.plan(goal)
            if steps:
                steps_models = [PlanStep(**s) for s in steps]
                if self._is_mock_compatible(goal, [s.model_dump() for s in steps_models]):
                    plan = Plan(goal=goal, steps=steps_models)
                    self.last_plan_source = "llm"
                    self.cache.set(key, {"goal": goal, "steps": [s.model_dump() for s in steps_models]})
                    self.log("Created LLM plan with", len(steps_models), "steps")
                    return plan
                self.log("Rejected an LLM plan incompatible with the mock environment; using the rule fallback.")
                self.last_plan_source = "rule_fallback_invalid_llm_plan"

        # Fallback
        plan = self._rule_plan(goal)
        if self.use_llm and self.last_plan_source == "rule":
            self.last_plan_source = "rule_fallback_llm_unavailable"
        self.cache.set(key, {"goal": goal, "steps": [s.model_dump() for s in plan.steps]})
        return plan
