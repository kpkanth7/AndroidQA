from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any

class PlanStep(BaseModel):
    id: int
    description: str
    intent: Literal["open_app","open_settings","tap","type","toggle","wait","verify"]
    target: Optional[str] = None
    params: Dict[str, Any] = {}

class Plan(BaseModel):
    goal: str
    steps: List[PlanStep]

class Action(BaseModel):
    name: Literal["open_settings","tap","type","toggle","wait"]
    target: Optional[str] = None
    params: Dict[str, Any] = {}

class Observation(BaseModel):
    screen: str
    ui_tree: Dict[str, Any]
    info: Dict[str, Any] = {}

class ActionResult(BaseModel):
    ok: bool
    observation: Observation
    message: str = ""
    bug: Optional[str] = None

class Verification(BaseModel):
    passed: bool
    reason: str
    need_replan: bool = False

class LogRecord(BaseModel):
    event: Literal["run_metadata","run_metrics","agent_message","llm","plan","plan_rejection","plan_fallback","action","verify","replan","finish"]
    payload: Dict[str, Any]
