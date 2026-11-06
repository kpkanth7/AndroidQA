from __future__ import annotations
from .base import Agent
from typing import List, Dict, Any
from ..llm.openai_client import LLMClient
from ..utils.schemas import Verification

class Supervisor(Agent):
    def __init__(self, model: str|None = None, use_llm: bool = False) -> None:
        self.llm = LLMClient(model=model) if use_llm else None

    def recovery_decision(self, verification: Verification, replans_remaining: int) -> str:
        if verification.passed:
            return "continue"
        if verification.need_replan and replans_remaining > 0:
            return "replan"
        return "fail"

    def summarize(self, goal: str, log_records: List[Dict[str, Any]]) -> str:
        # Try LLM; if not available, fallback to a deterministic summary
        prompt = (
            "You are the QA supervisor. Summarize the run in 5 bullet points: "
            "goal, plan length, bugs detected, recovery actions, final status."
        )
        if self.llm:
            return self.llm.summarize(prompt, log_records, fallback=True)
        return LLMClient.fallback_summary(log_records)
