"""One metered Groq decision on a reconstructed, isolated rules-run state."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from .agents.executor import Executor
from .agents.planner import Planner
from .agents.verifier import Verifier
from .envs.mock_android import MockAndroidEnv
from .llm.openai_client import LLMClient
from .mock_contracts import mock_contract_for_goal

DAILY_LIMIT = 24
MODEL = "openai/gpt-oss-20b"


def configured() -> bool:
    return os.getenv("LLM_PROVIDER") == "groq" and all(
        os.getenv(key) for key in ("GROQ_API_KEY", "UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN")
    )


def reserve_attempt() -> bool:
    """An atomic increment reserves one attempt across processes and restarts."""
    key = "androidqa:decision:" + datetime.now(timezone.utc).date().isoformat()
    commands = [["INCR", key], ["EXPIRE", key, 172800]]
    request = Request(os.environ["UPSTASH_REDIS_REST_URL"].rstrip("/") + "/multi-exec",
                      data=json.dumps(commands).encode(), headers={
                          "Authorization": "Bearer " + os.environ["UPSTASH_REDIS_REST_TOKEN"],
                          "Content-Type": "application/json",
                      })
    with urlopen(request, timeout=5) as response:
        increment, expiry = json.load(response)
    count = increment.get("result")
    if type(count) is not int or count < 1 or expiry.get("result") != 1:
        raise ValueError("Budget store unavailable")
    return count <= DAILY_LIMIT


def execute(executor, history, intent, target, params):
    before = executor.obs
    outcome = executor.execute(intent, target, params)
    history.append(dict(intent=intent, target=target, params=params, ok=outcome.ok,
                        observation=outcome.observation.model_dump(), previous_info=before.info,
                        previous_toggles=before.ui_tree.get("toggles", {})))
    return outcome


def decision(goal: str, inject_failure: bool, completed: int):
    executor = Executor(MockAndroidEnv("toggle" if inject_failure else None))
    verifier = Verifier(goal)
    history = []
    with tempfile.TemporaryDirectory() as tmp:
        planner = Planner(cache_path=str(Path(tmp) / "plan.json"))
        pending = list(mock_contract_for_goal(goal))
        for _ in range(completed):
            if not pending:
                return 400, {"error": "This playback position is outside the run."}
            step = pending.pop(0)
            result = execute(executor, history, step["intent"], step["target"], step["params"])
            if not result.ok:
                recovery = planner.recovery_plan(goal, executor.obs, history)
                pending = [s.model_dump() for s in recovery.steps]

        if not pending:
            return 400, {"error": "This test has finished. Choose an earlier step."}
        client = LLMClient(model=MODEL)
        if client.provider != "groq" or client.client is None:
            return 503, {"error": "The live planner is unavailable."}
        try:
            if not reserve_attempt():
                return 429, {"error": "Today's 24 live attempts have been used. More are available at midnight UTC."}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return 503, {"error": "The daily allowance could not be checked. Try again later."}
        planner.llm = client
        client.single_decision = True
        before = executor.obs
        action = planner.plan_next(goal, before, history)
        call = client.calls[-1]
        response = dict(model=MODEL, latency_ms=call["latency_ms"], before=before.model_dump())
        if action is None:
            return 200, {**response, "accepted": False, "reason": planner.last_action_rejection}
        outcome = execute(executor, history, action.intent, action.target, action.params)
        check = verifier.verify_result(action, outcome, history)
        return 200, {**response, "accepted": True, "action": action.model_dump(), "ok": outcome.ok,
                     "message": outcome.message, "after": executor.obs.model_dump(),
                     "check": check.model_dump(), "goal_check": verifier.verify_goal(executor.obs, history).model_dump()}
