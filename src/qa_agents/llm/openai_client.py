import json
import os
import re
import time
from typing import Any, List, Dict, Optional
from dotenv import load_dotenv

from ..mock_contracts import mock_contract_instruction

load_dotenv(override=False)

OBSERVATION_PROMPT_VERSION = "visible-settings-v3"

def _strip_code_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)


class MalformedPlanError(ValueError):
    """The provider responded, but not with the promised plan schema."""


class LLMClient:
    """OpenAI-compatible LLM client with OpenAI and Groq provider support."""
    def __init__(self, model: Optional[str] = None) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        if self.provider not in {"openai", "groq"}:
            raise ValueError("LLM_PROVIDER must be either 'openai' or 'groq'.")

        defaults = {
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY", "").strip(),
                "model": "gpt-4o-mini",
                "base_url": os.getenv("OPENAI_BASE_URL", "").strip() or None,
            },
            "groq": {
                "api_key": os.getenv("GROQ_API_KEY", "").strip(),
                "model": "openai/gpt-oss-20b",
                "base_url": os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").strip(),
            },
        }
        config = defaults[self.provider]
        self.api_key = config["api_key"]
        self.model = model or os.getenv("LLM_MODEL", "").strip() or config["model"]
        self.base_url = config["base_url"]
        self.client = None
        self.calls: List[Dict[str, Any]] = []
        if self.api_key:
            try:
                from openai import OpenAI
                kwargs: Dict[str, Any] = {"api_key": self.api_key, "max_retries": 0, "timeout": 15.0}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self.client = OpenAI(**kwargs)
            except Exception:
                self.client = None

    def metadata(self) -> Dict[str, Any]:
        """Safe-to-log configuration; deliberately excludes credentials."""
        return {
            "provider": self.provider,
            "model": self.model,
            "configured": self.client is not None,
        }

    def _record_call(self, operation: str, latency_ms: float, response: Any = None, error: Exception | None = None) -> None:
        usage = getattr(response, "usage", None)
        self.calls.append({
            "operation": operation,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": round(latency_ms, 2),
            "success": error is None,
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            } if usage else None,
            "error_type": type(error).__name__ if error else None,
        })

    @staticmethod
    def _parse_plan_steps(content: str) -> List[Dict[str, Any]]:
        try:
            data = json.loads(_strip_code_fences(content))
        except json.JSONDecodeError as error:
            raise MalformedPlanError("The provider did not return JSON.") from error
        if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
            raise MalformedPlanError("The provider response must contain a list named 'steps'.")

        cleaned = []
        next_id = 1
        for step in data["steps"]:
            if not isinstance(step, dict):
                raise MalformedPlanError("Every plan step must be an object.")
            intent = str(step.get("intent", "")).strip()
            params = step.get("params", {}) or {}
            if not isinstance(params, dict):
                raise MalformedPlanError("Plan-step params must be an object.")
            if intent not in ["open_app", "open_settings", "tap", "type", "toggle", "wait", "verify"]:
                continue
            try:
                step_id = int(step.get("id", next_id))
            except (TypeError, ValueError, OverflowError) as error:
                raise MalformedPlanError("Plan-step id must be an integer.") from error
            cleaned.append({
                "id": step_id,
                "description": str(step.get("description", intent)).strip(),
                "intent": intent,
                "target": step.get("target", None),
                "params": params,
            })
            next_id = cleaned[-1]["id"] + 1
        if not cleaned:
            raise MalformedPlanError("The provider returned no valid plan steps.")
        return cleaned

    def plan(self, goal: str) -> Optional[List[Dict[str, Any]]]:
        """
        Ask the model to produce a JSON list of plan steps for the given goal.
        Returns a Python list of step dicts, or None if not available.
        """
        if not self.client:
            return None
        system = (
            "You are a senior mobile QA planner. "
            "Output ONLY strict JSON with a top-level key 'steps' (no prose). "
            "Each step must have: id (int), description (str), "
            "intent one of ['open_app','open_settings','tap','type','toggle','wait','verify'], "
            "target (str or null), params (object)."
        )
        user = (
            f"Goal: {goal}\n"
            "Constraints:\n"
            "- Prefer minimal steps.\n"
            "- Use 'open_settings' to open Settings app.\n"
            "- Use 'tap' for list items like 'Network & Internet'.\n"
            "- Use 'toggle' for switches like 'Wi‑Fi'.\n"
            + mock_contract_instruction(goal)
            + "Return JSON ONLY:\n"
            "{\n"
            "  \"steps\": [\n"
            "    {\"id\":1, \"description\":\"...\", \"intent\":\"open_settings\", \"target\":null, \"params\":{}},\n"
            "    {\"id\":2, \"description\":\"...\", \"intent\":\"tap\", \"target\":\"Network & Internet\", \"params\":{}},\n"
            "    ...\n"
            "  ]\n"
            "}\n"
        )
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        for attempt in range(2):
            started_at = time.perf_counter()
            latency_ms = None
            resp = None
            operation = "plan" if attempt == 0 else "plan_retry"
            try:
                resp = self.client.chat.completions.create(**request)
                latency_ms = (time.perf_counter() - started_at) * 1000
                content = resp.choices[0].message.content or ""
                steps = self._parse_plan_steps(content)
                self._record_call(operation, latency_ms, response=resp)
                return steps
            except MalformedPlanError as error:
                self._record_call(operation, latency_ms, response=resp, error=error)
                continue
            except Exception as error:
                elapsed = (time.perf_counter() - started_at) * 1000 if latency_ms is None else latency_ms
                self._record_call(operation, elapsed, response=resp, error=error)
                return None
        return None

    def plan_next(self, goal: str, observation: dict, history: list) -> Optional[Dict[str, Any]]:
        """Choose one action using the latest screen and completed action history."""
        if not self.client:
            return None
        system = (
            "You control Android Settings through the supplied device observations. Select exactly one next action "
            "to accomplish the user's goal using only the current observation and history. "
            "Output JSON with a steps array containing exactly one object with id, description, "
            "intent, target and params. Allowed intents: open_settings, tap, toggle, wait, verify. "
            "Tap only visible list items or buttons. Toggle only visible switches, with params.state 'on' or 'off'. "
            "Use the exact visible label as target; do not invent element IDs or targets from other screens. "
            "Use verify on the requested setting, with params.expected_state 'on' or 'off', "
            "only after accomplishing every part of the goal. "
            "open_settings returns to the main settings screen. wait uses params.seconds."
        )
        visible = {"screen": observation["screen"], "ui_tree": observation["ui_tree"]}
        visible_history = []
        for entry in history:
            if "feedback" in entry:
                visible_history.append({"feedback": entry["feedback"]})
                continue
            after = entry["observation"]
            visible_history.append({
                "intent": entry["intent"], "target": entry["target"], "params": entry["params"],
                "ok": entry["ok"], "previous_toggles": entry.get("previous_toggles", {}),
                "observation": {"screen": after["screen"], "ui_tree": after["ui_tree"]},
            })
        user = json.dumps({"goal": goal, "observation": visible, "history": visible_history}, ensure_ascii=False)
        for attempt in range(2):
            started_at = time.perf_counter()
            latency_ms = None
            response = None
            operation = "plan_next" if attempt == 0 else "plan_next_retry"
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=0.2, response_format={"type": "json_object"},
                )
                latency_ms = (time.perf_counter() - started_at) * 1000
                steps = self._parse_plan_steps(response.choices[0].message.content or "")
                if len(steps) != 1:
                    raise MalformedPlanError("Expected exactly one action.")
                self._record_call(operation, latency_ms, response=response)
                return steps[0]
            except MalformedPlanError as error:
                self._record_call(operation, latency_ms, response=response, error=error)
            except Exception as error:
                elapsed = (time.perf_counter() - started_at) * 1000 if latency_ms is None else latency_ms
                self._record_call(operation, elapsed, response=response, error=error)
                return None
        return None

    def summarize(self, instruction: str, log: List[Dict[str, Any]], fallback: bool = True) -> str:
        if self.client:
            started_at = time.perf_counter()
            try:
                content = [
                    {"type":"text","text": instruction},
                    {"type":"text","text": "Here is the JSON log of the run:"},
                    {"type":"text","text": json.dumps(log, indent=2)},
                ]
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role":"system","content":"You are a concise QA supervisor."},
                        {"role":"user","content": content},
                    ],
                    temperature=0.2,
                )
                self._record_call("summarize", (time.perf_counter() - started_at) * 1000, response=resp)
                return resp.choices[0].message.content
            except Exception as error:
                self._record_call("summarize", (time.perf_counter() - started_at) * 1000, error=error)
                pass
        if fallback:
            return self.fallback_summary(log)
        return "Summary unavailable."

    @staticmethod
    def fallback_summary(log: List[Dict[str, Any]]) -> str:
        steps = [r for r in log if r.get("event") == "action"]
        verifs = [r for r in log if r.get("event") == "verify"]
        bugs = [r for r in steps if not r.get("payload", {}).get("ok")]
        replans = [r for r in log if r.get("event") == "replan"]
        messages = [r for r in log if r.get("event") == "agent_message"]
        done = [r for r in log if r.get("event") == "finish"]
        plan = next((r for r in log if r.get("event") == "plan"), {})
        goal = plan.get("payload", {}).get("goal", "N/A")
        status = done[0]["payload"]["status"] if done else "unknown"
        reason = done[0]["payload"].get("reason") if done else None
        return (
            f"- Goal: {goal}\n"
            f"- Plan length: {len(steps)} actions\n"
            f"- Bugs detected: {len(bugs)}\n"
            f"- Verifications run: {len(verifs)}\n"
            f"- Replans: {len(replans)}. Role handoffs: {len(messages)}\n"
            f"- Final status: {status}"
            + (f"\n- Finish reason: {reason}" if reason else "")
        )
