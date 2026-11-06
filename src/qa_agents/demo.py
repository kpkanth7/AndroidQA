"""Small HTTP interface for the existing mock QA runner."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .benchmark import SCENARIOS

ASSETS = Path(__file__).parent / "demo_assets"
RUN_LOCK = threading.Lock()
REQUEST_TIMES: deque[float] = deque()


class DemoHandler(BaseHTTPRequestHandler):
    rule_only = False

    def log_message(self, format, *args):
        # Do not log request bodies, provider output or credentials.
        return

    def reply(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def json_reply(self, status: int, data: dict) -> None:
        self.reply(status, json.dumps(data).encode(), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        if urlsplit(self.path).path == "/api/config":
            self.json_reply(200, {"environment": "mock", "planners": ["rule"] if self.rule_only else ["rule", "llm"]})
            return
        files = {"/": ("index.html", "text/html"), "/style.css": ("style.css", "text/css"), "/app.js": ("app.js", "text/javascript")}
        asset = files.get(urlsplit(self.path).path)
        if not asset:
            self.json_reply(404, {"error": "Not found."})
            return
        self.reply(200, (ASSETS / asset[0]).read_bytes(), asset[1] + "; charset=utf-8")

    def do_POST(self) -> None:
        if self.path != "/api/run":
            self.json_reply(404, {"error": "Not found."})
            return
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            self.json_reply(403, {"error": "Request origin is not allowed."})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 1 or size > 1024:
                raise ValueError()
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict) or body.get("scenario") not in SCENARIOS:
                raise ValueError()
            if body.get("planner") not in ("rule", "llm") or type(body.get("inject_failure")) is not bool:
                raise ValueError()
        except (ValueError, TypeError, json.JSONDecodeError):
            self.json_reply(400, {"error": "Invalid request."})
            return

        self.json_reply(*self.run_scenario(body))

    def run_scenario(self, body: dict) -> tuple[int, dict]:
        if self.rule_only and body["planner"] != "rule":
            return 403, {"error": "This public demo supports only the rule-based planner."}
        if not RUN_LOCK.acquire(blocking=False):
            return 429, {"error": "Another run is in progress. Please try again shortly."}
        try:
            now = time.monotonic()
            while REQUEST_TIMES and now - REQUEST_TIMES[0] > 60:
                REQUEST_TIMES.popleft()
            if len(REQUEST_TIMES) >= 10:
                return 429, {"error": "Demo limit reached. Please wait a minute."}
            REQUEST_TIMES.append(now)
            with tempfile.TemporaryDirectory(prefix="android-qa-demo-") as run_dir:
                command = [sys.executable, "-m", "qa_agents.run_test", "--env", "mock", "--goal", SCENARIOS[body["scenario"]], "--logs_dir", run_dir]
                if body["planner"] == "llm":
                    command.append("--llm_planner")
                if body["inject_failure"]:
                    command.extend(["--inject-failure-once", "toggle"])
                try:
                    result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
                    trace = json.loads((Path(run_dir) / "qa_run.json").read_text())
                    report = (Path(run_dir) / "report.md").read_text()
                except subprocess.TimeoutExpired:
                    return 504, {"error": "The run exceeded the demo's 120-second time limit."}
                except (OSError, ValueError):
                    return 503, {"error": "The run could not complete. Please try again."}
                return 200, {"exit_code": result.returncode, "events": trace, "report": report}
        finally:
            RUN_LOCK.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule-only", action="store_true", help="Disable LLM requests, including direct API requests, for a public demo.")
    args = parser.parse_args()
    port = int(os.environ.get("PORT", "7860"))
    handler = type("ConfiguredDemoHandler", (DemoHandler,), {"rule_only": args.rule_only})
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    print(f"Android QA demo available on port {port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
