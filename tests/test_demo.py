import http.client
import json
import subprocess
import threading
from http.server import ThreadingHTTPServer

from qa_agents.demo import DemoHandler, REQUEST_TIMES, RUN_LOCK


def test_demo_runs_and_rejects_invalid_requests(monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    reply = DemoHandler.json_reply
    lock_states = []

    def record_lock_state(handler, status, data):
        if handler.command == "POST" and status in (200, 504):
            lock_states.append(RUN_LOCK.locked())
        reply(handler, status, data)

    monkeypatch.setattr(DemoHandler, "json_reply", record_lock_state)
    REQUEST_TIMES.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), DemoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=20)
    try:
        conn.request("GET", "/api/config")
        response = conn.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["planners"] == ["rule", "llm"]

        conn.request("GET", "/")
        response = conn.getresponse()
        assert response.status == 200
        assert b"Android simulation" in response.read()

        payload = json.dumps({"scenario": "wifi", "planner": "rule", "inject_failure": True})
        conn.request("POST", "/api/run", payload, {"Content-Type": "application/json"})
        response = conn.getresponse()
        result = json.loads(response.read())
        assert response.status == 200
        assert any(event["event"] == "replan" for event in result["events"])
        assert any(event["event"] == "finish" and event["payload"]["status"] == "passed" for event in result["events"])

        payload = json.dumps({"scenario": "bluetooth", "planner": "llm", "inject_failure": False})
        conn.request("POST", "/api/run", payload, {"Content-Type": "application/json"})
        response = conn.getresponse()
        result = json.loads(response.read())
        assert response.status == 200
        assert any(event["event"] == "plan_fallback" for event in result["events"])
        assert any(event["event"] == "finish" and event["payload"]["status"] == "passed" for event in result["events"])

        conn.request("POST", "/api/run", '{"scenario":"anything"}', {"Content-Type": "application/json"})
        response = conn.getresponse()
        assert response.status == 400
        response.read()

        def time_out(command, *, timeout, **kwargs):
            raise subprocess.TimeoutExpired(command, timeout)

        monkeypatch.setattr(subprocess, "run", time_out)
        conn.request("POST", "/api/run", payload, {"Content-Type": "application/json"})
        response = conn.getresponse()
        assert response.status == 504
        assert "120-second time limit" in json.loads(response.read())["error"]
        assert lock_states == [False, False, False]

        with RUN_LOCK:
            conn.request("POST", "/api/run", payload, {"Content-Type": "application/json"})
            response = conn.getresponse()
            assert response.status == 429
            assert "Another run is in progress" in json.loads(response.read())["error"]
    finally:
        conn.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_rule_only_rejects_llm_even_with_credentials_and_runs_recovery(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-must-never-be-used")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    handler = type("PublicDemoHandler", (DemoHandler,), {"rule_only": True})
    REQUEST_TIMES.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=20)
    try:
        conn.request("GET", "/api/config")
        response = conn.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == {"environment": "mock", "planners": ["rule"]}

        with monkeypatch.context() as patch:
            def unexpected_run(*args, **kwargs):
                raise AssertionError("Disabled LLM must not start a runner")

            patch.setattr(subprocess, "run", unexpected_run)
            payload = json.dumps({"scenario": "wifi", "planner": "llm", "inject_failure": False})
            conn.request("POST", "/api/run", payload, {"Content-Type": "application/json"})
            response = conn.getresponse()
            assert response.status == 403
            assert "rule-based" in json.loads(response.read())["error"]
            assert not REQUEST_TIMES

        payload = json.dumps({"scenario": "wifi", "planner": "rule", "inject_failure": True})
        conn.request("POST", "/api/run", payload, {"Content-Type": "application/json"})
        response = conn.getresponse()
        result = json.loads(response.read())
        assert response.status == 200
        assert any(event["event"] == "replan" for event in result["events"])
        assert any(event["event"] == "finish" and event["payload"]["status"] == "passed" for event in result["events"])
        assert not any(event["event"] == "llm" for event in result["events"])
    finally:
        conn.close()
        server.shutdown()
        server.server_close()
        thread.join()
