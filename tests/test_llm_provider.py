from types import SimpleNamespace

from qa_agents.llm.openai_client import LLMClient


def test_groq_uses_openai_compatible_configuration(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")

    client = LLMClient()

    assert client.metadata() == {
        "provider": "groq",
        "model": "openai/gpt-oss-20b",
        "configured": True,
    }
    assert str(client.client.base_url) == "https://api.groq.com/openai/v1/"


def test_invalid_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unsupported")

    try:
        LLMClient()
    except ValueError as error:
        assert "LLM_PROVIDER" in str(error)
    else:
        raise AssertionError("Expected an invalid provider to fail")


def test_malformed_plan_is_retried_once(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    client = LLMClient()
    responses = iter([
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"steps": {}}'))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=(
            '{"steps": [{"id": 1, "description": "Open", "intent": "open_settings", '
            '"target": null, "params": {}}]}'
        )))]),
    ])
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: next(responses)))
    )

    steps = client.plan("Open settings")

    assert steps and steps[0]["intent"] == "open_settings"
    assert [call["operation"] for call in client.calls] == ["plan", "plan_retry"]
    assert client.calls[0]["error_type"] == "MalformedPlanError"
    assert client.calls[1]["success"] is True
