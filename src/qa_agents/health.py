from __future__ import annotations

import argparse
import json

from .llm.openai_client import LLMClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the configured LLM provider without exposing credentials.")
    parser.add_argument("--model", help="Optional model override to validate.")
    args = parser.parse_args()

    client = LLMClient(model=args.model)
    metadata = client.metadata()
    if not client.client:
        print(json.dumps({**metadata, "healthy": False, "reason": "No usable API key is configured."}, indent=2))
        return 1

    try:
        models = client.client.models.list()
        available = {item.id for item in models.data}
        healthy = client.model in available
        print(json.dumps({
            **metadata,
            "healthy": healthy,
            "configured_model_available": healthy,
        }, indent=2))
        return 0 if healthy else 1
    except Exception as error:
        print(json.dumps({
            **metadata,
            "healthy": False,
            "reason": f"Provider request failed: {type(error).__name__}",
        }, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
