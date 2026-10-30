#!/usr/bin/env python
from __future__ import annotations

import asyncio
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tendril.inference.base_client import BaseLLMClient  # noqa: E402
from tendril.inference.openrouter_client import OpenRouterClient  # noqa: E402


def completion_response():
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        model_extra={"cost": 0.001},
        prompt_tokens_details=None,
        completion_tokens_details=None,
        cost_details=None,
    )
    message = SimpleNamespace(
        content='{"O": 2}',
        model_extra={},
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
        thinking=None,
    )
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        usage=usage,
        model="qwen/qwen3.8-27b",
        provider="test-provider",
    )


def bare_client(client_class):
    client = client_class.__new__(client_class)
    BaseLLMClient.__init__(
        client,
        "qwen/qwen3.8-27b",
        providers=["provider-a"],
    )
    return client


def test_openrouter_generate_uses_shared_params_and_parser():
    client = bare_client(OpenRouterClient)
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return completion_response()

    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    text, usage = asyncio.run(
        client._generate_async(
            [{"role": "user", "content": "prompt"}],
            {"max_tokens": 20, "temperature": 0},
            {"return_usage": True},
        )
    )

    assert text == '{"O": 2}'
    assert usage["cost"] == 0.001
    assert usage["provider"] == "test-provider"
    assert captured["extra_body"]["usage"] == {"include": True}
    assert captured["extra_body"]["provider"] == {"order": ["provider-a"]}


class _Verdict:
    def __init__(self, *, restricted: bool, reason: str) -> None:
        self.restricted = restricted
        self.reason = reason


def test_openrouter_structured_forwards_provider_params():
    client = bare_client(OpenRouterClient)
    client.client = object()
    captured = {}

    async def structured(rr_client, **kwargs):
        captured["client"] = rr_client
        captured.update(kwargs)
        return kwargs["schema"](restricted=True, reason="ok")

    with patch("tendril.inference.rr_client.rr.structured", new=structured):
        result = asyncio.run(
            client._generate_structured_async(
                [{"role": "user", "content": "prompt"}],
                _Verdict,
                {"max_tokens": 20, "temperature": 0},
                {"reasoning": {"enabled": False}},
                retries=1,
                strict=True,
                fallback=True,
            )
        )

    assert result.restricted is True
    assert captured["client"] is client.client
    assert captured["model"] == "qwen/qwen3.8-27b"
    assert captured["retries"] == 1
    assert captured["max_tokens"] == 20
    assert captured["temperature"] == 0
    assert "response_format" not in captured
    assert captured["extra_body"]["reasoning"] == {"enabled": False}
    assert captured["extra_body"]["usage"] == {"include": True}


def test_openrouter_title_header():
    client = bare_client(OpenRouterClient)
    client.kwargs["title"] = "TENDRIL Restriction Filter"
    headers = client._default_headers()
    assert headers is not None
    assert headers["X-Title"] == "TENDRIL Restriction Filter"


def test_openrouter_batch_uses_shared_params_and_parser():
    client = bare_client(OpenRouterClient)
    client.client = object()
    captured = {}

    async def run_batch(rr_client, requests, **kwargs):
        captured["client"] = rr_client
        captured["requests"] = requests
        captured["kwargs"] = kwargs
        return [completion_response()]

    with (
        patch(
            "tendril.inference.rr_client.rr.run_batch",
            new=run_batch,
        ),
        patch(
            "tendril.inference.rr_client.rr.tags",
            side_effect=lambda **kwargs: nullcontext(),
        ),
    ):
        result = asyncio.run(
            client._generate_batch_async(
                [
                    {
                        "messages": [{"role": "user", "content": "prompt"}],
                        "params": {"max_tokens": 20, "temperature": 0},
                        "kwargs": {},
                    }
                ],
                progress=False,
                batch_id="batch-1",
                tags={"experiment": "test"},
            )
        )

    text, usage = result[0]
    assert text == '{"O": 2}'
    assert usage["cost"] == 0.001
    assert captured["requests"][0]["extra_body"]["usage"] == {"include": True}
    assert captured["kwargs"]["batch_id"] == "batch-1"


def _run_all():
    failures = []
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures.append(name)
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(
        f"\n{len(tests)} tests, {len(failures)} failure(s)"
        if failures
        else f"\nAll {len(tests)} tests passed."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
