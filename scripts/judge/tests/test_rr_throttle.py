#!/usr/bin/env python
"""Throttle resolution for ResponsibleRequest judge clients.

Runnable directly (no pytest required):

    .venv/bin/python scripts/judge/tests/test_rr_throttle.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import judge_common as jc  # noqa: E402
from tendril.inference.rr_client import resolve_throttle  # noqa: E402
from tendril.inference.settings import settings  # noqa: E402


def _restore(names: dict[str, object]) -> None:
    for name, value in names.items():
        setattr(settings, name, value)


def test_yaml_throttle_overrides_provider_defaults():
    saved = {
        "openrouter_rr_rpm": settings.openrouter_rr_rpm,
        "openrouter_rr_max_concurrency": settings.openrouter_rr_max_concurrency,
    }
    settings.openrouter_rr_rpm = 300
    settings.openrouter_rr_max_concurrency = 32
    try:
        throttle = resolve_throttle(
            "openrouter",
            {
                "client": "openrouter",
                "max_workers": 1,
                "throttle": {"max_rpm": 40, "mode": "adaptive"},
            },
            {
                "max_workers": 8,
                "throttle": {"max_rpm": 100, "min_rpm": 5, "start_rpm": 10},
            },
        )
    finally:
        _restore(saved)
    assert throttle.max_rpm == 40
    assert throttle.min_rpm == 5
    assert throttle.adaptive is True
    assert throttle.max_concurrency == 1


def test_openrouter_fixed_default_uses_max_workers():
    saved = {
        "openrouter_rr_rpm": settings.openrouter_rr_rpm,
        "openrouter_rr_max_concurrency": settings.openrouter_rr_max_concurrency,
    }
    settings.openrouter_rr_rpm = 60
    settings.openrouter_rr_max_concurrency = 10
    try:
        throttle = resolve_throttle(
            "openrouter",
            {"client": "openrouter", "max_workers": 10},
            {},
        )
    finally:
        _restore(saved)
    assert throttle.max_rpm == 60
    assert throttle.adaptive is False
    assert throttle.max_concurrency == 10


def test_runtime_max_workers_fills_concurrency_when_model_omits_it():
    throttle = resolve_throttle(
        "openai",
        {"client": "openai"},
        {"max_workers": 3},
    )
    assert throttle.adaptive is False
    assert throttle.max_concurrency == 3


def test_openrouter_rr_is_rejected():
    try:
        jc.make_client("openrouter-rr", "qwen/qwen3.8-27b")
    except ValueError as exc:
        assert "openrouter-rr" in str(exc)
    else:
        raise AssertionError("openrouter-rr should be an unknown client")


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
