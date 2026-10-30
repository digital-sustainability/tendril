"""Synchronous facade over ``responsible_request.AsyncOpenAI``.

Every OpenAI-compatible chat client in this package subclasses
:class:`ResponsibleRequestClient`. Experiment YAML is resolved by
:func:`resolve_throttle` and :func:`responsible_request_kwargs`; direct
construction keeps the provider defaults from :mod:`tendril.inference.settings`.

:meth:`ResponsibleRequestClient.generate_structured` asks
``responsible_request.structured`` for a Pydantic model instead of raw text.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import Any, TypeVar

import responsible_request as rr
from pydantic import BaseModel

from .base_client import BaseLLMClient
from .settings import settings

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_THROTTLE_KEYS = frozenset(
    {
        "mode",
        "max_rpm",
        "min_rpm",
        "start_rpm",
        "max_concurrency",
        "high_ratio",
        "recover_ratio",
        "cooldown_s",
    }
)


def throttle_profile(client_name: str) -> dict[str, Any]:
    """Provider defaults used when YAML omits a throttle field."""
    if client_name == "openrouter":
        rpm = settings.openrouter_rr_rpm
        return {
            "mode": "fixed",
            "max_rpm": rpm,
            "min_rpm": rpm,
            "start_rpm": rpm,
            "max_concurrency": settings.openrouter_rr_max_concurrency,
            "high_ratio": 3.0,
            "recover_ratio": 1.5,
            "cooldown_s": 120.0,
        }
    rpm = 60.0
    return {
        "mode": "fixed",
        "max_rpm": rpm,
        "min_rpm": rpm,
        "start_rpm": rpm,
        "max_concurrency": 10,
        "high_ratio": 3.0,
        "recover_ratio": 1.5,
        "cooldown_s": 120.0,
    }


def _throttle_block(source: Mapping[str, Any] | None) -> dict[str, Any]:
    if not source:
        return {}
    block = source.get("throttle")
    if block is None:
        return {}
    if not isinstance(block, Mapping):
        raise ValueError("throttle must be a mapping")
    unknown = set(block) - _THROTTLE_KEYS
    if unknown:
        raise ValueError(
            f"unknown throttle keys {sorted(unknown)}; valid: {sorted(_THROTTLE_KEYS)}"
        )
    return dict(block)


def resolve_throttle(
    client_name: str,
    model_cfg: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> rr.ThrottleConfig:
    """Build a throttle config from provider defaults overlaid with YAML.

    Model ``throttle`` overrides runtime ``throttle``. RPM and mode stay at the
    provider default when both blocks are omitted. ``max_concurrency`` falls
    back to the model's ``max_workers``, then ``runtime.max_workers``, then the
    provider default.
    """
    profile = throttle_profile(client_name)
    merged = _throttle_block(runtime)
    merged.update(_throttle_block(model_cfg))
    mode = merged.get("mode", profile["mode"])
    if mode not in ("adaptive", "fixed"):
        raise ValueError(f"throttle.mode must be 'adaptive' or 'fixed', got {mode!r}")

    if "max_concurrency" in merged:
        max_concurrency = int(merged["max_concurrency"])
    elif model_cfg and model_cfg.get("max_workers") is not None:
        max_concurrency = int(model_cfg["max_workers"])
    elif runtime and runtime.get("max_workers") is not None:
        max_concurrency = int(runtime["max_workers"])
    else:
        max_concurrency = int(profile["max_concurrency"])

    max_rpm = float(merged.get("max_rpm", profile["max_rpm"]))
    if mode == "fixed":
        return rr.ThrottleConfig.fixed(max_rpm, max_concurrency=max_concurrency)
    return rr.ThrottleConfig(
        max_rpm=max_rpm,
        min_rpm=float(merged.get("min_rpm", profile["min_rpm"])),
        start_rpm=float(merged.get("start_rpm", profile["start_rpm"])),
        max_concurrency=max_concurrency,
        high_ratio=float(merged.get("high_ratio", profile["high_ratio"])),
        recover_ratio=float(merged.get("recover_ratio", profile["recover_ratio"])),
        cooldown_s=float(merged.get("cooldown_s", profile["cooldown_s"])),
    )


def _nested_option(model_cfg: Mapping[str, Any] | None, runtime: Mapping[str, Any] | None, key: str) -> Any:
    value = None
    if runtime and key in runtime:
        value = runtime[key]
    if model_cfg and key in model_cfg:
        value = model_cfg[key]
    return value


def cost_option(
    model_cfg: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> rr.CostConfig | bool:
    """Track cost on every request. A YAML ``budget_usd`` also stops the run."""
    budget = _nested_option(model_cfg, runtime, "budget_usd")
    if budget is None:
        return True
    return rr.CostConfig(budget_usd=float(budget))


def cache_option(
    model_cfg: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> bool | None:
    """Return ``True`` only when YAML sets ``cache: true``."""
    enabled = _nested_option(model_cfg, runtime, "cache")
    if enabled:
        return True
    return None


def responsible_request_kwargs(
    model_cfg: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    log_sqlite: str | Path,
    run_name: str | None = None,
    run_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Keyword arguments for a judge-run client: YAML wins over env defaults."""
    options: dict[str, Any] = {
        "rr_log_sqlite": log_sqlite,
        "throttle": resolve_throttle(str(model_cfg.get("client", "")), model_cfg, runtime),
        "cost": cost_option(model_cfg, runtime),
    }
    if runtime.get("timeout") is not None:
        options["timeout"] = runtime["timeout"]
    if runtime.get("max_retries") is not None:
        options["max_retries"] = runtime["max_retries"]
    cache = cache_option(model_cfg, runtime)
    if cache is not None:
        options["cache"] = cache
    if run_name is not None or run_metadata:
        options["run"] = rr.RunConfig(
            name=run_name,
            metadata=dict(run_metadata or {}),
        )
    return options


def message_thinking(message: Any) -> Any:
    """Reasoning text from an OpenAI-compatible chat message, if the provider sent one."""
    model_extra = getattr(message, "model_extra", None) or {}
    return (
        getattr(message, "reasoning", None)
        or getattr(message, "reasoning_content", None)
        or getattr(message, "reasoning_details", None)
        or getattr(message, "thinking", None)
        or model_extra.get("reasoning")
        or model_extra.get("reasoning_content")
        or model_extra.get("reasoning_details")
        or model_extra.get("thinking")
    )


def basic_usage(
    response: Any,
    model_name: str,
    *,
    resolved_model: bool,
) -> tuple[str | None, dict[str, Any] | None]:
    """Content plus the usage fields shared by OpenAI-compatible chat responses."""
    if not getattr(response, "choices", None):
        return None, None
    choice = response.choices[0]
    message = choice.message
    usage = response.usage
    usage_info: dict[str, Any] = {
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None,
        "finish_reason": getattr(choice, "finish_reason", None),
        "thinking": message_thinking(message),
    }
    if resolved_model:
        usage_info["resolved_model"] = getattr(response, "model", model_name)
    return message.content, usage_info


def chat_api_params(
    model_name: str,
    messages: Sequence[Mapping[str, str]],
    params: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    *,
    temperature_default: float = 0.7,
    top_p_default: float = 0.95,
    top_k: str = "never",
    repetition_penalty: str = "frequency",
) -> dict[str, Any]:
    """Map judge params onto an OpenAI chat-completions body.

    ``top_k`` is ``"always"`` (OpenAI, default 40), ``"if_set"`` (Together),
    or ``"never"``. ``repetition_penalty`` is ``"frequency"`` (map onto
    ``frequency_penalty``), ``"native"`` (pass through), or ``"omit"``.
    """
    api_params: dict[str, Any] = {
        "model": model_name,
        "messages": list(messages),
        "max_tokens": params.get("max_tokens", 1000),
        "temperature": params.get("temperature", temperature_default),
        "seed": params.get("seed", 42),
        "top_p": params.get("top_p", top_p_default),
    }
    if top_k == "always":
        api_params["top_k"] = params.get("top_k", 40)
    elif top_k == "if_set" and "top_k" in params:
        api_params["top_k"] = params["top_k"]

    if repetition_penalty == "frequency" and "repetition_penalty" in params:
        api_params["frequency_penalty"] = min(
            2.0, max(0.0, float(params["repetition_penalty"]) - 1.0)
        )
    elif repetition_penalty == "native" and "repetition_penalty" in params:
        api_params["repetition_penalty"] = params["repetition_penalty"]
    if "frequency_penalty" in params and repetition_penalty != "native":
        api_params["frequency_penalty"] = params["frequency_penalty"]
    if "presence_penalty" in params and repetition_penalty != "native":
        api_params["presence_penalty"] = params["presence_penalty"]

    response_format = kwargs.get("response_format")
    reasoning_effort = kwargs.get("reasoning_effort")
    timeout = kwargs.get("timeout")
    if response_format:
        api_params["response_format"] = response_format
    if reasoning_effort:
        api_params["reasoning_effort"] = reasoning_effort
    if timeout:
        api_params["timeout"] = timeout
    return api_params


class ResponsibleRequestClient(BaseLLMClient):
    """Sync ``generate`` / ``generate_batch`` over one background event loop.

    ``responsible-request`` keeps throttle state per event loop. One loop for
    the lifetime of the client means single calls and batches share one limiter.
    """

    client_name = "rr"
    track_resolved_model = False

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super().__init__(model_name, **kwargs)
        api_key = self._api_key()
        if not api_key:
            raise ValueError(self._missing_key_message())

        self._loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"rr-{self.client_name}-{model_name}".replace("/", "_"),
            daemon=True,
        )
        self._thread.start()
        self._loop_ready.wait()
        self._closed = False
        log_sqlite = Path(kwargs.get("rr_log_sqlite") or self._default_log_sqlite()).resolve()
        try:
            self.client = self._submit(self._create_client(api_key, log_sqlite)).result()
        except Exception:
            self._closed = True
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            raise

    def _api_key(self) -> str | None:
        raise NotImplementedError

    def _missing_key_message(self) -> str:
        return f"API key for client {self.client_name!r} is not set."

    def _base_url(self) -> str | None:
        return None

    def _default_headers(self) -> dict[str, str] | None:
        return None

    def _default_log_sqlite(self) -> Path:
        raise NotImplementedError

    def _console_level(self) -> str:
        return "INFO"

    def _default_throttle(self) -> rr.ThrottleConfig:
        return resolve_throttle(self.client_name)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        self._loop.run_forever()

    def _submit(self, coroutine: Any) -> Future:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    def _throttle(self) -> rr.ThrottleConfig:
        throttle = self.kwargs.get("throttle")
        if isinstance(throttle, rr.ThrottleConfig):
            return throttle
        if isinstance(throttle, Mapping):
            return resolve_throttle(self.client_name, {"throttle": dict(throttle)})
        return self._default_throttle()

    async def _create_client(self, api_key: str, log_sqlite: Path) -> rr.AsyncOpenAI:
        log_sqlite.parent.mkdir(parents=True, exist_ok=True)
        create_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": self.kwargs.get("timeout", settings.llm_timeout),
            "max_retries": self.kwargs.get("max_retries", settings.llm_max_retries),
            "throttle": self._throttle(),
            "log": rr.LogConfig(
                console_level=self._console_level(),
                sqlite=log_sqlite,
            ),
            "default_params": rr.reproducible(seed=42),
            "cost": self.kwargs.get("cost", True),
        }
        base_url = self._base_url()
        if base_url:
            create_kwargs["base_url"] = base_url
        headers = self._default_headers()
        if headers:
            create_kwargs["default_headers"] = headers
        cache = self.kwargs.get("cache")
        if cache:
            create_kwargs["cache"] = cache
        run = self.kwargs.get("run")
        if run is not None:
            create_kwargs["run"] = run
        return rr.AsyncOpenAI(**create_kwargs)

    def _api_params(
        self,
        messages: Sequence[Mapping[str, str]],
        params: Mapping[str, Any],
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        return chat_api_params(self.model_name, messages, params, kwargs)

    def _response_value(
        self,
        response: Any,
        *,
        return_usage: bool,
    ) -> str | None | tuple[str | None, dict[str, Any] | None]:
        text, usage = basic_usage(
            response,
            self.model_name,
            resolved_model=self.track_resolved_model,
        )
        if return_usage:
            return text, usage
        return text

    async def _generate_async(
        self,
        messages: Sequence[Mapping[str, str]],
        params: Mapping[str, Any],
        kwargs: Mapping[str, Any],
    ) -> tuple[str | None, dict[str, Any] | None]:
        response = await self.client.chat.completions.create(
            **self._api_params(messages, params, kwargs)
        )
        value = self._response_value(response, return_usage=True)
        if isinstance(value, tuple):
            return value
        return value, None

    def generate(
        self,
        messages: list[dict[str, str]],
        params: dict[str, Any],
        **kwargs: Any,
    ) -> str | None | tuple[str | None, dict[str, Any] | None]:
        response_text, usage = self._submit(
            self._generate_async(messages, params, kwargs)
        ).result()
        return (response_text, usage) if kwargs.get("return_usage", False) else response_text

    async def _generate_structured_async(
        self,
        messages: Sequence[Mapping[str, str]],
        schema: type[SchemaT],
        params: Mapping[str, Any],
        kwargs: Mapping[str, Any],
        *,
        retries: int,
        strict: bool,
        fallback: bool,
    ) -> SchemaT:
        api_params = self._api_params(messages, params, kwargs)
        model = api_params.pop("model")
        chat_messages = api_params.pop("messages")
        api_params.pop("response_format", None)
        return await rr.structured(
            self.client,
            model=model,
            messages=chat_messages,
            schema=schema,
            retries=retries,
            strict=strict,
            fallback=fallback,
            **api_params,
        )

    def generate_structured(
        self,
        messages: Sequence[Mapping[str, str]],
        schema: type[SchemaT],
        params: Mapping[str, Any] | None = None,
        *,
        retries: int = 2,
        strict: bool = True,
        fallback: bool = True,
        **kwargs: Any,
    ) -> SchemaT:
        """Return a validated ``schema`` instance via ``responsible_request.structured``.

        Provider params (throttle, reasoning, ``extra_body``) come from the same
        mapping as :meth:`generate`. The schema supplies ``response_format``.
        """
        return self._submit(
            self._generate_structured_async(
                messages,
                schema,
                params or {},
                kwargs,
                retries=retries,
                strict=strict,
                fallback=fallback,
            )
        ).result()

    async def _generate_batch_async(
        self,
        requests: Sequence[Mapping[str, Any]],
        *,
        progress: bool,
        batch_id: str | None,
        tags: Mapping[str, Any] | None,
    ) -> list[Any]:
        api_requests = [
            self._api_params(
                list(request["messages"]),
                dict(request.get("params", {})),
                dict(request.get("kwargs", {})),
            )
            for request in requests
        ]
        with rr.tags(**dict(tags or {})):
            responses = await rr.run_batch(
                self.client,
                api_requests,
                progress=progress,
                return_exceptions=True,
                batch_id=batch_id,
            )
        converted: list[Any] = []
        for response in responses:
            if isinstance(response, BaseException):
                converted.append(response)
                continue
            value = self._response_value(response, return_usage=True)
            converted.append(value if isinstance(value, tuple) else (value, None))
        return converted

    def generate_batch(
        self,
        requests: Sequence[Mapping[str, Any]],
        *,
        progress: bool = True,
        batch_id: str | None = None,
        tags: Mapping[str, Any] | None = None,
    ) -> list[Any]:
        """Generate many completions through one paced ``rr.run_batch``."""
        return self._submit(
            self._generate_batch_async(
                requests,
                progress=progress,
                batch_id=batch_id,
                tags=tags,
            )
        ).result()

    async def _close_async(self) -> None:
        await self.client.close()

    def close(self) -> None:
        if self._closed:
            return
        self._submit(self._close_async()).result()
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
