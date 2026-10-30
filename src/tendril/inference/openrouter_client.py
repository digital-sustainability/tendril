from collections.abc import Mapping, Sequence
from typing import Any

from .rr_client import ResponsibleRequestClient, message_thinking
from .settings import settings


def _usage_scalar(obj: Any, name: str) -> Any:
    """Read a scalar usage-accounting field from an OpenRouter usage object.

    OpenRouter extends the OpenAI ``CompletionUsage`` with non-standard fields
    (``cost``, ``provider``, ``native_tokens_*``). Depending on the SDK version
    they are exposed either as plain attributes or parked in ``model_extra`` —
    check both. Anything that is not a JSON scalar (nested detail objects,
    unmodelled SDK mocks) yields None, so unknown response shapes degrade to
    null instead of leaking objects into persisted result rows.
    """
    value = getattr(obj, name, None)
    if value is None:
        value = (getattr(obj, "model_extra", None) or {}).get(name)
    return value if isinstance(value, (bool, int, float, str)) else None


def _usage_object(obj: Any, name: str) -> Any:
    """Read a nested detail object (``prompt_tokens_details`` & friends)."""
    value = getattr(obj, name, None)
    if value is None:
        value = (getattr(obj, "model_extra", None) or {}).get(name)
    return value


def _detail_scalar(details: Any, name: str) -> Any:
    """Read a scalar from a token-details object (dict, pydantic model or None)."""
    if details is None:
        return None
    if isinstance(details, dict):
        value = details.get(name)
    else:
        value = getattr(details, name, None)
        if value is None:
            value = (getattr(details, "model_extra", None) or {}).get(name)
    return value if isinstance(value, (bool, int, float, str)) else None


class OpenRouterClient(ResponsibleRequestClient):
    """OpenRouter client with fixed-rate ResponsibleRequest pacing and logs."""

    client_name = "openrouter"

    def _api_key(self) -> str | None:
        return settings.openrouter_api_key

    def _missing_key_message(self) -> str:
        return (
            "OPENROUTER_API_KEY environment variable not set. "
            "Get your key at https://openrouter.ai/keys"
        )

    def _base_url(self) -> str | None:
        return "https://openrouter.ai/api/v1"

    def _default_headers(self) -> dict[str, str] | None:
        title = self.kwargs.get("title") or settings.openrouter_app_name
        return {
            "HTTP-Referer": settings.openrouter_referer,
            "X-Title": str(title),
        }

    def _default_log_sqlite(self):
        return settings.openrouter_rr_log_sqlite

    def _console_level(self) -> str:
        return str(self.kwargs.get("console_level") or settings.openrouter_rr_console_level)

    def _api_params(
        self,
        messages: Sequence[Mapping[str, str]],
        params: Mapping[str, Any],
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        response_format = kwargs.get("response_format", None)
        reasoning = kwargs.get("reasoning", None)
        reasoning_effort = kwargs.get("reasoning_effort", None)
        timeout = kwargs.get("timeout", None)

        providers = kwargs.get("providers") or self.kwargs.get("providers")
        if isinstance(providers, list):
            providers = {"order": providers}
        provider = kwargs.get("provider") or self.kwargs.get("provider")
        routing = kwargs.get("routing") or self.kwargs.get("routing")

        if routing == "lowest-price":
            providers = None
            provider = None
        elif routing:
            if not providers:
                providers = {}
            providers["sort"] = routing

        if provider and not providers:
            providers = {"order": [provider]}

        api_params: dict[str, Any] = {
            "model": self.model_name,
            "messages": list(messages),
            "max_tokens": params.get("max_tokens", 1000),
            "temperature": params.get("temperature", 0.7),
            "seed": params.get("seed", 42),
            "top_p": params.get("top_p", 0.95),
        }

        if timeout:
            api_params["timeout"] = timeout
        if response_format:
            api_params["response_format"] = response_format

        extra_body: dict[str, Any] = {"usage": {"include": True}}
        if reasoning is not None:
            extra_body["reasoning"] = (
                {"effort": reasoning} if isinstance(reasoning, str) else reasoning
            )
        elif reasoning_effort:
            extra_body["reasoning"] = {"effort": reasoning_effort}
            api_params["reasoning_effort"] = reasoning_effort

        if providers:
            extra_body["provider"] = providers

        api_params["extra_body"] = extra_body

        llm_trace_id = kwargs.get("llm_trace_id")
        if llm_trace_id:
            api_params["extra_headers"] = {"X-Request-ID": llm_trace_id}

        return api_params

    def _response_value(
        self,
        response: Any,
        *,
        return_usage: bool,
    ) -> str | None | tuple[str | None, dict[str, Any] | None]:
        if len(response.choices) > 0:
            choice = response.choices[0]
            message = choice.message
            response_text = message.content
            if return_usage:
                usage_info: dict[str, Any] = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                    "completion_tokens": response.usage.completion_tokens if response.usage else None,
                    "total_tokens": response.usage.total_tokens if response.usage else None,
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "resolved_model": getattr(response, "model", self.model_name),
                    "thinking": message_thinking(message),
                }
                usage = response.usage
                cost = _usage_scalar(usage, "cost")
                if cost is not None:
                    usage_info["cost"] = cost
                provider = _usage_scalar(usage, "provider")
                if provider is None:
                    provider = _usage_scalar(response, "provider")
                usage_info["provider"] = provider
                usage_info["native_prompt_tokens"] = _usage_scalar(usage, "native_tokens_prompt")
                usage_info["native_completion_tokens"] = _usage_scalar(
                    usage, "native_tokens_completion"
                )
                usage_info["native_reasoning_tokens"] = _usage_scalar(
                    usage, "native_tokens_reasoning"
                )
                usage_info["cached_prompt_tokens"] = _detail_scalar(
                    _usage_object(usage, "prompt_tokens_details"), "cached_tokens"
                )
                usage_info["reasoning_tokens"] = _detail_scalar(
                    _usage_object(usage, "completion_tokens_details"), "reasoning_tokens"
                )
                usage_info["upstream_inference_cost"] = _detail_scalar(
                    _usage_object(usage, "cost_details"), "upstream_inference_cost"
                )
                return response_text, usage_info
            return response_text

        if return_usage:
            return None, None
        return None
