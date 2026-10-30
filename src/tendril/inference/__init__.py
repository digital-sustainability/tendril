"""LLM clients for the TENDRIL LLM-as-a-judge experiments.

``OpenRouterClient`` implements ``BaseLLMClient.generate(messages, params, **kwargs)``
and talks to the OpenRouter API through ``responsible_request.AsyncOpenAI``.

Credentials come from the environment or the project-root ``.env`` file
(see ``tendril.inference.settings``). Judge YAML ``throttle``, ``timeout``, and
``max_retries`` override those defaults when a runner constructs the client.

    from tendril.inference import OpenRouterClient

    client = OpenRouterClient("qwen/qwen3.8-flash")
    response = client.generate(
        messages=[{"role": "user", "content": "Is this answer correct?"}],
        params={"max_tokens": 500, "temperature": 0.0},
        return_usage=True,  # -> (text, usage_info)
    )
"""

from .base_client import BaseLLMClient
from .openrouter_client import OpenRouterClient
from .rr_client import resolve_throttle, responsible_request_kwargs
from .settings import settings

__all__ = [
    "BaseLLMClient",
    "OpenRouterClient",
    "resolve_throttle",
    "responsible_request_kwargs",
    "settings",
]
