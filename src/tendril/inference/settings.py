"""Centralized configuration for the LLM clients.

Reads environment variables in precedence order: process environment first,
then the project-root ``.env`` file, then the defaults declared below. Import
the shared ``settings`` singleton rather than calling ``os.getenv`` at the
point of use:

    from tendril.inference.settings import settings

    api_key = settings.openrouter_api_key

Only the variables needed by the OpenRouter client are kept here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from tendril.paths import REPOSITORY_PATH


def repo_root() -> Path:
    """Directory that contains ``pyproject.toml`` and ``.env``."""
    return REPOSITORY_PATH


_PROJECT_ROOT = repo_root()
_ENV_FILE = _PROJECT_ROOT / ".env"

if _ENV_FILE.exists():
    load_dotenv(dotenv_path=_ENV_FILE)


class Settings:
    """Typed view over the environment configuration used by the LLM clients."""

    def __init__(self):
        # ── LLM provider credentials ──────────────────────────────────────────
        # Optional: each client raises a targeted error at construction if its
        # own key is required but missing.
        self.openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
        # ── OpenRouter request metadata (provider-side log correlation) ───────
        self.openrouter_referer: str = os.getenv("OPENROUTER_REFERER", "https://tendril-project.local")
        self.openrouter_app_name: str = os.getenv("OPENROUTER_APP_NAME", "TENDRIL")
        self.openrouter_rr_rpm: float = float(
            os.getenv("OPENROUTER_RR_RPM", "60")
        )
        self.openrouter_rr_max_concurrency: int = int(
            os.getenv("OPENROUTER_RR_MAX_CONCURRENCY", "10")
        )
        self.openrouter_rr_console_level: str = os.getenv(
            "OPENROUTER_RR_CONSOLE_LEVEL", "INFO"
        )
        openrouter_log_sqlite = Path(
            os.getenv(
                "OPENROUTER_RR_LOG_SQLITE",
                "logs/responsible_request.sqlite",
            )
        )
        self.openrouter_rr_log_sqlite: Path = (
            openrouter_log_sqlite
            if openrouter_log_sqlite.is_absolute()
            else _PROJECT_ROOT / openrouter_log_sqlite
        )

        # ── Outbound LLM client resilience (every LLM SDK client) ─────────────
        # llm_timeout: per-request timeout (seconds). llm_max_retries: SDK-level
        # retries for transient failures (connection/timeout/429/5xx).
        self.llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "60.0"))
        self.llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))


#: Import this shared instance; do not construct ``Settings()`` ad hoc.
settings = Settings()