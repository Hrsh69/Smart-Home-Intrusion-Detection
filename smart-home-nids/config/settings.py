"""Runtime settings loaded from environment / .env file.

Usage:
    from config.settings import SETTINGS
    print(SETTINGS.SMTP_HOST)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key, "").lower()
    if val in ("1", "true", "yes"):
        return True
    if val in ("0", "false", "no"):
        return False
    return default


@dataclass(frozen=True)
class Settings:
    """Application-wide runtime settings."""

    # ── Paths ────────────────────────────────────────────────────────────
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
    DB_PATH: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "nids.db")

    # ── Email Alerts ─────────────────────────────────────────────────────
    SMTP_HOST: str = field(default_factory=lambda: _env("SMTP_HOST", "smtp.gmail.com"))
    SMTP_PORT: int = field(default_factory=lambda: _env_int("SMTP_PORT", 587))
    SMTP_USER: str = field(default_factory=lambda: _env("SMTP_USER"))
    SMTP_PASS: str = field(default_factory=lambda: _env("SMTP_PASS"))
    ALERT_EMAIL_TO: str = field(default_factory=lambda: _env("ALERT_EMAIL_TO"))
    EMAIL_ALERTS_ENABLED: bool = field(default_factory=lambda: _env_bool("EMAIL_ALERTS_ENABLED", False))

    # ── Telegram Alerts ──────────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: str = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN"))
    TELEGRAM_CHAT_ID: str = field(default_factory=lambda: _env("TELEGRAM_CHAT_ID"))
    TELEGRAM_ALERTS_ENABLED: bool = field(default_factory=lambda: _env_bool("TELEGRAM_ALERTS_ENABLED", False))

    # ── Desktop Notifications ────────────────────────────────────────────
    DESKTOP_ALERTS_ENABLED: bool = field(default_factory=lambda: _env_bool("DESKTOP_ALERTS_ENABLED", True))

    # ── Dashboard ────────────────────────────────────────────────────────
    DASHBOARD_REFRESH_SECONDS: int = field(default_factory=lambda: _env_int("DASHBOARD_REFRESH_SECONDS", 5))
    DASHBOARD_THEME: str = field(default_factory=lambda: _env("DASHBOARD_THEME", "dark"))
    DASHBOARD_PAGE_SIZE: int = field(default_factory=lambda: _env_int("DASHBOARD_PAGE_SIZE", 50))

    # ── Alert Rate Limiting ──────────────────────────────────────────────
    ALERT_COOLDOWN_SECONDS: int = field(default_factory=lambda: _env_int("ALERT_COOLDOWN_SECONDS", 60))

    # ── Severity Thresholds ──────────────────────────────────────────────
    CONFIDENCE_THRESHOLD_HIGH: float = 0.8
    CONFIDENCE_THRESHOLD_MEDIUM: float = 0.5

    # ── Model ────────────────────────────────────────────────────────────
    RF_N_ESTIMATORS: int = field(default_factory=lambda: _env_int("RF_N_ESTIMATORS", 300))
    RF_MAX_DEPTH: Optional[int] = None


SETTINGS = Settings()
