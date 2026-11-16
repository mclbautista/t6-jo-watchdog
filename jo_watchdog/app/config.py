"""Configuration: non-secret values from config.yaml, secrets from env.

A bad non-secret value fails boot (Pydantic validation) rather than failing a
message later. Secrets never appear in the config file or in logs.

Every key in the configuration reference (issue #27) is declared here. A few
keys with no sensible default (page ids, handles) are optional on the model so
tests and partial tools can build a ``Config``, and ``require_live_keys`` makes
them mandatory when the service boots.
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

_HANDLE_RE = re.compile(r"^@?[A-Za-z0-9_]{5,32}$")
_HHMM_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")

_NON_EMPTY_KEYS = ("llm_model_hint", "ledger_path")

# Keys the running service cannot do without; checked by ``require_live_keys``.
LIVE_REQUIRED_KEYS = (
    "standing_cc_telegram_handle",
    "owner_telegram_handle",
)

# What the service needs from its environment (T6_API_URL is not secret, but it is just as
# required: the reads go through the t6 API).
SECRET_KEYS = (
    "TELEGRAM_BOT_TOKEN",
    "T6_API_URL",
    "T6_API_SECRET",
)

# Gateway mode (#95): the module secret replaces the Telegram token.
GATEWAY_SECRET_KEY = "WATCHDOG_SECRET"


class FilterEntry(BaseModel):
    """One chat (and optionally one topic) and the t6-core filter rule that applies to it.

    The rule's question and thresholds live in t6-core; only its name is configured here.
    """

    model_config = ConfigDict(extra="forbid")

    chat_id: int
    topic_id: int | None = None
    rule: str = Field(min_length=1, max_length=200)


class Config(BaseModel):
    """Validated runtime configuration."""

    model_config = ConfigDict(extra="forbid")

    # Telegram scope filter
    intake_chat_id: int
    intake_topic_id: int
    captured_reaction: str = "👍"
    flagged_reaction: str = "🤔"

    # Ops topic (#23): where capture reports and system messages go
    ops_chat_id: int
    ops_topic_id: int
    standing_cc_telegram_handle: str | None = None
    header_date_tolerance_days: int = Field(default=1, ge=0)
    # #101: alert ops when intake chat message ids jump (messages that never arrived)
    message_gap_alerts: bool = True

    # Telegram transport (#95): ``direct`` polls Telegram with the bot token; ``gateway``
    # reads and posts through the Sixto gateway with ``WATCHDOG_SECRET`` instead.
    telegram_mode: Literal["direct", "gateway"] = "direct"
    gateway_url: str | None = None

    # Polling
    poll_timeout: int = Field(default=50, ge=1, le=50)  # Telegram long-poll seconds
    poll_interval: float = Field(default=1.0, gt=0)

    # Classify/extract through the t6 API (``llm run``): the model the API's provider list
    # names (the "latest" alias, no pin), and how long to wait for the call. The API's own
    # provider timeout is 60 s, so a shorter wait here would cut off a call that is still running.
    llm_model_hint: str = "deepseek-chat"
    llm_timeout_seconds: float = Field(default=70, ge=70)

    # Filter before extraction (#153): ``llm filter`` per message, by chat and topic. ``None``
    # filters the intake topic with the rule ``job_order``; ``[]`` filters nothing. Count first:
    # while ``filter_drops`` is false the verdict is stored and every message is still extracted.
    filters: list[FilterEntry] | None = None
    filter_drops: bool = False

    # Live resolution caches (#22)
    cache_refresh_hours: float = Field(default=6, gt=0)
    cache_stale_alert_hours: float = Field(default=24, gt=0)

    # Ledger
    ledger_path: str = "./data/ledger.db"

    # The Department rules run in the t6 API (#151). An old config.yaml that still has the
    # alias map boots; the value is ignored (``T6_DEPARTMENT_ALIASES`` is the API's).
    department_aliases: dict[str, str] | None = None

    # How long to wait for the page of a raised Job Order before the retry schedule applies.
    raise_poll_seconds: float = Field(default=60, ge=0)

    # Reliability and heartbeat (#25)
    owner_telegram_handle: str | None = None
    retry_schedule_seconds: list[int] = Field(default_factory=lambda: [30, 120, 300, 600])
    deadman_stale_minutes: int = Field(default=10, gt=0)
    heartbeat_time: str = "07:00"
    timezone: str = "Asia/Manila"

    # Deliverable matching (the t6 API's ``llm decide``, #26, #147)
    t_link: float = Field(default=0.90, ge=0, le=1)
    t_hint: float = Field(default=0.50, ge=0, le=1)
    deliverable_link_enabled: bool = False

    # Rollback control (#27)
    dry_run: bool = False

    @field_validator("captured_reaction", "flagged_reaction")
    @classmethod
    def _non_empty_reaction(cls, value: str, info: ValidationInfo) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must be an emoji")
        return value

    @field_validator("standing_cc_telegram_handle", "owner_telegram_handle")
    @classmethod
    def _telegram_handle(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        value = value.strip()
        if not _HANDLE_RE.match(value):
            raise ValueError(f"not a Telegram handle: {value!r}")
        return value if value.startswith("@") else f"@{value}"

    @field_validator("heartbeat_time")
    @classmethod
    def _hhmm(cls, value: str) -> str:
        if not _HHMM_RE.match(value):
            raise ValueError("heartbeat_time must be HH:MM (24-hour)")
        return value

    @field_validator("timezone")
    @classmethod
    def _known_zone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone: {value!r}") from exc
        return value

    @field_validator("retry_schedule_seconds")
    @classmethod
    def _retry_schedule(cls, value: list[int]) -> list[int]:
        if not value or any(step <= 0 for step in value):
            raise ValueError("retry_schedule_seconds must be a non-empty list of positive ints")
        return value

    def filter_rule(self, chat_id: int, topic_id: int | None) -> str | None:
        """The filter rule for a chat and topic, or ``None`` when it is not filtered."""
        entries = self.filters
        if entries is None:
            entries = [
                FilterEntry(
                    chat_id=self.intake_chat_id, topic_id=self.intake_topic_id, rule="job_order"
                )
            ]
        for entry in entries:
            if entry.chat_id == chat_id and entry.topic_id in (None, topic_id):
                return entry.rule
        return None

    @model_validator(mode="after")
    def _cross_field(self) -> Config:
        if not self.intake_chat_id or not self.intake_topic_id:
            raise ValueError("intake_chat_id and intake_topic_id must be set")
        if not self.ops_chat_id or not self.ops_topic_id:
            raise ValueError("ops_chat_id and ops_topic_id must be set")
        if self.t_hint > self.t_link:
            raise ValueError("t_hint must not exceed t_link")
        if self.telegram_mode == "gateway":
            parts = urlsplit(self.gateway_url or "")
            if parts.scheme not in ("http", "https") or not parts.netloc:
                raise ValueError("gateway_url must be an http(s) URL when telegram_mode is gateway")
        if self.cache_stale_alert_hours < self.cache_refresh_hours:
            raise ValueError("cache_stale_alert_hours must be at least cache_refresh_hours")
        return self


def _production_problems(config: Config) -> list[str]:
    """Why ``config`` is not fit to boot against live providers.

    Messages name the key and the rule, never the value.
    """
    problems = [
        f"{key} is missing"
        for key in LIVE_REQUIRED_KEYS
        if getattr(config, key) in (None, "", 0)
    ]
    for key in _NON_EMPTY_KEYS:
        if not getattr(config, key).strip():
            problems.append(f"{key} must not be empty")
    for key in ("intake_topic_id", "ops_topic_id"):
        if getattr(config, key) <= 0:
            problems.append(f"{key} must be a positive topic id")
    # Chat ids are not topic ids: supergroups are legitimately negative, so only
    # zero is rejected (the model already refuses it).
    return problems


def require_live_keys(config: Config) -> None:
    """Fail boot when the config is unfit to run against live providers.

    Checks structure only, so the example file passes it; ``placeholder_keys``
    separately flags the FILL IN values that must be replaced.
    """
    problems = _production_problems(config)
    if problems:
        raise ValueError("config.yaml is not valid for production: " + "; ".join(problems))


def placeholder_keys(config: Config) -> list[str]:
    """Keys still holding the example file's FILL IN values (not production-ready)."""
    placeholders = []
    for key in ("standing_cc_telegram_handle", "owner_telegram_handle"):
        if str(getattr(config, key)).startswith(("@standing_cc_handle", "@owner_handle")):
            placeholders.append(key)
    if config.intake_chat_id == -1000000000000:
        placeholders.append("intake_chat_id")
    if config.ops_chat_id == -1000000000001:
        placeholders.append("ops_chat_id")
    return placeholders


# Keys renamed to role names (the standing cc person). Old files still load, with a warning.
_RENAMED_KEYS = {
    "leizette_telegram_handle": "standing_cc_telegram_handle",
}


# Keys of earlier releases that no longer do anything (the t6 API holds those rules and keys).
# An old config.yaml still loads: each is dropped and named in one warning, never its value.
REMOVED_KEYS = frozenset(
    {
        "job_orders_data_source_id",
        "contacts_data_source_id",
        "projects_data_source_id",
        "deliverables_data_source_id",
        "notion_version",
        "deepseek_base_url",
        "deepseek_model",
        "deepseek_timeout_seconds",
        "jev_url",
        "jev_model",
        "jev_timeout_seconds",
        "fuzzy_min_score",
        "fuzzy_min_lead",
        "honorifics",
        # The API adds the standing cc (``T6_STANDING_CC``); the old name of the key is ignored too.
        "standing_cc_contact_page_id",
        "leizette_contact_page_id",
    }
)


def _apply_renamed_keys(data: dict) -> dict:
    for old, new in _RENAMED_KEYS.items():
        if old in data:
            warnings.warn(
                f"config key {old!r} is renamed to {new!r}; update config.yaml",
                DeprecationWarning,
                stacklevel=3,
            )
            value = data.pop(old)
            data.setdefault(new, value)
    return data


def load_config(path: str | Path) -> Config:
    """Load non-secret config from a YAML file."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    removed = sorted(REMOVED_KEYS & data.keys())
    if removed:
        for key in removed:
            del data[key]
        logging.getLogger("jo_watchdog").warning(
            "config keys ignored (removed): %s", ", ".join(removed)
        )
    return Config(**_apply_renamed_keys(data))


def _secret(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value


def required_secrets(telegram_mode: str = "direct") -> tuple[str, ...]:
    """The secrets the given mode needs: gateway mode swaps the bot token for the module secret."""
    if telegram_mode == "gateway":
        return tuple(
            GATEWAY_SECRET_KEY if name == "TELEGRAM_BOT_TOKEN" else name for name in SECRET_KEYS
        )
    return SECRET_KEYS


def missing_secrets(telegram_mode: str = "direct") -> list[str]:
    """Names of the secrets the mode needs (``SECRET_KEYS`` by default) that are unset."""
    return [name for name in required_secrets(telegram_mode) if not os.environ.get(name)]


def telegram_token() -> str:
    return _secret("TELEGRAM_BOT_TOKEN")


def gateway_secret() -> str:
    return _secret(GATEWAY_SECRET_KEY)


def t6_api_url() -> str:
    return _secret("T6_API_URL")


def t6_api_secret() -> str:
    return _secret("T6_API_SECRET")

