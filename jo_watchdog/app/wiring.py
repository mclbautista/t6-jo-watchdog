"""Production adapter and worker construction, shared by ``run`` and ``smoke.py``.

The smoke check must exercise the same wiring the service runs: caches, the Jev
matcher and the Deliverable source all present. Both entry points build their
worker here, and a caller replaces individual adapters (for example a recording
Telegram) rather than assembling a worker by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from jo_watchdog.app.caches import CacheManager
from jo_watchdog.app.config import (
    Config,
    deepseek_token,
    gateway_secret,
    notion_token,
    openrouter_token,
    t6_api_secret,
    t6_api_url,
    telegram_token,
)
from jo_watchdog.app.worker import Worker
from jo_watchdog.ledger.db import Ledger


@dataclass
class Adapters:
    """The five external adapters; ``closers`` release whatever the real ones opened."""

    telegram: object
    notion: object
    t6api: object
    classifier: object
    matcher: object | None
    closers: list = field(default_factory=list)

    def with_(self, **replacements) -> Adapters:
        """A copy with some adapters swapped; the swapped-out real ones stay closable."""
        return replace(self, **replacements)

    async def close(self) -> None:
        for closer in self.closers:
            await closer()


def build_adapters(config: Config) -> Adapters:
    """The real aiogram, httpx, t6 API, DeepSeek and Jev adapters, from the environment."""
    from t6.client import Client

    from jo_watchdog.adapters.deepseek import DeepSeekClassifier
    from jo_watchdog.adapters.jev import JevMatcher
    from jo_watchdog.adapters.notion import NotionAdapter
    from jo_watchdog.adapters.t6api import T6Api

    notion = NotionAdapter(notion_token())
    t6api = T6Api(Client(t6_api_url(), t6_api_secret()))
    classifier = DeepSeekClassifier(
        deepseek_token(),
        base_url=config.deepseek_base_url,
        model=config.deepseek_model,
        timeout_seconds=config.deepseek_timeout_seconds,
    )
    matcher = JevMatcher(
        openrouter_token(),
        url=config.jev_url,
        model=config.jev_model,
        timeout=config.jev_timeout_seconds,
    )

    telegram, close_telegram = _build_telegram(config)
    return Adapters(
        telegram=telegram,
        notion=notion,
        t6api=t6api,
        classifier=classifier,
        matcher=matcher,
        closers=[notion.close, classifier.close, matcher.close, close_telegram],
    )


def _build_telegram(config: Config):
    """The Telegram client for ``telegram_mode`` and the coroutine that closes it."""
    if config.telegram_mode == "gateway":
        from jo_watchdog.adapters.gateway import GatewayTelegramClient

        client = GatewayTelegramClient(config.gateway_url, gateway_secret())
        return client, client.close

    from aiogram import Bot

    from jo_watchdog.adapters.telegram import AiogramTelegramClient

    bot = Bot(token=telegram_token())

    async def close_bot() -> None:
        await bot.session.close()

    return AiogramTelegramClient(bot), close_bot


def build_worker(config: Config, ledger: Ledger, adapters: Adapters, **worker_options) -> Worker:
    """The production worker: caches loaded; Notion writes Job Orders and lists Deliverables."""
    caches = CacheManager(ledger=ledger, directory=adapters.t6api, config=config)
    caches.load_persisted()  # last good snapshots first, then the live refresh
    return Worker(
        ledger=ledger,
        telegram=adapters.telegram,
        notion=adapters.notion,
        job_orders=adapters.t6api,
        classifier=adapters.classifier,
        config=config,
        caches=caches,
        matcher=adapters.matcher,
        deliverables=adapters.notion,
        **worker_options,
    )


def missing_wiring(worker: Worker) -> list[str]:
    """Names of the production collaborators this worker lacks (empty when fully wired)."""
    return [
        name
        for name in ("caches", "matcher", "deliverables")
        if getattr(worker, name) is None
    ]
