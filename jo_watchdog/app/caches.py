"""Six-hourly People, Projects and Department caches, persisted in the ledger.

The two lists are read through the t6 API (``person list`` and ``project list``).

Every successful, non-empty refresh is written to ``cache_snapshot`` and becomes
the live snapshot. On restart the latest persisted snapshot loads before the
first live refresh, so an API outage at boot does not blank every relation. A
failed refresh writes nothing, keeps the previous snapshot, and reports a stale
cache only once it is older than ``cache_stale_alert_hours``.

If a kind has never loaded, ``ready`` is False and planning must hold (the
message stays put, no retry attempt is spent and no unresolved finding is made).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel

from jo_watchdog.adapters.protocols import DirectoryReader, Matcher
from jo_watchdog.app.config import Config
from jo_watchdog.core.resolve import (
    MentionEntity,
    PeopleResolution,
    Person,
    ProjectEntry,
    ProjectResolution,
    mention_results,
    merge_leads,
    normalize_handle,
    people_mentions,
    people_resolution,
    resolve_project,
)
from jo_watchdog.ledger.db import Ledger

log = logging.getLogger("jo_watchdog.caches")

KINDS = ("people", "projects")
# Bump when the cached payload shape changes; older snapshots are then ignored.
FORMAT_VERSION = 2

# Live Notion property names (schema is the contract). The Notion page parsers below are
# no longer used by the service; the smoke tuning check and recorded-page tests still read pages.
CONTACT_NAME = "Name"
CONTACT_USER_ID = "TelegramUserID"
CONTACT_HANDLE = "TelegramHandle"
CONTACT_ALIASES = "Aliases"
PROJECT_NAME = "Name"
JO_DEPARTMENT = "Department"


class StaleCache(BaseModel):
    """A cache older than the alert threshold after a failed refresh (rendered by #23)."""

    kind: str
    fetched_at: str
    hours_stale: float
    error: str


# -- Notion page parsing ----------------------------------------------------


def _plain(prop: dict[str, Any] | None) -> str:
    if not prop:
        return ""
    kind = prop.get("type")
    if kind in ("title", "rich_text"):
        return "".join(part.get("plain_text", "") for part in prop.get(kind) or []).strip()
    if kind == "select":
        return ((prop.get("select") or {}).get("name") or "").strip()
    return ""


def person_from_page(page: dict[str, Any]) -> Person:
    props = page.get("properties", {})
    return Person(
        page_id=page["id"],
        name=_plain(props.get(CONTACT_NAME)),
        telegram_user_id=_plain(props.get(CONTACT_USER_ID)),
        telegram_handle=_plain(props.get(CONTACT_HANDLE)),
        aliases=[a.strip() for a in _plain(props.get(CONTACT_ALIASES)).split(",") if a.strip()],
    )


def project_from_page(page: dict[str, Any]) -> ProjectEntry:
    props = page.get("properties", {})
    return ProjectEntry(
        page_id=page["id"],
        name=_plain(props.get(PROJECT_NAME)),
    )


def person_from_api(item: dict[str, Any]) -> Person:
    """A ``person list`` item. Aliases are a list."""
    user_id = item.get("telegram_user_id")
    return Person(
        page_id=item["page_id"],
        name=item.get("name") or "",
        telegram_user_id="" if user_id is None else str(user_id),
        telegram_handle=item.get("telegram_handle") or "",
        aliases=list(item.get("aliases") or ()),
    )


def project_from_api(item: dict[str, Any]) -> ProjectEntry:
    """A ``project list`` item."""
    return ProjectEntry(page_id=item["page_id"], name=item.get("name") or "")


def department_options(data_source: dict[str, Any]) -> list[str]:
    prop = data_source.get("properties", {}).get(JO_DEPARTMENT) or {}
    options = (prop.get("select") or {}).get("options") or []
    return [o["name"] for o in options if o.get("name")]


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


# -- the cache manager --------------------------------------------------------


class CacheManager:
    """Owns the live snapshots and their refresh schedule."""

    def __init__(
        self,
        *,
        ledger: Ledger,
        directory: DirectoryReader,
        matcher: Matcher,
        config: Config,
        clock: Callable[[], datetime] = _utcnow,
    ):
        self.ledger = ledger
        self.directory = directory
        self.matcher = matcher
        self.config = config
        self.clock = clock
        self.people: list[Person] | None = None
        self.projects: list[ProjectEntry] | None = None
        self.fetched_at: dict[str, str] = {}
        self.last_attempt: datetime | None = None
        self._stale_reported: set[str] = set()

    # -- loading -------------------------------------------------------------

    def load_persisted(self) -> None:
        """Load the latest persisted snapshot per kind (before the first live refresh)."""
        for kind in KINDS:
            snapshot = self.ledger.latest_cache_snapshot(kind)
            if snapshot is None or snapshot["format_version"] != FORMAT_VERSION:
                continue
            if not snapshot["payload"]:
                continue
            self._apply(kind, snapshot["payload"], snapshot["fetched_at"])

    def _apply(self, kind: str, payload: list, fetched_at: str) -> None:
        if kind == "people":
            self.people = [Person(**p) for p in payload]
        elif kind == "projects":
            self.projects = [ProjectEntry(**p) for p in payload]
        self.fetched_at[kind] = fetched_at

    async def _fetch(self, kind: str) -> list:
        if kind == "people":
            return [
                person_from_api(p).model_dump(mode="json") for p in await self.directory.people()
            ]
        return [
            project_from_api(p).model_dump(mode="json") for p in await self.directory.projects()
        ]

    async def refresh(self) -> list[StaleCache]:
        """Refresh every kind; return stale-cache events for failed refreshes past the threshold."""
        now = self.clock()
        self.last_attempt = now
        stale: list[StaleCache] = []
        for kind in KINDS:
            try:
                payload = await self._fetch(kind)
                if not payload:
                    raise ValueError(f"{kind} refresh returned no items")
            except Exception as exc:  # any failure keeps the previous snapshot
                log.warning("cache refresh failed for %s at %s: %s", kind, _iso(now), exc)
                event = self._stale_event(kind, now, str(exc))
                if event is not None:
                    stale.append(event)
                continue
            fetched_at = _iso(now)
            self.ledger.save_cache_snapshot(kind, FORMAT_VERSION, payload, len(payload), fetched_at)
            self._apply(kind, payload, fetched_at)
            self._stale_reported.discard(kind)
        return stale

    def _stale_event(self, kind: str, now: datetime, error: str) -> StaleCache | None:
        fetched_at = self.fetched_at.get(kind)
        if fetched_at is None or kind in self._stale_reported:
            return None
        age = now - _parse_iso(fetched_at)
        if age <= timedelta(hours=self.config.cache_stale_alert_hours):
            return None
        self._stale_reported.add(kind)
        return StaleCache(
            kind=kind,
            fetched_at=fetched_at,
            hours_stale=round(age.total_seconds() / 3600, 1),
            error=error,
        )

    def refresh_due(self) -> bool:
        if self.last_attempt is None:
            return True
        return self.clock() - self.last_attempt >= timedelta(hours=self.config.cache_refresh_hours)

    async def refresh_if_due(self) -> list[StaleCache]:
        if not self.refresh_due():
            return []
        return await self.refresh()

    # -- resolution ------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """True once every kind has a loaded snapshot; until then planning holds."""
        return self.people is not None and self.projects is not None

    async def resolve_people(
        self,
        assignees_raw: str | None,
        cc_raw: str | None,
        entities: list[MentionEntity] | None = None,
    ) -> PeopleResolution:
        """Match the message's mentions with one ``person find`` call.

        Raises ``T6ApiError`` when the API cannot answer: the worker holds the message.
        """
        if self.people is None:
            raise RuntimeError("People cache has not loaded; hold planning")
        mentions = people_mentions(assignees_raw, cc_raw, entities or [])
        asked = [m for m in mentions if not m.hedged]
        # One call: the mentions, then the words of each name with no separator (``Mira Lena``).
        queries = [m.query for m in asked] + [w for m in asked for w in m.words]
        answers = await self.matcher.find_contacts(queries) if queries else []
        matches = iter(answers[: len(asked)])
        word_matches = iter(answers[len(asked) :])
        groups = [
            mention_results(
                m, None if m.hedged else next(matches), [next(word_matches) for _ in m.words]
            )
            for m in mentions
        ]
        return people_resolution(merge_leads(mentions, groups))

    def contact_for_telegram_user(self, user_id: int | None, username: str | None) -> str | None:
        """The one Contact whose TelegramUserID or TelegramHandle matches, else None.

        Used only for the source block's bracketed name (#24); never a relation.
        """
        people = self.people or []
        found = []
        if user_id is not None:
            found = [p for p in people if p.telegram_user_id.strip() == str(user_id)]
        handle = normalize_handle(username or "")
        if not found and handle:
            found = [p for p in people if normalize_handle(p.telegram_handle) == handle]
        return found[0].name if len(found) == 1 else None

    async def resolve_project(
        self, header_text: str | None, project_raw: str | None
    ) -> ProjectResolution:
        """The Project the API finds in the posted header line (``project resolve``), else in
        the model's reading of the project (``project_raw``).

        Raises ``T6ApiError`` when the API cannot answer: the worker holds the message.
        """
        if self.projects is None:
            raise RuntimeError("Projects cache has not loaded; hold planning")
        match = None
        for text in (header_text, project_raw):
            if text and text.strip():
                match = await self.matcher.find_project(text.strip())
                if match.page_id:
                    break
        return resolve_project((project_raw or "").strip() or header_text, match)

    def project_name(self, page_id: str | None) -> str | None:
        """The cached name of a resolved Project page, for the matcher's ``state``."""
        for project in self.projects or []:
            if project.page_id == page_id:
                return project.name
        return None
