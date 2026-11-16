"""Adapter contracts, expressed as plain models and Protocols.

The worker depends only on these shapes, so tests substitute fake adapters and
the real adapters stay behind the seam (aiogram/httpx are never imported by the
worker). ``IncomingUpdate`` is the normalized shape the real Telegram adapter
maps an aiogram ``Update`` into.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from jo_watchdog.core.extraction import Extraction
from jo_watchdog.core.matching import Candidate, MatchResult
from jo_watchdog.core.resolve import ContactMatch, ProjectMatch


class IncomingUpdate(BaseModel):
    """A normalized Telegram update, decoupled from aiogram's model tree."""

    update_id: int
    chat_id: int
    message_id: int
    message_thread_id: int | None = None
    text: str = ""
    raw: dict[str, Any]
    # An ``edited_message`` update (#24): ``raw`` holds ``edited_message``, and
    # ``edit_date`` is Telegram's unix time of the edit.
    edited: bool = False
    edit_date: int | None = None


class GatewayRef(BaseModel):
    """What a gateway client returns from ``send_message``: the post is queued, not delivered.

    The ref is the gateway's own number, never a Telegram message ID. A client that
    returns a plain ``int`` has delivered the post and that int is the real ID (#109).
    """

    ref: int


class NotifyStatus(BaseModel):
    """The gateway's answer to ``GET /notify/{ref}``.

    ``unknown`` is the client's reading of a 404: the gateway has forgotten the ref.
    """

    state: Literal["queued", "sent", "failed", "unknown"]
    telegram_message_id: int | None = None
    error: str | None = None


class PageRef(BaseModel):
    """The reference Notion returns after creating a page."""

    id: str
    url: str


ClassifierOutcome = Literal[
    "ok", "invalid_json", "schema_violation", "empty_content", "timeout", "api_error"
]


class ClassifyResult(BaseModel):
    """One classify/extract call: the validated result or its failure class, plus trace."""

    outcome: ClassifierOutcome
    extraction: Extraction | None = None
    error: str | None = None
    requested_model: str
    returned_model: str | None = None
    prompt_tag: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None
    retry_after: float | None = None  # seconds, from the provider's Retry-After header


@runtime_checkable
class TelegramClient(Protocol):
    async def get_updates(self, offset: int, timeout: int) -> list[IncomingUpdate]: ...

    async def set_reaction(self, chat_id: int, message_id: int, emoji: str) -> None: ...

    async def send_message(self, chat_id: int, topic_id: int, text: str) -> int:
        """Post Telegram-HTML ``text`` to a forum topic; return the new message id.

        A client that sets ``accepts_idempotency_key`` (the gateway client, #95) also
        takes ``idempotency_key=`` and returns a ``GatewayRef`` (queued, not delivered)
        instead; ``notify_status`` then reports the real message id.
        """
        ...


@runtime_checkable
class NotionClient(Protocol):
    """The Job Order write. Reads of Job Orders go through ``JobOrderReader``."""

    async def create_page(self, payload: dict) -> PageRef: ...


@runtime_checkable
class JobOrderReader(Protocol):
    """Job Order recovery and read-back, over the t6 API (``jo find``, ``jo get``)."""

    async def find_job_orders(self, telegram_key: str) -> list[PageRef]:
        """Job Orders whose Telegram Key equals ``telegram_key``, oldest first.

        Reserved for recovering from an uncertain write; normal success reads
        back by id.
        """
        ...

    async def read_job_order(self, page_id: str) -> dict:
        """The Job Order on a page: ``name``, ``telegram_key``, ``date``, ``deadline``."""
        ...


@runtime_checkable
class DirectoryReader(Protocol):
    """The read side the six-hour caches need, over the t6 API."""

    async def people(self) -> list[dict[str, Any]]:
        """``person list`` items."""
        ...

    async def projects(self) -> list[dict[str, Any]]:
        """``project list`` items."""
        ...

    async def departments(self) -> list[str]:
        """Department names, MISC included."""
        ...


@runtime_checkable
class ContactMatcher(Protocol):
    """Contact matching by the t6 API (``person find``)."""

    async def find_contacts(self, mentions: list[str]) -> list[ContactMatch]:
        """One answer for each mention, in order, from one call.

        Raises ``T6ApiError`` when the API cannot answer; the caller holds the message.
        """
        ...


@runtime_checkable
class ProjectMatcher(Protocol):
    """Project Code lookup by the t6 API (``project resolve`` in text)."""

    async def find_project(self, text: str) -> ProjectMatch:
        """The Project in rotation that the code in ``text`` names, or an empty match.

        Raises ``T6ApiError`` when the API cannot answer; the caller holds the message.
        """
        ...


@runtime_checkable
class Matcher(ContactMatcher, ProjectMatcher, Protocol):
    """Both lookups; the t6 API seam provides them."""


@runtime_checkable
class Classifier(Protocol):
    async def classify(self, text: str, origin_lines: list[str] | None) -> ClassifyResult:
        """Extract one message. The t6 API masks names and credentials; ``text`` is as posted."""
        ...


@runtime_checkable
class DeliverableMatcher(Protocol):
    """Answers one match/maybe/no decision per candidate (#26)."""

    async def match(
        self, state: str, project_name: str, candidates: list[Candidate]
    ) -> MatchResult: ...


@runtime_checkable
class DeliverableSource(Protocol):
    """Live, project-scoped Deliverable candidates (#26)."""

    async def deliverable_candidates(self, project_page_id: str) -> list[Candidate]: ...
