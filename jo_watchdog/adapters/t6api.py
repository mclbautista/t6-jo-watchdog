"""The watchdog's seam over the t6 API: directory reads, matching and Job Order read-back.

``t6.client.Client`` is synchronous (urllib, 10 s default timeout) and the worker is
asyncio, so every call runs in a worker thread: a slow or hung API call never stalls
the poll loop or the deadman. An overall ``call_timeout`` bounds how long a caller
waits even if the socket never times out.

A failed call (``Unavailable``, ``Refused``, ``APIError``, or a call that outlives
``call_timeout``) raises ``T6ApiError``. The callers treat it as they treated a Notion
failure: a cache refresh keeps the stale snapshot, a recovery or read-back step is retried, and
a message waiting on a match is held until the API answers.

Reads are unmasked only when the API key's service holds the ``unmasked`` grant; without
it Project and Job Order names come back as placeholders and resolution fails.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import Any

from t6.client import APIError, Client, Unavailable
from t6.verbs import (
    DeliverableCandidatesInput,
    DepartmentsInput,
    Done,
    JoFindInput,
    JoGetInput,
    PersonFindInput,
    PersonListInput,
    ProjectListInput,
    ProjectResolveInput,
    Refused,
)

from jo_watchdog.adapters.protocols import PageRef
from jo_watchdog.core.matching import Candidate
from jo_watchdog.core.payload import page_url_for
from jo_watchdog.core.resolve import ContactMatch, ProjectMatch

CALL_TIMEOUT_SECONDS = 30.0
# ``person find`` takes at most this many mentions in one call.
MAX_MENTIONS = 50
# A masked read names a page "[kind N]"; a bracketed token is never a real Deliverable name.
_PLACEHOLDER = re.compile(r"\[[^\]]*\d+\]")


class T6ApiError(Exception):
    """A t6 API call failed: transport down, refused, an HTTP error, or no answer in time."""


class T6Api:
    """Async reads over a sync ``t6.client.Client``."""

    def __init__(self, client: Client, *, call_timeout: float = CALL_TIMEOUT_SECONDS):
        self._client = client
        self._call_timeout = call_timeout

    async def _call(self, method: Callable[[Any], Any], inputs: Any) -> Any:
        try:
            outcome = await asyncio.wait_for(
                asyncio.to_thread(method, inputs), timeout=self._call_timeout
            )
        except TimeoutError as exc:
            raise T6ApiError(f"t6 API gave no answer in {self._call_timeout:g} s") from exc
        except APIError as exc:
            raise T6ApiError(f"t6 API error: {exc}") from exc
        if isinstance(outcome, Unavailable):
            raise T6ApiError(f"t6 API unavailable: {outcome.reason}")
        if isinstance(outcome, Refused):
            raise T6ApiError(f"t6 API refused ({outcome.rule}): {outcome.reason}")
        if not isinstance(outcome, Done):
            raise T6ApiError(f"t6 API answered {outcome.outcome}, want done")
        return outcome.result

    # -- directory (the six-hour caches) ---------------------------------------------

    async def people(self) -> list[dict[str, Any]]:
        """``person list``: every Contact."""
        return (await self._call(self._client.person_list, PersonListInput()))["people"]

    async def projects(self) -> list[dict[str, Any]]:
        """``project list``: every Project."""
        return (await self._call(self._client.project_list, ProjectListInput()))["projects"]

    async def departments(self) -> list[str]:
        """``departments`` with MISC included, so the list is the same as Notion's."""
        result = await self._call(self._client.departments, DepartmentsInput(include_misc=True))
        return [d["name"] for d in result["departments"] if d.get("name")]

    # -- matching ----------------------------------------------------------------------

    async def find_contacts(self, mentions: list[str]) -> list[ContactMatch]:
        """``person find`` with ``mentions``: one answer for each, in order."""
        matches: list[ContactMatch] = []
        for start in range(0, len(mentions), MAX_MENTIONS):
            batch = mentions[start : start + MAX_MENTIONS]
            result = await self._call(
                self._client.person_find, PersonFindInput(mentions=batch)
            )
            replies = result["results"]
            if len(replies) != len(batch):
                raise T6ApiError(f"person find answered {len(replies)} for {len(batch)} mentions")
            matches.extend(_contact_match(reply) for reply in replies)
        return matches

    async def find_project(self, text: str) -> ProjectMatch:
        """``project resolve`` with ``in_text``: the Project in rotation, rotation-aware."""
        result = await self._call(
            self._client.project_resolve, ProjectResolveInput(text=text, in_text=True)
        )
        match = result.get("match")
        if match and not result.get("needs_review"):
            return ProjectMatch(page_id=match["page_id"], code=match.get("project_code"))
        return ProjectMatch()

    async def deliverable_candidates(self, project_page_id: str) -> list[Candidate]:
        """``deliverable candidates``: every Deliverable of the Project, in the API's order.

        The matcher compares names, so a masked or empty name fails the call; it needs the
        ``unmasked`` grant.
        """
        result = await self._call(
            self._client.deliverable_candidates,
            DeliverableCandidatesInput(project=project_page_id),
        )
        candidates: list[Candidate] = []
        for item in result["candidates"]:
            name = item.get("name")
            if not name or _PLACEHOLDER.search(name):
                raise T6ApiError(
                    "deliverable candidates returned a masked or empty name; "
                    "the API key needs the unmasked grant"
                )
            candidates.append(Candidate(page_id=item["page_id"], name=name))
        return candidates

    # -- Job Orders (recovery and read-back) -----------------------------------------

    async def find_job_orders(self, telegram_key: str) -> list[PageRef]:
        """``jo find``: Job Orders with this Telegram Key, oldest first."""
        result = await self._call(self._client.jo_find, JoFindInput(telegram_key=telegram_key))
        return [
            PageRef(id=order["page_id"], url=page_url_for(order["page_id"]))
            for order in result["job_orders"]
        ]

    async def read_job_order(self, page_id: str) -> dict[str, Any]:
        """``jo get`` by page ID: the ``job_order`` dict (name, telegram_key, date, deadline)."""
        return (await self._call(self._client.jo_get, JoGetInput(ref=page_id)))["job_order"]


def _contact_match(reply: dict[str, Any]) -> ContactMatch:
    """A match is resolved; candidates to review are ambiguous; nothing is unresolved."""
    match = reply.get("match")
    if match:
        return ContactMatch(
            status="resolved",
            method=reply.get("method"),
            page_id=match["page_id"],
            name=match.get("name") or "",
        )
    if reply.get("needs_review") and reply.get("candidates"):
        return ContactMatch(status="ambiguous")
    return ContactMatch(status="unresolved")
