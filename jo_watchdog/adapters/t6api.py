"""The watchdog's seam over the t6 API: directory reads, matching and Job Orders.

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
    JoRaiseInput,
    JoRaiseMessage,
    PersonFindInput,
    PersonListInput,
    Preview,
    ProjectListInput,
    ProjectResolveInput,
    Refused,
)

from jo_watchdog.adapters.protocols import PageRef
from jo_watchdog.core.job_order import JobOrderPlan, RaisePreview
from jo_watchdog.core.matching import Candidate
from jo_watchdog.core.payload import build_raise_fields, page_url_for
from jo_watchdog.core.resolve import ContactMatch, ProjectMatch

CALL_TIMEOUT_SECONDS = 30.0
# ``person find`` takes at most this many mentions in one call.
MAX_MENTIONS = 50
# A masked read names a page "[kind N]"; a bracketed token is never a real Deliverable name.
_PLACEHOLDER = re.compile(r"\[[^\]]*\d+\]")


class T6ApiError(Exception):
    """A t6 API call failed: transport down, refused, an HTTP error, or no answer in time."""


class ApiUnreachable(T6ApiError):
    """No answer: the transport is down or the call timed out. The only error that may wait."""


class KeyAlreadyHeld(T6ApiError):
    """``jo raise`` refused: a Job Order with this Telegram Key was already raised."""


class T6Api:
    """Async calls over a sync ``t6.client.Client``."""

    def __init__(self, client: Client, *, call_timeout: float = CALL_TIMEOUT_SECONDS):
        self._client = client
        self._call_timeout = call_timeout

    async def _call(
        self, method: Callable[[Any], Any], inputs: Any, *, expect: type = Done
    ) -> Any:
        try:
            outcome = await asyncio.wait_for(
                asyncio.to_thread(method, inputs), timeout=self._call_timeout
            )
        except TimeoutError as exc:
            raise ApiUnreachable(f"t6 API gave no answer in {self._call_timeout:g} s") from exc
        except APIError as exc:
            raise T6ApiError(f"t6 API error: HTTP {exc.status}") from exc
        if isinstance(outcome, Unavailable):
            raise ApiUnreachable(f"t6 API unavailable: {outcome.reason}")
        if isinstance(outcome, Refused):
            if outcome.rule == "telegram_key":
                raise KeyAlreadyHeld(f"t6 API refused ({outcome.rule})")
            raise T6ApiError(f"t6 API refused ({outcome.rule})")
        if not isinstance(outcome, expect):
            want = expect.model_fields["outcome"].default
            raise T6ApiError(f"t6 API answered {outcome.outcome}, want {want}")
        return outcome.result if isinstance(outcome, Done) else outcome

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

    # -- Job Orders (raise, recovery and read-back) ------------------------------------

    async def preview_job_order(self, inputs: JoRaiseInput) -> RaisePreview:
        """``jo raise`` without a confirm: what the API would write, and the token to confirm.

        With a ``message`` the API builds the Name, number, Department, deadline and both
        keys; they come back here. Nothing is written.
        """
        preview = await self._call(self._client.jo_raise, inputs, expect=Preview)
        plan = preview.plan
        return RaisePreview(
            token=preview.token,
            name=plan.get("name"),
            jo_number=plan.get("jo_number"),
            department=plan.get("department"),
            deadline=plan.get("deadline"),
            deadline_source=plan.get("deadline_source") or "blank",
            deadline_findings=plan.get("deadline_findings") or [],
            department_findings=plan.get("department_findings") or [],
            telegram_key=plan.get("telegram_key"),
            duplicate_key=plan.get("duplicate_key"),
        )

    async def confirm_job_order(self, inputs: JoRaiseInput, token: str) -> str:
        """``jo raise`` with the preview's token and the same key: the T6 Job Order ID.

        The reply has no page ID; the page appears a few seconds later (``job_order_page``).
        """
        result = await self._call(
            self._client.jo_raise, inputs.model_copy(update={"confirm": token})
        )
        return result["jo_id"]

    async def job_order_page(self, jo_id: str) -> str | None:
        """``jo get`` by T6 Job Order ID: the page ID once the page exists, else None."""
        result = await self._call(self._client.jo_get, JoGetInput(ref=jo_id))
        return result["job_order"].get("page_id") or None

    async def find_job_orders(self, chat_id: int, message_id: int) -> list[PageRef]:
        """``jo find`` by chat and message ID: Job Orders raised from this message, oldest first."""
        result = await self._call(
            self._client.jo_find, JoFindInput(chat_id=str(chat_id), message_id=str(message_id))
        )
        return [
            PageRef(id=order["page_id"], url=page_url_for(order["page_id"]))
            for order in result["job_orders"]
            if order.get("page_id")
        ]

    async def read_job_order(self, page_id: str) -> dict[str, Any]:
        """``jo get`` by page ID: the ``job_order`` dict (name, telegram_key, date, deadline)."""
        return (await self._call(self._client.jo_get, JoGetInput(ref=page_id)))["job_order"]


def raise_input(plan: JobOrderPlan, idempotency_key: str) -> JoRaiseInput:
    """The ``jo raise`` input for a stored plan.

    A plan with a ``raise_message`` sends it, so the API builds the rule fields; any other
    plan (stored before #151) sends its own fields. With no Project, ``no_project`` is set:
    nothing picks one.
    """
    fields = build_raise_fields(plan)
    message = JoRaiseMessage.model_validate(plan.raise_message) if plan.raise_message else None
    project = message.project if message else fields.get("project_id")
    return JoRaiseInput(
        message=message,
        fields=fields,
        no_project=not project,
        idempotency_key=idempotency_key,
    )


def _contact_match(reply: dict[str, Any]) -> ContactMatch:
    """A match is resolved; candidates to review are ambiguous; nothing is unresolved."""
    match = reply.get("match")
    if match:
        return ContactMatch(
            status="resolved",
            method=reply.get("method"),
            page_id=match["page_id"],
            name=match.get("name") or "",
            t6_id=match.get("t6_id"),
        )
    if reply.get("needs_review") and reply.get("candidates"):
        return ContactMatch(status="ambiguous")
    return ContactMatch(status="unresolved")
