"""The watchdog's seam over the t6 API: directory reads and Job Order find and read-back.

``t6.client.Client`` is synchronous (urllib, 10 s default timeout) and the worker is
asyncio, so every call runs in a worker thread: a slow or hung API call never stalls
the poll loop or the deadman. An overall ``call_timeout`` bounds how long a caller
waits even if the socket never times out.

A failed call (``Unavailable``, ``Refused``, ``APIError``, or a call that outlives
``call_timeout``) raises ``T6ApiError``. The callers treat it as they treated a Notion
failure: a cache refresh keeps the stale snapshot, a recovery or read-back step is retried.

Reads are unmasked only when the API key's service holds the ``unmasked`` grant; without
it Project and Job Order names come back as placeholders and resolution fails.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from t6.client import APIError, Client, Unavailable
from t6.verbs import (
    DepartmentsInput,
    Done,
    JoFindInput,
    JoGetInput,
    PersonListInput,
    ProjectListInput,
    Refused,
)

from jo_watchdog.adapters.protocols import PageRef
from jo_watchdog.core.payload import page_url_for

CALL_TIMEOUT_SECONDS = 30.0


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
