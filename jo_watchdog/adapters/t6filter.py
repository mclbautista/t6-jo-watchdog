"""The intake filter through the t6 API's ``llm filter``: the rule's question lives in t6-core.

The watchdog names a rule (from its config), sends the message text and the text it replies to,
and gets ``pass``, ``drop`` or ``unsure``. The API masks the names. The verdict is advice; the
worker decides what a ``drop`` does. This adapter never raises and always fails open: a
refusal, an error, a timeout or an unreadable answer is the verdict ``unavailable`` with the
failure class (never any text), and the message goes on to extraction.
"""

from __future__ import annotations

import asyncio
from typing import NamedTuple

from t6.client import APIError, Client, Unavailable
from t6.verbs import Done, LLMFilterInput, Refused

VERDICTS = ("pass", "drop", "unsure")
# Slack on top of the client's own socket timeout for the thread to come back.
WAIT_MARGIN_SECONDS = 5.0


class FilterResult(NamedTuple):
    verdict: str  # pass, drop, unsure, or unavailable
    failure: str | None = None


class T6Filter:
    def __init__(
        self,
        client: Client,
        *,
        timeout_seconds: float = 70,
        wait_margin: float = WAIT_MARGIN_SECONDS,
    ):
        self._client = client
        self._wait = timeout_seconds + wait_margin

    async def filter(
        self, rule: str, text: str, reply_to_text: str | None = None
    ) -> FilterResult:
        request = LLMFilterInput(rule=rule, text=text, reply_to_text=reply_to_text)
        try:
            outcome = await asyncio.wait_for(
                asyncio.to_thread(self._client.llm_filter, request), timeout=self._wait
            )
        except TimeoutError:
            return FilterResult("unavailable", "timeout")
        except APIError:
            return FilterResult("unavailable", "api_error")
        except Exception:  # fail open on anything the client can throw
            return FilterResult("unavailable", "transport")
        if isinstance(outcome, Unavailable):
            return FilterResult(
                "unavailable", "timeout" if "timed out" in outcome.reason else "transport"
            )
        if isinstance(outcome, Refused):
            return FilterResult("unavailable", str(outcome.failure or outcome.rule)[:60])
        if not isinstance(outcome, Done):
            return FilterResult("unavailable", "api_error")
        verdict = (outcome.result or {}).get("verdict")
        if verdict not in VERDICTS:
            return FilterResult("unavailable", "invalid_body")
        return FilterResult(verdict)
