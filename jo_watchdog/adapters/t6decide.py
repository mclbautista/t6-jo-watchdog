"""Deliverable matching through the t6 API's ``llm decide``: the watchdog calls no provider.

One request per Job Order: the text and the Project name go in ``state``, the question names
one candidate Deliverable, and the API asks the Decisions provider for ``match``, ``maybe`` or
``no`` with probabilities for each. The API masks the registered names in the state and in the
candidate names before the provider sees them, and answers by candidate index; the page ids
stay here. ``decide`` and its thresholds stay in the watchdog (``core/matching.py``).

A call that cannot be made or answered is a ``MatcherError``, which the match stage records as
one failed attempt. ``retry_after`` of a rate-limit refusal is capped at 30 seconds.
"""

from __future__ import annotations

import asyncio
from typing import Any

from t6.client import APIError, Client, Unavailable
from t6.verbs import Done, LLMDecideInput, Refused

from jo_watchdog.core.matching import (
    Candidate,
    CandidateDecision,
    MatchResult,
    sums_to_one,
)

TASK_NAME = "jo_watchdog_match"
PROMPT_TAG = "match-v1"
# The API puts each candidate's name where the marker stands.
QUESTION = "Does this Job Order belong to the Deliverable '{candidate}'?"
# The model's default is the API's: no model hint is sent.
REQUESTED_MODEL = "t6-api-default"

# Part of the prompt: change the wording only together with PROMPT_TAG.
CRITERIA = {
    "match": "The Job Order is clearly work for this Deliverable.",
    "maybe": "The Job Order could be for this Deliverable, but the text doesn't make it clear.",
    "no": "The Job Order is not for this Deliverable.",
}

# The inline retry after a rate limit never waits longer than this.
MAX_RETRY_AFTER_SECONDS = 30.0
# Slack on top of the client's own socket timeout for the thread to come back.
WAIT_MARGIN_SECONDS = 5.0


class MatcherError(Exception):
    """One failed matcher call: an unreachable API, a refusal, or an invalid answer."""

    def __init__(self, failure: str, message: str, *, retry_after: float | None = None):
        super().__init__(f"{failure}: {message}")
        self.failure = failure
        self.message = message
        self.retry_after = retry_after


def _capped(seconds: float | None) -> float | None:
    if seconds is None:
        return None
    return max(0.0, min(float(seconds), MAX_RETRY_AFTER_SECONDS))


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class T6DecideMatcher:
    """``DeliverableMatcher`` backed by ``llm decide``."""

    prompt_tag = PROMPT_TAG

    def __init__(self, client: Client, *, timeout_seconds: float = 70):
        self._client = client
        self._wait = timeout_seconds + WAIT_MARGIN_SECONDS

    @staticmethod
    def request(state: str, project_name: str, candidates: list[Candidate]) -> LLMDecideInput:
        return LLMDecideInput(
            task_name=TASK_NAME,
            state={"project": project_name, "job_order": state},
            question=QUESTION,
            criteria=dict(CRITERIA),
            candidates=[c.name for c in candidates],
        )

    async def match(
        self, state: str, project_name: str, candidates: list[Candidate]
    ) -> MatchResult:
        request = self.request(state, project_name, candidates)
        try:
            outcome = await asyncio.wait_for(
                asyncio.to_thread(self._client.llm_decide, request), timeout=self._wait
            )
        except TimeoutError as exc:
            raise MatcherError("timeout", "no answer from the t6 API in time") from exc
        except APIError as exc:
            raise MatcherError("api_error", f"API error: HTTP {exc.status}") from exc

        if isinstance(outcome, Unavailable):
            failure = "timeout" if "timed out" in outcome.reason else "transport"
            raise MatcherError(failure, f"t6 API unavailable: {outcome.reason}"[:500])
        if isinstance(outcome, Refused):
            raise MatcherError(
                outcome.failure or outcome.rule,
                f"refused ({outcome.rule})",
                retry_after=_capped(outcome.retry_after),
            )
        if not isinstance(outcome, Done):
            raise MatcherError("api_error", f"t6 API answered {outcome.outcome}")
        return self._result(outcome.result, candidates)

    @staticmethod
    def _result(run: dict[str, Any], candidates: list[Candidate]) -> MatchResult:
        answers = run.get("answers")
        if not isinstance(answers, list) or sorted(
            a.get("index") if isinstance(a, dict) else None for a in answers
        ) != list(range(len(candidates))):
            raise MatcherError("invalid_body", "the answers do not match the candidates")
        by_index = {a["index"]: a for a in answers}
        decisions = []
        for i, candidate in enumerate(candidates):
            p = by_index[i].get("probabilities") or {}
            try:
                match, maybe, no = (float(p[k]) for k in ("match", "maybe", "no"))
            except (KeyError, TypeError, ValueError) as exc:
                raise MatcherError("invalid_body", f"probabilities for {i} are incomplete") from exc
            if not all(0 <= x <= 1 for x in (match, maybe, no)) or not sums_to_one(
                match, maybe, no
            ):
                raise MatcherError("invalid_body", f"probabilities for {i} do not sum to 1")
            decisions.append(
                CandidateDecision(page_id=candidate.page_id, p_match=match, p_maybe=maybe, p_no=no)
            )
        tokens = run.get("tokens") if isinstance(run.get("tokens"), dict) else {}
        model = run.get("model")
        return MatchResult(
            decisions=decisions,
            requested_model=REQUESTED_MODEL,
            returned_model=model if isinstance(model, str) else None,
            input_tokens=_count(tokens.get("prompt")),
            output_tokens=_count(tokens.get("completion")),
        )
