"""Extraction through the t6 API's ``llm run``: the watchdog calls no LLM and holds no LLM key.

The API masks the registered names (and credentials) in the input before the provider sees
them, puts the real names back into the output, and classes every failure. This adapter
builds the request (prompt, schema, few-shot ``examples``), runs the synchronous
``t6.client.Client`` in a worker thread, and maps what comes back onto the ledger's six
outcomes. It never raises: a call that cannot be made or answered is an ``api_error`` or
``timeout`` result, which the worker records as one failed attempt (held, then retried on the
ledger's schedule, nothing raised until extraction succeeds).

``failure`` and ``retry_after`` of a refusal carry the provider's class and ``Retry-After``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import ValidationError
from t6.client import APIError, Client, Unavailable
from t6.verbs import Done, LLMRunInput, Refused

from jo_watchdog.adapters.protocols import ClassifierOutcome, ClassifyResult
from jo_watchdog.core.extraction import (
    PROMPT_TAG,
    PROMPT_TEMPLATE,
    Extraction,
    build_user_message,
    fewshot_examples,
)

TASK_NAME = "jo_watchdog_extract"
# Slack on top of the client's own socket timeout for the thread to come back.
WAIT_MARGIN_SECONDS = 5.0

# ``llm run`` failure classes (``output_valid`` false, or a refusal) -> the ledger's outcomes.
_FAILURES: dict[str, ClassifierOutcome] = {
    "timeout": "timeout",
    "api_error": "api_error",
    "empty": "empty_content",
    "invalid_json": "invalid_json",
    "schema_violation": "schema_violation",
    # The reply held a placeholder the API did not issue, or a registered name unmasked:
    # the output is unusable, like one that breaks the schema.
    "unresolved_placeholder": "schema_violation",
    "leak": "schema_violation",
}


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class T6LlmClassifier:
    def __init__(self, client: Client, *, model_hint: str, timeout_seconds: float = 70):
        self._client = client
        self._model_hint = model_hint
        self._wait = timeout_seconds + WAIT_MARGIN_SECONDS

    def _request(self, text: str, origin_lines: list[str] | None) -> LLMRunInput:
        return LLMRunInput(
            task_name=TASK_NAME,
            prompt_template=PROMPT_TEMPLATE,
            output_schema=Extraction.model_json_schema(),
            input_text=build_user_message(text, origin_lines),
            masking_preset="names",
            model_hint=self._model_hint,
            temperature_zero=True,
            examples=fewshot_examples(),
        )

    def _result(self, outcome: ClassifierOutcome, started: float, **fields) -> ClassifyResult:
        return ClassifyResult(
            outcome=outcome,
            requested_model=self._model_hint,
            prompt_tag=PROMPT_TAG,
            latency_ms=round((time.monotonic() - started) * 1000),
            **fields,
        )

    async def classify(
        self, text: str, origin_lines: list[str] | None = None
    ) -> ClassifyResult:
        started = time.monotonic()
        try:
            outcome = await asyncio.wait_for(
                asyncio.to_thread(self._client.llm_run, self._request(text, origin_lines)),
                timeout=self._wait,
            )
        except TimeoutError:
            return self._result("timeout", started, error="no answer from the t6 API in time")
        except APIError as exc:
            return self._result("api_error", started, error=str(exc)[:500])

        if isinstance(outcome, Unavailable):
            timed_out = "timed out" in outcome.reason
            return self._result(
                "timeout" if timed_out else "api_error",
                started,
                error=f"t6 API unavailable: {outcome.reason}"[:500],
            )
        if isinstance(outcome, Refused):
            failure = _FAILURES.get(outcome.failure or "", "api_error")
            return self._result(
                failure,
                started,
                error=f"refused ({outcome.rule}): {outcome.reason}"[:500],
                retry_after=outcome.retry_after,
            )
        if not isinstance(outcome, Done):
            return self._result("api_error", started, error=f"t6 API answered {outcome.outcome}")
        return self._done(outcome.result, started)

    def _done(self, run: dict[str, Any], started: float) -> ClassifyResult:
        tokens = run.get("tokens") if isinstance(run.get("tokens"), dict) else {}
        model = run.get("model")
        trace = {
            "returned_model": model if isinstance(model, str) else None,
            "input_tokens": _count(tokens.get("prompt")),
            "output_tokens": _count(tokens.get("completion")),
        }
        if not run.get("output_valid", True):
            failure = run.get("failure") or ""
            return self._result(
                _FAILURES.get(failure, "api_error"), started, error=failure or "invalid", **trace
            )
        try:
            extraction = Extraction.model_validate(run.get("output"))
        except ValidationError as exc:
            return self._result("schema_violation", started, error=str(exc)[:1000], **trace)
        return self._result("ok", started, extraction=extraction, **trace)
