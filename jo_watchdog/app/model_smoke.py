"""Typed-model smoke checks for extraction (the t6 API's ``llm run``) and Deliverable
matching (its ``llm decide``) (#61).

``smoke.py`` is manual and calls these against the live API; the offline tests call the
same functions over a fake client. Both go through the production adapters, so a response is
judged by the same validation the worker uses. A recorder sits underneath, so what is
reported (request count) is observed, not assumed.

Nothing here claims a live rate limit unless one came back, and a live call needs the
explicit authorization that ``smoke.py`` documents.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from jo_watchdog.adapters.t6decide import MatcherError, T6DecideMatcher
from jo_watchdog.adapters.t6llm import T6LlmClassifier
from jo_watchdog.app.smoke_gate import FAIL, MANUAL, PASS, Result
from jo_watchdog.core.matching import Candidate, decide

# The three fixed regression pairs, in request order (tests/fixtures/jev/regression_pairs.json).
REGRESSION_PAIRS = ("moonglow-jo-32", "somsom-0155", "ina-toc-30")

SECRET = "hunter22"
MASK_POSITIVE = f"SOMSOM - TOC - #0163\nTask: conform reel 3\npw: {SECRET}"
MASK_NEGATIVE = "SOMSOM - TOC - #0163\nTask: conform reel 3\nPurpose: for grading"


@dataclass
class ModelCheck:
    result: Result
    lines: list[str] = field(default_factory=list)
    requests_sent: int = 0


def _fail(lines: list[str], detail: str, sent: int) -> ModelCheck:
    return ModelCheck(Result(FAIL, detail), lines, sent)


# -- Extraction through llm run -------------------------------------------------------


class RecordingClient:
    """Wraps a ``t6.client.Client`` and keeps each ``llm run`` or ``llm decide`` request."""

    def __init__(self, inner):
        self._inner = inner
        self.requests: list = []

    def llm_run(self, inputs):
        self.requests.append(inputs)
        return self._inner.llm_run(inputs)

    def llm_decide(self, inputs):
        self.requests.append(inputs)
        return self._inner.llm_decide(inputs)


async def check_extraction(client, *, model_hint: str, timeout: float) -> ModelCheck:
    """One text with a credential and one without, through ``T6LlmClassifier``.

    The API masks the input, so the credential is sent as posted: the check is that the
    reply carries it nowhere. Each reply must be a validated ``Extraction`` with the returned
    model and token counts, and each case must cost exactly one request.
    """
    recorder = RecordingClient(client)
    classifier = T6LlmClassifier(recorder, model_hint=model_hint, timeout_seconds=timeout)
    lines: list[str] = []
    problems: list[str] = []
    verdicts: list[str] = []
    for label, raw, has_secret in (
        ("mask positive", MASK_POSITIVE, True),
        ("no mask", MASK_NEGATIVE, False),
    ):
        before = len(recorder.requests)
        result = await classifier.classify(raw, [])
        if len(recorder.requests) != before + 1:
            saw = len(recorder.requests) - before
            problems.append(f"{label}: expected one request, saw {saw}")
            continue
        if result.extraction is None:
            problems.append(f"{label}: {result.outcome}: {result.error}")
            continue
        if has_secret and SECRET in result.extraction.model_dump_json():
            problems.append(f"{label}: the reply carries the credential")
        if result.requested_model != model_hint:
            problems.append(f"{label}: requested model {result.requested_model!r}")
        if not result.returned_model:
            problems.append(f"{label}: no returned model snapshot")
        if result.input_tokens is None or result.output_tokens is None:
            problems.append(f"{label}: token counts missing")
        verdicts.append(result.extraction.verdict)
        lines.append(
            f"{label}: model {result.returned_model}; tokens "
            f"{result.input_tokens}/{result.output_tokens}; {result.latency_ms} ms; "
            f"verdict {result.extraction.verdict}"
        )
    sent_count = len(recorder.requests)
    if problems:
        return _fail(lines, "; ".join(problems), sent_count)
    if any(v != "jo" for v in verdicts):
        detail = f"structured output valid but verdicts were {verdicts}; review the model"
        return ModelCheck(Result(MANUAL, detail), lines, sent_count)
    return ModelCheck(
        Result(PASS, "structured output, snapshot and tokens; credential and no-credential cases"),
        lines,
        sent_count,
    )


# -- Jev --------------------------------------------------------------------------


class PairError(ValueError):
    """The regression fixtures do not hold exactly the three required pairs."""


def load_pairs(path: Path) -> list[dict]:
    """The three fixed regression pairs, in order. Anything short of all three is an error."""
    by_name = {p["name"]: p for p in json.loads(path.read_text(encoding="utf-8"))}
    missing = [n for n in REGRESSION_PAIRS if n not in by_name]
    if missing:
        raise PairError(f"missing regression pair fixture(s): {', '.join(missing)}")
    pairs = [by_name[n] for n in REGRESSION_PAIRS]
    for pair in pairs:
        if len(pair["candidates"]) != 1:
            raise PairError(f"{pair['name']} must name exactly one Deliverable")
    return pairs


async def check_decide(
    client,
    pairs_path: Path,
    *,
    timeout: float,
    t_link: float,
    t_hint: float,
) -> ModelCheck:
    """Send each regression pair through ``T6DecideMatcher`` and report what came back.

    One ``llm decide`` request for each pair: the Job Order text goes as posted (the API
    masks it). Fails, without sending anything, when a regression fixture is missing. Fails
    on any answer the production validation rejects.
    """
    try:
        pairs = load_pairs(pairs_path)
    except (PairError, OSError, ValueError, KeyError) as exc:
        return _fail(["requests sent: 0"], f"regression fixtures: {exc}", 0)

    recorder = RecordingClient(client)
    matcher = T6DecideMatcher(recorder, timeout_seconds=timeout)
    lines: list[str] = []
    mismatches = []
    snapshots: set[str] = set()
    for pair in pairs:
        candidate = Candidate(**pair["candidates"][0])
        try:
            result = await matcher.match(pair["source_text"], pair["project"]["name"], [candidate])
        except MatcherError as exc:
            lines.insert(0, f"requests sent: {len(recorder.requests)}")
            detail = f"{pair['name']}: {exc.failure}: {exc.message}"
            if exc.retry_after is not None:
                detail += f" (retry after {exc.retry_after:g}s)"
            return _fail(lines, detail, len(recorder.requests))
        if result.returned_model:
            snapshots.add(result.returned_model)
        outcome = decide([candidate], result, t_link=t_link, t_hint=t_hint, link_enabled=False)
        [verdict] = outcome.outcomes
        expected = pair["expected_verdicts"][verdict.page_id]
        lines.append(
            f"{pair['name']} vs {verdict.name}: match={verdict.p_match:.2f} "
            f"maybe={verdict.p_maybe:.2f} no={verdict.p_no:.2f} -> {verdict.verdict} "
            f"(expected {expected}); tokens {result.input_tokens}/{result.output_tokens}"
        )
        if verdict.verdict != expected:
            mismatches.append(f"{pair['name']} is {verdict.verdict}, expected {expected}")
    lines.insert(0, f"requests sent: {len(recorder.requests)}")
    lines.insert(1, f"returned snapshot: {', '.join(sorted(snapshots)) or 'none'}")
    if not snapshots:
        return _fail(lines, "response carried no model snapshot", len(recorder.requests))
    if mismatches:
        # Shadow mode exists to observe this; the owner reviews before linking is enabled.
        detail = "review before enabling linking: " + "; ".join(mismatches)
        return ModelCheck(Result(MANUAL, detail), lines, len(recorder.requests))
    return ModelCheck(
        Result(PASS, f"3 regression pairs, 3 requests, snapshot {', '.join(sorted(snapshots))}"),
        lines,
        len(recorder.requests),
    )
