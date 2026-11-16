"""Corpus scoring for the link and hint thresholds (#61).

Offline and read-only: it scores recorded Jev probabilities against the link and
hint thresholds, then retains the numbers in a results file for the owner to
review. It never edits config and never enables linking. (Contact matching is the
t6 API's rule and has no threshold here.)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from jo_watchdog.core.matching import CandidateDecision, _verdict

SCHEMA_VERSION = 2
LINK_GRID = (0.70, 0.80, 0.85, 0.90, 0.95, 0.98)
HINT_GRID = (0.30, 0.40, 0.50, 0.60, 0.70)


@dataclass(frozen=True)
class MatchSample:
    """One recorded Jev answer and the verdict the owner expects for it."""

    name: str
    p_match: float
    p_maybe: float
    p_no: float
    expected: str  # link | hint | none


def sweep_match(samples: Sequence[MatchSample]) -> list[dict]:
    rows = []
    for t_link in LINK_GRID:
        for t_hint in HINT_GRID:
            if t_hint > t_link:
                continue
            false_links = missed_links = wrong_hints = 0
            for s in samples:
                decision = CandidateDecision(
                    page_id=s.name, p_match=s.p_match, p_maybe=s.p_maybe, p_no=s.p_no
                )
                got = _verdict(decision, t_link=t_link, t_hint=t_hint)
                false_links += got == "link" and s.expected != "link"
                missed_links += s.expected == "link" and got != "link"
                wrong_hints += got != s.expected and not (got == "link" or s.expected == "link")
            rows.append(
                {
                    "t_link": t_link,
                    "t_hint": t_hint,
                    "false_links": int(false_links),
                    "missed_links": int(missed_links),
                    "wrong_hints": int(wrong_hints),
                }
            )
    return rows


def match_samples(pairs: Sequence[dict]) -> list[MatchSample]:
    """Samples from the recorded regression pairs (``tests/fixtures/jev/regression_pairs.json``)."""
    samples = []
    for pair in pairs:
        body = pair["response"]["body"]["answers"]
        for i, candidate in enumerate(pair["candidates"]):
            p = body[f"c{i}"]["probabilities"]
            samples.append(
                MatchSample(
                    name=pair["name"],
                    p_match=p["match"],
                    p_maybe=p["maybe"],
                    p_no=p["no"],
                    expected=pair["expected_verdicts"][candidate["page_id"]],
                )
            )
    return samples


def _digest(results: dict) -> str:
    canonical = json.dumps(results, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def build_report(
    *,
    samples: Sequence[MatchSample],
    current: dict[str, float],
    sample_source: str,
    previous: dict | None = None,
) -> dict:
    """The retained results: the sweep, the thresholds in force, and the review state.

    ``adopted`` is always the thresholds currently in config; only the owner changes them.
    """
    results = {
        "match_sample_source": sample_source,
        "match_samples": len(samples),
        "match_sweep": sweep_match(samples),
    }
    digest = _digest(results)
    review = {"status": "pending", "reviewed_by": None, "reviewed_at": None, "evidence": None}
    prior = (previous or {}).get("owner_review")
    if (
        prior
        and prior.get("status") == "reviewed"
        and (previous or {}).get("results_digest") == digest
        and (previous or {}).get("adopted") == current
    ):
        review = prior
    return {
        "schema_version": SCHEMA_VERSION,
        "adopted": current,
        "link_enabled": False,
        "recommendation": "keep the adopted thresholds; nothing here enables linking",
        "results": results,
        "results_digest": digest,
        "owner_review": review,
    }


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_report(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
