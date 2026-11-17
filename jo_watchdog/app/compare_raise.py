"""``compare-raise``: the new path against the Job Orders already written (#134, #151).

For the last N ledger messages that have a Job Order, it runs the path up to and including the
``jo raise`` preview (never a confirm: nothing is written or sent) and compares each built field
with what was written at the time, the stored plan. The report holds counts for each field and,
for a difference, the field name and ledger row id; never a posted text.

The API adds the standing cc at ``jo raise`` and the preview does not return the cc, so the new
side's ``cc`` is the watchdog's own plus the API's ``standing-cc`` (read once, as a page ID). A
stored plan that holds the standing cc is the same; with no standing cc on the API it differs.

A row is ``missing`` for every field when it cannot be compared: it has no stored plan, or the
new path could not be built (no stored extraction, or the API did not answer).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from jo_watchdog.core.payload import _plain, _start

FIELDS = (
    "name",
    "number",
    "department",
    "deadline",
    "project",
    "assignees",
    "cc",
    "telegram_key",
    "duplicate_key",
)
DEFAULT_MUST_MATCH = ("name", "number", "department", "deadline", "project", "assignees")
SAME, DIFFERENT, MISSING = "same", "different", "missing"


def _ids(prop: dict | None) -> frozenset[str]:
    return frozenset(item["id"] for item in (prop or {}).get("relation") or [])


def _instant(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(UTC).replace(second=0, microsecond=0)


def values(plan: dict, *, duplicate_key: str | None = None) -> dict[str, Any]:
    """The compared fields of a plan (stored, or built by the new path), Notion-shaped."""
    props = plan.get("properties") or {}
    department = ((props.get("Department") or {}).get("select") or {}).get("name") or ""
    return {
        "name": plan.get("name") or "",
        "number": _plain(props.get("JO Number")),
        "department": department,
        "deadline": _instant(_start(props.get("Deadline"))),
        "project": _ids(props.get("Projects")),
        "assignees": _ids(props.get("Assigned")),
        "cc": _ids(props.get("cc")),
        "telegram_key": plan.get("telegram_key") or "",
        "duplicate_key": plan.get("duplicate_key") or duplicate_key or "",
    }


def compare(stored: dict[str, Any], built: dict[str, Any]) -> dict[str, str]:
    """``same`` or ``different`` for each field both sides hold; a blank side differs from a value."""
    out = {}
    for name in FIELDS:
        if name in stored and name in built:
            a, b = stored[name], built[name]
            if name == "deadline":
                a, b = _instant(a) if isinstance(a, str) else a, _instant(b) if isinstance(b, str) else b
            out[name] = SAME if a == b or (not a and not b) else DIFFERENT
    return out


@dataclass
class Report:
    rows: int = 0
    compared: int = 0
    not_built: int = 0
    counts: dict[str, Counter] = field(
        default_factory=lambda: {name: Counter(same=0, different=0, missing=0) for name in FIELDS}
    )
    differences: list[tuple[str, int]] = field(default_factory=list)
    has_standing_cc: bool = False


async def standing_cc_page_id(api, directory) -> str | None:
    """The API's standing cc as a Contact page ID: ``standing-cc`` answers the T6 Person ID when
    the page is bound, else the page ID; ``person list`` (``directory``) says which page a T6 ID is. An answer
    no Contact holds stays as the API gave it (it then differs from any page ID)."""
    answer = await api.standing_cc()
    if not answer:
        return None
    for person in await directory.people():
        if answer in (person.get("t6_id"), person.get("page_id")):
            return person["page_id"]
    return answer


async def run_comparison(
    ledger,
    worker,
    *,
    last: int,
    must_match: tuple[str, ...],
    standing_cc: str | None = None,
) -> Report:
    """Compare the last ``last`` written messages; ``worker`` builds the plan (preview only).

    ``standing_cc`` is the API's standing cc as a page ID (None: the API has none).
    """
    report = Report(has_standing_cc=bool(standing_cc))
    for row in ledger.written_messages(last):
        report.rows += 1
        stored_plan = json.loads(row["plan_json"]) if row["plan_json"] else None
        built_plan = None
        if stored_plan:
            try:
                built_plan = await worker.build_plan(row)
            except Exception:  # a refused preview is a row the new path cannot build
                built_plan = None
        if stored_plan is None or built_plan is None:
            report.not_built += built_plan is None and stored_plan is not None
            for name in FIELDS:
                report.counts[name][MISSING] += 1
            continue
        report.compared += 1
        built = values(built_plan.model_dump(mode="json"))
        if standing_cc:
            built["cc"] = built["cc"] | {standing_cc}
        outcome = compare(values(stored_plan, duplicate_key=row.get("jo_key")), built)
        for name, verdict in outcome.items():
            report.counts[name][verdict] += 1
            if verdict == DIFFERENT:
                report.differences.append((name, row["id"]))
    return report


def exit_code(report: Report, must_match: tuple[str, ...]) -> int:
    """1 when a field of a must-match kind differs; 2 when no row could be compared at all."""
    if any(name in must_match for name, _ in report.differences):
        return 1
    if report.rows and not report.compared:
        return 2
    return 0


def render(report: Report) -> str:
    lines = [f"compared {report.compared} of {report.rows} written messages"]
    lines.append(f"standing cc on the API: {'yes' if report.has_standing_cc else 'no'}")
    if report.not_built:
        lines.append(f"{report.not_built} could not be built (see the log)")
    lines.append(f"{'field':<14}{'same':>6}{'different':>11}{'missing':>9}")
    for name in FIELDS:
        c = report.counts[name]
        lines.append(f"{name:<14}{c[SAME]:>6}{c[DIFFERENT]:>11}{c[MISSING]:>9}")
    lines += [f"different: {name} row {row_id}" for name, row_id in sorted(report.differences)]
    return "\n".join(lines)
