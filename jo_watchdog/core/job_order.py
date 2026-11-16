"""Pure core: the Job Order plan (#21).

Turns a validated extraction, plus the Name, number, Department, deadline and keys the t6 API
returned for the message (``jo raise`` preview, #151), into the Notion-shaped properties and
body blocks. The plan holds no rule of its own. It is stored before any write and replayed
as-is: ``build_raise_fields`` (payload.py) reads only the stored plan, never the extraction
or the clock.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal, Protocol
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from jo_watchdog.core.extraction import is_subtitle_value
from jo_watchdog.core.findings import ASAP, Finding
from jo_watchdog.core.payload import STATUS_NEEDS_INTAKE_REVIEW
from jo_watchdog.core.plan import (
    BLOCK_LIMIT,
    NOTES_PROPERTY,
    WATCHDOG_NOTES_PROPERTY,
    split_blocks,
)
from jo_watchdog.core.resolve import (
    PeopleResolution,
    ProjectResolution,
    relation_properties,
)

MANILA = ZoneInfo("Asia/Manila")
DeadlineSource = Literal["stated", "shift", "fallback", "blank"]

# Plans stored before #99 carried the watchdog block inside ``Notes`` behind this marker.
LEGACY_WATCHDOG_MARKER = "— added by the watchdog —"
ASAP_NOTE = "ASAP — deprioritize other JOs"

# Exact live property names, including the trailing space on `Blocker `.
TEXT_PROPERTIES: dict[str, str] = {
    "title": "Title",
    "task": "Task",
    "purpose": "Purpose",
    "source": "Source / Source Drive",
    "platform": "Platform",
    "link_settings": "Link Settings",
    "watermark": "Watermark",
    "mode_of_transfer": "Mode of Transfer",
    "blocker": "Blocker ",
    "pending": "Pending",
}


class Extraction(Protocol):
    """The raw fields this slice reads from the #20 extraction result."""

    header_raw: str | None
    jo_number_raw: str | None
    department_raw: str | None
    title: str | None
    task: str | None
    purpose: str | None
    source: str | None
    platform: str | None
    link_settings: str | None
    watermark: str | None
    mode_of_transfer: str | None
    blocker: str | None
    pending: str | None
    notes: str | None
    deadline_raw: str | None
    subtitle: str | None


class UnresolvedMention(BaseModel):
    """A People mention the resolver left out of the relations."""

    raw: str
    role: str  # "assignee" | "cc"
    status: str  # "unresolved" | "ambiguous"


class PlanResolution(BaseModel):
    """What resolution decided, stored with the plan so a resume never resolves again."""

    project_page_id: str | None = None
    department: str | None = None
    assigned: list[str] = Field(default_factory=list)
    cc: list[str] = Field(default_factory=list)
    unresolved: list[UnresolvedMention] = Field(default_factory=list)


class RaisePreview(BaseModel):
    """What ``jo raise`` would write for a message: the API's Name, number, Department,
    deadline and keys, and the token that confirms the raise (#151)."""

    token: str
    name: str | None = None
    jo_number: str | None = None
    department: str | None = None
    deadline: str | None = None  # ISO-8601 with its offset
    deadline_source: DeadlineSource = "blank"
    deadline_findings: list[dict[str, Any]] = Field(default_factory=list)
    department_findings: list[dict[str, Any]] = Field(default_factory=list)
    telegram_key: str | None = None
    duplicate_key: str | None = None


class JobOrderPlan(BaseModel):
    """The Notion-shaped pieces, stored at ``planned`` and replayed unchanged."""

    telegram_key: str
    name: str
    properties: dict[str, Any]
    body_blocks: list[str] = Field(default_factory=list)
    deadline_source: DeadlineSource = "blank"
    # The API's duplicate key for the message (``<Project page ID>|<number>``); None when it
    # has no Project or no number. Plans stored before #151 have none.
    duplicate_key: str | None = None
    # The ``jo raise`` message input (``JoRaiseMessage``) this plan was previewed with; the
    # write sends it again, so the API builds the rule fields again. None for a plan stored
    # before #151: it is written with its own fields.
    raise_message: dict | None = None
    findings: list[Finding] = Field(default_factory=list)
    # None when no caches are wired (nothing was resolved).
    resolution: PlanResolution | None = None
    # The duplicate decision made at plan time (#24), replayed unchanged:
    # ``{"code": "exact_repost" | "number_reuse", "url", "days", "jo_number"}``.
    duplicate: dict | None = None
    # The Deliverable match stage result (#26): candidate snapshot, per-candidate
    # probabilities and outcome. None when no matcher is wired.
    match: dict | None = None


def plain_text(prop: dict[str, Any] | None) -> str:
    """The text of a stored rich-text property, items joined; empty when absent."""
    return "".join(item["text"]["content"] for item in (prop or {}).get("rich_text") or [])


def _split_legacy_block(notes: str) -> tuple[str, str] | None:
    """Split pre-#99 ``<poster>\\n\\n<marker>\\n<lines>`` text into (poster, watchdog lines).

    The block is always last, so only the final marker line counts: a marker quoted
    inside the poster's own text is left alone. ``None`` when there is no block.
    """
    token = f"{LEGACY_WATCHDOG_MARKER}\n"
    if notes.startswith(token):
        return "", notes[len(token) :]
    at = notes.rfind(f"\n\n{token}")
    if at == -1:
        return None
    return notes[:at], notes[at + 2 + len(token) :]


def split_legacy_notes(properties: dict[str, Any]) -> dict[str, Any]:
    """Move a pre-#99 watchdog block out of ``Notes`` into ``Watchdog Notes``.

    A plan stored before the split holds ``<poster notes>\\n\\n<marker>\\n<lines>`` in
    ``Notes``. Replaying it must not write watchdog text to the coordinator's column.
    A plan that already has ``Watchdog Notes`` is new-format and is returned unchanged.
    """
    if WATCHDOG_NOTES_PROPERTY in properties:
        return properties
    legacy_notes = properties.get(NOTES_PROPERTY) or properties.get("Notes")
    legacy = _split_legacy_block(plain_text(legacy_notes))
    if legacy is None:
        if "Notes" in properties and NOTES_PROPERTY != "Notes":
            split = {k: v for k, v in properties.items() if k != "Notes"}
            split[NOTES_PROPERTY] = properties["Notes"]
            return split
        return properties
    poster, block = legacy
    split = {k: v for k, v in properties.items() if k not in (NOTES_PROPERTY, "Notes")}
    if poster:
        split[NOTES_PROPERTY] = {"rich_text": rich_text(poster)}
    if block:
        split[WATCHDOG_NOTES_PROPERTY] = {"rich_text": rich_text(block)}
    return split


def load_plan(stored: str | dict) -> JobOrderPlan:
    """Read a stored plan; a pre-#39 skeleton plan (no ``properties``) is upgraded in place."""
    data = json.loads(stored) if isinstance(stored, str) else dict(stored)
    if "properties" in data:
        plan = JobOrderPlan.model_validate(data)
        plan.properties = split_legacy_notes(plan.properties)
        return plan
    properties: dict[str, Any] = {
        "Name": {"title": rich_text(data["name"])},
        "Telegram Key": {"rich_text": rich_text(data["telegram_key"])},
        "Status": {"status": {"name": STATUS_NEEDS_INTAKE_REVIEW}},
    }
    # A skeleton plan's notes were watchdog-only (the earlier-row reference).
    skeleton_notes = data.get("notes") or ""
    legacy = _split_legacy_block(skeleton_notes)
    if legacy is not None:
        skeleton_notes = "\n\n".join(part for part in legacy if part)
    if skeleton_notes:
        properties[WATCHDOG_NOTES_PROPERTY] = {"rich_text": rich_text(skeleton_notes)}
    return JobOrderPlan(
        telegram_key=data["telegram_key"],
        name=data["name"],
        properties=properties,
        body_blocks=data.get("body_blocks", []),
        duplicate=data.get("duplicate"),
    )


def manila_iso(value: datetime) -> str:
    """ISO-8601 with the explicit ``+08:00`` offset."""
    return value.astimezone(MANILA).isoformat()


def notion_date(value: datetime) -> dict[str, Any]:
    """A Notion Date value for ``value``: the instant at its Manila UTC offset, no ``time_zone``.

    Notion rejects a ``time_zone`` next to a non-zero offset (#83), and with a
    ``time_zone`` it reads the wall clock of ``start`` in that zone and ignores the
    offset: a UTC start came back eight hours early (#85). A bare offset keeps the
    exact instant, which is what the read-back compares.
    """
    return {"date": {"start": manila_iso(value)}}


def _instant_date(date: dict[str, Any]) -> dict[str, Any]:
    """Drop ``time_zone`` from a stored date, keeping each instant at its Manila offset."""
    if "time_zone" not in date:
        return date
    fixed = {k: v for k, v in date.items() if k != "time_zone"}
    for edge in ("start", "end"):
        value = date.get(edge)
        if not isinstance(value, str):
            continue
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            continue
        if parsed.tzinfo is not None:
            fixed[edge] = manila_iso(parsed)
    return fixed


def normalize_date_properties(properties: dict[str, Any]) -> dict[str, Any]:
    """Make every Date property Notion-safe, including ones in plans stored earlier.

    Plans stored by earlier builds carry ``time_zone`` (with ``+08:00`` before #83, with
    UTC after it). Each instant is kept and re-expressed at the Manila offset without
    ``time_zone``, so a stuck row is replanned by replay and no second page is made.
    """
    return {
        name: (
            {**prop, "date": _instant_date(prop["date"])}
            if isinstance(prop, dict) and isinstance(prop.get("date"), dict)
            else prop
        )
        for name, prop in properties.items()
    }


def rich_text(value: str) -> list[dict]:
    """Split a text value into Notion rich-text items of at most 2000 characters."""
    return [
        {"type": "text", "text": {"content": value[i : i + BLOCK_LIMIT]}}
        for i in range(0, len(value), BLOCK_LIMIT)
    ]


def _clean(value: str | None) -> str:
    """Trim a verbatim field; absent values stay blank."""
    if value is None:
        return ""
    return value.replace("\xa0", " ").strip()


def _verbatim(value: str | None) -> str:
    """A verbatim field with its whitespace and NBSP; trimming only decides emptiness."""
    if value is None or not value.strip():
        return ""
    return value


def build_notes(poster_notes: str | None) -> str:
    """The coordinator's own notes, verbatim, for the ``Notes`` column."""
    return _verbatim(poster_notes)


def build_watchdog_notes(deadline_raw: str | None, asap: bool) -> str:
    """What the watchdog adds, one line per item, for the ``Watchdog Notes`` column.

    ``Earlier row:`` lines are appended by #24 after these lines.
    """
    added: list[str] = []
    deadline = _clean(deadline_raw)
    if deadline:
        added.append(f"Deadline as posted: {deadline}")
    if asap:
        added.append(ASAP_NOTE)
    return "\n".join(added)


def plan_job_order(
    extraction: Extraction,
    *,
    rules: RaisePreview,
    telegram_key: str,
    raw_text: str,
    posted_at: datetime,
    source_block: str,
    raise_message: dict | None = None,
) -> JobOrderPlan:
    """Build the plan for one Job Order.

    ``rules`` is the API's answer for the message (the ``jo raise`` preview): the plan takes
    its Name, number, Department, deadline and keys as they are. ``posted_at`` is the intake
    posting or forwarding time, never the forwarded original's time. ``source_block`` is the
    rendered provenance block (#24's ``provenance.source_block``). The API withholds credentials
    from what it writes, so the text is kept as posted.
    """
    name = rules.name or ""
    key = rules.telegram_key or telegram_key
    deadline = datetime.fromisoformat(rules.deadline) if rules.deadline else None
    findings = [
        Finding(code=f["code"], params=f.get("params") or {}) for f in rules.deadline_findings
    ]
    asap = any(f.code == ASAP for f in findings)

    properties: dict[str, Any] = {
        "Name": {"title": rich_text(name)},
        "Telegram Key": {"rich_text": rich_text(key)},
        "Status": {"status": {"name": STATUS_NEEDS_INTAKE_REVIEW}},
        "Date": notion_date(posted_at),
    }
    if rules.jo_number:
        properties["JO Number"] = {"rich_text": rich_text(rules.jo_number)}
    if rules.department:
        properties["Department"] = {"select": {"name": rules.department}}
    if deadline is not None:
        properties["Deadline"] = notion_date(deadline)
    for field, prop in TEXT_PROPERTIES.items():
        raw_value = getattr(extraction, field)
        value = _verbatim(raw_value) if field == "title" else _clean(raw_value)
        if value:
            properties[prop] = {"rich_text": rich_text(value)}
    subtitle = _clean(extraction.subtitle)
    if subtitle and is_subtitle_value(subtitle):
        properties["Subtitle"] = {"rich_text": rich_text(subtitle)}
    notes = build_notes(extraction.notes)
    if notes:
        properties[NOTES_PROPERTY] = {"rich_text": rich_text(notes)}
    watchdog_notes = build_watchdog_notes(extraction.deadline_raw, asap)
    if watchdog_notes:
        properties[WATCHDOG_NOTES_PROPERTY] = {"rich_text": rich_text(watchdog_notes)}

    body = [
        *split_blocks(source_block),
        *split_blocks(raw_text),
    ]

    return JobOrderPlan(
        telegram_key=key,
        name=name,
        properties=properties,
        body_blocks=body,
        deadline_source=rules.deadline_source,
        duplicate_key=rules.duplicate_key,
        raise_message=raise_message,
        findings=findings,
    )


def apply_resolution(
    plan: JobOrderPlan,
    *,
    people: PeopleResolution,
    project: ProjectResolution,
    department: str | None,
    department_findings: list[Finding],
) -> None:
    """Merge the #22 resolution into the plan: relations, findings and the stored decision.

    ``department`` and its findings are the API's (the ``jo raise`` preview).
    """
    plan.properties.update(relation_properties(people, project))
    plan.resolution = PlanResolution(
        project_page_id=project.page_id,
        department=department,
        assigned=list(people.assigned),
        cc=list(people.cc),
        unresolved=[
            UnresolvedMention(raw=m.raw, role=m.role, status=m.status)
            for m in people.mentions
            if m.status != "resolved"
        ],
    )
    plan.findings.extend(
        Finding(code=f.code, params=dict(f.params))
        for f in (*people.findings, *project.findings)
    )
    plan.findings.extend(department_findings)
