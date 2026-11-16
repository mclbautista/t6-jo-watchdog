"""Pure core: the ``jo raise`` fields from a stored plan, and read-back comparison."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from jo_watchdog.core.masker import mask_text

if TYPE_CHECKING:
    from jo_watchdog.core.job_order import JobOrderPlan

# The only Status the watchdog ever writes (job-order field mapping).
STATUS_NEEDS_INTAKE_REVIEW = "Needs Intake Review"


# The live property of a stored plan -> the ``jo raise`` field that writes it.
_TEXT_FIELDS = {
    "Title": "title",
    "Subtitle": "subtitle",
    "Task": "task",
    "Purpose": "purpose",
    "Source / Source Drive": "source",
    "Platform": "platform",
    "Link Settings": "link_settings",
    "Watermark": "watermark",
    "Mode of Transfer": "mode_of_transfer",
    "Blocker ": "blocker",
    "Pending": "pending",
    "JO Notes": "notes",
    "Watchdog Notes": "watchdog_notes",
}
_RELATION_FIELDS = {"Assigned": "assigned_ids", "cc": "cc_ids", "Deliverable": "deliverable_ids"}
# What the API builds from a message; a stored plan from before #151 sends them itself.
RULE_FIELDS = {
    "Name": "name",
    "JO Number": "number",
    "Department": "department",
    "Deadline": "deadline",
    "Telegram Key": "telegram_key",
}


def _plain(prop: Mapping[str, Any] | None) -> str:
    return "".join(
        item["text"]["content"] for item in (prop or {}).get("rich_text") or []
    ) or "".join(item["text"]["content"] for item in (prop or {}).get("title") or [])


def _start(prop: Mapping[str, Any] | None) -> str | None:
    date = (prop or {}).get("date") or {}
    start = date.get("start")
    if not isinstance(start, str):
        return None
    if date.get("time_zone"):  # plans stored before #83 carried one; keep the instant
        parsed = datetime.fromisoformat(start)
        if parsed.tzinfo is not None:
            return parsed.astimezone(_MANILA).isoformat()
    return start


def build_raise_fields(plan: Any) -> dict[str, Any]:
    """The ``fields`` of a ``jo raise`` from a stored plan (Notion-shaped properties).

    A plan with a ``raise_message`` leaves the Name, number, Department, deadline and Telegram
    Key out: the API builds them from the message. The Project goes in the message too, so the
    Projects relation is not a field. Any other stored plan (from before #151) sends all of them
    itself. The body is the source block and the posted text, one line per paragraph.
    """
    properties = plan.properties
    fields: dict[str, Any] = {
        "status": ((properties.get("Status") or {}).get("status") or {}).get(
            "name", STATUS_NEEDS_INTAKE_REVIEW
        ),
    }
    if start := _start(properties.get("Date")):
        fields["date"] = start
    for prop, field in _TEXT_FIELDS.items():
        if text := _plain(properties.get(prop)):
            fields[field] = text
    for prop, field in _RELATION_FIELDS.items():
        if prop in properties:
            fields[field] = [r["id"] for r in properties[prop].get("relation") or []]
    if plan.raise_message is None:
        projects = (properties.get("Projects") or {}).get("relation") or []
        if projects:
            fields["project_id"] = projects[0]["id"]
        for prop, field in RULE_FIELDS.items():
            if prop == "Deadline":
                value = _start(properties.get(prop))
            elif prop == "Department":
                value = ((properties.get(prop) or {}).get("select") or {}).get("name")
            else:
                value = _plain(properties.get(prop))
            if value:
                fields[field] = value
    body = "\n".join(mask_text(block) for block in plan.body_blocks)
    if body:
        fields["source_text"] = body
    return fields


def page_url_for(page_id: str) -> str:
    """A working Notion URL for a page id, for rows whose create-time URL was not kept."""
    return f"https://www.notion.so/{page_id.replace('-', '')}"


DATE_PROPERTIES = ("Date", "Deadline")
_MANILA = ZoneInfo("Asia/Manila")


def _minute(value: object, time_zone: str | None = None) -> datetime | None:
    """A Notion date start as an instant truncated to the minute (Notion drops seconds)."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        try:
            zone = ZoneInfo(time_zone) if time_zone else _MANILA
        except Exception:
            zone = _MANILA
        parsed = parsed.replace(tzinfo=zone)
    return parsed.replace(second=0, microsecond=0)


def _date_mismatch(name: str, planned: dict | None, read: dict | None) -> str | None:
    planned_start = (planned or {}).get("start")
    if planned_start is None:
        return None
    want = _minute(planned_start, (planned or {}).get("time_zone"))
    got = _minute((read or {}).get("start"), (read or {}).get("time_zone"))
    if want is not None and got is not None and want == got:
        return None
    shown = (read or {}).get("start")
    return f"{name} differs: planned {planned_start!r}, read {shown!r}"


def verify_readback(
    plan: JobOrderPlan,
    *,
    retrieved_name: str,
    retrieved_telegram_key: str,
    retrieved_dates: Mapping[str, dict | None] | None = None,
) -> list[str]:
    """Compare a retrieved Notion page against the stored plan.

    Returns a list of human-readable mismatch descriptions (empty when the
    written page matches the plan). Mismatches are recorded, never rewritten.
    Dates are compared by instant, to the minute, so an offset shift is caught
    while Notion's dropped seconds are not. Pass ``retrieved_dates`` to check them.
    """
    mismatches: list[str] = []
    if retrieved_name != plan.name:
        mismatches.append(f"Name differs: planned {plan.name!r}, read {retrieved_name!r}")
    if retrieved_telegram_key != plan.telegram_key:
        mismatches.append(
            f"Telegram Key differs: planned {plan.telegram_key!r}, read {retrieved_telegram_key!r}"
        )
    if retrieved_dates is not None:
        properties = getattr(plan, "properties", {}) or {}
        for name in DATE_PROPERTIES:
            planned = (properties.get(name) or {}).get("date")
            problem = _date_mismatch(name, planned, retrieved_dates.get(name))
            if problem:
                mismatches.append(problem)
    return mismatches


def readback_fields(job_order: Mapping[str, Any]) -> dict:
    """Extract ``name``, ``telegram_key`` and the ``dates`` from a ``jo get`` Job Order.

    ``date`` and ``deadline`` arrive as ISO text (a date, or a datetime with its
    offset). They are shaped as ``{"start": text}`` so the minute compare of
    ``verify_readback`` runs unchanged.
    """
    dates = {
        name: {"start": job_order[name.lower()]} if job_order.get(name.lower()) else None
        for name in DATE_PROPERTIES
    }
    return {
        "name": job_order.get("name") or "",
        "telegram_key": job_order.get("telegram_key") or "",
        "dates": dates,
    }
