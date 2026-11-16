"""Pure core: deadline resolution in Asia/Manila time.

The rule is t6's ``t6.domain.deadlines.deadline``. What stays here is a small
adapter: it turns each ``(name, End of Shift)`` pair the resolution slice hands
over into the ``Contact`` the rule takes, using this module's own shift parser,
so no shift format resolves differently than before (#135). The cache reads
(#133) will hand over parsed times, and the adapter then goes away.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime, time
from typing import Literal

from pydantic import BaseModel, Field
from t6.domain import Contact
from t6.domain.deadlines import deadline as t6_deadline
from t6.domain.deadlines import strip_hedges
from t6.domain.studio_time import MANILA

from jo_watchdog.core.findings import Finding

__all__ = [
    "MANILA",
    "DeadlineResult",
    "DeadlineSource",
    "parse_shift_hour",
    "resolve_deadline",
    "strip_hedges",
]

FLEXIBLE = "Flexible / Variable"

DeadlineSource = Literal["stated", "shift", "fallback", "blank"]

_SHIFT_OPTION = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*(AM|PM)\s*$", re.IGNORECASE)


class DeadlineResult(BaseModel):
    deadline: datetime | None = None
    source: DeadlineSource = "blank"
    findings: list[Finding] = Field(default_factory=list)


def parse_shift_hour(value: str | time | None) -> time | None:
    """Parse a Contacts ``End of Shift`` option (``7:00 PM``); None if flexible/blank.

    An already-parsed ``time`` (from the resolution slice) passes through.
    """
    if isinstance(value, time):
        return value
    if not value or value.strip() == FLEXIBLE:
        return None
    match = _SHIFT_OPTION.match(value)
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if not 1 <= hour <= 12 or minute > 59:
        return None
    if match.group(3).upper() == "AM":
        hour = 0 if hour == 12 else hour
    else:
        hour = 12 if hour == 12 else hour + 12
    return time(hour, minute)


def resolve_deadline(
    deadline_raw: str | None,
    posted_at: datetime,
    assignee_shifts: Sequence[tuple[str, str | time | None]] = (),
) -> DeadlineResult:
    """Resolve a posted deadline phrase to an Asia/Manila datetime.

    ``posted_at`` is the intake posting (or forwarding) time. ``assignee_shifts``
    holds ``(display name, End of Shift)`` for each resolved assignee; an empty
    sequence means no assignee resolved.
    """
    assignees = [
        Contact(page_id=f"assignee-{n}", name=name, end_of_shift=parse_shift_hour(value))
        for n, (name, value) in enumerate(assignee_shifts)
    ]
    result = t6_deadline(deadline_raw, posted_at, assignees)
    return DeadlineResult(
        deadline=result.deadline,
        source=result.source,
        findings=[Finding(code=f.code, params=f.params) for f in result.findings],
    )
