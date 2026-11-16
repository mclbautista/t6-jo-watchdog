"""Pure core: resolve raw People, Project and Department mentions to Notion values.

The extractor only proposes text; identity is decided elsewhere (resolve, don't
generate): Contacts and Projects by the t6 API, Departments against the cached
list. This module splits the posted text into mentions, turns the API's answers
into results and findings, and has no I/O.

Precision beats recall. Anything ambiguous stays blank and is reported as a
finding (code plus parameters, rendered later by the ops reporter); nothing is
backfilled later.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import time
from typing import Literal

from pydantic import BaseModel, Field

# Mention separators: commas, slashes, ampersands, " and ", and line breaks.
_SPLIT_RE = re.compile(r"\s*(?:,|/|&|\n|\band\b)\s*", re.IGNORECASE)
_HEDGE_RE = re.compile(r"\b(?:but|maybe|or|baka|if|kung)\b", re.IGNORECASE)
_CONDITION_RE = re.compile(r"\b(?:if|kung)\b.*$", re.IGNORECASE | re.DOTALL)
_LEADING_FOR_RE = re.compile(r"^for\s+", re.IGNORECASE)
_COURTESY_TAIL_RE = re.compile(
    r"(?:[\s,!.]+|^)(?:thank\w*|ty(?:vm+)?|you+\w*|u|po)[\s,!.]*$", re.IGNORECASE
)
_HANDLE_RE = re.compile(r"@([A-Za-z0-9_]{3,})")


# -- snapshot models ---------------------------------------------------------


class Person(BaseModel):
    """One Contacts page, as cached."""

    page_id: str
    name: str
    telegram_user_id: str = ""
    telegram_handle: str = ""
    aliases: list[str] = Field(default_factory=list)
    # A parsed time of day from the API; the raw option text ("10:00 PM", "Flexible /
    # Variable") or "" from a Notion page. Left to right: a time string reads as a time.
    end_of_shift: time | str = Field(default="", union_mode="left_to_right")


class ProjectEntry(BaseModel):
    """One Projects page, as cached."""

    page_id: str
    name: str


class Finding(BaseModel):
    """A finding code plus its parameters; the ops reporter renders the line."""

    code: str
    params: dict[str, str] = Field(default_factory=dict)


class ResolveSettings(BaseModel):
    """The resolution knobs from config."""

    department_aliases: dict[str, str] = Field(default_factory=dict)
    standing_cc_contact_page_id: str | None = None


class MentionEntity(BaseModel):
    """A Telegram ``mention`` or ``text_mention`` entity from the message.

    ``text`` is the entity's text as it appears in the message (``@handle`` for
    a mention, the display name for a text_mention).
    """

    text: str
    user_id: str | None = None
    handle: str | None = None


class MentionResult(BaseModel):
    """How one raw mention resolved."""

    raw: str
    role: str  # "assignee" | "cc"
    status: str  # "resolved" | "unresolved" | "ambiguous"
    method: str | None = None  # the API's method: "exact_id" | "normalized" | "fuzzy" | "llm"
    page_id: str | None = None
    contact_name: str | None = None


class AssigneeShift(BaseModel):
    """A resolved assignee's End of Shift, as handed to the deadline resolver."""

    page_id: str
    name: str
    end_of_shift: time | str = Field(union_mode="left_to_right")  # as on the Person


class PeopleResolution(BaseModel):
    """Relation values and findings for Assigned and cc."""

    assigned: list[str] = Field(default_factory=list)
    cc: list[str] = Field(default_factory=list)
    mentions: list[MentionResult] = Field(default_factory=list)
    assignee_shifts: list[AssigneeShift] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)


class ProjectResolution(BaseModel):
    page_id: str | None = None
    code: str | None = None
    findings: list[Finding] = Field(default_factory=list)


class DepartmentResolution(BaseModel):
    """``department`` is the exact live option name, or None when it failed validation."""

    department: str | None = None
    token: str | None = None
    findings: list[Finding] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.department is not None


# -- mentions --------------------------------------------------------------------


def normalize_handle(handle: str) -> str:
    return handle.strip().lstrip("@").casefold()


def _fold(text: str) -> str:
    return " ".join(text.casefold().split())


def _strip_courtesy(segment: str) -> str:
    """Drop a leading ``for`` and trailing thanks (``thank youuu``, ``tyvm``, ``po``)."""
    text = _LEADING_FOR_RE.sub("", segment.strip())
    while True:
        m = _COURTESY_TAIL_RE.search(text)
        if not m or m.start() == m.end():
            return text.strip()
        text = text[: m.start()].strip()


def _pieces(raw: str | None) -> list[tuple[str, bool]]:
    """Each mention of an ``assignees_raw``/``cc_raw`` value, with its lead flag.

    ``lead`` marks a single word written right before an ``@handle`` (``ate @user_four``):
    it may be a title for the handle or a person of its own, which the answers decide.
    """
    if not raw or not raw.strip():
        return []
    condition = _CONDITION_RE.search(raw)
    head = raw[: condition.start()] if condition else raw
    out = _split_head(head)
    if condition:
        tail = condition.group(0).strip()
        out = [*out[:-1], (f"{out[-1][0]} {tail}", False)] if out else [(tail, False)]
    return out


def split_mentions(raw: str | None) -> list[str]:
    """Split an ``assignees_raw``/``cc_raw`` value into individual mentions.

    ``@handles`` are split from the names around them. A hedged mention
    (``Ramon but maybe Lena``) is kept whole so it is reported as one ambiguous
    mention, quoted as written. A condition (``if abutin the next day``,
    ``kung sakaling ...``) is joined, complete, to the mention before it, so
    that mention is ambiguous and nothing is assigned on a condition.
    """
    return [text for text, _ in _pieces(raw)]


def _split_head(text: str) -> list[tuple[str, bool]]:
    out: list[tuple[str, bool]] = []
    for part in _SPLIT_RE.split(text):
        part = _strip_courtesy(part)
        if not part:
            continue
        if _HEDGE_RE.search(part):
            out.append((part, False))
            continue
        out.extend(_split_handles(part))
    return out


def _split_handles(part: str) -> list[tuple[str, bool]]:
    """``Ramon @user_two`` -> ``Ramon`` (lead), ``@user_two``."""
    pieces: list[tuple[str, bool]] = []
    last = 0
    for m in _HANDLE_RE.finditer(part):
        before = part[last : m.start()].strip()
        if before:
            pieces.append((before, len(before.split()) == 1))
        pieces.append((m.group(0), False))
        last = m.end()
    tail = part[last:].strip()
    if tail:
        pieces.append((tail, False))
    return [(t, lead) for t, lead in ((_strip_courtesy(t), lead) for t, lead in pieces) if t]


class Mention(BaseModel):
    """One mention after splitting: what was written, and what is asked of the API for it."""

    raw: str
    role: str  # "assignee" | "cc"
    query: str
    hedged: bool = False  # never sent: it ends ambiguous as written
    lead: bool = False  # a lone word right before an @handle
    words: list[str] = Field(default_factory=list)  # the words of a name with no separator


class ContactMatch(BaseModel):
    """The API's answer for one mention: ``person find`` boiled down to one status."""

    status: Literal["resolved", "ambiguous", "unresolved"]
    method: str | None = None  # the API's: exact_id, normalized, fuzzy, llm
    page_id: str | None = None
    name: str | None = None


def _entity_for(segment: str, entities: Sequence[MentionEntity]) -> MentionEntity | None:
    """The entity that is this mention itself, never one that merely overlaps it.

    A ``mention`` entity binds when its ``@handle`` is a whole token of the
    segment; a ``text_mention`` binds when its display text is the segment.
    ``Jo`` therefore never supplies identity for ``Lena``.
    """
    handles = {normalize_handle(m.group(1)) for m in _HANDLE_RE.finditer(segment)}
    for entity in entities:
        if not entity.text:
            continue
        if entity.text.startswith("@"):
            if normalize_handle(entity.text) in handles:
                return entity
        elif _fold(entity.text) == _fold(segment):
            return entity
    return None


def _query_for(segment: str, entities: Sequence[MentionEntity]) -> str:
    """What the API is asked: the Telegram ID or handle of a bound entity, else the text."""
    entity = _entity_for(segment, entities)
    if entity and entity.user_id:
        return entity.user_id
    if entity and entity.handle:
        return "@" + normalize_handle(entity.handle)
    return segment


def people_mentions(
    assignees_raw: str | None, cc_raw: str | None, entities: Sequence[MentionEntity] = ()
) -> list[Mention]:
    """Assignee mentions, then cc mentions, in the order written."""
    mentions = []
    for role, raw in (("assignee", assignees_raw), ("cc", cc_raw)):
        for text, lead in _pieces(raw):
            hedged = bool(_HEDGE_RE.search(text))
            words = text.split()
            mentions.append(
                Mention(
                    raw=text,
                    role=role,
                    query=_query_for(text, entities),
                    hedged=hedged,
                    lead=lead and not hedged,
                    words=words if not hedged and "@" not in text and len(words) >= 2 else [],
                )
            )
    return mentions


def mention_results(
    mention: Mention, match: ContactMatch | None, word_matches: Sequence[ContactMatch] = ()
) -> list[MentionResult]:
    """How one mention resolved: usually one result, several when ``Mira Lena`` is two people.

    A hedged mention has no match and ends ambiguous. A mention nobody knows splits into its
    words only if every word is exactly one known Contact (``word_matches``, one for each word).
    """
    if match is None:
        return [MentionResult(raw=mention.raw, role=mention.role, status="ambiguous")]
    resolved = match.status == "resolved"
    result = MentionResult(
        raw=mention.raw,
        role=mention.role,
        status=match.status,
        method=match.method if resolved else None,
        page_id=match.page_id if resolved else None,
        contact_name=match.name if resolved else None,
    )
    if match.status != "unresolved" or not word_matches:
        return [result]
    exact = ("exact_id", "normalized")
    if any(m.status != "resolved" or m.method not in exact for m in word_matches):
        return [result]
    return [
        MentionResult(
            raw=word,
            role=mention.role,
            status="resolved",
            method=m.method,
            page_id=m.page_id,
            contact_name=m.name,
        )
        for word, m in zip(mention.words, word_matches, strict=True)
    ]


def merge_leads(
    mentions: Sequence[Mention], groups: Sequence[list[MentionResult]]
) -> list[MentionResult]:
    """A lead word nobody knows belongs to the @handle after it (``ate @user_four``)."""
    out: list[MentionResult] = []
    skip = False
    for i, (mention, group) in enumerate(zip(mentions, groups, strict=True)):
        if skip:
            skip = False
            continue
        following = mentions[i + 1] if i + 1 < len(mentions) else None
        if (
            mention.lead
            and group[0].status == "unresolved"
            and following is not None
            and following.role == mention.role
            and following.raw.startswith("@")
        ):
            handle_result = groups[i + 1][0]
            out.append(
                handle_result.model_copy(update={"raw": f"{mention.raw} {following.raw}"})
            )
            skip = True
        else:
            out.extend(group)
    return out


def _mention_finding(result: MentionResult) -> Finding | None:
    if result.status == "resolved" and result.method in ("fuzzy", "llm"):
        return Finding(
            code="fuzzy_person",
            params={"raw": result.raw, "role": result.role, "contact": result.contact_name or ""},
        )
    if result.status == "unresolved":
        return Finding(code="unresolved_person", params={"raw": result.raw, "role": result.role})
    if result.status == "ambiguous":
        return Finding(code="ambiguous_person", params={"raw": result.raw, "role": result.role})
    return None


def _add_unique(ids: list[str], page_id: str) -> None:
    if page_id not in ids:
        ids.append(page_id)


def people_resolution(
    results: Sequence[MentionResult], people: Sequence[Person], settings: ResolveSettings
) -> PeopleResolution:
    """Assigned and cc from the mention results.

    The standing cc is always in cc; Coordinator stays blank.
    """
    out = PeopleResolution()
    for result in results:
        out.mentions.append(result)
        finding = _mention_finding(result)
        if finding:
            out.findings.append(finding)
        if result.status != "resolved" or result.page_id is None:
            continue
        _add_unique(out.assigned if result.role == "assignee" else out.cc, result.page_id)

    if settings.standing_cc_contact_page_id:
        _add_unique(out.cc, settings.standing_cc_contact_page_id)

    known = {p.page_id: p for p in people}
    names = {r.page_id: r.contact_name for r in out.mentions if r.page_id}
    for page_id in out.assigned:
        person = known.get(page_id)
        out.assignee_shifts.append(
            AssigneeShift(
                page_id=page_id,
                name=person.name if person else names.get(page_id) or "",
                end_of_shift=person.end_of_shift if person else "",
            )
        )
    return out


# -- projects ------------------------------------------------------------------


class ProjectMatch(BaseModel):
    """The API's answer for a project text: the Project in rotation, or nothing decided."""

    page_id: str | None = None
    code: str | None = None


def resolve_project(project_raw: str | None, match: ProjectMatch | None) -> ProjectResolution:
    """The Project the API decided, or Projects blank with an ``unresolved_project`` finding.

    Nothing is asked, and nothing is found, for a blank text (``match`` is None then).
    """
    if not project_raw or not project_raw.strip():
        return ProjectResolution()
    if match is not None and match.page_id:
        return ProjectResolution(page_id=match.page_id, code=match.code)
    return ProjectResolution(
        findings=[Finding(code="unresolved_project", params={"raw": project_raw.strip()})]
    )


# -- departments ----------------------------------------------------------------


def normalize_department_token(token: str, aliases: Mapping[str, str]) -> str:
    """Apply the configured header alias map, then uppercase."""
    stripped = " ".join(token.split())
    folded_aliases = {k.strip().casefold(): v for k, v in aliases.items()}
    return folded_aliases.get(stripped.casefold(), stripped).upper()


def resolve_department(
    token: str | None, options: Sequence[str], aliases: Mapping[str, str]
) -> DepartmentResolution:
    """Validate a header department token against the live ``Department`` options."""
    if not token or not token.strip():
        return DepartmentResolution()
    normalized = normalize_department_token(token, aliases)
    wanted = " ".join(normalized.split()).casefold()
    # More than one live option folding to the same spelling is ambiguous.
    matches = {o for o in options if " ".join(o.split()).casefold() == wanted}
    match = next(iter(matches)) if len(matches) == 1 else None
    if match is None:
        return DepartmentResolution(
            token=token.strip(),
            findings=[Finding(code="unknown_department", params={"token": token.strip()})],
        )
    return DepartmentResolution(department=match, token=token.strip())


def relation_properties(people: PeopleResolution, project: ProjectResolution) -> dict[str, dict]:
    """Notion relation values for Assigned, cc and Projects.

    Only resolved page ids are written; an unresolved Project leaves Projects
    empty. ``🗂️ Coordinator`` is never set in v1.
    """
    return {
        "Assigned": {"relation": [{"id": page_id} for page_id in people.assigned]},
        "cc": {"relation": [{"id": page_id} for page_id in people.cc]},
        "Projects": {"relation": [{"id": project.page_id}] if project.page_id else []},
    }


# -- Telegram entities ------------------------------------------------------------


def mention_entities(message: Mapping) -> list[MentionEntity]:
    """The ``mention`` and ``text_mention`` entities of a Telegram message.

    Telegram offsets count UTF-16 code units, so the text is sliced there.
    """
    # A photo or document carries its text as a caption, with its own entities.
    text = message.get("text") or message.get("caption") or ""
    entities = message.get("entities") or message.get("caption_entities") or []
    units = text.encode("utf-16-le")
    out: list[MentionEntity] = []
    for entity in entities:
        kind = entity.get("type")
        if kind not in ("mention", "text_mention"):
            continue
        start, length = entity.get("offset", 0), entity.get("length", 0)
        shown = units[start * 2 : (start + length) * 2].decode("utf-16-le", errors="ignore")
        if kind == "mention":
            out.append(MentionEntity(text=shown, handle=shown.lstrip("@")))
        else:
            user_id = (entity.get("user") or {}).get("id")
            out.append(
                MentionEntity(text=shown, user_id=str(user_id) if user_id is not None else None)
            )
    return out
