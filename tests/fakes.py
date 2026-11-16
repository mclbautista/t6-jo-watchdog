"""Fake adapters for whole-pipeline tests (real SQLite, fake I/O)."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from t6.verbs import Done, Refused

from jo_watchdog.adapters.protocols import ClassifyResult, IncomingUpdate, PageRef
from jo_watchdog.adapters.t6api import ApiUnreachable, T6ApiError
from jo_watchdog.app.caches import person_from_page, project_from_page
from jo_watchdog.core.extraction import EXTRACTED_FIELDS, PROMPT_TAG, Extraction
from jo_watchdog.core.resolve import ContactMatch, ProjectMatch

# 2026-09-28 03:00:00 UTC = 11:00 in Manila.
POSTED_AT_UNIX = 1790564400


def update(
    update_id: int,
    *,
    chat_id: int = -100123,
    topic_id: int = 5,
    message_id: int = 1,
    text: str = "first line",
    extra: dict | None = None,
) -> IncomingUpdate:
    message = {"chat": {"id": chat_id}, "date": POSTED_AT_UNIX, "text": text, **(extra or {})}
    return IncomingUpdate(
        update_id=update_id,
        chat_id=chat_id,
        message_id=message_id,
        message_thread_id=topic_id,
        text=text,
        raw={"update_id": update_id, "message": message},
    )


def extraction(verdict: str = "jo", **fields) -> Extraction:
    base = {field: None for field in EXTRACTED_FIELDS}
    base.update(
        verdict=verdict,
        confidence="high",
        reasons=f"Looks like {verdict}.",
        multiple_jo_suspected=False,
    )
    base.update(fields)
    return Extraction(**base)


def ok_result(verdict: str = "jo", **fields) -> ClassifyResult:
    return ClassifyResult(
        outcome="ok",
        extraction=extraction(verdict, **fields),
        requested_model="deepseek-chat",
        returned_model="deepseek-chat-v9",
        prompt_tag=PROMPT_TAG,
        input_tokens=1200,
        output_tokens=300,
        latency_ms=850,
    )


def failed_result(outcome: str = "timeout") -> ClassifyResult:
    return ClassifyResult(
        outcome=outcome,
        error=f"{outcome} happened",
        requested_model="deepseek-chat",
        prompt_tag=PROMPT_TAG,
        latency_ms=60000,
    )


class FakeClassifier:
    """Returns scripted results in order (the last one repeats); records inputs."""

    def __init__(self, *results: ClassifyResult):
        self.results = list(results) or [ok_result()]
        self.calls: list[tuple[str, list[str] | None]] = []

    async def classify(self, text, origin_lines=None) -> ClassifyResult:
        self.calls.append((text, origin_lines))
        if len(self.results) > 1:
            return self.results.pop(0)
        return self.results[0]


class FakeTelegram:
    """Returns queued updates; records reactions and ops posts."""

    def __init__(self, updates=None, *, fail_reaction: bool = False, fail_send: int = 0):
        self.updates = list(updates or [])
        self.reactions: list[tuple[int, int, str]] = []
        self.sent: list[tuple[int, int, str]] = []
        self.get_updates_calls: list[int] = []
        self.fail_reaction = fail_reaction
        self.fail_send = fail_send  # fail this many send_message calls first

    async def get_updates(self, offset: int, timeout: int) -> list[IncomingUpdate]:
        self.get_updates_calls.append(offset)
        # Emit queued updates whose id >= offset, then clear the queue so the
        # same batch is not re-emitted on the next poll.
        batch = [u for u in self.updates if u.update_id >= offset]
        self.updates = []
        return batch

    async def set_reaction(self, chat_id: int, message_id: int, emoji: str) -> None:
        if self.fail_reaction:
            raise RuntimeError("reaction not allowed")
        self.reactions.append((chat_id, message_id, emoji))

    async def send_message(self, chat_id: int, topic_id: int, text: str) -> int:
        if self.fail_send:
            self.fail_send -= 1
            raise RuntimeError("telegram send failed")
        self.sent.append((chat_id, topic_id, text))
        return 9000 + len(self.sent)


def live_date(date: dict | None) -> dict | None:
    """A Date as the live Notion API returned it in the #85 trial.

    With ``time_zone`` Notion takes the wall clock of ``start`` in that zone and drops
    the offset; seconds are dropped either way and ``time_zone`` reads back null.
    """
    if not date:
        return None
    start = datetime.fromisoformat(date["start"])
    if date.get("time_zone"):
        start = start.replace(tzinfo=ZoneInfo(date["time_zone"]))
    start = start.astimezone(ZoneInfo("Asia/Manila")).replace(second=0, microsecond=0)
    return {"start": start.isoformat(), "end": None, "time_zone": None}


_HEADER = re.compile(r"^\s*(?P<project>.+?)\s*-\s*(?P<department>[A-Za-z]+)\s*-\s*JO\s*#?(?P<number>\d+)", re.I)


class FakeJobOrders:
    """A fake t6 API for Job Orders: ``jo raise`` (preview, confirm), ``jo get`` and ``jo find``.

    It builds the rule fields from a ``JoRaiseMessage`` the way the API does, in the simplest
    way the tests need: the Name is the header line, the number and Department are read from a
    ``<PROJECT> - <DEPT> - JO #<N>`` header (the Department only if it is one of ``departments``,
    else a finding), the keys are ``<chat>:<message>`` and ``<project>|<number>``, and a deadline
    phrase answers what ``deadlines`` says (an ISO time, a source and optionally findings), else blank.

    ``created`` holds one Notion-shaped payload for each Job Order raised, so a test reads
    properties and children. ``page_delay`` makes the page appear after that many polls;
    ``lost_confirms`` makes that many confirms write and then fail to answer. ``down`` makes
    every call fail as an unreachable API does.
    """

    def __init__(
        self,
        *,
        departments=("TOC", "VID"),
        deadlines: dict | None = None,
        fail_create: bool = False,
        wrong_name: bool = False,
        fail_retrieve: int = 0,
        fail_find: int = 0,
        create_error: Exception | None = None,
        page_delay: int = 0,
        lost_confirms: int = 0,
    ):
        self.departments = list(departments)
        self.deadlines = deadlines or {}
        self.fail_create = fail_create
        self.wrong_name = wrong_name
        self.fail_retrieve = fail_retrieve  # fail this many ``jo get`` by page calls first
        self.fail_find = fail_find  # fail this many ``jo find`` calls first
        self.create_error = create_error  # raised instead of the default failure
        self.page_delay = page_delay
        self.lost_confirms = lost_confirms
        self.down = False
        self.withhold_down: Exception | None = None  # raised by ``text withhold``
        self.withhold_calls: list[list[str]] = []
        self.standing_cc_answer: str | None = None  # ``standing-cc``: a T6 ID or a page ID
        self.find_calls = 0
        self.create_calls = 0  # confirm attempts
        self.previews: list = []
        self.confirms: list = []
        self.created: list[dict] = []
        self.pages: dict[str, dict] = {}
        self.raised: dict[str, str] = {}  # idempotency key -> jo_id
        self.polls: dict[str, int] = {}
        self._counter = 0

    def _jo_gate(self) -> None:
        if self.down:
            raise ApiUnreachable("t6 API unavailable: connection refused")

    def _rules(self, inputs) -> dict:
        message = inputs.message
        if message is None:  # a plan from before #151 sends its own rule fields
            f = inputs.fields
            return dict(
                name=f.get("name"),
                jo_number=f.get("number"),
                department=f.get("department"),
                deadline=f.get("deadline"),
                deadline_source="stated" if f.get("deadline") else "blank",
                telegram_key=f.get("telegram_key"),
            )
        header = _HEADER.match(message.header_text)
        token = header["department"].upper() if header else (message.department_text or "").upper()
        department = token if token in self.departments else None
        digits = header["number"] if header else "".join(c for c in message.number_text or "" if c.isdigit())
        deadline, source, *rest = self.deadlines.get(message.deadline_text, (None, "blank"))
        return dict(
            name=message.header_text.strip(),
            jo_number=f"#{int(digits)}" if digits else None,
            department=department,
            deadline=deadline,
            deadline_source=source,
            deadline_findings=rest[0] if rest else [],
            department_findings=[{"code": "unknown_department", "params": {"token": token}}]
            if token and department is None
            else [],
            telegram_key=f"{message.chat_id}:{message.message_id}",
            duplicate_key=f"{message.project}|{int(digits)}" if message.project and digits else None,
        )

    async def preview_job_order(self, inputs):
        from jo_watchdog.core.job_order import RaisePreview

        self._jo_gate()
        self.previews.append(inputs)
        return RaisePreview(token=f"token-{len(self.previews)}", **self._rules(inputs))

    async def standing_cc(self) -> str | None:
        self._jo_gate()
        return self.standing_cc_answer

    async def withhold_texts(self, texts: list[str]) -> list[str]:
        """``text withhold``: a stand-in that hides the value after ``password:``, in order."""
        self.withhold_calls.append(list(texts))
        if self.withhold_down:
            raise self.withhold_down
        return [re.sub(r"(?i)(password:\s*)\S+", r"\1[withheld]", text) for text in texts]

    async def confirm_job_order(self, inputs, token: str) -> str:
        self._jo_gate()
        self.create_calls += 1
        self.confirms.append(inputs)
        if self.fail_create:
            raise self.create_error or T6ApiError("t6 API unavailable: raise failed")
        key = inputs.idempotency_key
        if key not in self.raised:  # the same key is the same Job Order, as in the API
            self._counter += 1
            jo_id = f"t6j_{self._counter}"
            page_id = f"page-{self._counter}"
            rules = self._rules(inputs)
            self.raised[key] = jo_id
            self.created.append(self._payload(inputs, rules))
            date = (inputs.fields or {}).get("date")
            self.pages[page_id] = {
                "id": page_id,
                "jo_id": jo_id,
                "url": f"https://www.notion.so/{page_id.replace('-', '')}",
                "name": "WRONG NAME" if self.wrong_name else rules["name"],
                "telegram_key": rules["telegram_key"],
                "date": live_date({"start": date}) if date else None,
                "deadline": live_date({"start": rules["deadline"]}) if rules["deadline"] else None,
            }
            self.polls[jo_id] = 0
        if self.lost_confirms:
            self.lost_confirms -= 1
            raise T6ApiError("t6 API gave no answer in 30 s")
        return self.raised[key]

    async def job_order_page(self, jo_id: str):
        self._jo_gate()
        self.polls[jo_id] += 1
        if self.polls[jo_id] <= self.page_delay:
            return None
        return next(p["id"] for p in self.pages.values() if p["jo_id"] == jo_id)

    def _visible(self, page: dict) -> bool:
        return self.page_delay == 0 or self.polls[page["jo_id"]] > self.page_delay

    async def find_job_orders(self, chat_id: int, message_id: int):
        self._jo_gate()
        self.find_calls += 1
        if self.fail_find:
            self.fail_find -= 1
            raise T6ApiError("t6 API unavailable: find failed")
        key = f"{chat_id}:{message_id}"
        return [
            PageRef(id=page["id"], url=page["url"])
            for page in self.pages.values()
            if page["telegram_key"] == key and self._visible(page)
        ]

    async def read_job_order(self, page_id: str) -> dict:
        self._jo_gate()
        if self.fail_retrieve:
            self.fail_retrieve -= 1
            raise T6ApiError("t6 API unavailable: get failed")
        page = self.pages[page_id]
        return {
            "page_id": page_id,
            "name": page["name"],
            "telegram_key": page["telegram_key"],
            "date": (page["date"] or {}).get("start"),
            "deadline": (page["deadline"] or {}).get("start"),
        }

    @staticmethod
    def _payload(inputs, rules: dict) -> dict:
        """The Notion-shaped page a raise writes, from the fields and the rule values."""
        fields = inputs.fields or {}

        def text(value):
            return {"rich_text": [{"type": "text", "text": {"content": value}}]}

        names = {
            "title": "Title", "subtitle": "Subtitle", "task": "Task", "purpose": "Purpose",
            "source": "Source / Source Drive", "platform": "Platform",
            "link_settings": "Link Settings", "watermark": "Watermark",
            "mode_of_transfer": "Mode of Transfer", "blocker": "Blocker ", "pending": "Pending",
            "notes": "JO Notes", "watchdog_notes": "Watchdog Notes",
        }
        props: dict = {
            "Name": {"title": [{"text": {"content": rules["name"]}}]},
            "Telegram Key": text(rules["telegram_key"]),
            "Status": {"status": {"name": fields.get("status")}},
        }
        if fields.get("date"):
            props["Date"] = {"date": {"start": fields["date"]}}
        if rules.get("jo_number"):
            props["JO Number"] = text(rules["jo_number"])
        if rules.get("department"):
            props["Department"] = {"select": {"name": rules["department"]}}
        if rules.get("deadline"):
            props["Deadline"] = {"date": {"start": rules["deadline"]}}
        for field, prop in names.items():
            if field in fields:
                props[prop] = text(fields[field])
        for field, prop in (("assigned_ids", "Assigned"), ("cc_ids", "cc"), ("deliverable_ids", "Deliverable")):
            if field in fields:
                props[prop] = {"relation": [{"id": i} for i in fields[field]]}
        project = inputs.message.project if inputs.message else fields.get("project_id")
        props["Projects"] = {"relation": [{"id": project}] if project else []}
        children = [
            {"paragraph": {"rich_text": [{"type": "text", "text": {"content": line}}]}}
            for line in (fields.get("source_text") or "").split("\n")
        ]
        return {"properties": props, "children": children}


NOTION_FIXTURES = Path(__file__).parent / "fixtures" / "notion"


def _recorded(name: str) -> dict:
    return json.loads((NOTION_FIXTURES / name).read_text(encoding="utf-8"))


class RecordedDirectory:
    """A fake t6 API client for the caches: the recorded Contacts, Projects and Department
    fixtures, shaped as ``person list``, ``project list`` and ``departments`` answer.

    ``fail`` makes every call raise as an API outage would; ``empty_people`` answers an
    empty Contacts list.
    """

    def __init__(self, contact_files=("contacts_query_page1.json", "contacts_query_page2.json")):
        self.contact_files = contact_files
        self.fail = False
        self.empty_people = False
        self.calls: list[str] = []

    def _enter(self, name: str) -> None:
        self.calls.append(name)
        if self.fail:
            raise ApiUnreachable("t6 API unavailable: 503 down")

    async def people(self) -> list[dict]:
        self._enter("people")
        if self.empty_people:
            return []
        people = []
        for name in self.contact_files:
            for page in _recorded(name)["results"]:
                person = person_from_page(page)
                user_id = person.telegram_user_id
                people.append(
                    {
                        "page_id": person.page_id,
                        "name": person.name,
                        "telegram_user_id": int(user_id) if user_id.isdigit() else None,
                        "telegram_handle": person.telegram_handle or None,
                        "aliases": person.aliases,
                    }
                )
        return people

    async def projects(self) -> list[dict]:
        self._enter("projects")
        return [
            {
                "page_id": p.page_id,
                "name": p.name,
                "project_code": (page["properties"]["Project Code"]["select"] or {}).get("name"),
            }
            for page in _recorded("projects_query.json")["results"]
            for p in [project_from_page(page)]
        ]



def resolved(
    page_id: str, name: str, method: str = "normalized", t6_id: str | None = None
) -> ContactMatch:
    """A ``person find`` answer with a match (``t6_id``: the Contact's T6 Person ID)."""
    return ContactMatch(
        status="resolved", method=method, page_id=page_id, name=name, t6_id=t6_id
    )


def project(page_id: str, code: str) -> ProjectMatch:
    """A ``project resolve`` answer with the Project in rotation."""
    return ProjectMatch(page_id=page_id, code=code)


class FakeMatcher:
    """Canned ``person find`` and ``project resolve`` answers, keyed by the case-folded text.

    A text with no canned answer is unresolved. ``down`` makes every call fail the way an
    unreachable API does.
    """

    def __init__(
        self,
        contacts: dict[str, ContactMatch] | None = None,
        projects: dict[str, ProjectMatch] | None = None,
    ):
        self.contacts = {k.casefold(): v for k, v in (contacts or {}).items()}
        self.projects = {k.casefold(): v for k, v in (projects or {}).items()}
        self.calls: list[list[str]] = []
        self.project_calls: list[str] = []
        self.down = False

    async def find_project(self, text: str) -> ProjectMatch:
        self.project_calls.append(text)
        if self.down:
            raise ApiUnreachable("t6 API unavailable: connection refused")
        folded = text.casefold()
        if folded in self.projects:
            return self.projects[folded]
        words = set(re.findall(r"[\w-]+", folded))  # a code inside longer text (``in_text``)
        return next((m for code, m in self.projects.items() if code in words), ProjectMatch())

    async def find_contacts(self, mentions: list[str]) -> list[ContactMatch]:
        self.calls.append(list(mentions))
        if self.down:
            raise ApiUnreachable("t6 API unavailable: connection refused")
        return [
            self.contacts.get(m.casefold(), ContactMatch(status="unresolved")) for m in mentions
        ]


LLM_RECORDED = json.loads(
    (Path(__file__).parent / "fixtures" / "llm_run_recorded.json").read_text(encoding="utf-8")
)


def llm_answer(name: str):
    """A recorded ``llm run`` outcome: ``Done`` or ``Refused``."""
    recorded = LLM_RECORDED[name]
    model = Done if recorded["outcome"] == "done" else Refused
    return model.model_validate(recorded)


class FakeLlmClient:
    """Stands in for ``t6.client.Client``: scripted ``llm run`` answers (an outcome or an
    exception; the last repeats), and the requests it was sent."""

    def __init__(self, *answers):
        self.answers = list(answers) or [llm_answer("jo")]
        self.requests: list = []

    def llm_run(self, inputs):
        self.requests.append(inputs)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


def decide_answer(*probabilities, model="typesafe/jev-1.14-20261014", prompt=210, completion=31):
    """A ``llm decide`` ``Done`` for ``(match, maybe, no)`` triples, one per candidate in order."""
    return Done(
        result={
            "answers": [
                {
                    "index": i,
                    "choice": max(zip(p, ("match", "maybe", "no"), strict=True))[1],
                    "probabilities": dict(zip(("match", "maybe", "no"), p, strict=True)),
                }
                for i, p in enumerate(probabilities)
            ],
            "model": model,
            "provider": "decisions",
            "tokens": {"prompt": prompt, "completion": completion, "total": prompt + completion},
            "zdr": False,
        }
    )


class FakeDecideClient:
    """Stands in for ``t6.client.Client``: scripted ``llm decide`` answers (an outcome or an
    exception; the last repeats), and the requests it was sent."""

    def __init__(self, *answers):
        self.answers = list(answers) or [decide_answer((0.95, 0.04, 0.01), (0.02, 0.08, 0.90))]
        self.requests: list = []

    def llm_decide(self, inputs):
        self.requests.append(inputs)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer
