"""People and Project resolution over the recorded Notion fixtures and corpus."""

import json
import re
import tempfile
import unittest
from pathlib import Path

from jo_watchdog.adapters.t6api import T6ApiError
from jo_watchdog.app.caches import (
    CacheManager,
    person_from_page,
    project_from_page,
)
from jo_watchdog.app.config import Config
from jo_watchdog.core.resolve import (
    ContactMatch,
    MentionEntity,
    PeopleResolution,
    ProjectMatch,
    ProjectResolution,
    ResolveSettings,
    relation_properties,
    resolve_project,
    split_mentions,
)
from jo_watchdog.ledger.db import Ledger
from tests.fakes import FakeMatcher, project, resolved

FIXTURES = Path(__file__).parent / "fixtures"
NOTION = FIXTURES / "notion"

PRIYA = "c0000000-0000-4000-8000-000000000001"
RAMON = "c0000000-0000-4000-8000-000000000002"
LENA = "c0000000-0000-4000-8000-000000000003"
MIRA = "c0000000-0000-4000-8000-000000000004"
PAUL = "c0000000-0000-4000-8000-000000000005"
NINA = "c0000000-0000-4000-8000-000000000006"
JOSHUA = "c0000000-0000-4000-8000-000000000007"
RAE = "c0000000-0000-4000-8000-000000000008"
JASON = "c0000000-0000-4000-8000-000000000009"
LARA = "c0000000-0000-4000-8000-000000000013"


def _load(name):
    return json.loads((NOTION / name).read_text(encoding="utf-8"))


def people():
    pages = (
        _load("contacts_query_page1.json")["results"]
        + _load("contacts_query_page2.json")["results"]
    )
    return [person_from_page(p) for p in pages]


def projects():
    return [project_from_page(p) for p in _load("projects_query.json")["results"]]


SETTINGS = ResolveSettings(standing_cc_contact_page_id=PRIYA)


ANSWERS = {
    "Ramon": resolved(RAMON, "Ramon Reyes"),
    "Lena": resolved(LENA, "Lena Cruz"),
    "ate Lena": resolved(LENA, "Lena Cruz"),
    "Mira": resolved(MIRA, "Mira Santos"),
    "Paul": resolved(PAUL, "Paul Garcia", "fuzzy"),
    "@user_two": resolved(RAE, "Rae", "normalized"),
    "@user_four": resolved(JASON, "Jason", "normalized"),
    "@user_six": resolved(LARA, "Lara", "normalized"),
    "@priya": resolved(PRIYA, "Priya", "normalized"),
    "5550001": resolved(MIRA, "Mira Santos", "exact_id"),
    "Kim": ContactMatch(status="ambiguous"),
}
for _match in ANSWERS.values():  # registered in T6, so no "no Person ID" finding
    if _match.page_id:
        _match.t6_id = f"t6n_{_match.page_id}"


class ResolvedBy(unittest.IsolatedAsyncioTestCase):
    """``CacheManager.resolve_people`` over a fake matcher with canned ``person find`` answers."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ledger = Ledger(Path(tmp.name) / "ledger.db")
        self.addCleanup(self.ledger.close)
        self.matcher = FakeMatcher(ANSWERS)
        config = Config(
            intake_chat_id=-100123,
            intake_topic_id=5,
            ops_chat_id=-100999,
            ops_topic_id=7,
            standing_cc_contact_page_id=PRIYA,
        )
        self.caches = CacheManager(
            ledger=self.ledger, directory=None, matcher=self.matcher, config=config
        )
        self.caches._apply(
            "people", [p.model_dump(mode="json") for p in people()], "2026-10-22T00:00:00Z"
        )
        self.caches._apply(
            "projects", [p.model_dump(mode="json") for p in projects()], "2026-10-22T00:00:00Z"
        )

    async def resolve(self, assignees, cc=None, entities=()) -> PeopleResolution:
        return await self.caches.resolve_people(assignees, cc, list(entities))


class FixtureParsingTests(unittest.TestCase):
    def test_contacts_fields_read_by_live_property_names(self):
        by_id = {p.page_id: p for p in people()}
        self.assertEqual(by_id[NINA].telegram_handle, "@user_three")
        self.assertEqual(by_id[NINA].aliases, ["Nina", "sam", "KUYA sam"])
        self.assertEqual(by_id[MIRA].telegram_user_id, "5550001")

    def test_projects(self):
        self.assertEqual(len(projects()), 12)
        self.assertTrue(all(p.name for p in projects()))


class MentionSplittingTests(unittest.TestCase):
    CASES = {
        "Paul / Ramon": ["Paul", "Ramon"],
        "Paul, Ramon / Mira, Lena": ["Paul", "Ramon", "Mira", "Lena"],
        "Ramon & @user_two": ["Ramon", "@user_two"],
        "Paul and Ramon": ["Paul", "Ramon"],
        "Paul\nRamon": ["Paul", "Ramon"],
        "for @user_one  thank youuu": ["@user_one"],
        "@user_eight thankies!!": ["@user_eight"],
        "@user_two tyvmmmm": ["@user_two"],
        "ate Nina thank u!": ["ate Nina"],
        "Lena po": ["Lena"],
        "Mira / @user_two kung sakaling umabot kay user_two": [
            "Mira",
            "@user_two kung sakaling umabot kay user_two",
        ],
        "ate Lena / Ramon, if abutin the next day": [
            "ate Lena",
            "Ramon if abutin the next day",
        ],
        "Ramon @user_two // @user_three": ["Ramon", "@user_two", "@user_three"],
        "@user_three  @user_one": ["@user_three", "@user_one"],
        "ate @user_four / Ramon": ["ate", "@user_four", "Ramon"],
        "Ramon but maybe Lena": ["Ramon but maybe Lena"],
        "": [],
    }

    def test_corpus_shapes(self):
        for raw, expected in self.CASES.items():
            with self.subTest(raw=raw):
                self.assertEqual(split_mentions(raw), expected)


class PeopleResolutionTests(ResolvedBy):
    async def test_statuses_follow_the_api_answer(self):
        out = await self.resolve("Ramon / Kim / Zzyzx")
        self.assertEqual(
            [(m.raw, m.status, m.method, m.page_id) for m in out.mentions],
            [
                ("Ramon", "resolved", "normalized", RAMON),
                ("Kim", "ambiguous", None, None),
                ("Zzyzx", "unresolved", None, None),
            ],
        )
        self.assertEqual(out.assigned, [RAMON])
        self.assertEqual(
            [(f.code, f.params["raw"]) for f in out.findings],
            [("ambiguous_person", "Kim"), ("unresolved_person", "Zzyzx")],
        )

    async def test_one_call_carries_every_mention_of_the_message(self):
        await self.resolve("Ramon, Lena", "@user_two")
        self.assertEqual(self.matcher.calls, [["Ramon", "Lena", "@user_two"]])

    async def test_no_mentions_make_no_call(self):
        out = await self.resolve(None, "")
        self.assertEqual(self.matcher.calls, [])
        self.assertEqual(out.cc, [PRIYA])

    async def test_a_text_mention_entity_asks_by_telegram_user_id(self):
        entity = MentionEntity(text="Jadey", user_id="5550001")
        out = await self.resolve("Jadey", entities=[entity])
        self.assertEqual(self.matcher.calls, [["5550001"]])
        self.assertEqual((out.mentions[0].method, out.assigned), ("exact_id", [MIRA]))

    async def test_a_mention_entity_asks_by_handle(self):
        entity = MentionEntity(text="@user_two", handle="user_two")
        out = await self.resolve("@user_two", entities=[entity])
        self.assertEqual(self.matcher.calls, [["@user_two"]])
        self.assertEqual(out.assigned, [RAE])

    async def test_an_entity_cannot_supply_identity_for_another_mention(self):
        entity = MentionEntity(text="Jo", user_id="123")
        await self.resolve("Lena", entities=[entity])
        self.assertEqual(self.matcher.calls, [["Lena"]])

    async def test_a_handle_entity_binds_only_a_whole_handle_token(self):
        entity = MentionEntity(text="@rae", handle="rae")
        await self.resolve("@raed", entities=[entity])
        self.assertEqual(self.matcher.calls, [["@raed"]])

    async def test_a_bare_telegram_id_is_passed_on_as_written(self):
        out = await self.resolve("5550001")
        self.assertEqual(self.matcher.calls, [["5550001"]])
        self.assertEqual(out.assigned, [MIRA])

    async def test_a_fuzzy_match_is_written_with_a_finding(self):
        out = await self.resolve("Paul / Ramon")
        self.assertEqual(out.assigned, [PAUL, RAMON])
        self.assertEqual([f.code for f in out.findings], ["fuzzy_person"])
        self.assertEqual(out.findings[0].params["contact"], "Paul Garcia")

    async def test_standing_cc_not_duplicated_when_named(self):
        out = await self.resolve("Lena", "@priya, @user_six")
        self.assertEqual(out.cc, [PRIYA, LARA])

    async def test_a_hedged_mention_is_ambiguous_and_never_sent(self):
        out = await self.resolve("Ramon but maybe Lena / Kim", "@nobody")
        self.assertEqual(self.matcher.calls, [["Kim", "@nobody"]])
        self.assertEqual(out.assigned, [])
        self.assertEqual(
            [(f.code, f.params["raw"], f.params["role"]) for f in out.findings],
            [
                ("ambiguous_person", "Ramon but maybe Lena", "assignee"),
                ("ambiguous_person", "Kim", "assignee"),
                ("unresolved_person", "@nobody", "cc"),
            ],
        )

    async def test_conditional_mention_stays_ambiguous_with_complete_wording(self):
        out = await self.resolve("Lena if available", "Mira kung free")
        self.assertEqual(self.matcher.calls, [])
        self.assertEqual(out.assigned, [])
        self.assertEqual(out.cc, [PRIYA])
        self.assertEqual(
            [(m.raw, m.status) for m in out.mentions],
            [("Lena if available", "ambiguous"), ("Mira kung free", "ambiguous")],
        )

    async def test_unconditional_mentions_beside_a_conditional_one_still_resolve(self):
        out = await self.resolve("ate Lena / Ramon, if abutin the next day")
        self.assertEqual(self.matcher.calls, [["ate Lena", "ate", "Lena"]])
        self.assertEqual(out.assigned, [LENA])
        self.assertEqual(out.findings[0].params["raw"], "Ramon if abutin the next day")

    async def test_names_without_separator_split_only_when_every_word_is_known(self):
        out = await self.resolve("Ramon @user_two Mira Lena")
        self.assertEqual(out.assigned, [RAMON, RAE, MIRA, LENA])
        self.assertEqual(out.findings, [])
        self.assertEqual(len(self.matcher.calls), 1)
        out = await self.resolve("Mira Zzyzx")
        self.assertEqual(out.assigned, [])
        self.assertEqual(out.findings[0].params["raw"], "Mira Zzyzx")

    async def test_a_title_before_a_handle_belongs_to_the_handle(self):
        out = await self.resolve("ate @user_four / Ramon")
        self.assertEqual(self.matcher.calls[0][:3], ["ate", "@user_four", "Ramon"])
        self.assertEqual(out.assigned, [JASON, RAMON])
        self.assertEqual(out.findings, [])
        self.assertEqual(out.mentions[0].raw, "ate @user_four")

    async def test_a_known_name_before_a_handle_is_its_own_person(self):
        out = await self.resolve("Ramon @user_two")
        self.assertEqual(out.assigned, [RAMON, RAE])

    async def test_an_unknown_title_and_handle_make_one_unresolved_mention(self):
        out = await self.resolve("ate @nobody")
        self.assertEqual([(m.raw, m.status) for m in out.mentions], [("ate @nobody", "unresolved")])

    async def test_assignees_go_to_the_api_as_t6_person_ids(self):
        self.matcher.contacts["ramon"] = resolved(RAMON, "Ramon Reyes", t6_id="t6n_ramon")
        self.matcher.contacts["mira"] = resolved(MIRA, "Mira Santos", t6_id="t6n_mira")
        self.matcher.contacts["lena"] = resolved(LENA, "Lena Cruz")  # not registered yet
        out = await self.resolve("Ramon, Mira / Lena", "Ramon")
        self.assertEqual(out.assigned, [RAMON, MIRA, LENA])
        self.assertEqual(out.assignee_ids, ["t6n_ramon", "t6n_mira"])
        # The one left out of the API call is a finding, not silence; cc never needs an ID.
        self.assertEqual(
            [(f.code, f.params["raw"]) for f in out.findings],
            [("assignee_no_person_id", "Lena")],
        )

    async def test_no_assignee_resolved_gives_no_person_ids(self):
        out = await self.resolve("@user_eight")
        self.assertEqual(out.assignee_ids, [])

    async def test_no_standing_cc_config_means_no_standing_cc(self):
        self.caches.settings = ResolveSettings()
        out = await self.resolve("Ramon")
        self.assertEqual(out.cc, [])

    async def test_when_the_api_cannot_answer_nothing_is_resolved(self):
        self.matcher.down = True
        with self.assertRaises(T6ApiError):
            await self.resolve("Ramon")


PROJECT_ANSWERS = {
    # a code only one Project holds
    "MFG": project("p-mfg", "MFG"),
    # a code several Projects hold: one is in rotation, so the API names it
    "TWIN": project("p-twin-in", "TWIN"),
    # a code inside longer text
    "MFG - TOC - Season 2": project("p-mfg", "MFG"),
    "COLD CUT": project("p-cold-cut", "COLD CUT"),
    # a code shared where none or several Projects are in rotation: the API names none
    "NONE-IN": ProjectMatch(),
    "TWO-IN": ProjectMatch(),
}


class ProjectTests(ResolvedBy):
    def setUp(self):
        super().setUp()
        self.matcher.projects = {k.casefold(): v for k, v in PROJECT_ANSWERS.items()}

    async def test_a_unique_code(self):
        out = await self.caches.resolve_project("MFG", None)
        self.assertEqual((out.page_id, out.code, out.findings), ("p-mfg", "MFG", []))
        self.assertEqual(self.matcher.project_calls, ["MFG"])

    async def test_a_code_reused_by_several_projects_with_one_in_rotation(self):
        out = await self.caches.resolve_project("TWIN", None)
        self.assertEqual((out.page_id, out.findings), ("p-twin-in", []))

    async def test_a_reused_code_with_none_or_several_in_rotation_stays_unresolved(self):
        for text in ("NONE-IN", "TWO-IN"):
            with self.subTest(text=text):
                out = await self.caches.resolve_project(text, None)
                self.assertIsNone(out.page_id)
                self.assertEqual(
                    [(f.code, f.params) for f in out.findings],
                    [("unresolved_project", {"raw": text})],
                )

    async def test_a_code_inside_longer_text(self):
        out = await self.caches.resolve_project("  MFG - TOC - Season 2 ", None)
        self.assertEqual(out.page_id, "p-mfg")
        self.assertEqual(self.matcher.project_calls, ["MFG - TOC - Season 2"])

    async def test_the_models_project_is_asked_when_the_header_names_none(self):
        out = await self.caches.resolve_project("Unrelated header", "MFG")
        self.assertEqual(out.page_id, "p-mfg")
        self.assertEqual(self.matcher.project_calls, ["Unrelated header", "MFG"])

    async def test_a_header_match_is_not_asked_twice(self):
        out = await self.caches.resolve_project("MFG - TOC - Season 2", "TWIN")
        self.assertEqual(out.page_id, "p-mfg")
        self.assertEqual(self.matcher.project_calls, ["MFG - TOC - Season 2"])

    async def test_an_unknown_text_is_unresolved_with_a_finding(self):
        out = await self.caches.resolve_project("NOPE", None)
        self.assertIsNone(out.page_id)
        self.assertEqual([f.code for f in out.findings], ["unresolved_project"])

    async def test_a_blank_text_asks_nothing_and_finds_nothing(self):
        for text in (None, "", "  "):
            out = await self.caches.resolve_project(text, None)
            self.assertEqual((out.page_id, out.findings), (None, []))
        self.assertEqual(self.matcher.project_calls, [])

    async def test_when_the_api_cannot_answer_nothing_is_resolved(self):
        self.matcher.down = True
        with self.assertRaises(T6ApiError):
            await self.caches.resolve_project("MFG", None)


class RelationPropertyTests(unittest.TestCase):
    def test_only_resolved_ids_and_no_coordinator(self):
        people_out = PeopleResolution(assigned=[RAMON], cc=[PRIYA])
        props = relation_properties(people_out, resolve_project("NOPE", None))
        self.assertEqual(
            props,
            {
                "Assigned": {"relation": [{"id": RAMON}]},
                "cc": {"relation": [{"id": PRIYA}]},
                "Projects": {"relation": []},
            },
        )
        self.assertNotIn("🗂️ Coordinator", props)
        named = relation_properties(people_out, ProjectResolution(page_id="p-mfg", code="MFG"))
        self.assertEqual(named["Projects"], {"relation": [{"id": "p-mfg"}]})


class CorpusPeopleTests(ResolvedBy):
    """Every corpus JO resolves without error; Priya is always in cc."""

    ASSIGNED_RE = re.compile(r"^\s*assigned(?: to)?\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
    CC_RE = re.compile(r"^\s*cc\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)

    async def test_every_jo_resolves_and_keeps_standing_cc(self):
        corpus = json.loads((FIXTURES / "redacted_corpus.json").read_text(encoding="utf-8"))
        for i, text in enumerate(corpus):
            a = self.ASSIGNED_RE.search(text)
            c = self.CC_RE.search(text)
            out = await self.resolve(a.group(1) if a else None, c.group(1) if c else None)
            self.assertIn(PRIYA, out.cc, i)
            self.assertEqual(len(out.assigned), len(set(out.assigned)), i)
            for mention in out.mentions:
                if mention.status != "resolved":
                    self.assertIsNone(mention.page_id)


if __name__ == "__main__":
    unittest.main()


class MentionEntityTests(unittest.TestCase):
    def test_slices_by_utf16_offsets_and_keeps_user_ids(self):
        from jo_watchdog.core.resolve import mention_entities

        message = {
            "text": "😀 @raed and Benjamin",
            "entities": [
                {"type": "mention", "offset": 3, "length": 5},
                {"type": "text_mention", "offset": 13, "length": 8, "user": {"id": 222}},
                {"type": "bold", "offset": 0, "length": 2},
            ],
        }
        self.assertEqual(
            mention_entities(message),
            [
                MentionEntity(text="@raed", handle="raed"),
                MentionEntity(text="Benjamin", user_id="222"),
            ],
        )
