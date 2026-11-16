"""The t6 API seam: outcome mapping, MISC in departments, threads, and the cache over it."""

import asyncio
import tempfile
import threading
import unittest
from datetime import UTC, datetime, time
from pathlib import Path

from t6.client import APIError, Unavailable
from t6.verbs import Ambiguous, Done, Refused

from jo_watchdog.adapters.protocols import (
    ContactMatcher,
    DeliverableSource,
    DirectoryReader,
    JobOrderReader,
    Matcher,
    ProjectMatcher,
)
from jo_watchdog.adapters.t6api import T6Api, T6ApiError
from jo_watchdog.app.caches import CacheManager, person_from_api
from jo_watchdog.core.deadline import resolve_deadline
from jo_watchdog.ledger.db import Ledger
from tests.fakes import FakeMatcher, resolved
from tests.test_caches import config

PERSON = {
    "page_id": "c-ana",
    "name": "Ana Reyes",
    "telegram_user_id": 111,
    "telegram_handle": "@ana_r",
    "aliases": ["Ana", "Ate A"],
    "end_of_shift": "21:00:00",
}


class FakeClient:
    """Stands in for ``t6.client.Client``: scripted outcomes, recorded inputs."""

    def __init__(self, **answers):
        self.answers = answers
        self.inputs: dict[str, object] = {}
        self.threads: dict[str, int] = {}

    def _answer(self, name, inputs):
        self.inputs[name] = inputs
        self.threads[name] = threading.get_ident()
        answer = self.answers[name]
        if isinstance(answer, Exception):
            raise answer
        return answer() if callable(answer) else answer

    def person_list(self, inputs):
        return self._answer("person_list", inputs)

    def project_list(self, inputs):
        return self._answer("project_list", inputs)

    def departments(self, inputs):
        return self._answer("departments", inputs)

    def person_find(self, inputs):
        return self._answer("person_find", inputs)

    def project_resolve(self, inputs):
        return self._answer("project_resolve", inputs)

    def deliverable_candidates(self, inputs):
        return self._answer("deliverable_candidates", inputs)

    def jo_find(self, inputs):
        return self._answer("jo_find", inputs)

    def jo_get(self, inputs):
        return self._answer("jo_get", inputs)


class OutcomeMappingTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_seam_satisfies_both_protocols(self):
        api = T6Api(FakeClient())
        self.assertIsInstance(api, DirectoryReader)
        self.assertIsInstance(api, JobOrderReader)

    async def test_unavailable_refused_api_error_and_ambiguous_raise_one_error(self):
        for answer, text in (
            (Unavailable("connection refused"), "unavailable: connection refused"),
            (Refused(rule="permission", reason="no grant"), "refused (permission): no grant"),
            (APIError("API returned HTTP 502", 502), "HTTP 502"),
            (Ambiguous(candidates=["a", "b"]), "ambiguous"),
        ):
            with self.subTest(text=text):
                api = T6Api(FakeClient(person_list=answer))
                with self.assertRaises(T6ApiError) as caught:
                    await api.people()
                self.assertIn(text, str(caught.exception))

    async def test_departments_ask_for_misc_and_return_names_in_order(self):
        client = FakeClient(
            departments=Done(
                result={"departments": [{"id": "1", "name": "TOC"}, {"id": "2", "name": "MISC"}]}
            )
        )
        self.assertEqual(await T6Api(client).departments(), ["TOC", "MISC"])
        self.assertTrue(client.inputs["departments"].include_misc)

    async def test_find_returns_page_refs_oldest_first_as_the_api_gave_them(self):
        client = FakeClient(
            jo_find=Done(result={"job_orders": [{"page_id": "p-1"}, {"page_id": "p-2"}]})
        )
        refs = await T6Api(client).find_job_orders("-100:7")
        self.assertEqual([r.id for r in refs], ["p-1", "p-2"])
        self.assertTrue(refs[0].url.endswith("p1"))
        self.assertEqual(client.inputs["jo_find"].telegram_key, "-100:7")

    async def test_read_job_order_returns_the_job_order(self):
        order = {"page_id": "p-1", "name": "N", "telegram_key": "-100:7", "date": None}
        client = FakeClient(jo_get=Done(result={"job_order": order}))
        self.assertEqual(await T6Api(client).read_job_order("p-1"), order)
        self.assertEqual(client.inputs["jo_get"].ref, "p-1")


def found(match, method="normalized", *, review=False, candidates=()):
    return {
        "match": match,
        "method": method,
        "score": 1.0,
        "needs_review": review,
        "reason": None,
        "candidates": list(candidates),
    }


class FindContactsTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_call_answers_every_mention_in_order(self):
        client = FakeClient(
            person_find=Done(
                result={
                    "results": [
                        found({"page_id": "c-ana", "name": "Ana Reyes"}, "exact_id"),
                        found(None, "review", review=True, candidates=[{"page_id": "c-1"}]),
                        found(None, "none"),
                    ]
                }
            )
        )
        matches = await T6Api(client).find_contacts(["@ana_r", "Kim", "Zzyzx"])
        self.assertEqual(client.inputs["person_find"].mentions, ["@ana_r", "Kim", "Zzyzx"])
        self.assertEqual(
            [(m.status, m.method, m.page_id, m.name) for m in matches],
            [
                ("resolved", "exact_id", "c-ana", "Ana Reyes"),
                ("ambiguous", None, None, None),
                ("unresolved", None, None, None),
            ],
        )

    async def test_more_mentions_than_one_call_takes_go_in_batches(self):
        calls = []

        def answer():
            batch = calls[-1]
            return Done(result={"results": [found(None, "none") for _ in batch]})

        class Batching(FakeClient):
            def person_find(self, inputs):
                calls.append(inputs.mentions)
                return answer()

        matches = await T6Api(Batching()).find_contacts([f"n{i}" for i in range(120)])
        self.assertEqual([len(c) for c in calls], [50, 50, 20])
        self.assertEqual(len(matches), 120)

    async def test_a_down_or_refusing_api_raises(self):
        for answer in (Unavailable("down"), Refused(rule="input", reason="too many")):
            with self.subTest(answer=answer):
                api = T6Api(FakeClient(person_find=answer))
                with self.assertRaises(T6ApiError):
                    await api.find_contacts(["Ana"])

    async def test_the_seam_satisfies_the_matcher_protocol(self):
        self.assertIsInstance(T6Api(FakeClient()), ContactMatcher)


def resolved_project(match, *, review=False):
    return {
        "match": match,
        "method": "normalized",
        "score": 1.0,
        "needs_review": review,
        "reason": None,
        "candidates": [],
    }


class FindProjectTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_text_is_searched_for_a_code_and_the_match_is_the_project(self):
        match = {"page_id": "p-mfg", "name": "Show", "project_code": "MFG"}
        client = FakeClient(project_resolve=Done(result=resolved_project(match)))
        found_project = await T6Api(client).find_project("MFG - TOC")
        self.assertEqual((found_project.page_id, found_project.code), ("p-mfg", "MFG"))
        sent = client.inputs["project_resolve"]
        self.assertEqual((sent.text, sent.in_text), ("MFG - TOC", True))

    async def test_no_match_or_a_review_leaves_the_project_unresolved(self):
        for result in (resolved_project(None), resolved_project(None, review=True)):
            with self.subTest(result=result):
                client = FakeClient(project_resolve=Done(result=result))
                self.assertIsNone((await T6Api(client).find_project("NOPE")).page_id)

    async def test_a_down_api_raises(self):
        api = T6Api(FakeClient(project_resolve=Unavailable("down")))
        with self.assertRaises(T6ApiError):
            await api.find_project("MFG")

    async def test_the_seam_satisfies_the_matcher_protocols(self):
        self.assertIsInstance(T6Api(FakeClient()), ProjectMatcher)
        self.assertIsInstance(T6Api(FakeClient()), Matcher)


class DeliverableCandidatesTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def answer(*items):
        return Done(result={"candidates": [
            {"page_id": i, "name": n, "status": None, "deliverable_class": None} for i, n in items
        ]})

    async def test_candidates_keep_the_api_order_and_the_page_id_and_name_shape(self):
        client = FakeClient(
            deliverable_candidates=self.answer(
                ("d2", "Moonglow Archive"), ("d1", "Moonglow FTR SCR")
            )
        )
        found = await T6Api(client).deliverable_candidates("proj-1")
        self.assertEqual(
            [(c.page_id, c.name) for c in found],
            [("d2", "Moonglow Archive"), ("d1", "Moonglow FTR SCR")],
        )
        self.assertEqual(client.inputs["deliverable_candidates"].project, "proj-1")

    async def test_a_deliverable_only_in_notion_comes_through_with_its_name(self):
        client = FakeClient(deliverable_candidates=self.answer(("d9", "Notion-only Promo")))
        found = await T6Api(client).deliverable_candidates("proj-1")
        self.assertEqual([(c.page_id, c.name) for c in found], [("d9", "Notion-only Promo")])

    async def test_no_candidates_is_an_empty_list(self):
        client = FakeClient(deliverable_candidates=self.answer())
        self.assertEqual(await T6Api(client).deliverable_candidates("proj-1"), [])

    async def test_a_masked_or_missing_name_fails_the_call(self):
        for name in ("[deliverable 3]", None, ""):
            with self.subTest(name=name):
                client = FakeClient(deliverable_candidates=self.answer(("d1", name)))
                with self.assertRaises(T6ApiError):
                    await T6Api(client).deliverable_candidates("proj-1")

    async def test_a_down_api_raises(self):
        api = T6Api(FakeClient(deliverable_candidates=Unavailable("down")))
        with self.assertRaises(T6ApiError):
            await api.deliverable_candidates("proj-1")

    async def test_the_seam_satisfies_the_source_protocol(self):
        self.assertIsInstance(T6Api(FakeClient()), DeliverableSource)


class ThreadingTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_hung_call_does_not_stall_the_event_loop_and_times_out(self):
        release = threading.Event()
        self.addCleanup(release.set)
        client = FakeClient(person_list=lambda: release.wait(5))
        api = T6Api(client, call_timeout=0.2)
        ticks = 0

        async def heartbeat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        try:
            with self.assertRaises(T6ApiError) as caught:
                await api.people()
        finally:
            beat.cancel()
        self.assertIn("no answer", str(caught.exception))
        self.assertGreater(ticks, 5)  # the loop kept running while the call hung
        self.assertNotEqual(client.threads["person_list"], threading.get_ident())


class CachesOverTheApiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ledger = Ledger(Path(tmp.name) / "ledger.db")
        self.addCleanup(self.ledger.close)
        self.now = datetime(2026, 10, 22, tzinfo=UTC)

    def client(self, **overrides):
        answers = dict(
            person_list=Done(result={"people": [PERSON]}),
            project_list=Done(
                result={"projects": [{"page_id": "proj-1", "name": "Show", "project_code": "SHW"}]}
            ),
            departments=Done(result={"departments": [{"id": "1", "name": "TOC"}]}),
        )
        answers.update(overrides)
        return FakeClient(**answers)

    def manager(self, client):
        return CacheManager(
            ledger=self.ledger,
            directory=T6Api(client),
            matcher=FakeMatcher({"Ana": resolved("c-ana", "Ana Reyes")}),
            config=config(),
            clock=lambda: self.now,
        )

    def test_a_contact_arrives_with_aliases_as_a_list_and_a_parsed_shift(self):
        person = person_from_api(PERSON)
        self.assertEqual(person.aliases, ["Ana", "Ate A"])
        self.assertEqual(person.end_of_shift, time(21))
        self.assertEqual(person.telegram_user_id, "111")

    def test_a_flexible_shift_and_missing_ids_become_blank(self):
        person = person_from_api({"page_id": "c-1", "name": "Bo", "end_of_shift": None})
        self.assertEqual((person.end_of_shift, person.telegram_user_id), ("", ""))
        self.assertEqual(person.aliases, [])

    async def test_refresh_loads_the_three_lists_and_a_restart_keeps_the_shift_a_time(self):
        await self.manager(self.client()).refresh()
        restarted = self.manager(self.client())
        restarted.load_persisted()
        self.assertTrue(restarted.ready)
        self.assertEqual(restarted.people[0].end_of_shift, time(21))
        self.assertEqual(restarted.projects[0].name, "Show")
        self.assertEqual(restarted.departments, ["TOC"])

    async def test_a_parsed_shift_reaches_the_deadline_rule_unchanged(self):
        caches = self.manager(self.client())
        await caches.refresh()
        shifts = (await caches.resolve_people("Ana", None)).assignee_shifts
        posted = datetime(2026, 10, 22, 9, 0, tzinfo=UTC)
        result = resolve_deadline(
            "end of shift", posted, [(s.name, s.end_of_shift) for s in shifts]
        )
        self.assertEqual((result.source, result.deadline.hour), ("shift", 21))

    async def test_an_api_outage_keeps_the_last_snapshot(self):
        caches = self.manager(self.client())
        await caches.refresh()
        down = self.manager(self.client(person_list=Unavailable("down")))
        down.load_persisted()
        self.now = datetime(2026, 10, 22, 6, tzinfo=UTC)
        await down.refresh()
        self.assertEqual([p.name for p in down.people], ["Ana Reyes"])
        self.assertEqual(self.ledger.cache_snapshot_count("people"), 1)

    async def test_a_refusal_for_a_missing_grant_keeps_the_snapshot_too(self):
        caches = self.manager(self.client())
        await caches.refresh()
        refused = self.manager(self.client(project_list=Refused(rule="permission", reason="x")))
        refused.load_persisted()
        await refused.refresh()
        self.assertEqual(refused.projects[0].name, "Show")


if __name__ == "__main__":
    unittest.main()
