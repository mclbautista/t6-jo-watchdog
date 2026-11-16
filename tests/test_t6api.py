"""The t6 API seam: outcome mapping, MISC in departments, threads, and the cache over it."""

import asyncio
import tempfile
import threading
import unittest
from datetime import UTC, datetime
from pathlib import Path

from t6.client import APIError, Unavailable
from t6.verbs import Ambiguous, Done, Refused

from jo_watchdog.adapters.protocols import (
    ContactMatcher,
    DeliverableSource,
    DirectoryReader,
    JobOrderApi,
    Matcher,
    ProjectMatcher,
)
from jo_watchdog.adapters.t6api import KeyAlreadyHeld, T6Api, T6ApiError
from jo_watchdog.app.caches import CacheManager, person_from_api
from jo_watchdog.ledger.db import Ledger
from tests.fakes import FakeMatcher, resolved
from tests.test_caches import config

PERSON = {
    "page_id": "c-ana",
    "name": "Ana Reyes",
    "telegram_user_id": 111,
    "telegram_handle": "@ana_r",
    "aliases": ["Ana", "Ate A"],
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

    def jo_raise(self, inputs):
        return self._answer("jo_raise", inputs)


class OutcomeMappingTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_seam_satisfies_both_protocols(self):
        api = T6Api(FakeClient())
        self.assertIsInstance(api, DirectoryReader)
        self.assertIsInstance(api, JobOrderApi)

    async def test_unavailable_refused_api_error_and_ambiguous_raise_one_error(self):
        for answer, text in (
            (Unavailable("connection refused"), "unavailable: connection refused"),
            (Refused(rule="permission", reason="no grant"), "refused (permission)"),
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
        refs = await T6Api(client).find_job_orders(-100, 7)
        self.assertEqual([r.id for r in refs], ["p-1", "p-2"])
        self.assertTrue(refs[0].url.endswith("p1"))
        asked = client.inputs["jo_find"]
        self.assertEqual((asked.chat_id, asked.message_id, asked.telegram_key), ("-100", "7", None))

    async def test_read_job_order_returns_the_job_order(self):
        order = {"page_id": "p-1", "name": "N", "telegram_key": "-100:7", "date": None}
        client = FakeClient(jo_get=Done(result={"job_order": order}))
        self.assertEqual(await T6Api(client).read_job_order("p-1"), order)
        self.assertEqual(client.inputs["jo_get"].ref, "p-1")


PLAN = {
    "name": "MFG - TOC - JO #74",
    "jo_number": "#74",
    "department": "TOC",
    "deadline": "2026-10-23T21:00:00+08:00",
    "deadline_source": "shift",
    "deadline_findings": [{"code": "shift_fallback", "params": {}}],
    "department_findings": [],
    "telegram_key": "-100:7",
    "duplicate_key": "proj|74",
    "jo_id": None,
}


def raise_inputs():
    from t6.verbs import JoRaiseInput, JoRaiseMessage

    message = JoRaiseMessage(
        header_text="MFG - TOC - JO #74",
        posted_at=datetime(2026, 10, 22, 9, 0, tzinfo=UTC),
        chat_id="-100",
        message_id="7",
    )
    return JoRaiseInput(message=message, fields={}, idempotency_key="watchdog-jo--100:7")


class RaiseTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_preview_returns_what_the_api_built_and_the_token(self):
        from t6.verbs import Preview

        client = FakeClient(jo_raise=Preview(plan=PLAN, token="tok"))
        preview = await T6Api(client).preview_job_order(raise_inputs())
        self.assertEqual(
            (preview.token, preview.name, preview.jo_number, preview.department),
            ("tok", "MFG - TOC - JO #74", "#74", "TOC"),
        )
        self.assertEqual(preview.deadline, "2026-10-23T21:00:00+08:00")
        self.assertEqual(preview.deadline_findings, [{"code": "shift_fallback", "params": {}}])
        self.assertEqual((preview.telegram_key, preview.duplicate_key), ("-100:7", "proj|74"))
        self.assertIsNone(client.inputs["jo_raise"].confirm)

    async def test_the_confirm_sends_the_token_with_the_same_key_and_returns_the_job_order_id(self):
        client = FakeClient(jo_raise=Done(result={**PLAN, "jo_id": "t6j_1", "page_id": None}))
        jo_id = await T6Api(client).confirm_job_order(raise_inputs(), "tok")
        self.assertEqual(jo_id, "t6j_1")
        sent = client.inputs["jo_raise"]
        self.assertEqual((sent.confirm, sent.idempotency_key), ("tok", "watchdog-jo--100:7"))

    async def test_a_preview_answer_where_a_done_is_wanted_is_an_error(self):
        from t6.verbs import Preview

        client = FakeClient(jo_raise=Preview(plan=PLAN, token="tok"))
        with self.assertRaises(T6ApiError):
            await T6Api(client).confirm_job_order(raise_inputs(), "tok")

    async def test_a_refusal_that_the_key_is_held_is_its_own_error(self):
        client = FakeClient(jo_raise=Refused(rule="telegram_key", reason="Telegram Key is held"))
        with self.assertRaises(KeyAlreadyHeld):
            await T6Api(client).preview_job_order(raise_inputs())

    async def test_the_page_is_none_until_it_exists(self):
        for page, expected in ((None, None), ("p-1", "p-1")):
            client = FakeClient(jo_get=Done(result={"job_order": {"page_id": page}}))
            self.assertEqual(await T6Api(client).job_order_page("t6j_1"), expected)
            self.assertEqual(client.inputs["jo_get"].ref, "t6j_1")

    async def test_an_outage_on_preview_or_confirm_is_one_error(self):
        for answer in (Unavailable("connection refused"), APIError("API returned HTTP 503", 503)):
            api = T6Api(FakeClient(jo_raise=answer))
            with self.assertRaises(T6ApiError):
                await api.preview_job_order(raise_inputs())
            with self.assertRaises(T6ApiError):
                await api.confirm_job_order(raise_inputs(), "tok")


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

    def test_a_contact_arrives_with_aliases_as_a_list(self):
        person = person_from_api(PERSON)
        self.assertEqual(person.aliases, ["Ana", "Ate A"])
        self.assertEqual(person.telegram_user_id, "111")

    def test_missing_ids_become_blank(self):
        person = person_from_api({"page_id": "c-1", "name": "Bo"})
        self.assertEqual(person.telegram_user_id, "")
        self.assertEqual(person.aliases, [])

    async def test_refresh_loads_the_two_lists_and_a_restart_keeps_them(self):
        await self.manager(self.client()).refresh()
        restarted = self.manager(self.client())
        restarted.load_persisted()
        self.assertTrue(restarted.ready)
        self.assertEqual(restarted.people[0].name, "Ana Reyes")
        self.assertEqual(restarted.projects[0].name, "Show")

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
