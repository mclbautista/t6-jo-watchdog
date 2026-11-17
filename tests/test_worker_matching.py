"""The worker runs the Deliverable match stage between planning and the one Notion write (#26)."""

import json
import unittest

from t6.verbs import Refused

from jo_watchdog.adapters.t6decide import MatcherError, T6DecideMatcher
from jo_watchdog.app.worker import Worker
from tests.fakes import FakeClassifier, FakeDecideClient, decide_answer, ok_result
from tests.test_match_stage import CANDS, LINKABLE, FakeDeliverables, FakeMatcher
from tests.test_worker_planning import KEY, Harness


class MatchHarness(Harness):
    def __init__(self, test, matcher, deliverables=None, *, link_enabled=False, **kwargs):
        super().__init__(test, **kwargs)
        fields = {k: v for k, v in kwargs.items() if k != "text"}
        self.config = self.config.model_copy(update={"deliverable_link_enabled": link_enabled})
        self.matcher = matcher
        self.deliverables = deliverables if deliverables is not None else FakeDeliverables(CANDS)
        self.worker = Worker(
            ledger=self.ledger,
            telegram=self.telegram,
            job_orders=self.notion,
            classifier=FakeClassifier(ok_result(**fields)),
            config=self.config,
            caches=self.caches,
            matcher=matcher,
            deliverables=self.deliverables,
        )

    def stage(self) -> dict:
        return self.ledger.stage_result(KEY, "match")


GOOD = decide_answer((0.95, 0.04, 0.01), (0.02, 0.08, 0.90))
ONE_OF_TWO = decide_answer((0.95, 0.04, 0.01))  # no answer for the second candidate


class WorkerInvalidAnswerTests(unittest.IsolatedAsyncioTestCase):
    """Real adapter over a fake t6 API into the real ledger (#55, #147)."""

    async def run_with(self, *answers, link_enabled=True):
        client = FakeDecideClient(*answers)
        h = MatchHarness(self, T6DecideMatcher(client), link_enabled=link_enabled)
        await h.run()
        return h, client.requests

    def llm_rows(self, h):
        return [
            dict(r)
            for r in h.ledger._conn.execute(
                "SELECT outcome, requested_model, returned_model, prompt_tag,"
                " input_tokens, output_tokens, detail_json FROM llm_calls"
                " WHERE provider = 'jev' ORDER BY id"
            )
        ]

    async def test_an_answer_that_misses_a_candidate_is_never_a_link(self):
        h, seen = await self.run_with(ONE_OF_TWO)
        self.assertEqual(len(seen), 2)  # one inline retry
        self.assertNotIn("Deliverable", h.notion.created[0]["properties"])
        self.assertEqual(h.stage()["outcome"], "skipped")
        self.assertEqual(h.codes().count("matcher_skipped"), 1)
        self.assertEqual(h.ledger.get_message(KEY)["state"], "done")
        rows = self.llm_rows(h)
        self.assertEqual([r["outcome"] for r in rows], ["invalid_body", "invalid_body"])
        for r in rows:
            self.assertEqual(r["prompt_tag"], "match-v1")
            detail = json.loads(r["detail_json"])
            self.assertEqual(
                [(c["page_id"], c["name"]) for c in detail["candidates"]],
                [(c.page_id, c.name) for c in CANDS],
            )

    async def test_probabilities_that_do_not_sum_to_one_are_never_a_link(self):
        h, _ = await self.run_with(decide_answer((0.9, 0.5, 0.5), (0.02, 0.08, 0.90)))
        self.assertEqual(h.stage()["outcome"], "skipped")
        self.assertNotIn("Deliverable", h.notion.created[0]["properties"])

    async def test_a_refusal_is_recorded_by_its_class_and_the_row_still_completes(self):
        refused = Refused(rule="provider", reason="LLM provider failed", failure="timeout")
        h, seen = await self.run_with(refused)
        self.assertEqual(len(seen), 2)
        self.assertEqual([r["outcome"] for r in self.llm_rows(h)], ["timeout", "timeout"])
        self.assertEqual(h.ledger.get_message(KEY)["state"], "done")

    async def test_a_valid_answer_still_links(self):
        h, _ = await self.run_with(GOOD)
        self.assertEqual(
            h.notion.created[0]["properties"]["Deliverable"], {"relation": [{"id": "d-ftr"}]}
        )
        rows = self.llm_rows(h)
        self.assertEqual([r["outcome"] for r in rows], ["ok"])
        self.assertEqual(rows[0]["returned_model"], "typesafe/jev-1.14-20261014")
        self.assertEqual((rows[0]["input_tokens"], rows[0]["output_tokens"]), (210, 31))


class WorkerMatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_shadow_mode_runs_jev_keeps_the_decision_and_writes_no_relation(self):
        h = MatchHarness(self, FakeMatcher(LINKABLE))
        await h.run()
        self.assertNotIn("Deliverable", h.notion.created[0]["properties"])
        self.assertEqual(len(h.matcher.calls), 1)
        self.assertEqual(h.matcher.calls[0][1], "Manufacturing")
        stage = h.stage()
        self.assertEqual(stage["outcome"], "hinted")
        self.assertFalse(stage["link_enabled"])
        self.assertEqual([c["page_id"] for c in stage["candidates"]], ["d-ftr", "d-promo"])
        self.assertIn("deliverable_hint", h.codes())
        self.assertEqual(h.plan()["match"]["outcome"], "hinted")
        calls = h.ledger._conn.execute(
            "SELECT provider, purpose, prompt_tag, returned_model FROM llm_calls"
            " WHERE provider = 'jev'"
        ).fetchall()
        self.assertEqual(
            [tuple(c) for c in calls],
            [("jev", "deliverable_match", "match-v1", "typesafe/jev-1.14-20261014")],
        )

    async def test_hint_report_shows_probability_and_tags_no_one(self):
        h = MatchHarness(self, FakeMatcher(LINKABLE))
        await h.run()
        text = " ".join(t for _, _, t in h.telegram.sent)
        self.assertIn("Possible Deliverable: Moonglow FTR SCR (match probability 0.95)", text)

    async def test_enabled_linking_writes_the_relation(self):
        h = MatchHarness(self, FakeMatcher(LINKABLE), link_enabled=True)
        await h.run()
        self.assertEqual(
            h.notion.created[0]["properties"]["Deliverable"], {"relation": [{"id": "d-ftr"}]}
        )
        self.assertEqual(h.stage()["outcome"], "linked")

    async def test_matcher_failure_still_writes_and_queues_one_line(self):
        h = MatchHarness(
            self, FakeMatcher(MatcherError("timeout", "a"), MatcherError("timeout", "b"))
        )
        await h.run()
        self.assertEqual(len(h.notion.created), 1)
        self.assertNotIn("Deliverable", h.notion.created[0]["properties"])
        self.assertEqual(h.stage()["outcome"], "skipped")
        self.assertEqual(h.codes().count("matcher_skipped"), 1)
        self.assertEqual(h.ledger.get_message(KEY)["state"], "done")

    async def test_unresolved_project_skips_the_call(self):
        h = MatchHarness(self, FakeMatcher(), text="ZZZ - TOC - JO #74\nTask: cut teaser")
        await h.run()
        self.assertEqual(h.matcher.calls, [])
        self.assertEqual(h.stage()["outcome"], "none")
        self.assertEqual(len(h.notion.created), 1)

    async def test_planned_plan_replays_without_a_second_call(self):
        h = MatchHarness(self, FakeMatcher(LINKABLE), link_enabled=True)
        h.config.dry_run = True
        await h.run()
        self.assertEqual(len(h.matcher.calls), 1)
        h.config.dry_run = False
        await h.worker.drain_once()
        self.assertEqual(len(h.matcher.calls), 1)
        self.assertEqual(
            json.loads(h.ledger.get_message(KEY)["plan_json"])["properties"]["Deliverable"],
            {"relation": [{"id": "d-ftr"}]},
        )
        self.assertEqual(len(h.notion.created), 1)

    async def test_no_matcher_wired_leaves_planning_unchanged(self):
        h = Harness(self)
        await h.run()
        self.assertIsNone(h.ledger.stage_result(KEY, "match"))
        self.assertEqual(len(h.notion.created), 1)
