"""Offline tests for the typed-model smoke checks (#61). No network, no live model calls."""

import copy
import json
import unittest
from pathlib import Path

from t6.verbs import Refused

from jo_watchdog.app.model_smoke import (
    MASK_NEGATIVE,
    MASK_POSITIVE,
    REGRESSION_PAIRS,
    SECRET,
    PairError,
    check_decide,
    check_extraction,
    load_pairs,
)
from jo_watchdog.app.smoke_gate import FAIL, MANUAL, PASS
from tests.fakes import LLM_RECORDED, FakeDecideClient, FakeLlmClient, decide_answer, llm_answer

FIXTURES = Path(__file__).parent / "fixtures"
PAIRS_PATH = FIXTURES / "jev" / "regression_pairs.json"
PAIRS = json.loads(PAIRS_PATH.read_text(encoding="utf-8"))

# The fixtures' expected verdicts: hint, link, link.
GOOD = [(0.41, 0.52, 0.07), (0.94, 0.05, 0.01), (0.92, 0.07, 0.01)]
TIMEOUT = 5


def answers(probabilities, **kwargs):
    """One ``llm decide`` answer for each pair, in request order (one candidate each)."""
    return [decide_answer(p, **kwargs) for p in probabilities]


async def decide_check(*replies, path=PAIRS_PATH):
    client = FakeDecideClient(*replies)
    check = await check_decide(client, path, timeout=TIMEOUT, t_link=0.9, t_hint=0.5)
    return check, client


class DecideRequestTests(unittest.IsolatedAsyncioTestCase):
    def test_all_three_fixed_pairs_are_loaded_in_order(self):
        pairs = load_pairs(PAIRS_PATH)
        self.assertEqual([p["name"] for p in pairs], list(REGRESSION_PAIRS))

    async def test_one_request_per_pair_with_the_job_order_as_posted(self):
        check, client = await decide_check(*answers(GOOD))
        self.assertEqual(check.result.status, PASS, check.result.detail)
        self.assertEqual(len(client.requests), 3)
        self.assertEqual(check.requests_sent, 3)
        for request, pair in zip(client.requests, PAIRS, strict=True):
            self.assertEqual(request.state["project"], pair["project"]["name"])
            self.assertEqual(request.state["job_order"], pair["source_text"])
            self.assertEqual(request.candidates, [pair["candidates"][0]["name"]])

    async def test_it_reports_the_snapshot_tokens_and_every_pair(self):
        check, _ = await decide_check(*answers(GOOD))
        text = "\n".join(check.lines)
        self.assertIn("typesafe/jev-1.14-20261014", text)
        for pair in PAIRS:
            self.assertIn(f"{pair['name']} vs {pair['candidates'][0]['name']}", text)
        self.assertIn("match=0.41 maybe=0.52 no=0.07", text)
        self.assertIn("tokens 210/31", text)

    async def test_a_refusal_reports_its_class_and_retry_after_and_fails(self):
        refused = Refused(
            rule="provider", reason="LLM provider failed", failure="api_error", retry_after=7.0
        )
        check, client = await decide_check(refused)
        self.assertEqual(check.result.status, FAIL)
        self.assertIn("api_error", check.result.detail)
        self.assertIn("retry after 7s", check.result.detail)
        self.assertEqual(len(client.requests), 1)

    async def test_an_answer_that_misses_the_candidate_fails_through_production_validation(self):
        check, _ = await decide_check(*answers(GOOD[:2]), decide_answer())
        self.assertEqual(check.result.status, FAIL)
        self.assertIn("invalid_body", check.result.detail)

    async def test_probabilities_that_do_not_sum_to_one_fail(self):
        check, _ = await decide_check(*answers([(0.9, 0.5, 0.5)] + GOOD[1:]))
        self.assertEqual(check.result.status, FAIL)

    async def test_missing_snapshot_fails(self):
        check, _ = await decide_check(*answers(GOOD, model=""))
        self.assertEqual(check.result.status, FAIL)

    async def test_unexpected_verdict_needs_owner_review_not_a_pass(self):
        check, _ = await decide_check(*answers([(0.1, 0.1, 0.8)] + GOOD[1:]))
        self.assertEqual(check.result.status, MANUAL)
        self.assertIn("moonglow-jo-32 is none, expected hint", check.result.detail)

    async def test_linking_is_never_enabled_by_the_check(self):
        check, _ = await decide_check(*answers(GOOD))
        self.assertNotIn("linked", "\n".join(check.lines).lower())


class DecideFixtureTests(unittest.IsolatedAsyncioTestCase):
    def write(self, tmp, pairs) -> Path:
        path = Path(tmp) / "pairs.json"
        path.write_text(json.dumps(pairs), encoding="utf-8")
        return path

    async def test_a_missing_fixture_fails_and_sends_nothing(self):
        import tempfile

        for dropped in range(3):
            with self.subTest(dropped=PAIRS[dropped]["name"]), tempfile.TemporaryDirectory() as tmp:
                pairs = [p for i, p in enumerate(PAIRS) if i != dropped]
                check, client = await decide_check(*answers(GOOD), path=self.write(tmp, pairs))
                self.assertEqual(check.result.status, FAIL)
                self.assertIn(PAIRS[dropped]["name"], check.result.detail)
                self.assertEqual(client.requests, [])
                self.assertEqual(check.requests_sent, 0)

    async def test_a_pair_with_two_deliverables_is_rejected(self):
        import tempfile

        pairs = copy.deepcopy(PAIRS)
        pairs[0]["candidates"].append(pairs[1]["candidates"][0])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PairError):
                load_pairs(self.write(tmp, pairs))

    async def test_an_unreadable_fixture_fails(self):
        check, _ = await decide_check(*answers(GOOD), path=FIXTURES / "jev" / "absent.json")
        self.assertEqual(check.result.status, FAIL)
        self.assertEqual(check.requests_sent, 0)


async def extraction(*answers):
    client = FakeLlmClient(*answers)
    return await check_extraction(client, model_hint="deepseek-chat", timeout=5), client


class ExtractionSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_credential_and_no_credential_cases_both_run_as_posted(self):
        check, client = await extraction(llm_answer("jo"))
        self.assertEqual(check.result.status, PASS, check.result.detail)
        self.assertEqual(check.requests_sent, 2)
        positive, negative = (request.input_text for request in client.requests)
        self.assertEqual(positive, MASK_POSITIVE)  # the t6 API withholds it, not the watchdog
        self.assertEqual(negative, MASK_NEGATIVE)
        self.assertIn("pw:", MASK_POSITIVE)

    async def test_a_reply_that_carries_the_credential_fails(self):
        recorded = json.loads(json.dumps(LLM_RECORDED["jo"]))
        recorded["result"]["output"]["notes"] = f"pw: {SECRET}"
        leaky = type(llm_answer("jo")).model_validate(recorded)
        check, _ = await extraction(leaky)
        self.assertEqual(check.result.status, FAIL)
        self.assertIn("credential", check.result.detail)

    async def test_reports_snapshot_and_token_metadata(self):
        check, _ = await extraction(llm_answer("jo"))
        recorded = LLM_RECORDED["jo"]["result"]
        text = "\n".join(check.lines)
        self.assertIn(recorded["model"], text)
        self.assertIn(f"{recorded['tokens']['prompt']}/{recorded['tokens']['completion']}", text)

    async def test_invalid_structured_output_fails(self):
        for case in ("malformed_json", "schema_mismatch", "empty_content", "provider_failure"):
            with self.subTest(case=case):
                check, _ = await extraction(llm_answer(case))
                self.assertEqual(check.result.status, FAIL)

    async def test_non_jo_verdict_needs_review_not_a_pass(self):
        check, _ = await extraction(llm_answer("not_jo"))
        self.assertEqual(check.result.status, MANUAL)
