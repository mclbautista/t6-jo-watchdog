"""The Deliverable matcher over the t6 API's ``llm decide``: request shape and outcome mapping."""

import unittest

from t6.client import APIError, Unavailable
from t6.verbs import Ambiguous, Refused

from jo_watchdog.adapters.protocols import DeliverableMatcher
from jo_watchdog.adapters.t6decide import (
    CRITERIA,
    MAX_RETRY_AFTER_SECONDS,
    PROMPT_TAG,
    QUESTION,
    MatcherError,
    T6DecideMatcher,
)
from jo_watchdog.core.matching import Candidate
from tests.fakes import FakeDecideClient, decide_answer

CANDS = [
    Candidate(page_id="d-ftr", name="Moonglow FTR SCR"),
    Candidate(page_id="d-promo", name="Moonglow Promo"),
]
GOOD = decide_answer((0.95, 0.04, 0.01), (0.02, 0.08, 0.90))


def matcher_for(*answers) -> tuple[T6DecideMatcher, FakeDecideClient]:
    client = FakeDecideClient(*answers)
    return T6DecideMatcher(client), client


class RequestShapeTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_state_the_candidate_names_the_question_and_the_criteria_go_as_written(self):
        matcher, client = matcher_for(GOOD)
        await matcher.match("Task: upload\nPW: hunter2", "Moonglow", CANDS)

        (request,) = client.requests
        self.assertEqual(
            request.state, {"project": "Moonglow", "job_order": "Task: upload\nPW: hunter2"}
        )
        self.assertEqual(request.candidates, ["Moonglow FTR SCR", "Moonglow Promo"])
        self.assertEqual(request.question, QUESTION)
        self.assertEqual(request.question.count("{candidate}"), 1)
        self.assertEqual(request.criteria, CRITERIA)
        self.assertIsNone(request.model_hint)
        self.assertFalse(request.require_zdr)  # the written exception to zero retention

    async def test_the_adapter_satisfies_the_matcher_protocol_and_tags_the_prompt(self):
        matcher, _ = matcher_for()
        self.assertIsInstance(matcher, DeliverableMatcher)
        self.assertEqual(matcher.prompt_tag, PROMPT_TAG)


class AnswerTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_answers_map_back_to_page_ids_with_the_trace(self):
        matcher, _ = matcher_for(GOOD)
        result = await matcher.match("x", "Moonglow", CANDS)

        self.assertEqual(
            [(d.page_id, d.p_match, d.p_maybe, d.p_no) for d in result.decisions],
            [("d-ftr", 0.95, 0.04, 0.01), ("d-promo", 0.02, 0.08, 0.90)],
        )
        self.assertEqual(result.returned_model, "typesafe/jev-1.14-20261014")
        self.assertEqual((result.input_tokens, result.output_tokens), (210, 31))

    async def test_answers_out_of_order_still_map_by_index(self):
        done = decide_answer((0.95, 0.04, 0.01), (0.02, 0.08, 0.90))
        done.result["answers"].reverse()
        matcher, _ = matcher_for(done)
        result = await matcher.match("x", "Moonglow", CANDS)
        self.assertEqual([d.page_id for d in result.decisions], ["d-ftr", "d-promo"])
        self.assertEqual(result.decisions[0].p_match, 0.95)

    async def test_an_answer_off_the_candidate_set_is_an_invalid_body(self):
        for name, answer in {
            "missing": decide_answer((0.95, 0.04, 0.01)),
            "extra": decide_answer((0.9, 0.05, 0.05), (0.9, 0.05, 0.05), (0.9, 0.05, 0.05)),
            "not a list": GOOD.model_copy(update={"result": {"answers": "x"}}),
        }.items():
            with self.subTest(name):
                matcher, _ = matcher_for(answer)
                with self.assertRaises(MatcherError) as ctx:
                    await matcher.match("x", "Moonglow", CANDS)
                self.assertEqual(ctx.exception.failure, "invalid_body")

    async def test_probabilities_that_do_not_sum_to_one_are_an_invalid_body(self):
        matcher, _ = matcher_for(decide_answer((0.9, 0.5, 0.5), (0.02, 0.08, 0.90)))
        with self.assertRaises(MatcherError) as ctx:
            await matcher.match("x", "Moonglow", CANDS)
        self.assertEqual(ctx.exception.failure, "invalid_body")


class FailureTests(unittest.IsolatedAsyncioTestCase):
    async def fails(self, answer) -> MatcherError:
        matcher, _ = matcher_for(answer)
        with self.assertRaises(MatcherError) as ctx:
            await matcher.match("x", "Moonglow", CANDS)
        return ctx.exception

    async def test_an_unreachable_api_is_a_transport_failure_and_a_timeout_a_timeout(self):
        down = await self.fails(Unavailable(reason="connection refused"))
        slow = await self.fails(Unavailable(reason="timed out"))
        self.assertEqual((down.failure, slow.failure), ("transport", "timeout"))

    async def test_an_api_error_is_an_api_error(self):
        self.assertEqual((await self.fails(APIError("HTTP 500"))).failure, "api_error")

    async def test_a_refusal_carries_the_providers_class(self):
        for failure in ("timeout", "api_error", "schema_violation", "policy"):
            with self.subTest(failure):
                refused = Refused(rule="provider", reason="LLM provider failed", failure=failure)
                self.assertEqual((await self.fails(refused)).failure, failure)

    async def test_a_refusal_without_a_class_names_its_rule(self):
        refused = Refused(rule="masking", reason="Unresolved registered names")
        self.assertEqual((await self.fails(refused)).failure, "masking")

    async def test_a_rate_limit_wait_is_kept_but_never_longer_than_thirty_seconds(self):
        for asked, kept in ((7.0, 7.0), (600.0, MAX_RETRY_AFTER_SECONDS), (-3.0, 0.0)):
            with self.subTest(asked=asked):
                refused = Refused(
                    rule="provider", reason="x", failure="api_error", retry_after=asked
                )
                self.assertEqual((await self.fails(refused)).retry_after, kept)

    async def test_the_error_text_holds_no_request_text(self):
        refused = Refused(rule="provider", reason="LLM provider failed", failure="timeout")
        error = await self.fails(refused)
        self.assertNotIn("hunter2", str(error))

    async def test_an_outcome_that_is_not_an_answer_is_an_api_error(self):
        error = await self.fails(Ambiguous(candidates=[]))
        self.assertEqual(error.failure, "api_error")


if __name__ == "__main__":
    unittest.main()


class ErrorTextTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_refusal_and_an_api_error_keep_no_provider_text(self):
        from t6.client import APIError
        from t6.verbs import Refused

        from jo_watchdog.adapters.t6decide import MatcherError, T6DecideMatcher
        from tests.fakes import FakeDecideClient

        for answer in (
            Refused(rule="provider", reason="echo: SECRET", failure="api_error"),
            APIError("HTTP 400: SECRET", 400),
        ):
            matcher = T6DecideMatcher(FakeDecideClient(answer))
            with self.assertRaises(MatcherError) as caught:
                await matcher.match("state", "proj", [])
            self.assertNotIn("SECRET", str(caught.exception))
