"""The prompt start check, the extraction hold and non-retried refusals (#159)."""

import re
import unittest

from t6.client import Unavailable
from t6.verbs import Done, Refused

from jo_watchdog.adapters.protocols import ClassifyResult, PromptCheck
from jo_watchdog.adapters.t6llm import CHECK_TEXT, T6LlmClassifier
from jo_watchdog.app.worker import PROMPT_RECHECK
from jo_watchdog.core.extraction import (
    PROMPT_TAG,
    PROMPT_TEMPLATE,
    Extraction,
    fewshot_examples,
)
from tests.clockwork import Pipeline
from tests.fakes import FakeClassifier, FakeLlmClient, ok_result, update

TEMPLATE_REFUSAL = Refused(
    rule="masking", reason="Unresolved registered names in the prompt template"
)
SCHEMA_REFUSAL = Refused(rule="masking", reason="Unresolved registered names in the output schema")
INPUT_REFUSAL = Refused(rule="masking", reason="Unresolved registered names in the masked input")
DRY_DONE = Done(
    result={
        "output": None,
        "model": "deepseek-chat",
        "provider": "deepseek",
        "tokens": {"prompt": 0, "completion": 0, "total": 0},
        "output_valid": True,
        "dry_run": True,
    }
)


def classifier_for(*answers) -> tuple[T6LlmClassifier, FakeLlmClient]:
    client = FakeLlmClient(*answers)
    return T6LlmClassifier(client, model_hint="deepseek-chat"), client


def refused_result(refusal: str, *, hold: bool) -> ClassifyResult:
    return ClassifyResult(
        outcome="api_error",
        error=f"refused ({refusal})",
        requested_model="deepseek-chat",
        prompt_tag=PROMPT_TAG,
        refusal=refusal,
        hold=hold,
    )


class PromptGuardTests(unittest.TestCase):
    def test_the_prompt_and_schema_hold_no_example_code(self):
        # A three-letter uppercase token (a code shape) may only appear as a JSON-style
        # field word such as "JSON"; no example Project or department code is allowed.
        allowed = {"JSON"}
        for name, text in (
            ("template", PROMPT_TEMPLATE),
            ("schema", str(Extraction.model_json_schema())),
        ):
            tokens = set(re.findall(r"\b[A-Z]{3}\b", text)) - allowed
            self.assertEqual(tokens, set(), name)

    def test_the_project_line_uses_shapes_not_examples(self):
        (line,) = [x for x in PROMPT_TEMPLATE.splitlines() if x.startswith("- project_raw")]
        self.assertIn("<CODE>", line)
        self.assertNotIn("e.g.", line)


class CheckPromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_check_is_a_dry_run_of_the_real_request_with_a_fixed_text(self):
        classifier, client = classifier_for(DRY_DONE)
        check = await classifier.check_prompt()

        self.assertEqual(check.status, "ok")
        (request,) = client.requests
        self.assertTrue(request.dry_run)
        self.assertEqual(request.input_text, CHECK_TEXT)
        self.assertEqual(request.prompt_template, PROMPT_TEMPLATE)
        self.assertEqual(request.output_schema, Extraction.model_json_schema())
        self.assertEqual(len(request.examples), len(fewshot_examples()))

    async def test_a_real_run_is_not_a_dry_run(self):
        classifier, client = classifier_for()
        await classifier.classify("x", None)
        self.assertFalse(client.requests[0].dry_run)

    async def test_a_refusal_by_rule_is_refused_with_rule_and_reason(self):
        for refusal in (TEMPLATE_REFUSAL, SCHEMA_REFUSAL, INPUT_REFUSAL):
            with self.subTest(refusal.reason):
                classifier, _ = classifier_for(refusal)
                check = await classifier.check_prompt()
                self.assertEqual(check.status, "refused")
                self.assertEqual(check.reason, f"masking: {refusal.reason}")

    async def test_template_schema_and_policy_rules_are_refused_too(self):
        for rule in ("template", "schema", "policy"):
            with self.subTest(rule):
                classifier, _ = classifier_for(Refused(rule=rule, reason="why"))
                check = await classifier.check_prompt()
                self.assertEqual((check.status, check.reason), ("refused", f"{rule}: why"))

    async def test_an_api_that_does_not_answer_is_unavailable_not_refused(self):
        provider = Refused(rule="provider", reason="LLM provider failed", failure="api_error")
        for answer in (Unavailable("connection refused"), Unavailable("timed out"), provider):
            with self.subTest(answer):
                classifier, _ = classifier_for(answer)
                self.assertEqual((await classifier.check_prompt()).status, "unavailable")


class RefusedExtractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_refusal_by_rule_is_not_an_outage_outcome(self):
        cases = [
            (TEMPLATE_REFUSAL, True),
            (SCHEMA_REFUSAL, True),
            (Refused(rule="policy", reason="Provider does not meet retention policy"), True),
            (INPUT_REFUSAL, False),  # this message alone
            (Refused(rule="size", reason="LLM request exceeds the size limit"), False),
            (Refused(rule="allow", reason="not allowed"), False),
        ]
        for refusal, hold in cases:
            with self.subTest(refusal.rule + refusal.reason):
                classifier, client = classifier_for(refusal)
                result = await classifier.classify("x", None)
                self.assertEqual(result.refusal, f"{refusal.rule}: {refusal.reason}")
                self.assertEqual(result.hold, hold)
                self.assertIsNone(result.extraction)
                self.assertEqual(len(client.requests), 1)

    async def test_an_output_leak_and_a_provider_failure_stay_retried_failures(self):
        leak = Refused(rule="masking", reason="Output contains names", failure="leak")
        provider = Refused(rule="provider", reason="LLM provider failed", failure="api_error")
        for refusal in (leak, provider):
            with self.subTest(refusal.rule):
                classifier, _ = classifier_for(refusal)
                self.assertIsNone((await classifier.classify("x", None)).refusal)


def jo(update_id=1, message_id=42, text="Task: edit"):
    return update(update_id, message_id=message_id, text=text)


class ExtractionHoldTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_refused_prompt_holds_extraction_with_one_alert_and_no_attempt(self):
        classifier = FakeClassifier(refused_result("masking: in the prompt template", hold=True))
        p = Pipeline(self, [jo(1, 42), jo(2, 43)], classifier=classifier)
        for _ in range(3):
            await p.worker.run_once()

        self.assertEqual(len(classifier.calls), 1)  # one refused call, then nothing is taken
        row = p.message()
        self.assertEqual(
            (row["state"], row["attempts"], row["exhausted_at"]), ("received", 0, None)
        )
        (alert,) = p.outbox("outage_alert")
        self.assertIn("masking: in the prompt template", alert["text"])
        self.assertIn("Extraction is on hold", alert["text"])

    async def test_the_start_check_holds_before_any_message_is_taken(self):
        classifier = FakeClassifier(
            checks=(PromptCheck(status="refused", reason="masking: in the output schema"),)
        )
        p = Pipeline(self, [jo()], classifier=classifier)
        await p.worker.check_prompt()
        await p.worker.run_once()

        self.assertEqual(classifier.calls, [])
        self.assertEqual(p.message()["state"], "received")
        (alert,) = p.outbox("outage_alert")
        self.assertIn("masking: in the output schema", alert["text"])

    async def test_the_check_re_runs_every_15_minutes_and_a_pass_releases_the_hold(self):
        refused = PromptCheck(status="refused", reason="schema: bad")
        checks = (refused, refused, PromptCheck(status="ok"))
        classifier = FakeClassifier(ok_result(), checks=checks)
        p = Pipeline(self, [jo()], classifier=classifier)
        await p.worker.check_prompt()
        self.assertEqual(classifier.check_calls, 1)

        p.clock.advance(seconds=PROMPT_RECHECK.total_seconds() - 1)
        await p.worker.run_once()
        self.assertEqual(classifier.check_calls, 1)  # not yet due

        p.clock.advance(seconds=1)
        await p.worker.run_once()
        self.assertEqual((classifier.check_calls, classifier.calls), (2, []))  # still refused

        p.clock.advance(seconds=PROMPT_RECHECK.total_seconds())
        await p.worker.run_once()
        self.assertEqual(classifier.check_calls, 3)
        self.assertEqual(len(classifier.calls), 1)  # released: the message is taken
        self.assertEqual(len(p.outbox("outage_alert")), 1)  # one alert for the whole hold

    async def test_an_unavailable_api_at_start_does_not_hold(self):
        classifier = FakeClassifier(checks=(PromptCheck(status="unavailable"),))
        p = Pipeline(self, [jo()], classifier=classifier)
        await p.worker.check_prompt()
        await p.worker.run_once()

        self.assertIsNone(p.worker.extraction_hold)
        self.assertEqual(len(classifier.calls), 1)
        self.assertEqual(p.outbox("outage_alert"), [])


class MessageRefusalTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_refusal_of_one_message_is_given_up_at_once_with_one_alert(self):
        refusal = "masking: Unresolved registered names in the masked input"
        classifier = FakeClassifier(refused_result(refusal, hold=False))
        p = Pipeline(self, [jo()], classifier=classifier)
        await p.worker.run_once()
        p.clock.advance(seconds=3600)
        await p.worker.run_once()

        row = p.message()
        self.assertEqual((row["state"], row["attempts"]), ("received", 1))
        self.assertIsNotNone(row["exhausted_at"])
        self.assertEqual(len(classifier.calls), 1)  # never retried
        self.assertIsNone(p.worker.extraction_hold)
        (alert,) = p.outbox("outage_alert")
        self.assertIn(refusal, alert["text"])
        self.assertNotIn("Task: edit", alert["text"])


if __name__ == "__main__":
    unittest.main()
