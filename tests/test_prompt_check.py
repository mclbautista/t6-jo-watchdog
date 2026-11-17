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


def refused_result(refusal: str) -> ClassifyResult:
    return ClassifyResult(
        outcome="api_error",
        error=f"refused ({refusal})",
        requested_model="deepseek-chat",
        prompt_tag=PROMPT_TAG,
        refusal=refusal,
    )


class PromptGuardTests(unittest.TestCase):
    def test_the_prompt_and_schema_hold_no_example_code(self):
        # A three-letter uppercase token (a code shape) may only be "JSON".
        for name, text in (
            ("template", PROMPT_TEMPLATE),
            ("schema", str(Extraction.model_json_schema())),
        ):
            tokens = set(re.findall(r"\b[A-Z]{3}\b", text)) - {"JSON"}
            self.assertEqual(tokens, set(), name)

    def test_the_header_example_line_uses_only_shapes(self):
        (line,) = [x for x in PROMPT_TEMPLATE.splitlines() if x.startswith("- project_raw")]
        quoted = re.findall(r'"([^"]*)"', line)
        self.assertTrue(quoted)
        for example in quoted:
            self.assertRegex(example, r"<[^<>]+>", example)  # every example holds a <...> shape
            self.assertIsNone(re.search(r"[A-Za-z0-9]", re.sub(r"<[^<>]+>|JO #", "", example)))


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
    async def test_a_refusal_by_rule_carries_rule_and_reason_and_is_not_retried_as_outage(self):
        for rule in ("masking", "template", "schema", "policy", "allow", "size"):
            with self.subTest(rule):
                classifier, client = classifier_for(Refused(rule=rule, reason="why"))
                result = await classifier.classify("x", None)
                self.assertEqual(result.refusal, f"{rule}: why")
                self.assertIsNone(result.extraction)
                self.assertEqual(len(client.requests), 1)

    async def test_an_output_leak_and_a_provider_failure_stay_retried_failures(self):
        leak = Refused(rule="masking", reason="Output contains names", failure="leak")
        provider = Refused(rule="provider", reason="LLM provider failed", failure="api_error")
        for refusal in (leak, provider):
            with self.subTest(refusal.rule):
                classifier, _ = classifier_for(refusal)
                self.assertIsNone((await classifier.classify("x", None)).refusal)

    async def test_a_dry_run_ignored_by_an_older_api_is_unsupported_not_ok(self):
        real = Done(result={"output": None, "output_valid": True, "tokens": {}})  # no dry_run
        classifier, _ = classifier_for(real)
        self.assertEqual((await classifier.check_prompt()).status, "unsupported")


def jo(update_id=1, message_id=42, text="Task: edit"):
    return update(update_id, message_id=message_id, text=text)


REFUSED = PromptCheck(status="refused", reason="masking: in the output schema")
OK = PromptCheck(status="ok")


class ExtractionHoldTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_real_refusal_and_a_refused_dry_run_hold_with_one_alert_and_no_attempt(self):
        classifier = FakeClassifier(refused_result("template: why"), checks=(REFUSED,))
        p = Pipeline(self, [jo(1, 42), jo(2, 43)], classifier=classifier)
        for _ in range(3):
            await p.worker.run_once()

        self.assertEqual(len(classifier.calls), 1)  # one refused call, then nothing is taken
        row = p.message()
        self.assertEqual(
            (row["state"], row["attempts"], row["exhausted_at"]), ("received", 0, None)
        )
        (alert,) = p.outbox("outage_alert")  # once, although two messages were held
        self.assertIn("masking: in the output schema", alert["text"])
        self.assertIn("Extraction is on hold", alert["text"])

    async def test_every_refusing_rule_holds_when_the_dry_run_is_refused(self):
        for rule in ("masking", "template", "schema", "policy", "allow", "size"):
            with self.subTest(rule):
                classifier = FakeClassifier(
                    refused_result(f"{rule}: why"),
                    checks=(PromptCheck(status="refused", reason=f"{rule}: why"),),
                )
                p = Pipeline(self, [jo()], classifier=classifier)
                await p.worker.run_once()
                self.assertEqual(p.worker.extraction_hold, f"{rule}: why")
                self.assertEqual(p.message()["attempts"], 0)

    async def test_the_start_check_holds_before_any_message_is_taken(self):
        classifier = FakeClassifier(checks=(REFUSED,))
        p = Pipeline(self, [jo()], classifier=classifier)
        await p.worker.check_prompt()
        await p.worker.run_once()

        self.assertEqual(classifier.calls, [])
        self.assertEqual(p.message()["state"], "received")
        (alert,) = p.outbox("outage_alert")
        self.assertIn("masking: in the output schema", alert["text"])

    async def test_the_check_re_runs_every_15_minutes_and_a_pass_releases_the_hold(self):
        classifier = FakeClassifier(ok_result(), checks=(REFUSED, REFUSED, OK))
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

    async def test_an_api_that_ignores_dry_run_does_not_hold(self):
        classifier = FakeClassifier(checks=(PromptCheck(status="unsupported"),))
        p = Pipeline(self, [jo()], classifier=classifier)
        await p.worker.check_prompt()
        await p.worker.run_once()

        self.assertIsNone(p.worker.extraction_hold)
        self.assertEqual(len(classifier.calls), 1)


class MessageRefusalTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_refusal_with_an_ok_dry_run_gives_the_message_up_without_a_hold(self):
        refusal = "masking: Unresolved registered names in the masked input"
        classifier = FakeClassifier(refused_result(refusal), checks=(OK,))
        p = Pipeline(self, [jo()], classifier=classifier)
        await p.worker.run_once()
        p.clock.advance(seconds=3600)
        await p.worker.run_once()

        row = p.message()
        self.assertEqual((row["state"], row["attempts"]), ("received", 1))
        self.assertIsNotNone(row["exhausted_at"])
        self.assertEqual(len(classifier.calls), 1)  # never retried
        self.assertEqual(classifier.check_calls, 1)  # no re-check: nothing is held
        self.assertIsNone(p.worker.extraction_hold)
        (alert,) = p.outbox("outage_alert")
        self.assertIn(f"Extraction refused by the t6 API ({refusal})", alert["text"])
        self.assertIn("message given up", alert["text"])
        self.assertNotIn("attempts", alert["text"])
        self.assertNotIn("Task: edit", alert["text"])

    async def test_a_refusal_when_the_dry_run_cannot_say_is_retried_as_an_api_error(self):
        for status in ("unavailable", "unsupported"):
            with self.subTest(status):
                classifier = FakeClassifier(
                    refused_result("masking: why"), checks=(PromptCheck(status=status),)
                )
                p = Pipeline(self, [jo()], classifier=classifier)
                await p.worker.run_once()

                row = p.message()
                self.assertIsNone(row["exhausted_at"])
                self.assertIsNotNone(row["next_attempt_at"])  # the normal schedule
                self.assertIsNone(p.worker.extraction_hold)
                self.assertEqual(p.outbox("outage_alert"), [])


if __name__ == "__main__":
    unittest.main()
