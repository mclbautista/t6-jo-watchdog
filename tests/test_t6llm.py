"""The extraction adapter over the t6 API's ``llm run``: request shape and outcome mapping."""

import asyncio
import json
import threading
import unittest

from jsonschema import Draft202012Validator
from t6.client import APIError, Unavailable
from t6.verbs import Ambiguous, Refused

from jo_watchdog.adapters.protocols import Classifier
from jo_watchdog.adapters.t6llm import T6LlmClassifier
from jo_watchdog.core.extraction import (
    PROMPT_TAG,
    PROMPT_TEMPLATE,
    Extraction,
    fewshot_examples,
)
from tests.fakes import LLM_RECORDED, FakeLlmClient, llm_answer

HINT = "deepseek-chat"


def classifier_for(*answers) -> tuple[T6LlmClassifier, FakeLlmClient]:
    client = FakeLlmClient(*answers)
    return T6LlmClassifier(client, model_hint=HINT), client


class RequestShapeTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_request_asks_for_names_masked_json_at_temperature_zero(self):
        classifier, client = classifier_for()
        await classifier.classify("MFG - TOC - JO #74\nTask: upload", None)

        (request,) = client.requests
        self.assertEqual(request.masking_preset, "names")
        self.assertEqual(request.model_hint, HINT)
        self.assertTrue(request.temperature_zero)
        self.assertFalse(request.require_zdr)  # the written exception to zero retention
        self.assertEqual(request.prompt_template, PROMPT_TEMPLATE)
        self.assertEqual(request.input_text, "MFG - TOC - JO #74\nTask: upload")
        self.assertEqual(request.output_schema, Extraction.model_json_schema())

    async def test_the_text_goes_as_posted_with_the_origin_block_after_it(self):
        classifier, client = classifier_for()
        text = "Task: x\nPW: probe-secret"
        await classifier.classify(text, ["Origin type: channel", "Forwarded by: Lei"])

        sent = client.requests[0].input_text
        self.assertTrue(sent.startswith(text + "\n\n--- forward origin"))
        self.assertIn("Forwarded by: Lei", sent)

    async def test_the_few_shot_turns_go_in_the_examples_list(self):
        classifier, client = classifier_for()
        await classifier.classify("x", None)

        examples = client.requests[0].examples
        self.assertEqual(len(examples), len(fewshot_examples()))
        schema = Draft202012Validator(Extraction.model_json_schema())
        for example, committed in zip(examples, fewshot_examples(), strict=True):
            self.assertEqual(example.input_text, committed["input_text"])
            self.assertEqual(example.output, committed["output"])
            schema.validate(example.output)  # the API refuses an example the schema rejects

    async def test_the_schema_is_a_valid_json_schema(self):
        Draft202012Validator.check_schema(Extraction.model_json_schema())

    async def test_the_adapter_satisfies_the_classifier_protocol(self):
        self.assertIsInstance(classifier_for()[0], Classifier)


class OutcomeTests(unittest.IsolatedAsyncioTestCase):
    async def test_verdicts_validate_and_carry_trace(self):
        for name in ("jo", "unsure", "not_jo"):
            with self.subTest(name):
                classifier, _ = classifier_for(llm_answer(name))
                result = await classifier.classify("x", None)
                self.assertEqual(result.outcome, "ok")
                self.assertEqual(result.extraction.verdict, name)
                self.assertEqual(result.requested_model, HINT)
                self.assertEqual(result.returned_model, "deepseek-chat")
                self.assertEqual(result.prompt_tag, PROMPT_TAG)
                self.assertEqual(result.input_tokens, 6180)
                self.assertEqual(result.output_tokens, 214)
                self.assertIsNotNone(result.latency_ms)

    async def test_jo_fields_stay_verbatim(self):
        classifier, _ = classifier_for(llm_answer("jo"))
        result = await classifier.classify("x", None)
        self.assertEqual(result.extraction.deadline_raw, "September 29 / Tues 5AM")
        self.assertEqual(result.extraction.jo_number_raw, "JO #74")

    async def test_failure_classes_map_to_the_ledger_outcomes(self):
        cases = {
            "empty_content": "empty_content",
            "malformed_json": "invalid_json",
            "schema_mismatch": "schema_violation",
            "unresolved_placeholder": "schema_violation",
            "wrong_shape": "schema_violation",  # valid for the schema, refused by Extraction
            "provider_failure": "api_error",
            "provider_timeout": "timeout",
            "leak": "schema_violation",
            "policy": "api_error",
        }
        for recording, outcome in cases.items():
            with self.subTest(recording):
                classifier, client = classifier_for(llm_answer(recording))
                result = await classifier.classify("x", None)
                self.assertEqual(result.outcome, outcome)
                self.assertIsNone(result.extraction)
                self.assertTrue(result.error)
                self.assertEqual(len(client.requests), 1)  # no repair call

    async def test_a_failed_output_keeps_the_trace_the_api_gave(self):
        classifier, _ = classifier_for(llm_answer("malformed_json"))
        result = await classifier.classify("x", None)
        self.assertEqual((result.returned_model, result.input_tokens), ("deepseek-chat", 6180))

    async def test_retry_after_of_a_refusal_is_carried(self):
        classifier, _ = classifier_for(llm_answer("rate_limited"))
        result = await classifier.classify("x", None)
        self.assertEqual((result.outcome, result.retry_after), ("api_error", 90.0))

    async def test_a_refusal_without_a_failure_class_is_an_api_error(self):
        refused = Refused(rule="size", reason="LLM request exceeds the configured size limit")
        classifier, _ = classifier_for(refused)
        result = await classifier.classify("x", None)
        self.assertEqual(result.outcome, "api_error")
        self.assertIn("size", result.error)

    async def test_a_down_api_is_an_api_error_and_a_slow_one_a_timeout(self):
        for answer, outcome in (
            (Unavailable("connection refused"), "api_error"),
            (Unavailable("<urlopen error timed out>"), "timeout"),
            (APIError("API returned HTTP 502: bad gateway", 502), "api_error"),
            (Ambiguous(candidates=[]), "api_error"),
        ):
            with self.subTest(answer=answer):
                classifier, _ = classifier_for(answer)
                result = await classifier.classify("x", None)
                self.assertEqual(result.outcome, outcome)
                self.assertIsNone(result.extraction)
                self.assertIsNone(result.retry_after)

    async def test_unsafe_trace_values_are_dropped(self):
        recorded = json.loads(json.dumps(LLM_RECORDED["jo"]))
        recorded["result"]["model"] = ["x"]
        recorded["result"]["tokens"] = {"prompt": "10", "completion": True}
        classifier, _ = classifier_for(type(llm_answer("jo")).model_validate(recorded))
        result = await classifier.classify("x", None)
        self.assertEqual(result.outcome, "ok")
        self.assertEqual(
            (result.returned_model, result.input_tokens, result.output_tokens), (None, None, None)
        )


class ThreadingTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_hung_call_times_out_without_stalling_the_loop(self):
        release = threading.Event()
        self.addCleanup(release.set)

        class Hung:
            def llm_run(self, inputs):
                release.wait(5)

        classifier = T6LlmClassifier(Hung(), model_hint=HINT)
        classifier._wait = 0.2
        ticks = 0

        async def heartbeat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        try:
            result = await classifier.classify("x", None)
        finally:
            beat.cancel()
        self.assertEqual(result.outcome, "timeout")
        self.assertGreater(ticks, 5)


if __name__ == "__main__":
    unittest.main()


class ErrorTextTests(unittest.IsolatedAsyncioTestCase):
    """What is stored or alerted holds no message text and no model output."""

    async def test_a_schema_violation_keeps_the_error_count_and_field_paths_only(self):
        from t6.verbs import Done

        answer = Done(
            outcome="done",
            result={
                "output": {"verdict": "SECRET message text", "confidence": "high"},
                "output_valid": True,
                "model": "m",
                "tokens": {},
            },
        )
        classifier, _ = classifier_for(answer)
        result = await classifier.classify("x", None)
        self.assertEqual(result.outcome, "schema_violation")
        self.assertNotIn("SECRET", result.error)
        self.assertIn("ValidationError", result.error)
        self.assertIn("verdict", result.error)

    async def test_a_refusal_and_an_api_error_keep_no_provider_text(self):
        for answer in (
            Refused(rule="provider", reason="echo: SECRET text", failure="api_error"),
            APIError("HTTP 400: SECRET text", 400),
        ):
            classifier, _ = classifier_for(answer)
            result = await classifier.classify("x", None)
            self.assertNotIn("SECRET", result.error)
