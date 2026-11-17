"""Classification stage in the whole pipeline: real SQLite, fake adapters."""

import json
import unittest

from t6.client import Unavailable

from jo_watchdog.adapters.t6llm import T6LlmClassifier
from jo_watchdog.ledger.states import State
from tests.fakes import FakeClassifier, FakeLlmClient, failed_result, llm_answer, ok_result, update
from tests.test_worker import make_worker

KEY = "-100123:42"


def rows(ledger, sql, *args):
    return [dict(r) for r in ledger._conn.execute(sql, args).fetchall()]


class VerdictTests(unittest.IsolatedAsyncioTestCase):
    async def run_one(self, *results, text="MFG - TOC - JO #74\nTask: upload", extra=None):
        classifier = FakeClassifier(*results)
        worker, ledger, telegram, notion, cleanup = make_worker(
            [update(7, message_id=42, text=text, extra=extra)], classifier=classifier
        )
        self.addCleanup(cleanup)
        await worker.run_once()
        return ledger, telegram, notion, classifier

    async def test_jo_is_written_with_trace_and_stage_result(self):
        ledger, telegram, notion, _ = await self.run_one(ok_result("jo", task="upload"))
        msg = ledger.get_message(KEY)
        self.assertEqual(msg["state"], "done")
        self.assertEqual(msg["verdict"], "jo")
        self.assertEqual(msg["posted_at"], "2026-09-28T11:00:00+08:00")
        self.assertEqual(len(notion.created), 1)

        result = ledger.stage_result(KEY, "extract")
        self.assertEqual(result["extraction"]["task"], "upload")
        self.assertEqual(result["findings"], [])
        self.assertEqual(result["trace"]["returned_model"], "deepseek-chat-v9")
        self.assertEqual(result["trace"]["requested_model"], "deepseek-chat")
        self.assertEqual(result["trace"]["prompt_tag"], "extract-v3")
        self.assertEqual(result["trace"]["input_tokens"], 1200)

        (call,) = rows(ledger, "SELECT * FROM llm_calls")
        self.assertEqual(call["provider"], "deepseek")
        self.assertEqual(call["purpose"], "classify_extract")
        self.assertEqual(call["outcome"], "ok")
        self.assertEqual(call["masker_version"], "t6-api")
        self.assertEqual(call["latency_ms"], 850)

    async def test_unsure_is_written_and_records_a_finding(self):
        ledger, telegram, notion, _ = await self.run_one(ok_result("unsure"))
        self.assertEqual(ledger.get_message(KEY)["state"], "done")
        self.assertEqual(len(notion.created), 1)
        findings = ledger.stage_result(KEY, "extract")["findings"]
        self.assertEqual(findings, [{"code": "unsure", "reasons": "Looks like unsure."}])

    async def test_not_jo_is_ignored_with_its_first_line_as_written(self):
        ledger, telegram, notion, _ = await self.run_one(
            ok_result("not_jo"), text="PW: hunter2 salamat po\nmore"
        )
        msg = ledger.get_message(KEY)
        self.assertEqual(msg["state"], "ignored")
        self.assertEqual(msg["ignore_reason"], "not_jo")
        self.assertEqual(notion.created, [])
        self.assertEqual(telegram.reactions, [])
        (logged,) = rows(ledger, "SELECT * FROM ignore_log")
        self.assertEqual(logged["reason"], "not_jo")
        self.assertEqual(logged["first_line"], "PW: hunter2 salamat po")
        self.assertEqual(ledger.stage_result(KEY, "extract")["extraction"]["verdict"], "not_jo")

    async def test_no_text_bypasses_the_model(self):
        ledger, telegram, notion, classifier = await self.run_one(
            text="", extra={"forward_origin": {"type": "channel", "chat": {"title": "T6"}}}
        )
        self.assertEqual(classifier.calls, [])
        msg = ledger.get_message(KEY)
        self.assertEqual(msg["state"], "ignored")
        self.assertEqual(msg["ignore_reason"], "no_text")
        self.assertEqual(rows(ledger, "SELECT reason FROM ignore_log"), [{"reason": "no_text"}])
        self.assertEqual(rows(ledger, "SELECT * FROM llm_calls"), [])
        self.assertEqual(notion.created, [])

    async def test_multiple_jo_writes_one_row_and_flags_a_split(self):
        text = "JO #1\nTask: a\n\nJO #2\nTask: b"
        ledger, telegram, notion, _ = await self.run_one(
            ok_result("jo", multiple_jo_suspected=True, task="a"), text=text
        )
        self.assertEqual(len(notion.created), 1)
        plan = json.loads(ledger.get_message(KEY)["plan_json"])
        self.assertIn("Task: b", "\n".join(plan["body_blocks"]))  # full source kept
        result = ledger.stage_result(KEY, "extract")
        self.assertEqual(result["extraction"]["task"], "a")
        self.assertEqual([f["code"] for f in result["findings"]], ["multiple_jo"])


class ModelInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_classifier_gets_the_text_as_posted_and_forward_origin_only(self):
        classifier = FakeClassifier()
        extra = {
            "from": {"first_name": "Lei"},
            "forward_origin": {"type": "hidden_user", "sender_user_name": "Ben", "date": 1},
        }
        worker, ledger, *_ , cleanup = make_worker(
            [update(7, message_id=42, text="Task: x\nPW: hunter2", extra=extra)],
            classifier=classifier,
        )
        self.addCleanup(cleanup)
        await worker.run_once()
        ((text, origin),) = classifier.calls
        self.assertEqual(text, "Task: x\nPW: hunter2")  # the t6 API masks it
        self.assertEqual(
            origin,
            ["Origin type: hidden_user", "Forwarded by: Lei", "Originally from: Ben"],
        )

    async def test_non_forward_sends_no_origin(self):
        classifier = FakeClassifier()
        worker, ledger, *_, cleanup = make_worker([update(7, message_id=42)], classifier=classifier)
        self.addCleanup(cleanup)
        await worker.run_once()
        self.assertIsNone(classifier.calls[0][1])

    async def test_worker_to_llm_adapter_sends_one_request_with_the_text_as_posted(self):
        client = FakeLlmClient(llm_answer("jo"))
        extra = {"from": {"first_name": "Lei"}, "forward_origin": {"type": "channel"}}
        worker, ledger, *_, cleanup = make_worker(
            [update(7, message_id=42, text="MFG - TOC - JO #74\nPW: hunter2", extra=extra)],
            classifier=T6LlmClassifier(client, model_hint="deepseek-chat"),
        )
        self.addCleanup(cleanup)
        await worker.run_once()
        (request,) = client.requests
        self.assertTrue(request.input_text.startswith("MFG - TOC - JO #74\nPW: hunter2\n\n---"))
        self.assertEqual(request.masking_preset, "names")  # names and credentials: the API's job
        (call,) = rows(ledger, "SELECT * FROM llm_calls")
        self.assertEqual((call["outcome"], call["prompt_tag"]), ("ok", "extract-v3"))
        self.assertEqual((call["input_tokens"], call["output_tokens"]), (6180, 214))


class FailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_failure_class_stays_queued_without_a_row(self):
        outcomes = ("invalid_json", "schema_violation", "empty_content", "timeout", "api_error")
        for outcome in outcomes:
            with self.subTest(outcome):
                worker, ledger, telegram, notion, cleanup = make_worker(
                    [update(7, message_id=42)], classifier=FakeClassifier(failed_result(outcome))
                )
                await worker.run_once()
                msg = ledger.get_message(KEY)
                self.assertEqual(msg["state"], "received")
                self.assertEqual(msg["attempts"], 1)
                self.assertIsNotNone(msg["next_attempt_at"])
                self.assertIn(outcome, msg["last_error"])
                self.assertEqual(notion.created, [])
                self.assertEqual(ledger.stage_result(KEY, "extract"), None)
                (call,) = rows(ledger, "SELECT outcome FROM llm_calls")
                self.assertEqual(call["outcome"], outcome)
                cleanup()

    async def test_an_unusable_llm_answer_is_one_retryable_failed_attempt(self):
        cases = {
            "empty_content": "empty_content",
            "malformed_json": "invalid_json",
            "schema_mismatch": "schema_violation",
            "wrong_shape": "schema_violation",
            "provider_failure": "api_error",
            "provider_timeout": "timeout",
        }
        for recording, outcome in cases.items():
            with self.subTest(recording):
                client = FakeLlmClient(llm_answer(recording))
                worker, ledger, telegram, notion, cleanup = make_worker(
                    [update(7, message_id=42)],
                    classifier=T6LlmClassifier(client, model_hint="deepseek-chat"),
                )
                await worker.run_once()  # no uncaught error
                msg = ledger.get_message(KEY)
                self.assertEqual(msg["state"], "received")
                self.assertEqual(msg["attempts"], 1)
                self.assertIsNotNone(msg["next_attempt_at"])
                self.assertEqual(len(client.requests), 1)  # no repair call
                self.assertEqual(notion.created, [])
                (call,) = rows(ledger, "SELECT outcome FROM llm_calls")
                self.assertEqual(call["outcome"], outcome)
                cleanup()

    async def test_a_down_api_holds_the_message_and_retries_then_extraction_succeeds(self):
        client = FakeLlmClient(Unavailable("connection refused"), llm_answer("jo"))
        worker, ledger, telegram, notion, cleanup = make_worker(
            [update(7, message_id=42)],
            classifier=T6LlmClassifier(client, model_hint="deepseek-chat"),
        )
        self.addCleanup(cleanup)
        await worker.run_once()
        msg = ledger.get_message(KEY)
        self.assertEqual((msg["state"], msg["attempts"]), ("received", 1))
        self.assertEqual(notion.created, [])  # nothing is raised until extraction succeeds

        ledger._conn.execute("UPDATE messages SET next_attempt_at = '2000-01-01T00:00:00Z'")
        await worker.drain_once()
        msg = ledger.get_message(KEY)
        self.assertEqual((msg["state"], msg["attempts"]), ("done", 0))
        self.assertEqual(len(notion.created), 1)

    async def test_retry_waits_then_succeeds_and_resets_attempts(self):
        classifier = FakeClassifier(failed_result("timeout"), ok_result("jo"))
        worker, ledger, telegram, notion, cleanup = make_worker(
            [update(7, message_id=42)], classifier=classifier
        )
        self.addCleanup(cleanup)
        await worker.run_once()
        await worker.drain_once()  # not due yet
        self.assertEqual(len(classifier.calls), 1)

        ledger._conn.execute("UPDATE messages SET next_attempt_at = '2000-01-01T00:00:00Z'")
        await worker.drain_once()
        msg = ledger.get_message(KEY)
        self.assertEqual(msg["state"], State.DONE.value)
        self.assertEqual(msg["attempts"], 0)
        self.assertIsNone(msg["last_error"])
        self.assertEqual(len(notion.created), 1)
        self.assertEqual(
            [r["outcome"] for r in rows(ledger, "SELECT outcome FROM llm_calls ORDER BY id")],
            ["timeout", "ok"],
        )

    async def test_extracted_message_resumes_without_reclassifying(self):
        classifier = FakeClassifier()
        worker, ledger, telegram, notion, cleanup = make_worker(
            [update(7, message_id=42)], classifier=classifier, fail_create=True
        )
        self.addCleanup(cleanup)
        await worker.poll_once()
        await worker.classify(ledger.get_message(KEY))
        self.assertEqual(ledger.get_message(KEY)["state"], "extracted")
        notion.fail_create = False
        await worker.drain_once()
        self.assertEqual(ledger.get_message(KEY)["state"], "done")
        self.assertEqual(len(classifier.calls), 1)


if __name__ == "__main__":
    unittest.main()
