"""Filter before extraction (#153): the config, the ``llm filter`` call, count first and fail open."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from pydantic import ValidationError
from t6.client import APIError, Unavailable
from t6.verbs import Done, Refused

from jo_watchdog.adapters.t6filter import FilterResult, T6Filter
from jo_watchdog.app import __main__ as entry
from jo_watchdog.app.config import Config
from jo_watchdog.ledger.db import Ledger
from tests.fakes import FakeClassifier, ok_result, update
from tests.test_config import MINIMAL
from tests.test_worker import make_worker

KEY = "-100123:42"


def filter_done(verdict="pass"):
    return Done(
        outcome="done",
        result={
            "verdict": verdict,
            "probabilities": {"match": 0.9, "maybe": 0.05, "no": 0.05},
            "rule": "job_order",
            "model": "m",
            "provider": "decisions",
            "tokens": {"prompt": 10, "completion": 2},
            "zdr": False,
        },
    )


def refused(failure="rate_limit"):
    return Refused(
        outcome="refused", rule="provider", reason="busy", failure=failure, retry_after=3
    )


class FakeFilterClient:
    def __init__(self, *answers):
        self.answers = list(answers) or [filter_done()]
        self.requests = []

    def llm_filter(self, inputs):
        self.requests.append(inputs)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


class FakeFilter:
    """Stands in for ``T6Filter``: scripted verdicts; records what it was asked."""

    def __init__(self, *verdicts):
        self.verdicts = list(verdicts) or ["pass"]
        self.calls = []

    async def filter(self, rule, text, reply_to_text=None):
        self.calls.append((rule, text, reply_to_text))
        verdict = self.verdicts.pop(0) if len(self.verdicts) > 1 else self.verdicts[0]
        return FilterResult(verdict)


class ConfigTests(unittest.TestCase):
    def test_the_default_filters_the_intake_topic_with_job_order_and_drops_nothing(self):
        config = Config(**MINIMAL)
        self.assertFalse(config.filter_drops)
        self.assertEqual(
            config.filter_rule(MINIMAL["intake_chat_id"], MINIMAL["intake_topic_id"]), "job_order"
        )

    def test_an_unlisted_chat_or_topic_has_no_rule(self):
        config = Config(**MINIMAL)
        self.assertIsNone(config.filter_rule(MINIMAL["intake_chat_id"], 999))
        self.assertIsNone(config.filter_rule(-1, MINIMAL["intake_topic_id"]))

    def test_entries_name_a_chat_an_optional_topic_and_a_rule(self):
        config = Config(
            **MINIMAL,
            filters=[{"chat_id": 1, "topic_id": 2, "rule": "a"}, {"chat_id": 3, "rule": "b"}],
        )
        self.assertEqual(config.filter_rule(1, 2), "a")
        self.assertIsNone(config.filter_rule(1, 9))
        self.assertEqual(config.filter_rule(3, 77), "b")

    def test_an_empty_list_filters_nothing(self):
        config = Config(**MINIMAL, filters=[])
        self.assertIsNone(config.filter_rule(MINIMAL["intake_chat_id"], MINIMAL["intake_topic_id"]))

    def test_a_rule_name_is_required_and_no_rule_text_is_accepted(self):
        for entry_ in (
            {"chat_id": 1},
            {"chat_id": 1, "rule": ""},
            {"chat_id": 1, "rule": "a", "question": "q"},
        ):
            with self.assertRaises(ValidationError):
                Config(**MINIMAL, filters=[entry_])


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def ask(self, *answers, **kw):
        client = FakeFilterClient(*answers)
        return client, await T6Filter(client).filter("job_order", "the text", **kw)

    async def test_each_verdict_comes_back(self):
        for verdict in ("pass", "drop", "unsure"):
            client, result = await self.ask(filter_done(verdict))
            self.assertEqual(result, FilterResult(verdict))

    async def test_the_request_is_the_rule_the_text_and_the_reply_text(self):
        client, _ = await self.ask(reply_to_text="the parent")
        (request,) = client.requests
        self.assertEqual(
            (request.rule, request.text, request.reply_to_text),
            ("job_order", "the text", "the parent"),
        )

    async def test_every_failure_is_unavailable_with_its_class_and_never_raises(self):
        cases = {
            "unavailable": Unavailable(reason="connection refused"),
            "timeout": Unavailable(reason="timed out"),
            "refused": refused("rate_limit"),
            "api_error": APIError("boom"),
            "bad body": Done(outcome="done", result={"verdict": "maybe"}),
        }
        for name, answer in cases.items():
            with self.subTest(name):
                _, result = await self.ask(answer)
                self.assertEqual(result.verdict, "unavailable")
                self.assertTrue(result.failure)
                self.assertNotIn("the text", result.failure)

    async def test_a_slow_call_is_unavailable_after_the_wait(self):
        class Slow:
            def llm_filter(self, inputs):
                import time

                time.sleep(0.5)
                return filter_done()

        result = await T6Filter(Slow(), timeout_seconds=0, wait_margin=0.05).filter("r", "t")
        self.assertEqual(result, FilterResult("unavailable", "timeout"))


def make(text="MFG - TOC - JO #74\nTask: cut", message_filter=None, verdict="jo", **config):
    worker, ledger, telegram, notion, cleanup = make_worker(
        [update(7, message_id=42, text=text)],
        classifier=FakeClassifier(ok_result(verdict)),
        **config,
    )
    worker.message_filter = message_filter
    return worker, ledger, notion, cleanup


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def run_one(self, flt, **kw):
        worker, ledger, notion, cleanup = make(message_filter=flt, **kw)
        self.addCleanup(cleanup)
        await worker.run_once()
        return worker, ledger, notion

    async def test_count_first_extracts_every_message_and_stores_the_verdict(self):
        flt = FakeFilter("drop")
        worker, ledger, notion = await self.run_one(flt)
        msg = ledger.get_message(KEY)
        self.assertEqual(msg["filter_verdict"], "drop")
        self.assertEqual(msg["verdict"], "jo")
        self.assertEqual(msg["state"], "done")
        self.assertEqual(len(notion.created), 1)
        self.assertEqual(worker.classifier.calls[0][0], "MFG - TOC - JO #74\nTask: cut")

    async def test_the_call_names_the_rule_and_passes_the_text(self):
        flt = FakeFilter()
        await self.run_one(flt)
        self.assertEqual(flt.calls, [("job_order", "MFG - TOC - JO #74\nTask: cut", None)])

    async def test_a_real_reply_sends_the_text_it_replies_to(self):
        flt = FakeFilter()
        worker, ledger, telegram, notion, cleanup = make_worker(
            [
                update(
                    7,
                    message_id=42,
                    text="Task: x",
                    extra={"reply_to_message": {"text": "MFG JO #1"}},
                )
            ]
        )
        self.addCleanup(cleanup)
        worker.message_filter = flt
        await worker.run_once()
        self.assertEqual(flt.calls, [("job_order", "Task: x", "MFG JO #1")])

    async def test_the_topic_root_is_not_a_reply(self):
        flt = FakeFilter()
        worker, ledger, telegram, notion, cleanup = make_worker(
            [
                update(
                    7,
                    message_id=42,
                    text="Task: x",
                    extra={"reply_to_message": {"text": "topic", "forum_topic_created": {}}},
                )
            ]
        )
        self.addCleanup(cleanup)
        worker.message_filter = flt
        await worker.run_once()
        self.assertEqual(flt.calls, [("job_order", "Task: x", None)])

    async def test_drops_on_skips_extraction_and_raises_nothing(self):
        flt = FakeFilter("drop")
        worker, ledger, notion = await self.run_one(flt, filter_drops=True)
        msg = ledger.get_message(KEY)
        self.assertEqual(msg["state"], "ignored")
        self.assertEqual(msg["filter_verdict"], "drop")
        self.assertEqual(msg["ignore_reason"], "filtered")
        self.assertIsNone(msg["verdict"])
        self.assertEqual(worker.classifier.calls, [])
        self.assertEqual(notion.created, [])

    async def test_pass_and_unsure_go_on_to_extraction_with_drops_on(self):
        for verdict in ("pass", "unsure"):
            with self.subTest(verdict):
                worker, ledger, notion = await self.run_one(FakeFilter(verdict), filter_drops=True)
                msg = ledger.get_message(KEY)
                self.assertEqual((msg["filter_verdict"], msg["state"]), (verdict, "done"))
                self.assertEqual(len(notion.created), 1)

    async def test_a_failed_filter_goes_on_to_extraction_as_unavailable_in_both_modes(self):
        for drops in (False, True):
            with self.subTest(filter_drops=drops):
                worker, ledger, notion = await self.run_one(
                    FakeFilter("unavailable"), filter_drops=drops
                )
                msg = ledger.get_message(KEY)
                self.assertEqual((msg["filter_verdict"], msg["state"]), ("unavailable", "done"))
                self.assertEqual(len(notion.created), 1)

    async def test_an_unfiltered_chat_or_topic_is_not_asked(self):
        for overrides in (
            {"filters": []},
            {"filters": [{"chat_id": -100123, "topic_id": 99, "rule": "r"}]},
        ):
            with self.subTest(overrides):
                flt = FakeFilter("drop")
                worker, ledger, notion = await self.run_one(flt, filter_drops=True, **overrides)
                self.assertEqual(flt.calls, [])
                msg = ledger.get_message(KEY)
                self.assertIsNone(msg["filter_verdict"])
                self.assertEqual(msg["state"], "done")

    async def test_a_worker_without_a_filter_changes_nothing(self):
        worker, ledger, notion = await self.run_one(None)
        self.assertIsNone(ledger.get_message(KEY)["filter_verdict"])
        self.assertEqual(ledger.get_message(KEY)["state"], "done")

    async def test_an_extraction_retry_does_not_ask_the_filter_again(self):
        from tests.fakes import failed_result

        flt = FakeFilter("pass")
        worker, ledger, telegram, notion, cleanup = make_worker(
            [update(7, message_id=42, text="Task: x")],
            classifier=FakeClassifier(failed_result("timeout"), ok_result("jo")),
        )
        self.addCleanup(cleanup)
        worker.message_filter = flt
        await worker.poll_once()
        await worker.drain_once()
        ledger._conn.execute("UPDATE messages SET next_attempt_at = '2000-01-01T00:00:00Z'")
        await worker.drain_once()
        self.assertEqual(len(flt.calls), 1)
        self.assertEqual(ledger.get_message(KEY)["state"], "done")

    async def test_the_filter_text_never_reaches_the_log(self):
        flt = FakeFilter("drop")
        with self.assertLogs("jo_watchdog", level="DEBUG") as logs:
            import logging

            logging.getLogger("jo_watchdog").debug("start")
            await self.run_one(flt)
        self.assertNotIn("Task: cut", "\n".join(logs.output))


class ReportTests(unittest.TestCase):
    def ledger(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ledger = Ledger(Path(tmp.name) / "ledger.db")
        self.addCleanup(ledger.close)
        return ledger

    def add(self, ledger, n, filter_verdict, verdict, created="2026-11-10T01:00:00Z"):
        ledger._conn.execute(
            "INSERT INTO messages (chat_id, message_id, telegram_key, update_id, raw_source,"
            " text, first_line, state, created_at, updated_at, filter_verdict, verdict)"
            " VALUES (-1, ?, ?, ?, '{}', 'SECRET text', 'SECRET line', 'done', ?, ?, ?, ?)",
            (n, f"-1:{n}", n, created, created, filter_verdict, verdict),
        )

    def test_counts_by_pair_for_a_date_range(self):
        ledger = self.ledger()
        self.add(ledger, 1, "pass", "jo")
        self.add(ledger, 2, "pass", "jo")
        self.add(ledger, 3, "drop", "jo")
        self.add(ledger, 4, "drop", "not_jo")
        self.add(ledger, 5, "drop", None)
        self.add(ledger, 6, None, "jo")
        self.add(ledger, 7, "pass", "jo", created="2026-10-01T00:00:00Z")
        counts = ledger.filter_counts("2026-11-01", "2026-12-01")
        self.assertEqual(
            counts,
            {
                ("pass", "jo"): 2,
                ("drop", "jo"): 1,
                ("drop", "not_jo"): 1,
                ("drop", "none"): 1,
                ("none", "jo"): 1,
            },
        )

    def test_the_command_prints_counts_only(self):
        ledger = self.ledger()
        self.add(ledger, 1, "drop", "jo")
        ledger._conn.execute("PRAGMA wal_checkpoint(FULL)")
        config_path = Path(tempfile.mkdtemp()) / "config.yaml"
        self.addCleanup(config_path.unlink)
        config_path.write_text(
            "intake_chat_id: -100123\nintake_topic_id: 5\nops_chat_id: -100999\nops_topic_id: 7\n"
            f"ledger_path: {ledger._path}\n"
        )
        out = io.StringIO()
        with redirect_stdout(out):
            code = entry.main(
                ["filter-report", "--config", str(config_path), "--since", "2026-11-01"]
            )
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("drop", text)
        self.assertIn("jo", text)
        for secret in ("SECRET", "-100123", "-1:1"):
            self.assertNotIn(secret, text)

    def test_the_command_takes_since_and_until(self):
        with mock.patch.object(entry, "run_filter_report", return_value=0) as run:
            entry.main(["filter-report", "--since", "2026-11-01", "--until", "2026-11-15"])
        run.assert_called_once_with("config.yaml", "2026-11-01", "2026-11-15")
