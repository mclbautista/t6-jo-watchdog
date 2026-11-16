"""Contract tests for the capture boundary: the watchdog masks nothing itself.

The owner-protected ledger keeps the text as posted. The text to a model and to a Job Order
goes through the t6 API, which masks the names and withholds the credentials there; what the
watchdog logs and reports holds counts and ids, never the posted text.
"""

import logging
import tempfile
import unittest
from pathlib import Path

from jo_watchdog.app.config import Config
from jo_watchdog.app.worker import Worker
from jo_watchdog.ledger.db import Ledger
from tests.fakes import FakeClassifier, FakeJobOrders, FakeTelegram, update

SOURCE = "PW: swordfish-credential\nordinary detail"


def make_config() -> Config:
    return Config(
        intake_chat_id=-100123,
        intake_topic_id=5,
        ops_chat_id=-100999,
        ops_topic_id=7,
        captured_reaction="✅",
    )


class CaptureSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_text_stays_as_posted_in_the_ledger_and_reaches_the_api_and_no_log(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        data_dir = Path(tmp.name) / "data"
        data_dir.mkdir()
        ledger = Ledger(data_dir / "ledger.db")
        self.addCleanup(ledger.close)
        telegram = FakeTelegram([update(7, message_id=42, text=SOURCE)])
        notion = FakeJobOrders(wrong_name=True)
        worker = Worker(
            ledger=ledger,
            telegram=telegram,
            job_orders=notion,
            classifier=FakeClassifier(),
            config=make_config(),
        )

        with self.assertLogs("jo_watchdog.worker", level=logging.WARNING) as captured:
            await worker.run_once()

        message = ledger.get_message("-100123:42")
        assert message is not None
        self.assertEqual(message["text"], SOURCE)
        # The watchdog keeps no masked copy of its own: the columns hold the values that make
        # the previous image leave the row alone on a rollback, and no text.
        self.assertEqual(message["masked_text"], "")
        self.assertEqual(message["masked_count"], 0)

        # The text goes to the API as posted: the API withholds the credential on its write.
        [raised] = notion.confirms
        self.assertEqual(raised.message.header_text, "PW: swordfish-credential")
        self.assertTrue(raised.fields["source_text"].endswith(SOURCE))

        # A log line holds the property that differs, never the text.
        structured_logs = "\n".join(captured.output)
        self.assertNotIn("swordfish-credential", structured_logs)
        self.assertIn("read-back mismatch for -100123:42: Name", structured_logs)

    async def test_ignore_log_keeps_the_first_line_as_written(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        data_dir = Path(tmp.name) / "data"
        data_dir.mkdir()
        ledger = Ledger(data_dir / "ledger.db")
        self.addCleanup(ledger.close)
        telegram = FakeTelegram([update(7, message_id=42, topic_id=999, text=SOURCE)])
        notion = FakeJobOrders()
        worker = Worker(
            ledger=ledger,
            telegram=telegram,
            job_orders=notion,
            classifier=FakeClassifier(),
            config=make_config(),
        )

        await worker.poll_once()

        row = ledger._conn.execute("SELECT first_line FROM ignore_log").fetchone()
        assert row is not None
        self.assertEqual(row["first_line"], "PW: swordfish-credential")


if __name__ == "__main__":
    unittest.main()
