"""Job Orders are raised through ``jo raise`` only: the two-step flow, its recovery and its inputs (#134, #151)."""

import tempfile
import unittest
from pathlib import Path

from jo_watchdog.adapters.t6api import KeyAlreadyHeld
from jo_watchdog.ledger.db import Ledger
from tests.test_worker_planning import KEY, Harness


class RaiseFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_preview_comes_first_then_one_confirm_with_the_same_key(self):
        h = Harness(self, task="Cut")
        await h.run()
        self.assertGreaterEqual(len(h.notion.previews), 1)
        [confirmed] = h.notion.confirms
        self.assertEqual(confirmed.idempotency_key, f"watchdog-jo-{KEY}")
        self.assertEqual(h.notion.previews[-1].idempotency_key, confirmed.idempotency_key)
        row = h.ledger.get_message(KEY)
        self.assertEqual(row["state"], "done")
        self.assertEqual(row["jo_id"], "t6j_1")
        self.assertEqual(row["notion_page_id"], "page-1")

    async def test_the_raw_posted_facts_reach_jo_raise_and_the_api_builds_the_rest(self):
        h = Harness(self, task="Cut", deadline_raw="Sep 30 end of shift")
        await h.run()
        message = h.notion.confirms[0].message
        self.assertEqual(message.header_text, "MFG - TOC - JO #74")
        self.assertEqual(message.deadline_text, "Sep 30 end of shift")
        self.assertEqual((message.chat_id, message.message_id), ("-100123", "42"))
        # The Name, number, Department and deadline are the API's, not the watchdog's.
        props = h.notion.created[0]["properties"]
        self.assertEqual(props["JO Number"]["rich_text"][0]["text"]["content"], "#74")
        self.assertEqual(props["Department"], {"select": {"name": "TOC"}})
        self.assertEqual(props["Deadline"], {"date": {"start": "2026-09-30T22:00:00+08:00"}})

    async def test_the_duplicate_key_is_the_one_the_preview_returned(self):
        h = Harness(self, task="Cut")
        await h.run()
        self.assertEqual(h.plan()["duplicate_key"], "proj-mfg|74")
        self.assertEqual(h.ledger.get_message(KEY)["jo_key"], "proj-mfg|74")

    async def test_a_late_page_keeps_the_row_writing_then_finishes_without_a_second_raise(self):
        h = Harness(self, task="Cut")
        h.notion.page_delay = 3
        h.worker.config = h.config.model_copy(update={"raise_poll_seconds": 0})
        await h.run()
        row = h.ledger.get_message(KEY)
        self.assertEqual(row["state"], "writing")
        self.assertEqual(row["jo_id"], "t6j_1")
        self.assertFalse(row["notion_page_id"])
        h.worker.config = h.config.model_copy(update={"raise_poll_seconds": 60})

        async def no_sleep(_):
            return None

        h.worker.sleep = no_sleep
        h.ledger._conn.execute(
            "UPDATE messages SET next_attempt_at = '2000-01-01T00:00:00Z' WHERE telegram_key = ?",
            (KEY,),
        )
        await h.worker.drain_once()
        self.assertEqual(h.ledger.get_message(KEY)["state"], "done")
        self.assertEqual(h.notion.create_calls, 1)

    async def test_a_key_the_api_already_holds_adopts_the_page_that_jo_find_shows(self):
        h = Harness(self, task="Cut")
        real = h.notion.confirm_job_order
        finds = []

        async def held(inputs, token):
            await real(inputs, token)
            raise KeyAlreadyHeld("key held")

        h.notion.confirm_job_order = held
        original = h.notion.find_job_orders

        async def find(chat_id, message_id):
            finds.append(1)
            return [] if len(finds) == 1 else await original(chat_id, message_id)

        h.notion.find_job_orders = find
        await h.run()
        row = h.ledger.get_message(KEY)
        self.assertEqual(row["state"], "done")
        self.assertEqual(row["notion_page_id"], "page-1")
        self.assertEqual(h.notion.create_calls, 1)

    async def test_an_api_that_is_down_raises_nothing_and_the_row_waits(self):
        h = Harness(self, task="Cut")
        h.notion.down = True
        await h.run()
        row = h.ledger.get_message(KEY)
        self.assertNotEqual(row["state"], "done")
        self.assertEqual(h.notion.created, [])
        self.assertEqual(h.telegram.reactions, [])


class MigrationTests(unittest.TestCase):
    def test_migration_12_adds_a_nullable_jo_id_and_leaves_old_rows_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp) / "ledger.db")
            self.addCleanup(ledger.close)
            columns = {r["name"]: r for r in ledger._conn.execute("PRAGMA table_info(messages)")}
            self.assertIn("jo_id", columns)
            self.assertEqual(columns["jo_id"]["notnull"], 0)
