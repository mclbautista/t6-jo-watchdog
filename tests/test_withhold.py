"""Message text is sent through ``text withhold`` before the capture report and the dry-run plan."""

import unittest

from jo_watchdog.adapters.t6api import ApiUnreachable, T6ApiError
from jo_watchdog.core.report import DRY_RUN_HEADER, FLAGGED_HEADER
from tests.fakes import FakeClassifier, FakeJobOrders, ok_result, update
from tests.test_forwards_edits import Harness as EditHarness
from tests.test_forwards_edits import edit_update, message_update
from tests.test_outbox import Harness
from tests.test_worker import make_worker

HEADER = "MG - TOC - JO #41 password: something"
KEY = "-100123:42"


def unsure_classifier():
    return FakeClassifier(
        ok_result("unsure", reasons="The header says password: quoted", title="T", task="X")
    )


class CaptureReportTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_report_shows_no_credential_from_the_header_or_the_reasons(self):
        h = Harness(
            self, [update(1, message_id=42, text=HEADER)], classifier=unsure_classifier()
        )
        await h.worker.run_once()
        [(_, _, text)] = h.telegram.sent
        self.assertIn(FLAGGED_HEADER, text)
        self.assertIn("MG - TOC - JO #41", text)
        self.assertNotIn("something", text)
        self.assertNotIn("quoted", text)

    async def test_one_call_carries_every_text_of_the_report(self):
        h = Harness(
            self, [update(1, message_id=42, text=HEADER)], classifier=unsure_classifier()
        )
        await h.worker.run_once()
        self.assertEqual(len(h.notion.withhold_calls), 1)
        self.assertIn(HEADER, h.notion.withhold_calls[0])

    async def test_a_clean_capture_makes_no_call(self):
        h = Harness(self, [update(1, message_id=42, text="MG - TOC - JO #41")])
        await h.worker.run_once()
        self.assertEqual(h.notion.withhold_calls, [])

    async def test_when_the_api_fails_nothing_is_posted_and_the_step_is_retried(self):
        for error in (ApiUnreachable("down"), T6ApiError("refused")):
            with self.subTest(error=type(error).__name__):
                notion = FakeJobOrders()
                notion.withhold_down = error
                h = Harness(
                    self,
                    [update(1, message_id=42, text=HEADER)],
                    classifier=unsure_classifier(),
                    notion=notion,
                )
                await h.worker.run_once()
                self.assertEqual(h.telegram.sent, [])
                self.assertNotEqual(h.ledger.get_message(KEY)["state"], "done")


class DryRunPlanTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_plan_shows_no_credential_from_the_header(self):
        worker, ledger, telegram, notion, cleanup = make_worker(
            [update(7, message_id=42, text=HEADER)], dry_run=True
        )
        self.addCleanup(cleanup)
        await worker.run_once()
        await worker.run_once()
        [(_, _, text)] = [s for s in telegram.sent if DRY_RUN_HEADER in s[2]]
        self.assertIn("MG - TOC - JO #41", text)
        self.assertNotIn("something", text)
        self.assertEqual(len(notion.withhold_calls), 1)

    async def test_when_the_api_fails_no_plan_is_posted(self):
        for error in (ApiUnreachable("down"), T6ApiError("refused")):
            with self.subTest(error=type(error).__name__):
                worker, ledger, telegram, notion, cleanup = make_worker(
                    [update(7, message_id=42, text=HEADER)], dry_run=True
                )
                self.addCleanup(cleanup)
                notion.withhold_down = error
                await worker.run_once()
                await worker.run_once()
                self.assertEqual([s for s in telegram.sent if DRY_RUN_HEADER in s[2]], [])
                self.assertEqual(ledger.get_message(KEY)["state"], "extracted")


class EditReportTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_edit_report_shows_no_credential_from_the_name(self):
        h = EditHarness(self)
        await h.feed(message_update(1, 10, f"{HEADER}\nTask: original"))
        await h.feed(edit_update(2, 10, f"{HEADER}\nTask: changed"))
        edit_posts = [p for p in h.posts() if p.startswith("✏️")]
        self.assertEqual(len(edit_posts), 1)
        self.assertNotIn("something", edit_posts[0])
        self.assertIn("MG - TOC - JO #41", edit_posts[0])
        self.assertNotIn("something", "\n".join(h.posts()))

    async def test_an_edit_made_in_the_pipeline_is_reported_withheld(self):
        h = EditHarness(self)
        h.telegram.updates = [
            message_update(1, 10, f"{HEADER}\nTask: original"),
            edit_update(2, 10, f"{HEADER}\nTask: changed"),
        ]
        await h.worker.poll_once()
        await h.worker.drain_once()
        await h.worker.deliver_outbox()
        self.assertNotIn("something", "\n".join(h.posts()))
        self.assertEqual(len([p for p in h.posts() if p.startswith("✏️")]), 1)

    async def test_the_fallback_first_line_is_withheld_too(self):
        h = EditHarness(self)
        await h.feed(message_update(1, 10, f"{HEADER}\nTask: original"))
        h.ledger._conn.execute("UPDATE messages SET plan_json = NULL")
        await h.feed(edit_update(2, 10, "changed"))
        self.assertNotIn("something", "\n".join(h.posts()))

    async def test_when_the_api_fails_the_edit_is_not_recorded_or_posted(self):
        for error in (ApiUnreachable("down"), T6ApiError("refused")):
            with self.subTest(error=type(error).__name__):
                h = EditHarness(self)
                await h.feed(message_update(1, 10, f"{HEADER}\nTask: original"))
                h.notion.withhold_down = error
                with self.assertRaises(type(error)):
                    await h.feed(edit_update(2, 10, f"{HEADER}\nTask: changed"))
                self.assertEqual(h.edits(10), [])
                self.assertEqual([p for p in h.posts() if p.startswith("✏️")], [])


if __name__ == "__main__":
    unittest.main()
