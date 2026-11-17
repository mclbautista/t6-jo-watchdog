"""``compare-raise``: the new path up to the ``jo raise`` preview against what was written (#134, #151).

Nothing is confirmed, written or sent; the report holds counts, field names and ledger row ids.
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from jo_watchdog.app import __main__ as entry
from jo_watchdog.app.compare_raise import (
    DEFAULT_MUST_MATCH,
    compare,
    exit_code,
    render,
    run_comparison,
)
from tests.test_worker_planning import KEY, Harness


class Directory:
    """``person list``: the standing cc's page has a T6 Person ID."""

    async def people(self):
        return [
            {"page_id": "c-standing", "t6_id": "t6n_standing"},
            {"page_id": "c-other", "t6_id": "t6n_other"},
        ]


async def written(test, **fields) -> Harness:
    """A harness whose one message was captured and written."""
    h = Harness(test, task="Cut", deadline_raw="Sep 30 end of shift", **fields)
    await h.run()
    return h


class CompareTests(unittest.IsolatedAsyncioTestCase):
    async def run_comparison(self, h, *, last=50):
        return await run_comparison(
            h.ledger,
            h.worker,
            last=last,
            must_match=DEFAULT_MUST_MATCH,
        )

    async def test_a_row_written_by_the_same_rules_matches_in_every_field(self):
        h = await written(self)
        report = await self.run_comparison(h)
        self.assertEqual(report.compared, 1)
        self.assertEqual(report.differences, [])
        for field in DEFAULT_MUST_MATCH:
            self.assertEqual(report.counts[field], {"same": 1, "different": 0, "missing": 0})
        self.assertEqual(exit_code(report, DEFAULT_MUST_MATCH), 0)

    async def test_nothing_is_confirmed_written_or_sent(self):
        h = await written(self)
        confirms, created = len(h.notion.confirms), len(h.notion.created)
        outbox = h.ledger._conn.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        await self.run_comparison(h)
        self.assertEqual((len(h.notion.confirms), len(h.notion.created)), (confirms, created))
        self.assertEqual(h.ledger._conn.execute("SELECT COUNT(*) FROM outbox").fetchone()[0], outbox)
        self.assertGreaterEqual(len(h.notion.previews), 2)  # the plan's and the comparison's

    async def test_a_field_that_differs_is_named_with_the_row_id_only(self):
        h = await written(self)
        row = h.ledger.get_message(KEY)
        plan = json.loads(row["plan_json"])
        plan["properties"]["Department"] = {"select": {"name": "VID"}}
        plan["properties"]["JO Number"] = {"rich_text": [{"type": "text", "text": {"content": "#9"}}]}
        h.ledger._conn.execute(
            "UPDATE messages SET plan_json = ? WHERE telegram_key = ?", (json.dumps(plan), KEY)
        )
        report = await self.run_comparison(h)
        self.assertEqual(
            sorted(report.differences), sorted([("department", row["id"]), ("number", row["id"])])
        )
        self.assertEqual(report.counts["department"]["different"], 1)
        self.assertEqual(report.counts["name"]["same"], 1)
        self.assertEqual(exit_code(report, DEFAULT_MUST_MATCH), 1)
        text = render(report)
        self.assertIn(f"department row {row['id']}", text)
        self.assertNotIn("Cut", text)  # no posted text in the report

    async def test_the_api_standing_cc_is_added_to_the_new_side(self):
        h = await written(self)
        plan = json.loads(h.ledger.get_message(KEY)["plan_json"])
        plan["properties"]["cc"] = {"relation": [{"id": "c-ana"}, {"id": "c-standing"}]}
        h.ledger._conn.execute(
            "UPDATE messages SET plan_json = ? WHERE telegram_key = ?", (json.dumps(plan), KEY)
        )
        has = await run_comparison(
            h.ledger, h.worker, last=50, must_match=("cc",), standing_cc="c-standing"
        )
        self.assertEqual(has.counts["cc"]["different"], 1)  # c-ana was never on the new side
        plan["properties"]["cc"] = {"relation": [{"id": "c-standing"}]}
        h.ledger._conn.execute(
            "UPDATE messages SET plan_json = ? WHERE telegram_key = ?", (json.dumps(plan), KEY)
        )
        has = await run_comparison(
            h.ledger, h.worker, last=50, must_match=("cc",), standing_cc="c-standing"
        )
        self.assertEqual(has.counts["cc"], {"same": 1, "different": 0, "missing": 0})
        self.assertIn("standing cc on the API: yes", render(has))
        self.assertNotIn("c-standing", render(has))
        none = await run_comparison(h.ledger, h.worker, last=50, must_match=("cc",))
        self.assertEqual(none.counts["cc"]["different"], 1)
        self.assertEqual(exit_code(none, ("cc",)), 1)
        self.assertIn("standing cc on the API: no", render(none))

    async def test_a_t6_id_answer_is_turned_into_the_page_id_by_the_people_list(self):
        from jo_watchdog.app.compare_raise import standing_cc_page_id

        h = await written(self)

        for answer, want in (("t6n_standing", "c-standing"), ("c-standing", "c-standing"), (None, None)):
            h.notion.standing_cc_answer = answer
            self.assertEqual(await standing_cc_page_id(h.notion, Directory()), want)

    async def test_only_the_must_match_kinds_decide_the_exit_code(self):
        h = await written(self)
        plan = json.loads(h.ledger.get_message(KEY)["plan_json"])
        plan["properties"]["cc"] = {"relation": [{"id": "someone-else"}]}
        h.ledger._conn.execute(
            "UPDATE messages SET plan_json = ? WHERE telegram_key = ?", (json.dumps(plan), KEY)
        )
        report = await self.run_comparison(h)
        self.assertEqual(report.counts["cc"]["different"], 1)
        self.assertEqual(exit_code(report, DEFAULT_MUST_MATCH), 0)
        self.assertEqual(exit_code(report, ("cc",)), 1)

    async def test_a_plan_from_before_the_api_rules_compares_through_its_properties(self):
        h = await written(self)
        plan = json.loads(h.ledger.get_message(KEY)["plan_json"])
        for key in ("raise_message", "duplicate_key"):
            plan.pop(key, None)
        h.ledger._conn.execute(
            "UPDATE messages SET plan_json = ? WHERE telegram_key = ?", (json.dumps(plan), KEY)
        )
        report = await self.run_comparison(h)
        self.assertEqual(report.differences, [])

    async def test_a_row_the_api_cannot_preview_is_missing_not_different(self):
        h = await written(self)
        h.notion.down = True
        report = await self.run_comparison(h)
        self.assertEqual(report.compared, 0)
        self.assertEqual(report.not_built, 1)
        self.assertEqual(report.counts["name"], {"same": 0, "different": 0, "missing": 1})
        self.assertEqual(exit_code(report, DEFAULT_MUST_MATCH), 2)  # nothing was compared

    async def test_a_row_with_no_stored_plan_is_missing(self):
        h = await written(self)
        h.ledger._conn.execute("UPDATE messages SET plan_json = NULL WHERE telegram_key = ?", (KEY,))
        report = await self.run_comparison(h)
        self.assertEqual(report.counts["name"]["missing"], 1)
        self.assertEqual(report.differences, [])

    async def test_only_the_last_n_written_messages_are_taken(self):
        h = await written(self)
        self.assertEqual((await self.run_comparison(h, last=1)).compared, 1)
        h.ledger._conn.execute("UPDATE messages SET notion_page_id = NULL")
        self.assertEqual((await self.run_comparison(h)).compared, 0)

    def test_blank_on_both_sides_is_the_same_and_one_blank_side_differs(self):
        blank = {"name": "", "department": "", "deadline": ""}
        self.assertEqual(compare(blank, blank)["department"], "same")
        self.assertEqual(compare(blank, {**blank, "department": "TOC"})["department"], "different")

    def test_the_same_instant_in_another_offset_is_the_same_deadline(self):
        a = {"deadline": "2026-09-30T22:00:00+08:00"}
        b = {"deadline": "2026-09-30T14:00:00+00:00"}
        self.assertEqual(compare(a, b)["deadline"], "same")


class CommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_command_reads_a_copy_of_the_ledger_and_prints_counts(self):
        h = await written(self)
        config_path = Path(tempfile.mkdtemp()) / "config.yaml"
        self.addCleanup(config_path.unlink)
        config_path.write_text(
            "intake_chat_id: -100123\nintake_topic_id: 5\nops_chat_id: -100999\nops_topic_id: 7\n"
            f"ledger_path: {h.ledger._path}\n"
        )
        h.ledger._conn.execute("PRAGMA wal_checkpoint(FULL)")
        seen = {}

        h.notion.standing_cc_answer = "c-standing"

        def build(config, ledger):
            seen["ledger"] = ledger
            h.worker.ledger = ledger
            h.worker.caches.directory = Directory()
            return h.worker

        out = io.StringIO()
        with mock.patch.object(entry, "build_compare_worker", build), redirect_stdout(out):
            code = await entry.compare_raise(str(config_path), 5, DEFAULT_MUST_MATCH)
        self.assertEqual(code, 0)
        self.assertIn("compared 1 of 1", out.getvalue())
        self.assertIn("name", out.getvalue())
        self.assertIn("standing cc on the API: yes", out.getvalue())
        self.assertNotEqual(seen["ledger"]._path, h.ledger._path)  # a private copy, read only
        with self.assertRaises(Exception):
            seen["ledger"]._conn.execute("DELETE FROM messages")

    def test_the_command_takes_last_and_must_match(self):
        with mock.patch.object(entry, "run_compare_raise", return_value=0) as run:
            code = entry.main(
                ["compare-raise", "--config", "c.yaml", "--last", "7", "--must-match", "name,cc"]
            )
        self.assertEqual(code, 0)
        run.assert_called_once_with("c.yaml", 7, ("name", "cc"))

    def test_the_defaults_are_fifty_messages_and_the_six_rule_fields(self):
        with mock.patch.object(entry, "run_compare_raise", return_value=0) as run:
            entry.main(["compare-raise", "--config", "c.yaml"])
        run.assert_called_once_with("c.yaml", 50, DEFAULT_MUST_MATCH)
        self.assertEqual(
            DEFAULT_MUST_MATCH,
            ("name", "number", "department", "deadline", "project", "assignees"),
        )

    def test_an_unknown_must_match_field_is_refused(self):
        with self.assertRaises(SystemExit):
            entry.main(["compare-raise", "--must-match", "name,colour"])
