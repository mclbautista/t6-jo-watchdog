"""Tests for the pure Notion payload builder and read-back comparison."""

import unittest

from jo_watchdog.core.job_order import JobOrderPlan
from jo_watchdog.core.payload import (
    build_raise_fields,
    readback_fields,
    verify_readback,
)
from jo_watchdog.core.plan import Plan
from jo_watchdog.core.verify import readback_properties


def stored_plan(**overrides) -> JobOrderPlan:
    properties = {
        "Name": {"title": [{"type": "text", "text": {"content": "MFG - TOC - JO #74"}}]},
        "Telegram Key": {"rich_text": [{"type": "text", "text": {"content": "-100:1"}}]},
        "Status": {"status": {"name": "Needs Intake Review"}},
        "Date": {"date": {"start": "2026-09-28T11:00:00+08:00"}},
        "JO Number": {"rich_text": [{"type": "text", "text": {"content": "#74"}}]},
        "Department": {"select": {"name": "TOC"}},
        "Deadline": {"date": {"start": "2026-09-30T22:00:00+08:00"}},
        "Task": {"rich_text": [{"type": "text", "text": {"content": "Cut teaser"}}]},
        "Watchdog Notes": {"rich_text": [{"type": "text", "text": {"content": "a\nb"}}]},
        "Assigned": {"relation": [{"id": "c-ben"}]},
        "cc": {"relation": [{"id": "c-ana"}]},
        "Projects": {"relation": [{"id": "proj-mfg"}]},
    }
    return JobOrderPlan(
        telegram_key="-100:1",
        name="MFG - TOC - JO #74",
        properties=properties,
        body_blocks=["line one", "line two"],
        **overrides,
    )


class BuildRaiseFieldsTests(unittest.TestCase):
    def test_a_message_plan_leaves_the_rule_fields_to_the_api(self):
        fields = build_raise_fields(stored_plan(raise_message={"header_text": "x"}))
        for rule in ("name", "number", "department", "deadline", "telegram_key", "project_id"):
            self.assertNotIn(rule, fields)
        self.assertEqual(fields["status"], "Needs Intake Review")
        self.assertEqual(fields["date"], "2026-09-28T11:00:00+08:00")
        self.assertEqual(fields["task"], "Cut teaser")
        self.assertEqual(fields["watchdog_notes"], "a\nb")
        self.assertEqual(fields["assigned_ids"], ["c-ben"])
        self.assertEqual(fields["cc_ids"], ["c-ana"])
        self.assertEqual(fields["source_text"], "line one\nline two")

    def test_a_plan_stored_before_the_switch_sends_its_own_fields(self):
        fields = build_raise_fields(stored_plan())
        self.assertEqual(fields["name"], "MFG - TOC - JO #74")
        self.assertEqual(fields["number"], "#74")
        self.assertEqual(fields["department"], "TOC")
        self.assertEqual(fields["deadline"], "2026-09-30T22:00:00+08:00")
        self.assertEqual(fields["telegram_key"], "-100:1")
        self.assertEqual(fields["project_id"], "proj-mfg")

    def test_a_plan_with_no_body_sends_no_body(self):
        plan = stored_plan(raise_message={})
        plan.body_blocks = []
        self.assertNotIn("source_text", build_raise_fields(plan))


class VerifyReadbackTests(unittest.TestCase):
    def test_match_returns_no_mismatches(self):
        plan = Plan(name="N", telegram_key="k", body_blocks=["b"])
        self.assertEqual(verify_readback(plan, retrieved_name="N", retrieved_telegram_key="k"), [])

    def test_name_mismatch_is_reported(self):
        plan = Plan(name="N", telegram_key="k", body_blocks=["b"])
        mismatches = verify_readback(plan, retrieved_name="WRONG", retrieved_telegram_key="k")
        self.assertIn("Name", " ".join(mismatches))

    def test_telegram_key_mismatch_is_reported(self):
        plan = Plan(name="N", telegram_key="k", body_blocks=["b"])
        mismatches = verify_readback(plan, retrieved_name="N", retrieved_telegram_key="other")
        self.assertIn("Telegram Key", " ".join(mismatches))


class DateReadbackTests(unittest.TestCase):
    PLAN = JobOrderPlan(
        telegram_key="k",
        name="N",
        properties={
            "Date": {"date": {"start": "2026-10-27T17:20:02+08:00"}},
            "Deadline": {"date": {"start": "2026-10-30T05:00:00+08:00"}},
        },
    )

    def check(self, dates):
        return verify_readback(
            self.PLAN, retrieved_name="N", retrieved_telegram_key="k", retrieved_dates=dates
        )

    def test_live_shape_with_dropped_seconds_and_null_zone_matches(self):
        self.assertEqual(
            self.check(
                {
                    "Date": {"start": "2026-10-27T17:20:00.000+08:00", "time_zone": None},
                    "Deadline": {"start": "2026-10-30T05:00:00.000+08:00", "time_zone": None},
                }
            ),
            [],
        )

    def test_same_instant_at_another_offset_matches(self):
        dates = {
            "Date": {"start": "2026-10-27T09:20:00+00:00"},
            "Deadline": {"start": "2026-10-29T21:00:00Z"},
        }
        self.assertEqual(self.check(dates), [])

    def test_the_issue_85_eight_hour_shift_is_reported_for_both_properties(self):
        mismatches = self.check(
            {
                "Date": {"start": "2026-10-27T09:20:00.000+08:00", "time_zone": None},
                "Deadline": {"start": "2026-10-29T21:00:00.000+08:00", "time_zone": None},
            }
        )
        self.assertEqual(readback_properties(mismatches), ["Date", "Deadline"])

    def test_missing_date_is_reported(self):
        mismatches = self.check({"Date": None, "Deadline": {"start": "2026-10-30T05:00:00+08:00"}})
        self.assertEqual(readback_properties(mismatches), ["Date"])

    def test_naive_start_is_read_in_its_time_zone(self):
        dates = {
            "Date": {"start": "2026-10-27T17:20:00", "time_zone": "Asia/Manila"},
            "Deadline": {"start": "2026-10-30T05:00:00", "time_zone": "Asia/Manila"},
        }
        self.assertEqual(self.check(dates), [])

    def test_readback_fields_read_both_dates_from_a_job_order(self):
        job_order = {
            "name": "N",
            "telegram_key": "-1:2",
            "date": "2026-09-15T10:00:00+08:00",
            "deadline": None,
        }
        self.assertEqual(
            readback_fields(job_order),
            {
                "name": "N",
                "telegram_key": "-1:2",
                "dates": {"Date": {"start": "2026-09-15T10:00:00+08:00"}, "Deadline": None},
            },
        )

    def test_a_job_order_without_a_key_or_dates_reads_back_blank(self):
        self.assertEqual(
            readback_fields({"name": None, "telegram_key": None}),
            {"name": "", "telegram_key": "", "dates": {"Date": None, "Deadline": None}},
        )


if __name__ == "__main__":
    unittest.main()
