from jo_watchdog.core.plan import NOTES_PROPERTY

"""The Job Order plan: the API's Name, number and deadline, fields, Notes, and replay."""

import json
import unittest
from datetime import datetime
from types import SimpleNamespace

from jo_watchdog.core.job_order import (
    ASAP_NOTE,
    LEGACY_WATCHDOG_MARKER,
    MANILA,
    JobOrderPlan,
    RaisePreview,
    build_notes,
    build_watchdog_notes,
    load_plan,
    plan_job_order,
    rich_text,
)
from jo_watchdog.core.payload import build_raise_fields

POSTED = datetime(2026, 9, 15, 10, 0, tzinfo=MANILA)
WITHHELD = "[password withheld – see Telegram message]"

FIELDS = [
    "header_raw", "jo_number_raw", "project_raw", "department_raw", "title", "task", "purpose",
    "source", "platform", "link_settings", "watermark", "mode_of_transfer", "blocker", "pending",
    "notes", "deadline_raw", "subtitle",
]  # fmt: skip


def extraction(**values):
    return SimpleNamespace(**{name: values.get(name) for name in FIELDS})


def rules(**values):
    """What ``jo raise`` previews for a message (the API builds these, the plan takes them)."""
    base = dict(
        token="t",
        name="MFG - TOC - JO #74",
        jo_number="#74",
        department="TOC",
        telegram_key="-1001:42",
        duplicate_key="proj|74",
    )
    return RaisePreview(**{**base, **values})


def plan(ext, raw_text="MFG - TOC - JO #74\nbody", **kwargs):
    kwargs.setdefault("rules", rules())
    return plan_job_order(
        ext,
        telegram_key="-1001:42",
        raw_text=raw_text,
        posted_at=POSTED,
        source_block="Posted by: Priya (@priya)",
        **kwargs,
    )


def text_of(prop):
    return "".join(item["text"]["content"] for item in prop["rich_text"])


class RuleValuesTests(unittest.TestCase):
    def test_the_apis_name_number_department_and_keys_are_the_plans(self):
        p = plan(
            extraction(),
            rules=rules(name="SOMSOM - TOC - JO #0155", jo_number="#0155", duplicate_key="p|155"),
        )
        self.assertEqual(p.name, "SOMSOM - TOC - JO #0155")
        self.assertEqual(text_of(p.properties["JO Number"]), "#0155")
        self.assertEqual(p.properties["Department"], {"select": {"name": "TOC"}})
        self.assertEqual(p.duplicate_key, "p|155")

    def test_the_plan_applies_no_rule_of_its_own_to_the_header(self):
        # The posted text and the extraction say one thing, the API another: the API wins.
        p = plan(
            extraction(header_raw="ALTA-TO-JO04", jo_number_raw="#9", department_raw="OE"),
            raw_text="ALTA-TO-JO04",
            rules=rules(name="ALTA - TOC - JO #04", jo_number="#04"),
        )
        self.assertEqual(p.name, "ALTA - TOC - JO #04")
        self.assertEqual(text_of(p.properties["JO Number"]), "#04")

    def test_no_department_and_no_number_write_neither(self):
        p = plan(extraction(), rules=rules(name="JOB ORDER", jo_number=None, department=None))
        self.assertEqual(p.name, "JOB ORDER")
        self.assertNotIn("JO Number", p.properties)
        self.assertNotIn("Department", p.properties)

    def test_a_message_plan_keeps_its_raise_message(self):
        p = plan(extraction(), raise_message={"header_text": "x"})
        self.assertEqual(p.raise_message, {"header_text": "x"})
        self.assertIsNone(plan(extraction()).raise_message)


class FieldTests(unittest.TestCase):
    def test_text_fields_are_verbatim_with_exact_property_names(self):
        p = plan(
            extraction(
                header_raw="MFG - TOC - JO #74",
                title="Show Alpha FTR",
                task="D14-15 Upload synced dailies",
                purpose="For prod and NF review",
                source="Offline Owen",
                platform="Frame.io",
                link_settings="N/A",
                watermark="Follow netflix guidelines",
                mode_of_transfer="Contenthub, share thru PIX",
                blocker="Waiting for waiver",
                pending="Upload to PIX",
            )
        )
        expected = {
            "Title": "Show Alpha FTR",
            "Task": "D14-15 Upload synced dailies",
            "Purpose": "For prod and NF review",
            "Source / Source Drive": "Offline Owen",
            "Platform": "Frame.io",
            "Link Settings": "N/A",
            "Watermark": "Follow netflix guidelines",
            "Mode of Transfer": "Contenthub, share thru PIX",
            "Blocker ": "Waiting for waiver",
            "Pending": "Upload to PIX",
        }
        for prop, value in expected.items():
            self.assertEqual(text_of(p.properties[prop]), value, prop)

    def test_absent_values_stay_blank(self):
        p = plan(extraction(header_raw="MFG - TOC - JO #74", watermark="  "))
        for prop in [
            "Title", "Watermark", "Blocker ", "Subtitle", "Notes", "Watchdog Notes", "Deadline",
        ]:  # fmt: skip
            self.assertNotIn(prop, p.properties)

    def test_status_date_and_telegram_key(self):
        p = plan(extraction(header_raw="MFG - TOC - JO #74"))
        self.assertEqual(p.properties["Status"], {"status": {"name": "Needs Intake Review"}})
        self.assertEqual(
            p.properties["Date"],
            {"date": {"start": "2026-09-15T10:00:00+08:00"}},
        )
        self.assertEqual(text_of(p.properties["Telegram Key"]), "-1001:42")

    def test_subtitle_is_only_burned_in_or_none(self):
        for value, expected in [
            ("Burned-in: English", "Burned-in: English"),
            ("None", "None"),
            ("English", None),
            (None, None),
        ]:
            p = plan(extraction(header_raw="MFG - TOC - JO #74", subtitle=value))
            actual = text_of(p.properties["Subtitle"]) if "Subtitle" in p.properties else None
            self.assertEqual(actual, expected, value)

    def test_poster_subtitle_placeholders_survive_as_posted(self):
        for value in ["-", "N/A", "none"]:
            p = plan(extraction(header_raw="MFG - TOC - JO #74", subtitle=value))
            self.assertEqual(text_of(p.properties["Subtitle"]), value)

    def test_title_and_notes_keep_whitespace_and_nbsp(self):
        p = plan(
            extraction(
                header_raw="MFG - TOC - JO #74", title="  keep\u00a0this\n", notes="  a\u00a0b\n"
            )
        )
        self.assertEqual(text_of(p.properties["Title"]), "  keep\u00a0this\n")
        self.assertEqual(text_of(p.properties[NOTES_PROPERTY]), "  a\u00a0b\n")

    def test_whitespace_only_title_and_notes_are_blank(self):
        p = plan(extraction(header_raw="MFG - TOC - JO #74", title="\u00a0 ", notes=" \n"))
        self.assertNotIn("Title", p.properties)
        self.assertNotIn(NOTES_PROPERTY, p.properties)

    def test_poster_notes_and_watchdog_notes_go_to_separate_columns(self):
        poster = " " + "x" * 2500 + "\n"
        p = plan(
            extraction(header_raw="MFG - TOC - JO #74", notes=poster, deadline_raw="ASAP"),
            rules=rules(deadline_findings=[{"code": "asap"}]),
        )
        for column in (NOTES_PROPERTY, "Watchdog Notes"):
            items = p.properties[column]["rich_text"]
            self.assertTrue(all(len(i["text"]["content"]) <= 2000 for i in items), column)
        self.assertEqual(text_of(p.properties[NOTES_PROPERTY]), poster)
        self.assertEqual(
            text_of(p.properties["Watchdog Notes"]),
            f"Deadline as posted: ASAP\n{ASAP_NOTE}",
        )

    def test_notes_column_never_carries_watchdog_text(self):
        p = plan(
            extraction(header_raw="MFG - TOC - JO #74", deadline_raw="ASAP"),
            rules=rules(deadline_findings=[{"code": "asap"}]),
        )
        self.assertNotIn(NOTES_PROPERTY, p.properties)
        self.assertEqual(
            text_of(p.properties["Watchdog Notes"]),
            f"Deadline as posted: ASAP\n{ASAP_NOTE}",
        )

    def test_watchdog_notes_absent_when_watchdog_adds_nothing(self):
        p = plan(extraction(header_raw="MFG - TOC - JO #74", notes="poster only"))
        self.assertEqual(text_of(p.properties[NOTES_PROPERTY]), "poster only")
        self.assertNotIn("Watchdog Notes", p.properties)

    def test_stored_plan_with_legacy_marker_block_is_split_on_load(self):
        legacy = JobOrderPlan(
            telegram_key="-1001:42",
            name="MFG - TOC - JO #74",
            properties={
                "Name": {"title": rich_text("MFG - TOC - JO #74")},
                "Notes": {
                    "rich_text": rich_text(
                        f"poster note\n\n{LEGACY_WATCHDOG_MARKER}\nDeadline as posted: 1pm"
                    )
                },
            },
        )
        loaded = load_plan(legacy.model_dump_json())
        self.assertEqual(text_of(loaded.properties[NOTES_PROPERTY]), "poster note")
        self.assertEqual(
            text_of(loaded.properties["Watchdog Notes"]), "Deadline as posted: 1pm"
        )

    def test_stored_plan_with_only_legacy_block_drops_notes(self):
        legacy = JobOrderPlan(
            telegram_key="-1001:42",
            name="n",
            properties={
                "Notes": {"rich_text": rich_text(f"{LEGACY_WATCHDOG_MARKER}\nEarlier row: u")},
            },
        )
        loaded = load_plan(legacy.model_dump_json())
        self.assertNotIn("Notes", loaded.properties)
        self.assertEqual(text_of(loaded.properties["Watchdog Notes"]), "Earlier row: u")

    def test_marker_quoted_in_new_format_poster_notes_is_left_alone(self):
        quoted = f"see earlier JO\n\n{LEGACY_WATCHDOG_MARKER}\nsomething posted by hand"
        stored = JobOrderPlan(
            telegram_key="-1001:42",
            name="n",
            properties={
                NOTES_PROPERTY: {"rich_text": rich_text(quoted)},
                "Watchdog Notes": {"rich_text": rich_text("Deadline as posted: 1pm")},
            },
        )
        loaded = load_plan(stored.model_dump_json())
        self.assertEqual(text_of(loaded.properties[NOTES_PROPERTY]), quoted)
        self.assertEqual(
            text_of(loaded.properties["Watchdog Notes"]), "Deadline as posted: 1pm"
        )

    def test_skeleton_plan_with_marker_block_drops_the_marker_line(self):
        loaded = load_plan(
            {
                "telegram_key": "-1001:1",
                "name": "n",
                "notes": f"{LEGACY_WATCHDOG_MARKER}\nEarlier row: u (exact re-post)",
            }
        )
        self.assertEqual(
            text_of(loaded.properties["Watchdog Notes"]), "Earlier row: u (exact re-post)"
        )

    def test_skeleton_plan_notes_are_watchdog_notes(self):
        loaded = load_plan(
            {"telegram_key": "-1001:1", "name": "n", "notes": "Earlier row: u (exact re-post)"}
        )
        self.assertNotIn("Notes", loaded.properties)
        self.assertEqual(
            text_of(loaded.properties["Watchdog Notes"]), "Earlier row: u (exact re-post)"
        )

    def test_deadline_property_source_and_findings_are_the_apis(self):
        p = plan(
            extraction(deadline_raw="September 16, 2am"),
            rules=rules(
                deadline="2026-09-16T02:00:00+08:00",
                deadline_source="stated",
                deadline_findings=[{"code": "deadline_past", "params": {"days": 3}}],
            ),
        )
        self.assertEqual(
            p.properties["Deadline"],
            {"date": {"start": "2026-09-16T02:00:00+08:00"}},
        )
        self.assertEqual(p.deadline_source, "stated")
        self.assertEqual([(f.code, f.params) for f in p.findings], [("deadline_past", {"days": 3})])

    def test_no_deadline_from_the_api_writes_none(self):
        p = plan(extraction(deadline_raw="whenever"), rules=rules(deadline_source="blank"))
        self.assertNotIn("Deadline", p.properties)
        self.assertEqual(p.deadline_source, "blank")

    def test_body_is_source_block_then_masked_raw_text(self):
        p = plan(extraction(), raw_text="MFG - TOC - JO #74\nPW: secret")
        self.assertEqual(
            p.body_blocks,
            ["Posted by: Priya (@priya)", f"MFG - TOC - JO #74\nPW: {WITHHELD}"],
        )

    def test_long_raw_text_splits_into_2000_character_blocks(self):
        p = plan(extraction(), raw_text="a" * 4500)
        self.assertTrue(all(len(b) <= 2000 for b in p.body_blocks))
        self.assertEqual("".join(p.body_blocks[1:]), "a" * 4500)


class NotesTests(unittest.TestCase):
    def test_poster_notes_verbatim(self):
        self.assertEqual(build_notes("waiting for waiver"), "waiting for waiver")

    def test_watchdog_notes_items_in_order(self):
        self.assertEqual(
            build_watchdog_notes("September 10, asap", True),
            "Deadline as posted: September 10, asap\nASAP — deprioritize other JOs",
        )

    def test_watchdog_notes_deadline_alone(self):
        self.assertEqual(
            build_watchdog_notes("Tonight if keri or tom", False),
            "Deadline as posted: Tonight if keri or tom",
        )

    def test_nothing_to_add(self):
        self.assertEqual(build_notes(None), "")
        self.assertEqual(build_notes(""), "")
        self.assertEqual(build_watchdog_notes(None, False), "")
        self.assertEqual(build_watchdog_notes("  ", False), "")

    def test_notes_are_masked(self):
        self.assertEqual(build_notes("pw: abc"), f"pw: {WITHHELD}")

    def test_plan_notes_keep_hedges_and_asap(self):
        p = plan(
            extraction(
                header_raw="SOY - AE - JOB ORDER 25",
                deadline_raw="September 10, asap",
                notes="pls prioritize",
            ),
            rules=rules(
                deadline_findings=[
                    {"code": "asap"},
                    {"code": "shift_fallback"},
                    {"code": "deadline_past"},
                ]
            ),
        )
        self.assertEqual(text_of(p.properties[NOTES_PROPERTY]), "pls prioritize")
        self.assertEqual(
            text_of(p.properties["Watchdog Notes"]),
            "Deadline as posted: September 10, asap\nASAP — deprioritize other JOs",
        )
        self.assertEqual([f.code for f in p.findings], ["asap", "shift_fallback", "deadline_past"])

    def test_unparsed_deadline_is_kept_in_watchdog_notes(self):
        p = plan(
            extraction(deadline_raw="kung kelan matapos yung render"),
            rules=rules(deadline_findings=[{"code": "deadline_unparsed"}]),
        )
        self.assertNotIn("Deadline", p.properties)
        self.assertIn(
            "Deadline as posted: kung kelan matapos yung render",
            text_of(p.properties["Watchdog Notes"]),
        )
        self.assertEqual([f.code for f in p.findings], ["deadline_unparsed"])

    def test_long_notes_split_into_2000_character_items(self):
        p = plan(extraction(notes="n" * 4100))
        items = p.properties[NOTES_PROPERTY]["rich_text"]
        self.assertEqual([len(i["text"]["content"]) for i in items], [2000, 2000, 100])


class ReplayTests(unittest.TestCase):
    def test_stored_plan_replays_to_identical_fields(self):
        p = plan(
            extraction(deadline_raw="within shift", notes="note", title="T"),
            rules=rules(deadline="2026-09-15T19:00:00+08:00", deadline_source="shift"),
            raise_message={"header_text": "MFG - TOC - JO #74"},
        )
        fields = build_raise_fields(p)
        replayed = JobOrderPlan.model_validate(json.loads(p.model_dump_json()))
        self.assertEqual(replayed, p)
        self.assertEqual(build_raise_fields(replayed), fields)

    def test_fields_carry_the_body_one_line_each(self):
        p = plan(extraction(), raise_message={})
        self.assertEqual(
            build_raise_fields(p)["source_text"],
            "Posted by: Priya (@priya)\nMFG - TOC - JO #74\nbody",
        )


if __name__ == "__main__":
    unittest.main()


class LoadPlanTests(unittest.TestCase):
    def test_a_skeleton_plan_is_upgraded_to_a_replayable_one(self):
        legacy = {
            "name": "First line",
            "telegram_key": "-100:1",
            "body_blocks": ["body"],
            "notes": "Earlier row: x",
            "duplicate": {"code": "exact_repost"},
        }
        plan = load_plan(legacy)
        fields = build_raise_fields(plan)
        self.assertEqual(fields["status"], "Needs Intake Review")
        self.assertEqual(fields["name"], "First line")
        self.assertEqual(fields["telegram_key"], "-100:1")
        self.assertNotIn("notes", fields)
        self.assertEqual(fields["watchdog_notes"], "Earlier row: x")
        self.assertEqual(plan.duplicate, {"code": "exact_repost"})
