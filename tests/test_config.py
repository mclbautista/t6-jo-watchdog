"""Configuration reference: defaults, validation, and the example files."""

import re
import tempfile
import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from jo_watchdog.app.config import (
    LIVE_REQUIRED_KEYS,
    SECRET_KEYS,
    Config,
    load_config,
    placeholder_keys,
    require_live_keys,
)

ROOT = Path(__file__).resolve().parent.parent

MINIMAL = dict(
    intake_chat_id=-100123,
    intake_topic_id=5,
    ops_chat_id=-100999,
    ops_topic_id=9,
)


def make(**overrides) -> Config:
    return Config(**{**MINIMAL, **overrides})


class DefaultsTests(unittest.TestCase):
    def test_reference_defaults(self):
        config = make()
        self.assertEqual(config.llm_model_hint, "deepseek-chat")
        self.assertEqual(config.llm_timeout_seconds, 70)
        self.assertIsNone(config.department_aliases)
        self.assertEqual(config.raise_poll_seconds, 60)
        self.assertEqual(config.cache_refresh_hours, 6)
        self.assertEqual(config.cache_stale_alert_hours, 24)
        self.assertEqual(config.header_date_tolerance_days, 1)
        self.assertEqual(config.retry_schedule_seconds, [30, 120, 300, 600])
        self.assertEqual(config.deadman_stale_minutes, 10)
        self.assertEqual(config.heartbeat_time, "07:00")
        self.assertEqual(config.timezone, "Asia/Manila")
        self.assertEqual(config.t_link, 0.90)
        self.assertEqual(config.t_hint, 0.50)

    def test_rollout_controls_default_off_independently(self):
        config = make()
        self.assertFalse(config.dry_run)
        self.assertFalse(config.deliverable_link_enabled)
        self.assertFalse(make(dry_run=True).deliverable_link_enabled)
        self.assertFalse(make(deliverable_link_enabled=True).dry_run)


class ValidationTests(unittest.TestCase):
    def test_bad_values_fail(self):
        bad = [
            {"timezone": "Mars/Olympus"},
            {"heartbeat_time": "7am"},
            {"heartbeat_time": "24:00"},
            {"retry_schedule_seconds": []},
            {"retry_schedule_seconds": [30, 0]},
            {"t_link": 1.5},
            {"t_hint": 0.95, "t_link": 0.90},
            {"fuzzy_min_score": 90},  # removed with the Contact resolver (#144)
            {"fuzzy_min_lead": 8},
            {"honorifics": ["ate"]},
            {"llm_timeout_seconds": 60},  # the API waits up to 60 s for the provider
            {"job_orders_data_source_id": "ds-1"},  # removed with the Notion client (#143)
            {"contacts_data_source_id": "ds-1"},
            {"projects_data_source_id": "ds-1"},
            {"deliverables_data_source_id": "ds-1"},
            {"notion_version": "2026-03-11"},
            {"deepseek_base_url": "x"},  # removed with the DeepSeek client
            {"jev_url": "https://x.example"},  # removed with the Jev client (#147)
            {"jev_model": "x"},
            {"jev_timeout_seconds": 30},
            {"deepseek_model": "x"},
            {"deepseek_timeout_seconds": 60},
            {"cache_refresh_hours": 0},
            {"cache_refresh_hours": 30, "cache_stale_alert_hours": 24},
            {"owner_telegram_handle": "has space"},
            {"captured_reaction": " "},
            {"flagged_reaction": " "},
            {"raise_poll_seconds": -1},
            {"unknown_key": 1},
        ]
        for overrides in bad:
            with self.subTest(overrides=overrides), self.assertRaises(ValidationError):
                make(**overrides)

    def test_an_old_department_alias_map_is_accepted_and_ignored(self):
        config = make(department_aliases={"TO": "OE"})
        self.assertIn("department_aliases", config.model_fields_set)
        self.assertNotIn("department_aliases", make().model_fields_set)

    def test_handles_are_normalized_with_at_sign(self):
        config = make(owner_telegram_handle="cyril_owner", standing_cc_telegram_handle="@priya")
        self.assertEqual(config.owner_telegram_handle, "@cyril_owner")
        self.assertEqual(config.standing_cc_telegram_handle, "@priya")

    def test_live_keys_are_required_at_boot(self):
        with self.assertRaises(ValueError) as ctx:
            require_live_keys(make())
        for key in LIVE_REQUIRED_KEYS:
            self.assertIn(key, str(ctx.exception))


LIVE = dict(
    standing_cc_telegram_handle="@priya",
    owner_telegram_handle="@cyril_owner",
)


def live(**overrides) -> Config:
    return make(**{**LIVE, **overrides})


class ProductionReadinessTests(unittest.TestCase):
    def test_valid_configuration_boots(self):
        require_live_keys(live())

    def test_negative_group_chat_ids_are_legitimate(self):
        require_live_keys(live(intake_chat_id=-1001234567890, ops_chat_id=-4242))

    def test_audited_invalid_values_are_rejected(self):
        bad = {
            "standing_cc_telegram_handle": [""],
            "intake_topic_id": [-5],
            "ops_topic_id": [-1],
            "llm_model_hint": ["", "  "],
            "ledger_path": ["", " "],
        }
        for key, values in bad.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError) as ctx:
                        require_live_keys(live(**{key: value}))
                    self.assertIn(key, str(ctx.exception))

    def test_all_problems_are_reported_without_values(self):
        with self.assertRaises(ValueError) as ctx:
            require_live_keys(live(intake_topic_id=-7, ledger_path="  "))
        message = str(ctx.exception)
        self.assertIn("intake_topic_id", message)
        self.assertIn("ledger_path", message)

    def test_placeholders_are_distinct_from_readiness(self):
        example = yaml.safe_load((ROOT / "config.example.yaml").read_text(encoding="utf-8"))
        config = Config(**example)
        require_live_keys(config)  # structurally valid...
        self.assertEqual(  # ...yet still not production-ready
            set(placeholder_keys(config)),
            {
                "intake_chat_id",
                "ops_chat_id",
                "standing_cc_telegram_handle",
                "owner_telegram_handle",
            },
        )
        self.assertEqual(placeholder_keys(live()), [])


class RemovedKeyTests(unittest.TestCase):
    OLD_EXAMPLE = ROOT / "tests" / "fixtures" / "config.origin-main.yaml"

    def test_the_config_example_before_stage_2_still_loads_and_names_each_removed_key(self):
        with self.assertLogs("jo_watchdog", level="WARNING") as logs:
            config = load_config(self.OLD_EXAMPLE)
        text = "\n".join(logs.output)
        for key in ("job_orders_data_source_id", "notion_version", "deepseek_model", "jev_url",
                    "fuzzy_min_score", "honorifics", "standing_cc_contact_page_id"):
            self.assertIn(key, text)
        self.assertEqual(len(logs.records), 1)  # one line for all the removed keys
        self.assertEqual(config.intake_topic_id, 2)

    def test_the_warning_holds_key_names_never_values(self):
        data = yaml.safe_load(self.OLD_EXAMPLE.read_text(encoding="utf-8"))
        data["jev_url"] = "https://secret.invalid/path"
        path = Path(tempfile.mkdtemp()) / "c.yaml"
        self.addCleanup(path.unlink)
        path.write_text(yaml.safe_dump(data))
        with self.assertLogs("jo_watchdog", level="WARNING") as logs:
            load_config(path)
        self.assertNotIn("secret.invalid", "\n".join(logs.output))

    def test_a_key_that_was_never_a_config_key_is_still_refused(self):
        data = yaml.safe_load(self.OLD_EXAMPLE.read_text(encoding="utf-8"))
        data["never_a_key"] = 1
        path = Path(tempfile.mkdtemp()) / "c.yaml"
        self.addCleanup(path.unlink)
        path.write_text(yaml.safe_dump(data))
        with self.assertRaises(ValidationError):
            load_config(path)


class ExampleFilesTests(unittest.TestCase):
    def test_config_example_lists_every_key_and_boots(self):
        data = yaml.safe_load((ROOT / "config.example.yaml").read_text(encoding="utf-8"))
        # department_aliases is accepted and ignored (#151), so the example leaves it out.
        self.assertEqual(set(data), set(Config.model_fields) - {"department_aliases"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
            require_live_keys(load_config(path))

    def test_config_example_topics_are_not_the_general_topic(self):
        data = yaml.safe_load((ROOT / "config.example.yaml").read_text(encoding="utf-8"))
        self.assertNotEqual(data["intake_topic_id"], 1)
        self.assertNotEqual(data["ops_topic_id"], 1)

    def test_env_example_lists_every_secret(self):
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        names = set(re.findall(r"^([A-Z0-9_]+)=", text, re.MULTILINE))
        self.assertEqual(names, set(SECRET_KEYS))


if __name__ == "__main__":
    unittest.main()


class RenamedKeysTests(unittest.TestCase):
    def test_old_standing_cc_keys_still_load_with_a_deprecation_warning(self):
        import warnings

        example = (ROOT / "config.example.yaml").read_text(encoding="utf-8")
        kept = "".join(
            line for line in example.splitlines(keepends=True)
            if not line.startswith("standing_cc_")
        )
        legacy = (
            "leizette_contact_page_id: 'c0000000-0000-4000-8000-0000000000cc'\n"
            "leizette_telegram_handle: '@priya'\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(kept + legacy, encoding="utf-8")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                config = load_config(path)
        self.assertFalse(hasattr(config, "standing_cc_contact_page_id"))  # removed: ignored
        self.assertEqual(config.standing_cc_telegram_handle, "@priya")
        self.assertEqual(len([w for w in caught if issubclass(w.category, DeprecationWarning)]), 1)
