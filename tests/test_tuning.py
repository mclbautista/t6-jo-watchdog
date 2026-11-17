"""Offline tests for the link and hint threshold scoring and the retained results (#61)."""

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from jo_watchdog.app.config import Config
from jo_watchdog.app.tuning import (
    MatchSample,
    build_report,
    match_samples,
    read_report,
    sweep_match,
    write_report,
)

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures"
RETAINED = ROOT / "docs" / "tuning" / "threshold-results.json"
PAIRS = json.loads((FIXTURES / "jev" / "regression_pairs.json").read_text(encoding="utf-8"))
DEFAULTS = Config.model_fields


def current() -> dict:
    return {
        "t_link": DEFAULTS["t_link"].default,
        "t_hint": DEFAULTS["t_hint"].default,
    }


def report(**overrides) -> dict:
    args = dict(samples=match_samples(PAIRS), current=current(), sample_source="test")
    return build_report(**{**args, **overrides})


class MatchSweepTests(unittest.TestCase):
    def test_recorded_pairs_give_samples_with_expected_verdicts(self):
        samples = match_samples(PAIRS)
        self.assertEqual([s.expected for s in samples], ["hint", "link", "link"])

    def test_default_thresholds_make_no_false_links_on_the_recorded_pairs(self):
        rows = {(r["t_link"], r["t_hint"]): r for r in sweep_match(match_samples(PAIRS))}
        self.assertEqual(rows[(0.90, 0.50)]["false_links"], 0)
        self.assertEqual(rows[(0.90, 0.50)]["missed_links"], 0)

    def test_a_loose_link_threshold_shows_its_false_link(self):
        sample = MatchSample("maybe-case", 0.75, 0.2, 0.05, expected="hint")
        rows = {(r["t_link"], r["t_hint"]): r for r in sweep_match([sample])}
        self.assertEqual(rows[(0.70, 0.50)]["false_links"], 1)
        self.assertEqual(rows[(0.90, 0.50)]["false_links"], 0)


class RetainedResultsTests(unittest.TestCase):
    def test_report_never_enables_linking_or_changes_thresholds(self):
        r = report()
        self.assertFalse(r["link_enabled"])
        self.assertEqual(r["adopted"], current())
        self.assertFalse(DEFAULTS["deliverable_link_enabled"].default)

    def test_new_results_start_with_a_pending_owner_review(self):
        self.assertEqual(report()["owner_review"]["status"], "pending")

    def test_owner_review_survives_regeneration_only_for_identical_results(self):
        first = report()
        first["owner_review"] = {
            "status": "reviewed", "reviewed_by": "Cyril", "reviewed_at": "2026-10-25T00:00:00Z",
            "evidence": "kept defaults",
        }  # fmt: skip
        self.assertEqual(report(previous=first)["owner_review"]["status"], "reviewed")
        changed = report(previous=first, samples=[MatchSample("x", 0.75, 0.2, 0.05, "hint")])
        self.assertEqual(changed["owner_review"]["status"], "pending")

    def test_round_trip_through_the_results_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "results.json"
            r = report()
            write_report(path, r)
            self.assertEqual(read_report(path), r)
            self.assertIsNone(read_report(Path(tmp) / "absent.json"))

    def test_retained_results_are_current_and_match_config_defaults(self):
        retained = read_report(RETAINED)
        self.assertIsNotNone(retained, f"{RETAINED} missing; run smoke.py --only threshold_tuning")
        self.assertEqual(retained["adopted"], current())
        self.assertFalse(retained["link_enabled"])
        regenerated = report(
            previous=retained, sample_source=retained["results"]["match_sample_source"]
        )
        self.assertEqual(retained["results_digest"], regenerated["results_digest"])

    def test_example_config_carries_the_adopted_thresholds(self):
        example = yaml.safe_load((ROOT / "config.example.yaml").read_text(encoding="utf-8"))
        self.assertEqual({k: example[k] for k in current()}, current())
        self.assertFalse(example.get("deliverable_link_enabled", False))
