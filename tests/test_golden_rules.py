"""Golden tests for the rules imported from t6 (#135).

The fixtures in ``tests/golden`` hold what the watchdog's own masker, header
parser and deadline resolver returned before the import, for invented inputs.
The imported rules must return the same, so the swap changes nothing.
"""

import json
import unittest
from datetime import datetime, time
from pathlib import Path

from jo_watchdog.core.deadline import resolve_deadline
from jo_watchdog.core.header import (
    canonical_name,
    normalize_department,
    parse_header,
    parse_number,
)
from jo_watchdog.core.masker import outbound_text

GOLDEN = Path(__file__).resolve().parent / "golden"


def load(name: str):
    return json.loads((GOLDEN / name).read_text(encoding="utf-8"))


class MaskerGoldenTests(unittest.TestCase):
    def test_masker_output_is_unchanged(self):
        rows = load("masker_golden.json")
        self.assertGreaterEqual(len(rows), 1000)
        for row in rows:
            result = outbound_text(row["in"])
            self.assertEqual(
                (result.text, result.withheld_count, result.masker_version),
                (row["text"], row["count"], row["version"]),
                row["in"],
            )


class HeaderGoldenTests(unittest.TestCase):
    def test_header_rules_are_unchanged(self):
        data = load("header_golden.json")
        aliases = data["aliases"]
        for row in data["rows"]:
            header = parse_header(row["header"], aliases[row["aliases"]])
            parsed = (
                None
                if header is None
                else {
                    "project": header.project,
                    "department": header.department,
                    "number": header.number,
                }
            )
            label = (row["header"], row["aliases"])
            self.assertEqual(parsed, row["parsed"], label)
            self.assertEqual(canonical_name(header, "fallback"), row["name"], label)
            self.assertEqual(canonical_name(header, "fallback", "TOC"), row["name_dept"], label)
        for token, index, expected in data["departments"]:
            self.assertEqual(normalize_department(token, aliases[index]), expected, token)
        for text, expected in data["numbers"]:
            self.assertEqual(parse_number(text), expected, text)


class DeadlineGoldenTests(unittest.TestCase):
    def test_deadline_rule_is_unchanged(self):
        data = load("deadline_golden.json")
        posted = [datetime.fromisoformat(p) for p in data["posted"]]
        shifts = [
            [(name, time.fromisoformat(v["time"]) if isinstance(v, dict) else v) for name, v in s]
            for s in data["shifts"]
        ]
        for row in data["rows"]:
            result = resolve_deadline(row["phrase"], posted[row["posted"]], shifts[row["shifts"]])
            got = {
                "deadline": result.deadline.isoformat() if result.deadline else None,
                "source": result.source,
                "findings": [{"code": f.code, "params": f.params} for f in result.findings],
            }
            want = {k: row[k] for k in ("deadline", "source", "findings")}
            self.assertEqual(got, want, (row["phrase"], row["posted"], row["shifts"]))


if __name__ == "__main__":
    unittest.main()
