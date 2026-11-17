import unittest

from pydantic import BaseModel, ValidationError

from jo_watchdog.core.safe_error import describe


class Row(BaseModel):
    verdict: int
    nested: dict[str, int]


class SafeErrorTests(unittest.TestCase):
    def test_a_validation_error_keeps_class_count_and_paths_and_no_value(self):
        with self.assertRaises(ValidationError) as caught:
            Row.model_validate({"verdict": "SECRET text", "nested": {"a": "MORE text"}})
        text = describe(caught.exception)
        self.assertNotIn("SECRET", text)
        self.assertNotIn("MORE", text)
        self.assertIn("2 errors", text)
        self.assertIn("verdict", text)
        self.assertIn("nested.a", text)

    def test_another_error_keeps_class_and_message(self):
        self.assertEqual(describe(KeyError("k")), "KeyError: 'k'")
