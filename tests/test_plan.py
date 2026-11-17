"""Unit tests for the pure core: the block splitter."""

import unittest

from jo_watchdog.core.plan import split_blocks


class SplitBlocksTests(unittest.TestCase):
    def test_short_text_is_one_block(self):
        self.assertEqual(split_blocks("hello", limit=2000), ["hello"])

    def test_text_over_limit_is_split(self):
        text = "a" * 2500
        blocks = split_blocks(text, limit=2000)
        self.assertEqual(len(blocks), 2)
        self.assertEqual("".join(blocks), text)
        self.assertTrue(all(len(b) <= 2000 for b in blocks))

    def test_split_prefers_newline_boundaries(self):
        # 1200 + 1200 chars joined by newline; limit 2000 should keep lines intact
        text = "x" * 1200 + "\n" + "y" * 1200
        blocks = split_blocks(text, limit=2000)
        self.assertEqual(blocks, ["x" * 1200, "y" * 1200])

    def test_long_single_line_hard_splits(self):
        text = "z" * 4500  # no newlines
        blocks = split_blocks(text, limit=2000)
        self.assertEqual(len(blocks), 3)
        self.assertTrue(all(len(b) <= 2000 for b in blocks))
        self.assertEqual("".join(blocks), text)


if __name__ == "__main__":
    unittest.main()
