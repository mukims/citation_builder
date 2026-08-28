"""Tests for Agent 1's handling of source PDFs.

Agent 1 moves each PDF out of raw/ after reading it. Where it moves them to is
the whole record of what happened: a paper filed under processed/ is treated as
done and never looked at again.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import agent1_extractor as ex


class TestUniqueDestination(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_uses_the_plain_name_when_free(self):
        self.assertEqual(
            ex._unique_destination(self.tmp.name, "paper.pdf"),
            os.path.join(self.tmp.name, "paper.pdf"),
        )

    def test_does_not_overwrite_an_existing_file(self):
        """Two source papers can share a basename."""
        open(os.path.join(self.tmp.name, "paper.pdf"), "w").close()

        dest = ex._unique_destination(self.tmp.name, "paper.pdf")

        self.assertEqual(dest, os.path.join(self.tmp.name, "paper_2.pdf"))

    def test_keeps_counting_past_the_first_collision(self):
        for name in ("paper.pdf", "paper_2.pdf"):
            open(os.path.join(self.tmp.name, name), "w").close()

        self.assertEqual(
            ex._unique_destination(self.tmp.name, "paper.pdf"),
            os.path.join(self.tmp.name, "paper_3.pdf"),
        )


class TestRunExtractorFiling(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw = os.path.join(self.tmp.name, "raw")
        os.makedirs(self.raw)
        self.citations_path = os.path.join(self.tmp.name, "extracted_citations.json")

        for target, value in [("RAW_DIR", self.raw),
                              ("EXTRACTED_CITATIONS_PATH", self.citations_path)]:
            patcher = patch.object(ex, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _drop(self, name):
        path = os.path.join(self.raw, name)
        with open(path, "wb") as f:
            f.write(b"%PDF-1.4")
        return path

    def test_successful_pdf_goes_to_processed(self):
        self._drop("good.pdf")

        with patch.object(ex, "extract_citations", return_value=["[1] A. Author, 2020"]):
            ex.run_extractor()

        self.assertTrue(os.path.exists(os.path.join(self.raw, "processed", "good.pdf")))
        self.assertFalse(os.path.exists(os.path.join(self.raw, "good.pdf")))

    def test_pdf_yielding_nothing_goes_to_failed_not_processed(self):
        """The regression: an unreadable paper was filed as done and lost."""
        self._drop("scanned.pdf")

        with patch.object(ex, "extract_citations", return_value=[]):
            ex.run_extractor()

        self.assertTrue(os.path.exists(os.path.join(self.raw, "failed", "scanned.pdf")))
        self.assertFalse(os.path.exists(os.path.join(self.raw, "processed", "scanned.pdf")))

    def test_nothing_is_written_when_every_pdf_failed(self):
        self._drop("scanned.pdf")

        with patch.object(ex, "extract_citations", return_value=[]):
            ex.run_extractor()

        self.assertFalse(os.path.exists(self.citations_path))

    def test_a_failure_does_not_discard_a_sibling_success(self):
        self._drop("good.pdf")
        self._drop("scanned.pdf")

        def fake(path):
            return [] if "scanned" in path else ["[1] A. Author, 2020"]

        with patch.object(ex, "extract_citations", side_effect=fake):
            ex.run_extractor()

        with open(self.citations_path) as f:
            self.assertEqual(json.load(f), ["[1] A. Author, 2020"])
        self.assertTrue(os.path.exists(os.path.join(self.raw, "failed", "scanned.pdf")))
        self.assertTrue(os.path.exists(os.path.join(self.raw, "processed", "good.pdf")))

    def test_citations_are_deduplicated_across_runs(self):
        self._drop("a.pdf")
        with patch.object(ex, "extract_citations", return_value=["[1] Shared ref"]):
            ex.run_extractor()

        self._drop("b.pdf")
        with patch.object(ex, "extract_citations", return_value=["[1] Shared ref", "[2] New"]):
            ex.run_extractor()

        with open(self.citations_path) as f:
            self.assertEqual(json.load(f), ["[1] Shared ref", "[2] New"])

    def test_same_basename_twice_does_not_clobber(self):
        """Second paper named good.pdf must not replace the first."""
        self._drop("good.pdf")
        with patch.object(ex, "extract_citations", return_value=["[1] X"]):
            ex.run_extractor()

        self._drop("good.pdf")
        with patch.object(ex, "extract_citations", return_value=["[2] Y"]):
            ex.run_extractor()

        processed = sorted(os.listdir(os.path.join(self.raw, "processed")))
        self.assertEqual(processed, ["good.pdf", "good_2.pdf"])


if __name__ == "__main__":
    unittest.main()
