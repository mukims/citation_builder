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
from agent1_extractor import _match_citation_line, _reference_section


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


class TestCitationLinePatterns(unittest.TestCase):
    def test_bracket_marker_with_text_on_the_same_line(self):
        self.assertEqual(
            _match_citation_line("[3] C. Liu, Graphene-based supercapacitor, Nano Lett. 10 (2010)"),
            (3, "C. Liu, Graphene-based supercapacitor, Nano Lett. 10 (2010)"),
        )

    def test_bracket_marker_alone_on_its_line(self):
        """Many journals put the marker on its own line, text underneath.

        The pattern used to require whitespace plus content after the bracket,
        so a reference list typeset this way matched nothing at all — while the
        numbered-heading pattern happily captured "1. Introduction" instead.
        """
        self.assertEqual(_match_citation_line("[1]"), (1, ""))

    def test_numbered_marker_still_matches(self):
        self.assertEqual(
            _match_citation_line("12. A. Author, Some Paper, J. Phys. (2020)"),
            (12, "A. Author, Some Paper, J. Phys. (2020)"),
        )

    def test_ordinary_prose_does_not_match(self):
        self.assertIsNone(_match_citation_line("The results in Figure 3 show that"))


class TestReferenceSectionScoping(unittest.TestCase):
    """The "N." pattern matches numbered section headings just as readily as
    citations, so it must only be applied after the reference list begins."""

    BODY = [
        "1. Introduction",
        "Storing electrical charge has been known since ancient times.",
        "2. Experimental methods",
        "MnFe2O4 nano-ferrites were prepared by hydrothermal method.",
        "References",
        "[1]",
        "B.E. Conway, Electrochemical Supercapacitors, Springer US, 2013.",
    ]

    def test_returns_only_lines_after_the_heading(self):
        self.assertEqual(_reference_section(self.BODY), self.BODY[5:])

    def test_section_headings_are_excluded(self):
        kept = _reference_section(self.BODY)
        self.assertNotIn("1. Introduction", kept)
        self.assertNotIn("2. Experimental methods", kept)

    def test_heading_match_is_case_insensitive(self):
        for heading in ("REFERENCES", "references", "Bibliography", "Works Cited"):
            with self.subTest(heading=heading):
                self.assertEqual(_reference_section(["body", heading, "[1] X"]), ["[1] X"])

    def test_numbered_heading_form_is_recognised(self):
        self.assertEqual(_reference_section(["body", "5. References", "[1] X"]), ["[1] X"])

    def test_falls_back_to_the_whole_document_when_absent(self):
        """Papers without an explicit heading keep working as before."""
        lines = ["[1] A. Author", "[2] B. Author"]
        self.assertEqual(_reference_section(lines), lines)

    def test_the_last_heading_wins(self):
        """'References' can appear in a table of contents before the real list."""
        lines = ["References", "(contents entry)", "References", "[1] Real"]
        self.assertEqual(_reference_section(lines), ["[1] Real"])


if __name__ == "__main__":
    unittest.main()
