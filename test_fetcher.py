"""Tests for Agent 2's paper naming.

Filenames are how the fetcher decides whether it already has a paper. They used
to embed the citation's position in the fetch queue, which is recomputed every
run, so the same work downloaded under a new name each time its reference string
was spelled differently — 21 of the 167 PDFs in the working corpus were
byte-identical duplicates of another file.
"""

import unittest

from agent2_fetcher import paper_filename


class TestPaperFilename(unittest.TestCase):
    def test_same_doi_gives_same_name_regardless_of_queue_position(self):
        """The property the old scheme lacked: identity, not fetch order."""
        a = paper_filename("Half-metallic graphene nanoribbons", doi="10.1038/nature05180")
        b = paper_filename("Half-metallic graphene nanoribbons", doi="10.1038/nature05180")
        self.assertEqual(a, b)

    def test_title_capitalisation_does_not_split_one_paper(self):
        """Crossref capitalisation varies; it must not create a second file."""
        self.assertEqual(
            paper_filename("Half-metallic graphene nanoribbons", doi="10.1038/nature05180"),
            paper_filename("Half-Metallic Graphene Nanoribbons", doi="10.1038/nature05180"),
        )

    def test_trailing_punctuation_does_not_split_one_paper(self):
        self.assertEqual(
            paper_filename("Graphene nanoribbons", doi="10.1/x"),
            paper_filename("Graphene nanoribbons.", doi="10.1/x"),
        )

    def test_distinct_papers_get_distinct_names(self):
        self.assertNotEqual(
            paper_filename("Paper A", doi="10.1038/nature05180"),
            paper_filename("Paper B", doi="10.1038/nature12952"),
        )

    def test_same_title_different_doi_stays_distinct(self):
        """Titles collide across papers; the DOI is what disambiguates."""
        self.assertNotEqual(
            paper_filename("Graphene", doi="10.1038/aaa"),
            paper_filename("Graphene", doi="10.1038/bbb"),
        )

    def test_doi_is_preferred_over_arxiv_id(self):
        name = paper_filename("T", doi="10.1038/nature05180", arxiv_id="1234.5678")
        self.assertIn("10_1038_nature05180", name)
        self.assertNotIn("arxiv", name)

    def test_arxiv_id_used_when_there_is_no_doi(self):
        name = paper_filename("Some Preprint", arxiv_id="1234.5678v2")
        self.assertIn("arxiv_1234_5678v2", name)

    def test_title_only_fallback_is_still_stable(self):
        """No identifier at all still beats a queue index — it repeats."""
        self.assertEqual(paper_filename("Only A Title"), paper_filename("Only A Title"))

    def test_unknown_title_does_not_produce_a_bare_extension(self):
        """Crossref returns 'Unknown' when it cannot resolve the reference."""
        name = paper_filename("Unknown", doi="10.1/x")
        self.assertTrue(name.startswith("untitled_"))
        self.assertTrue(name.endswith(".pdf"))

    def test_path_separators_and_punctuation_are_sanitised(self):
        """DOIs contain slashes and dots; neither may reach the filesystem."""
        name = paper_filename("A/B: study (part 1)", doi="10.1038/s41586-020-2649-2")
        self.assertNotIn("/", name)
        self.assertNotIn(":", name)
        self.assertTrue(name.endswith(".pdf"))

    def test_name_stays_within_a_sane_length(self):
        name = paper_filename("x" * 400, doi="y" * 400)
        self.assertLessEqual(len(name), 100)


if __name__ == "__main__":
    unittest.main()
