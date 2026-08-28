"""Tests for Agent 2's paper naming.

Filenames are how the fetcher decides whether it already has a paper. They used
to embed the citation's position in the fetch queue, which is recomputed every
run, so the same work downloaded under a new name each time its reference string
was spelled differently — 21 of the 167 PDFs in the working corpus were
byte-identical duplicates of another file.
"""

import unittest

import requests

from agent2_fetcher import is_transient, paper_filename


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


class TestTransientClassification(unittest.TestCase):
    """Anything recorded in failed_downloads.json is skipped by every later run.

    So the question is not "did this attempt fail" but "does this failure tell
    us the paper is unavailable". A rate limit does not.
    """

    @staticmethod
    def _http_error(status):
        response = requests.Response()
        response.status_code = status
        return requests.HTTPError(f"{status} Client Error", response=response)

    def test_rate_limit_is_transient(self):
        """The case seen in practice: Crossref 429s on the anonymous pool."""
        self.assertTrue(is_transient(self._http_error(429)))

    def test_server_errors_are_transient(self):
        for status in (500, 502, 503, 504):
            with self.subTest(status=status):
                self.assertTrue(is_transient(self._http_error(status)))

    def test_timeouts_and_connection_drops_are_transient(self):
        self.assertTrue(is_transient(requests.Timeout("timed out")))
        self.assertTrue(is_transient(requests.ConnectionError("reset")))

    def test_not_found_is_permanent(self):
        """404 is a real answer about the paper — record it and move on."""
        self.assertFalse(is_transient(self._http_error(404)))

    def test_forbidden_is_permanent(self):
        """403 means the publisher refuses us, which retrying will not change."""
        self.assertFalse(is_transient(self._http_error(403)))

    def test_error_without_a_response_is_permanent(self):
        """A parse error or similar carries no HTTP status to judge by."""
        self.assertFalse(is_transient(ValueError("malformed record")))


if __name__ == "__main__":
    unittest.main()
