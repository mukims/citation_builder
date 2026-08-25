import types
import unittest
from unittest.mock import patch

from agent5_batch_citer import _batch_needs_citation, _cite_keys


def _reply(text):
    """Wrap *text* in the object shape ollama.chat returns."""
    return types.SimpleNamespace(message=types.SimpleNamespace(content=text))


class TestCiteKeyExtraction(unittest.TestCase):
    def test_no_citation(self):
        """A sentence the model declined to cite yields no keys."""
        self.assertEqual(_cite_keys("Disorder is ubiquitous in graphene systems."), set())

    def test_single_and_multi_key(self):
        """Both \\cite{a} and the comma-separated \\cite{a,b} form are recognised."""
        self.assertEqual(_cite_keys("Ballistic transport occurs \\cite{cite_2}."), {"cite_2"})
        self.assertEqual(
            _cite_keys("This is well established \\cite{cite_1, cite_4}."),
            {"cite_1", "cite_4"},
        )

    def test_multiple_citations_in_one_sentence(self):
        """Separate \\cite commands in the same sentence are all collected."""
        text = "Shown in graphene \\cite{cite_1} and in MoS2 \\cite{cite_9}."
        self.assertEqual(_cite_keys(text), {"cite_1", "cite_9"})

    def test_empty_braces_ignored(self):
        """A malformed \\cite{} contributes no key rather than an empty one."""
        self.assertEqual(_cite_keys("A claim \\cite{}."), set())


class TestCitationNeedParsing(unittest.TestCase):
    """Verdicts must bind to the sentence the model numbered, not to line order.

    The earlier parser appended one boolean per line of the response, so any
    preamble or blank line shifted every verdict onto the wrong sentence.
    """

    SENTENCES = ["First claim.", "Second claim.", "Third claim."]
    EXPECTED = [True, False, True]

    def _run(self, response):
        with patch("agent5_batch_citer.ollama.chat", return_value=_reply(response)):
            return _batch_needs_citation(self.SENTENCES)

    def test_clean_list(self):
        """The well-formed case the prompt asks for."""
        self.assertEqual(self._run("1. YES\n2. NO\n3. YES"), self.EXPECTED)

    def test_preamble_line(self):
        """A conversational opener must not shift the verdicts by one."""
        self.assertEqual(
            self._run("Here are the answers:\n1. YES\n2. NO\n3. YES"), self.EXPECTED
        )

    def test_blank_lines_between_entries(self):
        """A double-spaced list must not interleave phantom False verdicts."""
        self.assertEqual(self._run("1. YES\n\n2. NO\n\n3. YES"), self.EXPECTED)

    def test_trailing_commentary(self):
        """Text after the list is ignored rather than parsed as a verdict."""
        self.assertEqual(
            self._run("1. YES\n2. NO\n3. YES\n\nLet me know if you need more."),
            self.EXPECTED,
        )

    def test_paren_and_lowercase_forms(self):
        """`1) yes` is accepted alongside `1. YES`."""
        self.assertEqual(self._run("1) yes\n2) no\n3) yes"), self.EXPECTED)

    def test_out_of_order_response(self):
        """Verdicts are placed by their number, not by the order received."""
        self.assertEqual(self._run("2. NO\n1. YES\n3. YES"), self.EXPECTED)

    def test_truncated_response_raises(self):
        """A short list must fail loudly rather than be padded with False."""
        with patch("shared.retry.time.sleep"):
            with self.assertRaises(ValueError) as ctx:
                self._run("1. YES\n2. NO")
        self.assertIn("3", str(ctx.exception))

    def test_unparseable_response_raises(self):
        """A response with no verdicts at all raises instead of returning all-False."""
        with patch("shared.retry.time.sleep"):
            with self.assertRaises(ValueError):
                self._run("I cannot determine this without more context.")


if __name__ == "__main__":
    unittest.main()
