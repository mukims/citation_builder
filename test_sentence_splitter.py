import unittest
from agent5_batch_citer import split_into_sentences

class TestSentenceSplitter(unittest.TestCase):
    def test_standard_boundaries(self):
        """Test sentence splitting with standard punctuation (periods, question marks, exclamation marks)."""
        text = "This is the first sentence. This is the second sentence! And the third one?"
        expected = [
            "This is the first sentence.",
            "This is the second sentence!",
            "And the third one?"
        ]
        self.assertEqual(split_into_sentences(text), expected)

    def test_scientific_abbreviations_casing(self):
        """Test that common scientific abbreviations (capitalized and lowercase) do not cause incorrect splits."""
        # Test Fig. / fig.
        text = "See Fig. 1 for details. This is the next sentence. Also check fig. 2."
        expected = [
            "See Fig. 1 for details.",
            "This is the next sentence.",
            "Also check fig. 2."
        ]
        self.assertEqual(split_into_sentences(text), expected)

        # Test Eq. / eq. / Eqs. / eqs.
        text = "Using Eq. 3, we obtain the limit. Compare with eq. 4 and eqs. 5-6."
        expected = [
            "Using Eq. 3, we obtain the limit.",
            "Compare with eq. 4 and eqs. 5-6."
        ]
        self.assertEqual(split_into_sentences(text), expected)

        # Test et al. / ET AL.
        text = "This was shown by Smith et al. in 2020. Another study by Jones et al. found otherwise."
        expected = [
            "This was shown by Smith et al. in 2020.",
            "Another study by Jones et al. found otherwise."
        ]
        self.assertEqual(split_into_sentences(text), expected)

    def test_multi_period_abbreviations(self):
        """Test abbreviations with multiple periods like i.e. and e.g."""
        text = "This is e.g. a classic example. That is i.e. the only explanation."
        expected = [
            "This is e.g. a classic example.",
            "That is i.e. the only explanation."
        ]
        self.assertEqual(split_into_sentences(text), expected)

    def test_academic_titles_and_common_abbrevs(self):
        """Test various other standard titles and abbreviations like Dr., Prof., approx., vs."""
        text = "Dr. Smith and Prof. Jones analyzed the sample vs. the control group. The result was approx. 95% accurate."
        expected = [
            "Dr. Smith and Prof. Jones analyzed the sample vs. the control group.",
            "The result was approx. 95% accurate."
        ]
        self.assertEqual(split_into_sentences(text), expected)

    def test_nested_punctuation(self):
        """Test correct handling of abbreviations nested inside parentheses or followed by punctuation."""
        text = "We studied the system (e.g. Fig. 1). This is a new sentence."
        expected = [
            "We studied the system (e.g. Fig. 1).",
            "This is a new sentence."
        ]
        self.assertEqual(split_into_sentences(text), expected)

    def test_consecutive_abbreviations(self):
        """Test consecutive abbreviations to ensure no weird interactions or incorrect splits."""
        text = "Compare Dr. Smith approx. vs. Prof. Jones."
        expected = [
            "Compare Dr. Smith approx. vs. Prof. Jones."
        ]
        self.assertEqual(split_into_sentences(text), expected)

if __name__ == "__main__":
    unittest.main()
