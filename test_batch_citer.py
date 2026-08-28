import json
import os
import tempfile
import types
import unittest
from unittest.mock import patch

import agent5_batch_citer as bc
from agent5_batch_citer import _batch_needs_citation, _cite_keys, check_citation_need


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


class TestCitationNeedBatching(unittest.TestCase):
    """One request for a whole draft makes the run hostage to a single reply.

    Batching is about blast radius: an unparseable answer should cost the
    sentences in that batch, not the document.
    """

    @staticmethod
    def _sentences(n):
        return [f"Claim number {i}." for i in range(n)]

    def test_splits_into_batches_of_the_configured_size(self):
        seen = []

        def fake(batch):
            seen.append(len(batch))
            return [True] * len(batch)

        with patch.object(bc, "_batch_needs_citation", side_effect=fake):
            check_citation_need(self._sentences(50), batch_size=20)

        self.assertEqual(seen, [20, 20, 10])

    def test_verdicts_stay_aligned_across_batch_boundaries(self):
        """Batch 2's answers must land on batch 2's sentences."""
        def fake(batch):
            return [s.endswith("1.") for s in batch]

        sentences = self._sentences(5)
        with patch.object(bc, "_batch_needs_citation", side_effect=fake):
            verdicts, failed = check_citation_need(sentences, batch_size=2)

        self.assertEqual(verdicts, [False, True, False, False, False])
        self.assertEqual(failed, 0)

    def test_a_failing_batch_does_not_lose_the_others(self):
        """The whole point: one bad reply costs one batch."""
        def fake(batch):
            if batch[0] == "Claim number 2.":
                raise ValueError("unparseable reply")
            return [True] * len(batch)

        with patch.object(bc, "_batch_needs_citation", side_effect=fake):
            verdicts, failed = check_citation_need(self._sentences(6), batch_size=2)

        self.assertEqual(verdicts, [True, True, None, None, True, True])
        self.assertEqual(failed, 1)

    def test_failed_sentences_are_none_not_false(self):
        """None means 'not judged'. False would claim no citation was needed —
        a claim nobody actually made, and one that hides the failure."""
        with patch.object(bc, "_batch_needs_citation", side_effect=ValueError("boom")):
            verdicts, failed = check_citation_need(self._sentences(3), batch_size=10)

        self.assertEqual(verdicts, [None, None, None])
        self.assertNotIn(False, verdicts)
        self.assertEqual(failed, 1)

    def test_exact_multiple_of_batch_size(self):
        seen = []
        with patch.object(bc, "_batch_needs_citation",
                          side_effect=lambda b: seen.append(len(b)) or [True] * len(b)):
            check_citation_need(self._sentences(40), batch_size=20)
        self.assertEqual(seen, [20, 20])

    def test_fewer_sentences_than_one_batch(self):
        with patch.object(bc, "_batch_needs_citation", return_value=[True, False]):
            verdicts, failed = check_citation_need(self._sentences(2), batch_size=20)
        self.assertEqual(verdicts, [True, False])
        self.assertEqual(failed, 0)

    def test_empty_input_makes_no_calls(self):
        with patch.object(bc, "_batch_needs_citation") as m:
            verdicts, failed = check_citation_need([], batch_size=20)
        m.assert_not_called()
        self.assertEqual((verdicts, failed), ([], 0))

    def test_default_batch_size_comes_from_config(self):
        seen = []
        with patch.object(bc, "CITATION_CHECK_BATCH_SIZE", 5):
            with patch.object(bc, "_batch_needs_citation",
                              side_effect=lambda b: seen.append(len(b)) or [True] * len(b)):
                check_citation_need(self._sentences(12))
        self.assertEqual(seen, [5, 5, 2])


class TestPartialFailureEndToEnd(unittest.TestCase):
    """A batch failing mid-draft must still produce output for the rest."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.draft = os.path.join(self.tmp.name, "draft.txt")
        self.out = os.path.join(self.tmp.name, "draft_cited.txt")
        with open(self.draft, "w") as f:
            f.write(" ".join(f"Scientific claim number {i} about graphene." for i in range(6)))

        # Batch size 2 over 6 sentences gives three batches, so a single
        # failure is genuinely partial rather than the whole draft.
        patcher = patch.object(bc, "CITATION_CHECK_BATCH_SIZE", 2)
        patcher.start()
        self.addCleanup(patcher.stop)

        for name, repl in [
            ("load_search_resources", lambda: (None, None, ["chunk"], [{"citation_source": "Src A"}])),
            ("hybrid_search", lambda *a, **k: [
                {"text": "ctx", "metadata": {"citation_source": "Src A"}, "rrf_score": 0.1}]),
            ("_cite_sentence_with_reasoning",
             lambda sentence, ctx: (sentence.rstrip(".") + " \\cite{cite_1}.", "because")),
        ]:
            patcher = patch.object(bc, name, repl)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_draft_is_written_despite_a_failed_batch(self):
        def fake(batch):
            if "number 2" in batch[0]:
                raise ValueError("unparseable")
            return [True] * len(batch)

        with patch.object(bc, "_batch_needs_citation", side_effect=fake):
            written = bc.run_batch_citer(self.draft, self.out)

        self.assertEqual(written, self.out)
        self.assertTrue(os.path.exists(self.out))

    def test_unjudged_sentences_are_left_verbatim(self):
        def fake(batch):
            if "number 2" in batch[0]:
                raise ValueError("unparseable")
            return [True] * len(batch)

        with patch.object(bc, "_batch_needs_citation", side_effect=fake):
            bc.run_batch_citer(self.draft, self.out)

        with open(self.out) as f:
            text = f.read()
        self.assertIn("Scientific claim number 2 about graphene.", text)
        self.assertIn("\\cite{cite_1}", text)

    def test_report_counts_unjudged_separately_from_not_needed(self):
        """The distinction the report exists to make."""
        def fake(batch):
            if "number 2" in batch[0]:
                raise ValueError("unparseable")
            return [True] * len(batch)

        with patch.object(bc, "_batch_needs_citation", side_effect=fake):
            bc.run_batch_citer(self.draft, self.out)

        with open(self.out.replace(".txt", "_report.md")) as f:
            report = f.read()
        self.assertIn("Not judged (check failed)", report)
        self.assertIn("| **Not judged (check failed)** | **2** |", report)

    def test_total_failure_writes_nothing(self):
        """With no verdicts at all there is no meaningful draft to produce."""
        with patch.object(bc, "_batch_needs_citation", side_effect=ValueError("boom")):
            written = bc.run_batch_citer(self.draft, self.out)

        self.assertIsNone(written)
        self.assertFalse(os.path.exists(self.out))


if __name__ == "__main__":
    unittest.main()
