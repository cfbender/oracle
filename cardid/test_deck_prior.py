"""The deck-prior sweep replays the app's re-rank and clear-answer rule faithfully."""

from __future__ import annotations

import unittest

import numpy as np

from .deck_prior import rerank, suggest, sweep

NAMES = ["Sol Ring", "Mind Stone", "Arcane Signet", "Fellwar Stone", "Thought Vessel", "Sol Ring"]


def queries(rows: list[tuple[list[int], list[float], int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.array([r[0] for r in rows])
    sims = np.array([r[1] for r in rows], dtype=float)
    targets = np.array([r[2] for r in rows])
    return idx, sims, targets


class RerankTest(unittest.TestCase):
    def test_adds_prior_to_listed_candidates_and_keeps_ties_in_order(self):
        order, scores = rerank(np.array([0.72, 0.70, 0.5]), np.array([0.0, 1.0, 0.0]), 0.03)
        self.assertEqual(order.tolist(), [1, 0, 2])
        np.testing.assert_allclose(scores, [0.73, 0.72, 0.5])
        order, _ = rerank(np.array([0.7, 0.7]), np.array([1.0, 1.0]), 0.03)
        self.assertEqual(order.tolist(), [0, 1])


class SweepTest(unittest.TestCase):
    def test_near_tie_in_list_becomes_a_clear_right_answer(self):
        # Truth (Sol Ring) leads by 0.05: not clear alone, clear with a 0.03 prior.
        idx, sims, targets = queries([([0, 1, 2], [0.75, 0.70, 0.60], 0)])
        rows = {row["prior"]: row for row in sweep(idx, sims, targets, NAMES, priors=(0.0, 0.03), deck_size=1)}
        self.assertEqual(rows[0.0]["in_list"]["auto_rate"], 0.0)
        self.assertEqual(rows[0.03]["in_list"]["auto_rate"], 1.0)
        self.assertEqual(rows[0.03]["in_list"]["auto_precision"], 1.0)

    def test_off_list_lookalike_can_steal_a_close_answer(self):
        # Truth leads its lookalike by 0.02; off-list, a 0.03 prior on the lookalike flips it.
        idx, sims, targets = queries([([0, 1, 2], [0.72, 0.70, 0.50], 0)])
        rows = {row["prior"]: row for row in sweep(idx, sims, targets, NAMES, priors=(0.0, 0.03), deck_size=1)}
        self.assertEqual(rows[0.0]["off_list"]["top1"], 1.0)
        self.assertEqual(rows[0.03]["off_list"]["top1"], 0.0)
        self.assertEqual(rows[0.03]["in_list"]["top1"], 1.0)

    def test_scores_by_name_so_another_printing_of_the_truth_is_right(self):
        # Gallery 5 is another Sol Ring art: top-1 by name is right even though the art differs.
        idx, sims, targets = queries([([5, 1], [0.9, 0.5], 0)])
        row = sweep(idx, sims, targets, NAMES, priors=(0.0,), deck_size=1)[0]
        self.assertEqual(row["in_list"]["top1"], 1.0)

    def test_mixes_scenarios_by_off_list_share(self):
        idx, sims, targets = queries([([0, 1, 2], [0.72, 0.70, 0.50], 0)])
        row = sweep(idx, sims, targets, NAMES, priors=(0.03,), deck_size=1, off_list_share=0.25)[0]
        self.assertAlmostEqual(row["mixed"]["top1"], 0.75)

    def test_suggest_rejects_priors_that_cost_auto_precision_or_off_list_mistakes(self):
        def row(prior, top1, precision, off_list_wrong):
            return {"prior": prior, "mixed": {"top1": top1, "auto_precision": precision}, "off_list": {"silent_wrong": off_list_wrong}}

        rows = [
            row(0.0, 0.90, 0.99, 0.002),
            row(0.03, 0.93, 0.988, 0.008),
            row(0.05, 0.94, 0.99, 0.03),  # mixed looks fine, but stolen cards get recorded wrong
            row(0.06, 0.95, 0.97, 0.01),  # silent mistakes overall
        ]
        self.assertEqual(suggest(rows)["prior"], 0.03)

    def test_counts_silent_mistakes(self):
        # Off-list, the lookalike already leads by 0.06; a 0.03 prior makes that a clear wrong answer.
        idx, sims, targets = queries([([1, 0, 2], [0.76, 0.70, 0.50], 0)])
        rows = {row["prior"]: row for row in sweep(idx, sims, targets, NAMES, priors=(0.0, 0.03), deck_size=1)}
        self.assertEqual(rows[0.0]["off_list"]["silent_wrong"], 0.0)
        self.assertEqual(rows[0.03]["off_list"]["silent_wrong"], 1.0)
        self.assertEqual(rows[0.03]["in_list"]["silent_wrong"], 0.0)


if __name__ == "__main__":
    unittest.main()
