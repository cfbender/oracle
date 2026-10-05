"""The export-time hub penalty: CSLS r per art, its clipping, and when export skips it."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from . import hubs
from .detect import FRAME_NAMES


class HubScoresTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.embeddings = rng.normal(size=(9, 4)).astype(np.float32)
        self.frames = np.array([0, 6, 0, 1, 6, 0, 1, 0, 6])
        self.cuts = rng.normal(size=(7, len(FRAME_NAMES), 4)).astype(np.float32)
        self.owners = [[2], [], [4, 5], [], [], [0], []]

    def brute_force(self, k):
        r = []
        for art, (vec, frame) in enumerate(zip(self.embeddings, self.frames, strict=True)):
            sims = [self.cuts[card, frame] @ vec for card in range(len(self.cuts)) if art not in self.owners[card]]
            r.append(np.mean(sorted(sims)[-k:]))
        return np.array(r, np.float32)

    def test_mean_of_the_k_best_cuts_for_its_own_frame_without_its_own_card(self):
        for k in (1, 3):
            with self.subTest(k=k):
                np.testing.assert_allclose(hubs.hub_scores(self.embeddings, self.frames, self.cuts, self.owners, k), self.brute_force(k), rtol=1e-5)
        chunked = hubs.hub_scores(self.embeddings, self.frames, self.cuts, self.owners, 3, chunk=2)
        np.testing.assert_allclose(chunked, self.brute_force(3), rtol=1e-5)
        with self.assertRaises(ValueError):
            hubs.hub_scores(self.embeddings, self.frames, self.cuts, self.owners, 7)

    def test_penalty_is_the_clipped_excess_over_the_median(self):
        r = np.array([0.2, 0.3, 0.4, 0.5, 0.9], np.float32)
        np.testing.assert_allclose(hubs.hub_penalties(r, 0.25, 0.1), [0, 0, 0, 0.025, 0.1], atol=1e-7)
        self.assertTrue((hubs.hub_penalties(r) >= 0).all())  # never a bonus: scores stay <= cosine

    def test_two_part_scans_own_both_halves(self):
        arts = [{"id": "a"}, {"id": "room"}, {"id": "room-1"}, {"id": "b", "printings": [{"id": "b-reprint"}]}]
        paths = [Path(f"{name}.jpg") for name in ("room", "b-reprint", "nothing", "room-1")]
        self.assertEqual(hubs.card_owners(paths, arts), [[1, 2], [3], [], [1, 2]])


class ExportHubPenaltyTest(unittest.TestCase):
    def index(self, n=6):
        rng = np.random.default_rng(1)
        vectors = rng.normal(size=(n, 3)).astype(np.float32)
        return SimpleNamespace(
            embeddings=vectors / np.linalg.norm(vectors, axis=1, keepdims=True),
            frames=np.array([0, 0, 1, 6, 0, 1][:n]),
            arts=[{"id": f"art-{i}", "name": f"Art {i}"} for i in range(n)],
            embed=lambda crops: np.stack([crops.reshape(len(crops), -1, 3).mean(axis=1)[:, i] for i in range(3)], axis=1).astype(np.float32),
        )

    def scans(self, root: Path, n: int) -> Path:
        rng = np.random.default_rng(2)
        for i in range(n):
            cv2.imwrite(str(root / f"art-{i}.jpg"), rng.integers(0, 256, (680, 488, 3), dtype=np.uint8))
        return root

    def test_skipped_without_enough_scans_or_weight(self):
        with tempfile.TemporaryDirectory() as temp:
            cards = self.scans(Path(temp), 4)
            with redirect_stdout(io.StringIO()) as out:
                penalty, info = hubs.export_hub_penalty(self.index(), cards, min_cards=5)
            self.assertIn("no hub penalty: 4 card scans", out.getvalue())
            self.assertIsNone(info)
            self.assertFalse(penalty.any())
            penalty, info = hubs.export_hub_penalty(self.index(), cards, weight=0, min_cards=1)
            self.assertIsNone(info)
            self.assertFalse(penalty.any())

    def test_records_what_was_applied(self):
        with tempfile.TemporaryDirectory() as temp:
            cards = self.scans(Path(temp), 8)
            index = self.index()
            with redirect_stdout(io.StringIO()):
                penalty, info = hubs.export_hub_penalty(index, cards, weight=0.5, k=2, cap=0.2, min_cards=5)
            paths = hubs.background_cards(cards)
            cuts = hubs.background_cuts(index.embed, paths)
            self.assertEqual(cuts.shape, (8, len(FRAME_NAMES), 3))
            r = hubs.hub_scores(index.embeddings, index.frames, cuts, hubs.card_owners(paths, index.arts), 2)
            np.testing.assert_allclose(penalty, hubs.hub_penalties(r, 0.5, 0.2))
            self.assertEqual(
                {k: info[k] for k in ("weight", "neighbours", "cap", "cards", "penalised")},
                {"weight": 0.5, "neighbours": 2, "cap": 0.2, "cards": 8, "penalised": int((penalty > 0).sum())},
            )
            self.assertEqual(info["cards_fingerprint"], hubs.fingerprint(paths))
            self.assertEqual(len(hubs.background_cards(cards, limit=3)), 3)


if __name__ == "__main__":
    unittest.main()
