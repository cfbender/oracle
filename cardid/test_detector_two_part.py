"""Detector scan selection, sampling, and grouped geometry metrics (CPU, no downloads)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import httpx
import numpy as np
import torch

from . import scryfall, synth, train_detector
from .evaluate_layouts import detector_geometry
from .test_scryfall import ABRADE


def printing(card_id="room", **changes):
    return dict(
        ABRADE,
        **{
            "id": card_id,
            "layout": "split",
            "image_uris": {"normal": f"https://scan.test/{card_id}"},
            "card_faces": [{"name": "A", "type_line": "Enchantment — Room"}, {"name": "B"}],
            **changes,
        },
    )


class DetectorTwoPartTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def bank(self):
        # Deliberately unequal printing counts: group balancing is not uniform-file sampling.
        groups = {**{f"room{i}": "room" for i in range(8)}, "split": "split", "aftermath": "aftermath", "flip": "flip"}
        (self.root / "two-part.json").write_text(json.dumps(groups))
        return synth.CardBank([self.root / f"{key}.jpg" for key in ["ordinary", *groups]])

    def test_selection_keeps_every_printing_once_and_rejects_unusable_scans(self):
        cards = [
            printing("room"),
            printing("translation", lang="de"),
            printing("split", card_faces=[{"name": "A"}, {"name": "B"}]),
            printing("aftermath", card_faces=[{"name": "A"}, {"name": "B"}], keywords=["Aftermath"]),
            printing("flip", layout="flip"),
            ABRADE,
        ]
        for changes in (
            {"digital": True},
            {"games": ["arena"]},
            {"image_status": "placeholder"},
            {"image_uris": {"art_crop": "montage"}},
            {"card_faces": [{"name": "A"}] * 3},
            {"promo_types": ["playtest"]},
            {"type_line": "Card"},
        ):
            self.assertEqual(scryfall.two_part_cards([printing("bad", **changes)]), [])
        entries = scryfall.two_part_cards(cards + cards)
        self.assertEqual([e["id"] for e in entries], ["aftermath", "flip", "room", "split", "translation"])
        self.assertEqual([e["layout_group"] for e in entries], ["aftermath", "flip", "room", "split", "room"])
        self.assertEqual(entries, scryfall.two_part_cards(cards[::-1]))
        self.assertEqual(entries[2]["card_url"], "https://scan.test/room")

    def test_paginated_download_is_additive_retries_and_preserves_portrait_bytes(self):
        image = np.zeros((88, 63, 3), np.uint8)
        image[:22, :17] = [31, 90, 180]  # asymmetric, so rotations/crops cannot pass
        _, encoded = cv2.imencode(".png", image)
        calls = []
        fail = True

        def respond(request):
            calls.append(str(request.url))
            if request.url.host == "api.scryfall.com":
                self.assertEqual(request.url.params["unique"], "prints")
                self.assertIn("include:multilingual", request.url.params["q"])
                return httpx.Response(200, json={"data": [printing()], "has_more": True, "next_page": "https://pages.test/2"})
            if request.url.host == "pages.test":
                return httpx.Response(200, json={"data": [printing("flip", layout="flip")], "has_more": False})
            return httpx.Response(503 if fail and request.url.path == "/flip" else 200, content=encoded.tobytes())

        (self.root / "ordinary.jpg").write_bytes(b"existing")
        with (
            httpx.Client(transport=httpx.MockTransport(respond)) as client,
            patch.object(scryfall, "CARD_DIR", self.root),
            patch.object(scryfall.time, "sleep"),
        ):
            with self.assertRaisesRegex(SystemExit, "1 two-part downloads failed"):
                scryfall.download_two_part_cards(client)
            self.assertEqual(json.loads((self.root / "two-part.json").read_text()), {"room": "room"})
            self.assertEqual((self.root / "room.jpg").read_bytes(), encoded.tobytes())
            fail = False
            scryfall.download_two_part_cards(client)
            scryfall.download_two_part_cards(client)
        self.assertEqual(json.loads((self.root / "two-part.json").read_text()), {"room": "room", "flip": "flip"})
        self.assertEqual(calls.count("https://scan.test/room"), 1)
        self.assertEqual(calls.count("https://scan.test/flip"), 2)
        self.assertEqual((self.root / "ordinary.jpg").read_bytes(), b"existing")

    def test_sampling_balances_groups_at_ten_percent_not_file_counts(self):
        bank = self.bank()
        rng = np.random.default_rng(4)
        picked = bank.groups[[bank.sample_index(rng) for _ in range(20000)]]
        self.assertAlmostEqual(float((picked != "ordinary").mean()), 0.10, delta=0.01)
        for group in ("room", "split", "aftermath", "flip"):
            self.assertAlmostEqual(float((picked == group).mean()), 0.025, delta=0.004)
        alias = synth.CardBank([self.root / "room0-1.jpg"])
        self.assertEqual(alias.groups.tolist(), ["room"])
        self.assertEqual(alias.sample_index(rng), 0)  # no ordinary scans is still usable
        unmarked = synth.CardBank([self.root / "a.jpg", self.root / "b.jpg"])
        a, b = np.random.default_rng(8), np.random.default_rng(8)
        self.assertEqual([unmarked.sample_index(a) for _ in range(10)], [int(b.integers(2)) for _ in range(10)])

    def test_random_clicked_and_neighbour_draws_use_sampler_but_explicit_index_wins(self):
        bank = self.bank()
        image = np.full((88, 63, 3), 120, np.uint8)
        arts = synth.ArtBank([])
        with patch.object(bank, "sample_index", return_value=3) as pick, patch.object(bank, "load", return_value=image) as load:
            _, quad = synth.render_scene(np.random.default_rng(5), bank, arts)
            self.assertGreater(pick.call_count, 1)  # clicked card plus neighbours/occluders
            self.assertTrue(all(call.args[0] == 3 for call in load.call_args_list))
            # Same renderer, explicit two-part clicked card. Geometry cannot follow content.
            _, fixed_quad = synth.render_scene(np.random.default_rng(5), bank, arts, target_index=7)
            np.testing.assert_array_equal(quad, fixed_quad)
            self.assertIn(7, [call.args[0] for call in load.call_args_list])
        width = np.linalg.norm(quad[1] - quad[0])
        height = np.linalg.norm(quad[3] - quad[0])
        # Perspective perturbs the 63:88 rectangle, but must not swap its printed axes.
        self.assertLess(width, height)
        self.assertGreater(width / height, 0.5)

    def test_validation_strata_metrics_and_empty_group(self):
        bank = self.bank()
        indices = train_detector.validation_targets(bank, 400, 999)
        groups = bank.groups[indices]
        self.assertEqual(int((groups != "ordinary").sum()), 100)
        self.assertEqual(set(groups), {"ordinary", "room", "split", "aftermath", "flip"})
        np.testing.assert_array_equal(indices, train_detector.validation_targets(bank, 400, 999))
        quad = np.float32([[0, 0], [100, 0], [100, 140], [0, 140]])
        quads = np.stack([quad] * 4)
        pred = quads + np.float32([0, 20, 40, 0])[:, None, None] * np.float32([1, 0])
        with patch.object(train_detector, "predict_scenes", return_value=(pred, pred, np.tile([0, -1], (4, 1)))):
            metrics = train_detector.eval_scenes(None, None, quads, torch.device("cpu"), groups=np.array(["ordinary", "room", "flip", "ordinary"]))
        self.assertEqual(metrics["two_part"], {"n": 2, "err": 0.3, "err_mean": 0.3, "hit": 0.0})
        self.assertEqual(metrics["room"]["err"], 0.2)
        self.assertEqual(metrics["ordinary"]["hit"], 1.0)
        self.assertEqual(metrics["synth"]["hit"], 0.5)
        self.assertEqual(metrics["split"], {"n": 0, "err": None, "err_mean": None, "hit": None})

    def test_validation_cache_tracks_files_manifest_and_backgrounds(self):
        bank = self.bank()
        arts = synth.ArtBank([])
        image = np.zeros((256, 256, 3), np.uint8)
        quad = np.float32([[30, 20], [93, 20], [93, 108], [30, 108]])
        with (
            patch.object(train_detector, "DATA_DIR", self.root),
            patch.object(train_detector, "CardBank", side_effect=lambda: bank),
            patch.object(train_detector, "ArtBank", side_effect=lambda: arts),
            patch.object(synth.ImageBank, "build"),
            patch.object(synth, "render_scene", return_value=(image, quad)) as render,
        ):
            np.savez(self.root / "det-val-8-999.npz", scenes=["stale"], quads=["stale"])
            first = train_detector.val_scenes(8, 0)
            second = train_detector.val_scenes(8, 0)
            self.assertEqual(render.call_count, 8)
            for a, b in zip(first, second, strict=True):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(first[2], bank.groups[train_detector.validation_targets(bank, 8, 999)])
            bank.groups[1] = "split"
            train_detector.val_scenes(8, 0)
            self.assertEqual(render.call_count, 16)
            old_path = bank.path
            bank = synth.CardBank([*bank.paths, self.root / "new.jpg"])
            self.assertNotEqual(old_path, bank.path)  # decoded ImageBank invalidates on file names
            train_detector.val_scenes(8, 0)
            self.assertEqual(render.call_count, 24)
            arts = synth.ArtBank([self.root / "new-art.jpg"])
            train_detector.val_scenes(8, 0)
            self.assertEqual(render.call_count, 32)

    def test_geometry_distinguishes_physical_axis_from_corner_roll(self):
        quad = synth.quad_from_pose(230, 190, 100, 23, None)
        self.assertEqual(detector_geometry(quad, quad + np.float32([5, 0])), "ok")
        self.assertEqual(detector_geometry(quad, np.roll(quad, 2, axis=0)), "rot180")
        self.assertEqual(detector_geometry(quad, np.roll(quad, 1, axis=0)), "other")
        perpendicular = synth.quad_from_pose(235, 185, 100, 113, None)
        self.assertEqual(detector_geometry(quad, perpendicular), "perpendicular")
        self.assertEqual(detector_geometry(quad, quad + np.float32([40, 0])), "other")
        self.assertEqual(detector_geometry(quad, quad + np.float32([14.9, 0])), "ok")
        self.assertEqual(detector_geometry(quad, quad + np.float32([15.1, 0])), "other")
