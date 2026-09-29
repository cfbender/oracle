"""Phone-scanner detector scenes (CPU, no downloads): geometry, profile plumbing, cache keys."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from . import scene_datasets, synth, train_detector
from .constants import DET_INPUT, SCENE
from .scene_geometry import quad_short
from .scene_renderer import render_phone_scene


class PhoneScenesTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "two-part.json").write_text(json.dumps({}))
        self.bank = synth.CardBank([self.root / f"{i}.jpg" for i in range(3)])
        self.arts = synth.ArtBank([])
        card = np.full((350, 250, 3), 90, np.uint8)
        card[20:170, 20:230] = [200, 60, 40]
        self.enterContext(patch.object(self.bank, "load", return_value=card))

    def test_phone_scenes_hold_one_large_upright_card_over_the_frame_centre(self):
        centre = (DET_INPUT / 2, DET_INPUT / 2)
        for seed in range(40):
            scene, quad = render_phone_scene(np.random.default_rng(seed), self.bank, self.arts)
            self.assertEqual((scene.shape, scene.dtype), ((DET_INPUT, DET_INPUT, 3), np.uint8))
            short = quad_short(quad) * SCENE / DET_INPUT
            self.assertTrue(85 <= short <= 380, f"seed {seed}: short side {short:.0f} native px")
            top = quad[1] - quad[0]
            tilt = abs(np.degrees(np.arctan2(top[1], top[0])))
            self.assertTrue(tilt <= 20 or tilt >= 160, f"seed {seed}: card rotated {tilt:.0f} degrees")
            inside = cv2.pointPolygonTest(quad.reshape(-1, 1, 2), centre, measureDist=False)
            self.assertGreaterEqual(inside, 0, f"seed {seed}: frame centre off the card")

    def test_profiles_are_plumbed_and_table_scenes_are_unchanged(self):
        default = synth.render_scene(np.random.default_rng(5), self.bank, self.arts)
        table = synth.render_scene(np.random.default_rng(5), self.bank, self.arts, profile="table")
        np.testing.assert_array_equal(default[1], table[1])
        with self.assertRaises(ValueError):
            synth.render_scene(np.random.default_rng(5), self.bank, self.arts, profile="nope")
        with patch.object(synth.ImageBank, "build"), patch.object(scene_datasets, "render_scene", return_value=default) as render:
            dataset = scene_datasets.SceneDataset(1, cards=self.bank, arts=self.arts, profile="phone")
            dataset[0]
        self.assertEqual(render.call_args.kwargs["profile"], "phone")

    def test_phone_validation_scenes_are_cached_separately(self):
        image = np.zeros((DET_INPUT, DET_INPUT, 3), np.uint8)
        quad = np.float32([[30, 20], [93, 20], [93, 108], [30, 108]])
        with (
            patch.object(train_detector, "DATA_DIR", self.root),
            patch.object(train_detector, "CardBank", side_effect=lambda: self.bank),
            patch.object(train_detector, "ArtBank", side_effect=lambda: self.arts),
            patch.object(synth.ImageBank, "build"),
            patch.object(scene_datasets, "render_scene", return_value=(image, quad)) as render,
        ):
            train_detector.val_scenes(4, 0)
            train_detector.val_scenes(4, 0, profile="phone")
            train_detector.val_scenes(4, 0, profile="phone")
        self.assertEqual(render.call_count, 8)
        self.assertEqual({call.kwargs["profile"] for call in render.call_args_list}, {"table", "phone"})

    def test_held_out_detector_eval_uses_only_trusted_outlines(self):
        from . import real

        for capture_id in ("imported", "drawn"):
            (self.root / capture_id).mkdir()
            (self.root / capture_id / "crop.jpg").write_bytes(b"jpg")
        quad = [[0, 0], [10, 0], [10, 14], [0, 14]]
        imported = {"capture_id": "imported", "label": "a", "quad": quad, "quad_source": "detector", "top5": []}
        drawn = {**imported, "capture_id": "drawn", "quad_source": "manual"}
        with patch.object(real, "REAL_DIR", self.root):
            self.assertIsNone(train_detector.real_eval_set([imported]))
            self.assertEqual([r["capture_id"] for r in train_detector.real_eval_set([imported, drawn]).rows], ["drawn"])


if __name__ == "__main__":
    unittest.main()
