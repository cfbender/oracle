"""Two-part scan geometry, grouping, and exported crop contracts without a checkpoint."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import httpx
import numpy as np
import onnxruntime as ort
import torch

from . import scryfall
from .bundle import rgba
from .data import to_tensor
from .detect import CARD_H, CARD_W, FRAME_NAMES, art_crops, frame_crop, frame_of
from .export import export_graph
from .graphs import EmbedGraph, SearchGraph
from .real import art_from_card
from .synth import draw_card
from .test_scryfall import ABRADE


def gradient():
    y, x = np.mgrid[:CARD_H, :CARD_W]
    return np.stack([x, y * 0.6, x * 0.2 + y * 0.3], axis=-1).astype(np.uint8)


class LayoutTest(unittest.TestCase):
    def test_metadata_uses_type_and_keyword_not_names_or_aspect(self):
        room = dict(
            ABRADE,
            layout="split",
            illustration_id="shared",
            image_uris={"normal": "https://scan.test/card.jpg"},
            card_faces=[{"name": "Mirror", "type_line": "Enchantment — Room", "illustration_id": "shared"}, {"name": "Realm"}],
        )
        for group, changes in [
            ("room", {}),
            ("aftermath", {"keywords": ["Aftermath"], "card_faces": [{"name": "Commit"}, {"name": "Memory"}]}),
            ("split", {"card_faces": [{"name": "Room in the name is not a type"}, {"name": "Ice"}]}),
            ("flip", {"layout": "flip"}),
        ]:
            with self.subTest(group=group), tempfile.TemporaryDirectory() as temp:
                card = {**room, **changes}
                path = Path(temp) / "bulk.gz"
                with gzip.open(path, "wt") as out:
                    translation = dict(card, id="translation", lang="de", card_faces=[dict(f, illustration_id="shared") for f in card["card_faces"]])
                    for c in [card, translation, dict(card, id="pending", image_status="missing", image_uris={})]:
                        out.write(json.dumps(c) + "\n")
                entries = scryfall.usable_entries(path)
                self.assertEqual(len(entries), 2)
                self.assertEqual([a["face"] for a in entries], [0, 1])
                self.assertTrue(entries[1]["id"].endswith("-1"))
                self.assertNotEqual(entries[0]["illustration_id"], entries[1]["illustration_id"])
                for i, entry in enumerate(entries):
                    self.assertEqual(len(entry["printings"]), 3)
                    self.assertEqual(entry["url"], "https://scan.test/card.jpg")
                    self.assertEqual(entry["layout_group"], group)
                    self.assertEqual(frame_of(0.4, entry["layout"], i, group), f"{group}_{i}")
                    self.assertEqual(frame_of(2.8, entry["layout"], i, group), f"{group}_{i}")
                self.assertFalse(scryfall.usable(dict(card, image_uris={"art_crop": "combined"})))
                for n in (0, 1, 3, 5):
                    self.assertFalse(scryfall.supported_faces(dict(card, card_faces=[{"name": "Part"}] * n)))

    def test_native_crops_have_the_expected_location_and_orientation(self):
        card = gradient()
        # Explicit source pixels independently pin box, face order and rotation direction.
        # In a clockwise crop, top-left comes from the source bottom-left, not top-right.
        expected = {
            "room_0": ((317, 33), (169, 132)),
            "room_1": ((165, 33), (17, 132)),
            "split_0": ((314, 40), (197, 121)),
            "split_1": ((149, 40), (33, 121)),
            "aftermath_0": ((40, 18), (116, 230)),
            "aftermath_1": ((197, 206), (319, 137)),
            "flip_0": ((110, 21), (228, 121)),
            "flip_1": ((228, 227), (110, 127)),
        }
        self.assertEqual(FRAME_NAMES[:6], ["modern", "old", "extended", "tall", "right", "left"])
        for frame, (first, last) in expected.items():
            with self.subTest(frame=frame):
                crop = frame_crop(card, frame)
                np.testing.assert_array_equal(crop[0, 0], card[first])
                np.testing.assert_array_equal(crop[-1, -1], card[last])
                # Real correction training must normalize orientation too.
                real = art_from_card(card, frame=frame)
                np.testing.assert_allclose(real[8:-8, 8:-8], cv2.resize(crop, (128, 128))[8:-8, 8:-8], atol=2)

    def test_download_cuts_art_but_keeps_full_scan_for_synthetic_scenes(self):
        card = gradient()
        _, encoded = cv2.imencode(".png", card)
        entry = {
            "id": "half-1",
            "face": 1,
            "layout": "split",
            "layout_group": "aftermath",
            "url": "https://scan.test/card",
            "card_url": "https://scan.test/card",
        }
        with (
            tempfile.TemporaryDirectory() as temp,
            httpx.Client(
                transport=httpx.MockTransport(lambda r: httpx.Response(200, content=encoded.tobytes(), headers={"content-type": "image/png"}))
            ) as client,
        ):
            root = Path(temp)
            with patch.object(scryfall, "REQUEST_GAP_S", 0):
                self.assertEqual(scryfall.fetch_image(client, entry, root), ("half-1", True))
                image = cv2.imread(str(root / "half-1.jpg"))
                self.assertEqual(image.shape[:2], (70, 123))
                np.testing.assert_allclose(image.astype(float), frame_crop(card, "aftermath_1").astype(float), atol=5)
                (root / "half-1.jpg").unlink()
                scryfall.fetch_image(client, entry, root, "card_url")
                np.testing.assert_array_equal(cv2.imread(str(root / "half-1.jpg")), card)

    def test_synthetic_target_uses_full_scan_not_an_enlarged_half(self):
        class Bank:
            def __len__(self):
                return 2

            def load(self, index, _short):
                self.index = index
                return gradient()

        bank = Bank()
        canvas = np.zeros((CARD_H, CARD_W, 3), np.float32)
        quad = np.float32([[0, 0], [CARD_W, 0], [CARD_W, CARD_H], [0, CARD_H]])
        draw_card(canvas, np.random.default_rng(0), bank, quad, shadow=False, index=1)
        self.assertEqual(bank.index, 1)
        np.testing.assert_allclose(canvas[50:-50, 50:-50], gradient()[50:-50, 50:-50], atol=2)

    def test_exported_embed_crops_match_python_for_every_frame(self):
        # Identity embedder makes every pixel observable, unlike a random CNN which can
        # hide a wrong crop behind nearly identical embeddings. Also execute real ONNX.
        graph = EmbedGraph(torch.nn.Identity())
        scene = rgba(gradient())
        quad = np.float32([[0, 0], [CARD_W, 0], [CARD_W, CARD_H], [0, CARD_H]])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "embed.onnx"
            export_graph(
                graph, (torch.from_numpy(scene), torch.from_numpy(quad)), path, ["scene", "quad"], ["crops"], dynamic={"scene": {0: "height", 1: "width"}}
            )
            actual = ort.InferenceSession(str(path)).run(None, {"scene": scene, "quad": quad})[0]
        expected = to_tensor(art_crops(gradient())).numpy()
        self.assertEqual(actual.shape, (14, 3, 128, 128))
        np.testing.assert_allclose(actual, expected, atol=0.018)

    def test_search_scores_only_the_gallery_rows_own_frame(self):
        frames = np.array([13, 6, 0])
        gallery = np.eye(3, dtype=np.float32)
        queries = np.zeros((14, 3), np.float32)
        queries[0] = [1, 1, 0.1]  # tempting scores in the wrong frame
        queries[13, 0] = 0.3
        queries[6, 1] = 0.8
        graph = SearchGraph(gallery, frames, np.array([0.02, 0.02, 0]), 3, torch.float32)
        ids, scores = graph(torch.from_numpy(queries))
        self.assertEqual(ids.tolist(), [1, 0, 2])
        np.testing.assert_allclose(scores.numpy(), [0.78, 0.28, 0.1], atol=1e-6)
