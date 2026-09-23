"""Gallery coverage, face geometry, and append-only refresh regressions (no downloads)."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import scryfall
from .detect import frame_of

ABRADE = {
    "id": "d1ed5b20-94f6-455f-80a5-e4ed360167be",
    "oracle_id": "f9db72dc-9a5b-48a4-a86e-7464d9a2166a",
    "name": "Abrade",
    "set": "soa",
    "collector_number": "102",
    "layout": "normal",
    "lang": "ja",
    "games": ["paper"],
    "image_status": "highres_scan",
    "digital": False,
    "image_uris": {"art_crop": "https://cards.scryfall.io/art_crop/front/d/1/d1ed5b20-94f6-455f-80a5-e4ed360167be.jpg"},
}
# Relevant fields from the real bulk records. MDFCs have no top-level image_uris;
# prepare/adventure have card_faces but share a single top-level art crop.
JADZI = {
    **{k: v for k, v in ABRADE.items() if k != "image_uris"},
    "id": "b0a96416-9ee5-4202-a99f-e09db8794567",
    "oracle_id": "ca79cd3f-13ba-4d11-b885-48a504ae69aa",
    "name": "Jadzi, Oracle of Arcavios // Journey to the Oracle",
    "set": "stx",
    "collector_number": "325",
    "layout": "modal_dfc",
    "lang": "en",
    "card_faces": [
        {
            "name": "Jadzi, Oracle of Arcavios",
            "image_uris": {"art_crop": "https://cards.scryfall.io/art_crop/front/b/0/b0a96416-9ee5-4202-a99f-e09db8794567.jpg"},
        },
        {"name": "Journey to the Oracle", "image_uris": {"art_crop": "https://cards.scryfall.io/art_crop/back/b/0/b0a96416-9ee5-4202-a99f-e09db8794567.jpg"}},
    ],
}
STUDIOUS = {
    **ABRADE,
    "id": "24f888dd-785c-4089-a89c-03f9080130ed",
    "oracle_id": "58b0c737-0a84-4f9a-b3b7-300c5de43874",
    "name": "Studious First-Year // Rampant Growth",
    "set": "sos",
    "collector_number": "162",
    "layout": "prepare",
    "lang": "en",
    "image_uris": {"art_crop": "https://cards.scryfall.io/art_crop/front/2/4/24f888dd-785c-4089-a89c-03f9080130ed.jpg"},
    "card_faces": [{"name": "Studious First-Year"}, {"name": "Rampant Growth"}],
}


class ScryfallTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.bulk = self.root / "unique-artwork.jsonl.gz"

    def entries(self, cards):
        with gzip.open(self.bulk, "wt") as out:
            for card in cards:
                out.write(json.dumps(card) + "\n")
        return scryfall.usable_entries(self.bulk)

    def test_reported_cards_and_face_identity(self):
        entries = self.entries([ABRADE, JADZI, STUDIOUS])
        self.assertEqual(
            [e["name"] for e in entries], ["Abrade", "Jadzi, Oracle of Arcavios", "Journey to the Oracle", "Studious First-Year // Rampant Growth"]
        )
        self.assertEqual([e["id"] for e in entries], [ABRADE["id"], JADZI["id"], JADZI["id"] + "-1", STUDIOUS["id"]])
        self.assertEqual([e["face"] for e in entries], [0, 0, 1, 0])
        self.assertEqual(entries[0]["lang"], "ja")
        self.assertEqual(entries[2]["url"], JADZI["card_faces"][1]["image_uris"]["art_crop"])
        self.assertEqual(entries[2]["layout"], "modal_dfc")
        self.assertEqual(entries[3]["url"], STUDIOUS["image_uris"]["art_crop"])

    def test_rejects_digital_nonpaper_and_unready_images_in_any_language(self):
        for change in [{"digital": True}, {"games": ["arena"]}, {"image_status": "placeholder"}, {"image_status": "missing"}, {"image_uris": {}}]:
            with self.subTest(change=change):
                self.assertFalse(scryfall.usable({**ABRADE, **change}))
        self.assertTrue(scryfall.usable({**ABRADE, "lang": "zhs", "image_status": "lowres"}))
        self.assertFalse(scryfall.usable({**JADZI, "digital": True}))

    def test_layouts_with_shared_crops_and_exclusions(self):
        for layout in ["prepare", "adventure", "meld"]:
            with self.subTest(layout=layout):
                self.assertEqual(len(self.entries([{**STUDIOUS, "layout": layout}])), 1)
        for layout in ["split", "flip", "art_series"]:
            with self.subTest(layout=layout):
                self.assertFalse(scryfall.usable({**ABRADE, "layout": layout}))
                self.assertFalse(scryfall.usable({**JADZI, "layout": layout}))

    def test_all_separate_side_layouts_and_missing_front_do_not_renumber_back(self):
        for layout in ["transform", "modal_dfc", "reversible_card", "double_faced_token"]:
            with self.subTest(layout=layout):
                card = {**JADZI, "layout": layout}
                self.assertEqual(len(self.entries([card])), 2)
                card["card_faces"] = [{"name": "Missing scan"}, JADZI["card_faces"][1]]
                entries = self.entries([card])
                self.assertEqual(len(entries), 1)
                self.assertEqual(entries[0]["id"], JADZI["id"] + "-1")
                self.assertEqual(entries[0]["face"], 1)

    def test_reversible_faces_use_face_oracle_id(self):
        card = {k: v for k, v in JADZI.items() if k != "oracle_id"}
        card["layout"] = "reversible_card"
        card["card_faces"] = [dict(face, oracle_id=f"oracle-{i}") for i, face in enumerate(JADZI["card_faces"])]
        self.assertEqual([e["oracle_id"] for e in self.entries([card])], ["oracle-0", "oracle-1"])

    def test_metadata_and_update_preserve_existing_order_ids_and_splits(self):
        normal = {**ABRADE, "id": "11111111-1111-1111-1111-111111111111", "lang": "en"}
        cards = [normal, ABRADE, JADZI, STUDIOUS]
        entries = self.entries(cards)
        # Simulate an old sampled arts.json, with one held-out art and an obsolete entry.
        existing = [dict(entries[0], split="eval"), {"id": "old-art", "split": "train"}]
        for key in ["face", "lang", "layout", "collector_number"]:
            existing[0].pop(key)
        arts_path = self.root / "arts.json"
        arts_path.write_text(json.dumps(existing))
        art_dir = self.root / "art"
        art_dir.mkdir()
        for art in existing:
            (art_dir / f"{art['id']}.jpg").touch()

        def fetch(_client, entry):
            (art_dir / f"{entry['id']}.jpg").touch()
            return entry["id"], True

        def refresh(*_args):
            self.entries(cards)
            return self.bulk

        with (
            patch.object(scryfall, "DATA_DIR", self.root),
            patch.object(scryfall, "ART_DIR", art_dir),
            patch.object(scryfall, "download_bulk", return_value=self.bulk),
            patch.object(scryfall, "fetch_image", side_effect=fetch) as download,
        ):
            with patch("sys.argv", ["scryfall", "--metadata"]):
                scryfall.main()
            metadata = json.loads(arts_path.read_text())
            self.assertEqual(len(metadata), 2)
            self.assertEqual(metadata[0], dict(entries[0], split="eval"))
            download.assert_not_called()
            # A fresh bulk has the same records. Exercise the actual --update CLI twice.
            with patch.object(scryfall, "download_bulk", side_effect=refresh):
                with patch("sys.argv", ["scryfall", "--update"]):
                    scryfall.main()
                    updated = json.loads(arts_path.read_text())
                    scryfall.main()
            self.assertEqual(json.loads(arts_path.read_text()), updated)
            self.assertEqual(updated[:2], metadata)
            self.assertEqual(updated[2:], [dict(e, split="train") for e in entries[1:]])
            self.assertEqual(download.call_count, 4)
            self.assertEqual(scryfall.add_metadata(updated, entries), 0)

    def test_face_frame_uses_its_own_aspect(self):
        for layout in ["prepare", "transform", "modal_dfc", "reversible_card", "double_faced_token", "meld"]:
            self.assertEqual(frame_of(1.37, layout), "modern")
            self.assertEqual(frame_of(1.62, layout), "extended")
        self.assertEqual(frame_of(0.88, "double_faced_token"), "tall")
        self.assertEqual(frame_of(0.415, "class"), "left")
        self.assertEqual(frame_of(0.415, "transform"), "right")
