"""Gallery coverage, face geometry, and append-only refresh regressions (no downloads)."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import scryfall
from .detect import frame_of
from .gallery import runtime_metadata

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
    def test_runtime_metadata_keeps_embedding_order_but_defers_siblings(self):
        entries = [
            {
                "id": "back-1",
                "name": "Back",
                "face": 1,
                "lang": "ja",
                "url": "crop",
                "split": "train",
                "printings": [{"id": "sibling", "name": "Other name", "lang": "de", "set": "fin"}],
            },
            {"id": "front", "name": "Front"},
        ]
        arts, printings = runtime_metadata(entries, ["2015", "1993"])
        self.assertEqual(
            arts,
            [
                {"id": "back-1", "name": "Back", "face": 1, "lang": "ja", "url": "crop", "frame": "2015", "printing_count": 1},
                {"id": "front", "name": "Front", "frame": "1993", "printing_count": 0},
            ],
        )
        self.assertEqual(printings, {"back-1": entries[0]["printings"]})
        self.assertIn("printings", entries[0])

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

    def test_hub_cards_are_rejected_and_retired_from_an_older_arts_json(self):
        # Bind // Liberate (cmb1) is a `split`-layout sketch card in a plain frame; its flat
        # crop became a hub for degraded real-camera queries. Unfinity stays: it is `funny`
        # but not playtest. World Championship decklists are `token`-layout inserts typed
        # as a bare "Card"; real tokens keep their type line.
        bind = {**STUDIOUS, "id": "22222222-2222-2222-2222-222222222222", "layout": "split", "set": "cmb1", "set_type": "funny", "promo_types": ["playtest"]}
        unfinity = {**ABRADE, "id": "33333333-3333-3333-3333-333333333333", "set": "unf", "set_type": "funny", "promo_types": []}
        decklist = {**ABRADE, "id": "44444444-4444-4444-4444-444444444444", "layout": "token", "set": "wc01", "set_type": "memorabilia", "type_line": "Card"}
        token = {**ABRADE, "id": "55555555-5555-5555-5555-555555555555", "layout": "token", "set": "tmt", "type_line": "Token Creature — Turtle"}
        self.assertFalse(scryfall.usable(bind))
        self.assertFalse(scryfall.usable(decklist))
        self.assertTrue(scryfall.usable(unfinity))
        self.assertTrue(scryfall.usable(token))
        excluded = set()
        with gzip.open(self.bulk, "wt") as out:
            for card in [bind, ABRADE, unfinity]:
                out.write(json.dumps(card) + "\n")
        entries = scryfall.usable_entries(self.bulk, excluded)
        self.assertEqual([e["set"] for e in entries], ["soa", "unf"])
        self.assertEqual(excluded, {bind["id"], f"{bind['id']}-1"})

        # An arts.json exported before the rule keeps both halves at their indices, so a
        # resumed checkpoint's classes and the eval split do not move; only the flag changes.
        old = [
            {"id": bind["id"], "split": "train", "illustration_id": "bind:face:0", "url": "x"},
            {"id": f"{bind['id']}-1", "split": "eval", "illustration_id": "bind:face:1", "url": "x"},
            dict(entries[0], split="eval"),
        ]
        self.assertEqual(scryfall.add_metadata(old, entries, excluded), 2)
        self.assertEqual([a["id"] for a in old], [bind["id"], f"{bind['id']}-1", ABRADE["id"]])
        self.assertEqual([a.get("excluded", False) for a in old], [True, True, False])
        self.assertEqual([a.get("alias_of") for a in old], [None, None, None])
        self.assertEqual([scryfall.embeds(a) for a in old], [False, False, True])
        self.assertEqual(scryfall.add_metadata(old, entries, excluded), 0)
        self.assertEqual(len(scryfall.extend_to_all(old, entries, excluded)), 4)  # Unfinity is added, Bind is not re-added

    def test_layouts_with_shared_crops_and_exclusions(self):
        for layout in ["prepare", "adventure", "meld"]:
            with self.subTest(layout=layout):
                self.assertEqual(len(self.entries([{**STUDIOUS, "layout": layout}])), 1)
        for layout in ["battle", "art_series"]:
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

    def test_siblings_share_one_crop_but_keep_printing_treatments_and_languages(self):
        extended = dict(ABRADE, illustration_id="shared-art", lang="en", frame_effects=["extendedart"], border_color="black")
        # A normal legendary crown is not a special treatment. Its higher collector number
        # must not cause the lexically earlier extended-art scan to win.
        regular = dict(extended, id="regular", collector_number="200", frame_effects=["legendary"], frame="2015")
        japanese = dict(regular, id="japanese", lang="ja")
        promo = dict(regular, id="promo", promo=True, set="psoa")
        pending = dict(regular, id="pending-scan", image_status="missing", image_uris={})
        orphan = dict(pending, id="no-art", illustration_id="unscanned-art")
        entries = self.entries([extended, japanese, promo, pending, regular, orphan, dict(regular, id="digital", digital=True)])
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["id"], "regular")
        self.assertEqual(entries[0]["illustration_id"], "shared-art")
        siblings = {p["id"]: p for p in entries[0]["printings"]}
        self.assertEqual(set(siblings), {extended["id"], "regular", "japanese", "promo", "pending-scan"})
        self.assertEqual(siblings[extended["id"]]["frame_effects"], ["extendedart"])
        self.assertEqual(siblings["regular"]["scryfall_frame"], "2015")
        self.assertEqual(siblings["japanese"]["lang"], "ja")
        self.assertTrue(siblings["promo"]["promo"])
        self.assertNotIn("url", siblings["regular"])
        # Bulk order cannot switch which representative a fresh gallery downloads.
        self.assertEqual(entries, self.entries([regular, promo, japanese, extended, pending]))

    def test_existing_extended_id_crop_and_split_survive_new_regular_printing(self):
        extended = dict(ABRADE, illustration_id="shared-art", lang="en", frame_effects=["extendedart"])
        original = self.entries([extended])[0]
        original.pop("illustration_id")
        original.pop("printings")
        original["split"] = "eval"
        existing = [dict(original)]
        entries = self.entries([extended, dict(extended, id="regular", frame_effects=[]), STUDIOUS])
        updated = scryfall.extend_to_all(existing, entries)
        self.assertEqual({k: updated[0][k] for k in original}, original)
        self.assertEqual(len(updated), 2)
        self.assertEqual(updated[1]["id"], STUDIOUS["id"])
        self.assertEqual(updated[1]["split"], "train")
        self.assertEqual(len(updated[0]["printings"]), 2)
        self.assertEqual(scryfall.extend_to_all(updated, entries), updated)

    def test_repeated_reverse_art_preserves_ids_and_splits_but_only_embeds_eval_row(self):
        from . import data
        from .gallery import printing_index

        first = dict(JADZI, card_faces=[dict(face, illustration_id=f"art-{i}") for i, face in enumerate(JADZI["card_faces"])])
        second = dict(first, id="another-printing", card_faces=[dict(first["card_faces"][0], illustration_id="different-front"), first["card_faces"][1]])
        entries = self.entries([first, second])
        self.assertEqual(len(entries), 3)  # two front artworks, one shared back
        existing = [{"id": first["id"] + "-1", "split": "train"}, {"id": second["id"] + "-1", "split": "eval"}]
        scryfall.add_metadata(existing, entries)
        self.assertEqual([(a["id"], a["split"]) for a in existing], [(first["id"] + "-1", "train"), (second["id"] + "-1", "eval")])
        self.assertEqual(existing[0]["alias_of"], second["id"] + "-1")
        self.assertNotIn("alias_of", existing[1])
        (self.root / "arts.json").write_text(json.dumps(existing))
        for a in existing:
            (self.root / f"{a['id']}.jpg").touch()
        with patch.object(data, "DATA_DIR", self.root), patch.object(data, "ART_DIR", self.root):
            canonical = data.load_arts()
        self.assertEqual(canonical, [existing[1]])
        self.assertEqual(printing_index(canonical), {first["id"] + "-1": 0, second["id"] + "-1": 0})
        self.assertEqual(scryfall.add_metadata(existing, entries), 0)

    def test_gallery_cache_identity_includes_order_membership_and_split_not_siblings(self):
        from .gallery import gallery_fingerprint

        arts = [{"id": "a", "split": "train"}, {"id": "b", "split": "eval"}]
        digest = gallery_fingerprint(arts)
        self.assertNotEqual(digest, gallery_fingerprint(arts[::-1]))
        self.assertNotEqual(digest, gallery_fingerprint([arts[0], {"id": "c", "split": "eval"}]))
        self.assertNotEqual(digest, gallery_fingerprint([arts[0], {"id": "b", "split": "train"}]))
        self.assertEqual(digest, gallery_fingerprint([dict(a, printings=[{"id": "new"}]) for a in arts]))
