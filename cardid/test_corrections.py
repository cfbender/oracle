"""CPU-only importer, publish gate, and local/SSH-shell publisher regression tests."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
from PIL import Image

from .corrections import latest_labels, merge, pull
from .nightly import fingerprint, publish_allowed, run

CID = "00000000-0000-0000-0000-000000000001"  # independently known train hash
LABEL = "11111111-1111-1111-1111-111111111111"


class CorrectionsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.real = self.root / "real"
        image = np.zeros((500, 400, 3), dtype=np.uint8)
        image[20:370, 40:290] = [230, 40, 10]
        image[20:190, 40:160] = [20, 220, 40]  # asymmetric top-left patch catches reversed warps
        out = io.BytesIO()
        Image.fromarray(image).save(out, format="JPEG", quality=95)
        self.jpeg = out.getvalue()
        self.row = {"capture_id": CID, "label": LABEL, "click": [123, 234], "quad": [[40, 20], [290, 20], [290, 370], [40, 370]], "up_vote": 0.92}

    def test_warp_dedup_relabel_and_skip(self):
        self.assertTrue(merge(self.row, self.jpeg, self.real))
        self.assertFalse(merge(self.row, self.jpeg, self.real))
        card = Image.open(self.real / CID / "card.png")
        self.assertEqual(card.size, (250, 350))
        self.assertGreater(card.getpixel((30, 40))[1], 200)
        self.assertGreater(card.getpixel((220, 300))[0], 200)
        row = latest_labels(self.real)[CID]
        self.assertNotIn("up_correct", row)
        self.assertEqual(row["split"], "train")
        relabelled = {**self.row, "label": "22222222-2222-2222-2222-222222222222"}
        self.assertTrue(merge(relabelled, self.jpeg, self.real))
        self.assertEqual(latest_labels(self.real)[CID]["label"], relabelled["label"])
        self.assertEqual(latest_labels(self.real)[CID]["split"], row["split"])
        merge({**self.row, "label": None}, self.jpeg, self.real)
        self.assertIsNone(latest_labels(self.real)[CID]["label"])
        self.assertFalse((self.real / CID / "card.png").exists())
        self.assertEqual(len((self.real / "labels.jsonl").read_text().splitlines()), 3)

    def test_missing_or_degenerate_quad_stays_pending(self):
        for quad in [None, [[1, 1]] * 4, [[1, 1], [50, 50], [1, 50], [50, 1]]]:
            merge({**self.row, "quad": quad}, self.jpeg, self.real)
            self.assertTrue((self.real / CID / "crop.jpg").exists())
            self.assertFalse((self.real / CID / "card.png").exists())

    def test_refuses_paths_bad_images_and_bombs(self):
        with self.assertRaises(ValueError):
            merge({**self.row, "capture_id": "../../escape"}, self.jpeg, self.real)
        out = io.BytesIO()
        Image.new("RGB", (641, 640)).save(out, format="JPEG")
        for jpeg in [b"x" * 150_001, out.getvalue(), b"not JPEG"]:
            with self.assertRaises((ValueError, OSError)):
                merge(self.row, jpeg, self.real)

    def test_filesystem_pull_uses_last_label_and_keeps_existing_local_samples(self):
        source = self.root / "source"
        (source / CID).mkdir(parents=True)
        (source / CID / "crop.jpg").write_bytes(self.jpeg)
        (source / "labels.jsonl").write_text(json.dumps({**self.row, "label": None}) + "\n" + json.dumps(self.row) + "\n")
        self.real.mkdir()
        local = {"capture_id": "local-capture", "label": "local-label"}
        (self.real / "labels.jsonl").write_text(json.dumps(local) + "\n")
        self.assertEqual(pull(self.real, from_dir=source), 1)
        self.assertEqual(pull(self.real, from_dir=source), 0)
        self.assertEqual(latest_labels(self.real)["local-capture"], local)

    def test_http_cursor_only_advances_after_success_and_retries_deduplicate(self):
        second = {**self.row, "capture_id": "00000000-0000-0000-0000-000000000002"}
        failed = False
        cursors = []

        def handle(request):
            nonlocal failed
            self.assertEqual(request.headers["Authorization"], "Bearer test-token")
            if request.url.path.endswith("corrections"):
                cursor = int(request.url.params["cursor"])
                cursors.append(cursor)
                return httpx.Response(200, json={"data": {"cursor": 2, "has_more": False, "corrections": [self.row, second] if cursor == 0 else []}})
            if second["capture_id"] in request.url.path and not failed:
                failed = True
                return httpx.Response(503)
            return httpx.Response(200, content=self.jpeg)

        client = httpx.Client(transport=httpx.MockTransport(handle), headers={"Authorization": "Bearer test-token"})
        with patch("cardid.corrections.httpx.Client", return_value=client), patch.dict("os.environ", CARDID_CORRECTIONS_TOKEN="test-token"):
            with self.assertRaises(httpx.HTTPStatusError):
                pull(self.real, "https://example.test")
        self.assertFalse((self.real / ".corrections-cursor.json").exists())
        for expected in [1, 0]:
            client = httpx.Client(transport=httpx.MockTransport(handle), headers={"Authorization": "Bearer test-token"})
            with patch("cardid.corrections.httpx.Client", return_value=client), patch.dict("os.environ", CARDID_CORRECTIONS_TOKEN="test-token"):
                self.assertEqual(pull(self.real, "https://example.test"), expected)
        self.assertEqual(cursors, [0, 0, 2])
        self.assertEqual(len((self.real / "labels.jsonl").read_text().splitlines()), 2)

    def test_gate_rejects_regression_empty_changed_eval_and_no_new_data(self):
        base = {"count": 5, "correct": 4, "captures": "same"}
        self.assertTrue(publish_allowed(True, base, base))
        self.assertTrue(publish_allowed(True, base, {**base, "correct": 5}))
        self.assertFalse(publish_allowed(False, base, base))
        for change in [{"correct": 3}, {"count": 6}, {"captures": "other"}, {"correct": 6}]:
            self.assertFalse(publish_allowed(True, base, {**base, **change}))
        self.assertFalse(publish_allowed(True, {**base, "count": 0}, {**base, "count": 0}))

    def test_dry_run_merges_without_marking_trained_and_seen_data_refuses_training(self):
        source = self.root / "source"
        (source / CID).mkdir(parents=True)
        (source / CID / "crop.jpg").write_bytes(self.jpeg)
        (source / "labels.jsonl").write_text(json.dumps(self.row) + "\n")
        state_dir = self.root / "state"
        state_dir.mkdir()
        args = argparse.Namespace(real_dir=self.real, state_dir=state_dir, server=None, from_dir=source, dry_run=True)
        with patch("cardid.nightly.command", side_effect=AssertionError("must not train/publish")), redirect_stdout(io.StringIO()) as output:
            run(args)
            self.assertIn("gate fixture equal: True", output.getvalue())
            self.assertIn("gate fixture regression: False", output.getvalue())
            self.assertFalse((state_dir / "state.json").exists())
            digest = fingerprint(list(latest_labels(self.real).values()))
            (state_dir / "state.json").write_text(json.dumps({"corrections": digest}))
            args.dry_run = False
            run(args)
            self.assertIn("REFUSED: no new usable corrections", output.getvalue())

    def bundle(self, name):
        from .publish import check_bundle

        bundle = self.root / "bundles" / name
        bundle.mkdir(parents=True)
        files = {}
        for filename in ["detector.onnx", "embed.onnx", "search.onnx", "arts.json"]:
            (bundle / filename).write_bytes(b"fixture")
            files[filename] = {"sha256": hashlib.sha256(b"fixture").hexdigest()}
        (bundle / "manifest.json").write_text(json.dumps({"version": name, "files": files}))
        check_bundle(bundle)
        return bundle

    def test_local_publish_preserves_corrections_previous_and_guards_current(self):
        from .publish import publish_local

        dest = self.root / "published"
        (dest / "corrections").mkdir(parents=True)
        marker = dest / "corrections" / "labels.jsonl"
        marker.write_text("keep me")
        for name in ["v1", "v2", "v3"]:
            publish_local(self.bundle(name), dest, keep=1)
        self.assertEqual((dest / "current").readlink(), Path("v3"))
        self.assertEqual((dest / "previous").readlink(), Path("v2"))
        self.assertTrue((dest / "v2").exists())
        self.assertFalse((dest / "v1").exists())
        self.assertEqual(marker.read_text(), "keep me")
        with self.assertRaises(SystemExit):
            publish_local(self.bundle("v4"), dest, keep=1, expected_current="stale")
        self.assertEqual((dest / "current").readlink(), Path("v3"))

    def test_remote_shell_preserves_previous_corrections_and_rejects_stale_baseline(self):
        from .publish import remote_script

        dest = self.root / "remote"
        (dest / "corrections").mkdir(parents=True)
        for name in ["v1", "v2", "v3"]:
            bundle = self.bundle(name)
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w:gz") as tar:
                tar.add(bundle, arcname=name)
            subprocess.run(["bash", "-c", remote_script(str(dest), name, 1)], input=stream.getvalue(), check=True, capture_output=True)
            if name == "v2":
                (dest / "current").unlink()
                (dest / "current").symlink_to(dest / "v2")
        self.assertTrue((dest / "corrections").exists())
        self.assertTrue((dest / "previous" / "manifest.json").exists())
        self.assertEqual((dest / "previous").resolve(), dest / "v2")
        self.assertFalse((dest / "v1").exists())
        result = subprocess.run(["bash", "-c", remote_script(str(dest), "v4", 1, "stale")], input=b"", capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((dest / "current").readlink(), Path("v3"))
