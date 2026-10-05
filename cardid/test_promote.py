"""promote copies an app's published pair into models/ and rewrites SHA256SUMS."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from . import promote
from .workflow import sha256


class PromoteTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.data, self.models = root / "data", root / "models"
        self.models.mkdir()
        (self.models / "recogniser.pt").write_bytes(b"old recogniser")
        (self.models / "detector.pt").write_bytes(b"same detector")
        run = self.data / "runs" / "retrain-1"
        run.mkdir(parents=True)
        (run / "best.pt").write_bytes(b"new recogniser")
        (self.data / "runs" / "det.pt").write_bytes(b"same detector")
        state = self.data / "nightly" / "the-gathering" / "state.json"
        state.parent.mkdir(parents=True)
        state.write_text(json.dumps({"checkpoint": str(run / "best.pt"), "detector": str(self.data / "runs" / "det.pt")}))
        self.enterContext(redirect_stdout(io.StringIO()))

    def test_copies_the_published_pair_and_names_its_bundle(self):
        bundle = self.data / "bundles" / "retrain-1"
        bundle.mkdir(parents=True)
        hashes = {kind: {"sha256": sha256(self.data / "runs" / path)} for kind, path in [("recogniser", "retrain-1/best.pt"), ("detector", "det.pt")]}
        (bundle / "manifest.json").write_text(json.dumps({"version": "retrain-1", **hashes}))
        message = promote.promote("the-gathering", self.data, self.models)
        self.assertIn("the-gathering bundle retrain-1", message)
        self.assertEqual((self.models / "recogniser.pt").read_bytes(), b"new recogniser")
        sums = (self.models / "SHA256SUMS").read_text().splitlines()
        self.assertEqual(sums, [f"{hashes['recogniser']['sha256']}  recogniser.pt", f"{hashes['detector']['sha256']}  detector.pt"])

    def test_refuses_a_profile_that_never_published(self):
        with self.assertRaisesRegex(SystemExit, "never published"):
            promote.promote("manavault", self.data, self.models)
        self.assertEqual((self.models / "recogniser.pt").read_bytes(), b"old recogniser")


if __name__ == "__main__":
    unittest.main()
