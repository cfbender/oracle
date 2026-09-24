"""Local publication: bundle checks, atomic `current`/`previous` swaps and pruning."""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from . import publish
from .export import SUMS, sha256
from .publish import check_bundle, publish_local

FILES = ("detector.onnx", "embed.onnx", "search.onnx", "arts.json", "printings.json")


class LocalPublishTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.dest = self.root / "published"
        self.clock = 1_700_000_000
        self.enterContext(redirect_stdout(io.StringIO()))

    def bundle(self, name: str) -> Path:
        bundle = self.root / "bundles" / name
        bundle.mkdir(parents=True)
        files = {}
        for filename in FILES:
            (bundle / filename).write_bytes(f"{name}:{filename}".encode())
            files[filename] = {"bytes": (bundle / filename).stat().st_size, "sha256": sha256(bundle / filename)}
        (bundle / "manifest.json").write_text(json.dumps({"version": name, "gallery": {"arts": 1}, "files": files}))
        return bundle

    def publish(self, name: str, keep: int | None, **kwargs) -> None:
        bundle = self.bundle(name)
        check_bundle(bundle)
        # copytree keeps the source directory's mtime, which is what pruning orders by;
        # step it explicitly so the order does not depend on filesystem timestamp resolution.
        self.clock += 60
        os.utime(bundle, (self.clock, self.clock))
        publish_local(bundle, self.dest, keep, **kwargs)

    def versions(self) -> list[str]:
        return sorted(p.name for p in self.dest.iterdir() if p.is_dir() and not p.is_symlink() and (p / "manifest.json").exists())

    def test_publish_swaps_current_and_previous_and_prunes_old_versions(self):
        (self.dest / "corrections").mkdir(parents=True)
        (self.dest / "corrections" / "labels.jsonl").write_text("keep")
        (self.dest / "notes.txt").write_text("not a bundle")
        for name in ("v1", "v2", "v3", "v4"):
            self.publish(name, keep=2)
        self.assertEqual((self.dest / "current").readlink(), Path("v4"))
        self.assertEqual((self.dest / "previous").readlink(), Path("v3"))
        self.assertEqual(self.versions(), ["v3", "v4"])
        self.assertEqual((self.dest / "corrections" / "labels.jsonl").read_text(), "keep")
        self.assertEqual((self.dest / "notes.txt").read_text(), "not a bundle")
        self.assertEqual(list((self.dest / ".incoming").iterdir()), [])
        self.assertFalse((self.dest / "current.tmp").exists() or (self.dest / "previous.tmp").is_symlink())
        for filename in FILES:
            self.assertEqual((self.dest / "current" / filename).read_bytes(), f"v4:{filename}".encode())
        self.assertTrue((self.dest / "current" / SUMS).exists())

    def test_current_and_previous_survive_pruning_even_when_oldest(self):
        for name in ("v1", "v2"):
            self.publish(name, keep=None)
        # Make the live versions the oldest by mtime; pruning must still protect them.
        for name in ("v1", "v2"):
            os.utime(self.dest / name, (1, 1))
        for name in ("v3", "v4", "v5"):
            self.clock += 60
            bundle = self.bundle(name)
            os.utime(bundle, (self.clock, self.clock))
            publish_local(bundle, self.dest, None)
        (self.dest / "previous").unlink()
        (self.dest / "previous").symlink_to("v1")
        (self.dest / "current").unlink()
        (self.dest / "current").symlink_to("v2")
        self.publish("v6", keep=1)  # previous becomes v2; v1 is no longer protected
        self.assertEqual(self.versions(), ["v2", "v6"])
        self.assertEqual((self.dest / "previous").readlink(), Path("v2"))

    def test_keep_none_keeps_every_version(self):
        for name in ("a1", "a2", "a3", "a4"):
            self.publish(name, keep=None)
        self.assertEqual(self.versions(), ["a1", "a2", "a3", "a4"])

    def test_refusals_leave_current_untouched(self):
        self.publish("v1", keep=3)
        with self.assertRaisesRegex(SystemExit, "already exists"):
            publish_local(self.root / "bundles" / "v1", self.dest, 3)
        with self.assertRaisesRegex(SystemExit, "changed during evaluation"):
            self.publish("v2", keep=3, expected_current="0" * 64)
        self.assertEqual((self.dest / "current").readlink(), Path("v1"))
        self.assertEqual(self.versions(), ["v1"])
        self.publish("v3", keep=3, expected_current=sha256(self.dest / "current" / "manifest.json"))
        self.assertEqual((self.dest / "current").readlink(), Path("v3"))
        broken = self.bundle("v4")
        with patch.object(publish.shutil, "copytree", side_effect=OSError("disk full")), self.assertRaises(OSError):
            publish_local(broken, self.dest, 3)
        self.assertEqual((self.dest / "current").readlink(), Path("v3"))
        self.assertFalse((self.dest / "v4").exists())

    def test_plain_directory_current_is_refused(self):
        (self.dest / "current").mkdir(parents=True)
        (self.dest / "current" / "manifest.json").write_text("{}")
        with self.assertRaisesRegex(SystemExit, "current must be a symlink"):
            self.publish("v1", keep=3)
        self.assertFalse((self.dest / "v1").exists())

    def test_check_bundle_rejects_bad_bundles(self):
        tampered = self.bundle("tampered")
        (tampered / "embed.onnx").write_bytes(b"changed")
        with self.assertRaisesRegex(SystemExit, "embed.onnx changed since export"):
            check_bundle(tampered)
        missing = self.bundle("missing")
        (missing / "search.onnx").unlink()
        with self.assertRaisesRegex(SystemExit, "missing search.onnx"):
            check_bundle(missing)
        renamed = self.bundle("renamed")
        manifest = json.loads((renamed / "manifest.json").read_text())
        (renamed / "manifest.json").write_text(json.dumps({**manifest, "version": "other"}))
        with self.assertRaisesRegex(SystemExit, "does not match the directory name"):
            check_bundle(renamed)
        for name in ("current", "previous", "corrections", ".hidden"):
            with self.subTest(name=name), self.assertRaisesRegex(SystemExit, "invalid bundle version"):
                check_bundle(self.bundle(name))

    def test_cli_publishes_to_a_local_directory(self):
        bundle = self.bundle("cli-v1")
        with patch("sys.argv", ["publish", str(bundle), "--to", str(self.dest), "--keep", "0"]):
            publish.main()
        self.assertEqual((self.dest / "current").readlink(), Path("cli-v1"))
        self.assertTrue((bundle / SUMS).exists())


if __name__ == "__main__":
    unittest.main()
