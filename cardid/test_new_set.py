"""The new-set driver: one gallery refresh, then a gallery-only retrain per app profile."""

from __future__ import annotations

import io
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from . import new_set


class NewSetTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = self.root / "config"
        self.config.mkdir()
        for name in ("the-gathering", "manavault"):
            (self.config / f"{name}.env").write_text(f"CARDID_PROFILE={name}\n")
        self.enterContext(patch.object(new_set, "DATA_DIR", self.root / "data"))
        self.enterContext(redirect_stdout(io.StringIO()))
        self.calls = []
        self.failing = set()

    def runner(self, cmd, cwd):
        self.calls.append(cmd[1:])
        failed = any(part.endswith(f"{name}.env") for part in cmd for name in self.failing) or ("scryfall" in self.failing and "cardid.scryfall" in cmd)
        return subprocess.CompletedProcess(cmd, 1 if failed else 0)

    def test_refreshes_the_gallery_once_then_every_profile_and_keeps_going(self):
        files = new_set.profile_files(self.config, [])
        self.assertEqual([f.stem for f in files], ["manavault", "the-gathering"])
        self.failing = {"manavault"}
        results = new_set.run_all(files, ["--force"], runner=self.runner)
        self.assertEqual(results, {"manavault": False, "the-gathering": True})
        self.assertEqual(self.calls[0], ["-m", "cardid.scryfall", "--update"])
        for call, path in zip(self.calls[1:], files, strict=True):
            self.assertEqual(call, ["-m", "cardid.retrain", "--env-file", str(path), "--gallery-only", "--no-update-gallery", "--force"])

    def test_train_runs_a_full_retrain_per_profile(self):
        files = new_set.profile_files(self.config, [])
        new_set.run_all(files, [], runner=self.runner, train=True)
        self.assertEqual(self.calls[0], ["-m", "cardid.scryfall", "--update"])
        for call, path in zip(self.calls[1:], files, strict=True):
            self.assertEqual(call, ["-m", "cardid.retrain", "--env-file", str(path), "--no-update-gallery"])

    def test_a_failed_gallery_update_publishes_nothing(self):
        self.failing = {"scryfall"}
        with self.assertRaisesRegex(SystemExit, "gallery update failed"):
            new_set.run_all(new_set.profile_files(self.config, []), [], runner=self.runner)
        self.assertEqual(len(self.calls), 1)

    def test_dry_runs_plan_without_downloading_and_names_select_profiles(self):
        files = new_set.profile_files(self.config, ["the-gathering"])
        new_set.run_all(files, ["--dry-run"], runner=self.runner)
        self.assertEqual(self.calls, [["-m", "cardid.retrain", "--env-file", str(files[0]), "--gallery-only", "--no-update-gallery", "--dry-run"]])
        with self.assertRaisesRegex(SystemExit, "no such profile"):
            new_set.profile_files(self.config, ["typo"])


if __name__ == "__main__":
    unittest.main()
