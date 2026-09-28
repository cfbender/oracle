"""GitHub release publishing (`--to github:OWNER/REPO`) against an in-memory fake `gh`."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from . import github
from .export import SUMS, sha256
from .publish import check_bundle
from .workflow import check_destination, find_manifest, remote_target, snapshot_bundle

FILES = ("detector.onnx", "embed.onnx", "search.onnx", "arts.json", "printings.json")


class FakeGitHub:
    """Just enough of `gh api`, `gh release create/view/download` and `gh repo view`."""

    def __init__(self, permission: str = "WRITE"):
        self.releases: list[dict] = []
        self.assets: dict[str, dict[str, bytes]] = {}
        self.permission = permission
        self.drop: set[str] = set()  # asset names a flaky upload loses
        self.calls: list[tuple[str, ...]] = []

    def release(self, tag: str, published_at: str, files: dict[str, bytes], draft: bool = False) -> None:
        self.releases.append({"tag_name": tag, "draft": draft, "published_at": None if draft else published_at})
        self.assets[tag] = files

    def __call__(self, *args: str) -> str:
        self.calls.append(args)
        if args[:2] == ("repo", "view"):
            return json.dumps({"viewerPermission": self.permission})
        if args[0] == "api":
            return json.dumps(self.releases)
        if args[:2] == ("release", "create"):
            tag = args[2]
            files = [a for a in args[3 : args.index("--repo")]]
            self.release(tag, "2026-10-01T00:00:00Z", {Path(f).name: Path(f).read_bytes() for f in files if Path(f).name not in self.drop})
            return ""
        if args[:2] == ("release", "view"):
            return json.dumps({"assets": [{"name": n, "size": len(b)} for n, b in self.assets[args[2]].items()]})
        if args[:2] == ("release", "download"):
            tag = args[2]
            directory = Path(args[args.index("--dir") + 1])
            patterns = [args[i + 1] for i, a in enumerate(args) if a == "--pattern"]
            for name, body in self.assets[tag].items():
                if not patterns or name in patterns:
                    (directory / name).write_bytes(body)
            return ""
        raise AssertionError(f"unexpected gh call {args}")


class GitHubPublishTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.enterContext(redirect_stdout(io.StringIO()))
        self.gh = FakeGitHub()
        self.enterContext(patch.object(github, "gh", self.gh))

    def bundle(self, name: str) -> Path:
        bundle = self.root / "bundles" / name
        bundle.mkdir(parents=True)
        files = {}
        for filename in FILES:
            (bundle / filename).write_bytes(f"{name}:{filename}".encode())
            files[filename] = {"bytes": (bundle / filename).stat().st_size, "sha256": sha256(bundle / filename)}
        (bundle / "manifest.json").write_text(json.dumps({"version": name, "gallery": {"arts": 1}, "files": files}))
        check_bundle(bundle)  # writes SHA256SUMS
        return bundle

    def published(self, bundle: Path, published_at: str) -> None:
        names = ["manifest.json", SUMS, *FILES]
        self.gh.release("scanner-bundle-" + bundle.name, published_at, {n: (bundle / n).read_bytes() for n in names})

    def test_targets_are_recognised_and_never_treated_as_ssh(self):
        self.assertEqual(github.github_repo("github:cfbender/manavault"), "cfbender/manavault")
        self.assertEqual(github.github_repo("github:cfbender/manavault/current"), "cfbender/manavault")
        self.assertIsNone(github.github_repo("nuc:/srv/cardid"))
        self.assertIsNone(remote_target("github:cfbender/manavault"))
        with self.assertRaises(SystemExit):
            github.github_repo("github:not a repo")

    def test_publish_creates_a_non_latest_release_with_every_file(self):
        bundle = self.bundle("v2")
        github.publish_github(bundle, "cfbender/manavault")
        create = next(c for c in self.gh.calls if c[:2] == ("release", "create"))
        self.assertEqual(create[2], "scanner-bundle-v2")
        self.assertIn("--latest=false", create)
        self.assertEqual(sorted(self.gh.assets["scanner-bundle-v2"]), sorted(["manifest.json", SUMS, *FILES]))

    def test_release_prefix_is_configurable(self):
        with patch.dict("os.environ", {github.PREFIX_ENV: "cardid-"}):
            github.publish_github(self.bundle("v3"), "cfbender/the-gathering")
        self.assertIn("cardid-v3", self.gh.assets)

    def test_refuses_an_existing_version_and_an_incomplete_upload(self):
        bundle = self.bundle("v4")
        self.published(bundle, "2026-09-01T00:00:00Z")
        with self.assertRaisesRegex(SystemExit, "already exists"):
            github.publish_github(bundle, "cfbender/manavault")
        self.gh.drop = {"search.onnx"}
        with self.assertRaisesRegex(SystemExit, "incomplete.*search.onnx"):
            github.publish_github(self.bundle("v5"), "cfbender/manavault")

    def test_current_is_the_newest_published_prefixed_release(self):
        old, new = self.bundle("old"), self.bundle("new")
        self.published(old, "2026-09-01T00:00:00Z")
        self.published(new, "2026-09-20T00:00:00Z")
        self.gh.release("scanner-bundle-draft", "", {}, draft=True)
        self.gh.release("v1.3.0", "2026-09-30T00:00:00Z", {})
        self.assertEqual(github.current_release("cfbender/manavault")["tag_name"], "scanner-bundle-new")

    def test_expected_current_guards_against_a_concurrent_publish(self):
        current = self.bundle("live")
        self.published(current, "2026-09-01T00:00:00Z")
        with self.assertRaisesRegex(SystemExit, "changed during evaluation"):
            github.publish_github(self.bundle("candidate"), "cfbender/manavault", expected_current="0" * 64)
        github.publish_github(self.bundle("candidate2"), "cfbender/manavault", expected_current=sha256(current / "manifest.json"))
        self.assertIn("scanner-bundle-candidate2", self.gh.assets)

    def test_retrain_helpers_read_manifest_and_snapshot_from_the_release(self):
        target = "github:cfbender/manavault"
        work = self.root / "work"
        work.mkdir()
        # First publication: nothing released yet is "missing", so a local bundle is the fallback.
        local = self.bundle("local")
        path, source = find_manifest(target, local.parent, work, require=True)
        self.assertEqual(source, str(local))

        self.published(local, "2026-09-01T00:00:00Z")
        path, source = find_manifest(target, local.parent, work, require=True)
        self.assertEqual(source, target + "/current")
        self.assertEqual(json.loads(path.read_text())["version"], "local")

        baseline, manifest, digest = snapshot_bundle(source, self.root / "snap-root")
        self.assertEqual(baseline.name, "local")
        self.assertEqual(digest, sha256(local / "manifest.json"))

    def test_destination_check_requires_write_access(self):
        check_destination("github:cfbender/manavault")
        self.gh.permission = "READ"
        with self.assertRaisesRegex(SystemExit, "write access"):
            check_destination("github:cfbender/manavault")

    def test_unreachable_repository_fails_early(self):
        def failing(*args):
            raise subprocess.CalledProcessError(1, ["gh", *args])

        with patch.object(github, "gh", failing), self.assertRaisesRegex(SystemExit, "cannot read"):
            check_destination("github:cfbender/manavault")


if __name__ == "__main__":
    unittest.main()
