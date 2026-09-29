"""GitHub releases as a bundle destination: `--to github:OWNER/REPO`.

ManaVault's server installs scanner bundles from its own GitHub releases, so publishing there
is a release per bundle instead of a copy onto a host:

    uv run python -m cardid.publish data/bundles/<version> --to github:cfbender/manavault

Each bundle becomes a release tagged `<prefix><version>` (`CARDID_RELEASE_PREFIX`, default
`scanner-bundle-`) with every bundle file as an asset. Releases are published but never marked
"latest": apps may read the latest release for their own update checks (ManaVault's Android
shell does). The destination's `current` is the newest published, non-draft release with the
prefix, the same rule the ManaVault updater applies. Rolling back means unpublishing (or
deleting) the newest release. Needs the GitHub CLI (`gh`) logged in with write access.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

PREFIX_ENV = "CARDID_RELEASE_PREFIX"
DEFAULT_PREFIX = "scanner-bundle-"
_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


def release_prefix() -> str:
    return os.environ.get(PREFIX_ENV) or DEFAULT_PREFIX


def github_repo(target: str | None) -> str | None:
    """`OWNER/REPO` for a `github:OWNER/REPO[/current]` target, None for any other target."""
    if not target or not target.startswith("github:"):
        return None
    repo = target.removeprefix("github:").strip("/").removesuffix("/current")
    if not _REPO.fullmatch(repo):
        raise SystemExit(f"invalid GitHub target {target!r}; expected github:OWNER/REPO")
    return repo


def gh(*args: str) -> str:
    """Run the GitHub CLI and return stdout; failures raise CalledProcessError."""
    result = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
    return result.stdout


def releases(repo: str) -> list[dict]:
    return json.loads(gh("api", f"repos/{repo}/releases?per_page=100"))


def current_release(repo: str) -> dict | None:
    """Newest published (non-draft) release whose tag has the bundle prefix."""
    prefix = release_prefix()
    published = [r for r in releases(repo) if not r.get("draft") and (r.get("tag_name") or "").startswith(prefix)]
    return max(published, key=lambda r: r.get("published_at") or "", default=None)


def download_current(repo: str, directory: Path, *patterns: str) -> Path:
    """Download the current release's assets (optionally only `patterns`) into `directory`.

    Raises FileNotFoundError when nothing is published yet, so callers can tell a first
    publication from a connection failure."""
    release = current_release(repo)
    if release is None:
        raise FileNotFoundError(f"no published {release_prefix()}* release in {repo}")
    directory.mkdir(parents=True, exist_ok=True)
    selection = [arg for pattern in patterns for arg in ("--pattern", pattern)]
    gh("release", "download", release["tag_name"], "--repo", repo, "--dir", str(directory), "--clobber", *selection)
    return directory


def check_repo(repo: str) -> None:
    """Refuse early (before training) when the repository is unreachable or read-only."""
    try:
        permission = json.loads(gh("repo", "view", repo, "--json", "viewerPermission"))["viewerPermission"]
    except (subprocess.CalledProcessError, OSError, ValueError, KeyError) as error:
        raise SystemExit(f"publish destination github:{repo}: cannot read the repository ({error}); check `gh auth status`") from None
    if permission not in {"ADMIN", "MAINTAIN", "WRITE"}:
        raise SystemExit(f"publish destination github:{repo}: {permission or 'no'} permission; releases need write access")


def publish_github(bundle: Path, repo: str, expected_current: str | None = None) -> None:
    from .export import SUMS
    from .workflow import sha256

    manifest = json.loads((bundle / "manifest.json").read_text())
    tag = release_prefix() + bundle.name
    if expected_current:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            try:
                download_current(repo, Path(tmp), "manifest.json")
            except FileNotFoundError:
                raise SystemExit("published bundle changed during evaluation (none is published now); refusing to publish") from None
            if sha256(Path(tmp) / "manifest.json") != expected_current:
                raise SystemExit("published bundle changed during evaluation; refusing to publish")
    if any(r.get("tag_name") == tag for r in releases(repo)):
        raise SystemExit(f"release {tag} already exists in {repo}; export with a new --version")

    names = ["manifest.json", SUMS, *sorted(manifest["files"])]
    files = [str(bundle / name) for name in names]
    arts = manifest.get("gallery", {}).get("arts")
    notes = f"Card recognition bundle `{bundle.name}` ({arts} artworks), exported by cardid."
    print(f"creating release {tag} in {repo} ({len(files)} assets) ...", flush=True)
    gh("release", "create", tag, *files, "--repo", repo, "--title", f"Scanner bundle {bundle.name}", "--notes", notes, "--latest=false")

    # gh uploads asset by asset; confirm the release holds every file at its exact size.
    uploaded = {a["name"]: a["size"] for a in json.loads(gh("release", "view", tag, "--repo", repo, "--json", "assets"))["assets"]}
    wrong = [name for name in names if uploaded.get(name) != (bundle / name).stat().st_size]
    if wrong:
        raise SystemExit(
            f"release {tag} is incomplete ({', '.join(wrong)}); delete it with `gh release delete {tag} --repo {repo} --cleanup-tag` and publish again"
        )
    print(f"published {bundle.name} -> github:{repo} ({tag})")
