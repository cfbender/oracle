"""Ship a bundle (`cardid.export`) to the server that serves it to browsers.

    uv run python -m cardid.publish data/bundles/2026-09-22-full-3 --to nuc:/srv/the-gathering/cardid
    uv run python -m cardid.publish data/bundles/2026-09-22-full-3 --to /mnt/gathering/cardid   # local path

The destination ends up as

    <dest>/<version>/{manifest.json, detector.onnx, embed.onnx, search.onnx, arts.json}
    <dest>/current -> <version>          (symlink, swapped atomically)

so whatever serves `<dest>/current/` (Phoenix from DATA_DIR/cardid, nginx, a bucket sync)
switches versions between two requests, never mid-download, and older versions stay around for
a rollback (`--keep` prunes all but the newest N). Remote destinations go over ssh: a tarball is
streamed to `tar -x` on the host, the manifest's sha256 sums are checked there, then the
symlink is swapped.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from .export import SUMS, sha256, write_sums

REQUIRED = ("manifest.json", "detector.onnx", "embed.onnx", "search.onnx", "arts.json")


def check_bundle(bundle: Path) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text())
    missing = [f for f in REQUIRED if not (bundle / f).exists()]
    if missing:
        raise SystemExit(f"{bundle}: missing {', '.join(missing)}")
    if manifest["version"] != bundle.name:
        raise SystemExit(f"{bundle}: manifest version {manifest['version']!r} does not match the directory name")
    bad = [name for name, f in manifest["files"].items() if sha256(bundle / name) != f["sha256"]]
    if bad:
        raise SystemExit(f"{bundle}: {', '.join(bad)} changed since export; re-export instead of publishing a modified bundle")
    if not (bundle / SUMS).exists():
        write_sums(bundle, manifest)
    return manifest


def remote_script(dest: str, version: str, keep: int | None) -> str:
    """Shell run on the host with the tarball on stdin: unpack aside, check sums, move into
    place, swap the symlink, prune."""
    d, v = shlex.quote(dest), shlex.quote(version)
    # real directories only (`-type d` skips the `current` symlink), newest first
    prune = (
        f"find {d} -mindepth 1 -maxdepth 1 -type d ! -name '.*' -printf '%T@ %p\\n' | sort -rn | tail -n +{keep + 1} | cut -d' ' -f2- | xargs -r rm -rf\n"
        if keep
        else ""
    )
    return (
        "set -euo pipefail\n"
        f"mkdir -p {d}/.incoming && rm -rf {d}/.incoming/{v}\n"
        f"tar -xzf - -C {d}/.incoming\n"
        f"(cd {d}/.incoming/{v} && sha256sum --quiet -c {SUMS})\n"
        f"rm -rf {d}/{v} && mv {d}/.incoming/{v} {d}/{v}\n"
        f"ln -sfn {v} {d}/current.tmp && mv -Tf {d}/current.tmp {d}/current\n"
        f"{prune}"
        f"echo published {v} && ls -l {d}/current"
    )


def publish_remote(bundle: Path, host: str, dest: str, keep: int | None) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / f"{bundle.name}.tar.gz"
        subprocess.run(["tar", "-czf", str(tarball), "-C", str(bundle.parent), bundle.name], check=True)
        print(f"uploading {tarball.stat().st_size / 1e6:.1f} MB to {host}:{dest}/{bundle.name} ...")
        with tarball.open("rb") as f:
            result = subprocess.run(["ssh", host, remote_script(dest, bundle.name, keep)], stdin=f)
    if result.returncode != 0:
        raise SystemExit(f"publish failed on {host} (exit {result.returncode})")


def publish_local(bundle: Path, dest: Path, keep: int | None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    incoming = dest / ".incoming" / bundle.name
    shutil.rmtree(incoming, ignore_errors=True)
    shutil.copytree(bundle, incoming)
    target = dest / bundle.name
    shutil.rmtree(target, ignore_errors=True)
    incoming.rename(target)
    tmp = dest / "current.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(bundle.name)
    tmp.replace(dest / "current")
    if keep:
        versions = sorted(
            (p for p in dest.iterdir() if p.is_dir() and not p.is_symlink() and not p.name.startswith(".")), key=lambda p: p.stat().st_mtime, reverse=True
        )
        for old in versions[keep:]:
            shutil.rmtree(old)
    print(f"published {bundle.name} -> {dest / 'current'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", help="bundle directory from cardid.export")
    ap.add_argument("--to", required=True, help="[user@]host:/path for ssh, or a local directory")
    ap.add_argument("--keep", type=int, default=3, help="versions to keep at the destination (0 keeps all)")
    args = ap.parse_args()

    bundle = Path(args.bundle).resolve()
    manifest = check_bundle(bundle)
    print(f"{bundle.name}: {manifest['gallery']['arts']} arts, {sum(f['bytes'] for f in manifest['files'].values()) / 1e6:.1f} MB")
    if ":" in args.to and not Path(args.to.split(":", 1)[0]).exists():
        host, dest = args.to.split(":", 1)
        publish_remote(bundle, host, dest, args.keep or None)
    else:
        publish_local(bundle, Path(args.to), args.keep or None)


if __name__ == "__main__":
    main()
