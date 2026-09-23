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
import fcntl
import json
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from .export import SUMS, sha256, write_sums
from .workflow import remote_target

REQUIRED = ("manifest.json", "detector.onnx", "embed.onnx", "search.onnx", "arts.json")


def check_bundle(bundle: Path) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text())
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", bundle.name) or bundle.name in {"current", "previous", "corrections"}:
        raise SystemExit("invalid bundle version")
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


def remote_script(dest: str, version: str, keep: int | None, expected_current: str | None = None) -> str:
    """Shell run on the host with the tarball on stdin: unpack aside, check sums, move into
    place, swap the symlink, prune."""
    d, v = shlex.quote(dest), shlex.quote(version)
    # Only bundle directories, never corrections or either protected symlink target.
    prune = (
        f"find {d} -mindepth 2 -maxdepth 2 -name manifest.json -printf '%T@ %h\\n' | sort -rn | tail -n +{keep + 1} | cut -d' ' -f2- | "
        f'while IFS= read -r old; do [ "$(readlink -f "$old")" = "$(readlink -f {d}/current)" ] || '
        f'[ "$(readlink -f "$old")" = "$(readlink -f {d}/previous || true)" ] || rm -rf -- "$old"; done\n'
        if keep
        else ""
    )
    guard = (
        f"test \"$(sha256sum {d}/current/manifest.json | cut -d' ' -f1)\" = {shlex.quote(expected_current)} || "
        f'{{ echo "publish: {d}/current changed since evaluation; refusing to publish" >&2; exit 1; }}\n'
        if expected_current
        else ""
    )
    return (
        "set -euo pipefail\n"
        f"mkdir -p {d}\nexec 9>{d}/.publish.lock\nflock -x 9\n"
        f"{guard}"
        f'test ! -e {d}/{v} || {{ echo "publish: {d}/{v} already exists; export with a new --version or remove it" >&2; exit 1; }}\n'
        f"mkdir -p {d}/.incoming && rm -rf {d}/.incoming/{v}\n"
        f"tar -xzf - -C {d}/.incoming\n"
        f"(cd {d}/.incoming/{v} && sha256sum --quiet -c {SUMS})\n"
        f"mv {d}/.incoming/{v} {d}/{v}\n"
        f'if [ -L {d}/current ]; then ln -sfn "$(readlink {d}/current)" {d}/previous.tmp && mv -Tf {d}/previous.tmp {d}/previous; '
        f'elif [ -d {d}/current ]; then echo "publish: {d}/current must be a symlink; move the copied directory to {d}/<version> and symlink it" >&2; exit 1; fi\n'
        f"ln -sfn {v} {d}/current.tmp && mv -Tf {d}/current.tmp {d}/current\n"
        f"{prune}"
        f"echo published {v} && ls -l {d}/current"
    )


def publish_remote(bundle: Path, host: str, dest: str, keep: int | None, expected_current: str | None = None) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / f"{bundle.name}.tar.gz"
        subprocess.run(["tar", "-czf", str(tarball), "-C", str(bundle.parent), bundle.name], check=True)
        print(f"uploading {tarball.stat().st_size / 1e6:.1f} MB to {host}:{dest}/{bundle.name} ...")
        # Run under bash explicitly: the script relies on pipefail and `{ ...; }` groups, which the
        # remote user's login shell (dash, fish, ...) may not accept.
        script = f"bash -c {shlex.quote(remote_script(dest, bundle.name, keep, expected_current))}"
        with tarball.open("rb") as f:
            result = subprocess.run(["ssh", host, script], stdin=f)
    if result.returncode != 0:
        raise SystemExit(f"publish failed on {host} (exit {result.returncode}); see the message above from the remote shell")


def publish_local(bundle: Path, dest: Path, keep: int | None, expected_current: str | None = None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / ".publish.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if expected_current and sha256(dest / "current" / "manifest.json") != expected_current:
            raise SystemExit("published bundle changed during evaluation; refusing to publish")
        _publish_local(bundle, dest, keep)


def _publish_local(bundle: Path, dest: Path, keep: int | None) -> None:
    current = dest / "current"
    if current.exists() and not current.is_symlink():
        raise SystemExit("current must be a symlink; migrate the copied directory before publishing")
    if (dest / bundle.name).exists():
        raise SystemExit("version already exists; export with a new --version")
    incoming = dest / ".incoming" / bundle.name
    shutil.rmtree(incoming, ignore_errors=True)
    shutil.copytree(bundle, incoming)
    target = dest / bundle.name
    incoming.rename(target)
    if current.is_symlink():
        previous = dest / "previous.tmp"
        previous.unlink(missing_ok=True)
        previous.symlink_to(current.readlink())
        previous.replace(dest / "previous")
    tmp = dest / "current.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(bundle.name)
    tmp.replace(dest / "current")
    if keep:
        versions = sorted(
            (p for p in dest.iterdir() if p.is_dir() and not p.is_symlink() and (p / "manifest.json").exists()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        protected = {current.resolve(), (dest / "previous").resolve()}
        for old in versions[keep:]:
            if old.resolve() not in protected:
                shutil.rmtree(old)
    print(f"published {bundle.name} -> {dest / 'current'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", help="bundle directory from cardid.export")
    ap.add_argument("--to", required=True, help="[user@]host:/path for ssh, or a local directory")
    ap.add_argument("--keep", type=int, default=3, help="versions to keep at the destination (0 keeps all)")
    ap.add_argument("--expected-current", help="refuse if current manifest SHA256 differs (nightly evaluation guard)")
    args = ap.parse_args()

    bundle = Path(args.bundle).resolve()
    manifest = check_bundle(bundle)
    print(f"{bundle.name}: {manifest['gallery']['arts']} arts, {sum(f['bytes'] for f in manifest['files'].values()) / 1e6:.1f} MB")
    remote = remote_target(args.to)
    if remote:
        publish_remote(bundle, *remote, args.keep or None, args.expected_current)
    else:
        publish_local(bundle, Path(args.to), args.keep or None, args.expected_current)


if __name__ == "__main__":
    main()
