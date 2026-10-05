"""Make an app's currently published checkpoints the committed starting weights in models/.

    mise run promote-models                         # The Gathering's published pair
    mise run promote-models -- --profile manavault  # ManaVault's (phone-tuned detector)

Reads data/nightly/<profile>/state.json, which retrain and nightly write only for a pair that
was published, copies both checkpoints into models/ and rewrites models/SHA256SUMS. Commit
the result yourself; the printed commit message names the bundle the weights came from.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from . import DATA_DIR, ML_DIR
from .workflow import sha256

FILES = {"recogniser": "recogniser.pt", "detector": "detector.pt"}


def published_version(bundles: Path, hashes: dict[str, str]) -> str | None:
    """The local bundle exported from exactly these checkpoints, if it is still on disk."""
    for manifest_path in sorted(bundles.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if all(manifest.get(kind, {}).get("sha256") == digest for kind, digest in hashes.items()):
            return manifest.get("version", manifest_path.parent.name)
    return None


def promote(profile: str, data: Path = DATA_DIR, models: Path = ML_DIR / "models") -> str:
    state_path = data / "nightly" / profile / "state.json"
    if not state_path.is_file():
        raise SystemExit(f"{state_path} not found: {profile} has never published from this machine")
    state = json.loads(state_path.read_text())
    # state.json names the recogniser "checkpoint".
    sources = {kind: Path(state.get("checkpoint" if kind == "recogniser" else kind) or "<unset>") for kind in FILES}
    for kind, path in sources.items():
        if not path.is_file():
            raise SystemExit(f"{kind}: published checkpoint {path} from {state_path} is missing")
    hashes = {kind: sha256(path) for kind, path in sources.items()}
    version = published_version(data / "bundles", hashes)
    if version is None:
        print("WARNING: no local bundle in data/bundles matches these checkpoints; the version is unknown", flush=True)
    for kind, path in sources.items():
        target = models / FILES[kind]
        if target.is_file() and sha256(target) == hashes[kind]:
            print(f"{kind}: unchanged ({path})", flush=True)
        else:
            shutil.copyfile(path, target)
            print(f"{kind}: {path} -> {target}", flush=True)
    (models / "SHA256SUMS").write_text("".join(f"{hashes[kind]}  {name}\n" for kind, name in FILES.items()))
    message = f"feat(models): promote {profile} bundle {version or 'checkpoints'} to starting weights"
    print(f"\nnext: git add models && git commit -m {json.dumps(message)}", flush=True)
    return message


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default="the-gathering", help="app profile whose published pair to promote (default: the-gathering)")
    args = parser.parse_args()
    promote(args.profile)


if __name__ == "__main__":
    main()
