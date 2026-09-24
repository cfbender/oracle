"""The desktop env file (~/.config/cardid.env): literal KEY=value lines, never executed.

    python -m cardid.envfile ~/.config/cardid.env   # NUL-separated KEY=value for nightly.sh

Each non-blank line is one `KEY=value` assignment, split with POSIX shell quoting rules
(`shlex`) but with no expansion of any kind: `$VAR`, `$(...)` and backticks stay literal
text. `#` starts a comment; an `export ` prefix is tolerated. Only `CARDID_*` keys are
applied, and a variable already in the environment wins over the file. `retrain` and
`nightly.sh` both read the file through this module, so it is parsed the same way everywhere.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import stat
import sys
from pathlib import Path

KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
PREFIX = "CARDID_"


def parse_env(path: Path) -> dict[str, str]:
    """Every assignment in the file, in order; SystemExit naming the line on anything else."""
    values: dict[str, str] = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        try:
            words = shlex.split(line, comments=True)
        except ValueError as error:
            raise SystemExit(f"{path}:{number}: {error} (quote values on one line; no shell expansion)") from None
        if words and words[0] == "export":
            words = words[1:]
        if not words:
            continue
        if len(words) != 1 or "=" not in words[0]:
            raise SystemExit(f"{path}:{number}: expected KEY=value (quote spaces; no shell expansion)")
        key, value = words[0].split("=", 1)
        if not KEY.fullmatch(key) or "\0" in value:
            raise SystemExit(f"{path}:{number}: invalid variable name {key!r}")
        values[key] = value
    return values


def pending(path: Path, environ=os.environ) -> dict[str, str]:
    """The file's `CARDID_*` values that the environment does not already set."""
    if not Path(path).exists():
        return {}
    warn_if_shared(Path(path))
    return {k: v for k, v in parse_env(path).items() if k.startswith(PREFIX) and k not in environ}


def load_env(path: Path) -> None:
    """Apply the file to `os.environ` (existing variables win)."""
    os.environ.update(pending(path))


def warn_if_shared(path: Path) -> None:
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        print(f"WARNING: {path} is accessible by other users (mode {stat.filemode(mode)}); it holds a token, chmod 600 it", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the env file's unset CARDID_* assignments, NUL-separated (for nightly.sh).")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    out = sys.stdout.buffer
    for key, value in pending(args.path).items():
        out.write(f"{key}={value}\0".encode())
    out.flush()


if __name__ == "__main__":
    main()
