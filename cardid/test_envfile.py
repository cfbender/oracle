"""The desktop env file is parsed as literal data by retrain and nightly.sh, never executed."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import ML_DIR
from .envfile import load_env, parse_env, pending

HOSTILE = """\
# comment line
export CARDID_SERVER=https://file.test   # trailing comment
CARDID_PUBLISH_TO='host:/path with spaces'
CARDID_CORRECTIONS_TOKEN="$(touch {marker})"
CARDID_CHECKPOINT='`touch {marker}`'
CARDID_DETECTOR=$HOME/det.pt
CARDID_EMPTY=
OTHER_VARIABLE=ignored
"""


class EnvFileTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.marker = self.root / "pwned"
        self.env = self.root / "cardid.env"
        self.env.write_text(HOSTILE.format(marker=self.marker))
        self.env.chmod(0o600)

    def test_values_are_literal(self):
        self.assertEqual(
            parse_env(self.env),
            {
                "CARDID_SERVER": "https://file.test",
                "CARDID_PUBLISH_TO": "host:/path with spaces",
                "CARDID_CORRECTIONS_TOKEN": f"$(touch {self.marker})",
                "CARDID_CHECKPOINT": f"`touch {self.marker}`",
                "CARDID_DETECTOR": "$HOME/det.pt",
                "CARDID_EMPTY": "",
                "OTHER_VARIABLE": "ignored",
            },
        )
        self.assertFalse(self.marker.exists())

    def test_only_unset_cardid_keys_are_applied(self):
        values = pending(self.env, {"CARDID_SERVER": "https://environment.test"})
        self.assertNotIn("CARDID_SERVER", values)
        self.assertNotIn("OTHER_VARIABLE", values)
        self.assertEqual(values["CARDID_PUBLISH_TO"], "host:/path with spaces")
        self.assertEqual(pending(self.root / "absent.env"), {})
        with patch.dict(os.environ, {"CARDID_SERVER": "https://environment.test"}, clear=True):
            load_env(self.env)
            self.assertEqual(os.environ["CARDID_SERVER"], "https://environment.test")
            self.assertEqual(os.environ["CARDID_DETECTOR"], "$HOME/det.pt")
            self.assertNotIn("OTHER_VARIABLE", os.environ)

    def test_malformed_lines_name_the_line(self):
        for text, message in (
            ("CARDID_A=1\nCARDID_B=1 CARDID_C=2\n", ":2: expected KEY=value"),
            ("echo hi\n", ":1: expected KEY=value"),
            ("CARDID_A='unterminated\n", ":1: No closing quotation"),
            ("1BAD=x\n", "invalid variable name"),
            ("CARDID-A=x\n", "invalid variable name"),
            ("CARDID_A=$(rm -rf ~) b\n", "expected KEY=value"),
        ):
            with self.subTest(text=text):
                self.env.write_text(text)
                with self.assertRaisesRegex(SystemExit, message):
                    parse_env(self.env)

    def test_world_readable_file_warns(self):
        self.env.chmod(0o644)
        with patch("sys.stderr") as stderr:
            pending(self.env, {})
        self.assertIn("chmod 600", "".join(call.args[0] for call in stderr.write.call_args_list))


class NightlyScriptTest(unittest.TestCase):
    """Run the real nightly.sh from a scratch copy with a `uv` shim: the env step runs the
    real parser; the training step just reports the environment it would have received."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        shutil.copy(ML_DIR / "nightly.sh", self.root / "nightly.sh")
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        uv = bin_dir / "uv"
        uv.write_text(
            "#!/bin/sh\n"
            "shift 3  # run --no-sync python\n"
            'case " $* " in *" cardid.nightly "*) env | grep "^CARDID_" | sort; exit 0;; esac\n'
            f'PYTHONPATH={ML_DIR} exec {sys.executable} "$@"\n'
        )
        uv.chmod(0o755)
        self.marker = self.root / "pwned"
        self.env_file = self.root / "cardid.env"
        self.env_file.write_text(HOSTILE.format(marker=self.marker))
        self.env_file.chmod(0o600)
        self.environ = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(self.root), "CARDID_ENV_FILE": str(self.env_file), "CARDID_BUDGET": "60s"}

    def run_script(self, **extra) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(self.root / "nightly.sh")], env={**self.environ, **extra}, capture_output=True, text=True, timeout=60)

    def test_env_file_is_parsed_not_sourced(self):
        result = self.run_script(CARDID_SERVER="https://environment.test")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = set(result.stdout.splitlines())
        self.assertIn("CARDID_SERVER=https://environment.test", lines)  # the environment wins
        self.assertIn("CARDID_PUBLISH_TO=host:/path with spaces", lines)
        self.assertIn(f"CARDID_CORRECTIONS_TOKEN=$(touch {self.marker})", lines)
        self.assertIn("CARDID_DETECTOR=$HOME/det.pt", lines)
        self.assertFalse(self.marker.exists())

    def test_malformed_env_file_aborts_before_training(self):
        self.env_file.write_text("CARDID_SERVER=https://x.test\ntouch pwned\n")
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(":2: expected KEY=value", result.stderr)
        self.assertNotIn("CARDID_SERVER", result.stdout)
        self.assertFalse((self.root / "data").exists())


if __name__ == "__main__":
    unittest.main()
