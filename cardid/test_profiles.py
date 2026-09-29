"""CARDID_PROFILE keeps each model line's bookkeeping apart on one machine."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from . import profiles
from .envfile import pending


class ProfilesTest(unittest.TestCase):
    def test_state_dir_is_per_profile(self):
        data = Path("/data")
        with patch.dict("os.environ", {}):
            os.environ.pop(profiles.ENV, None)
            self.assertEqual(profiles.state_dir(data), data / "nightly")
        with patch.dict("os.environ", {profiles.ENV: "manavault"}):
            self.assertEqual(profiles.state_dir(data), data / "nightly" / "manavault")
        with patch.dict("os.environ", {profiles.ENV: "../escape"}), self.assertRaises(SystemExit):
            profiles.state_dir(data)

    def test_templates_parse_and_name_their_profile(self):
        root = Path(__file__).resolve().parent.parent / "profiles"
        for name in ("manavault", "the-gathering"):
            with patch.dict("os.environ", {}, clear=True):
                values = pending(root / f"{name}.env.example")
            self.assertEqual(values["CARDID_PROFILE"], name)
            self.assertIn("CARDID_SOURCES", values)
            self.assertIn("CARDID_PUBLISH_TO", values)


if __name__ == "__main__":
    unittest.main()
