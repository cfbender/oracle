"""CARDID_SOURCES keeps each app's model line on its own real captures."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import sources
from .retrain import usable_rows

ROWS = [
    {"capture_id": "00000000-0000-0000-0000-00000000000a", "label": "x", "source": "webcam-table"},
    {"capture_id": "00000000-0000-0000-0000-00000000000b", "label": "x", "source": "manavault-scanner"},
    {"capture_id": "00000000-0000-0000-0000-00000000000c", "label": "x"},
]


class SourcesTest(unittest.TestCase):
    def test_unset_keeps_everything_and_a_list_selects(self):
        with patch.dict("os.environ", {}):
            os.environ.pop(sources.ENV, None)
            self.assertEqual(sources.keep(ROWS), ROWS)
        with patch.dict("os.environ", {sources.ENV: "manavault-scanner"}):
            self.assertEqual([r["source"] for r in sources.keep(ROWS)], ["manavault-scanner"])
        with patch.dict("os.environ", {sources.ENV: " capture , webcam-table "}):
            self.assertEqual([sources.row_source(r) for r in sources.keep(ROWS)], ["webcam-table", "capture"])

    def test_retrain_only_counts_selected_captures(self):
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "real"
            real.mkdir()
            (real / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ROWS))
            for row in ROWS:
                (real / row["capture_id"]).mkdir()
                (real / row["capture_id"] / "card.png").write_bytes(b"png")
            with patch.dict("os.environ", {sources.ENV: "manavault-scanner"}):
                self.assertEqual([r["source"] for r in usable_rows(Path(tmp))], ["manavault-scanner"])


if __name__ == "__main__":
    unittest.main()
