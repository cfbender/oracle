"""An exported bundle provides every field and graph name the browser runtimes read.

The consumers are the apps' TypeScript recognition code, read from checkouts next to this one
(or the directories in `CARDID_CONSUMERS`, separated by `:`):

- The Gathering: assets/react/src/features/webcam-table/recognition (`recognizer.worker.ts`)
- ManaVault: assets/react/src/pages/scan/recognition (`recognizer.ts`)

Each has `pipeline.ts` (`BundleConstants`) and `messages.ts` (`BundleInfo`, the manifest fields
the server forwards) plus a module holding the ONNX feed/output names. Constant, gallery and
file names are read from those sources, so renaming either side fails here instead of in a
browser. Without any consumer checkout the source checks are skipped.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import onnxruntime as ort
import torch

from . import ML_DIR, constants, data
from .detect import FRAME_NAMES
from .detector import CornerNet
from .export import SUMS, export_bundle, sha256
from .model import Embedder
from .publish import REQUIRED, check_bundle

# (recognition directory relative to the app checkout, module with the ONNX feed/output names)
KNOWN_CONSUMERS = {
    "the-gathering": ("assets/react/src/features/webcam-table/recognition", "recognizer.worker.ts"),
    "manavault": ("assets/react/src/pages/scan/recognition", "recognizer.ts"),
}


def consumers() -> list[tuple[Path, str]]:
    """(recognition directory, graph module) for every consumer checkout that exists."""
    found = []
    explicit = [Path(p) for p in os.environ.get("CARDID_CONSUMERS", "").split(":") if p]
    for directory in explicit:
        # The graph module is whichever reads the ONNX outputs (ManaVault's worker delegates).
        modules = [m for m in ("recognizer.worker.ts", "recognizer.ts") if (directory / m).is_file()]
        module = next((m for m in modules if "?.data" in (directory / m).read_text()), modules[0] if modules else "recognizer.ts")
        found.append((directory, module))
    if not explicit:
        for app, (relative, module) in KNOWN_CONSUMERS.items():
            directory = ML_DIR.parent / app / relative
            if (directory / "pipeline.ts").is_file():
                found.append((directory, module))
    return found


def bundle_file_names(messages: str) -> list[str]:
    """Files the browser fetches: `files: Record<"a" | "b", string>` or a `BUNDLE_FILES` list."""
    record = re.search(r"files: Record<(.*?),\s*string\s*>", messages, re.DOTALL)
    names = re.findall(r'"([\w.]+)"', record.group(1)) if record else []
    if not names:
        listed = re.search(r"BUNDLE_FILES = \[(.*?)\]", messages, re.DOTALL)
        names = re.findall(r'"([\w.]+)"', listed.group(1)) if listed else []
    return names


def interface_fields(source: str, name: str) -> list[str]:
    """Top-level field names of `interface <name> { ... }` in a TypeScript source."""
    body = re.search(rf"interface {name}\b[^{{]*\{{(.*?)\n\}}", source, re.DOTALL)
    assert body, f"interface {name} not found"
    fields = []
    for line in body.group(1).splitlines():
        match = re.match(r"  (\w+)\??:", line)  # two-space indent: top level only
        if match:
            fields.append(match.group(1))
    return fields


class ManifestContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())
        art_dir = cls.root / "art"
        art_dir.mkdir()
        rng = np.random.default_rng(0)
        arts = []
        for i in range(4):
            art_id = f"0000000{i}-0000-0000-0000-000000000000"
            cv2.imwrite(str(art_dir / f"{art_id}.jpg"), rng.integers(0, 256, (100, 137, 3), dtype=np.uint8))
            printing = {"id": art_id, "name": f"Card {i}", "set": "tst", "collector_number": str(i + 1), "layout": "normal", "face": 0, "lang": "en"}
            sibling = {**printing, "id": f"1000000{i}-0000-0000-0000-000000000000", "lang": "ja"}
            arts.append({**printing, "split": "train", "illustration_id": f"ill-{i}", "url": f"https://img.test/{i}.jpg", "printings": [printing, sibling]})
        (cls.root / "arts.json").write_text(json.dumps(arts))
        cls.checkpoint = cls.root / "runs" / "rec" / "best.pt"
        cls.detector = cls.root / "runs" / "det" / "best.pt"
        for path, model in ((cls.checkpoint, Embedder(pretrained=False)), (cls.detector, CornerNet(pretrained=False))):
            path.parent.mkdir(parents=True)
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
        cls.bundle = cls.root / "bundles" / "contract-v1"
        with patch.object(data, "DATA_DIR", cls.root), patch.object(data, "ART_DIR", art_dir), redirect_stdout(StringIO()):
            export_bundle(cls.checkpoint, cls.detector, cls.bundle, frame_penalty=0.02, topk=3, gallery_dtype="f16")
        cls.manifest = json.loads((cls.bundle / "manifest.json").read_text())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    def consumer_sources(self) -> list[tuple[Path, str]]:
        found = consumers()
        if not found:
            self.skipTest("no consumer checkout next to this one (../the-gathering, ../manavault) or in CARDID_CONSUMERS")
        return found

    def test_manifest_has_every_field_the_browser_reads(self):
        for directory, _module in self.consumer_sources():
            with self.subTest(consumer=str(directory)):
                self.check_manifest_fields(directory)

    def check_manifest_fields(self, directory: Path):
        pipeline = (directory / "pipeline.ts").read_text()
        messages = (directory / "messages.ts").read_text()
        m = self.manifest
        self.assertEqual(m["version"], "contract-v1")
        self.assertIsInstance(m["created"], str)
        constant_fields = interface_fields(pipeline, "BundleConstants")
        self.assertIn("refine_fill", constant_fields)  # the parser found the interface
        for field in constant_fields:
            with self.subTest(constant=field):
                self.assertIn(field, m["constants"])
        c = m["constants"]
        self.assertEqual(
            (c["scene"], c["det_input"], c["rotations"], c["refine_fill"], c["refine_min_side"], c["card_aspect"]),
            (constants.SCENE, constants.DET_INPUT, constants.ROTATIONS, constants.REFINE_FILL, constants.REFINE_MIN_SIDE, constants.CARD_ASPECT),
        )
        self.assertEqual(c["frame_names"], FRAME_NAMES)
        gallery_fields = re.findall(r"(\w+):", re.search(r"gallery: \{([^}]*)\}", messages).group(1))
        self.assertIn("topk", gallery_fields)
        for field in gallery_fields:
            with self.subTest(gallery=field):
                self.assertIn(field, m["gallery"])
        self.assertEqual((m["gallery"]["arts"], m["gallery"]["topk"], m["gallery"]["embed_dim"]), (4, 3, 128))
        file_names = bundle_file_names(messages)
        self.assertIn("detector.onnx", file_names)
        for name in [*file_names, "printings.json"]:
            with self.subTest(file=name):
                self.assertTrue((self.bundle / name).is_file())
        for name, entry in m["files"].items():
            self.assertEqual((entry["bytes"], entry["sha256"]), ((self.bundle / name).stat().st_size, sha256(self.bundle / name)))
        self.assertTrue(set(REQUIRED) - {"manifest.json"} <= set(m["files"]))
        self.assertIn("printings.json", m["files"])

    def test_bundle_passes_the_publisher_checks(self):
        self.assertEqual(check_bundle(self.bundle)["version"], "contract-v1")
        sums = dict(line.split("  ")[::-1] for line in (self.bundle / SUMS).read_text().splitlines())
        self.assertEqual(sums["manifest.json"], sha256(self.bundle / "manifest.json"))

    def test_onnx_names_used_by_the_browser(self):
        for directory, module in self.consumer_sources():
            worker = (directory / module).read_text()
            for name in ("window", "scene", "quad", "embeddings"):
                with self.subTest(consumer=str(directory), feed=name):
                    self.assertIn(f"{name}:", worker)  # feed names used by the runtime
            for name in ("quad", "up", "centre", "short", "indices", "scores"):
                with self.subTest(consumer=str(directory), output=name):
                    self.assertIn(f".{name}?.data", worker)  # outputs it reads by name

    def test_onnx_names_and_shapes(self):

        def session(name):
            return ort.InferenceSession(str(self.bundle / name), providers=["CPUExecutionProvider"])

        det, embed, search = session("detector.onnx"), session("embed.onnx"), session("search.onnx")
        self.assertEqual([i.name for i in det.get_inputs()], ["window"])
        self.assertEqual(det.get_inputs()[0].type, "tensor(uint8)")
        self.assertEqual([o.name for o in det.get_outputs()], ["quad", "up", "centre", "short"])
        self.assertEqual([i.name for i in embed.get_inputs()], ["scene", "quad"])
        self.assertEqual([i.name for i in search.get_inputs()], ["embeddings"])
        self.assertEqual([o.name for o in search.get_outputs()], ["indices", "scores"])

        size = self.manifest["constants"]["det_input"]
        quad, up, centre, short = det.run(None, {"window": np.full((size, size, 4), 128, np.uint8)})
        self.assertEqual((quad.shape, up.shape, centre.shape, np.asarray(short).size), ((4, 2), (2,), (2,), 1))
        scene = np.full((300, 400, 4), 90, np.uint8)
        (vectors,) = embed.run(None, {"scene": scene, "quad": np.float32([[100, 50], [200, 50], [200, 190], [100, 190]])})
        self.assertEqual(vectors.shape, (len(FRAME_NAMES), self.manifest["gallery"]["embed_dim"]))
        indices, scores = search.run(None, {"embeddings": vectors})
        self.assertEqual((indices.shape, scores.shape), ((3,), (3,)))
        self.assertTrue(set(indices.tolist()) <= set(range(4)))

    def test_gallery_metadata_matches_gallery_ts(self):
        arts = json.loads((self.bundle / "arts.json").read_text())
        printings = json.loads((self.bundle / "printings.json").read_text())
        self.assertEqual(len(arts), self.manifest["gallery"]["arts"])
        for art in arts:
            self.assertTrue({"id", "name", "set", "frame", "printing_count"} <= set(art))
            self.assertIn(art["frame"], self.manifest["constants"]["frame_names"])
            self.assertNotIn("printings", art)  # siblings load on demand from printings.json
            self.assertEqual(len(printings[art["id"]]), art["printing_count"])
            self.assertTrue(all({"id", "name", "set"} <= set(p) for p in printings[art["id"]]))


if __name__ == "__main__":
    unittest.main()
