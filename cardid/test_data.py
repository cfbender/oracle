"""Deserialization safety: pickle-free eval-query caches and weights-only checkpoint loads."""

from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import data


class Boom:
    """Unpickling this would run code; a pickle-free loader must never get that far."""

    def __reduce__(self):
        return (exec, ("raise SystemExit('pickle executed')",))


class QueryCacheTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.images = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
        self.targets = np.array([3, 5])
        self.infos = [{"width": 70, "strong_perspective": np.bool_(True), "jpeg_quality": np.int64(50), "blur_sigma": 0.5}, {"width": 90}]

    def test_round_trip_has_no_object_arrays(self):
        path = self.root / "q.npz"
        data.save_query_cache(path, self.images, self.targets, self.infos)
        with np.load(path, allow_pickle=False) as z:
            self.assertTrue(all(z[name].dtype != object for name in z.files))
        images, targets, infos = data.load_query_cache(path)
        np.testing.assert_array_equal(images, self.images)
        np.testing.assert_array_equal(targets, self.targets)
        self.assertEqual(infos, [{"width": 70, "strong_perspective": True, "jpeg_quality": 50, "blur_sigma": 0.5}, {"width": 90}])
        self.assertEqual(list(self.root.iterdir()), [path])  # the temp file was renamed into place

    def test_old_pickled_caches_are_ignored_not_unpickled(self):
        path = self.root / "old.npz"
        np.savez(path, images=self.images, targets=self.targets, infos=np.array([Boom(), {}], dtype=object))
        with redirect_stdout(io.StringIO()) as out:
            self.assertIsNone(data.load_query_cache(path))
        self.assertIn("rebuilding", out.getvalue())
        self.assertIsNone(data.load_query_cache(self.root / "missing.npz"))

    def test_cached_eval_queries_rebuilds_an_old_cache(self):
        arts = [{"id": "a", "split": "eval"}]
        with (
            patch.object(data, "DATA_DIR", self.root),
            patch.object(data, "build_eval_queries", return_value=(self.images, self.targets, self.infos[1:])) as build,
        ):
            first = data.cached_eval_queries(arts)
            cache = next(self.root.glob("eval-queries-*.npz"))
            np.savez(cache, images=self.images, targets=self.targets, infos=np.array([Boom()], dtype=object))
            with redirect_stdout(io.StringIO()):
                second = data.cached_eval_queries(arts)
            third = data.cached_eval_queries(arts)
        self.assertEqual(build.call_count, 2)
        for result in (first, second, third):
            self.assertEqual(result[2], [{"width": 90}])


class CheckpointLoadTest(unittest.TestCase):
    def test_detector_checkpoints_load_weights_only_and_refuse_pickled_code(self):
        import torch

        from .detector import CornerNet, load_checkpoint

        with tempfile.TemporaryDirectory() as temp:
            good, evil = Path(temp) / "good.pt", Path(temp) / "evil.pt"
            net = CornerNet(pretrained=False)
            torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, good)
            fresh = CornerNet(pretrained=False)
            load_checkpoint(fresh, good, torch.device("cpu"))
            torch.testing.assert_close(fresh.state_dict()["head.3.bias"], net.state_dict()["head.3.bias"])
            torch.save({"head.3.bias": Boom()}, evil)
            with self.assertRaises(Exception) as raised:
                load_checkpoint(CornerNet(pretrained=False), evil, torch.device("cpu"))
            self.assertNotIsInstance(raised.exception, SystemExit)


if __name__ == "__main__":
    unittest.main()
