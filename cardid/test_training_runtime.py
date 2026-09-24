"""Shared trainer setup: device/worker defaults, seeding, the seeded DataLoader, run metadata."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from . import ML_DIR, training_runtime
from .training_runtime import Runtime, add_runtime_args, default_counts, make_loader, seed_everything, setup, write_run_metadata


def parse(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_runtime_args(parser, "workers")
    parser.add_argument("--run", default="r")
    return parser.parse_args(list(argv))


class Draws(torch.utils.data.Dataset):
    """Module level so forkserver/spawn workers (Python 3.14 default) can unpickle it."""

    def __len__(self):
        return 4

    def __getitem__(self, i):
        return torch.tensor([np.random.random(), random.random(), torch.get_num_threads()])


def worker_draws(seed: int) -> torch.Tensor:
    """What one loader worker process sees for `seed`: global NumPy/Python draws and threads."""
    cv2.setNumThreads(0)  # as `setup` does before any worker forks
    loader = make_loader(Draws(), 2, Runtime(torch.device("cpu"), workers=1, threads=1, seed=seed))
    return torch.cat(list(loader))


class TrainingRuntimeTest(unittest.TestCase):
    def test_default_counts_split_cpu_and_favour_workers_on_gpu(self):
        self.assertEqual(default_counts(torch.device("cpu"), cores=16), (8, 8))
        self.assertEqual(default_counts(torch.device("cuda"), cores=16), (15, 2))
        self.assertEqual(default_counts(torch.device("cpu"), cores=1), (2, 2))

    def test_setup_resolves_flags_and_seeds_every_generator(self):
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        runtime = setup(parse("--device", "cpu", "--workers", "3", "--threads", "1", "--seed", "7"), "test")
        self.assertEqual(runtime, Runtime(device=torch.device("cpu"), workers=3, threads=1, seed=7))
        self.assertEqual(torch.get_num_threads(), 1)
        draws = (random.random(), np.random.random(), torch.rand(1).item())
        seed_everything(7)
        self.assertEqual((random.random(), np.random.random(), torch.rand(1).item()), draws)
        self.assertEqual(parse().seed, 0)  # the historical torch.manual_seed(0) default

    def test_loader_shuffle_follows_the_seed(self):
        data = torch.arange(64)

        def order(seed: int) -> list[int]:
            loader = make_loader(data, 8, Runtime(torch.device("cpu"), workers=0, threads=1, seed=seed))
            return torch.cat(list(loader)).tolist()

        self.assertEqual(order(3), order(3))
        self.assertNotEqual(order(3), order(4))
        self.assertEqual(sorted(order(3)), list(range(64)))

    def test_loader_worker_processes_are_seeded_and_single_threaded(self):
        # A fresh interpreter, like a trainer: forking this test process after other tests have
        # used OpenCV from their own threads can deadlock the child's cv2.setNumThreads.
        script = (
            "import json, torch; from cardid.test_training_runtime import worker_draws; "
            "print(json.dumps([worker_draws(5).tolist(), worker_draws(5).tolist(), worker_draws(6).tolist()]))"
        )
        result = subprocess.run([sys.executable, "-c", script], cwd=ML_DIR, capture_output=True, text=True, timeout=120, check=True)
        first, again, other = (torch.tensor(d) for d in json.loads(result.stdout.splitlines()[-1]))
        torch.testing.assert_close(first, again)
        self.assertFalse(torch.equal(first, other))
        self.assertTrue((first[:, 2] == 1).all())

    def test_run_metadata_records_seed_and_versions(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        args = parse("--seed", "11", "--run", "x")
        path = write_run_metadata(root, args, Runtime(torch.device("cpu"), workers=2, threads=3, seed=11))
        meta = json.loads(path.read_text())
        self.assertEqual(path.name, "run.json")
        self.assertEqual((meta["seed"], meta["args"]["seed"], meta["workers"], meta["threads"], meta["device"]), (11, 11, 2, 3, "cpu"))
        self.assertEqual(meta["versions"]["torch"], torch.__version__)
        self.assertEqual(meta["versions"]["numpy"], np.__version__)
        self.assertIn("torchvision", meta["versions"])

    def test_both_trainers_expose_the_runtime_flags(self):
        import contextlib
        import io
        from unittest.mock import patch

        from . import train, train_detector

        for module in (train, train_detector):
            with self.subTest(module.__name__), patch("sys.argv", [module.__name__, "--help"]), contextlib.redirect_stdout(io.StringIO()) as out:
                with self.assertRaises(SystemExit):
                    module.main()
            for flag in ("--seed", "--device", "--workers", "--threads"):
                self.assertIn(flag, out.getvalue())
        self.assertIs(train.setup, training_runtime.setup)


if __name__ == "__main__":
    unittest.main()
