"""CPU-only workflow tests: fake trainers/publisher, real files and manifest hashes."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import corrections, nightly, retrain, train
from .gallery import bundle_index
from .workflow import find_manifest, resolve_checkpoint, sha256, trained_checkpoint


class RetrainTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.runs = self.data / "runs"
        self.checkpoint = self.write(self.runs / "older" / "best.pt", b"recogniser")
        self.detector = self.write(self.runs / "not-named-det" / "last.pt", b"detector")
        self.bundle = self.data / "bundles" / "baseline"
        files = {}
        # The held-out label "art" is a sibling printing, which exported bundles keep in
        # printings.json rather than arts.json; "reprint" only exists in newer galleries.
        contents = {"arts.json": b'[{"id":"art-row"},{"id":"other"}]', "printings.json": b'{"art-row":[{"id":"art"}]}'}
        for name in ("detector.onnx", "embed.onnx", "search.onnx", "arts.json", "printings.json"):
            path = self.write(self.bundle / name, contents.get(name, b"[]"))
            files[name] = {"sha256": sha256(path)}
        self.manifest = {
            "version": "baseline",
            "recogniser": {"sha256": sha256(self.checkpoint)},
            "detector": {"sha256": sha256(self.detector)},
            "files": files,
        }
        self.manifest_path = self.write(self.bundle / "manifest.json", json.dumps(self.manifest).encode())
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.output = io.StringIO()
        self.redirect = redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)
        self.commands = []
        self.last_only = False
        self.fail_publish = False
        self.ssh_exit = 0
        self.manifest_exit = 0

    def write(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def args(self, *extra):
        return retrain.parse_args(["--env-file", str(self.root / "absent.env"), "--server", "https://example.test", "--to", "desktop:/cardid", *extra])

    def fake_runner(self, *cmd):
        self.commands.append(cmd)
        if cmd[0] == "ssh":
            if self.ssh_exit:
                raise subprocess.CalledProcessError(self.ssh_exit, cmd)
        elif cmd[0] == "rsync":
            dest = Path(cmd[-1])
            if cmd[-2].endswith("manifest.json"):
                if self.manifest_exit:
                    raise subprocess.CalledProcessError(self.manifest_exit, cmd)
                shutil.copyfile(self.manifest_path, dest)
            else:
                shutil.copytree(self.bundle, dest)
        elif cmd[2] in {"cardid.train", "cardid.train_detector"}:
            run = cmd[cmd.index("--run") + 1]
            self.write(self.runs / run / "last.pt", b"last")
            if not self.last_only:
                self.write(self.runs / run / "best.pt", b"best")
        elif cmd[2] == "cardid.export":
            out = self.data / "bundles" / cmd[cmd.index("--version") + 1]
            self.write(out / "arts.json", b'[{"id":"art-row"},{"id":"other"},{"id":"reprint"}]')
            self.write(out / "printings.json", b'{"art-row":[{"id":"art"}]}')
        elif cmd[2] == "cardid.publish" and self.fail_publish:
            raise subprocess.CalledProcessError(1, cmd)

    def run_pipeline(self, *extra, scorer=None):
        kwargs = {"scorer": scorer} if scorer else {}
        return retrain.run(self.args(*extra), data=self.data, runner=self.fake_runner, version="retrain-test", **kwargs)

    def real_rows(self, *, train=True, evaluation=True, known=True, newer=False):
        rows = []
        for split, enabled in (("train", train), ("eval", evaluation)):
            if enabled:
                rows.append({"capture_id": split, "label": "art" if known else "unknown", "split": split})
                self.write(self.data / "real" / split / "card.png", b"card")
        if newer:
            rows.append({"capture_id": "eval-newer", "label": "reprint", "split": "eval"})
            self.write(self.data / "real" / "eval-newer" / "card.png", b"card")
        self.write(self.data / "real" / "labels.jsonl", "\n".join(json.dumps(r) for r in rows).encode())
        self.write(self.data / "arts.json", b'[{"id":"art"}]')
        self.write(self.data / "art" / "art.jpg", b"art")
        return rows

    def scores(self, correct=3, *, captures="same"):
        values = iter(
            [
                {"count": 4, "correct": 3, "captures": "same", "top1": 0.75},
                {"count": 4, "correct": correct, "captures": captures, "top1": correct / 4},
            ]
        )
        return lambda *_: next(values)

    def test_manifest_match_beats_newer_mtime_and_stale_hint(self):
        newer = self.write(self.runs / "new" / "best.pt", b"experiment")
        os.utime(self.checkpoint, (1, 1))
        os.utime(newer, (200, 200))
        self.assertEqual(resolve_checkpoint("recogniser", self.runs, self.manifest, hint=newer), self.checkpoint)
        self.assertEqual(resolve_checkpoint("detector", self.runs, self.manifest), self.detector)
        self.assertIn("manifest SHA256 match", self.output.getvalue())

    def test_mtime_fallback_distinguishes_models_and_best_last(self):
        old = self.write(self.runs / "det-misleading" / "best.pt", b"")
        new = self.write(self.runs / "det-misleading" / "last.pt", b"")
        det = self.write(self.runs / "recogniser-misleading" / "best.pt", b"")
        for path, keys in ((old, ["head.weight", "features.0.weight"]), (new, ["head.weight", "features.0.weight"]), (det, ["head.0.weight", "stem.0.weight"])):
            torch.save(dict.fromkeys(keys, torch.zeros(1)), path)
        os.utime(old, (10, 10))
        os.utime(new, (20, 20))
        os.utime(det, (30, 30))
        # A manifest that matches nothing must warn and then choose by model type.
        manifest = {"recogniser": {"sha256": "missing"}}
        self.checkpoint.unlink()
        self.detector.unlink()
        self.assertEqual(resolve_checkpoint("recogniser", self.runs, manifest), new)
        self.assertEqual(resolve_checkpoint("detector", self.runs, None), det)
        self.assertIn("matches no local checkpoint", self.output.getvalue())
        self.assertIn("WARNING", self.output.getvalue())

    def test_explicit_flag_wins_over_manifest_and_missing_explicit_fails(self):
        explicit = self.write(self.root / "explicit.pt", b"explicit")
        self.assertEqual(resolve_checkpoint("recogniser", self.runs, self.manifest, explicit), explicit)
        with self.assertRaisesRegex(SystemExit, "does not exist"):
            resolve_checkpoint("detector", self.runs, self.manifest, self.root / "missing.pt")

    def test_no_runs_directory_and_strict_nightly_never_guess(self):
        with self.assertRaisesRegex(SystemExit, "no usable"):
            resolve_checkpoint("recogniser", self.root / "missing-runs", None)
        with self.assertRaisesRegex(SystemExit, "no local checkpoint matches"):
            resolve_checkpoint("recogniser", self.runs, {"recogniser": {"sha256": "missing"}}, strict=True)
        with self.assertRaisesRegex(SystemExit, "does not match"):
            resolve_checkpoint("recogniser", self.runs, self.manifest, self.detector, strict=True)

    def test_unreachable_target_uses_newest_local_manifest(self):
        newer = self.write(self.data / "bundles" / "newer" / "manifest.json", b"{}")
        os.utime(self.manifest_path, (1, 1))
        os.utime(newer, (2, 2))

        def unavailable(*cmd):
            raise subprocess.CalledProcessError(255, cmd)

        path, source = find_manifest("unreachable:/data", self.data / "bundles", self.root, unavailable)
        self.assertEqual((path, source), (newer, str(newer.parent)))
        self.assertIn("published manifest unavailable", self.output.getvalue())
        self.assertEqual(find_manifest(None, self.root / "none", self.root), (None, None))

    def test_env_precedence_and_source_flags(self):
        env = self.write(
            self.root / "settings.env",
            b"CARDID_SERVER=https://file.test\nCARDID_PUBLISH_TO='host:/path with spaces'\nCARDID_RETRAIN_EPOCHS=7\nCARDID_EPOCHS=2\nCARDID_CORRECTIONS_TOKEN='private token'\n",
        )
        args = retrain.parse_args(["--env-file", str(env)])
        self.assertEqual((args.server, args.to, args.epochs), ("https://file.test", "host:/path with spaces", 7))
        os.environ.update(CARDID_SERVER="https://environment.test", CARDID_RETRAIN_EPOCHS="6", CARDID_CORRECTIONS_DIR="/configured/inbox")
        args = retrain.parse_args(["--env-file", str(env), "--server", "https://flag.test", "--epochs", "5", "--to", "/target", "--checkpoint", "manual.pt"])
        self.assertEqual((args.server, args.from_dir, args.epochs, args.to, args.checkpoint), ("https://flag.test", None, 5, "/target", Path("manual.pt")))
        args = retrain.parse_args(["--env-file", str(env), "--from-dir", "/flag/inbox"])
        self.assertEqual((args.server, args.from_dir, args.epochs), (None, Path("/flag/inbox"), 6))
        self.assertNotIn("private token", self.output.getvalue())

    def test_defaults_ignore_nightly_epochs_and_validate_positive_epochs(self):
        os.environ["CARDID_EPOCHS"] = "2"
        args = self.args()
        self.assertEqual(args.epochs, 4)
        self.assertTrue(args.update_gallery)
        self.assertIsNone(args.detector_epochs)
        with self.assertRaises(SystemExit):
            self.args("--detector-epochs", "0")

    def test_exact_synthetic_command_sequence_and_state_update(self):
        self.write(self.data / "nightly" / "state.json", b'{"corrections":"already-seen"}')
        report = self.run_pipeline()
        checkpoint = str(self.runs / "retrain-test" / "best.pt")
        # Only the temporary rsync output path varies.
        expected = [
            ("ssh", "desktop", "test -d /cardid"),
            ("python", "-m", "cardid.corrections", "pull", "--server", "https://example.test"),
            ("python", "-m", "cardid.scryfall", "--update"),
            ("rsync", "-aL", "--", "desktop:/cardid/current/manifest.json", self.commands[3][-1]),
            ("python", "-m", "cardid.train", "--resume", str(self.checkpoint), "--epochs", "4", "--run", "retrain-test"),
            ("python", "-m", "cardid.export", "--checkpoint", checkpoint, "--detector", str(self.detector), "--version", "retrain-test", "--verify", "64"),
            ("python", "-m", "cardid.evaluate", "--method", "checkpoint", "--checkpoint", checkpoint, "--profile", "realistic"),
            (
                "python",
                "-m",
                "cardid.publish",
                str(self.data / "bundles" / "retrain-test"),
                "--to",
                "desktop:/cardid",
                "--expected-current",
                sha256(self.manifest_path),
            ),
        ]
        self.assertEqual(self.commands, expected)
        self.assertEqual(report["status"], "published")
        state = json.loads((self.data / "nightly" / "state.json").read_text())
        self.assertEqual((state["checkpoint"], state["detector"], state["corrections"]), (checkpoint, str(self.detector), "already-seen"))

    def test_missing_destination_fails_before_pulling(self):
        self.ssh_exit = 1
        with self.assertRaisesRegex(SystemExit, "not a directory there"):
            self.run_pipeline()
        self.assertEqual(self.commands, [("ssh", "desktop", "test -d /cardid")])
        self.assertEqual(json.loads((self.data / "retrain" / "retrain-test.json").read_text())["status"], "failed")
        self.ssh_exit = 255
        with self.assertRaisesRegex(SystemExit, "ssh connection failed"):
            retrain.run(self.args(), data=self.data, runner=self.fake_runner, version="retrain-test-2")
        # Quoted remote paths and local directories are checked the same way.
        with self.assertRaisesRegex(SystemExit, "is not a directory"):
            retrain.run(self.args("--to", str(self.root / "absent")), data=self.data, runner=self.fake_runner, version="retrain-test-3")
        (self.root / "with space").mkdir()
        retrain.run(self.args("--to", str(self.root / "with space")), data=self.data, runner=self.fake_runner, version="retrain-test-4")
        self.assertNotIn("ssh", [c[0] for c in self.commands[2:]])
        self.ssh_exit = 0
        self.run_pipeline("--to", "desktop:/path with spaces", "--no-publish")
        self.assertNotIn("ssh", [c[0] for c in self.commands[2:]])

    def test_first_publication_falls_back_without_current_guard(self):
        # The destination exists but nothing is published there yet: rsync exits 23.
        self.manifest_exit = 23
        report = self.run_pipeline()
        self.assertEqual(report["status"], "published")
        self.assertFalse(report["baseline_published"])
        self.assertEqual(report["baseline_source"], str(self.bundle))
        publish = self.commands[-1]
        self.assertEqual(publish[2], "cardid.publish")
        self.assertNotIn("--expected-current", publish)
        self.assertIn("nothing is published at desktop:/cardid/current yet", self.output.getvalue())
        self.assertIn("without the concurrent-publish guard", self.output.getvalue())
        train = next(c for c in self.commands if c[2] == "cardid.train")
        self.assertEqual(train[train.index("--resume") + 1], str(self.checkpoint))

    def test_unreadable_published_manifest_aborts_unless_no_publish(self):
        self.manifest_exit = 12
        with self.assertRaisesRegex(SystemExit, "published manifest unreadable"):
            self.run_pipeline()
        self.assertNotIn("cardid.train", [c[2] for c in self.commands if len(c) > 2])
        report = self.run_pipeline("--no-publish")
        self.assertEqual(report["status"], "not-published")
        self.assertFalse(report["baseline_published"])
        self.assertIn("cardid.evaluate", [c[2] for c in self.commands if len(c) > 2])

    def test_detector_last_only_and_real_train_without_eval(self):
        self.real_rows(evaluation=False)
        self.last_only = True
        report = self.run_pipeline("--no-update-gallery", "--detector-epochs", "3", "--workers", "2", "--no-publish")
        self.assertEqual(
            self.commands[2],
            ("python", "-m", "cardid.train", "--resume", str(self.checkpoint), "--epochs", "4", "--run", "retrain-test", "--workers", "2", "--real"),
        )
        self.assertEqual(
            self.commands[3],
            (
                "python",
                "-m",
                "cardid.train_detector",
                "--resume",
                str(self.detector),
                "--epochs",
                "3",
                "--run",
                "retrain-test-detector",
                "--workers",
                "2",
                "--real",
            ),
        )
        self.assertEqual(report["candidate_detector"], str(self.runs / "retrain-test-detector" / "last.pt"))
        self.assertEqual(report["candidate_checkpoint"], str(self.runs / "retrain-test" / "last.pt"))
        self.assertEqual(
            self.commands[4],
            (
                "python",
                "-m",
                "cardid.export",
                "--checkpoint",
                report["candidate_checkpoint"],
                "--detector",
                report["candidate_detector"],
                "--version",
                "retrain-test",
                "--verify",
                "64",
            ),
        )
        self.assertFalse((self.data / "nightly" / "state.json").exists())

    def test_unknown_train_labels_use_synthetic_only(self):
        self.real_rows(evaluation=False, known=False)
        self.run_pipeline("--no-publish")
        train = next(c for c in self.commands if c[:3] == ("python", "-m", "cardid.train"))
        self.assertNotIn("--real", train)

    def test_equal_held_out_score_publishes(self):
        self.real_rows()
        report = self.run_pipeline(scorer=self.scores())
        self.assertTrue(report["allowed"])
        self.assertEqual(report["baseline"]["correct"], 3)
        self.assertEqual(report["candidate"]["correct"], 3)
        self.assertEqual(self.commands[-1][2], "cardid.publish")

    def test_bundle_index_resolves_sibling_printings_from_the_on_demand_file(self):
        index = bundle_index(self.bundle)
        # A sibling printing maps to its embedded art's row; a bundle without printings.json still indexes arts.
        self.assertEqual((index["art"], index["art-row"], index["other"]), (0, 0, 1))
        (self.bundle / "printings.json").unlink()
        self.assertEqual(bundle_index(self.bundle), {"art-row": 0, "other": 1})

    def test_sibling_printing_labels_score_and_newer_labels_are_dropped(self):
        self.real_rows(newer=True)
        scored = []

        def scorer(bundle_path, rows, real):
            scored.append((bundle_path.name, [r["capture_id"] for r in rows]))
            return {"count": len(rows), "correct": len(rows), "captures": "same", "top1": 1.0}

        report = self.run_pipeline(scorer=scorer)
        # Both bundles score the same single sibling-labelled capture; the baseline cannot know "reprint".
        self.assertEqual(scored, [("baseline", ["eval"]), ("retrain-test", ["eval"])])
        self.assertEqual(report["dropped_captures"], ["eval-newer"])
        self.assertEqual(report["status"], "published")
        self.assertIn("eval-newer -> reprint", self.output.getvalue())

    def test_no_commonly_known_held_out_label_refuses(self):
        self.real_rows(known=False)
        with self.assertRaisesRegex(SystemExit, "no held-out label is known to both bundles"):
            self.run_pipeline(scorer=self.scores())
        self.assertNotIn("cardid.publish", [c[2] for c in self.commands if len(c) > 2])

    def test_regression_refuses_and_keeps_nightly_state(self):
        self.real_rows()
        state = self.write(self.data / "nightly" / "state.json", b'{"checkpoint":"old"}')
        with self.assertRaisesRegex(SystemExit, "regressed"):
            self.run_pipeline(scorer=self.scores(2))
        self.assertEqual(state.read_bytes(), b'{"checkpoint":"old"}')
        self.assertEqual(self.commands[-1][2], "cardid.evaluate")
        report = json.loads((self.data / "retrain" / "retrain-test.json").read_text())
        self.assertEqual(report["status"], "regressed")

    def test_force_allows_comparable_regression(self):
        self.real_rows()
        report = self.run_pipeline("--force", scorer=self.scores(2))
        self.assertFalse(report["allowed"])
        self.assertEqual(report["status"], "published")
        self.assertIn("--expected-current", self.commands[-1])

    def test_force_cannot_override_incomparable_evaluation(self):
        self.real_rows()
        with self.assertRaisesRegex(SystemExit, "incomparable"):
            self.run_pipeline("--force", scorer=self.scores(4, captures="different"))
        self.assertNotIn("cardid.publish", [c[2] for c in self.commands])

    def test_publish_failure_does_not_advance_state(self):
        self.fail_publish = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_pipeline()
        self.assertFalse((self.data / "nightly" / "state.json").exists())
        report = json.loads((self.data / "retrain" / "retrain-test.json").read_text())
        self.assertEqual(report["status"], "failed")

    def test_dataset_change_blocks_even_force(self):
        self.real_rows()

        def scorer(*_):
            self.write(self.data / "real" / "labels.jsonl", b"")
            return {"correct": 1, "count": 1, "captures": "same"}

        with self.assertRaisesRegex(SystemExit, "dataset changed"):
            self.run_pipeline("--force", scorer=scorer)
        self.assertNotIn("cardid.publish", [c[2] for c in self.commands])

    def test_dry_run_only_reads_manifest_and_prints_conditional_outputs(self):
        report = self.run_pipeline("--dry-run", "--detector-epochs", "2")
        # Read-only steps still run: the destination preflight and the manifest fetch.
        self.assertEqual(len(self.commands), 2)
        self.assertEqual(self.commands[0], ("ssh", "desktop", "test -d /cardid"))
        self.assertEqual(self.commands[1][:4], ("rsync", "-aL", "--", "desktop:/cardid/current/manifest.json"))
        self.assertFalse((self.runs / "retrain-test").exists())
        self.assertFalse((self.data / "nightly" / "state.json").exists())
        self.assertIn('"$candidate_detector"', self.output.getvalue())
        self.assertIn("else printf %s", self.output.getvalue())
        self.assertEqual(report["status"], "dry-run")

    def test_missing_training_output_fails(self):
        with self.assertRaisesRegex(SystemExit, "neither best.pt nor last.pt"):
            trained_checkpoint(self.runs / "absent")

    def test_correction_subprocess_explicit_server_overrides_configured_directory(self):
        os.environ["CARDID_CORRECTIONS_DIR"] = "/configured/inbox"
        with patch("sys.argv", ["corrections", "pull", "--server", "https://flag.test"]), patch.object(corrections, "pull", return_value=0) as pull:
            corrections.main()
        pull.assert_called_once_with(corrections.REAL, "https://flag.test", None)

    def test_filesystem_command_does_not_include_server(self):
        args = retrain.parse_args(["--env-file", str(self.root / "absent"), "--from-dir", "/inbox", "--no-publish"])
        retrain.run(args, data=self.data, runner=self.fake_runner, version="retrain-test")
        self.assertEqual(self.commands[0], ("python", "-m", "cardid.corrections", "pull", "--from-dir", "/inbox"))

    def test_real_training_selects_synthetic_queries_only_when_real_eval_is_unavailable(self):
        # Stop before constructing a model: exercise train.main's query-selection branch,
        # without downloads, dataloader processes or model training.
        arts = [{"id": "art", "split": "train"}]
        queries = np.zeros((3, 1))
        targets = np.array([0, 0, 0])
        for eval_rows, expected_calls in (([], 0), ([{"label": "unknown"}], 0), ([{"label": "art"}], 1)):
            with self.subTest(eval_rows=eval_rows), ExitStack() as stack:
                stack.enter_context(patch("sys.argv", ["train", "--real", "--run", "selection-test", "--workers", "1", "--threads", "1"]))
                stack.enter_context(patch.object(train, "RUNS_DIR", self.runs))
                for name in ("PairDataset", "RealDataset", "make_loader", "gallery_images", "art_frames"):
                    stack.enter_context(patch.object(train, name))
                stack.enter_context(patch.object(train, "load_arts", return_value=arts))
                stack.enter_context(patch.object(train, "load_labels", side_effect=lambda split, rows=eval_rows: rows if split == "eval" else []))
                stack.enter_context(patch.object(train, "cached_eval_queries", return_value=(queries, targets, [])))
                real_eval = stack.enter_context(patch.object(train, "real_eval_queries", return_value=(queries, targets, [])))
                stack.enter_context(patch.object(train, "Embedder", side_effect=RuntimeError("stop before model construction")))
                with self.assertRaisesRegex(RuntimeError, "stop before model construction"):
                    train.main()
                self.assertEqual(real_eval.call_count, expected_calls)
                self.assertEqual(json.loads((self.runs / "selection-test" / "run.json").read_text())["seed"], 0)

    def test_nightly_resolves_published_pair_despite_stale_state_and_env(self):
        rows = self.real_rows()
        rows[0]["source"] = "webcam-table"
        self.write(self.data / "real" / "labels.jsonl", "\n".join(json.dumps(r) for r in rows).encode())
        state_path = self.write(self.data / "nightly" / "state.json", b'{"checkpoint":"stale.pt","detector":"stale-det.pt"}')
        os.environ.update(CARDID_PUBLISH_TO="desktop:/cardid", CARDID_CHECKPOINT="also-stale.pt", CARDID_DETECTOR="also-stale-det.pt")
        args = self.args()
        args.real_dir, args.state_dir = self.data / "real", self.data / "nightly"
        with (
            patch.object(nightly, "DATA_DIR", self.data),
            patch.object(nightly, "REAL", args.real_dir),
            patch.object(nightly, "pull", return_value=1),
            patch.object(nightly, "command", side_effect=self.fake_runner),
            patch.object(nightly, "score", side_effect=self.scores()),
        ):
            nightly.run(args)
        training = next(c for c in self.commands if c[2] == "cardid.train")
        self.assertEqual(training[training.index("--resume") + 1], str(self.checkpoint))
        self.assertIn("--expected-current", self.commands[-1])
        state = json.loads(state_path.read_text())
        self.assertEqual(state["detector"], str(self.detector))
        self.assertIn("nightly-", state["checkpoint"])


if __name__ == "__main__":
    unittest.main()
