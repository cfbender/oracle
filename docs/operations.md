# Operations

How the training box retrains, publishes and ships new sets. The commands are in the
[README](../README.md#everyday-commands).

## Profiles: one model per app

Both apps start from the same `models/` weights but get their own model. Each app has a profile,
`~/.config/cardid/<profile>.env` (templates in `profiles/`, created by `mise run setup-profiles`):

| Key | Meaning |
|---|---|
| `CARDID_PROFILE` | name; bookkeeping goes in `data/nightly/<profile>/` |
| `CARDID_SERVER` | The Gathering's base URL, or ManaVault's `/api/scanner/corrections` URL |
| `CARDID_CORRECTIONS_TOKEN` | the server's read-only export token (ManaVault: `SCANNER_CORRECTIONS_TOKEN`) |
| `CARDID_CORRECTIONS_DIR` | instead of a server: a mounted or rsynced corrections export |
| `CARDID_PUBLISH_TO` | `user@host:/path` (ssh), a local directory, or `github:OWNER/REPO` |
| `CARDID_SOURCES` | real capture sources to train and gate on: `webcam-table`, `capture`, `manavault-scanner` |
| `CARDID_CHECKPOINT`, `CARDID_DETECTOR` | extra search hints for starting checkpoints (default `models/`) |
| `CARDID_DETECTOR_EPOCHS`, `CARDID_SCENE_PROFILE` | also fine-tune the detector, on `table` or `phone` scenes |
| `CARDID_RETRAIN_EPOCHS`, `CARDID_WORKERS` | recogniser epochs (default 4) and loader workers |

The ManaVault profile also fine-tunes the detector on phone scenes: one large, near-upright card
on a stand-like background, stronger keystone, foil glare, defocus and the app's dark padding
bars, with 20% table scenes mixed in.

The files are data, not shell. `cardid.envfile` parses one literal `KEY=value` per line; quote
values with spaces and put comments on their own lines. `$VAR`, `$(...)` and backticks stay
literal text, and a malformed line aborts the run naming its line number. Only `CARDID_*` keys
are used. Exported variables override the file, and flags override both. A warning is printed if
the file is readable by other users. Tokens never appear in command logs.

Moving an older single-file setup (`~/.config/cardid.env`, The Gathering only) to profiles:

```sh
mkdir -p ~/.config/cardid && mv ~/.config/cardid.env ~/.config/cardid/the-gathering.env
printf 'CARDID_PROFILE=the-gathering\nCARDID_SOURCES=webcam-table,capture\n' >> ~/.config/cardid/the-gathering.env
mkdir -p data/nightly/the-gathering && mv data/nightly/state.json data/nightly/the-gathering/ 2>/dev/null
mise run setup-profiles          # adds the ManaVault profile next to it
```

## What a retrain does

`mise run retrain-all` runs `cardid.retrain` once per profile after one gallery refresh;
`mise run gathering` / `manavault` run it for one app. Steps:

1. Check the publish destination exists, so a mistyped target fails before training.
2. Pull the app's corrections, refresh the gallery (`scryfall --update`).
3. Resolve starting checkpoints: the ones whose SHA256 matches the published
   `current/manifest.json`, searched across `data/runs/*/{best,last}.pt` and the
   `CARDID_CHECKPOINT`/`CARDID_DETECTOR` hints. Without a match it warns and takes the newest
   checkpoint of the right model type. `--checkpoint` / `--detector` always win.
4. Fine-tune the recogniser (with `--real` when gallery-backed train captures exist, otherwise
   synthetic only), and the detector if `CARDID_DETECTOR_EPOCHS` is set. Each trainer's
   `best.pt` is used, or `last.pt` if no epoch beat the starting model.
5. Export and verify 64 scenes for torch/ONNX parity.
6. Score the published bundle and the candidate on the same held-out real captures, end to end.
7. Publish if the candidate's top-1 is at least the baseline's.

The gate's rules:

- Labels resolve through each bundle's `arts.json` and `printings.json`, so a sibling printing
  counts as its artwork. Captures whose label only one bundle knows are listed, recorded as
  `dropped_captures`, and left out of both scores.
- `--force` publishes despite a regression. It cannot override missing baselines, incomparable
  sets, parity failures, labels that changed during training, or a published manifest that
  changed during the run.
- Without held-out captures the run warns and publishes with no real accuracy check.
- A destination without `current/manifest.json` is a first publication: it starts from the
  newest local bundle's checkpoints and publishes without the concurrent-publish guard.

Every run, including failures and dry runs, writes `data/retrain/<version>.json` with the
commands, chosen checkpoints, scores and status. A successful publish records the new pair in
`data/nightly/<profile>/state.json` (used by `promote-models` and the nightly loop). Retrain and
nightly share one lock, so runs never overlap; don't run other importers or trainers alongside.

Useful flags (after `--`): `--dry-run` (reads local data and the published manifest, prints the
plan; pulls and trains nothing), `--no-publish` (train, export and score for review),
`--no-update-gallery`, `--epochs N`, `--detector-epochs N`, `--from-dir DIR`.

### Trusted outlines for the detector

A capture's outline usually came from the detector, so training the detector on it would only
teach its own error; `trusted_quad` skips those. Outlines a person drew or confirmed arrive with
`quad_source: "manual"`:

- ManaVault, with **Check outlines** on, pauses on each logged scan to confirm or drag the corners.
- The Gathering: Shift+click the four corners of a card, then pick it.
- `cardid.capture`: drawing the quad by hand.

`corrections pull` keeps a `manual` outline unless it is implausible (a sliver, or two corners on
one spot) and re-warps `card.png` when an outline changes. Retrain passes `--real` to the
detector only when trusted outlines exist and records `detector_real_captures` in its report.

## New sets

New cards need gallery embeddings, not training. `mise run new-set` runs `scryfall --update` once,
then `retrain --gallery-only` for each profile:

- Each app re-exports exactly the checkpoints behind its own published bundle (SHA256 match
  required; it refuses rather than guess).
- The new bundle passes the same held-out gate and publishes as `gallery-<timestamp>`.
- `--profile NAME` (repeatable) limits it to some apps; `--dry-run` downloads nothing.

`scryfall --update` keeps the previous bulk file as `all-cards.jsonl.gz.previous`, preserves
existing art IDs and splits, appends new illustrations as train, and downloads only missing art.

## Publishing

`cardid.publish` ships an exported bundle. Retrain calls it; by hand:

```sh
uv run python -m cardid.publish data/bundles/<version> --to user@host:/srv/the-gathering/cardid
uv run python -m cardid.publish data/bundles/<version> --to github:cfbender/manavault
```

- **ssh / local directory:** verifies `SHA256SUMS`, streams a tarball into `<path>/.incoming`,
  re-verifies on the host, moves the version into place, repoints `current` atomically (the old
  one becomes `previous`) and prunes versions beyond `--keep` (default 3). A failed transfer
  never touches `current`. The host needs `bash`, `flock`, `tar` and GNU coreutils.
- **GitHub:** one release per bundle, tagged `scanner-bundle-<version>`, never marked latest
  (ManaVault's Android shell uses the latest release for app updates). It refuses an existing
  version and verifies uploaded sizes.

Versions are immutable; export each with a new `--version`. When ssh publishing fails, the
reason is the `publish: ...` line just above `publish failed on <host>`: the version already
exists, `current` is a directory instead of a symlink (move it to `<path>/<version>` and
`ln -s <version> current`), or `current` changed since the candidate was scored.

## Optional nightly loop

`nightly.sh` is a stricter, budgeted retrain for one profile: pull, resume training (2 epochs,
2 workers, reduced learning rates), export, score, publish if not worse. It refuses to train when
no new corrections arrived since its last completed run, needs both train and held-out captures,
and never refreshes the gallery (use `new-set`). It runs under `nice 15` and a two-hour budget
for the whole process group; nice limits CPU priority, not GPU use.

```sh
CARDID_ENV_FILE=~/.config/cardid/the-gathering.env bash nightly.sh --dry-run   # pulls; never trains or publishes
CARDID_ENV_FILE=~/.config/cardid/the-gathering.env bash nightly.sh
```

Logs go to `data/nightly/YYYY-MM-DD.log`, with a JSON report per run. `nightly.env.example` lists
its overrides (`CARDID_EPOCHS`, `CARDID_BUDGET`, `CARDID_NICE`).

`systemd/` has a sandboxed user service and a 04:00 America/New_York timer. They assume the
checkout is at `~/oracle`; ideally run them as a dedicated unprivileged account that owns only
the checkout, its `data/`, the env file and an SSH key for the publish target.

```sh
mkdir -p ~/.config/systemd/user
cp systemd/cardid-nightly.{service,timer} ~/.config/systemd/user/
systemctl --user edit cardid-nightly.service   # add: [Service] Environment=CARDID_ENV_FILE=%h/.config/cardid/the-gathering.env
systemctl --user daemon-reload
systemctl --user start cardid-nightly.service        # one run, no schedule
systemctl --user enable --now cardid-nightly.timer   # only to schedule it
journalctl --user -u cardid-nightly.service
```

The service is sandboxed (`ProtectSystem=strict`, `ProtectHome=read-only`, writable
`~/oracle/data` only), so accept the publish host's SSH key once beforehand, don't rely on SSH
`ControlMaster` sockets under `~/.ssh`, and add a local publish directory to `ReadWritePaths=` in
a drop-in. User units with these settings need unprivileged user namespaces. Enable lingering
(`loginctl enable-linger`) to run while logged out.
