# Contributing to the card recognition models

Oracle trains the models behind two apps: the card scanner in
[ManaVault](https://github.com/cfbender/manavault) (`/scan`, a phone over one card) and
click-to-identify in [The Gathering](https://github.com/cfbender/the-gathering)'s webcam table
(a webcam over a whole board). Both run the same kind of **bundle** in the browser:

- a **detector** (CornerNet) finds the card's four corners;
- a **recogniser** (MobileNetV3-Small) turns the card's art into a 128-number fingerprint;
- a **gallery** holds one fingerprint per distinct Scryfall artwork (~52k); the closest one wins.

Anyone can clone the repositories, train, evaluate, export a bundle and run it in a local copy of
either app. Only the maintainer publishes bundles that users receive, after a gate on real
held-out photos. Improvements come from four places, roughly in order of impact:

1. **Real photos with confirmed labels and trusted outlines** (see
   [Contribute real data](#contribute-real-data)). A few hundred phone scans took the scanner's
   recogniser from 96.9% to 100% top-1 on held-out scans.
2. **Training code**: augmentation, synthetic scenes, losses, frame templates.
3. **Gallery coverage**: printings, languages, frames and layouts the index misses.
4. **The browser runtimes** in the two apps: speed, camera handling, decision rules.

## The repositories

| Repository | What lives there | Where to look |
|---|---|---|
| [oracle](https://github.com/cfbender/oracle) | datasets, training, evaluation, ONNX export, publishing | `cardid/`, [README](README.md) (full reference) |
| [manavault](https://github.com/cfbender/manavault) | scanner UI, browser pipeline, training-data upload, bundle serving | `assets/react/src/pages/scan/`, `lib/manavault/scanner/`, [docs/scanner.md](https://github.com/cfbender/manavault/blob/main/docs/scanner.md) |
| [the-gathering](https://github.com/cfbender/the-gathering) | webcam table, browser pipeline, correction upload, bundle serving | `assets/react/src/features/webcam-table/recognition/`, `lib/the_gathering/card_id/`, [docs/webcam-table.md](https://github.com/cfbender/the-gathering/blob/main/docs/webcam-table.md) |

Clone them side by side. Oracle's contract test (`cardid/test_manifest_contract.py`) reads the
apps' TypeScript from sibling checkouts (or `CARDID_CONSUMERS`) and is skipped without them:

```sh
mkdir cards && cd cards
git clone https://github.com/cfbender/oracle.git
git clone https://github.com/cfbender/manavault.git       # optional: the scanner
git clone https://github.com/cfbender/the-gathering.git   # optional: the webcam table
```

## Set up Oracle

Oracle needs [mise](https://mise.jdx.dev) (it pins `uv`) and Python 3.11 or newer.

```sh
cd oracle
mise install
mise exec -- uv sync --extra cpu     # or --extra rocm on an AMD RX 9000 (gfx1201) Linux box
export UV_NO_SYNC=1                  # keep the installed torch extra for later `uv run`s
mise run test                        # CPU-only unit tests, about 15 s
```

`train`, `train_detector` and `evaluate` pick a device with `--device auto`: CUDA/ROCm, then
Apple's `mps`, then CPU.

- **CPU:** everything works, but training is slow (minutes per epoch on the 6k sample, much
  longer on the full gallery).
- **Apple silicon:** `--extra cpu` already includes Metal, so `auto` picks `mps`.
- **AMD:** `--extra rocm` installs AMD's wheels for gfx1201; see the README,
  "GPU training".
- **NVIDIA:** there is no `cuda` extra yet. On Linux, `--extra cpu` installs CPU-only wheels.
  Adding a `cuda` extra to `pyproject.toml` (next to `cpu` and `rocm`) is a welcome
  contribution.

`models/` holds the committed starting weights: the checkpoints behind the current production
bundle (README, "Starting weights"). You never have to train from scratch. Everything you
download or produce goes in `data/`, which Git ignores.

## Build a gallery

Training and evaluation need card art from [Scryfall](https://scryfall.com). The small sample is
enough to compare ideas; the full gallery is what ships.

```sh
uv run python -m cardid.scryfall --train 5000 --eval 1000   # bulk metadata (~393 MB) + 6k art crops
uv run python -m cardid.scryfall --all                      # later, optional: every artwork (~52k images, ~3 GB)
uv run python -m cardid.degrade <art_id>                    # see what the synthetic degradation does
```

The eval split is never trained on, so its score measures cards the model has not seen. Images
and card data belong to Scryfall and Wizards of the Coast. Keep them in `data/`, never commit
them, and follow [Scryfall's API guidelines](https://scryfall.com/docs/api).

## Measure first

Every change needs a before-and-after number on the same gallery, profile and seed. Numbers from
the 6k sample and the full gallery are not comparable.

```sh
# Recogniser on synthetic photos of unseen cards (profiles: see cardid/degrade.py PROFILES)
uv run python -m cardid.evaluate --method checkpoint --checkpoint models/recogniser.pt --profile realistic

# End to end on your own labelled real captures in data/real (detector included)
uv run python -m cardid.evaluate --method checkpoint --checkpoint models/recogniser.pt --real --detector models/detector.pt
```

The detector trainer prints its synthetic validation error and hit rate before the first epoch
and after every epoch. Its first line is the baseline for the same `--scene-profile`.

## Make a change

The README's "Code layout" maps the modules. The usual places to work:

| To improve | Look at |
|---|---|
| robustness to blur, glare, lighting, angles | `degrade.py` (`PROFILES`), `train.py` |
| finding the card (corners, orientation) | `scene_renderer.py` (the `table` and `phone` scene profiles), `train_detector.py`, `detector.py` |
| unusual frames (sagas, rooms, split cards, full art) | `detect.FRAMES`, `catalog.py`, README "Rooms, split/aftermath and flip geometry" |
| missing printings or languages | `catalog.py`, `scryfall.py`, README "Gallery coverage" |
| what the browser runs | `graphs.py`, `export.py`, `bundle.py` (the executable spec the apps port) |

Fine-tune from the starting weights rather than from ImageNet:

```sh
uv run python -m cardid.train --resume models/recogniser.pt --epochs 4 --run my-change
uv run python -m cardid.train_detector --resume models/detector.pt --epochs 4 --scene-profile phone --run my-det
```

Both trainers take `--seed` and write `data/runs/<run>/run.json` with their arguments and
library versions. Add `--real` to either one to mix in your labelled captures (the detector only
uses [trusted outlines](#contribute-real-data)). Each run writes `last.pt`, plus `best.pt` when an
epoch beats its starting checkpoint on the held-out data.

## Try it in an app

Export a bundle. The export runs a parity check between torch and ONNX, and fails if they
disagree:

```sh
uv run python -m cardid.export --checkpoint data/runs/my-change/best.pt --detector models/detector.pt --version my-change
uv run python -m cardid.bundle data/bundles/my-change --image photo.jpg --click 640,360   # reference runtime: top 5 + timings
```

**ManaVault.** Follow its README to run the app. Keep your bundle from being replaced by the
published one with `SCANNER_BUNDLE_SOURCE=off`, then install it from the manavault checkout
(installation moves the directory, so install a copy):

```sh
cp -r ../oracle/data/bundles/my-change data/scanner-incoming
SCANNER_BUNDLE_SOURCE=off mise exec -- mix run --no-start -e 'dir = "data/scanner-incoming"; {:ok, _} = Manavault.Scanner.Bundle.install(Jason.decode!(File.read!(Path.join(dir, "manifest.json"))), dir)'
SCANNER_BUNDLE_SOURCE=off mise exec -- mix phx.server    # then open /scan
```

No phone is needed. [docs/scanner.md](https://github.com/cfbender/manavault/blob/main/docs/scanner.md),
"Testing without a phone", turns a photo into a fake camera for Chromium.

**The Gathering.** Publish into its data directory from the oracle checkout, then run the app
(its README covers setup; development signs you in as a `dev` admin, and **Admin → Users** links
that account to a player so you can sit at a table):

```sh
uv run python -m cardid.publish data/bundles/my-change --to ../the-gathering/data/cardid
```

## Contribute real data

Real photos are the scarcest and most valuable input. They are also private: each one is a
frame from someone's camera, stored on that app's server. There is no public dataset, and
captures are never committed to any repository.

**Labels** come from explicit choices only: a scan you kept, a card you picked or corrected.
Automatic answers are never trusted. In ManaVault, deleting a scan marks its capture skipped.

**Outlines** train the detector only when a person drew or confirmed them
(`quad_source: "manual"`). The detector's own outline would teach it its own mistakes. To
contribute outlines:

- **ManaVault**: in scanner settings, turn on **Collect training data**, then **Check outlines**.
  Each scan pauses; tap **Looks right**, or drag the corners onto the card first.
- **The Gathering**: keep **Share card crops & picks for training** on, then Shift+click a card's
  four corners and pick the card.
- **Locally**: `uv run python -m cardid.capture --checkpoint models/recogniser.pt` opens a webcam
  page on `localhost:8765`. Confirm cards with `1`–`5`, and shift-drag a box when the outline is
  wrong. Captures go straight to `data/real/`.

**What helps most:** variety over volume. That means foils and etched cards, sleeves, old and
unusual frames, other languages, poor light, and every card the scanner gets wrong. Fix or
delete misreads rather than leaving them.

**Where the data goes:**

- **Your own server.** Set `SCANNER_CORRECTIONS_TOKEN` (ManaVault) or `CARDID_CORRECTIONS_TOKEN`
  (The Gathering) on the server, then pull the captures into your Oracle checkout:

  ```sh
  CARDID_CORRECTIONS_TOKEN=... uv run python -m cardid.corrections pull --server https://your-manavault/api/scanner/corrections
  CARDID_CORRECTIONS_TOKEN=... uv run python -m cardid.corrections pull --server https://your-gathering
  ```

- **The maintainer's instances.** If you scan there, your uploads already feed the official
  models.
- **Sharing with the project.** Don't send captures as a pull request. Open an issue to arrange
  it.

## Open a pull request

- **Oracle changes:**
  - Include the evidence: baseline and candidate numbers from the same commands, with the
    gallery size, profile and seed.
  - Attach the runs' `run.json`, or the `data/retrain/*.json` report from a retrain.
  - For a detector change, include the validation error and hit rate for both scene profiles
    it affects.
- **Leave `data/` and `models/` alone:**
  - Don't commit anything from `data/` or any exported bundle.
  - Don't commit new weights. `models/` changes only when the maintainer publishes a bundle
    trained from them, together with `SHA256SUMS` and the README table.
- **Keep the bundle format compatible with both apps:**
  - Run `mise run test` with the apps checked out beside Oracle, so the contract test runs
    instead of skipping.
  - A format change needs matching changes in both apps' recognition code (`recognition/` in
    each), merged before the new bundles are published.
- **App changes** follow each app's `AGENTS.md`, and should be checked against `cardid/bundle.py`
  (the reference runtime). Both apps' `recognition/pipeline.ts` and `recognizer.worker.ts` are
  ports of it.
- **Checks and commits:**
  - Before pushing Oracle changes, run `mise run test`, `uv run --no-sync ruff check cardid` and
    `uv run --no-sync ruff format --check cardid`.
  - Use [Conventional Commits](https://www.conventionalcommits.org) messages, as in every
    repository here.
- All three repositories are MPL-2.0; contributions are accepted under the same license.

## Publishing (maintainers)

Publishing is the only step that reaches users. Each app has its own model and profile, and one
command pulls captures, fine-tunes, evaluates and publishes:

```sh
mise run manavault    # phone scanner model → a scanner-bundle-* release on cfbender/manavault
mise run gathering    # webcam table model → The Gathering's server
mise run new-set      # a set released: refresh the gallery and ship it to both, no training
```

A run publishes only if the new bundle does at least as well as the published one on that app's
held-out real captures. See the README: "Two apps on one machine", "Trusted outlines for the
detector" and "One command". Rolling back a ManaVault bundle means unpublishing its release
(`gh release edit <tag> --repo cfbender/manavault --draft=true`). Servers reinstall the previous
bundle on their next check.
