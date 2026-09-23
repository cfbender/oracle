# Card recognition spike (M0)

Offline tooling for the webcam table's click-to-identify feature. Nothing here runs in the
Phoenix app; it produces numbers (go/no-go) and a versioned runtime bundle (three ONNX graphs
plus the gallery index, see [Shipping](#shipping-export-publish-refresh)) that the browser loads.

## One command

From the **repository root** on the Linux/RX 9070 XT desktop (existing gallery and training
checkpoints required), first-time setup:

```sh
mise install && mise exec -- uv sync --project ml --extra rocm
mkdir -p ~/.config; test -e ~/.config/cardid.env || install -m 600 ml/nightly.env.example ~/.config/cardid.env
${EDITOR:-nano} ~/.config/cardid.env
```

Set `CARDID_SERVER=https://your-server`, `CARDID_CORRECTIONS_TOKEN` (the server's read-only
export token), and `CARDID_PUBLISH_TO=user@host:/srv/the-gathering/cardid`. `rsync` and SSH
access are required. The file uses literal `KEY=value` assignments; quote spaces, do not use
shell expansion. Existing environment values override the file, and flags override both.
`CARDID_ENV_FILE` / `--env-file` selects another file. Tokens never appear in command logs.

```sh
mise run ml:retrain -- --dry-run             # resolve models and inspect the plan first
mise run ml:retrain                         # pull → gallery refresh → 4 epochs → export/verify → evaluate → publish
mise run ml:retrain -- --detector-epochs 4    # also fine-tune the resolved detector
mise run ml:retrain -- --no-publish          # train/export/evaluate for review, leave server and nightly state alone
mise run ml:retrain -- --from-dir /mnt/cardid/corrections --no-update-gallery --epochs 2
mise run ml:evaluate -- --method checkpoint --checkpoint data/runs/full-3/best.pt --profile realistic
mise run ml:export -- --checkpoint data/runs/full-3/best.pt --detector data/runs/det4/last.pt
mise run ml:publish -- data/bundles/<version> --to nuc:/srv/the-gathering/cardid
mise run ml:update-gallery
mise run ml:test
```

All tasks run in `ml/`, forward arguments after `--`, and preserve the installed torch extra
with `uv run --no-sync`. Use `--extra cpu` instead of `rocm` for CPU-only setup. The step tasks
are thin wrappers; **only `ml:retrain` reads the env file and orchestrates a gated publication**.
Manual commands below remain useful for individual experiments.

Retrain prefers SHA256 matches to the server's `current/manifest.json` across
`data/runs/*/{best,last}.pt`. If unreachable, it warns and uses the newest local bundle
manifest by mtime. If no checkpoint matches, it warns and selects the newest checkpoint of
the right model type (tensor names, not run-directory names). `--checkpoint` / `--detector`
always win; old `CARDID_CHECKPOINT` / `CARDID_DETECTOR` paths are additional search hints,
not overrides of the published hashes. To use an unpublished detector experiment explicitly:
`mise run ml:retrain -- --detector data/runs/det-two-part/last.pt`.

Usable, gallery-backed train captures enable `--real`; otherwise training is synthetic-only.
Train captures without real eval captures still mix into training, with synthetic checkpoint
selection. Both trainers use their new run's `best.pt`, or `last.pt` if no epoch beats the
starting model. `--epochs` defaults to 4 (`CARDID_RETRAIN_EPOCHS`); nightly's `CARDID_EPOCHS=2`
does not change it. `--workers` overrides `CARDID_WORKERS`, otherwise trainers choose for the
device. Other trainer hyperparameters use their normal defaults, not nightly's reduced rates.

Export always verifies 64 scenes. When held-out real captures exist, both ONNX bundles are
scored on the same raw crops using nightly's non-regression gate. Labels are resolved through
each bundle's `arts.json` plus `printings.json`, so a correction made through the printing
chooser (a sibling printing ID) counts as its shared artwork. Captures whose label only one
bundle knows (a printing that joined Scryfall after the baseline was exported, or one a later
gallery rule retired) are listed with a warning, recorded as `dropped_captures` in the report,
and left out of both scores; the run only refuses when no held-out label is common to both
bundles. A regression refuses
publication unless `--force`; missing labels, missing baselines, incomparable datasets,
export parity failures, changed labels or a changed server manifest cannot be forced.
Without held-out captures, retrain warns and permits publication with **no real accuracy
guarantee**; synthetic realistic top-1/top-5 are printed on every completed evaluation.
Use `--no-publish` to inspect results first. Unless `--no-publish` is set, retrain first checks
that `CARDID_PUBLISH_TO` is an existing directory (`ssh host test -d`, or a local `is_dir`) and
aborts before pulling or training if it is not, so a mistyped path fails in seconds rather than
after training. A missing `current/manifest.json` inside an existing destination means a first
publication: retrain resumes from the newest local bundle's checkpoints and publishes without
the `--expected-current` guard (it warns). Any other manifest fetch failure aborts, because
publication would fail anyway; `--no-publish` still falls back to the local bundle for offline
experiments.

Each run, including failures and dry runs, writes `data/retrain/<timestamp>.json` with commands,
selected paths, evaluation/gate results and publication status. Successful publication updates
nightly's checkpoint and detector paths without resetting its seen-corrections fingerprint.
Retrain shares nightly's lock, but has no automatic timer, nice level or two-hour time limit.
Do not run other trainers/importers concurrently. Dry runs only read local data/checkpoints
and fetch the current manifest; they do **not** pull corrections or refresh the gallery, so
the real run can choose `--real` differently after importing. Their shell-variable assignments
show the exact best/last choice that depends on future training output. No model scores are
invented. Reports and the lock are the only persistent dry-run writes.

The recognizer is a metric-learning CNN: a MobileNetV3-Small backbone maps the art box of a
card to a 128-d unit vector, and identification is cosine nearest-neighbor against one vector
per distinct Scryfall illustration (~52k). Each artwork carries its paper printing choices;
reprints and translations do not add duplicate embeddings. Adding a set means embedding
its new art, not retraining.

Scryfall's `art_crop` is a fixed template per card frame, so the query side cuts the same
templates out of the warped card: modern (0.08–0.92 wide from 0.115 down), old 1993/1997
frame, extended art, the tall art of full-art basics and most tokens, and the half-width art
of sagas (right half) and class/case cards (left half). `detect.FRAMES` holds the boxes,
measured by template-matching art crops back into card scans; each gallery art's frame is
read off its image aspect (plus the Scryfall layout for the half-width ones), every click
embeds all frame cuts in one batch (14 with the two-part layouts below), and each art is scored against the cut for its frame
(`index.frame_similarities`). On clean card scans through the orb model this took sagas from
0.25 to 1.00 top-1, class cards from 0 to 1.00 and full-art lands from 0.92 to 1.00 with
modern cards unchanged; the extra cuts cost ~50 ms on the orb CPU. The rare frames (tall,
saga, class: 1–2% of the gallery) carry a prior: their score is the similarity minus
`detect.FRAME_PENALTY` (0.02), because at webcam quality a full-art Plains or a saga
beat the truth by 0.01 in real evals while on clean scans no rare-frame impostor came close.
`--frame-penalty 0` on `evaluate`/`capture` turns it off for comparison.

## Setup

```sh
cd ml
uv sync --extra cpu                            # CPU-only torch from the pytorch index (see GPU training below)
export UV_NO_SYNC=1                            # preserve the installed torch extra
uv run python -m cardid.scryfall --train 5000 --eval 1000   # bulk metadata + art_crop sample
uv run python -m cardid.degrade <art_id>       # visual check of the synthetic webcam degradation
```

torch lives in the mutually exclusive `cpu` and `rocm` extras, so `uv sync` needs one of them
once. Export `UV_NO_SYNC=1` for subsequent `uv run` calls so they retain that extra.
A plain `uv sync` (no extra) removes torch again.

## Full catalog (bigger machine)

The 6k sample is enough to compare methods; the full gallery is ~52k artworks, which
is a harder retrieval problem. On a machine with the cores/RAM for it:

```sh
uv run python -m cardid.scryfall --all              # remaining distinct art crops as train; existing JPEGs are reused
uv run python -m cardid.scryfall --metadata         # backfill face/lang/illustration and refresh printing siblings (no image downloads)
uv run python -m cardid.train --epochs 16 --batch 256 --run full   # workers/threads default to the core count
uv run python -m cardid.evaluate --method checkpoint --checkpoint data/runs/full/best.pt --profile realistic
```

The sampled eval split is preserved while the gallery grows to every downloaded art;
legacy duplicate illustrations are handled as described below. First `gallery_images` call decodes all JPEGs
(a few minutes) and caches `data/gallery-<fingerprint>.npy` for later runs. Pixel, embedding,
and eval-query caches include gallery IDs/order/splits, so a deduplication cannot reuse stale
targets. Older count-only caches can be removed after migration.

### Gallery coverage, printings and face IDs

The **all_cards** bulk (`data/all-cards.jsonl.gz`) includes every language, including
Japanese-only alternate art such as SOA #102 Abrade. `unique_artwork` chose just one printing
per illustration, sometimes extended/borderless instead of ordinary; `default_cards` still
omits most translations. Neither supplies every printing choice.

We group paper, non-digital printing faces by `illustration_id` (or the printing face ID
when Scryfall has no illustration ID). Each group needs at least one `highres_scan`/`lowres`
art crop. Every supported-layout paper sibling is selectable, even one with a placeholder
scan, provided that illustration has a good scan elsewhere. Its preview uses Scryfall's
image for that exact printing, which may itself be a placeholder. Artworks without any
usable scan remain absent. No English restriction is applied. This changes the webcam
recognizer's gallery/search, not the separate English-only deck printing picker/catalog.

`printings` holds each sibling's exact ID, face name, set, collector number, language,
border color, Scryfall frame (`scryfall_frame`), frame effects and promo flag. Detector
`frame` remains separate. The picker expands an artwork into printing choices; both
searches accept `set:3ed`, `#40`, `lang:en` (English results sort first). Identical art cannot
identify the printing or language automatically: the numbered match is the existing
representative and the user chooses a sibling. New artworks prefer ordinary English,
non-promo scans; existing representatives and their crops are never replaced.
Some illustrations also span different names (Killbots, renamed tokens and misprints);
those choices explicitly show their own names, and searches use each printing's name.

In the checked bulk snapshot, Sol Talisman has 29 printing choices (4 English), Essence
Channeler 12 (4 English), and Nettlecyst 32 (6 English), each still one artwork. Revised
(`3ed`) grows from 2 choices to all 1,223 records (306 English), Unlimited from 0 to 302.
Revised Serra Angel, Lightning Bolt and Llanowar Elves each have all four language printings.
Early-core paper/digital filters reject none: 895 Revised records have placeholder scans,
but share scanned art and are now selectable. Alpha/Beta are included too.

- Single-art layouts: normal, leveler, saga, class, case, mutate, prototype, token,
  adventure, **prepare**, and **meld**. Prepare shares a top art box like adventure:
  Studious First-Year // Rampant Growth has one entry with the full combined name, not
  a second Rampant Growth artwork. Adventures were already supported.
- Separate-side layouts: **transform, modal_dfc, reversible_card, double_faced_token**.
  Each face with its own `image_uris.art_crop` becomes an entry using that face's name.
  STX #325 thus supplies both Jadzi, Oracle of Arcavios and Journey to the Oracle.
- Same-surface halves: **split** (Rooms, classic split, aftermath) and **flip**. Each half
  gets its own name/ID and a region cut from the shared `normal` scan; see the geometry below.
- Still excluded: **art_series**, **battle**, and novelty split cards with three or five parts.
  Representative prepare and meld scans have conventional top art boxes; inspected meld
  results include Ragnarok, Divine Deliverance and Mishra, Lost to Phyrexia.

IDs are the original Scryfall UUID for face 0 and `<uuid>-1` for face 1. Single-art entries
also record `face: 0`; all entries record `lang` and retain the card's `layout`. Missing
front images never renumber the back. Reversible cards use the face's oracle ID when the
parent has none. No ID or train/eval split changes for existing arts. Each side's frame is
classified from **its own crop aspect**: ordinary transform/MDFC backs use `modern`, while
extended/showcase, saga and token faces retain the corresponding existing frame. Inclusion
does not guarantee webcam accuracy for every unusual frame treatment.

An old gallery can contain duplicate illustrations (for example the same reverse side
paired with different fronts). Migration keeps every persisted row/ID/split, marks
duplicates with `alias_of`, and loads/exports one embedding row per illustration. A held-out
row wins over a training row so that shared artwork cannot leak into synthetic training.
All old and sibling printing labels map to the retained row. Exported indices can therefore
shift once during migration; every graph and `arts.json` is rebuilt together, never mixed
across versions. On the checked snapshot, 51,158 old face rows become 52,041 artwork rows
with 533,015 selectable printing faces. This is metadata growth, not 533k art downloads.

The API accepts the same gallery IDs for details and rulings, rejects malformed suffixes,
and keeps face-aware cache keys. Details select the face's image and rules; rulings are
fetched from the base card. Correction labels retain the suffix through export/import into
`data/real`. Exact sibling labels are preserved; real training opens the representative's
JPEG and real/nightly evaluation scores artwork identity, not whether an indistinguishable
printing was guessed. Only capture IDs remain plain UUIDs.

### Rooms, split/aftermath and flip geometry

Scryfall's `normal` image puts Rooms and classic splits sideways in a **portrait** scan.
Rooms are distinguished by `Room` in a face's type line, aftermath by the `Aftermath`
keyword, never by name or image aspect. `layout_group` preserves that distinction in
training metadata; `layout` remains Scryfall's original `split` or `flip`.

The checked all-language bulk contains no individual image URIs for these halves. Its
`art_crop` joins both artworks (or contains the shared central flip illustration). Most
records omit face 1's illustration ID; 26 repeat face 0's ID on face 1. Therefore these
regions use `<shared illustration_id>:face:0|1` as artwork keys, falling back to printing
identity when unavailable. This prevents translations with inconsistent face metadata from
collapsing the halves. Public gallery IDs remain `<printing UUID>` and `<printing UUID>-1`.
The source `url` is the whole `normal` image; only the downloaded training JPEG is cropped.

Measured on 488×680 scans of DSK #67 Mirror Room, APC #128 and MH2 #290 Fire // Ice,
DGM #135 Wear // Tear, AKH #211 Commit // Memory and #223 Cut // Ribbons, and CHK #202
Budoka Gardener and #131 Nezumi Shortfang. Fractions below refer to the stored portrait
scan. Classic split boxes deliberately use the common **interior** of old and modern art
windows, avoiding their differently positioned type bars. Flip art is shared in the center:
the two regions are left/right, not the upper/lower rules boxes. Unusual showcase treatments
are included but not individually calibrated.

| Frame | x0, y0, x1, y1 | Rotate crop upright |
|---|---|---|
| room_0 | .135, .485, .535, .910 | 90° clockwise |
| room_1 | .135, .050, .535, .475 | 90° clockwise |
| split_0 | .160, .565, .490, .900 | 90° clockwise |
| split_1 | .160, .095, .490, .430 | 90° clockwise |
| aftermath_0 | .075, .115, .925, .335 | none |
| aftermath_1 | .550, .565, .830, .915 | 90° counter-clockwise |
| flip_0 | .085, .315, .490, .655 | none |
| flip_1 | .510, .315, .915, .655 | 180° |

The original six frames retain their order; these eight append to `FRAME_NAMES`. Native
gallery cuts, Python queries, real-correction training and the exported embed graph apply
the same rotation. The browser passes the graph's tensor through without imposing a frame
count/order, so old six-frame bundles still load. New frames inherit the 0.02 rare-frame
penalty provisionally: they are less than 1% of the gallery and otherwise provide extra
opportunities for false matches. Tune this with real corrections, not clean scan scores.

In the inspected snapshot these layouts add **362 regions / 3,004 printing faces**:
Rooms 78/512, classic split 170/1,570, aftermath 62/818, flip 52/104. These are metadata
coverage counts, not an accuracy guarantee. Battles and art series remain excluded.

`synth` renders the actual full-card scans, preserving both artworks, text and orientations.
`render_scene(target_index=...)` lets evaluation choose the target without changing the
scene distribution. `evaluate_layouts` reports isolated-art retrieval separately from
full-card retrieval with known corners and optionally detector corners. Both halves are
valid card-identity targets in full scenes; these numbers do **not** measure which Room
door is unlocked or which flip ability is active. No such state is inferred by the app.
The old-crops control generously tries all six previous cuts against the new gallery.

Smoke evaluation (2026-09-23): `m0-arc/best.pt`, **not** production `full-3`, against the
original 6,000 arts plus all 362 new regions. Seed 2026, three realistic degraded queries
per art; twelve full scans × three synthetic scenes per group. Values are top-1/top-5:

| Layout | Isolated art (queries) | Clean scans | Scenes, known corners | Scenes, old cuts | Scenes, smoke detector |
|---|---|---|---|---|---|
| Room | 72.2% / 89.7% (234) | 100% / 100% | 75.0% / 77.8% | 0% / 5.6% | 5.6% / 5.6% |
| Split | 79.4% / 89.8% (510) | 91.7% / 100% | 58.3% / 72.2% | 2.8% / 2.8% | 0% / 0% |
| Aftermath | 60.2% / 81.2% (186) | 100% / 100% | 61.1% / 72.2% | 0% / 0% | 0% / 19.4% |
| Flip | 79.5% / 91.7% (156) | 100% / 100% | 55.6% / 66.7% | 11.1% / 22.2% | 25.0% / 38.9% |

The unchanged existing eval split (1,000 arts, 3,000 queries) scores 77.23%/88.03% with
6,000 gallery rows and 76.93%/87.63% with 6,362. These isolated-art queries do not measure
extra-frame false matches. No real corrections were available. The smoke detector is
`det-orb-up/last.pt`, not production `det4`. Geometry makes the new regions recognizable
without embedder retraining, but the low end-to-end smoke scores do **not** establish
production readiness. Evaluate the production pair before publishing; first investigate
detector corners/orientation if that gap remains, rather than blindly retraining the embedder.

For the full retraining cycle, use [One command](#one-command). To evaluate/export the
existing weights without training, run on the training box from `ml/` (use your actual paths):

```sh
uv sync --extra rocm
export UV_NO_SYNC=1
uv run python -m cardid.scryfall --update
uv run python -m cardid.scryfall --cards 3000
uv run python -m cardid.evaluate --method checkpoint --checkpoint data/runs/full-3/best.pt --profile realistic
uv run python -m cardid.evaluate_layouts --checkpoint data/runs/full-3/best.pt --detector data/runs/det4/last.pt
version="two-part-$(date -u +%Y%m%dT%H%M%SZ)"
uv run python -m cardid.export --checkpoint data/runs/full-3/best.pt --detector data/runs/det4/last.pt --version "$version" --verify 64
# Publish only after reviewing the evaluation (not performed by this code change):
uv run python -m cardid.publish "data/bundles/$version" --to nuc:/srv/the-gathering/cardid
```

For a small existing gallery, `evaluate_layouts --prepare-only` downloads only these new
regions without changing `arts.json` or its splits. Evaluation keeps its scans/report in
`data/layout-eval/`; remove that directory to refresh the bulk-derived evaluation selection.
If production evaluation warrants embedder fine-tuning, use
`uv run python -m cardid.train --resume data/runs/full-3/best.pt --epochs 4 --batch 256 --run two-part`,
then evaluate/export `data/runs/two-part/best.pt`. Add `--real` only after collecting usable
corrections; do not treat these synthetic measurements as real camera accuracy.
For detector fine-tuning after regenerating `data/cards`, use
`uv run python -m cardid.train_detector --resume data/runs/det4/last.pt --epochs 4 --samples 20000 --batch 64 --run det-two-part`,
then repeat evaluation with `--detector data/runs/det-two-part/best.pt` before exporting.

## GPU training (AMD RX 9070 XT / ROCm)

`train`, `train_detector` and `evaluate` take `--device auto|cpu|cuda|mps` (default `auto`,
which prefers `cuda`, then `mps`, then `cpu`; ROCm exposes AMD GPUs as `cuda`). On an Apple
silicon Mac the plain `--extra cpu` install already includes the Metal backend, so `auto` picks
`mps` with no extra setup; if an op turns out to be unsupported there, run with
`PYTORCH_ENABLE_MPS_FALLBACK=1` or `--device cpu`. The `rocm` extra installs AMD's own PyTorch wheels for gfx1201 with the ROCm 10.0
runtime bundled, so nothing but the `amdgpu` kernel driver is needed on the host
([AMD install page](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html);
Python 3.11–3.14, Linux only, ~1.5 GB download):

```sh
uv sync --extra rocm
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run python -m cardid.train --epochs 16 --batch 256 --run full       # prints "device: cuda (...)"
```

On GPU the model step is no longer the bottleneck, so `--workers` defaults to all cores but
one and `--threads` to 2; augmentation throughput is what limits batches/s. Checkpoints are
saved as CPU tensors either way, so `capture`, `bench` and `export` keep running on a
CPU-only install. `cardid` sets `TORCH_BLAS_PREFER_HIPBLASLT=0` before importing torch
because AMD lists GPU resets during training on the RX 9070 series as a known issue with the
hipBLASLt backend; export `TORCH_BLAS_PREFER_HIPBLASLT=1` to try the faster path.

If an epoch is slower than expected, `uv run python -m cardid.bench_loader --workers 15 8`
times the augmentation loader, the model step, and the post-epoch eval separately; whichever
is closest to the training loop's batch/s is the bottleneck. Pipeline experiments must be
modules like this rather than stdin scripts: Python 3.14 starts DataLoader workers with
`forkserver`, which re-imports `__main__` from its file path and fails for `<stdin>`.

To switch back to CPU wheels on the same machine: `uv sync --extra cpu`.

## Evaluate

Gallery = clean views of every downloaded art; queries = degraded views of the *eval* split,
which training never sees, so the number measures generalization to unseen sets.

```sh
uv run python -m cardid.evaluate --method dhash            # ManaVault-style baseline
uv run python -m cardid.evaluate --method phash --hash-size 16
uv run python -m cardid.evaluate --method pretrained       # ImageNet features, no training
uv run python -m cardid.evaluate --method checkpoint --checkpoint data/runs/m0/best.pt
```

`margin_for_99pct_precision` is the best-minus-second-best similarity above which 99% of
answers are right; `coverage_at_99pct_precision` is how often a click clears it. The UI
shows one card above the margin and top-3 below it; it never shows "no match".

## Train

```sh
uv run python -m cardid.train --epochs 12 --batch 128 --run m0
uv run python -m cardid.bench --checkpoint data/runs/m0/best.pt      # per-click CPU latency
```

Exporting a checkpoint for the app is covered in [Shipping](#shipping-export-publish-refresh).

## Real webcam captures (label → train → evaluate loop)

`cardid.capture` is the click-to-identify loop without the video call: it opens your camera
at 1080p in the browser, and every click sends a full-resolution crop to a local server that
finds the card quad, warps it, cuts the art box, embeds it and shows the top-5. Confirming a
candidate (or searching the right name) stores the capture under `data/real/` as training
and evaluation data.

```sh
uv run python -m cardid.capture --checkpoint data/runs/full/best.pt     # then open http://localhost:8765
```

Keys: `1`–`5` confirm a candidate, `/` search by name (add a set code to narrow a basic or
staple with hundreds of printings, and a collector number to pick one of a set's many:
`forest fin`, `forest fin 280`; results show `#number`), `S` skip, `F` flip to the other
orientation's candidates (with `--detector`), shift-drag a box around the card when the
automatic quad is missing or wrong. Candidates from a non-modern frame say so ("right art"). The header shows running top-1/top-5 over
what you have labeled and the server/round-trip milliseconds per click. Captures are split
80/20 into train/eval by a hash of their id, so relabeling never moves a sample.

Without `--detector`, quad detection is classical (Canny + contour quads containing the
click; the outermost of the nested card-shaped quads is the card edge, the smallest is the
inner frame line). It fails on busy playmats, sleeves and borderless cards; see the detector
section below. The stored `card.png` is the 250×350 warp, so the art box can be re-cut with
jitter at train time and the frame boxes can change without recapturing. Each label also stores the
crop and the quad used, which is what the detector trains and evaluates on.

```sh
uv run python -m cardid.evaluate --method checkpoint --checkpoint data/runs/full/best.pt --real
uv run python -m cardid.train --resume data/runs/full/best.pt --real --epochs 4 --run full-real
```

`--real` mixes the train-split captures into every epoch (each repeated `--real-repeat` times,
default 20, with only light box jitter and colour changes — the camera already supplied the
resolution loss) and picks `best.pt` by top-1 on the held-out eval-split captures.

## Card detector (M2): where is the card under the click?

`cardid.detector` replaces the edge finder with a MobileNetV3-Small that looks at the 640 px
window around the click (downscaled to 256) and predicts the card's *pose*: centre, short side
and rotation, plus small bounded per-corner residuals for camera perspective. The corners are
derived from a 63×88 rectangle at that pose, so the card's aspect ratio is built into the
output rather than checked afterwards. The head keeps the feature map's geometry (stride-16 and
stride-32 maps fused at 16×16 and flattened) instead of average-pooling it, since pooling
learns scale but not position or orientation. The loss is the corner error under the best of
the four cyclic corner orderings (a rotated card has no privileged first corner) plus a direct
pose term: centre, log size and the (cos 2θ, sin 2θ) angle vector. The corner loss alone has a
local minimum with the card turned 90° (each corner moves only ~0.2 short sides, less than at
60°), and the first detector fell into it on about half of all cards; the angle-vector L2 is
convex in the raw outputs and pulls straight out of it.

Regressing coordinates through a flattened fully connected head has a precision floor: on
synthetic scenes the pose-only detector plateaued around a tenth of the short side however
long it trained. So the network also predicts a class-agnostic *corner heatmap* at stride 4
(64×64) from a small FPN-style decoder over the stem's stride-8 and stride-4 features, trained
with CenterNet's penalty-reduced focal loss (`--heat-weight`, default 0.2) against Gaussians
at the four corners. At inference each pose corner snaps to the strongest heatmap peak within
~12% of the short side, refined to sub-pixel by a soft-argmax over the peak's 3×3
neighbourhood; a corner with no peak nearby keeps the pose estimate. The pose supplies the
ordering, the 90° disambiguation and a guaranteed answer; the heatmap supplies the precision.
Inference runs twice: once on the click window, then on a tight window around the first
estimate. It always returns a quad — a wrong one still gives the recogniser a guess to rank,
which beats "nothing found". `--resume` accepts checkpoints from before the heatmap head
(the decoder starts fresh, everything else warm-starts) and before the up output (the two new
head rows start fresh).

The pose is symmetric under a 180° turn, so the head also predicts which way is *up*: a unit
vector from the card's centre towards its printed top edge (`--up-weight`, default 1.0,
trained on rendered scenes only since a stored real quad may have been identified upside
down). `Detector.locate` returns the quad in printed order — corner 0 is the card's top-left —
so `warp_card` produces an upright card and the recogniser embeds one orientation. The earlier
approach, embedding both rotations and keeping the more confident one, was the main source of
misses on real captures: a text box warped upside down matches a sky or a text-heavy art (White
Ward, Look at Me I'm R&D) at ~0.67 similarity, more than a real photo of the art matches its
own scan: on 15 eval captures 4 of the 7 misses had the truth top-1 in the rotation the
confidence rule rejected (top-1 0.53 against an orientation oracle of 0.80).

The trainer reports `up`, the share of validation scenes where the vector points into the
correct half plane, and `up_big`, the same over cards at least 90 px wide in the 256 px
input — the size the refined pass of `Detector.locate` sees, so that is the deployed number.
`up` plateaus around 0.94 because the remainder is small (median 43 px) and occluded cards;
at 90–140 px the head is wrong on ~3%, at 140+ px on ~1%. The length of the vector is a
confidence (|up| ≥ 0.85 is right 98.7% of the time, below 0.5 about 75%), so `locate` runs
the refined window in four exact 90° rotations in one batch (`rotations=4`), turns the ups
back and sums them: 97.3% → 98.3% right at that size, and `locate_up` returns the vote
(summed length / rotations; 0.75+ is right 99.7% of the time). The corners come from the
unrotated window. `capture` shows the vote, still embeds the 180° turn so `F` flips to the
other orientation's card and candidates when the detector was wrong, stores `card.png`
upright with `up_correct` and `up_vote` per label and shows the running rate ("up ok"), and
`evaluate --real --detector <ckpt>` counts how often the up output rejected the rotation in
which the truth was top-1.

Training data is rendered by `cardid.synth` from full-card images composited onto busy
backgrounds (random art crops as playmats, flat desks, gradients), with sleeves (a ring
outside the card, a milky tint and glare over it), synthetic borderless cards (the image cut
inside its border), neighbouring and overlapping cards, dice, fingers, any rotation and mild
perspective, then webcam photometrics. Labeled real captures (`data/real`) can be mixed in
with `--real`, augmented by re-windowing the stored crop at random rotation and scale.

```sh
uv run python -m cardid.scryfall --cards 3000                                  # 488x680 card images into data/cards (~5 min)
uv run python -m cardid.synth --n 16 --out /tmp/scenes.png                     # eyeball the rendered scenes
uv run python -m cardid.train_detector --epochs 10 --samples 20000 --batch 64 --run det
uv run python -m cardid.train_detector --resume data/runs/det/best.pt --real --epochs 4 --run det-real
uv run python -m cardid.capture --checkpoint data/runs/full/best.pt --detector data/runs/det-real/best.pt
```

The trainer reports the median corner error as a fraction of the card's short side and the
share of samples under 5% ("hit", inside the recogniser's crop-jitter tolerance) on a fixed
synthetic validation set (`synth` = after heatmap snapping, `synth_pose` = the raw pose head,
so the gap shows what the heatmap buys) and on the held-out real captures (`real` = single
pass on the click window, `real_e2e` = the two-stage `Detector.locate` capture uses). With
only a handful of labeled eval captures the `real*` numbers step in coarse increments and are
mostly noise; grow `data/real` before trusting them. When the median is good but the mean is
not, `inspect_detector` shows what the tail is made of — percentiles, the share of gross
failures and how many of those are the card turned 90°, whether snapping helped or hurt,
small vs large cards — and renders the worst validation cases with target, pose, snapped
prediction and heatmap overlaid:

```sh
uv run python -m cardid.inspect_detector --checkpoint data/runs/det3/best.pt --out /tmp/worst.png
```

The number that matters
is identification accuracy with detector quads instead of the stored ones:

```sh
uv run python -m cardid.evaluate --method checkpoint --checkpoint data/runs/full/best.pt --real --detector data/runs/det/best.pt
uv run python -m cardid.evaluate --method checkpoint --checkpoint data/runs/full/best.pt --real --detector classical
```

Rendering is ~24 ms per scene on one core (was 60), so 8 workers give roughly 330 samples/s
and a 20k-sample epoch is ~60 s of rendering; on CPU the model step dominates instead.
The first run decodes the card images and a fixed 1,500-art subset of the gallery at half
resolution into memory-mapped banks under `data/cache/` (`cards-*.npy`, `arts-*.npy`;
~1 GB for 3,000 cards, shared by all workers through the page cache). JPEG decoding is
entropy-bound, ~5 ms per image whatever the reduced-size flag, and was a quarter of the render
time; the banks lose nothing because the 640 px window is downscaled 2.5x for the detector.
Delete the cache directory to rebuild it after changing `data/cards` or `data/art`.
`python -m cardid.profile_synth` prints ms/scene and the cProfile hot spots of the renderer.

`bench_loader --detector` times the renderer single-threaded, the same render inside N plain
processes (what the CPU does under all-core load, separately from the loader), the DataLoader
at each `--workers` count, and the CornerNet step, which is the first thing to run when epochs
take longer than that arithmetic says. The default is one worker per logical CPU minus one; on
an SMT desktop (8 cores / 16 threads) that oversubscribes the physical cores and 8 workers
deliver more samples/s than 15, so sweep `--workers 6 8 10 12` in the bench and pass the winner
to `train_detector`. Guest VMs report SMT topology too but often scale like full cores, so the
trainer does not guess. If the samples/s barely move with the worker count, the main process
is the cap: batches are staged in pinned host memory on GPU runs, which is slow to allocate on
some ROCm setups, so compare with `--no-pin` (both tools take it). The datasets ship uint8
scenes and normalise on the device, so each sample is 196 KB through the queue rather than
786 KB.

## M0 results (2026-09-22, 6k-art gallery, 3,000 queries from 1,000 unseen arts)

Top-1 retrieval accuracy. "Harsh" is the training distribution (56–140 px art, 25% of
queries with heavy off-center perspective, dice/card occlusion); "realistic" assumes the
detector quad was warped, art ≥70 px. `cov@99` = share of clicks whose margin clears the
99%-precision threshold, i.e. the UI can show a single card instead of top-3.

| method | harsh top-1 | harsh top-5 | realistic top-1 | realistic top-5 | realistic cov@99 |
|---|---|---|---|---|---|
| dHash-64 (ManaVault approach) | 0.32 | 0.45 | 0.43 | 0.58 | 0.12 |
| pHash-256 | 0.26 | 0.32 | – | – | – |
| MobileNetV3 ImageNet features, no training | 0.37 | 0.50 | 0.49 | 0.62 | 0.10 |
| **`m0`: MobileNetV3-S + InfoNCE, 12 epochs, batch 128** | **0.73** | **0.85** | **0.84** | **0.93** | **0.66** |
| `m0-arc`: + ArcFace head, 16 epochs, batch 256 | 0.65 | 0.78 | 0.77 | 0.88 | 0.58 |

Realistic top-1 by simulated art width for `m0`: 70–79 px 0.72, 80–109 px 0.82, 110–140 px 0.89.
Each `m0` epoch took ~60 s on 8 Xeon cores; the ArcFace head over 5k classes memorised the
train split (loss → 0.06) without helping unseen arts, so the recommended recipe is InfoNCE only.

Per-click compute on one CPU thread: embed 4.3 ms (torch eager) / 0.9 ms (onnxruntime),
cosine top-5 over a 49k×128 float32 gallery 0.34 ms. Recognition compute is not the latency
budget; capture, upload and detection are.

Open questions this run cannot answer: accuracy against the full 49k gallery (harder), and
real webcam crops versus the synthetic degradation. Those are the next runs.

## Shipping: export, publish, refresh

For pull → retrain → evaluate → publish, use [One command](#one-command). These individual
commands remain available for gallery-only exports and manual review.

The app never sees checkpoints. It loads a **bundle**: a versioned directory of three ONNX
graphs plus the gallery index, built once on the training machine and copied to the server.

```sh
uv sync --extra rocm                                  # once: pulls onnxruntime for the parity check
uv run python -m cardid.export --checkpoint data/runs/full-3/best.pt --detector data/runs/det4/last.pt
uv run python -m cardid.publish data/bundles/2026-09-22-full-3 --to nuc:/srv/the-gathering/cardid
```

`export` writes `data/bundles/<version>/` (default version `<UTC timestamp>-<checkpoint run name>`,
for example `2026-09-23T171512Z-full-3`; override with `--version` or `--out`). Published versions
are immutable on the host, so `export` refuses to overwrite an existing bundle directory unless
you pass `--force` for one that was never published:

| file | contents |
|---|---|
| `detector.onnx` | uint8 RGBA 256×256 window → card `quad` (4×2, window px, printed order), `up` (2), `centre` (2), `short` side. Runs the four 90° rotations, corner snapping, orientation vote and pose inside the graph. |
| `embed.onnx` | uint8 RGBA scene (any H×W) + quad → F×128 embeddings (currently F=14), one per frame cut (`detect.FRAMES`). The projective warp is a `GridSample`, so no OpenCV is needed in the browser. |
| `search.onnx` | frames + embeddings → top-k gallery indices and cosine scores. The gallery (f16 by default, `--gallery-dtype f32`) and the frame prior (`--frame-penalty`, default 0.02) are baked in; `--topk` defaults to 5. |
| `arts.json` | gallery index order → `id`, `name`, `set`, `collector_number`, `layout`, `face`, `lang`, `frame`, `illustration_id`, crop `url`, `printing_count`. No nested siblings. |
| `printings.json` | representative art ID → all selectable sibling printing records. Downloaded only on the first gallery search or printing expansion, shared and cached by bundle version. |
| `manifest.json` | version, checkpoint sha256s, gallery size, every constant the glue code needs (scene 640, detector input 256, refine fill 0.6 / min side 64, card 250×350, art input 128, frame names, opset 17), per-file bytes + sha256. |
| `SHA256SUMS` | what `publish` and the server verify. |

Graph sizes at 49k arts: detector 12.6 MB, embed 5.1 MB, search ≈13 MB (f16).
All-language printing metadata is much larger than the former ~7 MB flat gallery; allow
roughly 140 MB for uncompressed sibling metadata and ~393 MB for the compressed all-card bulk.
It lives in `printings.json`, not the worker's initial `arts.json`. An explicit search or
printing expansion downloads this whole optional file once; ordinary identification does not.
The browser caches bundle files by version; the search graph still has only ~52k rows.

The table creates no recognition worker or model requests on room entry. The first card
click or gallery search starts it and waits for warmup, with a loading indicator. The
two-second inference timeout starts only after warmup. Settings > Card scan reports
"Not loaded" until then. No idle prefetch or dialog splitting is enabled.

**Regenerate and publish a new immutable version to get the metadata split.** Use the
existing export/publish commands below and existing checkpoints; no retraining is needed.
Do not hand-edit a published `arts.json`: export writes checksums for both metadata files
and preserves exactly the embedding index order and graph parity verification. Training
`data/arts.json` keeps its siblings for correction-label resolution. Old browser bundles,
with embedded siblings or only representative printings, still work without regeneration,
but cannot gain printing coverage they never contained.

`export` ends with a parity check (`--verify N`, default 64, `0` to skip): it renders N
synthetic scenes, runs the torch pipeline and the bundle through onnxruntime on each, and fails
(exit 1, bundle left on disk for inspection) unless median corner error is under 1 px and the
top-1 agrees on ≥97% of scenes where torch's top-1 leads the runner-up by more than 0.02.
Disagreements inside that margin are float16/rounding noise, not export bugs.

`cardid.bundle` is the onnxruntime reference runtime and the executable spec for the browser
port: `uv run python -m cardid.bundle data/bundles/<version> --image frame.jpg --click 660,350`
prints the top-5 and per-stage timings. The glue a JS runtime writes around the three graphs
is in its module docstring (window resample → detector pass 1 → refine pass → embed → search).
On the orb CPU one click costs ~43 ms detector (both passes), 23 ms embed, 5 ms search.

`publish` verifies `SHA256SUMS`, tars the bundle and streams it over ssh (or copies to a local
directory) into `<path>/.incoming`, re-verifies the sums on the server, moves the version into
place and atomically repoints `<path>/current` at it, then prunes old versions beyond `--keep`
(default 3). A failed transfer or a tampered file never touches `current`. The Phoenix side
serves `DATA_DIR/cardid/current/*` and the browser caches by `manifest.json` version.

`publish failed on <host>` is the wrapper; the reason is the `publish: ...` line printed just
above it by the host: the version directory already exists (re-export with a new `--version`
or remove the leftover on the host), `current` is a copied directory instead of a symlink (move
it to `<path>/<version>` and `ln -s <version> current`), or, from the nightly, `current` changed
since the candidate was evaluated. The host needs `bash`, `flock`, `tar` and GNU coreutils; the
script is run under `bash -c` so the login shell does not matter.

### New sets

New cards need gallery embeddings, not retraining (the recogniser learned "compare arts", not
"these arts"). When a set releases:

```sh
uv run python -m cardid.scryfall --update        # fresh bulk file, new arts appended to data/arts.json, only new images downloaded
uv run python -m cardid.export --checkpoint data/runs/full-3/best.pt --detector data/runs/det4/last.pt
uv run python -m cardid.publish data/bundles/<version> --to nuc:/srv/the-gathering/cardid
```

`--update` keeps the previous bulk file as `all-cards.jsonl.gz.previous`, refreshes sibling
metadata, preserves existing art IDs and splits, and appends new illustrations as train.
The one-time duplicate-illustration migration described above removes aliases from the
exported index, not from `data/arts.json`. Retrain (`train
--real`) only when real-capture accuracy drifts, e.g. a new frame style the six cuts miss.

### Apply the expanded gallery to an existing deployment

This code change **does not regenerate the deployed bundle**. On the training box, from
`ml/`, use the existing checkpoints (no retraining required) and a fresh immutable version:

```sh
uv run python -m cardid.scryfall --update
version="gallery-$(date -u +%Y%m%dT%H%M%SZ)"
uv run python -m cardid.export --checkpoint data/runs/full-3/best.pt --detector data/runs/det4/last.pt --version "$version"
uv run python -m cardid.publish "data/bundles/$version" --to nuc:/srv/the-gathering/cardid
```

Use your currently published checkpoint/detector paths if different (`CARDID_CHECKPOINT`,
`CARDID_DETECTOR`, and `CARDID_PUBLISH_TO` in the nightly environment name these settings;
after a successful nightly run, its resume checkpoint is in `data/nightly/state.json`).
`--update` re-fetches the all-language metadata, backfills existing rows, refreshes printing
siblings, appends newly usable illustrations as train, and downloads only missing art files.
**No full art re-download or retraining is needed**: existing IDs/JPEGs are reused even
when a regular printing is now preferred for fresh galleries. Check its failed-download
count and rerun if needed before exporting. `--metadata` alone never adds artworks or
downloads art, but does refresh siblings from its cached bulk file.
Export's parity check, manifest and checksums are unchanged.

Both also retire rows that a newer `usable` rule rejects (`scryfall.hub_card`): Mystery
Booster / Playtest sketch cards (`promo_types` containing `playtest`, any layout) and
non-game inserts typed as a bare `Card` (World Championship decklists, bios and ads,
minigame cards) are near-textureless slabs whose embeddings sit close to everything, so
Bind // Liberate (cmb1) collected nine of sixteen real-camera misses and decklist cards most
wrong other-orientation hits. Such rows stay in `data/arts.json` with `"excluded": true`
so nothing renumbers, but they are dropped from training, evaluation, downloads and the
exported gallery. Rerun `scryfall --update` (or `--metadata`) and re-export to apply this to
an existing `arts.json`; no retraining is required.

### Training from in-app corrections

Every **explicit picker choice** (including confirming top-1, Shift+click, "Wrong card?",
and `/` search) can contribute a human-labelled crop. Clear automatic matches never do.
The small **Share card crops & picks for training** checkbox below the active board persists
in that browser's localStorage. Both the clicker and the camera owner must allow sharing;
older peers without a consent flag are excluded. Opting out affects future captures/uploads,
not samples already saved. Only a successful POST shows "Correction saved for training."

`POST /api/cardid/corrections` uses the signed-in session + CSRF and is limited to 30 requests
per user per minute. JPEG data URLs are capped at 190 KB encoded (the entire normal payload
is below 200 KB), dimensions at 640×640. The browser lowers JPEG quality if needed, without
downscaling. The server writes `DATA_DIR/cardid/corrections/labels.jsonl` and
`<capture_id>/crop.jpg`, plus internal owner/idempotency metadata. The log records the chosen
gallery Scryfall printing ID, click in crop pixels, ordered quad, up vote, original top-1,
similarity/margin, bundle version and deterministic capture-ID split. There is no video,
room ID, or player name in the export. Relabels append; last label per ID wins.

**The desktop importer creates `card.png`**, a 250×350 warp using `detect.warp_card`, beside
the original `crop.jpg` in `data/real`. No runtime Python/image decoder is added to Phoenix.
These are the exact image names `real.py` uses (not `crop.png`). The picker confirms identity,
not orientation or corners: `up_correct` is deliberately absent, `orientation=0` means the
detector's ordered quad was used. Review bad warps before detector training; do not mistake
model-generated quads for human geometry labels. Missing/degenerate quads remain pending
with no `card.png`, and `load_labels` excludes them until their geometry is repaired.

On the server, set `CARDID_CORRECTIONS_TOKEN` to a random secret (`openssl rand -hex 32`) and
`CARDID_CORRECTIONS_ADMIN_ID` to an enabled administrator's numeric user ID; the compose file
passes both through. This is a **read-only correction-export capability**, not a session or
general API token. Disabling/demoting that admin or rotating the token revokes it. Admin
cookie sessions can also export. The API returns up to 50 rows at
`GET /api/cardid/corrections?cursor=N` and JPEGs at `/api/cardid/corrections/:id/crop`.

Prefer [One command](#one-command) for the complete desktop loop. For individual steps on
the **Linux/ROCm desktop**, from `ml/`:

```sh
uv sync --extra rocm                              # once
install -m 600 nightly.env.example ~/.config/cardid.env  # create ~/.config first if needed
# Edit ~/.config/cardid.env: server URL, matching token, checkpoint, detector and SSH target.
set -a; . ~/.config/cardid.env; set +a
uv run python -m cardid.corrections pull
uv run python -m cardid.train --resume "$CARDID_CHECKPOINT" --real --epochs 2 --workers 2 --threads 2 --run corrections-1
uv run python -m cardid.evaluate --method checkpoint --checkpoint data/runs/corrections-1/best.pt --real
uv run python -m cardid.export --checkpoint data/runs/corrections-1/best.pt --detector "$CARDID_DETECTOR" --version corrections-1
uv run python -m cardid.publish data/bundles/corrections-1 --to "$CARDID_PUBLISH_TO"
```

HTTP pull requires HTTPS, keeps its cursor in `data/real/.corrections-cursor.json`, advances
only after a successful page, and deduplicates by capture ID/source record on retry. Existing
standalone captures are preserved. A relabel updates the existing capture without changing
its train/eval split; null labels remain skips. If restoring an older server backup, remove
the local cursor file to rescan; imports still deduplicate. Back up crops and labels together.
Keep this private dataset out of Git and restrict directory permissions on both machines.

For mounted storage or rsync, import rather than overwriting the desktop's label log:

```sh
rsync -a nuc:/srv/the-gathering/cardid/corrections/ data/corrections-inbox/
uv run python -m cardid.corrections pull --from-dir data/corrections-inbox
# Or: --from-dir /mnt/gathering/cardid/corrections
```

`publish` atomically switches `current`, retains the old target as `previous`, and prunes only
bundle directories, never corrections or either protected target. Versions are immutable:
use a new `--version` for each export. A legacy plain-directory `current` must first be moved
to its manifest version and replaced with a symlink. Phoenix serves the new bundle without
restarting; browsers pick it up on their next table load.

### Optional nightly loop (or the same guarded run by hand)

For the on-demand four-epoch cycle with gallery refresh and optional detector training, use
[One command](#one-command). Nightly deliberately retains its stricter correction-only gates.

`bash nightly.sh` runs pull → merge → resume training → export/parity check → held-out
evaluation → conditional publish. Unlike the individual commands above, it gates publication.
It deliberately does **not** run `scryfall --update`: gallery-only changes would otherwise
be hidden behind the no-new-corrections gate, bulk/art downloads spend the same two-hour
budget, and the baseline cannot score labels it does not yet contain. Refresh/export/publish
the gallery explicitly using the commands above before collecting corrections for newly
included faces. A future separate gallery-refresh job should have its own budget and
publication policy rather than silently changing this correction-training gate.
Requires `uv`, `rsync`, SSH access (or a mounted publish destination), GNU `timeout`, and
`flock` on the desktop; remote publication also requires `bash`, `flock`, tar and sha256sum.
No timer is enabled by installing or running this script.

```sh
bash nightly.sh --dry-run  # actually pulls/merges; exercises synthetic gate cases; never trains/publishes
bash nightly.sh            # identical to the scheduled run
```

Defaults: two epochs, two loader workers, two CPU threads, batch 64, reduced learning rates
(1e-4 head / 3e-5 backbone), nice 15 and a **two-hour wall-clock budget for the whole loop**.
See `nightly.env.example` for overrides and the filesystem alternative. ROCm still reports
device `cuda`. Nice limits CPU scheduling priority, not GPU usage; this is not GPU isolation.

The run snapshots the actual published bundle from `CARDID_PUBLISH_TO/current`, verifies its
checksums and that the resume/detector checkpoints match its manifest, then scores both the
baseline and candidate ONNX bundles **end-to-end from the same held-out raw crops**. This
tests deployed detection, orientation, embedding and gallery search, not training accuracy.
It requires both train and eval samples, rejects missing gallery labels/incomparable sets,
and publishes only when held-out top-1 is at least the baseline. A tiny eval set is noisy;
this is a non-regression check, not a statistical guarantee of improvement.

No new usable correction/relabel since the last completed run means **no training or publish**.
This also works if corrections were pulled manually earlier. Rejected candidates mark that
dataset as seen; failures/timeouts do not. Dry runs never mark data as trained. The current
best checkpoint and seen fingerprint live in `data/nightly/state.json`. Nightly now resolves
local checkpoints against the published manifest, so stale environment/state paths do not
block it after a manual publish; **it still refuses when no local model matches either hash**.
`ml:retrain` updates both paths automatically after publishing. Keep matching checkpoint files
on the desktop. Logs and exact correct/count/top-1 values live in
`data/nightly/YYYY-MM-DD.log` and per-run JSON reports. A lock prevents overlapping runs;
dataset changes or a changed published manifest abort publication. Keep previous checkpoints
and bundles for rollback. Do not run another label importer/trainer during this job.

Optional **user systemd units** assume checkout `~/the-gathering` (edit paths if different):

```sh
mkdir -p ~/.config/systemd/user
cp systemd/cardid-nightly.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user start cardid-nightly.service        # one run, no schedule
# Only if you decide to enable nightly runs:
systemctl --user enable --now cardid-nightly.timer
systemctl --user list-timers cardid-nightly.timer
journalctl --user -u cardid-nightly.service
# systemctl --user disable --now cardid-nightly.timer
```

The timer specifies `OnCalendar=*-*-* 04:00:00 America/New_York` and `Persistent=true`.
For execution while logged out, enable user lingering (`loginctl enable-linger "$USER"`).
Persistent timers catch up on the next boot/login, potentially during the day; disable
persistence if that is undesirable. The service stops all child processes after two hours.

Equivalent cron schedule (Cronie/cron with `CRON_TZ` support; no missed-run catch-up):

```cron
CRON_TZ=America/New_York
PATH=/home/cody/.local/bin:/usr/local/bin:/usr/bin:/bin
0 4 * * * /bin/bash /home/cody/the-gathering/ml/nightly.sh
```

Use your actual home path. Cron implementations without `CRON_TZ` must use a host timezone
of America/New_York; setting `TZ` only for the command does not change scheduling. Do not
enable both cron and the timer.

CPU-only checks: `uv run ruff check`, `uv run ruff format --check`, and
`uv run python -m cardid.corrections selftest` cover import/relabel/skip, interrupted HTTP
pulls, gate boundaries, and local/SSH-shell publication without a GPU or network server.

### Where to train

Training is the only heavy step and it is a batch job, so it does not need to live on the
server. The NUC has no GPU; at ~49k arts the 16-core desktop takes ~206 s per epoch on CPU, and
the NUC is a fraction of that machine, so expect 10–15 min per epoch (roughly an hour for a
`--real` fine-tune) — fine as a nightly job, too slow to iterate on. Measure before deciding:
`uv run python -m cardid.train --epochs 1 --run nuc-timing` prints seconds per epoch. The
recommended split is train on the desktop (or the 3080 Ti box with PCIe passthrough rather than
vGPU; that box needs a `cuda` torch extra in `pyproject.toml` alongside `cpu`/`rocm`, which
does not exist yet — `--extra cpu` installs CPU-only wheels) and `publish` to the NUC, which
only serves static files.

## Coral Edge TPU export (run on the host with the accelerator)

1. `uv add ai-edge-torch` and convert `Embedder` to TFLite with full-int8 post-training
   quantization, using a few hundred degraded crops as the calibration set.
2. `edgetpu_compiler model.tflite` — check the log says every op mapped to the TPU.
3. Runtime: `tflite_runtime` (feranick builds for Python 3.12) + `libedgetpu1-max`, load with
   `tflite.load_delegate("libedgetpu.so.1")`; `pycoral` itself is not needed.
