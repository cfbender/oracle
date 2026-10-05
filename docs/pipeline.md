# How recognition works

A click (The Gathering) or a phone frame (ManaVault) goes through three ONNX graphs in the
browser:

```text
640 px window ─▶ detector ─▶ card quad + up ─▶ embed (one cut per frame) ─▶ search ─▶ top-5 printings
```

## Recogniser

A MobileNetV3-Small maps a card's art box to a 128-d unit vector. Identification is cosine
nearest-neighbour against one vector per distinct Scryfall illustration (~52k). Reprints and
translations share their artwork's vector, so adding a set means embedding its new art, not
retraining.

Scryfall's `art_crop` is a fixed template per card frame, so the query side cuts the same
templates out of the warped card. `detect.FRAMES` holds the boxes, measured by template-matching
art crops back into card scans: modern, the 1993/1997 frame, extended art, the tall art of
full-art basics and most tokens, the half-width art of sagas (right) and class/case cards (left),
plus the eight two-part regions below. Every query embeds all cuts in one batch, and each gallery
art is scored against the cut for its own frame (`index.frame_similarities`). The rare frames
(tall, saga, class and the two-part regions) pay `detect.FRAME_PENALTY` (0.02), because at webcam
quality they otherwise beat the true card by a hair. `--frame-penalty 0` on `evaluate` or
`capture` turns it off.

## Gallery

`cardid.scryfall` builds `data/arts.json` from Scryfall's all-language `all_cards` bulk file.

- Paper, non-digital printing faces are grouped by `illustration_id` (or the face ID when there
  is none). A group needs at least one `highres_scan`/`lowres` art crop; every supported paper
  sibling is then selectable, even one whose own scan is a placeholder.
- `printings` holds each sibling's ID, face name, set, collector number, language, border,
  Scryfall frame, frame effects and promo flag. Identical art cannot tell printings or languages
  apart, so the app shows the representative and the user picks a sibling. Searches accept
  `set:3ed`, `#40` and `lang:en`.
- Layouts: normal, leveler, saga, class, case, mutate, prototype, token, adventure, prepare and
  meld have one art. Transform, modal DFC, reversible and double-faced tokens contribute one
  entry per face with its own art crop. Split (Rooms, classic split, aftermath) and flip cards
  contribute one region per half. Art series, battles and three- or five-part novelty splits are
  excluded.
- IDs are the Scryfall UUID for face 0 and `<uuid>-1` for face 1. IDs and train/eval splits never
  change once assigned.
- Duplicate illustrations from older galleries stay as rows marked `alias_of`; only one
  embedding per illustration is exported. A held-out row wins over a training row so shared art
  cannot leak into training.
- Rows a newer rule rejects (playtest/sketch cards, decklists, bios and other bare `Card` inserts)
  stay with `"excluded": true` so nothing renumbers, and are dropped from training, evaluation,
  downloads and export. Their near-textureless art otherwise matched everything.

All downloads (`cardid.downloads`) are HTTPS-only, size-capped, content-type checked, validated
and renamed into place, so an interrupted run leaves no partial file.

### Two-part geometry

Scryfall's `normal` image shows Rooms and classic splits sideways in a portrait scan, and has no
per-half art crops. Rooms are recognised by `Room` in a face's type line and aftermath by its
keyword. Each half gets a region of the stored portrait scan, rotated upright. Classic split boxes
use the common interior of old and modern art windows; flip art is shared in the centre, so its
regions are left/right halves.

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

These eight are appended after the original six in `FRAME_NAMES`; the browser does not assume a
frame count, so older six-frame bundles still load. Regions are keyed
`<illustration_id>:face:0|1` because Scryfall's face metadata for these halves is inconsistent.
`evaluate_layouts` scores isolated art, clean scans and synthetic scenes per layout. It does not
measure which Room door is unlocked or which flip side is active; nothing infers that.

## Detector

`cardid.detector` looks at a 640 px window (downscaled to 256) and predicts the card's pose
(centre, short side, rotation) with small per-corner perspective residuals. Corners come from a
63×88 rectangle at that pose, so the aspect ratio is built in. A stride-4 corner heatmap
(CenterNet focal loss) then snaps each corner to a nearby peak with sub-pixel refinement; the pose
supplies ordering and a guaranteed answer, the heatmap supplies precision. Inference runs twice:
on the click window, then on a tight window around the first estimate.

The pose is symmetric under a 180° turn, so the head also predicts **up**, a vector towards the
printed top. The refined pass runs in four 90° rotations in one batch and sums their votes, and
`Detector.locate` returns the quad in printed order, so the recogniser always embeds an upright
card. The vote's length is a confidence; `capture` still embeds the 180° turn so `F` can flip.

Training scenes come from `cardid.synth`: full-card scans composited onto playmats, desks and
gradients, with sleeves, glare, borderless cards, neighbours, dice, fingers, any rotation and
mild perspective, then webcam (`table`) or phone (`phone`) photometrics. Two-part cards are 10%
of draws, split evenly by layout, because their title bars along the long axis can otherwise pull
the predicted rectangle 90° off. Labelled real captures with trusted outlines mix in with `--real`.

## Bundle format

`cardid.export` writes `data/bundles/<version>/`. Apps never see checkpoints.

| file | contents |
|---|---|
| `detector.onnx` | uint8 RGBA 256×256 window → `quad` (4×2, printed order), `up`, `centre`, `short` side; rotations, snapping and the up vote run inside the graph |
| `embed.onnx` | uint8 RGBA scene (any H×W) + quad → F×128 embeddings, one per frame cut; the warp is a `GridSample`, so no OpenCV in the browser |
| `search.onnx` | frames + embeddings → top-k gallery indices and cosine scores; the gallery (f16 by default) and frame penalty are baked in |
| `arts.json` | gallery order → `id`, `name`, `set`, `collector_number`, `layout`, `face`, `lang`, `frame`, `illustration_id`, crop `url`, `printing_count` |
| `printings.json` | representative ID → all selectable sibling printings; large, fetched only on the first search or printing expansion |
| `manifest.json` | version, checkpoint SHA256s, gallery size, every constant the glue code needs (from `cardid/constants.py`), per-file bytes and SHA256 |
| `SHA256SUMS` | what `publish` and the servers verify |

Export ends with a parity check (`--verify N`, default 64): it renders N synthetic scenes and
fails unless torch and onnxruntime agree on corners (median under 1 px) and on top-1 for ≥97% of
scenes where torch's top-1 leads by more than 0.02. Never hand-edit a published bundle; re-export
under a new version.

`cardid.bundle` is the onnxruntime reference runtime and the spec both apps port
(`recognition/pipeline.ts`): `uv run python -m cardid.bundle data/bundles/<version> --image
frame.jpg --click 660,350` prints the top 5 and per-stage timings. Its docstring describes the
glue around the three graphs. `cardid/test_manifest_contract.py` checks the manifest against the
apps' TypeScript in sibling checkouts (or `CARDID_CONSUMERS`).
