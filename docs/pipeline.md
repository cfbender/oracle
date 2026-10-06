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
full-art basics and older tokens, the half-width art of sagas (right) and class/case cards (left),
the eight two-part regions below and the two token frames after them. Every query embeds all cuts
in one batch, and each gallery art is scored against the cut for its own frame
(`index.frame_similarities`). The rare frames (tall, token_tall, saga, class and the two-part
regions) pay `detect.FRAME_PENALTY` (0.02), because at webcam
quality they otherwise beat the true card by a hair. `--frame-penalty 0` on `evaluate` or
`capture` turns it off.

### Token frames

Scryfall's current token `art_crop` (684×570 or 684×722 px, about 60% of token arts) is wider than
any card's art box: 4.1–95.9% of the card's width from 11.5% down. By aspect alone it used to be
filed as `old` (1.20) or `tall` (0.947), whose cuts cover only about 65% and 83% of it, so a
token's true score on a clean scan was ~0.1 lower than an ordinary card's (0.88 against 0.99). `frame_of` now gives token layouts
the `token` frame (aspect 1.1–1.3, which also fits the older 630×550 and 613×498 templates better
than `old`) and `token_tall` (aspect 0.94–0.96). Older 0.92–0.93 templates stay `tall`, and
modern-frame double-faced tokens (1.37) stay `modern`. Both frames are appended after the two-part
ones, so a bundle's F grows from 14 to 16 (17 with `emblem`, below); `token_tall` pays the frame penalty like `tall`, and
`token` pays none, like `old`. The gallery embeddings do not change, only the query cuts.

Re-scoring the `retrain-20261005T145643894996Z` ManaVault bundle with these cuts (same weights,
fake quads so its embed graph cuts the token box) on 200 random token scans and 260 other cards
rendered as phone scenes raised token top-1 from 0.81 to 0.87 and the true token score from 0.75
to 0.82. Other cards were unchanged (0.913).

Emblems (layout `emblem`, which ManaVault scans as token fronts) have their own crop templates,
measured by template-matching all 106 distinct paper emblem illustrations into their scans. 92
are 603×576 (aspect 1.047) at x 0.098–0.908, y 0.119–0.674; 9 older ones are 602×605 (0.995) at
x 0.096–0.906, y 0.143–0.725; 5 use the 684×570 token template. By aspect alone the first two
would be `tall`, which covers them poorly (IoU 0.72 and 0.75). Emblems with an aspect of
0.97–1.1 get the `emblem` frame (the 603×576 box; IoU 0.87 for the older template), appended
after `token_tall` and penalised like `tall`; the rest follow the token rules. Emblems count as
tokens for `--token-repeat`.

### Hub arts

The same experiment showed why tokens went to Funeral Room: its `room_0` half is a hub. The
`room_0` cut of an ordinary card is a sideways strip of art and text box, and these strips look
alike from card to card. Funeral Room's art sits nearest their centre: it was in the top 5 for
12% of all queries (a typical art: 0.007%), and its mean top-10 similarity to other cards' cuts
was the highest of all 51k arts (0.69 against a median of 0.33). Ruin (`aftermath_1`) and a few
sagas and class cards are smaller hubs. The flat 0.02 penalty does not offset a hub that strong.
It used to win only when the true score was low, which describes tokens under the old cuts and
upside-down detections. Tokens were also read upside down more often (5% of phone scenes,
against 0.5% for other cards), and then the cuts land on aftermath hubs such as Road.
**Hub penalty.** Export applies a CSLS-style correction (`cardid.hubs`). Card scans stand in for
queries: for each gallery art, r is the mean of its 10 highest similarities to other cards'
cuts for its frame, and its own card never counts. The art pays a quarter of its r above the
gallery median, clipped to [0, 0.1]. The penalty is added to the frame penalty in `search.onnx`
and in `ArtIndex` (so `--verify` compares like with like). It only subtracts, so scores stay at
or below the cosine similarity, and the graph's inputs and outputs do not change.

The scans are `data/cards` (`scryfall --cards 3000`, the detector's scans; at most 3000, a seeded
sample beyond that), embedded with the checkpoint being exported, so the penalty follows the
model. Export prints the most penalised arts. The manifest records what was applied as
`gallery.hub_penalty`: `weight`, `neighbours`, `cap`, `cards`, `cards_fingerprint` (hash of the
scan names), `median_r`, `penalised` (arts paying > 0) and `max`. With fewer than 200 scans, or
`--hub-weight 0`, export warns and ships no hub penalty, and `hub_penalty` is `null`.

Measured on the same synthetic phone scenes as the token frames, with the published weights, 1000
other random cards as the background and the token frames on:

| | before | with hub penalty |
|---|---|---|
| Funeral Room in the top 5, all 2,120 queries | 11.6% | 3.1% |
| token top-1, phone scenes (600) | 0.870 | 0.870 |
| other cards top-1, phone scenes (780) | 0.913 | 0.917 |
| room and aftermath cards top-1, phone scenes (210) | 0.833 | 0.848 |
| correct with score ≥ 0.75: tokens / other cards | 0.790 / 0.827 | 0.785 / 0.822 |

The largest penalties went to Case of the Burning Masks and Funeral Room (0.077) and then sagas
(0.06–0.07); about half the gallery pays something, mostly a few thousandths.

## Gallery

`cardid.scryfall` builds `data/arts.json` from Scryfall's all-language `all_cards` bulk file.

- Paper, non-digital printing faces are grouped by `illustration_id` (or the face ID when there
  is none). A group needs at least one `highres_scan`/`lowres` art crop; every supported paper
  sibling is then selectable, even one whose own scan is a placeholder.
- `printings` holds each sibling's ID, face name, set, collector number, language, border,
  Scryfall frame, frame effects and promo flag. Identical art cannot tell printings or languages
  apart, so the app shows the representative and the user picks a sibling. Searches accept
  `set:3ed`, `#40` and `lang:en`.
- Layouts: normal, leveler, saga, class, case, mutate, prototype, token, emblem, adventure,
  prepare and meld have one art. An emblem that reuses its planeswalker's `illustration_id`
  (9 of 105) is keyed `<illustration_id>:emblem`, so it keeps its own crop and `emblem` row. Transform, modal DFC, reversible and double-faced tokens contribute one
  entry per face with its own art crop. Split (Rooms, classic split, aftermath) and flip cards
  contribute one region per half. Art series, battles and three- or five-part novelty splits are
  excluded.
- IDs are the Scryfall UUID for face 0 and `<uuid>-1` for face 1. IDs and train/eval splits never
  change once assigned.
- Duplicate illustrations from older galleries stay as rows marked `alias_of`; only one
  embedding per illustration is exported. A held-out row wins over a training row so shared art
  cannot leak into training.
- Rows a newer rule rejects stay with `"excluded": true` so nothing renumbers, and are dropped
  from training, evaluation, downloads and export. Their near-textureless art otherwise matched
  everything. The rule covers playtest/sketch cards, and cards with a face typed as the bare word
  `Card` that are not game pieces: memorabilia (decklists, bios, ads), minigames, checklists and
  the double-faced substitute. Bare-`Card` helpers from token sets (On an Adventure, The Monarch,
  City's Blessing, Day // Night, The Ring Tempts You) stay. A flagged row that becomes usable
  again loses the flag and keeps its index and split.

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
| `search.onnx` | `embeddings` (F×128) [+ `mask`] → `indices`, `scores`: top-k gallery indices and cosine scores; the gallery (f16 by default), per-art frames and frame penalty are baked in |
| `arts.json` | gallery order → `id`, `name`, `set`, `collector_number`, `layout`, `face`, `lang`, `frame`, `illustration_id`, crop `url`, `printing_count` |
| `printings.json` | representative ID → all selectable sibling printings; large, fetched only on the first search or printing expansion |
| `manifest.json` | version, checkpoint SHA256s, `gallery` (arts, dtype, embed_dim, frame_penalty, topk), every constant the glue code needs (from `cardid/constants.py`, with `frame_names`), `search_mask`, per-file bytes and SHA256 |
| `SHA256SUMS` | what `publish` and the servers verify |

**Search mask.** Exported with `--search-mask` (or `CARDID_SEARCH_MASK=1`, set in the ManaVault
profile), `search.onnx` takes a second required input after `embeddings`: `mask`, float32 of
shape `[N]` with N = `gallery.arts`, in `arts.json` order. Arts with a value above 0 compete; the
rest score −3 (`graphs.EXCLUDED_SCORE`, below any real score) before top-k. Ones reproduce the
unmasked graph exactly, and a client filters, for example tokens only, by setting ones where
`arts[i].layout` is `token`, `double_faced_token` or `emblem`. When fewer than k arts are kept, the extra
results have score −3 and should be dropped. The manifest's `search_mask` (`true`/`false`, absent in
older bundles) says which graph a bundle has, and so do the session's input names. Without the
flag the graph keeps its single `embeddings` input, which The Gathering feeds.

Export ends with a parity check (`--verify N`, default 64): it renders N synthetic scenes and
fails unless torch and onnxruntime agree on corners (median under 1 px) and on top-1 for ≥97% of
scenes where torch's top-1 leads by more than 0.02. Never hand-edit a published bundle; re-export
under a new version.

`cardid.bundle` is the onnxruntime reference runtime and the spec both apps port
(`recognition/pipeline.ts`); `Bundle.rank(embeddings, mask)` feeds ones when the graph takes a mask: `uv run python -m cardid.bundle data/bundles/<version> --image
frame.jpg --click 660,350` prints the top 5 and per-stage timings. Its docstring describes the
glue around the three graphs. `cardid/test_manifest_contract.py` checks the manifest against the
apps' TypeScript in sibling checkouts (or `CARDID_CONSUMERS`).
