# Card recognition spike (M0)

Offline tooling for the webcam table's click-to-identify feature. Nothing here runs in the
Phoenix app; it produces numbers (go/no-go) and, eventually, a model file the app loads.

The recognizer is a metric-learning CNN: a MobileNetV3-Small backbone maps the art box of a
card to a 128-d unit vector, and identification is cosine nearest-neighbor against one vector
per Scryfall unique artwork (~49k). Adding a set means embedding its art, not retraining.

## Setup

```sh
cd ml
uv sync --extra cpu                            # CPU-only torch from the pytorch index (see GPU training below)
uv run python -m cardid.scryfall --train 5000 --eval 1000   # bulk metadata + art_crop sample
uv run python -m cardid.degrade <art_id>       # visual check of the synthetic webcam degradation
```

torch lives in the mutually exclusive `cpu` and `rocm` extras, so `uv sync` needs one of them
once; later `uv run` calls keep whatever is installed. A plain `uv sync` (no extra) removes
torch again.

## Full catalog (bigger machine)

The 6k sample is enough to compare methods; the production gallery is ~49k artworks, which
is a harder retrieval problem. On a machine with the cores/RAM for it:

```sh
uv run python -m cardid.scryfall --all              # +43k art crops as train, ~80 min at 10 req/s
uv run python -m cardid.train --epochs 16 --batch 256 --run full   # workers/threads default to the core count
uv run python -m cardid.evaluate --method checkpoint --checkpoint data/runs/full/best.pt --profile realistic
```

The eval split stays the same 1,000 arts, so numbers are comparable with the sample runs;
the gallery grows to every downloaded art. First `gallery_images` call decodes all JPEGs
(a few minutes) and caches `data/gallery-<n>.npy` (2.4 GB) for later runs.

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
uv run python -m cardid.export --checkpoint data/runs/m0/best.pt     # ONNX for browser/server
```

## Real webcam captures (label → train → evaluate loop)

`cardid.capture` is the click-to-identify loop without the video call: it opens your camera
at 1080p in the browser, and every click sends a full-resolution crop to a local server that
finds the card quad, warps it, cuts the art box, embeds it and shows the top-5. Confirming a
candidate (or searching the right name) stores the capture under `data/real/` as training
and evaluation data.

```sh
uv run python -m cardid.capture --checkpoint data/runs/full/best.pt     # then open http://localhost:8765
```

Keys: `1`–`5` confirm a candidate, `/` search by name, `S` skip, shift-drag a box around the
card when the automatic quad is missing or wrong. The header shows running top-1/top-5 over
what you have labeled and the server/round-trip milliseconds per click. Captures are split
80/20 into train/eval by a hash of their id, so relabeling never moves a sample.

Without `--detector`, quad detection is classical (Canny + contour quads containing the
click; the outermost of the nested card-shaped quads is the card edge, the smallest is the
inner frame line). It fails on busy playmats, sleeves and borderless cards; see the detector
section below. The stored `card.png` is the 250×350 warp, so the art box can be re-cut with
jitter at train time and `ART_BOX` can change without recapturing. Each label also stores the
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
(the decoder starts fresh, everything else warm-starts).

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

## Coral Edge TPU export (run on the host with the accelerator)

1. `uv add ai-edge-torch` and convert `Embedder` to TFLite with full-int8 post-training
   quantization, using a few hundred degraded crops as the calibration set.
2. `edgetpu_compiler model.tflite` — check the log says every op mapped to the TPU.
3. Runtime: `tflite_runtime` (feranick builds for Python 3.12) + `libedgetpu1-max`, load with
   `tflite.load_delegate("libedgetpu.so.1")`; `pycoral` itself is not needed.
