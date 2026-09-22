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

`train` and `evaluate` take `--device auto|cpu|cuda` (default `auto`; ROCm exposes AMD GPUs as
`cuda`). The `rocm` extra installs AMD's own PyTorch wheels for gfx1201 with the ROCm 10.0
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

Quad detection is classical for now (Canny + contour quads containing the click; the
outermost of the nested card-shaped quads is the card edge, the smallest is the inner frame
line) and stands in for the M2 detector. The stored `card.png` is the 250×350 warp, so the art
box can be re-cut with jitter at train time and `ART_BOX` can change without recapturing.

```sh
uv run python -m cardid.evaluate --method checkpoint --checkpoint data/runs/full/best.pt --real
uv run python -m cardid.train --resume data/runs/full/best.pt --real --epochs 4 --run full-real
```

`--real` mixes the train-split captures into every epoch (each repeated `--real-repeat` times,
default 20, with only light box jitter and colour changes — the camera already supplied the
resolution loss) and picks `best.pt` by top-1 on the held-out eval-split captures.

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
