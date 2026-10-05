# Training and evaluation by hand

For the routine pull-train-publish loop use `mise run retrain-all` (see
[operations](operations.md)). This page covers the individual steps, for experiments.
Commands assume `export UV_NO_SYNC=1` after `uv sync --extra cpu` or `--extra rocm`; a plain
`uv sync` removes torch.

## Gallery

```sh
uv run python -m cardid.scryfall --train 5000 --eval 1000   # bulk metadata + a 6k art sample, enough to compare ideas
uv run python -m cardid.scryfall --all                      # every artwork (~52k), what ships
uv run python -m cardid.scryfall --metadata                 # refresh printings from the cached bulk file, no art downloads
uv run python -m cardid.scryfall --cards 3000 --two-part-cards   # full-card scans for detector scenes
uv run python -m cardid.degrade <art_id>                    # see the synthetic degradation
```

The eval split is never trained on, so its score measures unseen cards. Numbers from the 6k
sample and the full gallery are not comparable. The first run over a gallery decodes every JPEG
and caches `data/gallery-<fingerprint>.npy`; caches are keyed by gallery IDs, order and splits.

## Evaluate

```sh
uv run python -m cardid.evaluate --method checkpoint --checkpoint models/recogniser.pt --profile realistic
uv run python -m cardid.evaluate --method checkpoint --checkpoint models/recogniser.pt --real --detector models/detector.pt
uv run python -m cardid.evaluate --method dhash        # baselines: dhash, phash, pretrained (ImageNet features)
uv run python -m cardid.evaluate_layouts --checkpoint models/recogniser.pt --detector models/detector.pt
```

`--real` scores labelled captures in `data/real` end to end; `--detector classical` compares the
old edge finder. `margin_for_99pct_precision` is the best-minus-second similarity above which 99%
of answers are right, and `coverage_at_99pct_precision` is how often a query clears it.

**Deck-list prior.** The Gathering adds `DECK_PRIOR` (in its `deck-hint.ts`) to top-5 candidates
in the owner's linked deck list before its `CLEAR_MARGIN` rule. `--real --detector ... --deck-prior`
replays that over a sweep of priors, scoring each query both in the list and off it (the worst
case: its strongest wrong candidate is in the list), mixed by `--off-list-share` (default 10%).
Change `DECK_PRIOR` only when the suggested prior keeps auto-record precision within 0.5 pt and
off-list silent mistakes within 1 pt of no prior.

## Train

```sh
uv run python -m cardid.train --resume models/recogniser.pt --epochs 4 --run my-change
uv run python -m cardid.train --resume models/recogniser.pt --real --epochs 4 --run my-real
uv run python -m cardid.train_detector --resume models/detector.pt --epochs 4 --scene-profile phone --run my-det
uv run python -m cardid.export --checkpoint data/runs/my-change/best.pt --detector models/detector.pt --version my-change
```

Fine-tune from `models/` rather than ImageNet. InfoNCE alone beats adding the ArcFace head, which
memorises the train split. `--real` mixes train-split captures into every epoch (repeated
`--real-repeat` times, default 20, with light jitter) and picks `best.pt` by held-out real
top-1. Token arts (`token`, `double_faced_token`) are ~3% of the gallery, so each gets
`--token-repeat` pairs per epoch (default 3, ~9% of the synthetic pairs, ~6% more steps; 1 turns
it off). Copies keep their art index and get their own degradation. Both trainers take `--seed` and write `data/runs/<run>/run.json` with arguments, device and
library versions. Each run writes `last.pt`, and `best.pt` when an epoch beats the start.

The detector trainer reports median corner error as a fraction of the short side and the share
under 5% ("hit") on fixed synthetic validation (`synth`, `synth_pose` before heatmap snapping, and
per-layout `two_part`/`room`/`split`/`aftermath`/`flip`/`ordinary`), on held-out real outlines
(`real`, `real_e2e`), and `up`/`up_big` orientation accuracy. Its first line is the baseline.
Small real sets step in coarse increments; trust them only once `data/real` grows.

```sh
uv run python -m cardid.synth --n 16 --out /tmp/scenes.png                      # eyeball rendered scenes
uv run python -m cardid.inspect_detector --checkpoint data/runs/my-det/best.pt --out /tmp/worst.png   # worst cases
```

## Real captures

`data/real/` holds labelled captures from three sources, split 80/20 train/eval by a hash of the
capture ID (relabelling never moves one). Keep them out of Git; they are private camera frames.

- **The Gathering** (`webcam-table`): explicit picker choices on the table, from users who allow
  sharing. On its server set `CARDID_CORRECTIONS_TOKEN` (`openssl rand -hex 32`) and
  `CARDID_CORRECTIONS_ADMIN_ID` (an enabled admin's user ID).
- **ManaVault** (`manavault-scanner`): with **Collect training data** on, every logged scan;
  changing the printing or choosing **Wrong card?** relabels it, deleting the scan skips it. The
  token is the server's `SCANNER_CORRECTIONS_TOKEN`.
- **`cardid.capture`** (`capture`): a local webcam page.

```sh
uv run python -m cardid.corrections pull --server https://games.example.com
uv run python -m cardid.corrections pull --server https://manavault.example.com/api/scanner/corrections
uv run python -m cardid.corrections pull --from-dir data/corrections-inbox     # a rsynced export
```

Pull reads `CARDID_CORRECTIONS_TOKEN`, requires HTTPS, keeps a cursor in
`data/real/.corrections-cursor.json`, deduplicates on retry, and warps each crop to the 250×350
`card.png` training uses. Delete the cursor to rescan after restoring an older server backup.

`cardid.capture` opens `http://localhost:8765`: click a card, confirm with `1`–`5`, `/` to search
by name (add a set code and collector number to narrow, e.g. `forest fin 280`), `S` skip, `F` flip
orientation, shift-drag a box when the outline is wrong.

```sh
uv run python -m cardid.capture --checkpoint models/recogniser.pt --detector models/detector.pt
```

It binds to loopback and only accepts same-origin JSON from its own page; `--host` beyond loopback
needs `--allow-remote`, and browsers grant cameras only on localhost or HTTPS.

## GPU

`train`, `train_detector` and `evaluate` take `--device auto|cpu|cuda|mps`; `auto` prefers CUDA
(ROCm appears as `cuda`), then Apple's `mps`, then CPU.

- **AMD (RX 9000, gfx1201, Linux):** `uv sync --extra rocm` installs AMD's PyTorch wheels with the
  ROCm runtime bundled; only the `amdgpu` kernel driver is needed. Check with
  `uv run python -c "import torch; print(torch.cuda.is_available())"`. `cardid` sets
  `TORCH_BLAS_PREFER_HIPBLASLT=0` because AMD lists GPU resets with hipBLASLt on this series;
  export it as `1` to try the faster path.
- **Apple silicon:** `--extra cpu` includes Metal. Use `PYTORCH_ENABLE_MPS_FALLBACK=1` if an op
  is unsupported.
- **NVIDIA:** there is no `cuda` extra yet.

Checkpoints are saved as CPU tensors, so `capture`, `bench` and `export` run on CPU-only installs.

On GPU, augmentation is the bottleneck. `uv run python -m cardid.bench_loader --workers 15 8`
(add `--detector` for scene rendering) times the loader, model step and eval separately. On SMT
desktops fewer workers than logical CPUs often win, so sweep `--workers` and pass the winner to
the trainer; if worker count barely matters, compare `--no-pin`. `python -m cardid.profile_synth`
profiles the scene renderer. Pipeline experiments must be modules, not stdin scripts: Python 3.14
DataLoader workers re-import `__main__` from its file.

## Tests and code layout

`mise run test` runs the CPU unit tests (no GPU, server or weights needed). Before pushing, also
run `uv run --no-sync ruff check cardid` and `uv run --no-sync ruff format --check cardid`. With
the apps checked out beside Oracle, the manifest contract test runs instead of skipping.

Larger modules are split by concern, with the original module kept as the CLI:

- `scryfall` (downloads, images, CLI) over `catalog` (card records → `arts.json`) and `downloads`.
- `synth` over `image_bank`, `scene_geometry`, `scene_renderer` and `scene_datasets`.
- `detector` with `detector_checkpoint` (older head layouts) and `detector_overlay`.
- `constants` holds geometry the manifest ships; `training_runtime` holds device, worker, seed
  and run-metadata setup; `envfile` parses profile files; `workflow` holds retrain's shared steps.
