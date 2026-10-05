# Oracle

Card recognition for Magic cards. Oracle trains an art embedder (MobileNetV3-Small, 128-d) and a
card detector (CornerNet), then exports them as a versioned browser **bundle**: three ONNX graphs
plus the gallery index. Two apps load the bundle with onnxruntime-web:

| App | Uses it for | Bundles published to | Picked up |
|---|---|---|---|
| [The Gathering](https://github.com/cfbender/the-gathering) | click-to-identify on the webcam table | its server over ssh (`user@host:/srv/the-gathering/cardid`) | browsers on the next identification |
| [ManaVault](https://github.com/cfbender/manavault) | the `/scan` card scanner | a `scanner-bundle-<version>` GitHub release, never marked latest | servers check every 6 h; browsers on the next scanner open |

Each app has its own model, because a webcam over a table and a phone over one card are
different jobs. The Python package is `cardid` (`python -m cardid.<module>`).

To improve the models or try a bundle in either app, start with [CONTRIBUTING.md](CONTRIBUTING.md).

## Everyday commands

```sh
mise run retrain-all                 # new scans: retrain and publish both apps (-- --dry-run to preview)
mise run new-set                     # a set released: refresh the gallery and publish both apps, no training
mise run promote-models              # make The Gathering's published weights the committed models/
mise run test                        # CPU unit tests
```

`retrain-all` refreshes the gallery once. Then, for each app, it pulls that app's new captures,
fine-tunes from the checkpoints behind its published bundle, exports, and publishes unless
top-1 on its held-out real captures regresses. An app that fails does not stop the other.
Flags after `--` go to every run, for example `--no-publish` or `--profile manavault`.

One app at a time: `mise run gathering` or `mise run manavault`.

## Training box setup

```sh
git clone git@github.com:cfbender/oracle.git && cd oracle
mise install && mise exec -- uv sync --extra rocm    # --extra cpu without an AMD GPU
export UV_NO_SYNC=1                                  # keep the installed torch extra
gh auth login                                        # needed to publish ManaVault releases
mise run setup-profiles                              # creates ~/.config/cardid/{manavault,the-gathering}.env
${EDITOR:-nano} ~/.config/cardid/*.env               # fill in each server URL and token
```

Each profile names its app's corrections server, read-only token, publish target and capture
sources. The files are read as literal `KEY=value` data, never sourced. Every `*.env` in
`~/.config/cardid/` is an app to `retrain-all` and `new-set`. `CARDID_*` variables exported in
your shell override the files.

`data/` (gallery, images, captures, runs, bundles) is ignored by Git. Build a gallery once with
`uv run python -m cardid.scryfall --all` (~52k art crops, ~3 GB), or reuse an existing one.

## Starting weights

`models/recogniser.pt` and `models/detector.pt` are the checkpoints behind a published bundle, so
a fresh clone can fine-tune, evaluate or export without training from scratch.
`sha256sum -c models/SHA256SUMS` checks them. After a good retrain, `mise run promote-models`
(or `-- --profile manavault`) copies that app's published pair into `models/`, rewrites
`SHA256SUMS` and prints a commit message naming the bundle.

## Rolling back

- **ManaVault:** unpublish the newest release
  (`gh release edit <tag> --repo cfbender/manavault --draft=true`). Servers reinstall the
  previous one on their next check.
- **The Gathering:** the server keeps `previous` next to `current`; repoint the `current` symlink.

## More documentation

- [docs/operations.md](docs/operations.md): profiles, what a retrain checks before publishing,
  new sets, publishing, the optional nightly loop.
- [docs/training.md](docs/training.md): manual training, evaluation, real captures, the detector,
  GPU setup, code layout.
- [docs/pipeline.md](docs/pipeline.md): how recognition works, the gallery, frame, token and
  layout geometry, hub arts, the bundle format and manifest, and ManaVault's optional search
  mask (`search.onnx` input `mask`, manifest `search_mask`).
