# AGENTS.md

Oracle trains Magic card recognition models and exports the browser bundles that
The Gathering (webcam table) and ManaVault (card scanner) load. See README.md.

- `cardid/` is the Python package (`python -m cardid.<module>`); `models/` holds the committed
  starting checkpoints; `data/` is ignored and holds galleries, captures, runs and bundles.
- Use the pinned tools through mise: `mise install`, `mise exec -- uv sync --extra cpu`
  (or `--extra rocm` on the AMD training box), then `mise run test`.
- Keep the bundle format compatible with both apps. `cardid/test_manifest_contract.py` checks
  it against sibling checkouts (`../the-gathering`, `../manavault`) or `CARDID_CONSUMERS`.
- Publishing is visible to users: `publish --to github:...` creates a release that ManaVault
  servers install. Never mark bundle releases as latest, and ask before publishing.
- Every commit uses a Conventional Commits message, authored as the current user, with no
  co-author trailers.
