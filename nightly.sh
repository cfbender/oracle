#!/usr/bin/env bash
# Training desktop only. Safe by hand too; never installs/enables a timer.
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
env_file="${CARDID_ENV_FILE:-$HOME/.config/cardid.env}"
if [[ -f "$env_file" ]]; then
  set -a
  source "$env_file"
  set +a
fi
mkdir -p data/nightly
exec 9>data/nightly/run.lock
flock -n 9 || { echo 'Another nightly run is active'; exit 1; }
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
# timeout controls the whole process group, including dataloader workers and export.
timeout --signal=TERM --kill-after=30s "${CARDID_BUDGET:-2h}" \
  nice -n "${CARDID_NICE:-15}" uv run python -m cardid.nightly "$@" \
  2>&1 | tee -a "data/nightly/$(date -u +%Y-%m-%d).log"
