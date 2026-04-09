#!/usr/bin/env bash
set -euo pipefail

# Run environment setup and training on a remote Linux server.
# Usage:
#   bash scripts/server_train.sh user@host /remote/project/path [configs/train/default.yaml]

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 user@host /remote/project/path [train-config]" >&2
  exit 1
fi

TARGET_HOST="$1"
REMOTE_PATH="$2"
TRAIN_CONFIG="${3:-configs/train/default.yaml}"

ssh "$TARGET_HOST" \
  REMOTE_PATH="$REMOTE_PATH" \
  TRAIN_CONFIG="$TRAIN_CONFIG" \
  'bash -lc '
"'"'set -euo pipefail
cd "$REMOTE_PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --extra dev
uv run python scripts/healthcheck.py linux || true
uv run lumbar-cli train --config "$TRAIN_CONFIG"
'"'"''
