#!/usr/bin/env bash
set -euo pipefail

# Synchronize the project to a Linux training server using rsync.
# Usage:
#   bash scripts/sync_to_server.sh user@host /remote/project/path [--with-data]

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 user@host /remote/project/path [--with-data]" >&2
  exit 1
fi

TARGET_HOST="$1"
REMOTE_PATH="$2"
WITH_DATA="${3:-}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXCLUDES=(
  --exclude ".git/"
  --exclude ".venv/"
  --exclude "__pycache__/"
  --exclude ".pytest_cache/"
  --exclude ".mypy_cache/"
  --exclude ".ruff_cache/"
  --exclude "artifacts/"
  --exclude "reports/figures/"
)

if [[ "$WITH_DATA" != "--with-data" ]]; then
  EXCLUDES+=(--exclude "rsna-2024-lumbar-spine-degenerative-classification/")
fi

ssh "$TARGET_HOST" "mkdir -p '$REMOTE_PATH'"
rsync -avz --delete "${EXCLUDES[@]}" "$PROJECT_ROOT/" "$TARGET_HOST:$REMOTE_PATH/"

echo "Sync completed: $TARGET_HOST:$REMOTE_PATH"
if [[ "$WITH_DATA" == "--with-data" ]]; then
  echo "Dataset sync was enabled."
else
  echo "Dataset sync was skipped. Use --with-data if the server does not already have the dataset."
fi
