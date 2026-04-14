#!/usr/bin/env bash
set -euo pipefail

# Synchronize the project to a Linux training server using rsync.
# Usage:
#   SSH_PASSWORD=xxx SSH_PORT=48255 RSYNC_PARALLEL=6 bash scripts/sync_to_server.sh root@host /root/autodl-tmp/project --with-data

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 user@host /remote/project/path [--with-data]" >&2
  exit 1
fi

TARGET_HOST="$1"
REMOTE_PATH="$2"
WITH_DATA="${3:-}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR_NAME="rsna-2024-lumbar-spine-degenerative-classification"
LOCAL_DATASET_ROOT="$PROJECT_ROOT/$DATASET_DIR_NAME"
REMOTE_DATASET_ROOT="$REMOTE_PATH/$DATASET_DIR_NAME"
SSH_PASSWORD="${SSH_PASSWORD:-}"
SSH_PORT="${SSH_PORT:-22}"
RSYNC_PARALLEL="${RSYNC_PARALLEL:-4}"
RSYNC_CHUNK_SIZE="${RSYNC_CHUNK_SIZE:-24}"
RSYNC_BIN="${RSYNC_BIN:-$(command -v rsync)}"
RSYNC_SSH="ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no"

if [[ -z "$RSYNC_BIN" ]]; then
  echo "rsync is required but was not found on PATH." >&2
  exit 1
fi

SSH_CMD=(ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no)
RSYNC_CMD=(
  "$RSYNC_BIN"
  -a
  -z
  --human-readable
  --info=progress2
  --partial
  --append-verify
  --delete-delay
  -e
  "$RSYNC_SSH"
)
PROJECT_RSYNC_CMD=(
  "$RSYNC_BIN"
  -a
  -z
  --human-readable
  --info=progress2
  --partial
  --delete-delay
  -e
  "$RSYNC_SSH"
)

if [[ -n "$SSH_PASSWORD" ]]; then
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "sshpass is required when SSH_PASSWORD is set." >&2
    exit 1
  fi
  export SSHPASS="$SSH_PASSWORD"
  SSH_CMD=(sshpass -e ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no)
  RSYNC_SSH="sshpass -e ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no"
  RSYNC_CMD=(
    "$RSYNC_BIN"
    -a
    -z
    --human-readable
    --info=progress2
    --partial
    --append-verify
    --delete-delay
    -e
    "$RSYNC_SSH"
  )
  PROJECT_RSYNC_CMD=(
    "$RSYNC_BIN"
    -a
    -z
    --human-readable
    --info=progress2
    --partial
    --delete-delay
    -e
    "$RSYNC_SSH"
  )
fi

run_ssh() {
  "${SSH_CMD[@]}" "$TARGET_HOST" "$@"
}

run_rsync() {
  "${RSYNC_CMD[@]}" "$@"
}

run_project_rsync() {
  "${PROJECT_RSYNC_CMD[@]}" "$@"
}

sync_dataset_split() {
  local split_name="$1"
  local local_split_root="$LOCAL_DATASET_ROOT/$split_name"
  local remote_split_root="$REMOTE_DATASET_ROOT/$split_name"
  local entry_count

  if [[ ! -d "$local_split_root" ]]; then
    return 0
  fi

  run_ssh "mkdir -p '$remote_split_root'"
  entry_count="$(find "$local_split_root" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
  if [[ "$entry_count" == "0" ]]; then
    run_rsync "$local_split_root/" "$TARGET_HOST:$remote_split_root/"
    return 0
  fi

  export SSH_PASSWORD SSHPASS SSH_PORT TARGET_HOST RSYNC_BIN
  export REMOTE_SPLIT_ROOT="$remote_split_root"
  find "$local_split_root" -mindepth 1 -maxdepth 1 -type d -print0 |
    xargs -0 -n "$RSYNC_CHUNK_SIZE" -P "$RSYNC_PARALLEL" bash -c '
      set -euo pipefail
      ssh_transport="ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no"
      if [[ -n "${SSH_PASSWORD:-}" ]]; then
        export SSHPASS="${SSHPASS:-$SSH_PASSWORD}"
        ssh_transport="sshpass -e ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no"
      fi
      rsync_cmd=(
        "$RSYNC_BIN"
        -a
        -z
        --human-readable
        --partial
        --append-verify
        --info=progress2
        -e
        "$ssh_transport"
      )
      destination="$1"
      host="$2"
      shift 2
      "${rsync_cmd[@]}" "$@" "${host}:${destination}/"
    ' _ "$REMOTE_SPLIT_ROOT" "$TARGET_HOST"
}

PROJECT_EXCLUDES=(
  --exclude ".git/"
  --exclude ".venv/"
  --exclude "__pycache__/"
  --exclude ".pytest_cache/"
  --exclude ".mypy_cache/"
  --exclude ".ruff_cache/"
  --exclude "artifacts/"
  --exclude "reports/figures/"
  --exclude "$DATASET_DIR_NAME/"
)

run_ssh "mkdir -p '$REMOTE_PATH'"
# Project files are relatively small; checksum avoids false negatives when
# content changes keep the same size and land within the same timestamp second.
run_project_rsync --checksum "${PROJECT_EXCLUDES[@]}" "$PROJECT_ROOT/" "$TARGET_HOST:$REMOTE_PATH/"

if [[ "$WITH_DATA" == "--with-data" ]]; then
  if [[ ! -d "$LOCAL_DATASET_ROOT" ]]; then
    echo "Dataset directory not found: $LOCAL_DATASET_ROOT" >&2
    exit 1
  fi

  run_ssh "mkdir -p '$REMOTE_DATASET_ROOT'"
  run_rsync \
    --exclude "train_images/" \
    --exclude "test_images/" \
    "$LOCAL_DATASET_ROOT/" \
    "$TARGET_HOST:$REMOTE_DATASET_ROOT/"

  sync_dataset_split "train_images"
  sync_dataset_split "test_images"
  echo "Project and dataset sync completed: $TARGET_HOST:$REMOTE_PATH"
else
  echo "Project sync completed without dataset: $TARGET_HOST:$REMOTE_PATH"
fi
