#!/usr/bin/env bash
# Download best checkpoint(s) from the training server.
# Usage:
#   bash scripts/download_best_model.sh <run-id>
#   bash scripts/download_best_model.sh   # downloads the overall best run
#
# Requires: sshpass, SSH_PASSWORD env var (or interactive prompt)
# Server config is read from configs/deploy/default.yaml defaults.

set -euo pipefail

SERVER_HOST="${SERVER_HOST:-connect.westb.seetacloud.com}"
SERVER_PORT="${SERVER_PORT:-22194}"
SERVER_USER="${SERVER_USER:-root}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/root/autodl-tmp/dl-lumbar-degenerative-diseases}"
LOCAL_RUNS="./artifacts/runs"

RUN_ID="${1:-}"

if [ -z "$RUN_ID" ]; then
    echo "No run-id specified. Finding best run by val_macro_f1..."
    BEST=$(find "$LOCAL_RUNS" -name run_summary.json -exec \
        python3 -c "
import json, sys, os
best_f1, best_run = -1, ''
for path in sys.argv[1:]:
    try:
        data = json.load(open(path))
        f1 = data.get('best_val_macro_f1', data.get('val_macro_f1', -1))
        if f1 > best_f1:
            best_f1 = f1
            best_run = os.path.basename(os.path.dirname(path))
    except: pass
print(best_run)
" {} +)
    if [ -z "$BEST" ]; then
        echo "Error: no run_summary.json found in $LOCAL_RUNS"
        exit 1
    fi
    RUN_ID="$BEST"
    echo "Best run: $RUN_ID"
fi

DEST="$LOCAL_RUNS/$RUN_ID/best.ckpt"
if [ -f "$DEST" ]; then
    echo "Checkpoint already exists: $DEST"
    exit 0
fi

echo "Downloading $RUN_ID/best.ckpt from server..."

SSH_CMD="ssh -p $SERVER_PORT -o StrictHostKeyChecking=no"
if [ -n "${SSH_PASSWORD:-}" ]; then
    SSH_CMD="sshpass -p '$SSH_PASSWORD' $SSH_CMD"
fi

mkdir -p "$LOCAL_RUNS/$RUN_ID"
rsync -avz --progress -e "$SSH_CMD" \
    "$SERVER_USER@$SERVER_HOST:$REMOTE_PROJECT/artifacts/runs/$RUN_ID/best.ckpt" \
    "$DEST"

echo "Downloaded: $DEST ($(du -h "$DEST" | cut -f1))"
