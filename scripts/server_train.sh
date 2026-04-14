#!/usr/bin/env bash
set -euo pipefail

# Run environment setup and training on a remote Linux server.
# Usage:
#   SSH_PASSWORD=xxx SSH_PORT=48255 bash scripts/server_train.sh root@host /root/autodl-tmp/project [configs/train/default.yaml]

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 user@host /remote/project/path [train-config]" >&2
  exit 1
fi

TARGET_HOST="$1"
REMOTE_PATH="$2"
TRAIN_CONFIG="${3:-configs/train/default.yaml}"
SSH_PASSWORD="${SSH_PASSWORD:-}"
SSH_PORT="${SSH_PORT:-22}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
UV_BIN="${UV_BIN:-/root/miniconda3/bin/uv}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
CACHE_ROOT="${CACHE_ROOT:-/root/autodl-tmp/.cache}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

SSH_CMD=(ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no)
if [[ -n "$SSH_PASSWORD" ]]; then
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "sshpass is required when SSH_PASSWORD is set." >&2
    exit 1
  fi
  SSH_CMD=(sshpass -p "$SSH_PASSWORD" "${SSH_CMD[@]}")
fi

"${SSH_CMD[@]}" \
  "$TARGET_HOST" \
  env \
  REMOTE_PATH="$REMOTE_PATH" \
  TRAIN_CONFIG="$TRAIN_CONFIG" \
  PYTHON_BIN="$PYTHON_BIN" \
  UV_BIN="$UV_BIN" \
  PIP_INDEX_URL="$PIP_INDEX_URL" \
  PIP_TRUSTED_HOST="$PIP_TRUSTED_HOST" \
  HF_ENDPOINT="$HF_ENDPOINT" \
  CACHE_ROOT="$CACHE_ROOT" \
  FORCE_TRAIN="$FORCE_TRAIN" \
  'bash -s' <<'REMOTE_SCRIPT'
set -euo pipefail

export UV_INDEX_URL="$PIP_INDEX_URL"
export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"
export PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export TIMM_HOME="$CACHE_ROOT/timm"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

mkdir -p "$HF_HOME" "$TORCH_HOME" "$TIMM_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$REMOTE_PATH"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -x "$UV_BIN" ]]; then
  "$PYTHON_BIN" -m pip install --upgrade uv -i "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST"
fi

if [[ -f ".train.pid" ]] && [[ "$FORCE_TRAIN" != "1" ]]; then
  existing_pid="$(cat .train.pid)"
  if kill -0 "$existing_pid" >/dev/null 2>&1; then
    echo "Training is already running with PID $existing_pid. Set FORCE_TRAIN=1 to override." >&2
    exit 1
  fi
fi

"$UV_BIN" sync --extra dev --python "$PYTHON_BIN"
"$UV_BIN" run python scripts/healthcheck.py linux || true
"$UV_BIN" run python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_version={torch.version.cuda}")
if torch.cuda.is_available():
    print(f"device_name={torch.cuda.get_device_name(0)}")
PY

mkdir -p artifacts/logs
timestamp="$(date +%Y%m%d_%H%M%S)"
train_log="artifacts/logs/train_${timestamp}.log"
nohup env \
  PIP_INDEX_URL="$PIP_INDEX_URL" \
  UV_INDEX_URL="$UV_INDEX_URL" \
  PIP_TRUSTED_HOST="$PIP_TRUSTED_HOST" \
  HF_ENDPOINT="$HF_ENDPOINT" \
  HF_HOME="$HF_HOME" \
  TORCH_HOME="$TORCH_HOME" \
  TIMM_HOME="$TIMM_HOME" \
  MPLCONFIGDIR="$MPLCONFIGDIR" \
  XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  "$UV_BIN" run lumbar-cli train --config "$TRAIN_CONFIG" \
  >"$train_log" 2>&1 </dev/null &
train_pid="$!"

echo "$train_pid" > .train.pid
echo "TRAIN_PID=$train_pid"
echo "TRAIN_LOG=$REMOTE_PATH/$train_log"
sleep 5
tail -n 40 "$train_log" || true
REMOTE_SCRIPT
