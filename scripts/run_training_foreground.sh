#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TRAIN_CONFIG="${1:-configs/train/default.yaml}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/.cache/torch}"
export TIMM_HOME="${TIMM_HOME:-$PROJECT_ROOT/.cache/timm}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_ROOT/.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/.cache/xdg}"
export LUMBAR_TQDM="${LUMBAR_TQDM:-1}"

mkdir -p "$HF_HOME" "$TORCH_HOME" "$TIMM_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$PROJECT_ROOT"

if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  echo "Missing executable: $PROJECT_ROOT/.venv/bin/python" >&2
  echo "Please create the virtual environment first." >&2
  exit 1
fi

read_config_field() {
  "$PROJECT_ROOT/.venv/bin/python" -c "import sys, yaml; cfg = yaml.safe_load(open(sys.argv[1], encoding='utf-8')); value = cfg.get(sys.argv[2], ''); print(value if value is not None else '')" "$TRAIN_CONFIG" "$1"
}

prefetch_convnext_tiny_weights() {
  local model_name
  local pretrained
  local weights_url
  local weights_path
  local partial_path

  model_name="$(read_config_field model_name)"
  pretrained="$(read_config_field pretrained)"
  if [[ "$model_name" != "convnext_tiny_cbam" || "$pretrained" != "True" ]]; then
    return 0
  fi

  weights_url="${LUMBAR_CONVNEXT_TINY_WEIGHTS_URL:-https://download.pytorch.org/models/convnext_tiny-983f1562.pth}"
  weights_path="${LUMBAR_CONVNEXT_TINY_WEIGHTS:-$PROJECT_ROOT/.cache/pretrained/convnext_tiny-983f1562.pth}"
  partial_path="${weights_path}.partial"
  export LUMBAR_CONVNEXT_TINY_WEIGHTS="$weights_path"

  if [[ -f "$weights_path" ]]; then
    return 0
  fi

  mkdir -p "$(dirname "$weights_path")"
  echo "Prefetching ConvNeXt pretrained weights to $weights_path"
  curl \
    --location \
    --fail \
    --retry 20 \
    --retry-delay 3 \
    --retry-all-errors \
    --continue-at - \
    --output "$partial_path" \
    "$weights_url"
  mv "$partial_path" "$weights_path"
}

prefetch_convnext_tiny_weights

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PROJECT_ROOT/.venv/bin/python" -m dl_lumbar_dd.cli train --config "$TRAIN_CONFIG"
