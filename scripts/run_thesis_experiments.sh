#!/usr/bin/env bash
# Run all thesis experiments sequentially on GPU server.
# Usage: bash scripts/run_thesis_experiments.sh [--tier N]
#   --tier 1: Fair 5-model comparison only
#   --tier 2: Ablation experiments only
#   --tier 3: Multi-task experiments only
#   --tier 4: TTA experiments only
#   (no flag): run all tiers
set -euo pipefail

cd "$(dirname "$0")/.."

export LUMBAR_TQDM=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export TORCH_HOME=/root/autodl-tmp/.cache/torch

TIER="${1:-all}"

log_dir="artifacts/logs"
mkdir -p "$log_dir"

run_experiment() {
    local config="$1"
    local name
    name=$(basename "$config" .yaml)
    local logfile="$log_dir/thesis_${name}_$(date +%Y%m%d_%H%M%S).log"
    echo ""
    echo "================================================================"
    echo "[$(date '+%H:%M:%S')] START: $name"
    echo "  Config: $config"
    echo "  Log:    $logfile"
    echo "================================================================"
    if .venv/bin/lumbar-cli train --config "$config" > "$logfile" 2>&1; then
        echo "[$(date '+%H:%M:%S')] DONE:  $name ✓"
    else
        echo "[$(date '+%H:%M:%S')] FAIL:  $name ✗ (see $logfile)"
    fi
}

TIER1=(
    configs/train/thesis/fair_convnext_tiny_cbam.yaml
    configs/train/thesis/fair_densenet121_dense_reuse.yaml
    configs/train/thesis/fair_resnet101_3d.yaml
    configs/train/thesis/fair_swin_transformer.yaml
    configs/train/thesis/fair_vit_base_posenc.yaml
)

TIER2=(
    configs/train/thesis/ablation_no_fusion.yaml
    configs/train/thesis/ablation_no_aug.yaml
    configs/train/thesis/ablation_medium_aug.yaml
    configs/train/thesis/ablation_focal_loss.yaml
    configs/train/thesis/ablation_label_smoothing.yaml
)

TIER3=(
    configs/train/thesis/multitask_convnext_tiny_cbam.yaml
    configs/train/thesis/multitask_densenet121_dense_reuse.yaml
    configs/train/thesis/multitask_swin_transformer.yaml
)

TIER4=(
    configs/train/thesis/tta5_convnext.yaml
)

run_tier() {
    local tier_name="$1"
    shift
    local configs=("$@")
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  $tier_name"
    echo "╚══════════════════════════════════════════════════════════════╝"
    for cfg in "${configs[@]}"; do
        run_experiment "$cfg"
    done
}

case "$TIER" in
    --tier|1|--tier1)  run_tier "Tier 1: Fair 5-Model Comparison" "${TIER1[@]}" ;;
    2|--tier2)         run_tier "Tier 2: Ablation Experiments" "${TIER2[@]}" ;;
    3|--tier3)         run_tier "Tier 3: Multi-Task Learning" "${TIER3[@]}" ;;
    4|--tier4)         run_tier "Tier 4: TTA Experiments" "${TIER4[@]}" ;;
    all|--all|"")
        run_tier "Tier 1: Fair 5-Model Comparison" "${TIER1[@]}"
        run_tier "Tier 2: Ablation Experiments" "${TIER2[@]}"
        run_tier "Tier 3: Multi-Task Learning" "${TIER3[@]}"
        run_tier "Tier 4: TTA Experiments" "${TIER4[@]}"
        ;;
    *)
        echo "Usage: $0 [--tier N | all]"
        exit 1
        ;;
esac

echo ""
echo "================================================================"
echo "All experiments complete. Run evaluation:"
echo '  lumbar-cli compare --runs-root artifacts/runs --primary-metric val_macro_f1'
echo "================================================================"
