#!/bin/bash

#SBATCH --job-name=Sgdiff_test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=/share/home/u2515283058/sgdiff/test_bf_compare.log
#SBATCH --error=/share/home/u2515283058/sgdiff/test_bf_compare.err

set -euo pipefail

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate sgdiff

PROJECT_ROOT=/share/home/u2515283058/sgdiff
DATA_ROOT=/share/home/u2515283058/datasets/BF
ORIGINAL_CKPT="${PROJECT_ROOT}/checkpoint/sgdiff.pth"
FINETUNED_CKPT="${PROJECT_ROOT}/work_dirs/sgdiff_bf_style/iter_50000.pth"
OUTPUT_DIR="${PROJECT_ROOT}/results/bf_validation_compare"
MAX_SAMPLES="${MAX_SAMPLES:-}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

export HTTP_PROXY=http://211.67.63.75:3128
export HTTPS_PROXY=http://211.67.63.75:3128
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"

test -f "$ORIGINAL_CKPT" || { echo "找不到原版权重：$ORIGINAL_CKPT"; exit 1; }
test -f "$FINETUNED_CKPT" || { echo "找不到训练权重：$FINETUNED_CKPT"; exit 1; }

echo "作业 ID: ${SLURM_JOB_ID:-unknown}"
echo "运行节点: $(hostname)"
echo "原版权重: $ORIGINAL_CKPT"
echo "训练权重: $FINETUNED_CKPT"
nvidia-smi

TEST_ARGS=(
    --data-root "$DATA_ROOT"
    --original-ckpt "$ORIGINAL_CKPT"
    --finetuned-ckpt "$FINETUNED_CKPT"
    --output-dir "$OUTPUT_DIR"
    --num-inference-steps 100
    --up-inference-steps 35
)

if [[ -n "$MAX_SAMPLES" ]]; then
    TEST_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

python -u tools/test_bf_compare.py \
    "${TEST_ARGS[@]}"

echo "测试完成：$OUTPUT_DIR/comparison"
