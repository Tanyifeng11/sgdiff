#!/bin/bash
#SBATCH --job-name=sgdiff_bf_full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --chdir=/share/home/u2515283058/sgdiff
#SBATCH --output=/share/home/u2515283058/sgdiff/sgdiff_bf_full_%j.log
#SBATCH --error=/share/home/u2515283058/sgdiff/sgdiff_bf_full_%j.err

# 固定第二阶段权重，仅测试。中断后通过 OUTPUT_DIR 指定原目录即可续跑。
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate sgdiff
set -u

PROJECT_ROOT=/share/home/u2515283058/sgdiff
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export HTTP_PROXY="${HTTP_PROXY:-http://211.67.63.75:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-$HTTP_PROXY}"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"

CHECKPOINT="$PROJECT_ROOT/work_dirs/sgdiff_bf_style_v2/iter_50000.pth"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/bf_test_full/${SLURM_JOB_ID:-manual}/stage2}"
CLIP_MODEL="${TEST_CLIP_MODEL:-/share/home/u2515283058/Mymodel/models/clip/models/image_encoder}"
test -s "$CHECKPOINT" || { echo "找不到权重：$CHECKPOINT" >&2; exit 1; }

args=(configs/sgdiff/sgdiff-bf-style-64x64.py "$CHECKPOINT"
    --data-root /share/home/u2515283058/datasets/BF
    --split test --max-samples 0 --all-test-categories
    --seed 42 --output-dir "$OUTPUT_DIR"
    --num-inference-steps 100 --up-inference-steps 35
    --text-guidance 1.0 --style-guidance 1.2
    --metrics fid clip_i ssim --clip-model "$CLIP_MODEL")

GPU_COUNT=$(python -c 'import torch; assert torch.cuda.is_available(), "CUDA 不可用"; print(torch.cuda.device_count())')
echo "固定权重：$CHECKPOINT；GPU 数：$GPU_COUNT；输出：$OUTPUT_DIR"
srun --ntasks=1 --kill-on-bad-exit=1 python -u tools/test_sgdiff.py "${args[@]}" --mode prepare
extended_args=(--output-dir "$OUTPUT_DIR" --clip-model "$CLIP_MODEL"
    --mymodel-root "${MYMODEL_ROOT:-/share/home/u2515283058/Mymodel}")
srun --ntasks=1 --kill-on-bad-exit=1 python -u tools/evaluate_bf_extended.py "${extended_args[@]}" --check-only

# 单进程预先缓存风格编码器，避免多个 GPU 进程同时下载。
python -u - <<'PY'
import os
from mmagic.models.editors.sgdiff.clip_modules import ClipAttnEmbedding, _download
_download(ClipAttnEmbedding.MODELS['ViT-B/32'], os.path.expanduser('~/.cache/clip'))
PY

srun --ntasks=1 --kill-on-bad-exit=1 python -u -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="$GPU_COUNT" --max_restarts=0 \
    tools/test_sgdiff.py "${args[@]}" --mode generate

# 所有分片完成后汇总整个测试集，不能取各 GPU 的 FID 平均值。
srun --ntasks=1 --kill-on-bad-exit=1 python -u tools/test_sgdiff.py "${args[@]}" --mode evaluate
srun --ntasks=1 --kill-on-bad-exit=1 python -u tools/evaluate_bf_extended.py "${extended_args[@]}"
echo "扩展指标：$OUTPUT_DIR/metrics_extended.json"
echo "测试完成：$OUTPUT_DIR/metrics.json"
