#!/bin/bash
#SBATCH --job-name=sgdiff_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --chdir=/share/home/u2515283058/sgdiff
#SBATCH --output=/share/home/u2515283058/sgdiff/sgdiff_train_%j.log
#SBATCH --error=/share/home/u2515283058/sgdiff/sgdiff_train_%j.err

# 默认运行两阶段；也可提交 sbatch slurm_train_sgdiff.sh stage1 或 stage2。
# 双卡提交：sbatch --gres=gpu:2 --cpus-per-task=12 --mem=64G slurm_train_sgdiff.sh
# 团队账号如需计费项目号，在 sbatch 命令中加 --wckey=实际项目号。
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate sgdiff
set -u

PROJECT_ROOT=/share/home/u2515283058/sgdiff
STAGE="${1:-all}"
# 单独训练第二阶段时，可通过 sbatch --export 覆盖为旧第一阶段权重。
STAGE1_CKPT="${STAGE1_CKPT:-$PROJECT_ROOT/work_dirs/sgdiff_bf_glide_v2/iter_235000.pth}"

case "$STAGE" in
    all|stage1|stage2) ;;
    *) echo "阶段只能是 all、stage1 或 stage2。" >&2; exit 1 ;;
esac

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export HTTP_PROXY=http://211.67.63.75:3128
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTP_PROXY"

echo "作业：${SLURM_JOB_ID:-unknown}，节点：$(hostname)，阶段：$STAGE"
python -u -c 'import torch; assert torch.cuda.is_available(), "当前环境无法使用 CUDA"; print("PyTorch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0))'
GPU_COUNT=$(python -c 'import torch; print(torch.cuda.device_count())')
case "$GPU_COUNT" in
    1|2) ;;
    *) echo "此脚本支持单卡或双卡，当前可见 GPU 数：$GPU_COUNT" >&2; exit 1 ;;
esac

train_command=(python -u tools/train.py)
launcher=none
if [[ "$GPU_COUNT" == 2 ]]; then
    # Slurm 启动一个任务，由 torchrun 为每张 GPU 启动一个训练进程。
    train_command=(python -u -m torch.distributed.run --standalone
        --nnodes=1 --nproc_per_node=2 --max_restarts=0 tools/train.py)
    launcher=pytorch
fi

run_stage() {
    local config="$1" work_dir="$2" final_checkpoint="$3" global_batch="$4"
    shift 4
    if [[ -s "$work_dir/$final_checkpoint" ]]; then
        echo "此阶段已完成，跳过：$work_dir/$final_checkpoint"
        return
    fi
    local resume_args=()
    if [[ -f "$work_dir/last_checkpoint" ]]; then
        resume_args=(--resume)
        echo "从已有 checkpoint 续训：$work_dir"
    fi
    local per_gpu_batch=$((global_batch / GPU_COUNT))
    echo "GPU 数：$GPU_COUNT，每卡 batch：$per_gpu_batch，总 batch：$global_batch"
    srun --ntasks=1 --kill-on-bad-exit=1 "${train_command[@]}" \
        "$config" --amp --launcher "$launcher" "${resume_args[@]}" \
        --cfg-options env_cfg.dist_cfg.backend=nccl \
        "train_dataloader.batch_size=$per_gpu_batch" "$@"
}

if [[ "$STAGE" == all || "$STAGE" == stage1 ]]; then
    run_stage configs/sgdiff/sgdiff-bf-glide-64x64.py \
        work_dirs/sgdiff_bf_glide_v2 iter_235000.pth 8
fi

if [[ "$STAGE" == all || "$STAGE" == stage2 ]]; then
    if [[ ! -s "$STAGE1_CKPT" ]]; then
        echo "找不到第一阶段权重：$STAGE1_CKPT" >&2
        exit 1
    fi
    if [[ "$GPU_COUNT" == 2 ]]; then
        # 先由单进程准备 CLIP 缓存，避免两个 rank 同时写入同一权重文件。
        python -u - <<'PY'
import os
from mmagic.models.editors.sgdiff.clip_modules import ClipAttnEmbedding, _download

_download(ClipAttnEmbedding.MODELS['ViT-B/32'], os.path.expanduser('~/.cache/clip'))
PY
    fi
    run_stage configs/sgdiff/sgdiff-bf-style-64x64.py \
        work_dirs/sgdiff_bf_style_v2 iter_50000.pth 16 \
        "model.unet.pretrained_cfg.ckpt_path=$STAGE1_CKPT"
fi

echo "训练完成。固定样本保存在对应 work_dirs/*_v2/samples/。"
