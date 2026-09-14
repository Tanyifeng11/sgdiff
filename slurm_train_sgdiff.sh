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

# 默认运行两阶段，每阶段训练后自动测试；stage1/stage2 仅运行相应阶段及测试。
# 仅测试：sbatch slurm_train_sgdiff.sh test（第二阶段）或 test1（第一阶段）。
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
DATA_ROOT=/share/home/u2515283058/datasets/BF
TEST_SPLIT="${TEST_SPLIT:-validation}"
TEST_SAMPLES="${TEST_SAMPLES:-100}"
TEST_SEED="${TEST_SEED:-42}"
# 可指定 Mymodel 已生成的固定 split JSON，使两个工程测试同一批样本。
TEST_SPLIT_FILE="${TEST_SPLIT_FILE:-}"
TEST_CLIP_MODEL="${TEST_CLIP_MODEL:-openai/clip-vit-large-patch14}"
TEST_RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
TEST_OUTPUT_ROOT="${TEST_OUTPUT_ROOT:-$PROJECT_ROOT/results/bf_${TEST_SPLIT}/$TEST_RUN_ID}"

case "$STAGE" in
    all) stages=(1 2) ;;
    stage1|test1) stages=(1) ;;
    stage2|test) stages=(2) ;;
    *) echo "模式只能是 all、stage1、stage2、test 或 test1。" >&2; exit 1 ;;
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
test_command=(python -u tools/test_sgdiff.py)
launcher=none
if [[ "$GPU_COUNT" == 2 ]]; then
    # Slurm 启动一个任务，由 torchrun 为每张 GPU 启动一个训练进程。
    train_command=(python -u -m torch.distributed.run --standalone
        --nnodes=1 --nproc_per_node=2 --max_restarts=0 tools/train.py)
    test_command=(python -u -m torch.distributed.run --standalone
        --nnodes=1 --nproc_per_node=2 --max_restarts=0 tools/test_sgdiff.py)
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

for stage_number in "${stages[@]}"; do
    extra_train_args=()
    if [[ "$stage_number" == 1 ]]; then
        config=configs/sgdiff/sgdiff-bf-glide-64x64.py
        work_dir=work_dirs/sgdiff_bf_glide_v2
        final_checkpoint=iter_235000.pth
        global_batch=8
    else
        config=configs/sgdiff/sgdiff-bf-style-64x64.py
        work_dir=work_dirs/sgdiff_bf_style_v2
        final_checkpoint=iter_50000.pth
        global_batch=16
        if [[ "$STAGE" != test && ! -s "$STAGE1_CKPT" ]]; then
            echo "找不到第一阶段权重：$STAGE1_CKPT" >&2
            exit 1
        fi
        extra_train_args=("model.unet.pretrained_cfg.ckpt_path=$STAGE1_CKPT")
    fi
    test_checkpoint="$PROJECT_ROOT/$work_dir/$final_checkpoint"
    if [[ "$STAGE" == test1 ]]; then
        test_checkpoint="$STAGE1_CKPT"
    fi
    test_args=("$config" "$test_checkpoint"
        --data-root "$DATA_ROOT" --split "$TEST_SPLIT"
        --max-samples "$TEST_SAMPLES" --seed "$TEST_SEED" --split-seed 42
        --output-dir "$TEST_OUTPUT_ROOT/stage$stage_number"
        --num-inference-steps 100 --up-inference-steps 35
        --clip-model "$TEST_CLIP_MODEL")
    if [[ -n "$TEST_SPLIT_FILE" ]]; then
        test_args+=(--split-file "$TEST_SPLIT_FILE")
    fi
    # 先检查测试数据、指标依赖和缓存，再开始长时间训练。
    srun --ntasks=1 --kill-on-bad-exit=1 python -u tools/test_sgdiff.py \
        "${test_args[@]}" --mode prepare
    if [[ "$stage_number" == 2 ]]; then
        # 训练和测试前均由单进程准备 CLIP，避免两个 rank 同时写缓存。
        python -u - <<'PY'
import os
from mmagic.models.editors.sgdiff.clip_modules import ClipAttnEmbedding, _download

_download(ClipAttnEmbedding.MODELS['ViT-B/32'], os.path.expanduser('~/.cache/clip'))
PY
    fi
    if [[ "$STAGE" != test && "$STAGE" != test1 ]]; then
        run_stage "$config" "$work_dir" "$final_checkpoint" "$global_batch" \
            "${extra_train_args[@]}"
    fi
    # 每卡生成不同样本；等待所有分片完成后，单进程计算整个测试集的指标。
    srun --ntasks=1 --kill-on-bad-exit=1 "${test_command[@]}" \
        "${test_args[@]}" --mode generate
    srun --ntasks=1 --kill-on-bad-exit=1 python -u tools/test_sgdiff.py \
        "${test_args[@]}" --mode evaluate
done

echo "训练与测试完成，测试图片和指标：$TEST_OUTPUT_ROOT"
