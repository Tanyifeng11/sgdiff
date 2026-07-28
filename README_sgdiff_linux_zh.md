# SGDiff Linux（服务器）环境与训练

本文对应本仓库当前的 BF 训练代码和配置，适用于 NVIDIA GPU 服务器（A30 单卡即可开始训练）。数据集路径已配置为：

```text
/share/home/u2515283058/datasets/BF
```

## 1. 前置检查

登录服务器后执行：

```bash
nvidia-smi
```

确认能看到 A30。驱动版本只要支持 CUDA 11.3 或更高即可；下面安装的 PyTorch 自带 CUDA 11.3 运行时，不需要单独安装系统 CUDA Toolkit。

## 2. 创建 Conda 环境

以下命令假定服务器已安装 Miniconda/Anaconda。没有 Conda 时，先安装 Miniconda，再继续。

```bash
conda create -n sgdiff python=3.9 pip -y
conda activate sgdiff
python -m pip install --upgrade "pip<25"
```

项目使用的 MMagic 版本较旧，Python 3.12、PyTorch 2.x、NumPy 2.x 均不要使用。

## 3. 安装 PyTorch 和 OpenMMLab 依赖

```bash
python -m pip install \
  torch==1.10.0+cu113 \
  torchvision==0.11.0+cu113 \
  -f https://download.pytorch.org/whl/torch_stable.html

python -m pip install numpy==1.26.4 opencv-python==4.8.1.78
python -m pip install mmengine==0.10.7
python -m pip install \
  mmcv==2.1.0 \
  --no-deps \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

验证 CUDA：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

最后一项应为 `True`，并显示 A30。

## 4. 安装 SGDiff 运行依赖与项目

进入已上传或克隆到服务器的项目根目录（即含有 `setup.py` 和 `configs/` 的目录）：

```bash
cd /你的/SGDiff/项目目录

python -m pip install click requests regex tqdm
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

无需执行 `pip install .`：直接在仓库根目录运行训练脚本即可。不要安装 `av`；它与该版本组合容易引入无关的导入冲突。

最小导入检查：

```bash
python -c "import torch, mmcv, mmengine, mmagic; print(torch.__version__, mmcv.__version__, mmengine.__version__)"
```

## 5. 检查 BF 数据

```bash
find /share/home/u2515283058/datasets/BF/training -maxdepth 1 -type d | sort
```

训练集至少需要以下目录，且同名样本应成对存在：

```text
training/cloth/*.jpg
training/text/*.txt
training/texture/*.jpg
```

配置文件中的路径位于 [configs/sgdiff/sgdiff-bf-glide-64x64.py:4](configs/sgdiff/sgdiff-bf-glide-64x64.py)，第二阶段会继承它。

## 6. 单卡 A30 训练

建议先用单卡跑通。`--amp` 会启用混合精度，显著降低显存占用。

第一阶段（文本到 64×64 服装图）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/train.py \
  configs/sgdiff/sgdiff-bf-glide-64x64.py --amp
```

完成后确认生成：

```text
work_dirs/sgdiff_bf_glide/iter_235000.pth
```

第二阶段（纹理条件超分到 256×256）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/train.py \
  configs/sgdiff/sgdiff-bf-style-64x64.py --amp
```

第二阶段默认读取第一阶段的 `iter_235000.pth`。若 checkpoint 位于其他位置，用 `--cfg-options` 覆盖：

```bash
PYTHONPATH=. python tools/train.py configs/sgdiff/sgdiff-bf-style-64x64.py --amp \
  --cfg-options model.unet.pretrained_cfg.ckpt_path=/实际路径/iter_235000.pth
```

## 7. 四卡 A30（可选）

为保持与单卡配置相同的全局 batch size，四卡时需按下面命令降低每卡 batch size：

```bash
bash tools/dist_train.sh configs/sgdiff/sgdiff-bf-glide-64x64.py 4 --amp \
  --cfg-options train_dataloader.batch_size=2

bash tools/dist_train.sh configs/sgdiff/sgdiff-bf-style-64x64.py 4 --amp \
  --cfg-options train_dataloader.batch_size=4
```

## 8. 常见问题

- `torch.cuda.is_available()` 为 `False`：先确认 `nvidia-smi` 正常，再重新安装第 3 节的 CUDA 版 PyTorch。
- `ModuleNotFoundError: mmcv._ext`：MMCV 与 PyTorch/CUDA 不匹配，卸载 `mmcv` 后按第 3 节指定链接重装 `mmcv==2.1.0`。
- 显存不足：保留 `--amp`，并把 `train_dataloader.batch_size` 调小；单卡 batch size 变小后，学习率也应按比例降低。
- 预训练权重下载失败：服务器无外网时，先在有网机器下载配置中指定的 GLIDE 权重，再将 `ckpt_path` 修改为服务器本地文件路径。
