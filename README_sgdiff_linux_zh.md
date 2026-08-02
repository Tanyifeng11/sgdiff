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

## 6. 准备 GLIDE 预训练权重

MMagic 的 GLIDE 镜像（`download.openxlab.org.cn/models/mmediting/GLIDE/weight/glide_laion-64-256`）已下线，因此改用 OpenAI / LAION 发布的原始权重：

```text
checkpoint/laionide-v3-base.pt   # 第一阶段：GLIDE 64x64 基础模型
checkpoint/upsample.pt           # 第二阶段：GLIDE 64->256 超分模型
```

这两个文件沿用 [openai/glide-text2im](https://github.com/openai/glide-text2im) 的参数命名，与 MMagic 的命名不同（例如 `input_blocks` 对应 `in_blocks`、`out_layers.3` 对应 `conv_2.2`）。加载时由 [mmagic/models/editors/glide/glide_ckpt.py](mmagic/models/editors/glide/glide_ckpt.py) 自动改名，**不需要提前做离线转换**，直接填 `.pt` 路径即可。

路径写在这两处，默认按 `/share/home/u2515283058/sgdiff/checkpoint/` 书写：

- [configs/sgdiff/sgdiff-bf-glide-64x64.py](configs/sgdiff/sgdiff-bf-glide-64x64.py) 的 `glide_ckpt`
- [configs/sgdiff/sgdiff-bf-style-64x64.py](configs/sgdiff/sgdiff-bf-style-64x64.py) 的 `glide_up_ckpt`

若服务器上的项目目录不是 `/share/home/u2515283058/sgdiff`，改这两行即可，也可以在命令行覆盖：

```bash
--cfg-options model.pretrained_cfgs.unet_up.ckpt_path=/实际路径/checkpoint/upsample.pt
```

正式训练前建议先校验权重与网络结构是否完全匹配（只占内存，几十秒即可完成）：

```bash
PYTHONPATH=. python tools/model_converters/glide_openai_to_mmagic.py \
  checkpoint/upsample.pt \
  --config configs/sgdiff/sgdiff-bf-style-64x64.py --module unet_up
```

看到 `RESULT: the checkpoint matches the model.` 即表示可以开始训练。把 `--module` 换成 `unet`、配置换成第一阶段的 `sgdiff-bf-glide-64x64.py`，可以同样校验 `laionide-v3-base.pt`。

## 7. 单卡 A30 训练

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

## 8. 四卡 A30（可选）

为保持与单卡配置相同的全局 batch size，四卡时需按下面命令降低每卡 batch size：

```bash
bash tools/dist_train.sh configs/sgdiff/sgdiff-bf-glide-64x64.py 4 --amp \
  --cfg-options train_dataloader.batch_size=2

bash tools/dist_train.sh configs/sgdiff/sgdiff-bf-style-64x64.py 4 --amp \
  --cfg-options train_dataloader.batch_size=4
```

## 9. 常见问题

- `torch.cuda.is_available()` 为 `False`：先确认 `nvidia-smi` 正常，再重新安装第 3 节的 CUDA 版 PyTorch。
- `ModuleNotFoundError: mmcv._ext`：MMCV 与 PyTorch/CUDA 不匹配，卸载 `mmcv` 后按第 3 节指定链接重装 `mmcv==2.1.0`。
- 显存不足：保留 `--amp`，并把 `train_dataloader.batch_size` 调小；单卡 batch size 变小后，学习率也应按比例降低。
- 预训练权重下载失败：配置已改为读取第 6 节的本地 `.pt` 文件，不再联网下载。若报 `FileNotFoundError`，检查 `glide_ckpt` / `glide_up_ckpt` 的路径是否为服务器上的实际路径。
- 加载权重时报 `Missing key(s)` / `Unexpected key(s)`：说明 `.pt` 文件与配置里的网络结构不一致（例如下错了模型）。用第 6 节的校验命令定位，它会直接列出对不上的参数名。
