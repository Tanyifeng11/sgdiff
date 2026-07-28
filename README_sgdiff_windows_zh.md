# SGDiff Windows 环境配置与排障 README

本文档根据实际在 **Windows + Conda + PowerShell** 环境中为 SGDiff 项目排障、安装、推理的全过程整理而成，目标是：

- 用一套 **可落地** 的方式跑通 `inference.py`
- 避开项目原始依赖中会在 Windows / 老版本 PyTorch 上踩到的坑
- 记录最终验证通过的版本组合与必要补丁

> 适用场景：
> - Windows 10/11
> - NVIDIA GPU
> - PowerShell
> - 仅需先跑通 **推理**，不涉及训练复现

---

## 1. 项目背景与原则

SGDiff 仓库本身基于 **MMagic 1.1.0 系列代码**，但直接按照仓库默认依赖安装，在 Windows 上容易遇到以下问题：

1. `torch 1.10` 与某些新版本包不兼容
2. `mmcv` 装到 `2.2.0` 会与仓库内置的 `mmagic` 版本断言冲突
3. `av` 会破坏 `torchvision` 的导入链
4. `diffusers` / `transformers` / `huggingface_hub` 如果装到过新版本，会与 `torch 1.10` 或彼此产生兼容性问题
5. `mmagic` 内部 `__init__.py` 的全量导入，会把 **和 SGDiff 推理无关** 的模块一起拉进来，导致 `mediapipe` / `controlnet_aux` / `animatediff` 等支线报错

因此，本 README 采用以下原则：

- **固定老版本 PyTorch 栈**，与项目更一致
- **显式锁定关键版本**，避免 pip 自动升到过新版本
- **只保留推理真正需要的依赖**
- **对项目源码做最小补丁**，绕开无关模块的导入

---

## 2. 最终验证通过的环境版本

推荐版本组合如下：

- Python: `3.9`
- PyTorch: `1.10.0`
- torchvision: `0.11.0`
- torchaudio: `0.10.0`
- cudatoolkit: `11.3`
- mmcv: `2.1.0`
- mmengine: `<1.0.0`
- numpy: `1.26.4`
- opencv-python: `4.8.1.78`
- diffusers: `0.24.0`
- transformers: `4.46.3`
- tokenizers: `0.20.3`
- huggingface_hub: `0.25.2`

不建议：

- `mmcv==2.2.0`
- `diffusers` 最新版
- `transformers` 最新版
- `huggingface_hub` 最新版
- `numpy 2.x`
- `opencv-python 4.13.x`
- `av`

---

## 3. 新建环境

在 **PowerShell** 中执行：

```powershell
conda create -n sgdiff python=3.9 -y
conda activate sgdiff
```

---

## 4. 安装 PyTorch

```powershell
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c defaults --strict-channel-priority
```

安装后验证：

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

期望输出类似：

```text
1.10.0 11.3 True
```

---

## 5. 修复 Windows 下 DLL 搜索问题

这一项在 Windows 上非常重要。建议固化到 conda 环境：

```powershell
conda env config vars set CONDA_DLL_SEARCH_MODIFICATION_ENABLE=1
conda deactivate
conda activate sgdiff
```

再次验证 torch：

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

---

## 6. 安装 OpenMMLab 基础依赖

### 6.1 安装 pip / mim 基础工具

```powershell
python -m pip install -U pip setuptools wheel openmim
```

### 6.2 安装 mmcv

**不要** 用宽泛的：

```powershell
mim install "mmcv>=2.0.0"
```

因为它可能装到 `2.2.0`，与项目内置版本约束冲突。

请固定为：

```powershell
python -m pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html --trusted-host download.openmmlab.com --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

### 6.3 安装 mmengine

```powershell
python -m pip install "mmengine>=0.4.0,<1.0.0"
```

### 6.4 验证

```powershell
python -c "import mmcv, mmengine; print(mmcv.__version__); print(mmengine.__version__)"
```

期望：

- `mmcv` 为 `2.1.0`
- `mmengine` 正常导入

---

## 7. 安装 SGDiff 运行时依赖（手动锁版本）

### 7.1 先装 lmdb

```powershell
conda install -c conda-forge python-lmdb -y
```

### 7.2 安装 Hugging Face 相关依赖（必须锁版本）

```powershell
python -m pip install --force-reinstall "transformers==4.46.3" "tokenizers==0.20.3" "huggingface_hub==0.25.2" "diffusers==0.24.0"
```

### 7.3 降回 NumPy 1.x

```powershell
python -m pip install "numpy<2"
```

### 7.4 安装兼容的 OpenCV

先卸载冲突版本：

```powershell
python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python av
```

再安装兼容版本：

```powershell
python -m pip install "opencv-python==4.8.1.78"
```

### 7.5 安装其余常用依赖

```powershell
python -m pip install einops "face-alignment<=1.3.4" facexlib lpips mediapipe controlnet_aux pandas Pillow resize_right tensorboard
```

> 说明：
> - `mediapipe` / `controlnet_aux` 可能在后续仍通过无关导入链引发问题，因此下面还会做源码补丁。
> - `av` 不建议安装。

### 7.6 版本验证

```powershell
python -c "import numpy, cv2; print('numpy', numpy.__version__); print('cv2', cv2.__version__)"
python -c "import transformers, tokenizers, huggingface_hub, diffusers; print(transformers.__version__); print(tokenizers.__version__); print(huggingface_hub.__version__); print(diffusers.__version__)"
```

期望：

- numpy `1.26.4`
- cv2 `4.8.1`
- transformers `4.46.3`
- tokenizers `0.20.3`
- huggingface_hub `0.25.2`
- diffusers `0.24.0`

---

## 8. 安装项目本体

如果 `pip install -e .` 报 editable / build backend 相关错误，可以先不做 editable 安装。

对仅运行推理来说，**在仓库根目录直接运行 `inference.py` 就可以**。

如果仍希望安装本体，可尝试：

```powershell
python -m pip install . --no-deps --no-build-isolation
```

如果不安装本体，也可以继续下面的步骤，只要命令在项目根目录执行即可。

---

## 9. 必要源码补丁（关键）

为了跑通 SGDiff 推理，需要把 `mmagic` 中一些**与当前任务无关**的全量导入去掉，否则会在导入阶段拉起 `mediapipe`、`controlnet_aux`、`animatediff` 等支线。

### 9.1 精简 `mmagic/apis/inferencers/__init__.py`

先备份：

```powershell
Copy-Item .\mmagic\apis\inferencers\__init__.py .\mmagic\apis\inferencers\__init__.py.bak
```

改成以下内容：

```python
from .inference_functions import init_model

__all__ = ['init_model']
```

PowerShell 一键写入：

```powershell
@'
from .inference_functions import init_model

__all__ = ['init_model']
'@ | Set-Content .\mmagic\apis\inferencers\__init__.py -Encoding UTF8
```

### 9.2 精简 `mmagic/apis/__init__.py`

先备份：

```powershell
Copy-Item .\mmagic\apis\__init__.py .\mmagic\apis\__init__.py.bak
```

改成以下内容：

```python
from .inferencers.inference_functions import init_model

__all__ = ['init_model']
```

PowerShell 一键写入：

```powershell
@'
from .inferencers.inference_functions import init_model

__all__ = ['init_model']
'@ | Set-Content .\mmagic\apis\__init__.py -Encoding UTF8
```

### 9.3 精简 `mmagic/models/editors/__init__.py`

先备份：

```powershell
Copy-Item .\mmagic\models\editors\__init__.py .\mmagic\models\editors\__init__.py.bak
```

改成以下内容：

```python
from .sgdiff import SGDiff

__all__ = ['SGDiff']
```

PowerShell 一键写入：

```powershell
@'
from .sgdiff import SGDiff

__all__ = ['SGDiff']
'@ | Set-Content .\mmagic\models\editors\__init__.py -Encoding UTF8
```

### 9.4 修改 `inference.py` 的导入

将：

```python
from mmagic.apis import init_model
```

改为：

```python
from mmagic.apis.inferencers.inference_functions import init_model
```

PowerShell 一键替换：

```powershell
(Get-Content .\inference.py) -replace 'from mmagic\.apis import init_model','from mmagic.apis.inferencers.inference_functions import init_model' | Set-Content .\inference.py -Encoding UTF8
```

---

## 10. 权重与输入文件位置

确认以下路径存在：

```powershell
Test-Path .\checkpoint\sgdiff.pth
Test-Path .\examples\starry_night.jpg
```

注意：

- 是 `checkpoint`，不是 `checkpoints`
- 若返回 `True`，说明路径正确

---

## 11. 运行推理

在项目根目录运行：

```powershell
python -u .\inference.py --ckpt .\checkpoint\sgdiff.pth --img_path .\examples\starry_night.jpg --prompt "long sleeve jumpsuit" --output_path .\results.png
```

成功时通常会看到：

- 一行 checkpoint 加载提示
- 两段进度条
- 最终生成 `results.png`

检查输出：

```powershell
dir .\results.png
```

如果看到文件存在，例如：

```text
results.png
```

说明推理已经跑通。

---

## 12. 这张图是怎么生成出来的？

SGDiff 不是只看提示词，也不是只看参考图，而是同时用：

- `--prompt`：服装属性语义
- `--img_path`：参考风格图

例如下面这条命令：

```powershell
python .\inference.py --ckpt .\checkpoint\sgdiff.pth --img_path .\examples\starry_night.jpg --prompt "long sleeve jumpsuit" --output_path .\results.png
```

表示：

- 语义上生成“**长袖连体衣**”
- 风格上参考 `starry_night.jpg`

所以生成结果是 **“提示词控制服装内容 + 输入图控制风格/纹理”**。

---

## 13. 常见问题总结

### 13.1 `WinError 182` / `caffe2_detectron_ops_gpu.dll`

表现：

- `import torch` 即失败

解决：

- 新建全新 conda 环境
- 使用 `pytorch==1.10.0 + cudatoolkit=11.3`
- 设置 `CONDA_DLL_SEARCH_MODIFICATION_ENABLE=1`

### 13.2 `mmcv==2.2.0 is used but incompatible`

原因：

- 项目内置 `mmagic` 要求 `mmcv < 2.2.0`

解决：

- 固定安装 `mmcv==2.1.0`

### 13.3 `av` 导致 `torchvision` 导入报错

解决：

```powershell
python -m pip uninstall -y av
```

### 13.4 `diffusers` 报 `torch.xpu` 错误

原因：

- `diffusers` 太新，和 `torch 1.10` 不兼容

解决：

- 降级到 `diffusers==0.24.0`

### 13.5 `cached_download` 导入失败

原因：

- `diffusers 0.24.0` 与过新的 `huggingface_hub` 不兼容

解决：

- 固定 `huggingface_hub==0.25.2`

### 13.6 `mediapipe` / `controlnet_aux` / `animatediff` 无关报错

原因：

- `mmagic` 的 `__init__.py` 全量导入了无关模块

解决：

- 按第 9 节对几个 `__init__.py` 做精简补丁

### 13.7 `results.png` 不存在

排查顺序：

```powershell
$LASTEXITCODE
Get-ChildItem -Recurse -Filter results.png
```

若退出码非 0，则说明脚本失败，需要抓 traceback。

---

## 14. 建议

### 适合当前 README 的场景

- 先跑通 **推理**
- Windows 本机直接运行
- 不追求完全保持仓库原始安装逻辑

### 不建议的场景

- 直接复现训练
- 一味使用所有最新依赖
- 把 MMagic 主分支最新环境直接套到 SGDiff 上

### 如果后面还要继续深入

建议：

- 迁移到 **WSL2 / Ubuntu**
- 或单独为训练另建环境

因为这个项目公开仓库本身更偏向**推理可用**，训练闭环并不完整。

---

## 15. 一套最小可执行命令清单

```powershell
conda create -n sgdiff python=3.9 -y
conda activate sgdiff

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c defaults --strict-channel-priority
conda env config vars set CONDA_DLL_SEARCH_MODIFICATION_ENABLE=1
conda deactivate
conda activate sgdiff

python -m pip install -U pip setuptools wheel openmim
python -m pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html --trusted-host download.openmmlab.com --trusted-host pypi.org --trusted-host files.pythonhosted.org
python -m pip install "mmengine>=0.4.0,<1.0.0"

conda install -c conda-forge python-lmdb -y
python -m pip install --force-reinstall "transformers==4.46.3" "tokenizers==0.20.3" "huggingface_hub==0.25.2" "diffusers==0.24.0"
python -m pip install "numpy<2"
python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python av
python -m pip install "opencv-python==4.8.1.78"
python -m pip install einops "face-alignment<=1.3.4" facexlib lpips mediapipe controlnet_aux pandas Pillow resize_right tensorboard

# 修改源码补丁（见第 9 节）

python -u .\inference.py --ckpt .\checkpoint\sgdiff.pth --img_path .\examples\starry_night.jpg --prompt "long sleeve jumpsuit" --output_path .\results.png
```

---

## 16. 最终结论

经过上述版本固定与源码补丁后，SGDiff 已在 Windows + PowerShell + Conda 环境中成功跑通推理，并生成 `results.png`。

如果你后续要把这个 README 放回项目根目录，建议命名为：

```text
README_WINDOWS_SETUP_ZH.md
```

