# Copyright (c) OpenMMLab. All rights reserved.
import random
from pathlib import Path

import numpy as np
import torch
from mmengine.dataset import pseudo_collate
from mmengine.dist import barrier, is_main_process
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from PIL import Image

from mmagic.registry import DATASETS, HOOKS


@HOOKS.register_module()
class SGDiffVisualizationHook(Hook):
    """用固定验证样本分别检查基础模型和超分模型的生成结果。"""

    priority = 'NORMAL'

    def __init__(self, dataset, interval=5000, num_samples=2, seed=2022):
        self.dataset_cfg = dataset
        self.interval = interval
        self.num_samples = num_samples
        self.seed = seed
        self._dataset = None

    def before_train(self, runner):
        # 此时 Runner 已完成初始化和 checkpoint 恢复，检查实际使用的权重。
        model = self._unwrap(runner.model)
        if model.unet_up is not None:
            upsampler = self._unwrap(model.unet_up)
            if torch.count_nonzero(upsampler.out.conv.weight.detach()) == 0:
                raise RuntimeError(
                    '超分输出层权重全零，无法正常去噪。请使用原始 upsample.pt '
                    '恢复整个 unet_up，勿继续使用已经损坏的超分权重。')
        self._sample(runner, runner.iter)

    def after_train_iter(self,
                         runner,
                         batch_idx,
                         data_batch=None,
                         outputs=None):
        if self.every_n_train_iters(runner, self.interval):
            self._sample(runner, runner.iter + 1)

    @staticmethod
    def _unwrap(model):
        return model.module if is_model_wrapper(model) else model

    def _sample(self, runner, step):
        # 其他进程等待主进程采样，避免与下一轮分布式训练交错。
        barrier()
        try:
            if is_main_process():
                self._sample_on_main(runner, step)
        finally:
            barrier()

    @torch.no_grad()
    def _sample_on_main(self, runner, step):
        model = self._unwrap(runner.model)
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        module_modes = [(module, module.training)
                        for module in model.modules()]
        cuda_devices = (list(range(torch.cuda.device_count()))
                        if torch.cuda.is_available() else [])
        try:
            # 采样不推进训练使用的随机数序列，包括首次构建验证数据集。
            with torch.random.fork_rng(devices=cuda_devices):
                random.seed(self.seed)
                np.random.seed(self.seed)
                torch.manual_seed(self.seed)
                model.eval()
                if self._dataset is None:
                    self._dataset = DATASETS.build(self.dataset_cfg)
                output_dir = Path(runner.work_dir) / 'samples' / (
                    f'iter_{step:06d}')
                output_dir.mkdir(parents=True, exist_ok=True)
                for index in range(min(self.num_samples, len(self._dataset))):
                    # 逐张生成，避免验证时额外占用多个样本的显存。
                    batch = pseudo_collate([self._dataset[index]])
                    inputs = model.data_preprocessor(
                        batch, training=False)['inputs']
                    conditions = {
                        key: inputs[key]
                        for modality in model.modalities
                        for key in model.MODALITIES[modality]
                    }
                    sample_kwargs = dict(model.val_cfg)
                    sample_kwargs.update(
                        batch_size=1,
                        run_up=model.unet_up is not None,
                        show_progress=False)
                    if len(model.modalities) > 1:
                        result = model.infer_mm(**sample_kwargs, **conditions)
                    else:
                        result = model.infer(**sample_kwargs, **conditions)
                    sample_id = self._dataset.sample_ids[index]
                    self._save_image(result['low_res_samples'],
                                     output_dir / f'{sample_id}_64.png')
                    if model.unet_up is not None:
                        self._save_image(result['samples'],
                                         output_dir / f'{sample_id}_256.png')
                    self._save_image(inputs['img'],
                                     output_dir / f'{sample_id}_gt.png')
                    if 'style' in inputs:
                        self._save_image(inputs['style'],
                                         output_dir / f'{sample_id}_style.png')
                runner.logger.info(f'固定验证样本已保存至 {output_dir}')
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            # 按子模块原始状态恢复，保留冻结模块原有的 eval 状态。
            for module, training in module_modes:
                module.training = training

    @staticmethod
    def _save_image(tensor, path):
        pixels = tensor[0].detach().float().cpu().clamp(-1, 1)
        pixels = ((pixels + 1) * 127.5).round().to(torch.uint8)
        Image.fromarray(pixels.permute(1, 2, 0).numpy()).save(path)
