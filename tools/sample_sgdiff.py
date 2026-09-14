"""分别检查 SGDiff 的 64×64 基础生成和 256×256 超分结果。"""

import argparse
from copy import deepcopy
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='与 checkpoint 阶段对应的配置文件')
    parser.add_argument('checkpoint', help='训练保存的完整 checkpoint')
    parser.add_argument('--prompt', required=True, help='描述目标服装的文本')
    parser.add_argument('--style', help='第二阶段所需的风格参考图')
    parser.add_argument(
        '--output-dir', default='work_dirs/sgdiff_samples', help='图片保存目录')
    parser.add_argument('--device', default='cuda:0', help='采样设备，例如 cuda:0 或 cpu')
    parser.add_argument('--seed', type=int, default=0, help='固定随机种子')
    parser.add_argument(
        '--num-inference-steps', type=int, default=100, help='基础模型的 DDIM 步数')
    parser.add_argument(
        '--up-inference-steps', type=int, default=35, help='超分模型的 DDIM 步数')
    parser.add_argument(
        '--upsample-ckpt',
        help='显式覆盖整个超分模块的权重，例如原始 upsample.pt；用于修复旧 checkpoint')
    parser.add_argument('--skip-up', action='store_true', help='只检查 64×64 基础输出')
    args = parser.parse_args()
    if args.num_inference_steps <= 0 or args.up_inference_steps <= 0:
        parser.error('采样步数必须大于零。')
    if args.skip_up and args.upsample_ckpt:
        parser.error('--skip-up 与 --upsample-ckpt 不需要同时使用。')
    return args


def load_model(config, checkpoint_path, device, upsample_ckpt=None,
               run_up=True):
    import torch
    from mmengine.runner.checkpoint import _load_checkpoint

    import mmagic.models  # noqa: F401
    from mmagic.apis.inferencers.inference_functions import init_model
    from mmagic.models.editors.glide.glide_ckpt import (
        load_glide_state_dict, unwrap_state_dict)

    config = deepcopy(config)
    # 完整 checkpoint 已包含生成参数，不再依赖训练配置中的前一阶段路径。
    config.model.unet.pretrained_cfg = None
    config.model.pretrained_cfgs = None
    config.model.perceptual_loss = None
    style_cfg = config.model.unet.get('style_encoder_cfg')
    if style_cfg is not None:
        style_cfg.pop('pretrained', None)

    model = init_model(config, checkpoint=None, device='cpu')
    checkpoint = _load_checkpoint(checkpoint_path, map_location='cpu')
    state_dict = unwrap_state_dict(checkpoint)
    state_dict = {
        key[7:] if key.startswith('module.') else key: value
        for key, value in state_dict.items()
    }
    # VGG 仅用于训练损失；其余缺失、多余或尺寸错误的参数均不能忽略。
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith('perceptual_loss.')
    }
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            'checkpoint 与配置不匹配。第一阶段 checkpoint 必须搭配第一阶段配置，'
            '第二阶段 checkpoint 必须搭配相应的风格配置。\n'
            f'{error}') from error
    del checkpoint, state_dict

    if upsample_ckpt:
        if model.unet_up is None:
            raise ValueError('当前配置没有超分模块，不能使用 --upsample-ckpt。')
        # 必须在完整 checkpoint 之后覆盖整个模块，不能只修复最终输出层。
        up_state = load_glide_state_dict(upsample_ckpt, prefix='unet_up')
        model.unet_up.load_state_dict(up_state, strict=True)
        del up_state
        print(f'已用 {upsample_ckpt} 恢复整个超分模块。')

    if run_up and model.unet_up is not None:
        if torch.count_nonzero(model.unet_up.out.conv.weight).item() == 0:
            raise RuntimeError(
                '超分输出层权重全零。请用 --upsample-ckpt 指定原始 upsample.pt '
                '恢复整个模块，或用 --skip-up 先检查 64×64 基础模型。')
    model.to(device)
    model.eval()
    return model


def load_style(path, device):
    import numpy as np
    import torch
    from PIL import Image

    # 与 BFDataset 保持相同的 RGB、双三次缩放及 [-1, 1] 输入范围。
    resampling = getattr(Image, 'Resampling', Image)
    with Image.open(path) as image:
        image = image.convert('RGB').resize((256, 256), resampling.BICUBIC)
        array = np.asarray(image, dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float()
    return ((tensor - 127.5) / 127.5).unsqueeze(0).to(device)


def save_sample(sample, path):
    from PIL import Image

    image = sample[0].detach().float().cpu().clamp(-1, 1)
    array = ((image + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(array).save(path)
    print(f'已保存 {path.resolve()}（{image.shape[-1]}×{image.shape[-2]}）')


def main():
    args = parse_args()
    import torch
    from mmengine.config import Config
    from mmengine.runner import set_random_seed

    config = Config.fromfile(args.config)
    modalities = set(config.model.get('modalities', ['txt', 'style']))
    if modalities not in ({'txt'}, {'txt', 'style'}):
        raise ValueError('此脚本支持第一阶段文本模型和第二阶段文本加风格模型。')
    if 'style' in modalities and not args.style:
        raise ValueError('第二阶段采样必须通过 --style 提供风格参考图。')
    if 'style' not in modalities and args.style:
        raise ValueError('第一阶段配置不使用风格图；请去掉 --style 或改用第二阶段配置。')

    set_random_seed(args.seed)
    model = load_model(
        config, args.checkpoint, args.device, args.upsample_ckpt,
        run_up=not args.skip_up)
    conditions = dict(
        prompt=args.prompt,
        batch_size=1,
        num_inference_steps=args.num_inference_steps,
        up_inference_steps=args.up_inference_steps,
        run_up=not args.skip_up,
        show_progress=True)
    # 权重构造会消耗随机数；在生成前重新设种子，便于比较新旧 checkpoint。
    set_random_seed(args.seed)
    with torch.no_grad():
        if 'style' in modalities:
            outputs = model.infer_mm(
                style=load_style(args.style, args.device),
                modality_order_cfg=dict(style=1.2, txt=1.0),
                **conditions)
        else:
            outputs = model.infer(
                guidance_scale=1.0, modalities=['txt'], **conditions)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_sample(outputs['low_res_samples'], output_dir / 'base_64.png')
    if model.unet_up is not None and not args.skip_up:
        save_sample(outputs['samples'], output_dir / 'sample_256.png')


if __name__ == '__main__':
    main()
