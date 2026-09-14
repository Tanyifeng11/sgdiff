"""SGDiff 的配对图像评估；指标模型在生成结束后依次加载。"""

import csv
import gc
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _metric_names(names):
    names = tuple(dict.fromkeys(names))
    unknown = set(names) - {'fid', 'clip_i', 'ssim'}
    if unknown:
        raise ValueError(f'不支持的指标：{sorted(unknown)}')
    return names


def _open_rgb(path):
    with Image.open(path) as image:
        return image.convert('RGB')


def _release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_fid(device):
    # 局部加载项目原有实现，避免导入所有评估器及其无关依赖。
    path = (Path(__file__).resolve().parents[1] / 'mmagic' / 'evaluation'
            / 'functional' / 'fid_inception.py')
    spec = importlib.util.spec_from_file_location('sgdiff_fid_inception', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.InceptionV3(
        output_blocks=[3], resize_input=False, normalize_input=True,
        use_fid_inception=True, load_fid_inception=True).eval().to(device)


def _load_clip(model_name, device):
    from transformers import (AutoConfig, CLIPImageProcessor, CLIPModel,
                              CLIPVisionModelWithProjection)

    model_path = Path(model_name)
    nested = model_path / 'models' / 'image_encoder'
    source = str(nested) if (nested / 'config.json').is_file() else model_name
    config = AutoConfig.from_pretrained(source)
    model_cls = (CLIPVisionModelWithProjection
                 if config.model_type == 'clip_vision_model' else CLIPModel)
    model = model_cls.from_pretrained(source).eval().to(device)
    model.requires_grad_(False)
    processor_source = (model_name if (model_path / 'preprocessor_config.json')
                        .is_file() else source)
    if Path(source).is_dir() and not (
            Path(processor_source) / 'preprocessor_config.json').is_file():
        # 仅有视觉权重时，使用模型尺寸和 CLIP 官方归一化参数。
        vision_config = getattr(config, 'vision_config', config)
        size = vision_config.image_size
        print(f'本地 CLIP 缺少预处理配置，使用 CLIP 默认预处理，尺寸 {size}。',
              flush=True)
        processor = CLIPImageProcessor(
            size={'shortest_edge': size},
            crop_size={'height': size, 'width': size})
    else:
        processor = CLIPImageProcessor.from_pretrained(processor_source)
    return model, processor


def prepare_metrics(metric_names, clip_model, device='cpu'):
    """训练前检查依赖、下载并试加载指标权重，随后释放模型。"""
    for name in _metric_names(metric_names):
        print(f'检查评估指标：{name}', flush=True)
        if name == 'fid':
            from scipy.linalg import sqrtm  # noqa: F401
            model = _load_fid(device)
            del model
        elif name == 'clip_i':
            model, processor = _load_clip(clip_model, device)
            del model, processor
        else:
            from skimage.metrics import structural_similarity  # noqa: F401
        _release_memory()


@torch.no_grad()
def _extract_features(paths, model, device, batch_size, processor=None):
    features = []
    for offset in range(0, len(paths), batch_size):
        images = [_open_rgb(path) for path in paths[offset:offset + batch_size]]
        if processor is None:
            # 与 Mymodel 主流程对齐：PIL bicubic 299、RGB [0, 1]。
            batch = torch.stack([
                torch.from_numpy(np.asarray(image.resize((299, 299),
                                 Image.BICUBIC), dtype=np.float32) / 255.)
                .permute(2, 0, 1) for image in images]).to(device)
            output = model(batch)[0].flatten(1)
        else:
            inputs = {key: value.to(device) for key, value in
                      processor(images=images, return_tensors='pt').items()}
            output = (model.get_image_features(**inputs)
                      if hasattr(model, 'get_image_features')
                      else model(**inputs).image_embeds)
            output = torch.nn.functional.normalize(output.float(), dim=-1)
        features.append(output.float().cpu().numpy())
    return np.concatenate(features)


def _frechet_distance(generated, reference):
    """使用协方差矩阵平方根计算分布距离，不逐元素裁剪矩阵。"""
    from scipy.linalg import sqrtm

    generated, reference = np.asarray(generated, dtype=np.float64), np.asarray(
        reference, dtype=np.float64)
    if min(len(generated), len(reference)) < 2:
        raise ValueError('FID 至少需要 2 张生成图和 2 张真实图。')
    difference = generated.mean(0) - reference.mean(0)
    cov_g = np.atleast_2d(np.cov(generated, rowvar=False))
    cov_r = np.atleast_2d(np.cov(reference, rowvar=False))
    covariance_root = sqrtm(cov_g @ cov_r)
    if not np.isfinite(covariance_root).all():
        offset = np.eye(cov_g.shape[0]) * 1e-6
        covariance_root = sqrtm((cov_g + offset) @ (cov_r + offset))
    if np.iscomplexobj(covariance_root):
        if not np.allclose(np.diag(covariance_root).imag, 0, atol=1e-3):
            raise ValueError('FID 协方差平方根出现显著虚部，无法可靠计算。')
        covariance_root = covariance_root.real
    total = float(difference @ difference + np.trace(cov_g) + np.trace(cov_r))
    value = total - 2 * float(np.trace(covariance_root))
    if not np.isfinite(value) or value < -1e-6 * max(1., total):
        raise ValueError(f'FID 数值计算失败：{value}')
    return max(0., value)


def _ssim_full(generated, target):
    from skimage.metrics import structural_similarity

    image = _open_rgb(generated)
    reference = _open_rgb(target).resize(image.size, Image.BICUBIC)
    # 老版 skimage 没有 channel_axis；不得回退到非 SSIM 的近似指标。
    options = ({'channel_axis': 2} if 'channel_axis' in inspect.signature(
        structural_similarity).parameters else {'multichannel': True})
    return float(structural_similarity(np.asarray(image), np.asarray(reference),
                                      data_range=255, **options))


def evaluate_metrics(samples, generated_dir, output_dir,
                     metric_names=('fid', 'clip_i', 'ssim'),
                     clip_model='openai/clip-vit-large-patch14', device='cuda:0',
                     batch_size=16):
    """按样本 ID 配对，保存总体指标和逐图指标；FID 只在总体中报告。"""
    names = _metric_names(metric_names)
    if not samples or batch_size < 1:
        raise ValueError('评估需要非空样本和正整数 batch_size。')
    identifiers = [sample['id'] for sample in samples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('评估样本 ID 重复。')
    generated = [str(Path(generated_dir) / f'{key}.png') for key in identifiers]
    targets = [sample['target'] for sample in samples]
    styles = [sample['style'] for sample in samples]
    for path in generated + targets + (styles if 'clip_i' in names else []):
        if not Path(path).is_file():
            raise FileNotFoundError(f'缺少评估图像：{path}')
    rows = [dict(sample, generated=path) for sample, path in zip(samples, generated)]
    result = {'num_samples': len(samples), 'requested_metrics': list(names),
              'definitions': {}}
    if 'fid' in names:
        result['definitions']['fid'] = {
            'backend': 'mmagic.fid_inception.InceptionV3(use_fid_inception=True)',
            'feature_dim': 2048,
            'weights': 'pt_inception-2015-12-05-6726825d.pth',
            'preprocess': 'RGB; PIL bicubic 299x299; [0,1]; FID model -> [-1,1]'}
        if len(samples) < 2:
            result.update(fid=None, fid_reason='至少需要 2 个样本才能估计协方差。')
        else:
            print('计算 FID（标准 Inception 2048 维）。', flush=True)
            model = _load_fid(device)
            try:
                gen_features = _extract_features(generated, model, device, batch_size)
                ref_features = _extract_features(targets, model, device, batch_size)
            finally:
                del model
                _release_memory()
            result['fid'] = _frechet_distance(gen_features, ref_features)
            del gen_features, ref_features
    if 'clip_i' in names:
        print('计算 CLIP-I（生成图分别对 GT 和纹理图）。', flush=True)
        model, processor = _load_clip(clip_model, device)
        result['definitions']['clip_i'] = {
            'model': clip_model, 'backend': type(model).__name__,
            'preprocess': processor.to_dict(),
            'clip_i': '生成图与配对 GT 的归一化图像特征余弦相似度',
            'clip_i_style': '生成图与配对纹理图的归一化图像特征余弦相似度'}
        try:
            gen_features = _extract_features(
                generated, model, device, batch_size, processor)
            for key, paths in (('clip_i', targets), ('clip_i_style', styles)):
                ref_features = _extract_features(paths, model, device, batch_size, processor)
                values = np.sum(gen_features * ref_features, axis=1)
                result[key] = float(values.mean())
                result[key + '_std'] = float(values.std())
                for row, value in zip(rows, values):
                    row[key] = float(value)
            del gen_features, ref_features
        finally:
            del model, processor
            _release_memory()
    if 'ssim' in names:
        print('计算全图 RGB SSIM。', flush=True)
        values = [_ssim_full(path, target) for path, target in zip(generated, targets)]
        result.update(ssim_full=float(np.mean(values)), ssim_full_std=float(np.std(values)))
        result['definitions']['ssim_full'] = (
            'skimage structural_similarity; full-image RGB; data_range=255; '
            'GT PIL bicubic resize to generated size; no foreground mask')
        for row, value in zip(rows, values):
            row['ssim_full'] = value
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'metrics.json').open('w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    with (output / 'per_sample_metrics.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result
