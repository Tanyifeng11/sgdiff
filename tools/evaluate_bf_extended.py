"""复用 Mymodel 的评估口径，为 SGDiff 已生成结果补充对比指标。"""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PAIR_KEYS = ('clip_i_real', 'clip_i_texture', 'tpf_patch_sim', 'tpf_gram_l1',
             'tcf_lab_delta', 'struct_edge_f1', 'struct_iou',
             'leak_colored_frac', 'leak_edge_density',
             'prompt_color_delta_e', 'target_color_delta_e')


def configure_legacy_torchvision(backend):
    """仅替换旧 torchvision 的模型加载接口，保留原指标计算与归一化。"""
    from torchvision import models
    from torchvision.transforms import Normalize

    if hasattr(models, 'VGG19_Weights'):
        return

    def inception(device='cuda'):
        if getattr(backend, '_inception_v3', None) is None:
            model = models.inception_v3(pretrained=True, transform_input=False)
            model.fc = torch.nn.Identity()
            model.eval().requires_grad_(False)
            backend._inception_v3 = (model, Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        model, normalize = backend._inception_v3
        return model.to(device), normalize

    def vgg(device='cuda'):
        if getattr(backend, '_vgg_gram', None) is None:
            model = models.vgg19(pretrained=True).features.eval()
            model.requires_grad_(False)
            backend._vgg_gram = (model, Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        model, normalize = backend._vgg_gram
        return model.to(device), normalize

    backend._get_inception_v3 = inception
    backend._get_vgg_gram = vgg
    print('启用旧 torchvision 兼容接口：pretrained=True', flush=True)


def paired_file(directory, stem):
    matches = [p for p in directory.glob(stem + '.*')
               if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.webp')]
    if len(matches) != 1:
        raise ValueError(f'需要唯一配对图像：{directory / stem}，找到 {len(matches)} 个')
    return str(matches[0])


def summarize(rows):
    result = {}
    for key in PAIR_KEYS:
        values = [float(row[key]) for row in rows
                  if row.get(key) is not None and math.isfinite(float(row[key]))]
        result[key + '_mean'] = float(np.mean(values)) if values else None
        result[key + '_std'] = float(np.std(values)) if values else None
        result[key + '_valid'] = len(values)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, help='含 manifest.json 的生成目录')
    parser.add_argument('--mymodel-root', default='/share/home/u2515283058/Mymodel')
    parser.add_argument('--clip-model', default='/share/home/u2515283058/Mymodel/models/clip/models/image_encoder')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size 必须大于零')
    root = Path(args.mymodel_root).resolve()
    if not (root / 'eval' / 'metrics.py').is_file():
        raise FileNotFoundError(f'缺少 Mymodel 指标实现：{root}')
    sys.path.insert(0, str(root))
    from eval.metrics import evaluate_full, compute_clip_i_values
    from eval.distribution_diagnostics import compute_distribution_metrics
    from eval.eval_utils import prepare_evaluation_masks, json_safe
    from color_conflict_utils import extract_text_color, dominant_rgb_from_pil, delta_e_rgb
    from garment_mask_utils import mask_backend_info
    import eval.metrics as backend
    configure_legacy_torchvision(backend)

    output = Path(args.output_dir)
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    samples = manifest['samples']
    sketches = [paired_file(Path(s['target']).parent.parent / 'sketch',
                            Path(s['target']).stem) for s in samples]
    if args.check_only:
        # 提前验证 VGG/Inception 接口与权重，避免生成完成后才发现不兼容。
        backend._get_vgg_gram('cpu')
        backend._get_inception_v3('cpu')
        print(f'扩展指标预检查通过：{len(samples)} 个草图配对；mask_policy=sketch_only')
        return
    generated = [str(output / 'generated' / (s['id'] + '.png')) for s in samples]
    targets = [str(output / 'real' / (s['id'] + '.png')) for s in samples]
    textures = [str(output / 'texture' / (s['id'] + '.png')) for s in samples]
    for path in generated + targets + textures:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    rows = []
    with torch.no_grad():
        for index, (sample, gen, target, texture, sketch) in enumerate(
                zip(samples, generated, targets, textures, sketches)):
            with Image.open(gen) as image:
                image = image.convert('RGB')
                bundle = prepare_evaluation_masks(
                    image.size, sketch_path=sketch, target_path=target,
                    gen_path=gen, mask_policy='sketch_only')
                mask = bundle.get('garment')
                mask_image = Image.fromarray(mask.astype(np.uint8) * 255) if mask is not None else None
                row = dict(id=sample['id'], sample_id=sample['sample_id'],
                           **evaluate_full(gen, target_path=target, texture_path=texture,
                                           sketch_path=sketch, mask_bundle=bundle))
                row['prompt_color_delta_e'] = None
                row['target_color_delta_e'] = None
                if mask is not None and int(mask.sum()) >= 50:
                    gen_rgb = dominant_rgb_from_pil(image, mask_image)
                    _, text_rgb = extract_text_color(sample['prompt'])
                    if text_rgb is not None:
                        row['prompt_color_delta_e'] = delta_e_rgb(text_rgb, gen_rgb)
                    with Image.open(target) as reference:
                        row['target_color_delta_e'] = delta_e_rgb(
                            dominant_rgb_from_pil(reference.convert('RGB'), mask_image), gen_rgb)
                rows.append(row)
            print(f'扩展指标：{index + 1}/{len(samples)} {sample["id"]}', flush=True)
    # 释放逐图指标使用的 VGG，再加载 CLIP 和 Inception。
    backend._vgg_gram = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for key, reference in (('clip_i_real', targets), ('clip_i_texture', textures)):
        values = compute_clip_i_values(generated, reference, batch_size=args.batch_size,
                                       device=args.device, model_name=args.clip_model)
        for row, value in zip(rows, values):
            row[key] = float(value)
    backend._clip_model_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    distribution = compute_distribution_metrics(
        targets, generated, device=args.device, batch_size=args.batch_size,
        seed=42, kid_subsets=50, kid_subset_size=100)
    result = dict(num_samples=len(samples), **summarize(rows), **distribution)
    result['fid'] = distribution['legacy_fid']
    result['kid'] = distribution['kid_mean']
    result['definitions'] = dict(
        mymodel_root=str(root), mask_policy='sketch_only',
        mask_backend=mask_backend_info(), clip_model=args.clip_model,
        kid_backend='Mymodel extract_inception_features (torchvision Inception)',
        kid_seed=42, min_valid_pixels=50,
        manifest_sha256=hashlib.sha256((output / 'manifest.json').read_bytes()).hexdigest(),
        evaluation_size='generated image native size; no additional resize',
        implementation_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in ('eval/metrics.py', 'eval/eval_utils.py',
                                            'eval/distribution_diagnostics.py',
                                            'garment_mask_utils.py', 'color_conflict_utils.py')})
    (output / 'metrics_extended.json').write_text(
        json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with (output / 'per_sample_extended.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['id', 'sample_id', *PAIR_KEYS, 'metric_warnings'])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in writer.fieldnames})
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))
    unavailable = [key for key in PAIR_KEYS if result[key + '_valid'] < len(samples)]
    if unavailable:
        print('部分样本指标无效或不适用，请检查 valid 和逐图 metric_warnings：'
              + ', '.join(unavailable), flush=True)


if __name__ == '__main__':
    main()
