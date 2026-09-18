"""固定 BF 样本测试：准备清单、按 GPU 分片生成、统一计算指标。"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
TEST_CATEGORIES = ('top', 'outwear', 'pants', 'dress')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='与训练阶段对应的配置')
    parser.add_argument('checkpoint', help='该阶段的完整 checkpoint')
    parser.add_argument('--mode', choices=['all', 'prepare', 'generate', 'evaluate'],
                        default='all')
    parser.add_argument('--data-root', default='/share/home/u2515283058/datasets/BF')
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--split-file', help='可复用 Mymodel 的固定样本 JSON 清单')
    parser.add_argument('--all-test-categories', action='store_true',
                        help='读取 split 下全部类别目录，包括 bag 等额外类别')
    parser.add_argument('--max-samples', type=int, default=100, help='0 表示全部样本')
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--seed', type=int, default=42, help='逐样本种子为此值加 sample_id')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-inference-steps', type=int, default=100)
    parser.add_argument('--up-inference-steps', type=int, default=35)
    parser.add_argument('--text-guidance', type=float, default=1.0)
    parser.add_argument('--style-guidance', type=float, default=1.2)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--metrics', nargs='+', choices=['fid', 'clip_i', 'ssim'],
                        default=['fid', 'clip_i', 'ssim'])
    parser.add_argument('--clip-model', default='openai/clip-vit-large-patch14')
    parser.add_argument('--metric-batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.max_samples < 0 or min(args.num_inference_steps,
                                  args.up_inference_steps, args.metric_batch_size) < 1:
        parser.error('样本数不能为负数，采样步数与指标 batch 必须大于零。')
    return args


def image_index(directory, suffixes):
    if not directory.is_dir():
        raise FileNotFoundError(f'缺少测试目录：{directory}')
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in suffixes:
            if path.stem in result:
                raise ValueError(f'同名样本重复：{directory / path.stem}')
            result[path.stem] = path
    return result


def collect_samples(data_root, split='validation', max_samples=100,
                    split_seed=42, split_file=None, all_test_categories=False):
    """与 Mymodel 保持类顺序、抽样算法和 sample_id 对应关系。"""
    split_root = Path(data_root).resolve() / split
    samples = []
    if split_file:
        source = json.loads(Path(split_file).read_text(encoding='utf-8-sig'))
        for index, item in enumerate(source):
            def resolve(value):
                path = Path(value)
                return str(path if path.is_absolute() else split_root / path)

            target = resolve(item['target'])
            category = item.get('category', split)
            samples.append(dict(
                id=f'{category}__{Path(target).stem}',
                sample_id=str(item.get('sample_id', f'{index:06d}')),
                prompt=item['prompt'].strip(), target=target,
                style=resolve(item['texture']), category=category))
    else:
        # validation 为平铺目录；正式 test 沿用 Mymodel 的四个服装类别。
        categories = (split,) if (split_root / 'gt').is_dir() else TEST_CATEGORIES
        if all_test_categories and not (split_root / 'gt').is_dir():
            categories = sorted(path.name for path in split_root.iterdir()
                                if path.is_dir() and not path.name.startswith('.'))
        for category in categories:
            root = split_root if category == split else split_root / category
            targets = image_index(root / 'gt', IMAGE_SUFFIXES)
            styles = image_index(root / 'texture', IMAGE_SUFFIXES)
            texts = image_index(root / 'text', {'.txt'})
            if set(targets) != set(styles) or set(targets) != set(texts):
                raise ValueError(f'{root} 的 gt/texture/text 未按同名样本完整配对。')
            for stem in sorted(targets):
                samples.append(dict(
                    id=f'{category}__{stem}', sample_id=f'{len(samples):06d}',
                    prompt=' '.join(texts[stem].read_text(
                        encoding='utf-8-sig').strip().split()),
                    target=str(targets[stem]), style=str(styles[stem]),
                    category=category))
        if max_samples:
            random.Random(split_seed).shuffle(samples)
    if max_samples > len(samples):
        raise ValueError(f'只找到 {len(samples)} 个样本，无法抽取 {max_samples} 个。')
    if max_samples:
        samples = samples[:max_samples]
        if not split_file:
            for index, sample in enumerate(samples):
                sample['sample_id'] = f'{index:06d}'
    if not samples:
        raise ValueError('测试样本清单为空。')
    if len({sample['id'] for sample in samples}) != len(samples):
        raise ValueError('测试清单中存在重复样本。')
    for sample in samples:
        int(sample['sample_id'])
        if not sample['prompt']:
            raise ValueError(f"样本 {sample['id']} 的文本为空。")
        for key in ('target', 'style'):
            if not Path(sample[key]).is_file():
                raise FileNotFoundError(sample[key])
    return samples


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def prepare(args):
    from tools.sgdiff_metrics import prepare_metrics

    samples = collect_samples(args.data_root, args.split, args.max_samples,
                              args.split_seed, args.split_file,
                              args.all_test_categories)
    payload = dict(
        config=str(Path(args.config).resolve()),
        checkpoint=str(Path(args.checkpoint).resolve()),
        split=args.split, split_seed=args.split_seed, seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        up_inference_steps=args.up_inference_steps,
        text_guidance=args.text_guidance, style_guidance=args.style_guidance,
        samples=samples)
    path = Path(args.output_dir) / 'manifest.json'
    if path.is_file() and json.loads(path.read_text(encoding='utf-8')) != payload:
        raise ValueError(f'已有测试清单与本次参数不同，请使用新的 --output-dir：{path.parent}')
    # 在长时间训练之前检查测试数据、依赖及指标权重缓存。
    if not Path(args.config).is_file():
        raise FileNotFoundError(args.config)
    prepare_metrics(args.metrics, args.clip_model)
    write_json(path, payload)
    print(f'测试准备完成：{len(samples)} 个样本，清单：{path}', flush=True)


def read_manifest(args):
    path = Path(args.output_dir) / 'manifest.json'
    if not path.is_file():
        raise FileNotFoundError(f'请先运行 --mode prepare：{path}')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    expected = dict(config=str(Path(args.config).resolve()),
                    checkpoint=str(Path(args.checkpoint).resolve()),
                    split=args.split, seed=args.seed,
                    num_inference_steps=args.num_inference_steps,
                    up_inference_steps=args.up_inference_steps,
                    text_guidance=args.text_guidance, style_guidance=args.style_guidance)
    if any(manifest[key] != value for key, value in expected.items()):
        raise ValueError('测试参数与已准备的 manifest.json 不一致。')
    return manifest


def save_png(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    image.save(temporary, format='PNG')
    temporary.replace(path)


def generate(args):
    import numpy as np
    import torch
    from mmengine.config import Config
    from PIL import Image, ImageDraw
    from tools.sample_sgdiff import load_model, load_style

    manifest = read_manifest(args)
    samples = manifest['samples']
    rank, world_size = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = f'cuda:{local_rank}' if world_size > 1 else (
        'cuda:0' if args.device == 'cuda' else args.device)
    if device.startswith('cuda'):
        torch.cuda.set_device(torch.device(device))
    config = Config.fromfile(args.config)
    checkpoint = Path(args.checkpoint)
    stat = checkpoint.stat()
    signature = dict(checkpoint=str(checkpoint.resolve()), size=stat.st_size,
                     mtime_ns=stat.st_mtime_ns,
                     config_sha256=hashlib.sha256(repr(config.to_dict()).encode()).hexdigest())
    output = Path(args.output_dir)
    signature_path = output / 'generation.json'
    if signature_path.is_file():
        if json.loads(signature_path.read_text(encoding='utf-8')) != signature:
            raise ValueError('checkpoint 或配置已变化，请使用新的测试输出目录。')
    elif rank == 0:
        write_json(signature_path, signature)
    start, end = rank * len(samples) // world_size, (rank + 1) * len(samples) // world_size
    directories = ('generated', 'base_64', 'real', 'texture', 'comparison')
    pending = [sample for sample in samples[start:end] if not all(
        (output / name / f"{sample['id']}.png").is_file() for name in directories)]
    print(f'GPU 进程 {rank}/{world_size}：样本 [{start}, {end})，待生成 {len(pending)} 张', flush=True)
    if not pending:
        return
    model = load_model(config, args.checkpoint, device)
    use_style = 'style' in model.modalities
    run_up = model.unet_up is not None
    resampling = getattr(Image, 'Resampling', Image)
    for sample in pending:
        # 固定到样本的种子使单卡、双卡和中断重跑保持一致。
        seed = args.seed + int(sample['sample_id'])
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        kwargs = dict(prompt=sample['prompt'], batch_size=1, run_up=run_up,
                      num_inference_steps=args.num_inference_steps,
                      up_inference_steps=args.up_inference_steps, show_progress=False)
        with torch.no_grad():
            if use_style:
                result = model.infer_mm(
                    style=load_style(sample['style'], device),
                    modality_order_cfg=dict(style=args.style_guidance, txt=args.text_guidance),
                    **kwargs)
            else:
                result = model.infer(guidance_scale=args.text_guidance,
                                     modalities=['txt'], **kwargs)

        def to_image(tensor):
            pixels = tensor[0].detach().float().cpu().clamp(-1, 1)
            pixels = ((pixels + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
            return Image.fromarray(pixels)

        generated, base = to_image(result['samples']), to_image(result['low_res_samples'])
        with Image.open(sample['target']) as image:
            real = image.convert('RGB')
        with Image.open(sample['style']) as image:
            style = image.convert('RGB')
        panels = [('GT', real), ('Base 64', base)]
        if use_style:
            panels.insert(0, ('Texture', style))
        if run_up:
            panels.append(('Generated 256', generated))
        comparison = Image.new('RGB', (256 * len(panels), 280), 'white')
        draw = ImageDraw.Draw(comparison)
        for index, (label, image) in enumerate(panels):
            comparison.paste(image.resize((256, 256), resampling.BICUBIC), (256 * index, 24))
            draw.text((256 * index + 8, 5), label, fill='black')
        for directory, image in zip(directories, (generated, base, real, style, comparison)):
            save_png(image, output / directory / f"{sample['id']}.png")
        print(f"[GPU {rank}] {sample['id']}，seed={seed}，输出={generated.size}", flush=True)
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate(args):
    from tools.sgdiff_metrics import evaluate_metrics

    manifest = read_manifest(args)
    output = Path(args.output_dir)
    # 必须等所有分片结束并且样本齐全后，才计算全体分布的 FID。
    for sample in manifest['samples']:
        for directory in ('generated', 'base_64', 'real', 'texture', 'comparison'):
            path = output / directory / f"{sample['id']}.png"
            if not path.is_file():
                raise FileNotFoundError(f'测试生成未完成：{path}')
    samples = [dict(sample, target=str(output / 'real' / f"{sample['id']}.png"),
                    style=str(output / 'texture' / f"{sample['id']}.png"))
               for sample in manifest['samples']]
    metrics = evaluate_metrics(samples, output / 'generated', output,
                               args.metrics, args.clip_model, args.device,
                               args.metric_batch_size)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f'测试完成：{output}', flush=True)


def main():
    args = parse_args()
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1 and args.mode != 'generate':
        raise ValueError('多进程仅用于 --mode generate；准备清单和指标汇总使用单进程。')
    if args.mode in ('all', 'prepare'):
        prepare(args)
    if args.mode in ('all', 'generate'):
        generate(args)
    if args.mode in ('all', 'evaluate'):
        evaluate(args)


if __name__ == '__main__':
    main()
