"""Compare original and fine-tuned SGDiff checkpoints on BF validation."""

import argparse
import csv
import gc
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from mmagic.apis.inferencers.inference_functions import init_model


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate BF validation comparisons for two SGDiff models.')
    parser.add_argument(
        '--data-root',
        default='/share/home/u2515283058/datasets/BF',
        help='BF dataset root containing validation/.')
    parser.add_argument(
        '--original-config',
        default='configs/sgdiff/sgdiff-ddim-sg_fashion-64x64.py')
    parser.add_argument('--original-ckpt', default='checkpoint/sgdiff.pth')
    parser.add_argument(
        '--finetuned-config',
        default='configs/sgdiff/sgdiff-bf-style-64x64.py')
    parser.add_argument(
        '--finetuned-ckpt',
        default='work_dirs/sgdiff_bf_style/iter_50000.pth')
    parser.add_argument(
        '--output-dir', default='results/bf_validation_compare')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--num-inference-steps', type=int, default=100)
    parser.add_argument('--up-inference-steps', type=int, default=35)
    parser.add_argument(
        '--text-guidance', type=float, default=1.5,
        help='Classifier-free guidance scale for text.')
    parser.add_argument(
        '--style-guidance', type=float, default=2.0,
        help='Classifier-free guidance scale for style image.')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Only test the first N samples; omit to test all validation data.')
    return parser.parse_args()


def build_index(directory: Path, suffixes) -> Dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f'Missing validation directory: {directory}')
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    }


def collect_samples(data_root: Path, max_samples=None) -> List[dict]:
    data_root = Path(data_root)
    validation_root = data_root / 'validation'
    texts = build_index(validation_root / 'text', {'.txt'})
    styles = build_index(validation_root / 'texture', IMAGE_SUFFIXES)
    targets = build_index(validation_root / 'gt', IMAGE_SUFFIXES)

    sample_ids = sorted(set(texts) & set(styles) & set(targets))
    if max_samples is not None:
        sample_ids = sample_ids[:max_samples]
    if not sample_ids:
        raise RuntimeError(
            f'No complete text/texture/gt pairs in {validation_root}')

    return [{
        'id': sample_id,
        'text': texts[sample_id],
        'style': styles[sample_id],
        'target': targets[sample_id],
    } for sample_id in sample_ids]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_style(path: Path, device: str) -> torch.Tensor:
    resampling = getattr(Image, 'Resampling', Image)
    with Image.open(path) as image:
        image = image.convert('RGB').resize((256, 256), resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu()[0].clamp(-1, 1)
    array = ((tensor + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode='RGB')


def generate(model_name: str, config: str, checkpoint: str, samples: List[dict],
             output_dir: Path, args):
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'{model_name} checkpoint not found: {checkpoint}')

    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f'Loading {model_name} checkpoint: {checkpoint}')
    model = init_model(config, checkpoint, device=args.device)
    guidance = {'txt': args.text_guidance, 'style': args.style_guidance}

    for index, sample in enumerate(samples):
        output_path = model_dir / f"{sample['id']}.png"
        set_seed(args.seed + index)
        style = load_style(sample['style'], args.device)
        prompt = sample['text'].read_text(
            encoding='utf-8', errors='replace').strip()

        with torch.no_grad():
            result = model.infer_mm(
                style=style,
                prompt=prompt,
                modality_order_cfg=guidance,
                num_inference_steps=args.num_inference_steps,
                up_inference_steps=args.up_inference_steps,
                show_progress=False)
        tensor_to_image(result['samples']).save(output_path)
        print(f'[{model_name}] {index + 1}/{len(samples)} {sample["id"]}')

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_comparisons(samples: List[dict], output_dir: Path):
    comparison_dir = output_dir / 'comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / 'manifest.csv'
    resampling = getattr(Image, 'Resampling', Image)

    with manifest_path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=['id', 'prompt', 'style', 'target', 'original',
                        'finetuned', 'comparison'])
        writer.writeheader()
        for sample in samples:
            original_path = output_dir / 'original' / f"{sample['id']}.png"
            finetuned_path = output_dir / 'finetuned' / f"{sample['id']}.png"
            with Image.open(sample['style']) as style, \
                    Image.open(sample['target']) as target, \
                    Image.open(original_path) as original, \
                    Image.open(finetuned_path) as finetuned:
                panels = [
                    image.convert('RGB').resize((256, 256), resampling.BICUBIC)
                    for image in (style, target, original, finetuned)
                ]
            comparison = Image.new('RGB', (256 * len(panels), 256))
            for panel_index, panel in enumerate(panels):
                comparison.paste(panel, (256 * panel_index, 0))
            comparison_path = comparison_dir / f"{sample['id']}.jpg"
            comparison.save(comparison_path, quality=95)

            writer.writerow({
                'id': sample['id'],
                'prompt': sample['text'].read_text(
                    encoding='utf-8', errors='replace').strip(),
                'style': sample['style'],
                'target': sample['target'],
                'original': original_path,
                'finetuned': finetuned_path,
                'comparison': comparison_path,
            })


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for SGDiff inference.')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = collect_samples(Path(args.data_root), args.max_samples)
    print(f'Testing {len(samples)} BF validation samples.')

    # Load one model at a time so the two GLIDE pipelines never share GPU memory.
    generate('original', args.original_config, args.original_ckpt, samples,
             output_dir, args)
    generate('finetuned', args.finetuned_config, args.finetuned_ckpt, samples,
             output_dir, args)
    build_comparisons(samples, output_dir)
    print('Done. Each comparison image is: style | GT | original | finetuned')


if __name__ == '__main__':
    main()
