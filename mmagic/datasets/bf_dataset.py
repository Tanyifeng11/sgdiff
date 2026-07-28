import importlib.util
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from mmagic.registry import DATASETS


@DATASETS.register_module()
class BFDataset(Dataset):
    """Paired BF dataset used by the two SGDiff training stages."""

    def __init__(self,
                 data_root: str,
                 split: str = 'training',
                 target_dir: Optional[str] = None,
                 text_dir: str = 'text',
                 style_dir: Optional[str] = 'texture',
                 image_size: int = 64,
                 style_size: int = 256,
                 text_ctx: int = 128,
                 max_samples: Optional[int] = None):
        self.data_root = Path(data_root)
        self.split = split
        self.split_root = self.data_root / split
        self.target_dir = target_dir or (
            'cloth' if split == 'training' else 'gt')
        self.text_dir = text_dir
        self.style_dir = style_dir
        self.image_size = image_size
        self.style_size = style_size
        self.text_ctx = text_ctx
        self._tokenizer = None

        required_dirs = [
            self.split_root / self.target_dir,
            self.split_root / self.text_dir,
        ]
        if self.style_dir is not None:
            required_dirs.append(self.split_root / self.style_dir)
        for directory in required_dirs:
            if not directory.is_dir():
                raise FileNotFoundError(
                    f'BF modality directory does not exist: {directory}')

        stem_sets = []
        for directory in required_dirs:
            suffix = '.txt' if directory.name == self.text_dir else '.jpg'
            stem_sets.append({
                path.stem
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() == suffix
            })
        self.sample_ids = sorted(set.intersection(*stem_sets))
        if max_samples is not None:
            self.sample_ids = self.sample_ids[:max_samples]
        if not self.sample_ids:
            raise RuntimeError(f'No complete BF pairs found in {self.split_root}')

    def __len__(self):
        return len(self.sample_ids)

    def _get_tokenizer(self):
        if self._tokenizer is None:
            tokenizer_path = (
                Path(__file__).parents[1] / 'models' / 'editors' / 'glide' /
                'glide_tokenizer' / 'bpe.py')
            spec = importlib.util.spec_from_file_location(
                '_sgdiff_glide_bpe', tokenizer_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._tokenizer = module.get_encoder()
        return self._tokenizer

    @staticmethod
    def _load_image(path: Path, size: int) -> torch.Tensor:
        resampling = getattr(Image, 'Resampling', Image)
        with Image.open(path) as image:
            image = image.convert('RGB')
            image = image.resize((size, size), resampling.BICUBIC)
            array = np.asarray(image, dtype=np.uint8).copy()
        return torch.from_numpy(array).permute(2, 0, 1)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        prompt_path = self.split_root / self.text_dir / f'{sample_id}.txt'
        prompt = prompt_path.read_text(
            encoding='utf-8', errors='replace').strip()

        tokenizer = self._get_tokenizer()
        tokens = tokenizer.encode(prompt)
        tokens, token_mask = tokenizer.padded_tokens_and_mask(
            tokens, self.text_ctx)

        inputs = {
            'img':
            self._load_image(
                self.split_root / self.target_dir / f'{sample_id}.jpg',
                self.image_size),
            'tokens':
            torch.tensor(tokens, dtype=torch.long),
            'token_mask':
            torch.tensor(token_mask, dtype=torch.bool),
        }
        if self.style_dir is not None:
            inputs['style'] = self._load_image(
                self.split_root / self.style_dir / f'{sample_id}.jpg',
                self.style_size)
        return {'inputs': inputs}
