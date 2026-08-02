# Copyright (c) OpenMMLab. All rights reserved.
"""Convert / verify an original GLIDE checkpoint against an MMagic UNet.

The GLIDE weights released by OpenAI (``upsample.pt``) and by LAION
(``laionide-v3-base.pt``) store the parameters under the module names of
https://github.com/openai/glide-text2im, while MMagic uses its own names. The
training code translates the names on the fly, so no conversion is required to
start a run. This script is useful to

* check that a downloaded ``*.pt`` file really matches the UNet described by a
  config, before spending time on a training run, and
* write a converted MMagic-style checkpoint, if a plain checkpoint is
  preferred.

Examples:
    Verify the 64->256 upsampler used by the stage-2 config::

        python tools/model_converters/glide_openai_to_mmagic.py \
            checkpoint/upsample.pt \
            --config configs/sgdiff/sgdiff-bf-style-64x64.py --module unet_up

    Verify the 64x64 base model used by the stage-1 config::

        python tools/model_converters/glide_openai_to_mmagic.py \
            checkpoint/laionide-v3-base.pt \
            --config configs/sgdiff/sgdiff-bf-glide-64x64.py --module unet

    Write a converted checkpoint that can be loaded with ``prefix='unet_up'``::

        python tools/model_converters/glide_openai_to_mmagic.py \
            checkpoint/upsample.pt --out checkpoint/glide_up-mmagic.pth \
            --prefix unet_up
"""

import argparse
from collections import OrderedDict
from copy import deepcopy

import torch
from mmengine.config import Config
from mmengine.runner.checkpoint import _load_checkpoint

from mmagic.models.editors.glide.glide_ckpt import (
    convert_openai_glide_state_dict, is_openai_glide_state_dict,
    unwrap_state_dict)
from mmagic.registry import MODELS
from mmagic.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert or verify an original GLIDE checkpoint.')
    parser.add_argument('ckpt', help='path of the original GLIDE *.pt file')
    parser.add_argument(
        '--config', default=None, help='config used to verify the checkpoint')
    parser.add_argument(
        '--module',
        default='unet_up',
        help='name of the UNet inside `model` to verify, '
        '`unet` or `unet_up`')
    parser.add_argument(
        '--out', default=None, help='path to save the converted checkpoint')
    parser.add_argument(
        '--prefix',
        default=None,
        help='prefix to store the converted parameters under, e.g. `unet_up`. '
        'Defaults to the value of --module when --out is given.')
    return parser.parse_args()


def build_unet(config_path: str, module: str) -> torch.nn.Module:
    """Build ``model.<module>`` from ``config_path`` without loading weights.

    The pretrained and the extra encoder configs are dropped so that the result
    matches the module as it looks when the pretrained GLIDE weights are
    loaded, which happens before the style / edge encoders are attached.
    """
    cfg = Config.fromfile(config_path)
    if module not in cfg.model or cfg.model[module] is None:
        raise KeyError(f'`model.{module}` is not defined in {config_path}.')

    unet_cfg = deepcopy(cfg.model[module])
    for key in ('pretrained_cfg', 'style_encoder_cfg', 'edge_encoder_cfg'):
        if key in unet_cfg:
            unet_cfg[key] = None
    return MODELS.build(unet_cfg)


def verify(state_dict: OrderedDict, unet: torch.nn.Module) -> bool:
    """Compare ``state_dict`` with the parameters of ``unet``."""
    expected = unet.state_dict()
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    mismatched = sorted(
        key for key in set(state_dict) & set(expected)
        if tuple(state_dict[key].shape) != tuple(expected[key].shape))

    print(f'checkpoint parameters : {len(state_dict)}')
    print(f'model parameters      : {len(expected)}')

    def report(title, keys):
        print(f'{title:22}: {len(keys)}')
        for key in keys[:10]:
            print(f'    {key}')
        if len(keys) > 10:
            print(f'    ... and {len(keys) - 10} more')

    report('missing in checkpoint', missing)
    report('unexpected keys', unexpected)
    report('shape mismatches', mismatched)

    ok = not (missing or unexpected or mismatched)
    print('RESULT: ' + ('the checkpoint matches the model.'
                        if ok else 'the checkpoint does NOT match the model.'))
    return ok


def main():
    args = parse_args()
    register_all_modules()

    raw = unwrap_state_dict(_load_checkpoint(args.ckpt, map_location='cpu'))
    if is_openai_glide_state_dict(raw):
        state_dict = convert_openai_glide_state_dict(raw)
        print(f'Converted {len(state_dict)} parameters from the original '
              'GLIDE layout.')
    else:
        state_dict = OrderedDict(raw)
        print('The checkpoint already uses the MMagic layout; '
              'nothing to convert.')

    ok = True
    if args.config is not None:
        ok = verify(state_dict, build_unet(args.config, args.module))

    if args.out is not None:
        prefix = args.prefix if args.prefix is not None else args.module
        if prefix:
            state_dict = OrderedDict(
                (f'{prefix}.{key}', value)
                for key, value in state_dict.items())
        torch.save({'state_dict': state_dict}, args.out)
        print(f'Saved the converted checkpoint to {args.out}'
              + (f' under the prefix `{prefix}`.' if prefix else '.'))

    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
