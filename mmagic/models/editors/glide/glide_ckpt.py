# Copyright (c) OpenMMLab. All rights reserved.
"""Helpers for loading the original GLIDE checkpoints released by OpenAI.

The weights published by OpenAI (``base.pt``, ``upsample.pt``) and by the LAION
community (``laionide-v3-base.pt``) are flat ``state_dict`` files that follow
the module names of https://github.com/openai/glide-text2im.  MMagic
re-implements the very same network in
:class:`~mmagic.models.editors.ddpm.DenoisingUnet`, so the tensors correspond
one to one, but a few modules were given different names.  The helpers below
rewrite the original names into the MMagic ones, which makes a raw ``*.pt``
file usable anywhere a converted MMagic checkpoint is expected -- useful now
that the MMagic release of the 64->256 upsampler is no longer downloadable.

The naming differences are:

=========================  =========================================
original GLIDE             MMagic
=========================  =========================================
``time_embed.{0,2}``       ``time_embedding.blocks.{0,2}``
``input_blocks``           ``in_blocks``
``middle_block``           ``mid_blocks``
``output_blocks``          ``out_blocks``
``out.0`` / ``out.2``      ``out.gn`` / ``out.conv``
``*.in_layers.{0,2}``      ``*.conv_1.{0,2}``
``*.emb_layers.1``         ``*.norm_with_embedding.embedding_layer.1``
``*.out_layers.0``         ``*.norm_with_embedding.norm``
``*.out_layers.3``         ``*.conv_2.2``
``*.skip_connection``      ``*.shortcut``
``*.op``                   ``*.downsample``
=========================  =========================================

The text encoder (``transformer``, ``token_embedding``,
``positional_embedding``, ``padding_embedding``, ``final_ln``,
``transformer_proj``) and the attention blocks (``norm``, ``qkv``,
``encoder_kv``, ``proj_out``) already share the same names and are copied
unchanged.
"""

from collections import OrderedDict
from typing import Dict

import mmengine
from mmengine.runner.checkpoint import _load_checkpoint

# renames applied to the beginning of a key
_PREFIX_RENAMES = (
    ('time_embed.', 'time_embedding.blocks.'),
    ('input_blocks.', 'in_blocks.'),
    ('middle_block.', 'mid_blocks.'),
    ('output_blocks.', 'out_blocks.'),
    # ``ConvModule(order=('norm', 'act', 'conv'))`` names its children after
    # the layer type instead of using positional indices.
    ('out.0.', 'out.gn.'),
    ('out.2.', 'out.conv.'),
)

# renames applied inside a residual / sampling block
_INNER_RENAMES = (
    ('.in_layers.0.', '.conv_1.0.'),
    ('.in_layers.2.', '.conv_1.2.'),
    ('.emb_layers.1.', '.norm_with_embedding.embedding_layer.1.'),
    ('.out_layers.0.', '.norm_with_embedding.norm.'),
    ('.out_layers.3.', '.conv_2.2.'),
    ('.skip_connection.', '.shortcut.'),
    # only present when ``resblock_updown=False``
    ('.op.', '.downsample.'),
)

# keys that only exist in one of the two layouts, used to tell them apart
_OPENAI_MARKERS = ('input_blocks.', 'middle_block.', 'output_blocks.',
                   'time_embed.')
_MMAGIC_MARKERS = ('in_blocks.', 'mid_blocks.', 'out_blocks.',
                   'time_embedding.')


def _convert_key(key: str) -> str:
    """Translate a single original GLIDE parameter name to its MMagic name."""
    for src, dst in _PREFIX_RENAMES:
        if key.startswith(src):
            key = dst + key[len(src):]
            break
    for src, dst in _INNER_RENAMES:
        if src in key:
            key = key.replace(src, dst)
            break
    return key


def unwrap_state_dict(checkpoint: dict) -> dict:
    """Return the parameter dict of ``checkpoint``.

    Accepts both a bare ``state_dict`` and the usual wrappers written by
    training frameworks.
    """
    if not isinstance(checkpoint, dict):
        return checkpoint
    for wrapper in ('state_dict', 'model', 'module'):
        if isinstance(checkpoint.get(wrapper), dict):
            return checkpoint[wrapper]
    return checkpoint


def is_openai_glide_state_dict(state_dict: Dict) -> bool:
    """Check whether ``state_dict`` uses the original GLIDE module names."""
    has_openai = any(
        key.startswith(_OPENAI_MARKERS) for key in state_dict)
    has_mmagic = any(key.startswith(_MMAGIC_MARKERS) for key in state_dict)
    return has_openai and not has_mmagic


def convert_openai_glide_state_dict(state_dict: Dict) -> 'OrderedDict':
    """Rename an original GLIDE ``state_dict`` to the MMagic layout.

    Args:
        state_dict (dict): Parameters using the original GLIDE names.

    Returns:
        OrderedDict: The same tensors under their MMagic names.
    """
    converted = OrderedDict()
    for key, value in state_dict.items():
        new_key = _convert_key(key)
        if new_key in converted:
            raise RuntimeError(
                'Conversion of the GLIDE checkpoint is ambiguous: both '
                f'\'{key}\' and another parameter map to \'{new_key}\'.')
        converted[new_key] = value
    return converted


def load_glide_state_dict(ckpt_path: str,
                          prefix: str = '',
                          map_location: str = 'cpu') -> Dict:
    """Load a checkpoint for a GLIDE UNet, accepting both layouts.

    An original GLIDE ``*.pt`` file is renamed to the MMagic layout and
    ``prefix`` is ignored, because such a file stores a single UNet at the top
    level.  Any other checkpoint keeps the usual MMagic behaviour of selecting
    the ``prefix`` sub-module.

    Args:
        ckpt_path (str): Path or URL of the checkpoint.
        prefix (str): Sub-module to select from an MMagic checkpoint, e.g.
            ``'unet'`` or ``'unet_up'``. Defaults to ``''``.
        map_location (str): Same meaning as in :func:`torch.load`. Defaults to
            ``'cpu'``.

    Returns:
        dict: A ``state_dict`` ready for ``load_state_dict``.
    """
    checkpoint = _load_checkpoint(ckpt_path, map_location=map_location)
    state_dict = unwrap_state_dict(checkpoint)

    if is_openai_glide_state_dict(state_dict):
        mmengine.print_log(
            f'\'{ckpt_path}\' is an original GLIDE checkpoint; converting its '
            'parameter names to the MMagic layout.', 'current')
        return convert_openai_glide_state_dict(state_dict)

    if prefix:
        if not prefix.endswith('.'):
            prefix += '.'
        state_dict = OrderedDict({
            key[len(prefix):]: value
            for key, value in state_dict.items() if key.startswith(prefix)
        })
        if not state_dict:
            raise RuntimeError(
                f'\'{prefix}\' is not in the checkpoint \'{ckpt_path}\'.')
    return state_dict
