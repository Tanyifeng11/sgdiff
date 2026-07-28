_base_ = ['../_base_/gen_default_runtime.py']

work_dir = './work_dirs/sgdiff_bf_glide'
data_root = '/share/home/u2515283058/datasets/BF'

glide_ckpt = (
    'https://download.openmmlab.com/mmediting/glide/'
    'glide_laion-64x64-02afff47.pth')

model = dict(
    type='SGDiff',
    modalities=['txt'],
    cond_prob=dict(txt=0.2),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[127.5],
        std=[127.5],
        data_keys=None,
        non_image_keys=['tokens', 'token_mask']),
    unet=dict(
        type='MM2ImUNet',
        pretrained_cfg=dict(
            ckpt_path=glide_ckpt, prefix='unet', strict=True),
        style_encoder_cfg=None,
        fix_glide=False,
        image_size=64,
        base_channels=192,
        in_channels=3,
        resblocks_per_downsample=3,
        attention_res=(32, 16, 8),
        norm_cfg=dict(type='GN32', num_groups=32),
        dropout=0.1,
        num_classes=0,
        use_fp16=False,
        resblock_updown=True,
        attention_cfg=dict(
            type='MultiHeadAttentionBlock',
            num_heads=1,
            num_head_channels=64,
            use_new_attention_order=False,
            encoder_channels=512),
        use_scale_shift_norm=True,
        text_ctx=128,
        xf_width=512,
        xf_layers=16,
        xf_heads=8,
        xf_final_ln=True,
        xf_padding=True),
    diffusion_scheduler=dict(
        type='EditDDIMScheduler',
        variance_type='learned_range',
        beta_schedule='squaredcos_cap_v2'),
    unet_up=None,
    diffusion_scheduler_up=None,
    perceptual_loss=None,
    use_fp16=False)

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='BFDataset',
        data_root=data_root,
        split='training',
        target_dir='cloth',
        text_dir='text',
        style_dir=None,
        image_size=64,
        text_ctx=128))

optim_wrapper = dict(
    unet=dict(optimizer=dict(type='AdamW', lr=1e-4)))

train_cfg = dict(
    _delete_=True, type='IterBasedTrainLoop', max_iters=235000)
val_cfg = val_evaluator = val_dataloader = None
test_cfg = test_evaluator = test_dataloader = None

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        by_epoch=False,
        max_keep_ckpts=2,
        save_optimizer=True))

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo'))
