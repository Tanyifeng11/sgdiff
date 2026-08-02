_base_ = ['./sgdiff-bf-glide-64x64.py']

work_dir = './work_dirs/sgdiff_bf_style'
stage1_ckpt = './work_dirs/sgdiff_bf_glide/iter_235000.pth'
# Original GLIDE 64->256 upsampler released by OpenAI. It replaces the MMagic
# mirror `.../GLIDE/weight/glide_laion-64-256`, which is no longer available.
# The parameter names are translated to the MMagic layout while loading, so no
# offline conversion step is needed.
glide_up_ckpt = '/share/home/u2515283058/sgdiff/checkpoint/upsample.pt'

style_encoder_cfg = dict(
    _delete_=True,
    type='ClipAttnEmbedding',
    name='ViT-B/32',
    cross_attn_cfg=dict(
        type='MultiHeadAttentionBlock',
        in_channels=512,
        num_heads=4,
        encoder_channels=512),
    residual_cfg=dict(learned_length=128))

unet_up_cfg = dict(
    _delete_=True,
    type='SuperResText2ImUNet',
    image_size=256,
    base_channels=192,
    in_channels=3,
    output_cfg=dict(var='FIXED'),
    resblocks_per_downsample=2,
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
    xf_padding=True)

model = dict(
    modalities=['txt', 'style'],
    cond_prob=dict(txt=0.2, style=0.2),
    unet=dict(
        pretrained_cfg=dict(
            ckpt_path=stage1_ckpt, prefix='unet', strict=True),
        style_encoder_cfg=style_encoder_cfg,
        fix_glide=True),
    unet_up=unet_up_cfg,
    diffusion_scheduler_up=dict(
        _delete_=True,
        type='EditDDIMScheduler',
        variance_type='learned_range',
        beta_schedule='linear'),
    pretrained_cfgs=dict(
        unet_up=dict(
            ckpt_path=glide_up_ckpt, prefix='unet_up', strict=True)),
    perceptual_loss=dict(
        _delete_=True,
        type='PerceptualLoss',
        layer_weights={
            '3': 1.0,
            '8': 1.0,
            '13': 1.0,
            '22': 1.0,
            '31': 1.0
        },
        perceptual_weight=0.001,
        style_weight=0.0,
        norm_img=True,
        criterion='mse'),
    val_cfg=dict(
        num_inference_steps=100,
        up_inference_steps=35,
        modality_order_cfg=dict(style=1.2, txt=1.0)))

train_dataloader = dict(
    batch_size=16,
    dataset=dict(style_dir='texture', style_size=256))

optim_wrapper = dict(
    unet=dict(optimizer=dict(type='AdamW', lr=1e-5)))

train_cfg = dict(
    _delete_=True, type='IterBasedTrainLoop', max_iters=50000)
