import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import torch
from mmengine.optim import OptimWrapper, OptimWrapperDict
from mmengine.registry import init_default_scope
from torch import nn

from mmagic.models.diffusion_schedulers import EditDDIMScheduler
from mmagic.models.editors.glide import SuperResText2ImUNet
from mmagic.models.editors.sgdiff import MM2ImUNet, SGDiff
from mmagic.models.editors.sgdiff.training_utils import (
    GaussianDiffusionTrainingLoss)


class TinyTokenizer:
    n_vocab = 16

    def padded_tokens_and_mask(self, tokens, text_ctx):
        tokens = tokens[:text_ctx]
        length = len(tokens)
        return (tokens + [0] * (text_ctx - length),
                [True] * length + [False] * (text_ctx - length))


def tiny_unet_config():
    return dict(
        image_size=8,
        in_channels=3,
        base_channels=32,
        channels_cfg=[1],
        resblocks_per_downsample=1,
        attention_res=(8, ),
        norm_cfg=dict(type='GN32', num_groups=32),
        dropout=0.2,
        use_scale_shift_norm=True,
        attention_cfg=dict(
            type='MultiHeadAttentionBlock',
            num_heads=1,
            num_head_channels=32,
            encoder_channels=32),
        text_ctx=128,
        xf_width=32,
        xf_layers=1,
        xf_heads=1,
        xf_final_ln=True,
        xf_padding=True,
        tokenizer=TinyTokenizer())


def make_sgdiff(unet, **kwargs):
    return SGDiff(
        unet=unet,
        diffusion_scheduler=EditDDIMScheduler(num_train_timesteps=20),
        cond_prob=0,
        data_preprocessor=dict(
            type='DataPreprocessor',
            mean=[127.5],
            std=[127.5],
            data_keys=None,
            non_image_keys=['tokens', 'token_mask']),
        **kwargs)


class FixedPrediction(nn.Module):

    def __init__(self):
        super().__init__()
        self.epsilon = nn.Parameter(torch.tensor(0.25, dtype=torch.float16))
        self.variance = nn.Parameter(torch.tensor(0.0, dtype=torch.float16))
        self.seen_timesteps = None

    def forward(self, x_t, timesteps, **conditions):
        self.seen_timesteps = timesteps.clone()
        return torch.cat((self.epsilon.expand(x_t.shape),
                          self.variance.expand(x_t.shape)), dim=1)


class TinyStyleEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 32)
        self.dropout = nn.Dropout(0.1)

    def forward(self, image, text_emb):
        style = self.projection(image.mean(dim=(2, 3)))
        return text_emb + self.dropout(style[:, None])


class TrackingOptimWrapper(OptimWrapper):
    in_context = False

    @contextmanager
    def optim_context(self, model):
        with super().optim_context(model):
            self.in_context = True
            try:
                yield
            finally:
                self.in_context = False


class TestSGDiffTraining(unittest.TestCase):

    def setUp(self):
        init_default_scope('mmagic')
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.random_state = torch.random.fork_rng(devices=[])
        self.random_state.__enter__()
        torch.manual_seed(7)

    def tearDown(self):
        self.random_state.__exit__(None, None, None)
        torch.set_num_threads(self.previous_threads)

    def test_runner_initialization_preserves_pretrained_upsampler(self):
        upsampler = SuperResText2ImUNet(
            **tiny_unet_config(), output_cfg=dict(var='FIXED'))
        # 模拟真实预训练参数，覆盖构造器中刻意清零的输出和残差层。
        expected = {
            name: torch.full_like(value, (index + 1) / 1000)
            for index, (name, value) in enumerate(upsampler.state_dict().items())
        }
        # 仅替换文件读取；参数加载、父模块初始化仍走真实生产代码。
        with patch('mmagic.models.editors.glide.glide.load_glide_state_dict',
                   return_value=expected):
            model = make_sgdiff(
                MM2ImUNet(**tiny_unet_config(), fix_glide=False),
                unet_up=upsampler,
                pretrained_cfgs=dict(
                    unet_up=dict(ckpt_path='upsampler.pth', strict=True)))
        assert torch.count_nonzero(model.unet_up.out.conv.weight) > 0

        # Runner 在模型构造、预训练加载之后还会调用一次父模块初始化。
        model.init_weights()

        for name, value in model.unet_up.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        model.train()
        assert not model.unet_up.training
        assert all(not value.requires_grad for value in model.unet_up.parameters())


    def test_forward_diffusion_and_epsilon_reconstruction(self):
        scheduler = EditDDIMScheduler(beta_schedule='squaredcos_cap_v2')
        loss = GaussianDiffusionTrainingLoss(scheduler.betas)
        timesteps = torch.tensor([0, 333, 999])
        x_start = torch.linspace(-0.75, 0.75, 36).reshape(3, 3, 2, 2)
        noise = torch.full_like(x_start, 0.25)
        alpha_bar = torch.tensor(
            np.cumprod(1 - np.asarray(scheduler.betas, dtype=np.float64)),
            dtype=torch.float32)[timesteps, None, None, None]
        expected = alpha_bar.sqrt() * x_start + (1 - alpha_bar).sqrt() * noise

        x_t = loss.q_sample(x_start, timesteps, noise)
        torch.testing.assert_close(x_t, expected, rtol=1e-5, atol=1e-6)
        reconstructed = loss.predict_xstart(x_t, timesteps, noise)
        # 终端时间步的 alpha_bar 极小，FP32 相减会产生少量舍入误差。
        torch.testing.assert_close(reconstructed, x_start, rtol=0, atol=1e-3)


    def test_loss_accepts_fixed_noise_and_keeps_vlb_gradients_off_epsilon(self):
        loss = GaussianDiffusionTrainingLoss(
            EditDDIMScheduler(beta_schedule='squaredcos_cap_v2').betas)
        model = FixedPrediction()
        x_start = torch.linspace(-0.75, 0.75, 24).reshape(2, 3, 2, 2).half()
        noise = torch.linspace(-0.5, 0.5, 24).reshape_as(x_start).half()
        timesteps = torch.tensor([0, 999])
        output = loss(model, x_start, {}, timesteps=timesteps, noise=noise)

        torch.testing.assert_close(model.seen_timesteps, timesteps)
        for value in output.values():
            assert value.dtype == torch.float32
            assert torch.isfinite(value).all()
        torch.testing.assert_close(output['simple_loss'],
                                   (0.25 - noise.float()).square().mean())
        output['vlb_loss'].backward(retain_graph=True)
        assert model.epsilon.grad is None or model.epsilon.grad.item() == 0
        assert model.variance.grad is not None
        assert model.variance.grad.abs().item() > 0
        model.zero_grad(set_to_none=True)
        output['simple_loss'].backward()
        assert model.epsilon.grad.abs().item() > 0
        assert model.variance.grad is None or model.variance.grad.item() == 0


    def test_train_step_updates_style_through_frozen_backbone(self):
        source = MM2ImUNet(**tiny_unet_config(), fix_glide=False)
        # 小模型没有外部预训练文件，显式初始化原实现的 empty 文本参数。
        nn.init.normal_(source.positional_embedding, std=0.01)
        nn.init.normal_(source.padding_embedding, std=0.01)
        with patch(
                'mmagic.models.editors.sgdiff.mm2im_unet.load_glide_state_dict',
                return_value=source.state_dict()):
            unet = MM2ImUNet(
                **tiny_unet_config(),
                pretrained_cfg=dict(ckpt_path='stage1.pth', strict=True),
                fix_glide=True)
        unet.style_encoder = TinyStyleEncoder()
        model = make_sgdiff(unet)
        model.train()
        for name, module in unet.named_modules():
            if name and not name.startswith('style_encoder'):
                assert not module.training
        assert unet.style_encoder.training
        before = {name: value.detach().clone()
                  for name, value in unet.named_parameters()}
        optimizer = TrackingOptimWrapper(
            torch.optim.SGD(unet.parameters(), lr=0.01))

        def check_optim_context(module, args):
            assert optimizer.in_context

        handle = unet.register_forward_pre_hook(check_optim_context)
        batch = dict(inputs=dict(
            img=torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8),
            style=torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8),
            tokens=torch.ones((2, 128), dtype=torch.long),
            token_mask=torch.ones((2, 128), dtype=torch.bool)))
        try:
            logs = model.train_step(batch, OptimWrapperDict(unet=optimizer))
        finally:
            handle.remove()
        assert all(torch.isfinite(torch.as_tensor(value)) for value in logs.values())
        assert not optimizer.in_context
        changed_style = False
        for name, value in unet.named_parameters():
            if name.startswith('style_encoder'):
                changed_style |= not torch.equal(value, before[name])
            else:
                assert not value.requires_grad
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        assert changed_style

    def test_sampling_can_skip_upsampling_and_restores_state(self):
        unet = MM2ImUNet(**tiny_unet_config(), fix_glide=False)
        unet.style_encoder = TinyStyleEncoder()
        up_config = tiny_unet_config()
        up_config['image_size'] = 16
        upsampler = SuperResText2ImUNet(
            **up_config, output_cfg=dict(var='FIXED'))
        for module in (unet, upsampler):
            nn.init.normal_(module.positional_embedding, std=0.01)
            nn.init.normal_(module.padding_embedding, std=0.01)
        model = make_sgdiff(unet, unet_up=upsampler)
        model.diffusion_scheduler.set_timesteps(5)
        model.diffusion_scheduler_up.set_timesteps(4)
        for method in (model.infer, model.infer_mm):
            for run_up in (False, True):
                with self.subTest(method=method.__name__, run_up=run_up):
                    model.train(not run_up)
                    unet.out.gn.eval()
                    modes = [module.training for module in model.modules()]
                    schedules = [
                        (scheduler.num_inference_steps,
                         scheduler.timesteps.copy())
                        for scheduler in (model.diffusion_scheduler,
                                          model.diffusion_scheduler_up)
                    ]
                    kwargs = dict(
                        batch_size=1,
                        num_inference_steps=2,
                        up_inference_steps=2,
                        run_up=run_up,
                        tokens=torch.ones((1, 128), dtype=torch.long),
                        token_mask=torch.ones((1, 128), dtype=torch.bool),
                        style=torch.zeros((1, 3, 8, 8)))
                    if method.__name__ == 'infer_mm':
                        kwargs['modality_order_cfg'] = dict(txt=1., style=1.)
                    else:
                        kwargs['guidance_scale'] = 1.
                    output = method(**kwargs)
                    resolution = 16 if run_up else 8
                    assert output['samples'].shape == (1, 3, resolution,
                                                       resolution)
                    assert output['low_res_samples'].shape == (1, 3, 8, 8)
                    assert all(torch.isfinite(value).all()
                               for value in output.values())
                    if not run_up:
                        torch.testing.assert_close(output['samples'],
                                                   output['low_res_samples'])
                    assert [module.training
                            for module in model.modules()] == modes
                    for scheduler, (steps, timesteps) in zip(
                            (model.diffusion_scheduler,
                             model.diffusion_scheduler_up), schedules):
                        assert scheduler.num_inference_steps == steps
                        np.testing.assert_array_equal(scheduler.timesteps,
                                                      timesteps)

