"""Diagnose SGDiff stage-1 sampling failure.

Three checks in one run:
  A. Does the UNet behave identically when the timestep arrives as a torch
     tensor (training path) vs a numpy int64 (sampling path)?
  B. Does the trained checkpoint reproduce its logged eps-MSE (~0.015)?
  C. Do raw GLIDE weights and the trained checkpoint produce a real image?

Run from the repo root:  python tools/probe_sgdiff.py
"""
import numpy as np
import torch
from PIL import Image

from mmagic.apis.inferencers.inference_functions import init_model
from mmagic.datasets.bf_dataset import BFDataset

CONFIG = 'configs/sgdiff/sgdiff-bf-glide-64x64.py'
CKPT = 'work_dirs/sgdiff_bf_glide/iter_235000.pth'
DATA_ROOT = '/share/home/u2515283058/datasets/BF'
DEVICE = 'cuda:0'

torch.manual_seed(0)


def save(tensor, path):
    t = tensor.detach().float().cpu()[0].clamp(-1, 1)
    array = ((t + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(array, mode='RGB').save(path)
    return t


def stats(name, t):
    print(f'{name:24s} std={t.std():.4f}  mean={t.mean():+.4f}  '
          f'per-channel mean={[round(v, 3) for v in t.mean(dim=(1, 2)).tolist()]}')


# ---------------------------------------------------------------- trained model
model = init_model(CONFIG, CKPT, device=DEVICE)
unet = model.unet
unet.eval()                    # dropout=0.1 would otherwise pollute the A/B diff
loss = model.training_loss

# a real training sample, exactly as train_step would see it
dataset = BFDataset(data_root=DATA_ROOT, split='training', target_dir='cloth',
                    text_dir='text', style_dir=None, image_size=64, text_ctx=128)
sample = dataset[0]['inputs']
x0 = (sample['img'].float().unsqueeze(0).to(DEVICE) - 127.5) / 127.5
tokens = sample['tokens'].unsqueeze(0).to(DEVICE)
mask = sample['token_mask'].unsqueeze(0).to(DEVICE)

print('\n=== A/B: timestep as tensor (train) vs numpy int64 (sampling) ===')
print(f'{"t":>5} {"mse_tensor_t":>13} {"mse_numpy_t":>12} {"max|diff|":>11}')
for t_int in [0, 10, 100, 500, 990]:
    t_tensor = torch.tensor([t_int], device=DEVICE, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = loss.q_sample(x0, t_tensor, noise)
    with torch.no_grad():
        out_tensor = unet(x_t, t_tensor, tokens=tokens, token_mask=mask)
        out_numpy = unet(x_t, np.int64(t_int), tokens=tokens, token_mask=mask)
    a, b = out_tensor[:, :3], out_numpy[:, :3]
    print(f'{t_int:5d} {(a - noise).pow(2).mean():13.4f} '
          f'{(b - noise).pow(2).mean():12.4f} '
          f'{(a - b).abs().max():11.6f}')

model.diffusion_scheduler.set_timesteps(100)
ts = model.diffusion_scheduler.timesteps
print(f'\nsampling timesteps: dtype={ts.dtype} first={ts[:3]} last={ts[-3:]}')

print('\n=== sampling with the TRAINED checkpoint ===')
with torch.no_grad():
    out = model.infer(prompt='a red dress', batch_size=1,
                      guidance_scale=3.0, num_inference_steps=100)
stats('trained', save(out['samples'], 'probe_trained.png'))

del model
torch.cuda.empty_cache()

# ------------------------------------------------------------ raw GLIDE weights
print('\n=== sampling with RAW laionide weights (no checkpoint) ===')
raw = init_model(CONFIG, None, device=DEVICE)
with torch.no_grad():
    out = raw.infer(prompt='a red dress', batch_size=1,
                    guidance_scale=3.0, num_inference_steps=100)
stats('raw laionide', save(out['samples'], 'probe_raw_glide.png'))

print('\nWrote probe_trained.png and probe_raw_glide.png')
