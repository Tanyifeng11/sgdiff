import math

import numpy as np
import torch


def _extract(values, timesteps, shape):
    values = torch.as_tensor(
        values, device=timesteps.device, dtype=torch.float32)[timesteps]
    while values.ndim < len(shape):
        values = values[..., None]
    return values.expand(shape)


def _mean_flat(tensor):
    return tensor.mean(dim=tuple(range(1, tensor.ndim)))


def _normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (
        -1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) +
        (mean1 - mean2).pow(2) * torch.exp(-logvar2))


def _approx_standard_normal_cdf(tensor):
    return 0.5 * (
        1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) *
            (tensor + 0.044715 * tensor.pow(3))))


def _discretized_gaussian_log_likelihood(x, means, log_scales):
    centered_x = x - means
    inv_std = torch.exp(-log_scales)
    plus_in = inv_std * (centered_x + 1.0 / 255.0)
    min_in = inv_std * (centered_x - 1.0 / 255.0)
    cdf_plus = _approx_standard_normal_cdf(plus_in)
    cdf_min = _approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    log_cdf_delta = torch.log((cdf_plus - cdf_min).clamp(min=1e-12))
    return torch.where(
        x < -0.999, log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, log_cdf_delta))


class GaussianDiffusionTrainingLoss:
    """Training loss for an epsilon model with learned-range variance."""

    def __init__(self, betas):
        self.betas = np.asarray(betas, dtype=np.float64)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(
            1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(
            1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(
            1.0 / self.alphas_cumprod - 1.0)

        self.posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod))
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1],
                      self.posterior_variance[1:]))
        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) /
            (1.0 - self.alphas_cumprod))

    @property
    def num_timesteps(self):
        return len(self.betas)

    def q_sample(self, x_start, timesteps, noise):
        return (
            _extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) *
            x_start +
            _extract(self.sqrt_one_minus_alphas_cumprod, timesteps,
                     x_start.shape) * noise)

    def predict_xstart(self, x_t, timesteps, epsilon):
        return (
            _extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) *
            x_t -
            _extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape) *
            epsilon)

    def _posterior_mean(self, x_start, x_t, timesteps):
        return (
            _extract(self.posterior_mean_coef1, timesteps, x_t.shape) *
            x_start +
            _extract(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t)

    def _vlb(self, x_start, x_t, timesteps, epsilon, variance_values):
        true_mean = self._posterior_mean(x_start, x_t, timesteps)
        true_log_variance = _extract(
            self.posterior_log_variance_clipped, timesteps, x_t.shape)

        pred_xstart = self.predict_xstart(x_t, timesteps, epsilon.detach())
        model_mean = self._posterior_mean(pred_xstart, x_t, timesteps)
        min_log = true_log_variance
        max_log = _extract(np.log(self.betas), timesteps, x_t.shape)
        variance_fraction = (variance_values + 1.0) / 2.0
        model_log_variance = (
            variance_fraction * max_log +
            (1.0 - variance_fraction) * min_log)

        kl = _mean_flat(
            _normal_kl(true_mean, true_log_variance, model_mean,
                       model_log_variance)) / math.log(2.0)
        decoder_nll = -_mean_flat(
            _discretized_gaussian_log_likelihood(
                x_start,
                means=model_mean,
                log_scales=0.5 * model_log_variance)) / math.log(2.0)
        return torch.where(timesteps == 0, decoder_nll, kl).mean()

    def __call__(self, model, x_start, conditions):
        batch_size = x_start.shape[0]
        timesteps = torch.randint(
            0,
            self.num_timesteps, (batch_size, ),
            device=x_start.device,
            dtype=torch.long)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, timesteps, noise)
        model_output = model(x_t, timesteps, **conditions)

        channels = x_start.shape[1]
        if model_output.shape[1] != channels * 2:
            raise ValueError(
                'SGDiff training requires epsilon and learned variance '
                f'outputs, but received {model_output.shape[1]} channels.')
        epsilon, variance_values = torch.split(
            model_output, channels, dim=1)
        simple_loss = (epsilon - noise).pow(2).mean()
        vlb_loss = self._vlb(x_start, x_t, timesteps, epsilon,
                             variance_values)
        pred_xstart = self.predict_xstart(x_t, timesteps, epsilon)
        return {
            'simple_loss': simple_loss,
            'vlb_loss': vlb_loss,
            'pred_xstart': pred_xstart,
        }
