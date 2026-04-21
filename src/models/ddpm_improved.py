"""
Improved DDPM for financial time series -- ablation-ready.

All improvements are toggled via constructor flags so a single class
can instantiate the baseline or any combination of improvements:

  Phase 1:
  1. use_dit         -- Transformer denoiser instead of UNet
  2. use_vpred       -- v-prediction instead of epsilon-prediction
  3. use_self_cond   -- self-conditioning (feed x0 estimate back)
  4. use_sigmoid_schedule -- sigmoid noise schedule
  5. use_cross_attn  -- cross-attention conditioning instead of additive

  Phase 2:
  6. use_temporal_attn   -- multi-head self-attention after each ResBlock
  7. use_hetero_noise    -- heteroskedastic (volatility-scaled) noise injection
  8. use_aux_sf_loss     -- auxiliary stylized-fact losses (kurtosis + ACF)

  Phase 5:
  9.  use_acf_guidance   -- inference-time ACF guidance during DDIM sampling
  10. use_wavelet        -- wavelet-domain diffusion (train on wavelet coefficients)
  11. use_student_t_noise -- Student-t noise in forward process instead of Gaussian
"""

from __future__ import annotations

import os
import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.base_model import BaseGenerativeModel


# ---------------------------------------------------------------------------
# Sinusoidal embeddings (shared)
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        emb = math.log(10_000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ---------------------------------------------------------------------------
# Noise schedules
# ---------------------------------------------------------------------------

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.linspace(0, T, T + 1)
    alpha_bar = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(0.0001, 0.9999)


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


def sigmoid_beta_schedule(T: int, start: float = -3.0, end: float = 3.0, tau: float = 1.0) -> torch.Tensor:
    steps = torch.linspace(0, T, T + 1)
    v_start = torch.sigmoid(torch.tensor(start / tau))
    v_end = torch.sigmoid(torch.tensor(end / tau))
    alpha_bar = (-((steps / T) * (end - start) + start) / tau).sigmoid()
    alpha_bar = (alpha_bar - v_end) / (v_start - v_end)
    alpha_bar = alpha_bar.clamp(1e-5, 1.0)
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(0.0001, 0.9999)


# ---------------------------------------------------------------------------
# EMA wrapper
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module):
        with torch.no_grad():
            for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
                s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# 1-D U-Net (baseline denoiser)
# ---------------------------------------------------------------------------

class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.mlp_t = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.block1 = nn.Sequential(
            nn.GroupNorm(min(8, in_ch), in_ch), nn.SiLU(),
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(min(8, out_ch), out_ch), nn.SiLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.mlp_t(t_emb)[:, :, None]
        h = self.block2(h)
        return h + self.skip(x)


class TemporalAttention1D(nn.Module):
    """Self-attention over the temporal dimension for 1D sequences.

    Expects input shape (B, C, L), transposes to (B, L, C) for attention,
    then transposes back. Gives every timestep a global receptive field.
    """

    def __init__(self, channels: int, n_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=n_heads, batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        h = h.permute(0, 2, 1)  # (B, L, C)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1)  # (B, C, L)
        return residual + h


class UNet1D(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 64,
                 channel_mults: tuple = (1, 2, 4), time_dim: int = 128,
                 cond_dim: int = 0, use_cross_attn: bool = False,
                 out_channels: int | None = None,
                 use_temporal_attn: bool = False):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.out_channels = out_channels or in_channels

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim), nn.Linear(time_dim, time_dim), nn.SiLU(),
        )
        if cond_dim > 0:
            if use_cross_attn:
                self.cond_proj = nn.Linear(cond_dim, time_dim)
                self.cross_attn = nn.MultiheadAttention(
                    embed_dim=time_dim, num_heads=4, batch_first=True,
                )
                self.cross_norm = nn.LayerNorm(time_dim)
            else:
                self.cond_proj = nn.Sequential(
                    nn.Linear(cond_dim, time_dim), nn.SiLU(),
                    nn.Linear(time_dim, time_dim),
                )
                self.cross_attn = None
        else:
            self.cond_proj = None
            self.cross_attn = None

        self.init_conv = nn.Conv1d(in_channels, base_channels, 1)
        self.down_blocks = nn.ModuleList()
        self.down_pools = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        ch = base_channels
        channels = [ch]
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.down_blocks.append(ResBlock1D(ch, out_ch, time_dim))
            self.down_attns.append(
                TemporalAttention1D(out_ch) if use_temporal_attn
                else nn.Identity()
            )
            self.down_pools.append(nn.Conv1d(out_ch, out_ch, 2, stride=2, padding=0))
            ch = out_ch
            channels.append(ch)

        self.mid = ResBlock1D(ch, ch, time_dim)
        self.mid_attn = (
            TemporalAttention1D(ch) if use_temporal_attn else nn.Identity()
        )

        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.up_samples.append(nn.ConvTranspose1d(ch, out_ch, 2, stride=2))
            skip_ch = channels.pop()
            self.up_blocks.append(ResBlock1D(out_ch + skip_ch, out_ch, time_dim))
            self.up_attns.append(
                TemporalAttention1D(out_ch) if use_temporal_attn
                else nn.Identity()
            )
            ch = out_ch

        self.final_conv = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch), nn.SiLU(),
            nn.Conv1d(ch, self.out_channels, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor | None = None):
        t_emb = self.time_mlp(t)
        if cond is not None and self.cond_proj is not None:
            if self.use_cross_attn and self.cross_attn is not None:
                cond_emb = self.cond_proj(cond).unsqueeze(1)  # (B, 1, D)
                t_q = t_emb.unsqueeze(1)  # (B, 1, D)
                attn_out, _ = self.cross_attn(t_q, cond_emb, cond_emb)
                t_emb = self.cross_norm(t_emb + attn_out.squeeze(1))
            else:
                t_emb = t_emb + self.cond_proj(cond)

        x = self.init_conv(x)
        skips = [x]
        for block, attn, pool in zip(self.down_blocks, self.down_attns,
                                     self.down_pools):
            x = block(x, t_emb)
            x = attn(x)
            skips.append(x)
            x = pool(x)
        x = self.mid(x, t_emb)
        x = self.mid_attn(x)
        for up_sample, block, attn in zip(self.up_samples, self.up_blocks,
                                          self.up_attns):
            x = up_sample(x)
            skip = skips.pop()
            min_len = min(x.shape[-1], skip.shape[-1])
            x = torch.cat([x[..., :min_len], skip[..., :min_len]], dim=1)
            x = block(x, t_emb)
            x = attn(x)
        return self.final_conv(x)


# ---------------------------------------------------------------------------
# Diffusion Transformer (DiT) denoiser
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """Transformer block with adaptive layer norm (adaLN) for timestep conditioning."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, d_model), nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(d_model, 6 * d_model),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        h = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(h, h, h)
        x = x + gate_msa.unsqueeze(1) * attn_out

        h = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class DiT1D(nn.Module):
    """Diffusion Transformer for 1-D sequences (replaces UNet1D)."""

    def __init__(self, in_channels: int, seq_len: int, d_model: int = 256,
                 n_heads: int = 8, n_layers: int = 6, time_dim: int = 128,
                 cond_dim: int = 0, use_cross_attn: bool = False,
                 out_channels: int | None = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.d_model = d_model
        self.use_cross_attn = use_cross_attn

        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        if cond_dim > 0:
            if use_cross_attn:
                self.cond_proj = nn.Linear(cond_dim, d_model)
                self.cond_cross_attn = nn.MultiheadAttention(
                    embed_dim=d_model, num_heads=4, batch_first=True,
                )
                self.cond_norm = nn.LayerNorm(d_model)
            else:
                self.cond_proj = nn.Sequential(
                    nn.Linear(cond_dim, d_model), nn.SiLU(),
                    nn.Linear(d_model, d_model),
                )
                self.cond_cross_attn = None
        else:
            self.cond_proj = None
            self.cond_cross_attn = None

        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads) for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, self.out_channels)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor | None = None):
        """
        x: (B, C, L) noisy input
        t: (B,) timesteps
        cond: (B, cond_dim) optional conditioning
        """
        B, C, L = x.shape
        x = x.permute(0, 2, 1)  # (B, L, C)
        x = self.input_proj(x)  # (B, L, d_model)
        x = x + self.pos_emb[:, :L, :]

        c = self.time_mlp(t)  # (B, d_model)
        if cond is not None and self.cond_proj is not None:
            if self.use_cross_attn and self.cond_cross_attn is not None:
                cond_emb = self.cond_proj(cond).unsqueeze(1)
                c_q = c.unsqueeze(1)
                attn_out, _ = self.cond_cross_attn(c_q, cond_emb, cond_emb)
                c = self.cond_norm(c + attn_out.squeeze(1))
            else:
                c = c + self.cond_proj(cond)

        for block in self.blocks:
            x = block(x, c)

        x = self.final_norm(x)
        x = self.output_proj(x)  # (B, L, C)
        return x.permute(0, 2, 1)  # (B, C, L)


# ---------------------------------------------------------------------------
# Improved DDPM
# ---------------------------------------------------------------------------

class ImprovedDDPM(BaseGenerativeModel):
    """
    DDPM with configurable improvements for ablation study.

    Phase 1 flags:
        use_dit, use_vpred, use_self_cond, use_sigmoid_schedule, use_cross_attn

    Phase 2 flags:
        use_temporal_attn: multi-head self-attention after each ResBlock
        use_hetero_noise: volatility-scaled (heteroskedastic) noise injection
        use_aux_sf_loss: auxiliary kurtosis + ACF matching losses
    """

    def __init__(
        self,
        n_features: int = 16,
        seq_len: int = 60,
        T: int = 1000,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4),
        cond_dim: int = 0,
        cfg_drop_prob: float = 0.1,
        device: str = "cpu",
        # Phase 1 flags
        use_dit: bool = False,
        use_vpred: bool = False,
        use_self_cond: bool = False,
        use_sigmoid_schedule: bool = False,
        use_cross_attn: bool = False,
        # Phase 2 flags
        use_temporal_attn: bool = False,
        use_hetero_noise: bool = False,
        use_aux_sf_loss: bool = False,
        hetero_noise_k: int = 5,
        aux_sf_weight: float = 0.1,
        # Phase 5 flags
        use_acf_guidance: bool = False,
        acf_guidance_scale: float = 0.05,
        use_wavelet: bool = False,
        use_student_t_noise: bool = False,
        student_t_df: float = 5.0,
        # DiT-specific
        dit_d_model: int = 256,
        dit_n_heads: int = 8,
        dit_n_layers: int = 6,
    ):
        tag = "DDPM"
        parts = []
        if use_dit:
            parts.append("DiT")
        if use_vpred:
            parts.append("vpred")
        if use_self_cond:
            parts.append("selfcond")
        if use_sigmoid_schedule:
            parts.append("sigmoid")
        if use_cross_attn:
            parts.append("xattn")
        if use_temporal_attn:
            parts.append("tempattn")
        if use_hetero_noise:
            parts.append("hetero")
        if use_aux_sf_loss:
            parts.append("auxsf")
        if use_acf_guidance:
            parts.append("acfguide")
        if use_wavelet:
            parts.append("wavelet")
        if use_student_t_noise:
            parts.append("studentt")
        if parts:
            tag += "-" + "+".join(parts)

        super().__init__(name=tag, device=device)
        self.n_features = n_features
        self.seq_len = seq_len
        self.T = T
        self.cond_dim = cond_dim
        self.cfg_drop_prob = cfg_drop_prob
        self.use_dit = use_dit
        self.use_vpred = use_vpred
        self.use_self_cond = use_self_cond
        self.use_sigmoid_schedule = use_sigmoid_schedule
        self.use_cross_attn = use_cross_attn
        self.use_temporal_attn = use_temporal_attn
        self.use_hetero_noise = use_hetero_noise
        self.use_aux_sf_loss = use_aux_sf_loss
        self.hetero_noise_k = hetero_noise_k
        self.aux_sf_weight = aux_sf_weight
        self.use_acf_guidance = use_acf_guidance
        self.acf_guidance_scale = acf_guidance_scale
        self.use_wavelet = use_wavelet
        self.use_student_t_noise = use_student_t_noise
        self.student_t_df = student_t_df
        self._ref_acf = None

        if use_sigmoid_schedule:
            betas = sigmoid_beta_schedule(T)
        else:
            betas = cosine_beta_schedule(T)

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_schedule(betas, alphas, alpha_bar)

        n_pools = len(channel_mults)
        padded_len = seq_len
        while padded_len % (2 ** n_pools) != 0:
            padded_len += 1
        self.padded_len = padded_len

        in_ch = n_features * 2 if use_self_cond else n_features
        if use_hetero_noise:
            in_ch += n_features

        out_ch = n_features

        if use_dit:
            self.net = DiT1D(
                in_channels=in_ch,
                seq_len=padded_len,
                d_model=dit_d_model,
                n_heads=dit_n_heads,
                n_layers=dit_n_layers,
                cond_dim=cond_dim,
                use_cross_attn=use_cross_attn,
                out_channels=out_ch,
            ).to(self.device)
        else:
            self.net = UNet1D(
                in_channels=in_ch,
                base_channels=base_channels,
                channel_mults=channel_mults,
                cond_dim=cond_dim,
                use_cross_attn=use_cross_attn,
                out_channels=out_ch,
                use_temporal_attn=use_temporal_attn,
            ).to(self.device)

        self.ema: EMA | None = None
        self._train_step_counter = 0

    @staticmethod
    def _wavelet_encode(data: np.ndarray) -> np.ndarray:
        """Apply Haar wavelet transform along time axis. (N, T, D) -> (N, T, D)."""
        import pywt
        N, T, D = data.shape
        coeffs_list = []
        for i in range(N):
            asset_coeffs = []
            for d in range(D):
                coeffs = pywt.wavedec(data[i, :, d], "haar", level=2)
                flat = np.concatenate(coeffs)
                asset_coeffs.append(flat)
            coeffs_list.append(np.stack(asset_coeffs, axis=-1))
        result = np.stack(coeffs_list)
        return result[:, :T, :]

    @staticmethod
    def _wavelet_decode(coeffs_data: np.ndarray, orig_len: int) -> np.ndarray:
        """Invert Haar wavelet transform. (N, T, D) -> (N, orig_len, D)."""
        import pywt
        N, T, D = coeffs_data.shape
        dummy = np.zeros(orig_len)
        ref_coeffs = pywt.wavedec(dummy, "haar", level=2)
        coeff_lengths = [len(c) for c in ref_coeffs]

        result = np.zeros((N, orig_len, D), dtype=coeffs_data.dtype)
        for i in range(N):
            for d in range(D):
                flat = coeffs_data[i, :, d]
                split_coeffs = []
                idx = 0
                for length in coeff_lengths:
                    end = min(idx + length, len(flat))
                    c = flat[idx:end]
                    if len(c) < length:
                        c = np.pad(c, (0, length - len(c)))
                    split_coeffs.append(c)
                    idx = end
                rec = pywt.waverec(split_coeffs, "haar")
                result[i, :, d] = rec[:orig_len]
        return result

    def register_schedule(self, betas, alphas, alpha_bar):
        self.betas = betas.to(self.device)
        self.alphas = alphas.to(self.device)
        self.alpha_bar = alpha_bar.to(self.device)
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar).to(self.device)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar).to(self.device)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / alphas).to(self.device)
        self.posterior_variance = (
            betas * (1.0 - torch.cat([torch.tensor([1.0]), alpha_bar[:-1]])) / (1.0 - alpha_bar)
        ).to(self.device)

    def _extract(self, tensor: torch.Tensor, t: torch.Tensor, shape: tuple):
        out = tensor.gather(-1, t)
        return out.reshape(-1, *([1] * (len(shape) - 1)))

    def _compute_local_vol(self, x0: torch.Tensor) -> torch.Tensor:
        """Compute local volatility from squared returns: (B, C, L)."""
        k = self.hetero_noise_k
        sq = x0 ** 2
        kernel = torch.ones(1, 1, 2 * k + 1, device=x0.device) / (2 * k + 1)
        B, C, L = sq.shape
        sq_flat = sq.reshape(B * C, 1, L)
        local_var = F.conv1d(sq_flat, kernel, padding=k)
        local_vol = torch.sqrt(local_var + 1e-8).reshape(B, C, L)
        local_vol = local_vol / (local_vol.mean() + 1e-8)
        return local_vol

    def _sample_noise(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Sample noise: Gaussian by default, Student-t if enabled."""
        if self.use_student_t_noise:
            dist = torch.distributions.StudentT(df=self.student_t_df)
            noise = dist.sample(shape).to(device)
            noise = noise / (self.student_t_df / (self.student_t_df - 2)) ** 0.5
            return noise
        return torch.randn(shape, device=device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None,
                 local_vol: torch.Tensor | None = None):
        if noise is None:
            noise = self._sample_noise(x0.shape, x0.device)
        sqrt_ab = self._extract(self.sqrt_alpha_bar, t, x0.shape)
        sqrt_1m_ab = self._extract(self.sqrt_one_minus_alpha_bar, t, x0.shape)
        if self.use_hetero_noise and local_vol is not None:
            return sqrt_ab * x0 + sqrt_1m_ab * local_vol * noise
        return sqrt_ab * x0 + sqrt_1m_ab * noise

    def _get_target(self, x0: torch.Tensor, noise: torch.Tensor,
                    t: torch.Tensor) -> torch.Tensor:
        if self.use_vpred:
            sqrt_ab = self._extract(self.sqrt_alpha_bar, t, x0.shape)
            sqrt_1m_ab = self._extract(self.sqrt_one_minus_alpha_bar, t, x0.shape)
            return sqrt_ab * noise - sqrt_1m_ab * x0
        return noise

    def _predict_x0(self, x_t: torch.Tensor, t: torch.Tensor,
                    model_out: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self._extract(self.sqrt_alpha_bar, t, x_t.shape)
        sqrt_1m_ab = self._extract(self.sqrt_one_minus_alpha_bar, t, x_t.shape)
        if self.use_vpred:
            return sqrt_ab * x_t - sqrt_1m_ab * model_out
        return (x_t - sqrt_1m_ab * model_out) / sqrt_ab

    def _predict_noise(self, x_t: torch.Tensor, t: torch.Tensor,
                       model_out: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self._extract(self.sqrt_alpha_bar, t, x_t.shape)
        sqrt_1m_ab = self._extract(self.sqrt_one_minus_alpha_bar, t, x_t.shape)
        if self.use_vpred:
            return sqrt_1m_ab * x_t + sqrt_ab * model_out
        return model_out

    def _net_with_self_cond(self, net: nn.Module, x: torch.Tensor,
                            t: torch.Tensor, cond: torch.Tensor | None,
                            x0_self_cond: torch.Tensor | None = None,
                            local_vol: torch.Tensor | None = None):
        if self.use_self_cond:
            if x0_self_cond is None:
                x0_self_cond = torch.zeros_like(x)
            net_input = torch.cat([x, x0_self_cond], dim=1)
        else:
            net_input = x
        if self.use_hetero_noise:
            if local_vol is None:
                local_vol = torch.ones_like(x)
            net_input = torch.cat([net_input, local_vol], dim=1)
        return net(net_input, t, cond)

    @staticmethod
    def _diff_kurtosis(x: torch.Tensor) -> torch.Tensor:
        """Differentiable excess kurtosis over the last two dims."""
        flat = x.reshape(x.shape[0], -1)
        mu = flat.mean(dim=1, keepdim=True)
        diff = flat - mu
        var = (diff ** 2).mean(dim=1, keepdim=True).clamp(min=1e-8)
        kurt = (diff ** 4).mean(dim=1, keepdim=True) / (var ** 2) - 3.0
        return kurt.mean()

    @staticmethod
    def _diff_acf_abs(x: torch.Tensor, max_lag: int = 20) -> torch.Tensor:
        """Differentiable ACF of |returns| averaged across batch and features."""
        flat = x.reshape(x.shape[0], -1)
        abs_ret = flat.abs()
        mu = abs_ret.mean(dim=1, keepdim=True)
        centered = abs_ret - mu
        var = (centered ** 2).mean(dim=1, keepdim=True).clamp(min=1e-8)
        acf_vals = []
        for lag in range(1, max_lag + 1):
            cov = (centered[:, lag:] * centered[:, :-lag]).mean(dim=1, keepdim=True)
            acf_vals.append((cov / var).mean())
        return torch.stack(acf_vals)

    def p_losses(self, x0: torch.Tensor, t: torch.Tensor,
                 cond: torch.Tensor | None = None):
        noise = self._sample_noise(x0.shape, x0.device)

        local_vol = None
        if self.use_hetero_noise:
            local_vol = self._compute_local_vol(x0)

        x_noisy = self.q_sample(x0, t, noise, local_vol=local_vol)
        target = self._get_target(x0, noise, t)

        x0_self_cond = None
        if self.use_self_cond and random.random() > 0.5:
            with torch.no_grad():
                raw_out = self._net_with_self_cond(
                    self.net, x_noisy, t, cond, None, local_vol)
                x0_self_cond = self._predict_x0(x_noisy, t, raw_out).detach()
                x0_self_cond = x0_self_cond.clamp(-5, 5)

        pred = self._net_with_self_cond(
            self.net, x_noisy, t, cond, x0_self_cond, local_vol)
        loss = F.mse_loss(pred, target)

        if self.use_aux_sf_loss and self._train_step_counter % 5 == 0:
            x0_pred = self._predict_x0(x_noisy, t, pred).clamp(-5, 5)
            real_kurt = self._diff_kurtosis(x0)
            pred_kurt = self._diff_kurtosis(x0_pred)
            loss_kurt = (pred_kurt - real_kurt) ** 2

            real_acf = self._diff_acf_abs(x0)
            pred_acf = self._diff_acf_abs(x0_pred)
            loss_acf = F.mse_loss(pred_acf, real_acf)

            loss = loss + self.aux_sf_weight * (loss_kurt + loss_acf)

        self._train_step_counter += 1
        return loss

    def train(self, data: np.ndarray, cond: np.ndarray | None = None,
              epochs: int = 200, batch_size: int = 64, lr: float = 2e-4,
              ema_decay: float = 0.9999, **kwargs) -> dict:
        self.net.train()

        train_data = data
        if self.use_wavelet:
            train_data = self._wavelet_encode(data)

        if self.use_acf_guidance:
            flat = data.reshape(data.shape[0], -1)
            abs_ret = np.abs(flat)
            mu = abs_ret.mean(axis=1, keepdims=True)
            centered = abs_ret - mu
            var = centered.var(axis=1, keepdims=True) + 1e-8
            acf_vals = []
            for lag in range(1, 21):
                cov = (centered[:, lag:] * centered[:, :-lag]).mean(axis=1, keepdims=True)
                acf_vals.append((cov / var).mean())
            self._ref_acf = torch.tensor(acf_vals, dtype=torch.float32)

        x = torch.tensor(train_data, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        x = x.permute(0, 2, 1)  # (N, D, W)

        if x.shape[-1] < self.padded_len:
            pad = self.padded_len - x.shape[-1]
            x = F.pad(x, (0, pad))

        cond_tensor = None
        if cond is not None and self.cond_dim > 0:
            cond_tensor = torch.tensor(cond, dtype=torch.float32)

        if cond_tensor is not None:
            dataset = TensorDataset(x, cond_tensor)
        else:
            dataset = TensorDataset(x)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01)

        self.ema = EMA(self.net, decay=ema_decay)

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_data in loader:
                if cond_tensor is not None:
                    batch_x, batch_c = batch_data
                    batch_c = batch_c.to(self.device)
                    if random.random() < self.cfg_drop_prob:
                        batch_c = None
                else:
                    (batch_x,) = batch_data
                    batch_c = None

                batch_x = batch_x.to(self.device)
                t = torch.randint(0, self.T, (batch_x.shape[0],),
                                  device=self.device)
                loss = self.p_losses(batch_x, t, batch_c)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                self.ema.update(self.net)
                epoch_loss += loss.item()

            scheduler.step()
            avg = epoch_loss / max(len(loader), 1)
            losses.append(avg)
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"  [{self.name}] Epoch {epoch+1:4d}/{epochs} | "
                      f"loss={avg:.6f} | lr={scheduler.get_last_lr()[0]:.2e}")

        self.is_trained = True
        return {"losses": losses}

    def generate(self, n_samples: int, seq_len: int | None = None,
                 cond: np.ndarray | None = None, use_ddim: bool = True,
                 ddim_steps: int = 50, guidance_scale: float = 2.0,
                 **kwargs) -> np.ndarray:
        net = self.ema.shadow if self.ema is not None else self.net
        net.eval()

        if seq_len is None:
            seq_len = self.seq_len

        x = torch.randn(n_samples, self.n_features, self.padded_len,
                         device=self.device)

        cond_t = None
        if cond is not None and self.cond_dim > 0:
            cond_t = torch.tensor(cond, dtype=torch.float32, device=self.device)
            if cond_t.ndim == 1:
                cond_t = cond_t.unsqueeze(0).expand(n_samples, -1)

        if self.use_acf_guidance:
            if use_ddim:
                x = self._ddim_sample(x, net, ddim_steps, cond_t, guidance_scale)
            else:
                x0_self_cond = None
                for t_idx in reversed(range(self.T)):
                    x, x0_self_cond = self._p_sample_step(
                        x, t_idx, net, cond_t, guidance_scale, x0_self_cond)
        else:
            with torch.no_grad():
                if use_ddim:
                    x = self._ddim_sample(x, net, ddim_steps, cond_t, guidance_scale)
                else:
                    x0_self_cond = None
                    for t_idx in reversed(range(self.T)):
                        x, x0_self_cond = self._p_sample_step(
                            x, t_idx, net, cond_t, guidance_scale, x0_self_cond)

        x = x[:, :, :seq_len]
        result = x.permute(0, 2, 1).cpu().numpy()

        if self.use_wavelet:
            result = self._wavelet_decode(result, seq_len)

        return result

    def _p_sample_step(self, x, t_idx, net, cond, guidance_scale, x0_self_cond):
        B = x.shape[0]
        t = torch.full((B,), t_idx, device=self.device, dtype=torch.long)

        local_vol = None
        if self.use_hetero_noise and x0_self_cond is not None:
            local_vol = self._compute_local_vol(x0_self_cond)

        if cond is not None and guidance_scale > 1.0:
            out_c = self._net_with_self_cond(
                net, x, t, cond, x0_self_cond, local_vol)
            out_u = self._net_with_self_cond(
                net, x, t, None, x0_self_cond, local_vol)
            model_out = out_u + guidance_scale * (out_c - out_u)
        else:
            model_out = self._net_with_self_cond(
                net, x, t, cond, x0_self_cond, local_vol)

        pred_noise = self._predict_noise(x, t, model_out)
        x0_pred = self._predict_x0(x, t, model_out).clamp(-5, 5)

        sqrt_recip = self._extract(self.sqrt_recip_alpha, t, x.shape)
        beta = self._extract(self.betas, t, x.shape)
        sqrt_1m_ab = self._extract(self.sqrt_one_minus_alpha_bar, t, x.shape)

        mean = sqrt_recip * (x - beta / sqrt_1m_ab * pred_noise)

        if t_idx > 0:
            var = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(var) * noise
        else:
            x = mean

        return x, x0_pred

    def _acf_guidance_step(self, x: torch.Tensor, net: nn.Module,
                           t: torch.Tensor, cond: torch.Tensor | None,
                           step_idx: int, total_steps: int) -> torch.Tensor:
        """Apply inference-time ACF guidance: nudge x toward correct ACF profile."""
        if self._ref_acf is None or not self.use_acf_guidance:
            return x
        if step_idx % 5 != 0:
            return x

        x_in = x.detach().requires_grad_(True)
        model_out = net(x_in, t, cond)
        x0_pred = self._predict_x0(x_in, t, model_out).clamp(-5, 5)
        pred_acf = self._diff_acf_abs(x0_pred)
        ref = self._ref_acf.to(x.device)
        loss = F.mse_loss(pred_acf, ref)
        grad = torch.autograd.grad(loss, x_in)[0]
        scale = self.acf_guidance_scale * (step_idx / max(total_steps, 1))
        return x.detach() - scale * grad.detach()

    def _ddim_sample(self, x: torch.Tensor, net: nn.Module, n_steps: int,
                     cond: torch.Tensor | None,
                     guidance_scale: float) -> torch.Tensor:
        n_steps = min(n_steps, self.T)
        indices = np.linspace(0, self.T - 1, n_steps, dtype=int)
        timesteps = list(reversed(indices.tolist()))
        B = x.shape[0]
        x0_self_cond = None

        for i, t_cur in enumerate(timesteps):
            t = torch.full((B,), t_cur, device=self.device, dtype=torch.long)
            alpha_bar_t = self.alpha_bar[t_cur]

            local_vol = None
            if self.use_hetero_noise and x0_self_cond is not None:
                local_vol = self._compute_local_vol(x0_self_cond)

            if cond is not None and guidance_scale > 1.0:
                out_c = self._net_with_self_cond(
                    net, x, t, cond, x0_self_cond, local_vol)
                out_u = self._net_with_self_cond(
                    net, x, t, None, x0_self_cond, local_vol)
                model_out = out_u + guidance_scale * (out_c - out_u)
            else:
                model_out = self._net_with_self_cond(
                    net, x, t, cond, x0_self_cond, local_vol)

            pred_noise = self._predict_noise(x, t, model_out)
            x0_pred = self._predict_x0(x, t, model_out).clamp(-5, 5)

            if self.use_self_cond or self.use_hetero_noise:
                x0_self_cond = x0_pred

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_bar_next = self.alpha_bar[t_next]
            else:
                alpha_bar_next = torch.tensor(1.0, device=self.device)

            x = (alpha_bar_next.sqrt() * x0_pred +
                 (1 - alpha_bar_next).sqrt() * pred_noise)

            if self.use_acf_guidance:
                x = self._acf_guidance_step(x, net, t, cond, i, n_steps)

        return x

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_dict = {
            "net_state": self.net.state_dict(),
            "config": {
                "n_features": self.n_features,
                "seq_len": self.seq_len,
                "T": self.T,
                "padded_len": self.padded_len,
                "cond_dim": self.cond_dim,
                "cfg_drop_prob": self.cfg_drop_prob,
                "use_dit": self.use_dit,
                "use_vpred": self.use_vpred,
                "use_self_cond": self.use_self_cond,
                "use_sigmoid_schedule": self.use_sigmoid_schedule,
                "use_cross_attn": self.use_cross_attn,
                "use_temporal_attn": self.use_temporal_attn,
                "use_hetero_noise": self.use_hetero_noise,
                "use_aux_sf_loss": self.use_aux_sf_loss,
                "use_acf_guidance": self.use_acf_guidance,
                "use_wavelet": self.use_wavelet,
                "use_student_t_noise": self.use_student_t_noise,
            },
        }
        if self.ema is not None:
            save_dict["ema_state"] = self.ema.state_dict()
        torch.save(save_dict, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["net_state"])
        if "ema_state" in ckpt:
            self.ema = EMA(self.net)
            self.ema.load_state_dict(ckpt["ema_state"])
        self.is_trained = True
