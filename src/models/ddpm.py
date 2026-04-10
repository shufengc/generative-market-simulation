"""
Denoising Diffusion Probabilistic Model (DDPM) for financial time series.

Features:
  - 1-D U-Net noise predictor with sinusoidal time embeddings
  - Cosine / linear beta schedules
  - Classifier-free guidance for conditional generation
  - EMA (exponential moving average) of model weights
  - DDIM fast sampler (50 steps default vs 1000 for vanilla)
  - Full macro-regime conditioning pipeline
"""

from __future__ import annotations

import os
import copy
import math
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.base_model import BaseGenerativeModel


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
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
# Residual block with time conditioning
# ---------------------------------------------------------------------------

class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.mlp_t = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.block1 = nn.Sequential(
            nn.GroupNorm(min(8, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.mlp_t(t_emb)[:, :, None]
        h = self.block2(h)
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Simple 1-D U-Net noise predictor
# ---------------------------------------------------------------------------

class UNet1D(nn.Module):
    """
    Lightweight 1-D U-Net that predicts noise epsilon given (x_t, t).
    channel_mults control the depth; default gives 3 levels.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4),
        time_dim: int = 128,
        cond_dim: int = 0,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        if cond_dim > 0:
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            self.cond_proj = None

        self.init_conv = nn.Conv1d(in_channels, base_channels, 1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_pools = nn.ModuleList()
        ch = base_channels
        channels = [ch]
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.down_blocks.append(ResBlock1D(ch, out_ch, time_dim))
            self.down_pools.append(nn.Conv1d(out_ch, out_ch, 2, stride=2, padding=0))
            ch = out_ch
            channels.append(ch)

        # Bottleneck
        self.mid = ResBlock1D(ch, ch, time_dim)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.up_samples.append(nn.ConvTranspose1d(ch, out_ch, 2, stride=2))
            skip_ch = channels.pop()
            self.up_blocks.append(ResBlock1D(out_ch + skip_ch, out_ch, time_dim))
            ch = out_ch

        self.final_conv = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch),
            nn.SiLU(),
            nn.Conv1d(ch, in_channels, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None):
        """
        x: (B, C, L) noisy input
        t: (B,) integer diffusion timesteps
        cond: (B, cond_dim) optional conditioning vector
        """
        t_emb = self.time_mlp(t)
        if cond is not None and self.cond_proj is not None:
            t_emb = t_emb + self.cond_proj(cond)

        x = self.init_conv(x)

        skips = [x]
        for block, pool in zip(self.down_blocks, self.down_pools):
            x = block(x, t_emb)
            skips.append(x)
            x = pool(x)

        x = self.mid(x, t_emb)

        for up_sample, block in zip(self.up_samples, self.up_blocks):
            x = up_sample(x)
            skip = skips.pop()
            min_len = min(x.shape[-1], skip.shape[-1])
            x = torch.cat([x[..., :min_len], skip[..., :min_len]], dim=1)
            x = block(x, t_emb)

        return self.final_conv(x)


# ---------------------------------------------------------------------------
# EMA wrapper
# ---------------------------------------------------------------------------

class EMA:
    """Exponential moving average of model parameters."""

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
# Diffusion schedule helpers
# ---------------------------------------------------------------------------

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.linspace(0, T, T + 1)
    alpha_bar = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(0.0001, 0.9999)


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


# ---------------------------------------------------------------------------
# DDPM model wrapper
# ---------------------------------------------------------------------------

class DDPMModel(BaseGenerativeModel):
    """
    Full DDPM pipeline: schedule, training with EMA + CFG, DDIM sampling.
    """

    def __init__(
        self,
        n_features: int = 18,
        seq_len: int = 60,
        T: int = 1000,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4),
        schedule: str = "cosine",
        cond_dim: int = 0,
        cfg_drop_prob: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__(name="DDPM", device=device)
        self.n_features = n_features
        self.seq_len = seq_len
        self.T = T
        self.cond_dim = cond_dim
        self.cfg_drop_prob = cfg_drop_prob

        # Build noise schedule
        if schedule == "cosine":
            betas = cosine_beta_schedule(T)
        else:
            betas = linear_beta_schedule(T)

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(self.device)
        self.alphas = alphas.to(self.device)
        self.alpha_bar = alpha_bar.to(self.device)
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar).to(self.device)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar).to(self.device)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / alphas).to(self.device)
        self.posterior_variance = (
            betas * (1.0 - torch.cat([torch.tensor([1.0]), alpha_bar[:-1]])) / (1.0 - alpha_bar)
        ).to(self.device)

        # Pad seq_len to nearest power-of-2-divisible length for U-Net pooling
        n_pools = len(channel_mults)
        padded_len = seq_len
        while padded_len % (2 ** n_pools) != 0:
            padded_len += 1
        self.padded_len = padded_len

        # Build network
        self.net = UNet1D(
            in_channels=n_features,
            base_channels=base_channels,
            channel_mults=channel_mults,
            cond_dim=cond_dim,
        ).to(self.device)

        # EMA (initialized after first training call)
        self.ema: EMA | None = None

    def _extract(self, tensor: torch.Tensor, t: torch.Tensor, shape: tuple):
        out = tensor.gather(-1, t)
        return out.reshape(-1, *([1] * (len(shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        """Forward diffusion: add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._extract(self.sqrt_alpha_bar, t, x0.shape)
        sqrt_1m_ab = self._extract(self.sqrt_one_minus_alpha_bar, t, x0.shape)
        return sqrt_ab * x0 + sqrt_1m_ab * noise

    def p_losses(self, x0: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None):
        """Compute simplified DDPM loss: MSE between true noise and predicted noise."""
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        pred_noise = self.net(x_noisy, t, cond)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t_idx: int, cond: torch.Tensor | None = None,
                 net: nn.Module | None = None):
        """Single reverse diffusion step."""
        if net is None:
            net = self.net
        B = x.shape[0]
        t = torch.full((B,), t_idx, device=self.device, dtype=torch.long)

        pred_noise = net(x, t, cond)

        sqrt_recip = self._extract(self.sqrt_recip_alpha, t, x.shape)
        beta = self._extract(self.betas, t, x.shape)
        sqrt_1m_ab = self._extract(self.sqrt_one_minus_alpha_bar, t, x.shape)

        mean = sqrt_recip * (x - beta / sqrt_1m_ab * pred_noise)

        if t_idx > 0:
            var = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return mean + torch.sqrt(var) * noise
        return mean

    def train(self, data: np.ndarray, cond: np.ndarray | None = None,
              epochs: int = 200, batch_size: int = 64, lr: float = 1e-4,
              ema_decay: float = 0.9999, **kwargs) -> dict:
        """
        Train the DDPM on windowed return data with optional conditioning.

        Args:
            data: (N, seq_len, n_features) array of normalized return windows.
            cond: (N, cond_dim) array of conditioning vectors (optional).
        """
        self.net.train()

        x = torch.tensor(data, dtype=torch.float32)
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

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

        self.ema = EMA(self.net, decay=ema_decay)

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_data in loader:
                if cond_tensor is not None:
                    batch_x, batch_c = batch_data
                    batch_c = batch_c.to(self.device)
                    # Classifier-free guidance: randomly drop conditioning
                    if random.random() < self.cfg_drop_prob:
                        batch_c = None
                else:
                    (batch_x,) = batch_data
                    batch_c = None

                batch_x = batch_x.to(self.device)
                t = torch.randint(0, self.T, (batch_x.shape[0],), device=self.device)
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
                print(f"  Epoch {epoch+1:4d}/{epochs} | loss={avg:.6f} | lr={scheduler.get_last_lr()[0]:.2e}")

        self.is_trained = True
        return {"losses": losses}

    @torch.no_grad()
    def generate(self, n_samples: int, seq_len: int | None = None,
                 cond: np.ndarray | None = None, use_ddim: bool = True,
                 ddim_steps: int = 50, guidance_scale: float = 2.0,
                 **kwargs) -> np.ndarray:
        """
        Generate synthetic return sequences.

        Args:
            n_samples: number of sequences to generate
            seq_len: length of each sequence (defaults to training seq_len)
            cond: (cond_dim,) or (n_samples, cond_dim) conditioning vector
            use_ddim: use DDIM fast sampler (recommended)
            ddim_steps: number of DDIM steps (ignored if use_ddim=False)
            guidance_scale: classifier-free guidance strength (>1 = stronger conditioning)
        """
        net = self.ema.shadow if self.ema is not None else self.net
        net.eval()

        if seq_len is None:
            seq_len = self.seq_len

        x = torch.randn(n_samples, self.n_features, self.padded_len, device=self.device)

        cond_t = None
        if cond is not None and self.cond_dim > 0:
            cond_t = torch.tensor(cond, dtype=torch.float32, device=self.device)
            if cond_t.ndim == 1:
                cond_t = cond_t.unsqueeze(0).expand(n_samples, -1)

        if use_ddim:
            x = self._ddim_sample(x, net, ddim_steps, cond_t, guidance_scale)
        else:
            for t_idx in reversed(range(self.T)):
                if cond_t is not None and guidance_scale > 1.0:
                    x = self._cfg_step(x, t_idx, net, cond_t, guidance_scale)
                else:
                    x = self.p_sample(x, t_idx, cond_t, net)

        x = x[:, :, :seq_len]
        result = x.permute(0, 2, 1).cpu().numpy()
        return result

    def _ddim_sample(self, x: torch.Tensor, net: nn.Module, n_steps: int,
                     cond: torch.Tensor | None, guidance_scale: float) -> torch.Tensor:
        """DDIM deterministic sampler with optional classifier-free guidance."""
        n_steps = min(n_steps, self.T)
        indices = np.linspace(0, self.T - 1, n_steps, dtype=int)
        timesteps = list(reversed(indices.tolist()))

        B = x.shape[0]

        for i, t_cur in enumerate(timesteps):
            t = torch.full((B,), t_cur, device=self.device, dtype=torch.long)
            alpha_bar_t = self.alpha_bar[t_cur]

            # Predict noise with optional CFG
            if cond is not None and guidance_scale > 1.0:
                eps_cond = net(x, t, cond)
                eps_uncond = net(x, t, None)
                pred_noise = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                pred_noise = net(x, t, cond)

            # Predict x0
            x0_pred = (x - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()
            x0_pred = x0_pred.clamp(-5, 5)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_bar_next = self.alpha_bar[t_next]
            else:
                alpha_bar_next = torch.tensor(1.0, device=self.device)

            # DDIM update (deterministic, eta=0)
            x = alpha_bar_next.sqrt() * x0_pred + (1 - alpha_bar_next).sqrt() * pred_noise

        return x

    def _cfg_step(self, x: torch.Tensor, t_idx: int, net: nn.Module,
                  cond: torch.Tensor, guidance_scale: float) -> torch.Tensor:
        """Single reverse step with classifier-free guidance."""
        B = x.shape[0]
        t = torch.full((B,), t_idx, device=self.device, dtype=torch.long)

        eps_cond = net(x, t, cond)
        eps_uncond = net(x, t, None)
        pred_noise = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        sqrt_recip = self._extract(self.sqrt_recip_alpha, t, x.shape)
        beta = self._extract(self.betas, t, x.shape)
        sqrt_1m_ab = self._extract(self.sqrt_one_minus_alpha_bar, t, x.shape)

        mean = sqrt_recip * (x - beta / sqrt_1m_ab * pred_noise)

        if t_idx > 0:
            var = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return mean + torch.sqrt(var) * noise
        return mean

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.utils.config import DEFAULT_DEVICE
    parser = argparse.ArgumentParser(description="DDPM for financial time series")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--cond-dim", type=int, default=5)
    parser.add_argument("--regime", type=str, default=None, choices=["crisis", "calm", "normal"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/ddpm.pt")
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(os.path.dirname(__file__), "..", "..", "data")

    if args.train:
        windows = np.load(os.path.join(data_dir, "windows.npy"))
        n_features = windows.shape[2]
        seq_len = windows.shape[1]

        cond = None
        cond_path = os.path.join(data_dir, "window_cond.npy")
        if os.path.exists(cond_path):
            cond = np.load(cond_path)
            print(f"Using conditioning vectors: {cond.shape}")

        print(f"Training DDPM on {windows.shape[0]} windows ({seq_len} x {n_features})")

        model = DDPMModel(
            n_features=n_features,
            seq_len=seq_len,
            cond_dim=cond.shape[1] if cond is not None else 0,
            device=DEFAULT_DEVICE,
        )
        history = model.train(windows, cond=cond, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        model.save(args.checkpoint)
        print(f"Saved checkpoint -> {args.checkpoint}")

    if args.generate:
        windows = np.load(os.path.join(data_dir, "windows.npy"))
        n_features = windows.shape[2]
        seq_len = windows.shape[1]

        cond_dim = args.cond_dim
        cond_path = os.path.join(data_dir, "window_cond.npy")
        if not os.path.exists(cond_path):
            cond_dim = 0

        model = DDPMModel(
            n_features=n_features,
            seq_len=seq_len,
            cond_dim=cond_dim,
            device=DEFAULT_DEVICE,
        )
        model.load(args.checkpoint)

        gen_cond = None
        if args.regime and cond_dim > 0:
            from src.data.regime_labels import get_regime_conditioning_vectors
            gen_cond = get_regime_conditioning_vectors()[args.regime]

        synthetic = model.generate(args.n_samples, seq_len, cond=gen_cond)
        suffix = f"_{args.regime}" if args.regime else ""
        out_path = os.path.join(data_dir, f"generated_ddpm{suffix}.npy")
        np.save(out_path, synthetic)
        print(f"Generated {synthetic.shape} -> {out_path}")
