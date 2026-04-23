"""
Temporary improved DDPM variant.

This module keeps the original src.models.ddpm_improved implementation intact
and applies a focused set of training/sampling upgrades in a new file:

1) Min-SNR loss reweighting
2) Per-sample CFG conditioning drop during training
3) Reverse-step noise aligned with forward noise sampler
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.ddpm_improved import ImprovedDDPM as _BaseImprovedDDPM


class ImprovedDDPM(_BaseImprovedDDPM):
    """
    Drop-in replacement for ImprovedDDPM with three targeted upgrades.
    """

    def __init__(
        self,
        *args: Any,
        use_min_snr_loss: bool = True,
        min_snr_gamma: float = 5.0,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.use_min_snr_loss = use_min_snr_loss
        self.min_snr_gamma = min_snr_gamma

    def p_losses(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ):
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
                    self.net, x_noisy, t, cond, None, local_vol
                )
                x0_self_cond = self._predict_x0(x_noisy, t, raw_out).detach()
                x0_self_cond = x0_self_cond.clamp(-5, 5)

        pred = self._net_with_self_cond(
            self.net, x_noisy, t, cond, x0_self_cond, local_vol
        )

        # Base objective with optional Min-SNR timestep reweighting.
        mse = F.mse_loss(pred, target, reduction="none")
        per_sample = mse.mean(dim=(1, 2))
        if self.use_min_snr_loss:
            alpha_bar_t = self._extract(self.alpha_bar, t, x0.shape).reshape(-1)
            snr = alpha_bar_t / (1.0 - alpha_bar_t).clamp(min=1e-8)
            gamma = torch.full_like(snr, self.min_snr_gamma)
            weight = torch.minimum(snr, gamma) / snr.clamp(min=1e-8)
            loss = (weight * per_sample).mean()
        else:
            loss = per_sample.mean()

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

    def train(
        self,
        data: np.ndarray,
        cond: np.ndarray | None = None,
        epochs: int = 200,
        batch_size: int = 64,
        lr: float = 2e-4,
        ema_decay: float = 0.9999,
        **kwargs,
    ) -> dict:
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
        x = x.permute(0, 2, 1)

        if x.shape[-1] < self.padded_len:
            pad = self.padded_len - x.shape[-1]
            x = F.pad(x, (0, pad))

        cond_tensor = None
        if cond is not None and self.cond_dim > 0:
            cond_tensor = torch.tensor(cond, dtype=torch.float32)

        dataset = TensorDataset(x, cond_tensor) if cond_tensor is not None else TensorDataset(x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )

        self.ema = self.ema.__class__(self.net, decay=ema_decay) if self.ema is not None else None
        if self.ema is None:
            from src.models.ddpm_improved import EMA  # local import to avoid duplication
            self.ema = EMA(self.net, decay=ema_decay)

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_data in loader:
                if cond_tensor is not None:
                    batch_x, batch_c = batch_data
                    batch_c = batch_c.to(self.device)

                    # CFG drop is applied per sample (instead of dropping full batch).
                    if self.cfg_drop_prob > 0.0:
                        drop_mask = torch.rand(batch_c.shape[0], device=self.device) < self.cfg_drop_prob
                        if drop_mask.any():
                            batch_c = batch_c.clone()
                            batch_c[drop_mask] = 0.0
                else:
                    (batch_x,) = batch_data
                    batch_c = None

                batch_x = batch_x.to(self.device)
                t = torch.randint(0, self.T, (batch_x.shape[0],), device=self.device)
                loss = self.p_losses(batch_x, t, batch_c)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                self.ema.update(self.net)
                epoch_loss += loss.item()

            scheduler.step()
            avg = epoch_loss / max(len(loader), 1)
            losses.append(avg)
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(
                    f"  [{self.name}] Epoch {epoch+1:4d}/{epochs} | "
                    f"loss={avg:.6f} | lr={scheduler.get_last_lr()[0]:.2e}"
                )

        self.is_trained = True
        return {"losses": losses}

    def _p_sample_step(self, x, t_idx, net, cond, guidance_scale, x0_self_cond):
        B = x.shape[0]
        t = torch.full((B,), t_idx, device=self.device, dtype=torch.long)

        local_vol = None
        if self.use_hetero_noise and x0_self_cond is not None:
            local_vol = self._compute_local_vol(x0_self_cond)

        if cond is not None and guidance_scale > 1.0:
            out_c = self._net_with_self_cond(net, x, t, cond, x0_self_cond, local_vol)
            out_u = self._net_with_self_cond(net, x, t, None, x0_self_cond, local_vol)
            model_out = out_u + guidance_scale * (out_c - out_u)
        else:
            model_out = self._net_with_self_cond(net, x, t, cond, x0_self_cond, local_vol)

        pred_noise = self._predict_noise(x, t, model_out)
        x0_pred = self._predict_x0(x, t, model_out).clamp(-5, 5)

        sqrt_recip = self._extract(self.sqrt_recip_alpha, t, x.shape)
        beta = self._extract(self.betas, t, x.shape)
        sqrt_1m_ab = self._extract(self.sqrt_one_minus_alpha_bar, t, x.shape)
        mean = sqrt_recip * (x - beta / sqrt_1m_ab * pred_noise)

        if t_idx > 0:
            var = self._extract(self.posterior_variance, t, x.shape)
            # Keep reverse-process noise family consistent with forward process.
            noise = self._sample_noise(x.shape, x.device)
            x = mean + torch.sqrt(var) * noise
        else:
            x = mean

        return x, x0_pred
