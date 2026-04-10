"""
TimeGAN-inspired model for financial time series generation.

Architecture:
  - Embedding network: maps real space to latent space
  - Recovery network: maps latent space back to real space
  - Generator: produces latent sequences from noise
  - Discriminator: distinguishes real vs fake latent sequences
  - Supervisor: captures temporal dynamics in latent space

Reference: Yoon et al., "Time-series Generative Adversarial Networks", NeurIPS 2019
"""

from __future__ import annotations

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.base_model import BaseGenerativeModel


class RNNBlock(nn.Module):
    """GRU-based sequence model used across all TimeGAN components."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int = 2, use_spectral_norm: bool = False):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        fc = nn.Linear(hidden_dim, output_dim)
        if use_spectral_norm:
            fc = nn.utils.spectral_norm(fc)
        self.fc = fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.fc(h)


class TimeGANModel(BaseGenerativeModel):
    """TimeGAN for financial time series generation with improved stability."""

    def __init__(
        self,
        n_features: int = 18,
        seq_len: int = 60,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        n_layers: int = 2,
        device: str = "cpu",
    ):
        super().__init__(name="TimeGAN", device=device)
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.embedder = RNNBlock(n_features, hidden_dim, latent_dim, n_layers).to(self.device)
        self.recovery = RNNBlock(latent_dim, hidden_dim, n_features, n_layers).to(self.device)
        self.generator = RNNBlock(n_features, hidden_dim, latent_dim, n_layers).to(self.device)
        self.supervisor = RNNBlock(latent_dim, hidden_dim, latent_dim, n_layers).to(self.device)
        self.discriminator = RNNBlock(
            latent_dim, hidden_dim, 1, n_layers, use_spectral_norm=True
        ).to(self.device)

    def _random_noise(self, batch_size: int, seq_len: int) -> torch.Tensor:
        return torch.randn(batch_size, seq_len, self.n_features, device=self.device)

    def _gradient_penalty(self, real_h: torch.Tensor, fake_h: torch.Tensor) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP style regularization."""
        B = real_h.shape[0]
        alpha = torch.rand(B, 1, 1, device=self.device)
        interp = (alpha * real_h + (1 - alpha) * fake_h).requires_grad_(True)
        d_interp = self.discriminator(interp)
        grad = torch.autograd.grad(
            outputs=d_interp, inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True, retain_graph=True,
        )[0]
        grad_norm = grad.reshape(B, -1).norm(2, dim=1)
        return ((grad_norm - 1) ** 2).mean()

    def train(self, data: np.ndarray, epochs: int = 200, batch_size: int = 64,
              lr: float = 1e-3, gp_weight: float = 10.0, **kwargs) -> dict:
        x_tensor = torch.tensor(data, dtype=torch.float32)
        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(-1)

        dataset = TensorDataset(x_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        opt_ae = torch.optim.Adam(
            list(self.embedder.parameters()) + list(self.recovery.parameters()), lr=lr
        )
        opt_sup = torch.optim.Adam(self.supervisor.parameters(), lr=lr)
        opt_g = torch.optim.Adam(
            list(self.generator.parameters()) + list(self.supervisor.parameters()), lr=lr
        )
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr * 0.5)
        mse = nn.MSELoss()
        bce = nn.BCEWithLogitsLoss()

        losses = {"ae": [], "sup": [], "g": [], "d": []}

        ae_epochs = epochs // 3
        print(f"  Phase 1: Autoencoder pre-training ({ae_epochs} epochs)")
        for epoch in range(ae_epochs):
            epoch_loss = 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                h = self.embedder(batch)
                x_hat = self.recovery(h)
                loss = mse(x_hat, batch)
                opt_ae.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.embedder.parameters()) + list(self.recovery.parameters()), 1.0)
                opt_ae.step()
                epoch_loss += loss.item()
            losses["ae"].append(epoch_loss / len(loader))

        sup_epochs = epochs // 3
        print(f"  Phase 2: Supervisor pre-training ({sup_epochs} epochs)")
        for epoch in range(sup_epochs):
            epoch_loss = 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                with torch.no_grad():
                    h = self.embedder(batch)
                h_sup = self.supervisor(h[:, :-1, :])
                loss = mse(h_sup, h[:, 1:, :])
                opt_sup.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.supervisor.parameters(), 1.0)
                opt_sup.step()
                epoch_loss += loss.item()
            losses["sup"].append(epoch_loss / len(loader))

        joint_epochs = epochs - ae_epochs - sup_epochs
        print(f"  Phase 3: Joint adversarial training ({joint_epochs} epochs)")
        for epoch in range(joint_epochs):
            g_loss_total, d_loss_total = 0, 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                B = batch.shape[0]
                noise = self._random_noise(B, self.seq_len)

                # --- Train Generator ---
                h_real = self.embedder(batch)
                h_fake_raw = self.generator(noise)
                h_fake = self.supervisor(h_fake_raw)
                x_fake = self.recovery(h_fake)

                d_fake = self.discriminator(h_fake)
                g_adv_loss = bce(d_fake, torch.ones_like(d_fake))

                h_sup_real = self.supervisor(h_real[:, :-1, :])
                g_sup_loss = mse(h_sup_real, h_real[:, 1:, :].detach())

                # Moment matching on mean and variance
                g_moment = (
                    torch.abs(x_fake.mean(dim=0) - batch.mean(dim=0)).mean() +
                    torch.abs(x_fake.std(dim=0) - batch.std(dim=0)).mean()
                )

                g_loss = g_adv_loss + 10 * g_sup_loss + 1.0 * g_moment
                opt_g.zero_grad()
                g_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.generator.parameters()) + list(self.supervisor.parameters()), 1.0)
                opt_g.step()
                g_loss_total += g_loss.item()

                # --- Train Discriminator ---
                with torch.no_grad():
                    h_real_d = self.embedder(batch)
                    h_fake_d = self.supervisor(self.generator(self._random_noise(B, self.seq_len)))

                d_real = self.discriminator(h_real_d)
                d_fake = self.discriminator(h_fake_d)
                d_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))

                # Gradient penalty for stability
                gp = self._gradient_penalty(h_real_d.detach(), h_fake_d.detach())
                d_loss = d_loss + gp_weight * gp

                opt_d.zero_grad()
                d_loss.backward()
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                opt_d.step()
                d_loss_total += d_loss.item()

            losses["g"].append(g_loss_total / len(loader))
            losses["d"].append(d_loss_total / len(loader))
            if (epoch + 1) % 20 == 0:
                print(f"  Joint Epoch {epoch+1}/{joint_epochs} | G={losses['g'][-1]:.4f} D={losses['d'][-1]:.4f}")

        self.is_trained = True
        return losses

    @torch.no_grad()
    def generate(self, n_samples: int, seq_len: int | None = None, **kwargs) -> np.ndarray:
        if seq_len is None:
            seq_len = self.seq_len

        self.generator.eval()
        self.supervisor.eval()
        self.recovery.eval()

        noise = self._random_noise(n_samples, seq_len)
        h_fake = self.supervisor(self.generator(noise))
        x_fake = self.recovery(h_fake)
        return x_fake.cpu().numpy()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "embedder": self.embedder.state_dict(),
            "recovery": self.recovery.state_dict(),
            "generator": self.generator.state_dict(),
            "supervisor": self.supervisor.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "config": {
                "n_features": self.n_features,
                "seq_len": self.seq_len,
                "hidden_dim": self.hidden_dim,
                "latent_dim": self.latent_dim,
            },
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.embedder.load_state_dict(ckpt["embedder"])
        self.recovery.load_state_dict(ckpt["recovery"])
        self.generator.load_state_dict(ckpt["generator"])
        self.supervisor.load_state_dict(ckpt["supervisor"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.is_trained = True


if __name__ == "__main__":
    from src.utils.config import DEFAULT_DEVICE
    parser = argparse.ArgumentParser(description="TimeGAN for financial time series")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/timegan.pt")
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(os.path.dirname(__file__), "..", "..", "data")

    if args.train:
        windows = np.load(os.path.join(data_dir, "windows.npy"))
        print(f"Training TimeGAN on {windows.shape}")
        model = TimeGANModel(
            n_features=windows.shape[2], seq_len=windows.shape[1],
            device=DEFAULT_DEVICE,
        )
        model.train(windows, epochs=args.epochs)
        model.save(args.checkpoint)

    if args.generate:
        windows = np.load(os.path.join(data_dir, "windows.npy"))
        model = TimeGANModel(
            n_features=windows.shape[2], seq_len=windows.shape[1],
            device=DEFAULT_DEVICE,
        )
        model.load(args.checkpoint)
        synthetic = model.generate(args.n_samples)
        np.save(os.path.join(data_dir, "generated_timegan.npy"), synthetic)
        print(f"Generated {synthetic.shape}")
