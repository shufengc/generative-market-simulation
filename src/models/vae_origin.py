"""
Variational Autoencoder (VAE) for financial time series generation.

Uses a GRU encoder/decoder with a learned latent space.
The KL divergence term is annealed during training to avoid posterior collapse.
"""

from __future__ import annotations

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_model import BaseGenerativeModel


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, n_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.1)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        _, h = self.gru(x)
        h = h[-1]
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, seq_len: int, n_layers: int = 2):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True, dropout=0.1)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor):
        h = self.fc(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        h, _ = self.gru(h)
        return self.out(h)


class FinancialVAE(BaseGenerativeModel):
    """VAE for generating financial return sequences."""

    def __init__(
        self,
        n_features: int = 18,
        seq_len: int = 60,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        n_layers: int = 2,
        device: str = "cpu",
    ):
        super().__init__(name="VAE", device=device)
        self.n_features = n_features
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        self.encoder = Encoder(n_features, hidden_dim, latent_dim, n_layers).to(self.device)
        self.decoder = Decoder(latent_dim, hidden_dim, n_features, seq_len, n_layers).to(self.device)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def _vae_loss(self, x: torch.Tensor, x_recon: torch.Tensor,
                  mu: torch.Tensor, logvar: torch.Tensor, kl_weight: float = 1.0):
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_weight * kl_loss, recon_loss.item(), kl_loss.item()

    def train(self, data: np.ndarray, epochs: int = 200, batch_size: int = 64,
              lr: float = 1e-3, **kwargs) -> dict:
        x_tensor = torch.tensor(data, dtype=torch.float32)
        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(-1)

        dataset = TensorDataset(x_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        all_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.encoder.train(True)
        self.decoder.train(True)
        optimizer = torch.optim.Adam(all_params, lr=lr)

        losses = {"total": [], "recon": [], "kl": []}
        for epoch in range(epochs):
            total_loss, total_recon, total_kl = 0, 0, 0
            kl_weight = min(1.0, epoch / max(epochs * 0.3, 1))

            for (batch,) in loader:
                batch = batch.to(self.device)
                x_recon, mu, logvar = self.forward(batch)
                loss, rl, kl = self._vae_loss(batch, x_recon, mu, logvar, kl_weight)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()

                total_loss += loss.item()
                total_recon += rl
                total_kl += kl

            n = max(len(loader), 1)
            losses["total"].append(total_loss / n)
            losses["recon"].append(total_recon / n)
            losses["kl"].append(total_kl / n)

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs} | loss={losses['total'][-1]:.4f} "
                      f"recon={losses['recon'][-1]:.4f} kl={losses['kl'][-1]:.4f}")

        self.is_trained = True
        return losses

    @torch.no_grad()
    def generate(self, n_samples: int, seq_len: int | None = None, **kwargs) -> np.ndarray:
        self.decoder.eval()
        if seq_len is None:
            seq_len = self.seq_len

        z = torch.randn(n_samples, self.latent_dim, device=self.device)
        x_gen = self.decoder(z)
        return x_gen.cpu().numpy()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "config": {
                "n_features": self.n_features,
                "seq_len": self.seq_len,
                "latent_dim": self.latent_dim,
            },
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.decoder.load_state_dict(ckpt["decoder"])
        self.is_trained = True


if __name__ == "__main__":
    from src.utils.config import DEFAULT_DEVICE
    parser = argparse.ArgumentParser(description="VAE for financial time series")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/vae.pt")
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(os.path.dirname(__file__), "..", "..", "data")

    if args.train:
        windows = np.load(os.path.join(data_dir, "windows.npy"))
        print(f"Training VAE on {windows.shape}")
        model = FinancialVAE(
            n_features=windows.shape[2], seq_len=windows.shape[1],
            device=DEFAULT_DEVICE,
        )
        model.train(windows, epochs=args.epochs)
        model.save(args.checkpoint)

    if args.generate:
        windows = np.load(os.path.join(data_dir, "windows.npy"))
        model = FinancialVAE(
            n_features=windows.shape[2], seq_len=windows.shape[1],
            device=DEFAULT_DEVICE,
        )
        model.load(args.checkpoint)
        synthetic = model.generate(args.n_samples)
        np.save(os.path.join(data_dir, "generated_vae.npy"), synthetic)
        print(f"Generated {synthetic.shape}")