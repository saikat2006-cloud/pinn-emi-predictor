"""
model/pinn_model.py
===================
Defines:
  FourierEncoder  – maps (x,y,z,f) → 256-dim sinusoidal features
  PINN            – 6-layer network that predicts |E| (V/m)
"""

import torch
import torch.nn as nn
import math


class FourierEncoder(nn.Module):
    """
    Random Fourier Feature encoder.

    Maps a 4-D input (x, y, z, frequency) into a 128-D sinusoidal
    embedding so the network can represent the high-frequency
    oscillations of GHz EM fields without spectral bias.

    CPU-optimised: n_freq=64 → output dim = 128 (was 256).

    Parameters
    ----------
    in_dim  : input dimension (4 for x,y,z,f)
    n_freq  : number of random frequency components (output = 2*n_freq)
    scale   : std-dev of the random Gaussian matrix B
    """

    def __init__(self, in_dim: int = 4, n_freq: int = 64, scale: float = 10.0):
        super().__init__()
        B = torch.randn(in_dim, n_freq) * scale
        self.register_buffer("B", B)          # frozen, not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, 4)
        returns : (batch, 128)
        """
        proj = 2 * math.pi * (x @ self.B)     # (batch, 64)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for EMI prediction.
    CPU-optimised: smaller encoder (128-dim) and shallower network (4 layers).

    Architecture
    ------------
    FourierEncoder(128) →
    Linear(128,128) → GELU →
    Linear(128,128) → GELU →
    Linear(128, 64) → GELU →
    Linear( 64, 32) → GELU →
    Linear( 32,  1) → Softplus     ← ensures E ≥ 0
    """

    def __init__(self):
        super().__init__()
        self.encoder = FourierEncoder(in_dim=4, n_freq=64, scale=10.0)

        self.net = nn.Sequential(
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128,  64), nn.GELU(),
            nn.Linear( 64,  32), nn.GELU(),
            nn.Linear( 32,   1), nn.Softplus(),   # E-field magnitude ≥ 0
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x       : (batch, 4)   – [x_pos, y_pos, z_pos, frequency]
        returns : (batch,)     – predicted |E| in V/m
        """
        emb = self.encoder(x)
        return self.net(emb).squeeze(-1)
