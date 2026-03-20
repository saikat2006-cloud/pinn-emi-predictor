"""
model/physics_loss.py
=====================
Three loss components for the hybrid PINN objective.

ROOT CAUSE FIX: raw frequency values (1e8-3e9 Hz) make k^2 enormous,
which explodes the Helmholtz residual. Solution: normalize all inputs
to [-1, 1] before the network, and rescale the physics equation accordingly.

  L_data    - supervised MSE on 200 labeled analytical samples
  L_physics - normalized Helmholtz residual at collocation pts
  L_boundary- E=0 at conductor boundary points
"""

import torch
import torch.nn as nn

C = 3e8  # speed of light m/s

# Input normalization ranges — must match data/generate_data.py
X_MIN = torch.tensor([-0.05, -0.05, 0.001, 1e8], dtype=torch.float32)
X_MAX = torch.tensor([ 0.05,  0.05, 0.015, 3e9], dtype=torch.float32)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Scale physical inputs to [-1, 1]."""
    mn = X_MIN.to(x.device)
    mx = X_MAX.to(x.device)
    return 2.0 * (x - mn) / (mx - mn) - 1.0


def data_loss(model, X, y):
    X_norm = normalize(X)
    y_pred = model(X_norm)
    return nn.functional.mse_loss(y_pred, y)


def helmholtz_residual(model, coll_pts):
    """
    Helmholtz residual in normalized coordinates.
    Keeps all numbers in [-1, 1] range so k^2 * E never explodes.
    """
    x_norm = normalize(coll_pts).detach().requires_grad_(True)

    E = model(x_norm)  # (N,)

    # First-order gradients in normalized space
    dE = torch.autograd.grad(
        outputs=E,
        inputs=x_norm,
        grad_outputs=torch.ones_like(E),
        create_graph=True,
    )[0]  # (N, 4)

    # Second-order — sum spatial dims in one backward pass
    spatial_sum = dE[:, 0] + dE[:, 1] + dE[:, 2]

    d2E = torch.autograd.grad(
        outputs=spatial_sum,
        inputs=x_norm,
        grad_outputs=torch.ones_like(spatial_sum),
        create_graph=True,
    )[0]  # (N, 4)

    laplacian = d2E[:, 0] + d2E[:, 1] + d2E[:, 2]

    # k in normalized space: k_phys / scale, keeps magnitude ~1
    # scale = 2 / (x_max - x_min) for spatial dims
    L_scale = 0.1   # physical domain size ~0.1 m
    freq_norm = x_norm[:, 3]
    freq_phys = 0.5 * (freq_norm + 1.0) * (3e9 - 1e8) + 1e8
    k_phys = 2 * torch.pi * freq_phys / C   # ~2 to ~63 rad/m
    k_norm = k_phys * L_scale               # normalized: ~0.2 to ~6

    residual = laplacian + k_norm**2 * E
    return (residual**2).mean()


def boundary_loss(model, bnd_pts):
    bnd_norm = normalize(bnd_pts)
    E_bnd = model(bnd_norm)
    return (E_bnd**2).mean()


def total_loss(model, X_data, y_data, coll_pts, bnd_pts,
               w_data=1.0, w_phys=0.1, w_bnd=0.5):
    L_data = data_loss(model, X_data, y_data)
    L_phys = helmholtz_residual(model, coll_pts)
    L_bnd  = boundary_loss(model, bnd_pts)
    L_total = w_data * L_data + w_phys * L_phys + w_bnd * L_bnd
    return L_total, L_data, L_phys, L_bnd
