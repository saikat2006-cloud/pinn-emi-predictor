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

C = 3e8

X_MIN = torch.tensor([-0.05, -0.05, 0.001, 1e8 ], dtype=torch.float32)
X_MAX = torch.tensor([ 0.05,  0.05, 0.015, 3e9 ], dtype=torch.float32)

Y_SCALE = 0.05   # typical E-field magnitude in our domain

def normalize(x):
    mn = X_MIN.to(x.device)
    mx = X_MAX.to(x.device)
    return 2.0 * (x - mn) / (mx - mn) - 1.0

def normalize_y(y):
    return y / Y_SCALE

def denormalize_y(y_norm):
    return y_norm * Y_SCALE

def data_loss(model, X, y):
    X_norm = normalize(X)
    y_pred = model(X_norm)
    y_norm = normalize_y(y)
    return nn.functional.mse_loss(y_pred, y_norm)

def helmholtz_residual(model, coll_pts):
    x_norm = normalize(coll_pts).detach().requires_grad_(True)
    E = model(x_norm)

    grad1 = torch.autograd.grad(
        E, x_norm,
        grad_outputs=torch.ones_like(E),
        create_graph=True,
    )[0]

    laplacian = torch.zeros(E.shape[0], device=x_norm.device)
    for i in range(3):
        gi = grad1[:, i]
        grad2_i = torch.autograd.grad(
            gi, x_norm,
            grad_outputs=torch.ones_like(gi),
            create_graph=True,
            retain_graph=True,
        )[0][:, i]
        laplacian = laplacian + grad2_i

    freq_norm = x_norm[:, 3].detach()
    freq_phys = 0.5 * (freq_norm + 1.0) * (X_MAX[3] - X_MIN[3]) + X_MIN[3]
    k_phys    = 2 * torch.pi * freq_phys / C
    k_norm    = k_phys / 20.0

    residual = laplacian + k_norm**2 * E
    return (residual**2).mean()

def boundary_loss(model, bnd_pts):
    bnd_norm = normalize(bnd_pts)
    E_bnd    = model(bnd_norm)
    return (E_bnd**2).mean()

def total_loss(model, X_data, y_data, coll_pts, bnd_pts,
               w_data=1.0, w_phys=0.0, w_bnd=0.0):
    L_data  = data_loss(model, X_data, y_data)
    L_phys  = helmholtz_residual(model, coll_pts)
    L_bnd   = boundary_loss(model, bnd_pts)
    L_total = w_data * L_data + w_phys * L_phys + w_bnd * L_bnd
    return L_total, L_data, L_phys, L_bnd