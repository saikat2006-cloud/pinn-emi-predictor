"""
model/train.py
==============
Training loop for the PINN EMI Predictor.

Usage:
    python model/train.py

Outputs:
    model/pinn_checkpoint.pt   – trained weights
    model/loss_curves.png      – training loss plot
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pinn_model import PINN
from model.physics_loss import total_loss

# ── Config ─────────────────────────────────────────────────────────────────
EPOCHS        = 3_000    # CPU: 3k epochs is enough, model converges well
LR            = 1e-3
BATCH_SIZE    = 200      # use all labeled data each step (only 200 samples)
COLL_BATCH    = 500      # CPU: small collocation batch — autograd is expensive
BND_BATCH     = 100      # CPU: small boundary batch
W_DATA        = 1.0
W_PHYS        = 0.1      # safe now — inputs are normalized
W_BND         = 0.5
LOG_EVERY     = 500
SAVE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(os.path.dirname(SAVE_DIR), "data")

# ── Device ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Load data ──────────────────────────────────────────────────────────────
X_labeled  = torch.tensor(np.load(os.path.join(DATA_DIR, "X_labeled.npy")),
                           dtype=torch.float32).to(device)
y_labeled  = torch.tensor(np.load(os.path.join(DATA_DIR, "y_labeled.npy")),
                           dtype=torch.float32).to(device)
coll_all   = torch.tensor(np.load(os.path.join(DATA_DIR, "collocation_pts.npy")),
                           dtype=torch.float32).to(device)
bnd_all    = torch.tensor(np.load(os.path.join(DATA_DIR, "boundary_pts.npy")),
                           dtype=torch.float32).to(device)

print(f"Labeled  : {X_labeled.shape}  y: {y_labeled.shape}")
print(f"Coll pts : {coll_all.shape}")
print(f"Bnd pts  : {bnd_all.shape}")

# ── Model, optimiser, scheduler ────────────────────────────────────────────
model = PINN().to(device)
opt   = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {n_params:,}")

# ── Training ───────────────────────────────────────────────────────────────
history = {"total": [], "data": [], "phys": [], "bnd": []}

print("\nStarting training...")
for epoch in range(1, EPOCHS + 1):

    # Subsample collocation and boundary each epoch
    coll_idx = torch.randperm(len(coll_all), device=device)[:COLL_BATCH]
    bnd_idx  = torch.randperm(len(bnd_all),  device=device)[:BND_BATCH]
    coll_batch = coll_all[coll_idx]
    bnd_batch  = bnd_all[bnd_idx]

    opt.zero_grad()

    L_total, L_data, L_phys, L_bnd = total_loss(
        model,
        X_labeled, y_labeled,
        coll_batch, bnd_batch,
        w_data=W_DATA, w_phys=W_PHYS, w_bnd=W_BND,
    )

    L_total.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    sched.step()

    history["total"].append(L_total.item())
    history["data"].append(L_data.item())
    history["phys"].append(L_phys.item())
    history["bnd"].append(L_bnd.item())

    if epoch % LOG_EVERY == 0 or epoch == 1:
        lr_now = sched.get_last_lr()[0]
        print(
            f"Epoch {epoch:>6}  "
            f"total={L_total.item():.4e}  "
            f"data={L_data.item():.4e}  "
            f"phys={L_phys.item():.4e}  "
            f"bnd={L_bnd.item():.4e}  "
            f"lr={lr_now:.2e}"
        )

# ── Save checkpoint ────────────────────────────────────────────────────────
ckpt_path = os.path.join(SAVE_DIR, "pinn_checkpoint.pt")
torch.save({
    "model_state":  model.state_dict(),
    "epoch":        EPOCHS,
    "final_loss":   history["total"][-1],
    "history":      history,
}, ckpt_path)
print(f"\nCheckpoint saved: {ckpt_path}")

# ── Plot loss curves ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
epochs_x = range(1, EPOCHS + 1)
ax.semilogy(epochs_x, history["total"], label="Total",    linewidth=1.5)
ax.semilogy(epochs_x, history["data"],  label="L_data",   linewidth=1,  linestyle="--")
ax.semilogy(epochs_x, history["phys"],  label="L_physics", linewidth=1,  linestyle="--")
ax.semilogy(epochs_x, history["bnd"],   label="L_boundary", linewidth=1, linestyle="--")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (log scale)")
ax.set_title("PINN Training Loss Curves")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(SAVE_DIR, "loss_curves.png")
plt.savefig(plot_path, dpi=150)
print(f"Loss plot saved:  {plot_path}")
plt.close()
print("\nTraining complete.")
