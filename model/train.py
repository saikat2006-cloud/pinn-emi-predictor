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
import os, sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pinn_model import PINN
from model.physics_loss import total_loss

EPOCHS_WARMUP  = 5_000
EPOCHS_PHYSICS = 0
EPOCHS_TOTAL   = EPOCHS_WARMUP + EPOCHS_PHYSICS

LR          = 1e-3
COLL_BATCH  = 500
BND_BATCH   = 100

W_DATA_1, W_PHYS_1, W_BND_1 = 1.0, 0.0, 0.0
W_DATA_2, W_PHYS_2, W_BND_2 = 1.0, 0.0, 0.0

LOG_EVERY = 500
SAVE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(os.path.dirname(SAVE_DIR), "data")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

X_lab = torch.tensor(np.load(os.path.join(DATA_DIR, "X_labeled.npy")),       dtype=torch.float32).to(device)
y_lab = torch.tensor(np.load(os.path.join(DATA_DIR, "y_labeled.npy")),       dtype=torch.float32).to(device)
coll  = torch.tensor(np.load(os.path.join(DATA_DIR, "collocation_pts.npy")), dtype=torch.float32).to(device)
bnd   = torch.tensor(np.load(os.path.join(DATA_DIR, "boundary_pts.npy")),    dtype=torch.float32).to(device)

print(f"y labels — min:{y_lab.min():.5f}  max:{y_lab.max():.5f}  mean:{y_lab.mean():.5f}")

model = PINN().to(device)
opt   = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_TOTAL, eta_min=1e-5)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

history = {"total":[], "data":[], "phys":[], "bnd":[]}

print(f"\nPhase 1 — warmup data only  : {EPOCHS_WARMUP} epochs")
print(f"Phase 2 — data + physics    : {EPOCHS_PHYSICS} epochs\n")

for epoch in range(1, EPOCHS_TOTAL + 1):
    if epoch <= EPOCHS_WARMUP:
        wd, wp, wb = W_DATA_1, W_PHYS_1, W_BND_1
    else:
        wd, wp, wb = W_DATA_2, W_PHYS_2, W_BND_2

    ci = torch.randperm(len(coll), device=device)[:COLL_BATCH]
    bi = torch.randperm(len(bnd),  device=device)[:BND_BATCH]

    opt.zero_grad()
    L_total, L_data, L_phys, L_bnd = total_loss(
        model, X_lab, y_lab, coll[ci], bnd[bi],
        w_data=wd, w_phys=wp, w_bnd=wb,
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
        phase = "WARMUP " if epoch <= EPOCHS_WARMUP else "PHYSICS"
        print(
            f"[{phase}] Epoch {epoch:>5} | "
            f"total={L_total.item():.4e} | "
            f"data={L_data.item():.4e} | "
            f"phys={L_phys.item():.4e} | "
            f"bnd={L_bnd.item():.4e}"
        )

ckpt = os.path.join(SAVE_DIR, "pinn_checkpoint.pt")
torch.save({"model_state": model.state_dict(), "epoch": EPOCHS_TOTAL, "history": history}, ckpt)
print(f"\nCheckpoint saved: {ckpt}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogy(history["data"],  label="L_data")
ax.semilogy(history["phys"],  label="L_physics", linestyle="--")
ax.semilogy(history["bnd"],   label="L_boundary", linestyle="--")
ax.axvline(EPOCHS_WARMUP, color="gray", linestyle=":", label="physics ON")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "loss_curves.png"), dpi=150)
plt.close()
print("Loss plot saved.\nDone.")