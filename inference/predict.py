"""
inference/predict.py
====================
Loads the trained PINN checkpoint and provides:

  predict_heatmap()  – sweeps a 100×100 spatial grid above the PCB
  save_heatmap()     – renders and saves the inferno colormap PNG
  benchmark()        – measures MAE vs analytical ground-truth + timing

Usage:
    python inference/predict.py
"""

import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pinn_model import PINN

CKPT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "model", "pinn_checkpoint.pt")
C = 3e8


# ── Load model ─────────────────────────────────────────────────────────────
def load_model(ckpt_path: str = CKPT_PATH) -> tuple:
    """Load and return (model, device)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = PINN().to(device)
    ckpt   = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model loaded from {ckpt_path}  (epoch {ckpt['epoch']})")
    return model, device


# ── Predict heatmap ────────────────────────────────────────────────────────
def predict_heatmap(model: torch.nn.Module,
                    device: torch.device,
                    freq_hz: float   = 2.4e9,
                    z_height: float  = 0.005,
                    n: int           = 100,
                    xlim: tuple      = (-0.05, 0.05),
                    ylim: tuple      = (-0.05, 0.05)) -> tuple:
    """
    Predicts the |E| field over a 2-D grid above the PCB.

    Parameters
    ----------
    freq_hz   : frequency in Hz  (e.g. 2.4e9 for Wi-Fi band)
    z_height  : height above PCB in metres
    n         : grid resolution (n×n points)

    Returns
    -------
    (heatmap, xx, yy)  where heatmap is (n, n) array of |E| in V/m
    """
    xs = np.linspace(*xlim, n)
    ys = np.linspace(*ylim, n)
    xx, yy = np.meshgrid(xs, ys)

    pts = np.stack([
        xx.ravel(),
        yy.ravel(),
        np.full(n * n, z_height),
        np.full(n * n, freq_hz),
    ], axis=1).astype(np.float32)

    from model.physics_loss import normalize, denormalize_y
    t      = torch.tensor(pts, dtype=torch.float32).to(device)
    t_norm = normalize(t)

    t0 = time.perf_counter()
    with torch.no_grad():
        E = denormalize_y(model(t_norm)).cpu().numpy()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    heatmap = E.reshape(n, n)
    print(f"Inference: {n}×{n}={n*n} points in {elapsed_ms:.1f} ms")
    return heatmap, xx, yy, elapsed_ms


# ── Render and save ────────────────────────────────────────────────────────
def save_heatmap(heatmap: np.ndarray,
                 xx: np.ndarray,
                 yy: np.ndarray,
                 freq_hz: float,
                 z_height: float,
                 out_path: str = "emi_heatmap.png") -> None:
    """Render inferno colormap heatmap and save to PNG."""
    fig, ax = plt.subplots(figsize=(6, 5))

    extent = [xx.min() * 1000, xx.max() * 1000,
              yy.min() * 1000, yy.max() * 1000]   # convert to mm for display

    im = ax.imshow(heatmap, cmap="inferno", origin="lower",
                   extent=extent, aspect="equal")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|E| (V/m)", fontsize=10)

    freq_label = f"{freq_hz/1e9:.2f} GHz" if freq_hz >= 1e9 else f"{freq_hz/1e6:.0f} MHz"
    ax.set_title(f"EMI Field Prediction  |  {freq_label}  |  z = {z_height*1000:.1f} mm",
                 fontsize=11)
    ax.set_xlabel("x (mm)", fontsize=9)
    ax.set_ylabel("y (mm)", fontsize=9)

    # Colour legend annotation
    ax.text(0.02, 0.97,
            "White/Yellow = High EMI (fail zone)\nDark Purple = Safe zone",
            transform=ax.transAxes, fontsize=7,
            verticalalignment="top", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved: {out_path}")
    plt.close()


# ── Benchmark against analytical ground-truth ─────────────────────────────
def microstrip_efield_batch(pts: np.ndarray) -> np.ndarray:
    """Vectorised analytical E-field for benchmarking."""
    x, y, z, f = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    k  = 2 * np.pi * f / C
    r  = np.sqrt(x**2 + y**2 + z**2) + 1e-9
    Ez = (0.1 / (2 * np.pi * r)) * np.exp(-1j * k * r)
    return np.abs(Ez).astype(np.float32)


def benchmark(model: torch.nn.Module, device: torch.device, n: int = 500) -> None:
    """Compare PINN predictions against analytical ground truth."""
    np.random.seed(99)
    test_pts = np.zeros((n, 4), dtype=np.float32)
    test_pts[:, 0] = np.random.uniform(-0.05, 0.05, n)
    test_pts[:, 1] = np.random.uniform(-0.05, 0.05, n)
    test_pts[:, 2] = np.random.uniform(0.001, 0.015, n)
    test_pts[:, 3] = np.random.uniform(1e8,   3e9,   n)

    y_true = microstrip_efield_batch(test_pts)

    from model.physics_loss import normalize
    t_in   = torch.tensor(test_pts).to(device)
    t_norm = normalize(t_in)
    t0     = time.perf_counter()
    with torch.no_grad():
        y_pred = model(t_norm).cpu().numpy()
    elapsed = (time.perf_counter() - t0) * 1000

    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    rel  = mae / (np.mean(np.abs(y_true)) + 1e-9) * 100

    print("\n── Benchmark Results ──────────────────────────────")
    print(f"  Test points : {n}")
    print(f"  MAE         : {mae:.4f} V/m")
    print(f"  RMSE        : {rmse:.4f} V/m")
    print(f"  Rel. error  : {rel:.2f} %")
    print(f"  Inference   : {elapsed:.1f} ms  ({elapsed/n*1000:.2f} µs/point)")
    print("────────────────────────────────────────────────────\n")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, device = load_model()

    # 2.4 GHz Wi-Fi band heatmap
    hm, xx, yy, _ = predict_heatmap(model, device, freq_hz=2.4e9, z_height=0.005)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emi_heatmap_2_4GHz.png")
    save_heatmap(hm, xx, yy, freq_hz=2.4e9, z_height=0.005, out_path=out)

    # 1 GHz heatmap
    hm2, xx2, yy2, _ = predict_heatmap(model, device, freq_hz=1e9, z_height=0.005)
    out2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emi_heatmap_1GHz.png")
    save_heatmap(hm2, xx2, yy2, freq_hz=1e9, z_height=0.005, out_path=out2)

    # Benchmark
    benchmark(model, device)
