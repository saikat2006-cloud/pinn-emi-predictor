"""
data/generate_data.py
=====================
Generates all training data for the PINN EMI Predictor:
  - 200 labeled analytical samples  (closed-form microstrip E-field)
  - 40,000 collocation points        (physics supervision, no labels)
  -  5,000 boundary points           (E=0 at conductor surface)

Run:
    python data/generate_data.py
"""

import numpy as np
import os

np.random.seed(42)
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Analytical E-field for a microstrip trace ──────────────────────────────
def microstrip_efield(x, y, z, f, W=0.001, h=0.0016, I=0.1):
    """
    Closed-form approximation of the dominant E-field component
    radiated by a microstrip trace carrying current I.

    Parameters
    ----------
    x, y, z : float  spatial coordinates (metres)
    f        : float  frequency (Hz)
    W        : float  trace width (metres)
    h        : float  substrate height (metres)
    I        : float  trace current (amperes)

    Returns
    -------
    float  |E_z| magnitude (V/m)
    """
    k  = 2 * np.pi * f / 3e8            # wavenumber
    r  = np.sqrt(x**2 + y**2 + z**2) + 1e-9   # avoid divide-by-zero
    Ez = (I / (2 * np.pi * r)) * np.exp(-1j * k * r)
    return float(np.abs(Ez))


# ── 1. Labeled samples ─────────────────────────────────────────────────────
N_labeled = 200

X_labeled = np.zeros((N_labeled, 4))
X_labeled[:, 0] = np.random.uniform(-0.05, 0.05, N_labeled)   # x
X_labeled[:, 1] = np.random.uniform(-0.05, 0.05, N_labeled)   # y
X_labeled[:, 2] = np.random.uniform(0.001, 0.015, N_labeled)  # z (above PCB)
X_labeled[:, 3] = np.random.uniform(1e8,   3e9,   N_labeled)  # frequency

y_labeled = np.array([
    microstrip_efield(X_labeled[i, 0], X_labeled[i, 1],
                      X_labeled[i, 2], X_labeled[i, 3])
    for i in range(N_labeled)
], dtype=np.float32)

X_labeled = X_labeled.astype(np.float32)

np.save(os.path.join(SAVE_DIR, "X_labeled.npy"), X_labeled)
np.save(os.path.join(SAVE_DIR, "y_labeled.npy"), y_labeled)
print(f"[1/3] Labeled samples  : {X_labeled.shape}  saved.")


# ── 2. Collocation points (physics supervision) ────────────────────────────
N_coll = 40_000

coll_pts = np.zeros((N_coll, 4), dtype=np.float32)
coll_pts[:, 0] = np.random.uniform(-0.05, 0.05, N_coll)    # x
coll_pts[:, 1] = np.random.uniform(-0.05, 0.05, N_coll)    # y
coll_pts[:, 2] = np.random.uniform(0.001, 0.015, N_coll)   # z
coll_pts[:, 3] = np.random.uniform(1e8,   3e9,   N_coll)   # frequency

np.save(os.path.join(SAVE_DIR, "collocation_pts.npy"), coll_pts)
print(f"[2/3] Collocation pts  : {coll_pts.shape}  saved.")


# ── 3. Boundary points (E = 0 at conductor surface) ───────────────────────
N_bnd = 5_000

bnd_xy  = np.random.uniform(-0.05, 0.05, (N_bnd, 2))
bnd_z   = np.zeros((N_bnd, 1))
bnd_f   = np.random.uniform(1e8, 3e9, (N_bnd, 1))
bnd_pts = np.hstack([bnd_xy, bnd_z, bnd_f]).astype(np.float32)

np.save(os.path.join(SAVE_DIR, "boundary_pts.npy"), bnd_pts)
print(f"[3/3] Boundary pts     : {bnd_pts.shape}  saved.")

print("\nAll data files saved to:", SAVE_DIR)
print("  X_labeled.npy       (200, 4)")
print("  y_labeled.npy       (200,)")
print("  collocation_pts.npy (40000, 4)")
print("  boundary_pts.npy    (5000, 4)")
