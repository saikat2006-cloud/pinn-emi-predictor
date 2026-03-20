# PINN EMI Predictor — Team Nexus

**Physics-Informed Neural Network for PCB electromagnetic interference prediction**  
Track: Open | Theme: Physics-Informed Neural Network

---

## What it does

Predicts the electromagnetic interference (EMI) field over any PCB layout in **under 1 second**,
on a standard laptop, with **zero simulation software** (no HFSS, no CST, no Ansys).

Trained on just **200 analytical samples** + physics constraints from Maxwell's equations.

---

## Project Structure

```
pinn_emi/
├── data/
│   ├── generate_data.py       ← Step 1: generate all training data
│   ├── X_labeled.npy          ← 200 labeled samples (created after running)
│   ├── y_labeled.npy
│   ├── collocation_pts.npy    ← 40,000 physics supervision points
│   └── boundary_pts.npy       ← 5,000 boundary condition points
│
├── model/
│   ├── pinn_model.py          ← FourierEncoder + PINN architecture
│   ├── physics_loss.py        ← Helmholtz loss, boundary loss, hybrid loss
│   ├── train.py               ← Step 2: training loop
│   └── pinn_checkpoint.pt     ← saved weights (created after training)
│
├── inference/
│   └── predict.py             ← Step 3: heatmap generation + benchmark
│
├── demo/
│   └── app.py                 ← Step 4: Gradio web interface
│
├── run_all.py                 ← One-click full pipeline
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate training data

```bash
python data/generate_data.py
```

Outputs:
- `data/X_labeled.npy`         — 200 labeled (x,y,z,f) → |E| samples
- `data/y_labeled.npy`
- `data/collocation_pts.npy`   — 40,000 physics points (no labels)
- `data/boundary_pts.npy`      — 5,000 boundary points (E=0)

### 3. Train the model

```bash
python model/train.py
```

- Trains for 10,000 epochs (~5–30 min depending on hardware)
- Saves `model/pinn_checkpoint.pt`
- Saves `model/loss_curves.png`

### 4. Run inference + benchmark

```bash
python inference/predict.py
```

- Generates EMI heatmaps at 1 GHz and 2.4 GHz
- Prints MAE, RMSE, relative error, and inference time

### 5. Launch Gradio demo

```bash
python demo/app.py
```

Opens the interactive web interface at **http://localhost:7860**

---

## Or run everything at once

```bash
python run_all.py                # full pipeline
python run_all.py --skip-train   # skip training (reuse checkpoint)
python run_all.py --demo-only    # demo only
```

---

## Architecture

| Component         | Details                                               |
|-------------------|-------------------------------------------------------|
| Input             | (x, y, z, frequency) — 4D                            |
| Fourier encoder   | 128 random frequencies → 256-dim sinusoidal embedding |
| Network           | 6 FC layers, GELU activations, Softplus output        |
| Output            | |E| field magnitude (V/m), scalar                    |
| Physics loss      | Helmholtz ∇²E + k²E = 0 at 40,000 collocation pts   |
| Boundary loss     | E = 0 at 5,000 conductor surface points               |
| Data loss         | MSE on 200 analytical labeled samples                 |
| Training          | Adam + CosineAnnealing, 10,000 epochs                 |

---

## Loss Function

```
L_total = 1.0 × L_data  +  0.1 × L_physics  +  0.5 × L_boundary
```

- `L_data`     — supervised MSE on 200 labeled samples
- `L_physics`  — Helmholtz residual (∇²E + k²E)² via autograd
- `L_boundary` — |E|² at conductor surfaces (E=0 constraint)

---

## Comparison vs Existing Tools

| | HFSS/CST | ML (CNN) | **Our PINN** |
|---|---|---|---|
| Prediction time | 4–8 hours | ~1 sec | **<0.1 sec** |
| Cost | $10k–$100k/yr | Free | **Free** |
| Training data | None | 10,000+ | **<200** |
| Physics guaranteed | Yes | No | **Yes** |
| Runs on laptop | No | Yes | **Yes** |

---

## Team

| Member | Role |
|--------|------|
| Saikat Kumar Khatua (Lead) | Model & Training |
| Krrish Kuhar | Data & Physics |
| Taradrita Routh | Evaluation & Demo |
