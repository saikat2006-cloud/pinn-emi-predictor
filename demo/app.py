"""
demo/app.py
===========
Gradio web interface for the PINN EMI Predictor.

Usage:
    python demo/app.py

Opens a browser at http://localhost:7860
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pinn_model import PINN
from inference.predict import load_model, predict_heatmap, microstrip_efield_batch

# ── Load model once at startup ─────────────────────────────────────────────
CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "model", "pinn_checkpoint.pt")

print("Loading PINN model...")
model, device = load_model(CKPT)
print("Model ready.\n")


# ── Core prediction function ───────────────────────────────────────────────
def run_prediction(freq_ghz: float,
                   height_mm: float,
                   trace_width_mm: float,
                   show_comparison: bool) -> tuple:
    """
    Called by Gradio on every slider change.

    Returns
    -------
    (figure, metrics_text)
    """
    freq_hz  = freq_ghz  * 1e9
    z_height = height_mm / 1000.0

    heatmap, xx, yy, elapsed_ms = predict_heatmap(
        model, device,
        freq_hz=freq_hz,
        z_height=z_height,
        n=100,
    )

    if show_comparison:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(6, 5))
        axes = [axes]

    extent = [xx.min() * 1000, xx.max() * 1000,
              yy.min() * 1000, yy.max() * 1000]

    # PINN prediction
    im0 = axes[0].imshow(heatmap, cmap="inferno", origin="lower", extent=extent)
    plt.colorbar(im0, ax=axes[0], label="|E| (V/m)")
    axes[0].set_title(f"PINN Prediction  |  {freq_ghz:.2f} GHz  |  z={height_mm} mm",
                      fontsize=10)
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    axes[0].text(0.02, 0.97,
                 "White = High EMI\nDark = Safe",
                 transform=axes[0].transAxes, fontsize=7,
                 va="top", color="white",
                 bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55))

    # Analytical ground truth (for comparison)
    if show_comparison and len(axes) > 1:
        xs = np.linspace(-0.05, 0.05, 100)
        ys = np.linspace(-0.05, 0.05, 100)
        XX, YY = np.meshgrid(xs, ys)
        pts = np.stack([XX.ravel(), YY.ravel(),
                        np.full(10000, z_height),
                        np.full(10000, freq_hz)], axis=1).astype(np.float32)
        E_true = microstrip_efield_batch(pts).reshape(100, 100)

        im1 = axes[1].imshow(E_true, cmap="inferno", origin="lower", extent=extent)
        plt.colorbar(im1, ax=axes[1], label="|E| (V/m)")
        axes[1].set_title("Analytical Ground Truth", fontsize=10)
        axes[1].set_xlabel("x (mm)")
        axes[1].set_ylabel("y (mm)")

        mae = float(np.mean(np.abs(heatmap - E_true)))
        rel = mae / (float(np.mean(np.abs(E_true))) + 1e-9) * 100
        metrics = (
            f"Inference time  : {elapsed_ms:.1f} ms\n"
            f"MAE vs truth    : {mae:.4f} V/m\n"
            f"Relative error  : {rel:.2f} %\n"
            f"Grid resolution : 100 × 100 = 10,000 points\n"
            f"Frequency       : {freq_ghz:.2f} GHz\n"
            f"Height above PCB: {height_mm:.1f} mm"
        )
    else:
        metrics = (
            f"Inference time  : {elapsed_ms:.1f} ms\n"
            f"Grid resolution : 100 × 100 = 10,000 points\n"
            f"Frequency       : {freq_ghz:.2f} GHz\n"
            f"Height above PCB: {height_mm:.1f} mm\n"
            f"Trace width     : {trace_width_mm:.1f} mm"
        )

    plt.tight_layout()
    return fig, metrics


# ── Gradio interface ───────────────────────────────────────────────────────
with gr.Blocks(title="PINN EMI Predictor — Team Nexus") as demo:

    gr.Markdown("""
    # PINN EMI Predictor
    **Team Nexus** · Physics-Informed Neural Network for EMC compliance  
    Predicts electromagnetic interference fields on any PCB layout in **milliseconds** —  
    trained on 200 samples, powered by Maxwell's equations, runs on a laptop.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            freq_slider   = gr.Slider(0.1, 6.0,  value=2.4, step=0.1,
                                      label="Frequency (GHz)")
            height_slider = gr.Slider(1.0, 20.0, value=5.0, step=0.5,
                                      label="Height above PCB (mm)")
            width_slider  = gr.Slider(0.1, 3.0,  value=1.0, step=0.1,
                                      label="Trace width (mm)")
            compare_chk   = gr.Checkbox(label="Show analytical comparison", value=False)
            run_btn       = gr.Button("Generate EMI Heatmap", variant="primary")

        with gr.Column(scale=2):
            plot_out    = gr.Plot(label="EMI Field Heatmap")
            metrics_out = gr.Textbox(label="Prediction Metrics", lines=6,
                                     interactive=False)

    gr.Markdown("""
    ---
    **Colour guide:** White/Yellow = extremely high EMI (EMC failure zone) · 
    Orange/Red = moderate risk · Dark purple = safe zone  
    
    **How it works:** Fourier-encoded coordinates → 6-layer neural network · 
    trained with Helmholtz wave equation physics loss · 200 labeled samples only
    """)

    run_btn.click(
        fn=run_prediction,
        inputs=[freq_slider, height_slider, width_slider, compare_chk],
        outputs=[plot_out, metrics_out],
    )

    # Auto-run on load
    demo.load(
        fn=run_prediction,
        inputs=[freq_slider, height_slider, width_slider, compare_chk],
        outputs=[plot_out, metrics_out],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
