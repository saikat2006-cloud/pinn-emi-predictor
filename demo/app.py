"""
demo/app.py
===========
Gradio web interface for the PINN EMI Predictor.
Shows PCB trace overlay on heatmap so judges can see what geometry
the model learned.

Usage:
    python demo/app.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gradio as gr

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pinn_model import PINN
from inference.predict import load_model, predict_heatmap, microstrip_efield_batch
from model.physics_loss import denormalize_y

# ── Load model once at startup ─────────────────────────────────────────────
CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "model", "pinn_checkpoint.pt")

print("Loading PINN model...")
model, device = load_model(CKPT)
print("Model ready.\n")


# ── Draw PCB overlay on an axes ────────────────────────────────────────────
def draw_pcb_overlay(ax, trace_width_mm, extent):
    """
    Draws the PCB geometry on top of the heatmap so judges can see
    exactly what the model was trained on:
      - Green dashed rectangle  = PCB board boundary
      - Cyan filled rectangle   = microstrip trace (center, horizontal)
      - Cyan arrow              = current flow direction
      - Lime line               = ground plane (bottom edge)
    """
    tw = trace_width_mm / 2   # half-width for centering

    # Board outline
    board = mpatches.Rectangle(
        (-50, -50), 100, 100,
        linewidth=1.5, edgecolor="#00ff88",
        facecolor="none", linestyle="--", alpha=0.7
    )
    ax.add_patch(board)

    # Microstrip trace — runs full width along y=0
    trace = mpatches.Rectangle(
        (-50, -tw), 100, tw * 2,
        linewidth=1, edgecolor="cyan",
        facecolor="cyan", alpha=0.30
    )
    ax.add_patch(trace)

    # Trace border lines (top and bottom edge of trace)
    ax.axhline(y= tw, color="cyan", linewidth=0.8, alpha=0.7, linestyle="-")
    ax.axhline(y=-tw, color="cyan", linewidth=0.8, alpha=0.7, linestyle="-")

    # Current flow arrow
    ax.annotate(
        "", xy=(28, 0), xytext=(-28, 0),
        arrowprops=dict(arrowstyle="-|>", color="cyan",
                        lw=1.5, mutation_scale=12)
    )
    ax.text(0, tw + 2.5, "Current flow →",
            color="cyan", fontsize=7, ha="center", va="bottom", alpha=0.9)

    # Ground plane line at bottom of board
    ax.axhline(y=-48, color="#00ff88", linewidth=2.5, alpha=0.6)
    ax.text(-48, -46, "Ground plane",
            color="#00ff88", fontsize=6.5, va="bottom", alpha=0.8)

    # Labels
    ax.text(46, 46, "PCB", color="#00ff88",
            fontsize=7, ha="right", va="top", alpha=0.8)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="cyan",    alpha=0.5, label=f"Trace (w={trace_width_mm}mm)"),
        mpatches.Patch(facecolor="#00ff88", alpha=0.5, label="Board / GND"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              fontsize=6.5, facecolor="#111", labelcolor="white",
              framealpha=0.7, borderpad=0.4)


# ── Core prediction function ───────────────────────────────────────────────
def run_prediction(freq_ghz, height_mm, trace_width_mm, show_comparison):

    freq_hz  = freq_ghz  * 1e9
    z_height = height_mm / 1000.0

    heatmap, xx, yy, elapsed_ms = predict_heatmap(
        model, device,
        freq_hz=freq_hz,
        z_height=z_height,
        n=100,
    )

    extent = [xx.min() * 1000, xx.max() * 1000,
              yy.min() * 1000, yy.max() * 1000]

    if show_comparison:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        fig.patch.set_facecolor("#111111")
        for ax in axes:
            ax.set_facecolor("#111111")
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(6.5, 5.5))
        fig.patch.set_facecolor("#111111")
        ax0.set_facecolor("#111111")
        axes = [ax0]

    # ── PINN prediction plot ───────────────────────────────────────────────
    im0 = axes[0].imshow(heatmap, cmap="inferno",
                         origin="lower", extent=extent, aspect="equal")
    cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.set_label("|E| (V/m)", color="white", fontsize=9)
    cb0.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb0.ax.yaxis.get_ticklabels(), color="white")

    axes[0].set_title(
        f"PINN Prediction  |  {freq_ghz:.2f} GHz  |  z = {height_mm} mm",
        fontsize=10, color="white", pad=8
    )
    axes[0].set_xlabel("x (mm)", color="white", fontsize=9)
    axes[0].set_ylabel("y (mm)", color="white", fontsize=9)
    axes[0].tick_params(colors="white")
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#444")

    # EMI risk annotation
    axes[0].text(0.02, 0.97,
                 "White = High EMI risk\nDark  = Safe zone",
                 transform=axes[0].transAxes, fontsize=7,
                 va="top", color="white",
                 bbox=dict(boxstyle="round,pad=0.3",
                           fc="black", alpha=0.6, ec="#555"))

    # PCB overlay on PINN plot
    draw_pcb_overlay(axes[0], trace_width_mm, extent)

    # ── Analytical ground truth plot ───────────────────────────────────────
    if show_comparison and len(axes) > 1:
        xs = np.linspace(-0.05, 0.05, 100)
        ys = np.linspace(-0.05, 0.05, 100)
        XX, YY = np.meshgrid(xs, ys)
        pts = np.stack([XX.ravel(), YY.ravel(),
                        np.full(10000, z_height),
                        np.full(10000, freq_hz)], axis=1).astype(np.float32)
        E_true = microstrip_efield_batch(pts).reshape(100, 100)

        im1 = axes[1].imshow(E_true, cmap="inferno",
                             origin="lower", extent=extent, aspect="equal")
        cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        cb1.set_label("|E| (V/m)", color="white", fontsize=9)
        cb1.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb1.ax.yaxis.get_ticklabels(), color="white")

        axes[1].set_title("Analytical Ground Truth (Maxwell's Equations)",
                          fontsize=10, color="white", pad=8)
        axes[1].set_xlabel("x (mm)", color="white", fontsize=9)
        axes[1].set_ylabel("y (mm)", color="white", fontsize=9)
        axes[1].tick_params(colors="white")
        for spine in axes[1].spines.values():
            spine.set_edgecolor("#444")

        # PCB overlay on ground truth too
        draw_pcb_overlay(axes[1], trace_width_mm, extent)

        mae = float(np.mean(np.abs(heatmap - E_true)))
        rel = mae / (float(np.mean(np.abs(E_true))) + 1e-9) * 100
        metrics = (
            f"Inference time   : {elapsed_ms:.1f} ms  (HFSS takes 4-8 hours)\n"
            f"MAE vs truth     : {mae:.4f} V/m\n"
            f"Relative error   : {rel:.2f} %\n"
            f"Grid resolution  : 100 × 100 = 10,000 points\n"
            f"Frequency        : {freq_ghz:.2f} GHz\n"
            f"Height above PCB : {height_mm:.1f} mm\n"
            f"Trace width      : {trace_width_mm:.1f} mm"
        )
    else:
        metrics = (
            f"Inference time   : {elapsed_ms:.1f} ms  (HFSS takes 4-8 hours)\n"
            f"Grid resolution  : 100 × 100 = 10,000 points\n"
            f"Frequency        : {freq_ghz:.2f} GHz\n"
            f"Height above PCB : {height_mm:.1f} mm\n"
            f"Trace width      : {trace_width_mm:.1f} mm\n"
            f"Training samples : 200  (conventional ML needs 10,000+)"
        )

    plt.tight_layout()
    return fig, metrics


# ── Gradio interface ───────────────────────────────────────────────────────
with gr.Blocks(
    title="PINN EMI Predictor — Team Nexus",
    theme=gr.themes.Base()
) as demo:

    gr.Markdown("""
    # PINN EMI Predictor — Team Nexus
    **Physics-Informed Neural Network for PCB EMC Compliance**

    Predicts electromagnetic interference fields on a PCB layout in **milliseconds** —
    trained on only 200 samples, powered by Maxwell's equations, runs on a laptop.
    No HFSS. No CST. No $100,000 license.
    """)

    with gr.Row():
        with gr.Column(scale=1):

            gr.Markdown("### PCB & Simulation Parameters")

            freq_slider = gr.Slider(
                0.1, 6.0, value=2.4, step=0.1,
                label="Frequency (GHz)",
                info="Wi-Fi=2.4GHz  |  USB3=5GHz  |  Bluetooth=2.4GHz"
            )
            height_slider = gr.Slider(
                1.0, 20.0, value=5.0, step=0.5,
                label="Height above PCB (mm)",
                info="Distance of measurement plane above board"
            )
            width_slider = gr.Slider(
                0.1, 3.0, value=1.0, step=0.1,
                label="Trace width (mm)",
                info="Width of the microstrip trace on the PCB"
            )
            compare_chk = gr.Checkbox(
                label="Show analytical ground truth comparison",
                value=False
            )
            run_btn = gr.Button("Generate EMI Heatmap", variant="primary")

            gr.Markdown("""
            ---
            **PCB Geometry (trained on):**
            - Single microstrip trace (cyan bar)
            - Horizontal, center of board
            - FR4 substrate, 1.6mm thick
            - Ground plane beneath

            **Colour guide:**
            - White/Yellow = High EMI → **FAIL zone**
            - Orange/Red   = Moderate risk
            - Dark purple  = Safe zone ✓
            """)

        with gr.Column(scale=2):
            plot_out    = gr.Plot(label="EMI Field Heatmap")
            metrics_out = gr.Textbox(
                label="Prediction Metrics",
                lines=7,
                interactive=False
            )

    gr.Markdown("""
    ---
    **How it works:** Raw (x, y, z, f) coordinates →
    Fourier Feature Encoder (128-dim sinusoidal embedding) →
    4-layer neural network →
    |E| field prediction in V/m

    **Why it's fast:** Single forward pass through the network — no mesh, no solver, no iterations.
    """)

    run_btn.click(
        fn=run_prediction,
        inputs=[freq_slider, height_slider, width_slider, compare_chk],
        outputs=[plot_out, metrics_out],
    )

    demo.load(
        fn=run_prediction,
        inputs=[freq_slider, height_slider, width_slider, compare_chk],
        outputs=[plot_out, metrics_out],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        show_error=True,
    )