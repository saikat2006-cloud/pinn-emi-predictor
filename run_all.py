#!/usr/bin/env python3
"""
run_all.py
==========
One-click script to run the full PINN pipeline:

  Step 1 – Generate data
  Step 2 – Train model
  Step 3 – Run inference & benchmark
  Step 4 – Launch Gradio demo

Usage:
    python run_all.py              # full pipeline
    python run_all.py --skip-train # skip training (use saved checkpoint)
    python run_all.py --demo-only  # launch demo only
"""

import argparse
import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd: list, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    result = subprocess.run([sys.executable] + cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed: {label}")
        sys.exit(result.returncode)


parser = argparse.ArgumentParser()
parser.add_argument("--skip-train", action="store_true")
parser.add_argument("--demo-only",  action="store_true")
args = parser.parse_args()

if args.demo_only:
    run(["demo/app.py"], "Launching Gradio demo")
    sys.exit(0)

if not args.skip_train:
    run(["data/generate_data.py"], "Step 1 — Generating training data")
    run(["model/train.py"],        "Step 2 — Training PINN model")

run(["inference/predict.py"], "Step 3 — Running inference & benchmark")
run(["demo/app.py"],          "Step 4 — Launching Gradio demo")
