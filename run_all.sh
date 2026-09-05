#!/usr/bin/env bash
# Reproduce the whole LandmarkLens pipeline from scratch.
#
#   bash run_all.sh
#
# Assumes the venv exists and has requirements-dev.txt installed:
#   python -m venv .venv
#   .venv/Scripts/pip install --index-url https://download.pytorch.org/whl/cpu \
#       torch==2.5.1 torchvision==0.20.1
#   .venv/Scripts/pip install -r requirements-dev.txt
set -e

PY=${PY:-./.venv/Scripts/python.exe}
[ -x "$PY" ] || PY=python

echo "=== 1. Harvest dataset from Wikimedia Commons ==="
"$PY" -u scripts/harvest_dataset.py --per-class 110

echo "=== 2. Attribution / licensing record ==="
"$PY" -u scripts/make_data_sources.py

echo "=== 3. Stratified train/val/test split ==="
"$PY" -u scripts/split_dataset.py --val 0.15 --test 0.15 --seed 42

echo "=== 4. Experiment sweep (3 runs, logged to ./mlruns) ==="
"$PY" -u scripts/run_experiments.py --backbone resnet50 \
    --epochs-head 3 --epochs-ft 8 --batch-size 32 --workers 4

echo "=== 5. Evaluate the best model on the held-out test split ==="
"$PY" -u scripts/evaluate.py --checkpoint artifacts/best_model.pt

echo "=== 6. Grad-CAM example figures ==="
"$PY" -u scripts/make_gradcam_examples.py --checkpoint artifacts/best_model.pt

echo "=== 7. ONNX export + INT8 quantization ==="
"$PY" -u scripts/export_onnx.py --checkpoint artifacts/best_model.pt \
    --name best_model

echo "=== 8. Latency + accuracy benchmark ==="
"$PY" -u scripts/benchmark.py --name best_model \
    --checkpoint artifacts/best_model.pt --runs 200 --threads 4

echo "=== 9. Regenerate the measured-results report ==="
"$PY" -u scripts/make_report.py --out docs/RESULTS.md

echo
echo "Done. Serve the model with:"
echo "  PYTHONPATH=. $PY backend/app.py"
echo "Browse the experiment history with:"
echo "  mlflow ui --backend-store-uri ./mlruns"
