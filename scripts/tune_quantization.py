"""Find a static-INT8 configuration that keeps accuracy.

The default MinMax calibration cost this model ~30 points of top-1 accuracy,
which makes the 8x speedup worthless. This sweeps the knobs that matter --
calibration method, 7-bit weight range, and whether the precision-sensitive
stem convolution and final classifier are left in float32 -- and reports the
measured test accuracy and latency of each, so the deployed variant is chosen
on evidence rather than on the assumption that INT8 is free.

    python scripts/tune_quantization.py --checkpoint artifacts/best_model.pt
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.benchmark import load_test_tensors, acc_from_logits  # noqa: E402
from scripts.export_onnx import CalibrationReader, calibration_paths, \
    sensitive_nodes  # noqa: E402
from landmarklens.model import load_checkpoint  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.path.join(ROOT, "artifacts")

CONFIGS = [
    # (label, calib_method, reduce_range, exclude_sensitive)
    ("minmax", "minmax", False, False),
    ("percentile", "percentile", False, False),
    ("entropy", "entropy", False, False),
    ("percentile+excl", "percentile", False, True),
    ("entropy+excl", "entropy", False, True),
    ("percentile+excl+rr", "percentile", True, True),
]


def build(prepped: str, out: str, reader_factory, method: str,
          reduce_range: bool, exclude: bool) -> None:
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    methods = {"minmax": CalibrationMethod.MinMax,
               "entropy": CalibrationMethod.Entropy,
               "percentile": CalibrationMethod.Percentile}
    extra = {}
    if exclude:
        extra["nodes_to_exclude"] = sensitive_nodes(prepped)
    quantize_static(
        prepped, out, reader_factory(), quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
        calibrate_method=methods[method], per_channel=True,
        reduce_range=reduce_range, **extra,
    )


def time_session(sess, x, runs: int = 60, warmup: int = 10) -> float:
    name = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {name: x})
    s = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        s.append((time.perf_counter() - t0) * 1000)
    return statistics.median(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(ARTIFACTS,
                                                         "best_model.pt"))
    ap.add_argument("--name", default="best_model")
    ap.add_argument("--calib-images", type=int, default=160)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    model, classes, backbone, meta = load_checkpoint(args.checkpoint)
    size = int(meta.get("image_size", 224))

    fp32 = os.path.join(ARTIFACTS, f"{args.name}_fp32.onnx")
    if not os.path.exists(fp32):
        sys.exit(f"missing {fp32}; run scripts/export_onnx.py first")

    # Quantize from the fixed batch-1 graph: histogram calibrators cannot
    # resolve the symbolic batch axis of the dynamic-shape export.
    fixed = os.path.join(ARTIFACTS, f"{args.name}_fp32_fixed.onnx")
    if not os.path.exists(fixed):
        torch.onnx.export(model, torch.randn(1, 3, size, size), fixed,
                          opset_version=17, input_names=["input"],
                          output_names=["logits"], do_constant_folding=True)

    from onnxruntime.quantization.shape_inference import quant_pre_process
    prepped = os.path.join(ARTIFACTS, f"{args.name}_tune_prepped.onnx")
    quant_pre_process(fixed, prepped, skip_symbolic_shape=False)

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1

    X, y, test_classes = load_test_tensors(size)
    assert test_classes == classes
    print(f"test split n={len(X)}\n")

    input_name = ort.InferenceSession(
        fp32, so, providers=["CPUExecutionProvider"]).get_inputs()[0].name
    paths = calibration_paths(args.calib_images)

    def reader_factory():
        return CalibrationReader(input_name, paths, size)

    x1 = X[:1].numpy()

    # FP32 reference
    sess = ort.InferenceSession(fp32, so, providers=["CPUExecutionProvider"])
    outs = [sess.run(None, {input_name: X[i:i + 16].numpy()})[0]
            for i in range(0, len(X), 16)]
    ref1, ref3 = acc_from_logits(np.concatenate(outs), y.numpy())
    ref_ms = time_session(sess, x1)
    print(f"{'config':<22}{'top1':>9}{'top3':>9}{'median':>10}{'speedup':>9}"
          f"{'size':>9}")
    print("-" * 68)
    print(f"{'ONNX FP32 (reference)':<22}{ref1:>9.4f}{ref3:>9.4f}"
          f"{ref_ms:>9.2f}ms{1.0:>8.2f}x"
          f"{os.path.getsize(fp32) / 1024 ** 2:>8.1f}MB")

    results = [{"config": "onnx_fp32", "top1": ref1, "top3": ref3,
                "median_ms": ref_ms,
                "size_mb": os.path.getsize(fp32) / 1024 ** 2}]

    for label, method, rr, excl in CONFIGS:
        out = os.path.join(ARTIFACTS, f"{args.name}_tune_{label}.onnx")
        try:
            build(prepped, out, reader_factory, method, rr, excl)
            s = ort.InferenceSession(out, so,
                                     providers=["CPUExecutionProvider"])
            iname = s.get_inputs()[0].name
            outs = [s.run(None, {iname: X[i:i + 1].numpy()})[0]
                    for i in range(len(X))]
            t1, t3 = acc_from_logits(np.concatenate(outs), y.numpy())
            ms = time_session(s, x1)
            size = os.path.getsize(out) / 1024 ** 2
            print(f"{label:<22}{t1:>9.4f}{t3:>9.4f}{ms:>9.2f}ms"
                  f"{ref_ms / ms:>8.2f}x{size:>8.1f}MB")
            results.append({"config": label, "calib_method": method,
                            "reduce_range": rr, "exclude_sensitive": excl,
                            "top1": t1, "top3": t3, "median_ms": ms,
                            "size_mb": size})
        except Exception as exc:
            print(f"{label:<22}  FAILED: {str(exc)[:44]}")
            results.append({"config": label, "error": str(exc)[:300]})
        finally:
            if os.path.exists(out):
                os.remove(out)

    if os.path.exists(prepped):
        os.remove(prepped)

    path = os.path.join(ARTIFACTS, f"{args.name}_quant_tuning.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"reference_top1": ref1, "reference_top3": ref3,
                   "results": results}, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
