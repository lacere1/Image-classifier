"""Export the trained checkpoint to ONNX and produce quantized variants.

Three artifacts come out of this:

  <name>_fp32.onnx        ONNX graph, float32
  <name>_int8_dynamic.onnx  ONNX Runtime dynamic quantization (weights INT8)
  <name>_int8_static.onnx    ONNX Runtime static QDQ quantization, calibrated
                             on real training images

Dynamic quantization only quantizes MatMul/Gemm weights, which on a ResNet is
essentially just the final classifier -- expect a size win but little speedup.
Static QDQ quantization also quantizes the convolutions, which is where a CNN's
compute actually lives. Both are exported so the benchmark can show the
difference honestly rather than assuming INT8 is automatically faster.

A separate PyTorch dynamic-quantized checkpoint is also saved for comparison.

    python scripts/export_onnx.py --checkpoint artifacts/baseline_best.pt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from landmarklens.data import eval_transforms
from landmarklens.model import load_checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.path.join(ROOT, "artifacts")
SPLITS = os.path.join(ROOT, "data", "splits")


class CalibrationReader:
    """Feeds real training images to ONNX Runtime's static quantizer.

    Calibrating on actual photos (rather than random noise) is what keeps the
    INT8 activation ranges meaningful and the accuracy drop small.
    """

    def __init__(self, input_name: str, paths: list[str], size: int):
        self.input_name = input_name
        self.transform = eval_transforms(size)
        self.paths = paths
        self._it = None

    def _gen(self):
        for p in self.paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            yield {self.input_name:
                   self.transform(img).unsqueeze(0).numpy().astype(np.float32)}

    def get_next(self):
        if self._it is None:
            self._it = self._gen()
        return next(self._it, None)

    def rewind(self):
        self._it = None


def calibration_paths(n: int, seed: int = 7, splits: str = SPLITS) -> list[str]:
    paths = glob.glob(os.path.join(splits, "train", "*", "*"))
    random.Random(seed).shuffle(paths)
    return paths[:n]


def mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def sensitive_nodes(model_path: str) -> list[str]:
    """Names of the first Conv and the final Gemm/MatMul in the graph.

    The stem convolution sees raw normalised pixels with a wide dynamic range,
    and the classifier produces logits whose small differences decide the
    prediction -- both quantize badly. Together they are a tiny fraction of a
    ResNet50's FLOPs, so keeping them in float costs almost no speed.
    """
    import onnx

    graph = onnx.load(model_path).graph
    names: list[str] = []
    for node in graph.node:
        if node.op_type == "Conv":
            names.append(node.name)
            break
    for node in reversed(graph.node):
        if node.op_type in ("Gemm", "MatMul"):
            names.append(node.name)
            break
    return [n for n in names if n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--calib-images", type=int, default=160)
    ap.add_argument("--splits", default=SPLITS)
    ap.add_argument("--calib-method", default="percentile",
                    choices=["minmax", "entropy", "percentile"])
    ap.add_argument("--reduce-range", action="store_true", default=False,
                    help="use 7-bit weight range; helps on pre-VNNI CPUs")
    ap.add_argument("--exclude-sensitive", action="store_true", default=True,
                    help="keep the stem Conv and final Gemm in float32")
    ap.add_argument("--no-exclude-sensitive", dest="exclude_sensitive",
                    action="store_false")
    args = ap.parse_args()

    name = args.name or os.path.splitext(os.path.basename(args.checkpoint))[0]
    os.makedirs(ARTIFACTS, exist_ok=True)

    model, classes, backbone, meta = load_checkpoint(args.checkpoint)
    size = int(meta.get("image_size", 224))
    dummy = torch.randn(1, 3, size, size)

    results = {"name": name, "backbone": backbone, "image_size": size,
               "classes": classes, "artifacts": {}}

    # --- 1. FP32 ONNX ------------------------------------------------------
    fp32_path = os.path.join(ARTIFACTS, f"{name}_fp32.onnx")
    torch.onnx.export(
        model, dummy, fp32_path, opset_version=args.opset,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )
    print(f"FP32 ONNX          -> {fp32_path}  ({mb(fp32_path):.1f} MB)")
    results["artifacts"]["onnx_fp32"] = {"path": fp32_path,
                                         "size_mb": mb(fp32_path)}

    import onnx
    onnx.checker.check_model(onnx.load(fp32_path))
    print("  onnx.checker: OK")

    # --- 1b. Fixed-shape graph, used only as the quantization source -------
    # The histogram calibrators (percentile/entropy) fail on the graph above
    # with "Got invalid dimensions for input", because they augment the graph
    # to read intermediate tensors and cannot resolve the symbolic batch axis.
    # Serving is batch=1 anyway, so quantize from a batch-1 graph.
    fixed_path = os.path.join(ARTIFACTS, f"{name}_fp32_fixed.onnx")
    torch.onnx.export(
        model, dummy, fixed_path, opset_version=args.opset,
        input_names=["input"], output_names=["logits"],
        do_constant_folding=True,
    )
    print(f"FP32 ONNX (batch=1) -> {fixed_path}  ({mb(fixed_path):.1f} MB)"
          f"  [quantization source]")

    # --- 2. ONNX Runtime dynamic INT8 -------------------------------------
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat,
                                          QuantType, quantize_dynamic,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    prepped = os.path.join(ARTIFACTS, f"{name}_prepped.onnx")
    quant_pre_process(fixed_path, prepped, skip_symbolic_shape=False)

    dyn_path = os.path.join(ARTIFACTS, f"{name}_int8_dynamic.onnx")
    # Only MatMul/Gemm. Letting dynamic quantization touch Conv emits
    # ConvInteger nodes, for which the ORT CPU provider has no kernel -- the
    # resulting graph exports fine and then fails at session creation with
    # "Could not find an implementation for ConvInteger". Convolutions are
    # handled by the static QDQ path below instead.
    quantize_dynamic(prepped, dyn_path, weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"])
    print(f"INT8 dynamic ONNX  -> {dyn_path}  ({mb(dyn_path):.1f} MB)")
    results["artifacts"]["onnx_int8_dynamic"] = {"path": dyn_path,
                                                 "size_mb": mb(dyn_path)}

    # --- 3. ONNX Runtime static (QDQ) INT8 --------------------------------
    import onnxruntime as ort
    input_name = ort.InferenceSession(
        fp32_path, providers=["CPUExecutionProvider"]).get_inputs()[0].name

    paths = calibration_paths(args.calib_images, splits=args.splits)
    static_path = os.path.join(ARTIFACTS, f"{name}_int8_static.onnx")
    if paths:
        reader = CalibrationReader(input_name, paths, size)
        methods = {"minmax": CalibrationMethod.MinMax,
                   "entropy": CalibrationMethod.Entropy,
                   "percentile": CalibrationMethod.Percentile}
        # MinMax calibration on this ResNet50 cost ~30 points of accuracy: a
        # single outlier activation stretches the quantization range so far
        # that ordinary activations collapse into a handful of levels.
        # Percentile/Entropy clip the tail instead, and excluding the first
        # convolution and the final Gemm -- the two most precision-sensitive
        # layers, and a negligible share of total compute -- recovers the rest.
        extra = {}
        if args.exclude_sensitive:
            extra["nodes_to_exclude"] = sensitive_nodes(prepped)
        quantize_static(
            prepped, static_path, reader, quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
            calibrate_method=methods[args.calib_method],
            per_channel=True, reduce_range=args.reduce_range, **extra,
        )
        print(f"INT8 static ONNX   -> {static_path}  "
              f"({mb(static_path):.1f} MB)  "
              f"[{args.calib_method} calibration on {len(paths)} training "
              f"images, sensitive layers "
              f"{'excluded' if args.exclude_sensitive else 'included'}]")
        results["artifacts"]["onnx_int8_static"] = {
            "path": static_path, "size_mb": mb(static_path),
            "calibration_images": len(paths),
            "calibration_method": args.calib_method,
            "reduce_range": args.reduce_range,
            "exclude_sensitive": args.exclude_sensitive}
    else:
        print("INT8 static ONNX   -> SKIPPED (no calibration images found)")

    # --- 4. PyTorch dynamic INT8 ------------------------------------------
    qmodel = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8)
    pt_int8 = os.path.join(ARTIFACTS, f"{name}_int8_dynamic_torch.pt")
    torch.save({"state_dict": qmodel.state_dict(), "classes": classes,
                "backbone": backbone, "meta": meta}, pt_int8)
    print(f"INT8 dynamic Torch -> {pt_int8}  ({mb(pt_int8):.1f} MB)")
    results["artifacts"]["torch_int8_dynamic"] = {"path": pt_int8,
                                                  "size_mb": mb(pt_int8)}

    fp32_pt = mb(args.checkpoint)
    print(f"FP32 Torch ckpt    :  {fp32_pt:.1f} MB")
    results["artifacts"]["torch_fp32"] = {"path": args.checkpoint,
                                          "size_mb": fp32_pt}

    if os.path.exists(prepped):
        os.remove(prepped)

    out = os.path.join(ARTIFACTS, f"{name}_export.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
