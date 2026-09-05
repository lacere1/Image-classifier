"""Benchmark every model variant on the same hardware, in one process.

Measures single-image (batch=1) inference latency -- the number that matters for
a request/response API -- over N timed runs after a warmup, and reports mean,
median, p95 and standard deviation. It also re-scores the *whole test split*
with each variant so the accuracy cost of quantization is measured rather than
assumed.

Everything runs sequentially in one process with a fixed CPU thread count, so
the variants are compared under identical conditions.

    python scripts/benchmark.py --name baseline --runs 200
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import statistics
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from landmarklens.data import eval_transforms
from landmarklens.model import load_checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.path.join(ROOT, "artifacts")
SPLITS = os.path.join(ROOT, "data", "splits")


def timed(fn, x, runs: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn(x)
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(x)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {
        "runs": runs,
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": samples[int(0.95 * (len(samples) - 1))],
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "stdev_ms": statistics.pstdev(samples),
    }


def load_test_tensors(size: int, limit: int = 0, splits: str = SPLITS):
    """Preprocess the test split once; every variant scores the same tensors."""
    tf = eval_transforms(size)
    classes = sorted(d for d in os.listdir(os.path.join(splits, "test"))
                     if os.path.isdir(os.path.join(splits, "test", d)))
    idx = {c: i for i, c in enumerate(classes)}
    xs, ys = [], []
    for c in classes:
        for p in sorted(glob.glob(os.path.join(splits, "test", c, "*"))):
            try:
                xs.append(tf(Image.open(p).convert("RGB")))
                ys.append(idx[c])
            except Exception:
                continue
    if limit:
        xs, ys = xs[:limit], ys[:limit]
    return torch.stack(xs), torch.tensor(ys), classes


def acc_from_logits(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    top1 = float((logits.argmax(axis=1) == y).mean())
    k = min(3, logits.shape[1])
    top3_idx = np.argpartition(-logits, k - 1, axis=1)[:, :k]
    top3 = float((top3_idx == y[:, None]).any(axis=1).mean())
    return top1, top3


def score_torch(model, X, y, batch=16) -> tuple[float, float]:
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            outs.append(model(X[i:i + batch]).numpy())
    return acc_from_logits(np.concatenate(outs), y.numpy())


def score_onnx(sess, X, y, batch=16) -> tuple[float, float]:
    """Score a session, respecting a fixed batch dimension if it has one.

    The quantized graphs are exported at a fixed batch of 1 (the histogram
    calibrators cannot handle a symbolic batch axis), so feeding them batches
    would raise an input-shape error rather than a wrong number.
    """
    inp = sess.get_inputs()[0]
    name = inp.name
    if inp.shape and isinstance(inp.shape[0], int):
        batch = inp.shape[0]
    outs = []
    for i in range(0, len(X), batch):
        chunk = X[i:i + batch]
        if len(chunk) < batch:  # fixed-shape graph cannot take a short tail
            for j in range(len(chunk)):
                outs.append(sess.run(None, {name: chunk[j:j + 1].numpy()})[0])
            continue
        outs.append(sess.run(None, {name: chunk.numpy()})[0])
    return acc_from_logits(np.concatenate(outs), y.numpy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4,
                    help="fixed CPU thread count for every variant")
    ap.add_argument("--splits", default=SPLITS)
    ap.add_argument("--skip-accuracy", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    ckpt = args.checkpoint or os.path.join(ARTIFACTS, f"{args.name}_best.pt")

    model, classes, backbone, meta = load_checkpoint(ckpt)
    size = int(meta.get("image_size", 224))
    x1 = torch.randn(1, 3, size, size)

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1

    print(f"host      : {platform.processor() or platform.machine()}")
    print(f"python    : {platform.python_version()}   torch {torch.__version__}"
          f"   onnxruntime {ort.__version__}")
    print(f"threads   : {args.threads}   image size: {size}   "
          f"batch: 1   timed runs: {args.runs} (warmup {args.warmup})\n")

    X = y = None
    if not args.skip_accuracy:
        X, y, test_classes = load_test_tensors(size, splits=args.splits)
        assert test_classes == classes, "test split / checkpoint class mismatch"
        print(f"accuracy measured on the full test split (n={len(X)})\n")

    variants: list[tuple[str, dict]] = []

    # --- PyTorch FP32 ------------------------------------------------------
    def run_pt(x):
        with torch.no_grad():
            return model(x)
    entry = {"backend": "pytorch", "precision": "fp32",
             "size_mb": os.path.getsize(ckpt) / 1024 ** 2}
    entry.update(timed(run_pt, x1, args.runs, args.warmup))
    if X is not None:
        entry["top1"], entry["top3"] = score_torch(model, X, y)
    variants.append(("PyTorch FP32 (baseline)", entry))

    # --- PyTorch dynamic INT8 ---------------------------------------------
    qmodel = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8)

    def run_qpt(x):
        with torch.no_grad():
            return qmodel(x)
    pt_int8 = os.path.join(ARTIFACTS, f"{args.name}_int8_dynamic_torch.pt")
    entry = {"backend": "pytorch", "precision": "int8-dynamic",
             "size_mb": (os.path.getsize(pt_int8) / 1024 ** 2
                         if os.path.exists(pt_int8) else None)}
    entry.update(timed(run_qpt, x1, args.runs, args.warmup))
    if X is not None:
        entry["top1"], entry["top3"] = score_torch(qmodel, X, y)
    variants.append(("PyTorch INT8 dynamic", entry))

    # --- ONNX variants -----------------------------------------------------
    onnx_variants = [
        ("ONNX Runtime FP32", f"{args.name}_fp32.onnx", "fp32"),
        ("ONNX Runtime INT8 dynamic", f"{args.name}_int8_dynamic.onnx",
         "int8-dynamic"),
        ("ONNX Runtime INT8 static (QDQ)", f"{args.name}_int8_static.onnx",
         "int8-static"),
    ]
    xnp = x1.numpy()
    for label, fname, precision in onnx_variants:
        path = os.path.join(ARTIFACTS, fname)
        if not os.path.exists(path):
            print(f"  (skipping {label}: {fname} not found)")
            continue
        try:
            sess = ort.InferenceSession(path, so,
                                        providers=["CPUExecutionProvider"])
        except Exception as exc:
            # A variant that will not even load is a real result worth
            # reporting, not a reason to abandon the whole benchmark.
            print(f"  ({label} failed to load: {exc})")
            variants.append((label, {"backend": "onnxruntime",
                                     "precision": precision,
                                     "error": str(exc)[:200],
                                     "size_mb": os.path.getsize(path) / 1024 ** 2}))
            continue
        iname = sess.get_inputs()[0].name

        def run_onnx(x, _s=sess, _n=iname):
            return _s.run(None, {_n: x})

        entry = {"backend": "onnxruntime", "precision": precision,
                 "size_mb": os.path.getsize(path) / 1024 ** 2}
        entry.update(timed(run_onnx, xnp, args.runs, args.warmup))
        if X is not None:
            entry["top1"], entry["top3"] = score_onnx(sess, X, y)
        variants.append((label, entry))

    # --- Report ------------------------------------------------------------
    base = variants[0][1]["median_ms"]
    print(f"{'variant':<34}{'median':>9}{'mean':>9}{'p95':>9}"
          f"{'speedup':>9}{'size':>9}{'top1':>8}{'top3':>8}")
    print("-" * 95)
    for label, e in variants:
        if "median_ms" not in e:
            print(f"{label:<34}   FAILED TO LOAD: {e.get('error', '')[:44]}")
            continue
        sp = base / e["median_ms"]
        top1 = f"{e['top1']:.4f}" if "top1" in e else "  -  "
        top3 = f"{e['top3']:.4f}" if "top3" in e else "  -  "
        size = f"{e['size_mb']:.1f}MB" if e.get("size_mb") else "  -  "
        print(f"{label:<34}{e['median_ms']:>8.2f}ms{e['mean_ms']:>8.2f}ms"
              f"{e['p95_ms']:>8.2f}ms{sp:>8.2f}x{size:>9}{top1:>8}{top3:>8}")

    payload = {
        "host": {
            "processor": platform.processor(), "machine": platform.machine(),
            "system": platform.system(), "release": platform.release(),
            "python": platform.python_version(), "torch": torch.__version__,
            "onnxruntime": ort.__version__, "threads": args.threads,
            "cuda_available": torch.cuda.is_available(),
        },
        "config": {"name": args.name, "checkpoint": ckpt, "image_size": size,
                   "batch_size": 1, "runs": args.runs, "warmup": args.warmup,
                   "test_n": int(len(X)) if X is not None else 0},
        "variants": {label: e for label, e in variants},
    }
    out = os.path.join(ARTIFACTS, f"{args.name}_benchmark.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
