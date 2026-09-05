"""Render the measured results as Markdown, straight from the artifact JSON.

Every number in the README comes through here, so nothing is transcribed by
hand and nothing can drift from what was actually measured. Reads:

    data/split_stats.json                  dataset + split sizes
    artifacts/experiments.json             the experiment sweep
    artifacts/best_model_test_eval.json    held-out test results
    artifacts/best_model_benchmark.json    latency + accuracy per variant
    docs/gradcam/manifest.json             example figures

    python scripts/make_report.py             # print to stdout
    python scripts/make_report.py --out docs/RESULTS.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path: str):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return None
    with open(full, encoding="utf-8") as fh:
        return json.load(fh)


def pct(x) -> str:
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "-"


def dataset_section(out: list[str]) -> None:
    stats = load("data/split_stats.json")
    if not stats:
        out.append("_(dataset stats not generated yet)_\n")
        return
    t = stats["totals"]
    out.append(f"**{t['total']} images across "
               f"{sum(1 for v in stats['per_class'].values() if v['total'])} "
               f"classes.** Split with seed {stats['seed']} "
               f"({int((1 - stats['val_frac'] - stats['test_frac']) * 100)}% "
               f"train / {int(stats['val_frac'] * 100)}% val / "
               f"{int(stats['test_frac'] * 100)}% test), stratified per class.\n")
    out.append("| Split | Images |")
    out.append("|---|---:|")
    out.append(f"| Train | {t['train']} |")
    out.append(f"| Validation | {t['val']} |")
    out.append(f"| Test | {t['test']} |")
    out.append(f"| **Total** | **{t['total']}** |")
    out.append("")
    out.append("Per class:\n")
    out.append("| Class | Total | Train | Val | Test |")
    out.append("|---|---:|---:|---:|---:|")
    for slug, v in stats["per_class"].items():
        out.append(f"| `{slug}` | {v['total']} | {v['train']} | {v['val']} "
                   f"| {v['test']} |")
    out.append("")


def experiments_section(out: list[str]) -> None:
    exp = load("artifacts/experiments.json")
    if not exp:
        out.append("_(experiment sweep not run yet)_\n")
        return
    out.append("| Run | What differs | Best val top-1 | Best val top-3 | "
               "Best epoch | Train time |")
    out.append("|---|---|---:|---:|---:|---:|")
    blurbs = {
        "baseline": "standard augmentation, fine-tune LR 1e-4",
        "strong_aug": "strong augmentation (rotation, grayscale, erasing)",
        "high_lr": "standard augmentation, fine-tune LR 1e-3 (10x)",
    }
    for r in exp["runs"]:
        star = " **(best)**" if r["run_name"] == exp["best"] else ""
        out.append(f"| `{r['run_name']}`{star} | "
                   f"{blurbs.get(r['run_name'], '-')} | "
                   f"{pct(r['val_top1'])} | {pct(r['val_top3'])} | "
                   f"{r['epoch']} | {r['train_minutes']:.0f} min |")
    out.append("")


def test_section(out: list[str]) -> None:
    ev = load("artifacts/best_model_test_eval.json")
    if not ev:
        out.append("_(test evaluation not run yet)_\n")
        return
    out.append(f"Held-out test split, **n = {ev['n']}**, never seen during "
               f"training or model selection.\n")
    out.append(f"| Metric | Value |")
    out.append("|---|---:|")
    out.append(f"| Top-1 accuracy | **{pct(ev['top1'])}** |")
    out.append(f"| Top-3 accuracy | **{pct(ev['top3'])}** |")
    out.append(f"| Macro F1 | {ev['macro_f1']:.3f} |")
    out.append("")
    out.append("Per class (sorted worst F1 first -- these are the honest weak "
               "spots):\n")
    out.append("| Class | Precision | Recall | F1 | Support |")
    out.append("|---|---:|---:|---:|---:|")
    rows = sorted(ev["per_class"].items(), key=lambda kv: kv[1]["f1-score"])
    for name, m in rows:
        out.append(f"| `{name}` | {m['precision']:.2f} | {m['recall']:.2f} | "
                   f"{m['f1-score']:.2f} | {int(m['support'])} |")
    out.append("")
    if ev.get("top_confusions"):
        out.append("Most frequent confusions:\n")
        out.append("| Count | True | Predicted |")
        out.append("|---:|---|---|")
        for c in ev["top_confusions"]:
            out.append(f"| {c['count']} | `{c['true']}` | `{c['pred']}` |")
        out.append("")


def benchmark_section(out: list[str]) -> None:
    b = load("artifacts/best_model_benchmark.json")
    if not b:
        out.append("_(benchmark not run yet)_\n")
        return
    h, c = b["host"], b["config"]
    out.append(f"Measured on **{h['processor'] or h['machine']}** "
               f"({h['system']} {h['release']}), CPU only "
               f"(`torch.cuda.is_available() == {h['cuda_available']}`), "
               f"torch {h['torch']}, onnxruntime {h['onnxruntime']}, "
               f"{h['threads']} threads.\n")
    out.append(f"Batch size 1, {c['runs']} timed runs after {c['warmup']} "
               f"warmup runs, all variants in one process. Accuracy is the "
               f"full test split (n = {c['test_n']}).\n")
    out.append("| Variant | Median | Mean | p95 | Speedup | Size | Test top-1 "
               "| Test top-3 |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    base = None
    for label, e in b["variants"].items():
        if "median_ms" not in e:
            out.append(f"| {label} | — | — | — | — | "
                       f"{e.get('size_mb', 0):.1f} MB | — | "
                       f"failed to load |")
            continue
        if base is None:
            base = e["median_ms"]
        out.append(
            f"| {label} | {e['median_ms']:.2f} ms | {e['mean_ms']:.2f} ms | "
            f"{e['p95_ms']:.2f} ms | {base / e['median_ms']:.2f}x | "
            f"{e['size_mb']:.1f} MB | {pct(e.get('top1'))} | "
            f"{pct(e.get('top3'))} |")
    out.append("")


def gradcam_section(out: list[str]) -> None:
    man = load("docs/gradcam/manifest.json")
    if not man:
        out.append("_(Grad-CAM examples not generated yet)_\n")
        return
    out.append(f"{len(man['correct'])} correct-prediction panels and "
               f"{len(man['errors'])} misclassification panels in "
               f"`docs/gradcam/`. Each panel is the model's 224x224 view on "
               f"the left and the Grad-CAM overlay on the right.\n")
    if man["errors"]:
        out.append("| Figure | True | Predicted | Confidence |")
        out.append("|---|---|---|---:|")
        for e in man["errors"]:
            out.append(f"| `{os.path.basename(e['file'])}` | `{e['true']}` | "
                       f"`{e['pred']}` | {pct(e['confidence'])} |")
        out.append("")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out: list[str] = ["# LandmarkLens measured results\n",
                      "_Generated by `scripts/make_report.py` from the "
                      "artifact JSON. Every figure here was measured, not "
                      "estimated._\n"]
    out.append("## Dataset\n")
    dataset_section(out)
    out.append("## Experiment runs (MLflow)\n")
    experiments_section(out)
    out.append("## Held-out test results\n")
    test_section(out)
    out.append("## Inference benchmark\n")
    benchmark_section(out)
    out.append("## Grad-CAM examples\n")
    gradcam_section(out)

    text = "\n".join(out)
    if args.out:
        path = os.path.join(ROOT, args.out)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {path}")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
