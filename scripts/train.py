"""Train LandmarkLens and log the run to a local MLflow tracking store.

Two-phase transfer learning: a frozen-backbone warmup for the randomly
initialised head, then fine-tuning of layer4 at a lower learning rate. The best
epoch by validation top-1 is what gets checkpointed and logged as the MLflow
model artifact -- not the last epoch.

    python scripts/train.py --run-name baseline --epochs-head 3 --epochs-ft 8

MLflow writes to ./mlruns by default (file store, no server needed). Browse it
afterwards with:  mlflow ui --backend-store-uri ./mlruns
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import mlflow
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from landmarklens.data import build_loaders, class_weights
from landmarklens.model import build_model, save_checkpoint, set_trainable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(ROOT, "data", "splits")
ARTIFACTS = os.path.join(ROOT, "artifacts")


def accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> int:
    """Number of samples whose true label is inside the top-k predictions."""
    k = min(k, logits.shape[1])
    top = logits.topk(k, dim=1).indices
    return (top == targets.unsqueeze(1)).any(dim=1).sum().item()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, n, c1, c3 = 0.0, 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * y.size(0)
        n += y.size(0)
        c1 += accuracy(out, y, 1)
        c3 += accuracy(out, y, 3)
    return loss_sum / max(n, 1), c1 / max(n, 1), c3 / max(n, 1)


def run_phase(model, loader, val_loader, criterion, optimizer, scheduler,
              device, epochs, phase, epoch_offset, best, ckpt_path,
              classes, backbone, args):
    for e in range(epochs):
        epoch = epoch_offset + e + 1
        model.train()
        t0 = time.time()
        loss_sum, n, c1 = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            n += y.size(0)
            c1 += accuracy(out, y, 1)
        if scheduler is not None:
            scheduler.step()

        tr_loss, tr_acc = loss_sum / max(n, 1), c1 / max(n, 1)
        va_loss, va_acc, va_acc3 = evaluate(model, val_loader, criterion, device)
        secs = time.time() - t0

        mlflow.log_metrics({
            "train_loss": tr_loss, "train_top1": tr_acc,
            "val_loss": va_loss, "val_top1": va_acc, "val_top3": va_acc3,
            "epoch_seconds": secs,
            "lr": optimizer.param_groups[0]["lr"],
        }, step=epoch)

        star = ""
        if va_acc > best["val_top1"]:
            best.update(val_top1=va_acc, val_top3=va_acc3, epoch=epoch,
                        phase=phase)
            save_checkpoint(ckpt_path, model, classes, backbone, meta={
                "run_name": args.run_name, "best_epoch": epoch,
                "val_top1": va_acc, "val_top3": va_acc3, "phase": phase,
                "aug": args.aug, "image_size": args.image_size,
            })
            star = "  *best"

        print(f"[{phase}] epoch {epoch:>2}  {secs:6.1f}s  "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f}  |  "
              f"val loss {va_loss:.4f} top1 {va_acc:.3f} top3 {va_acc3:.3f}{star}",
              flush=True)
    return epoch_offset + epochs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--experiment", default="landmarklens")
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--epochs-head", type=int, default=3)
    ap.add_argument("--epochs-ft", type=int, default=8)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-ft", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--aug", default="standard",
                    choices=["light", "standard", "strong"])
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--balanced-loss", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", default=SPLITS,
                    help="dataset root containing train/val/test")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch CPU threads; 0 = leave default")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits = args.splits
    train_loader, val_loader, test_loader, classes = build_loaders(
        splits, batch_size=args.batch_size, workers=args.workers,
        aug=args.aug, size=args.image_size)
    print(f"device={device} threads={torch.get_num_threads()} "
          f"classes={len(classes)} "
          f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"test={len(test_loader.dataset)}", flush=True)

    model = build_model(len(classes), backbone=args.backbone,
                        dropout=args.dropout).to(device)

    weights = (class_weights(splits, classes).to(device)
               if args.balanced_loss else None)
    criterion = nn.CrossEntropyLoss(weight=weights,
                                    label_smoothing=args.label_smoothing)

    os.makedirs(ARTIFACTS, exist_ok=True)
    ckpt_path = os.path.join(ARTIFACTS, f"{args.run_name}_best.pt")

    mlflow.set_tracking_uri("file:" + os.path.join(ROOT, "mlruns").replace("\\", "/"))
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params({
            "backbone": args.backbone, "epochs_head": args.epochs_head,
            "epochs_ft": args.epochs_ft, "lr_head": args.lr_head,
            "lr_ft": args.lr_ft, "batch_size": args.batch_size,
            "augmentation": args.aug, "image_size": args.image_size,
            "dropout": args.dropout, "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "balanced_loss": args.balanced_loss, "seed": args.seed,
            "device": device, "num_classes": len(classes),
            "n_train": len(train_loader.dataset),
            "n_val": len(val_loader.dataset),
            "n_test": len(test_loader.dataset),
        })

        best = {"val_top1": -1.0, "val_top3": -1.0, "epoch": -1, "phase": ""}
        t_start = time.time()

        # --- phase 1: head only --------------------------------------------
        params = set_trainable(model, "head")
        opt = torch.optim.AdamW(params, lr=args.lr_head,
                                weight_decay=args.weight_decay)
        done = run_phase(model, train_loader, val_loader, criterion, opt, None,
                         device, args.epochs_head, "head", 0, best, ckpt_path,
                         classes, args.backbone, args)

        # --- phase 2: fine-tune layer4 -------------------------------------
        if args.epochs_ft > 0:
            params = set_trainable(model, "layer4")
            opt = torch.optim.AdamW(params, lr=args.lr_ft,
                                    weight_decay=args.weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.epochs_ft)
            run_phase(model, train_loader, val_loader, criterion, opt, sched,
                      device, args.epochs_ft, "finetune", done, best,
                      ckpt_path, classes, args.backbone, args)

        total_min = (time.time() - t_start) / 60
        print(f"\nbest val top1={best['val_top1']:.4f} "
              f"top3={best['val_top3']:.4f} at epoch {best['epoch']} "
              f"({best['phase']}); {total_min:.1f} min total", flush=True)

        mlflow.log_metrics({
            "best_val_top1": best["val_top1"],
            "best_val_top3": best["val_top3"],
            "best_epoch": best["epoch"],
            "train_minutes": total_min,
        })
        mlflow.log_artifact(ckpt_path, artifact_path="checkpoint")
        mlflow.log_artifact(os.path.splitext(ckpt_path)[0] + "_classes.json",
                            artifact_path="checkpoint")
        split_stats = os.path.join(ROOT, "data", "split_stats.json")
        if os.path.exists(split_stats):
            mlflow.log_artifact(split_stats, artifact_path="dataset")

        summary = {"run_name": args.run_name, "run_id": run.info.run_id,
                   "checkpoint": ckpt_path, **best,
                   "train_minutes": total_min}
        with open(os.path.join(ARTIFACTS, f"{args.run_name}_train.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
