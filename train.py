"""
DenseFish-v13 — Training Script
================================
Implements all settings from Table 1 of the paper:
  • SGD, momentum=0.937, weight_decay=5e-4
  • Cosine learning-rate decay
  • 5-epoch warm-up, 100 epochs total
  • Repulsion Loss activated at epoch 50  (λ_rep = 0.2)
  • FP16 mixed precision (via torch.cuda.amp)
  • Logging: per-epoch metrics to console + optional W&B
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model            import build_densefish_v13
from modules.losses   import DenseFishDetectionLoss
from modules.bhfg     import HICLoss
from data.dataset     import build_dataloader
from utils.metrics    import DetectionMetrics, CountingMetrics
from utils.callbacks  import CheckpointCallback, EarlyStoppingCallback


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Learning-rate helpers
# ──────────────────────────────────────────────────────────────────────────────

def cosine_lr(optimizer: torch.optim.Optimizer,
              epoch: int, total_epochs: int,
              warmup_epochs: int, lr0: float, lr_min: float = 1e-5):
    """
    Linear warm-up for `warmup_epochs`, then cosine decay to `lr_min`.
    Modifies optimizer lr in-place.
    """
    if epoch < warmup_epochs:
        scale = (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        scale    = lr_min / lr0 + 0.5 * (1 - lr_min / lr0) * (
            1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr0 * scale
    return lr0 * scale


# ──────────────────────────────────────────────────────────────────────────────
# 2.  One epoch
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model: nn.Module,
                    criterion: DenseFishDetectionLoss,
                    hic_loss: HICLoss,
                    dataloader, optimizer: torch.optim.Optimizer,
                    scaler: GradScaler,
                    epoch: int, device: torch.device,
                    log_freq: int = 20) -> Dict[str, float]:
    model.train()
    criterion.set_epoch(epoch)

    running = {k: 0.0 for k in
               ["loss_total", "loss_cls", "loss_box", "loss_rep"]}
    n_batches = len(dataloader)

    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        with autocast():
            outputs = model(images)
            losses  = criterion(outputs, targets)
            loss    = losses["loss_total"]

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()

        for k in running:
            running[k] += losses.get(k, torch.zeros(1)).item()

        if (i + 1) % log_freq == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d} [{i+1:4d}/{n_batches}]  "
                  f"loss={running['loss_total']/(i+1):.4f}  "
                  f"box={running['loss_box']/(i+1):.4f}  "
                  f"cls={running['loss_cls']/(i+1):.4f}  "
                  f"rep={running['loss_rep']/(i+1):.4f}  "
                  f"lr={lr_now:.6f}")

    return {k: v / n_batches for k, v in running.items()}


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model: nn.Module, criterion: DenseFishDetectionLoss,
             dataloader, device: torch.device,
             conf_thresh: float = 0.5) -> Dict[str, float]:
    model.eval()
    det_metrics = DetectionMetrics(num_classes=1)
    cnt_metrics = CountingMetrics()
    val_loss    = 0.0

    for images, targets in dataloader:
        images  = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        with autocast():
            outputs = model(images)
            losses  = criterion(outputs, targets)

        val_loss += losses["loss_total"].item()

        # decode predictions
        preds = model.predict(images, conf_threshold=conf_thresh)
        for pred, tgt in zip(preds, targets):
            det_metrics.update(pred, tgt)
            cnt_metrics.update(pred["boxes"].shape[0], tgt["n_fish"])

    n = len(dataloader)
    results = det_metrics.compute()
    results.update(cnt_metrics.compute())
    results["val_loss"] = val_loss / n
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # ── data ──────────────────────────────────────────────────────────────────
    train_loader = build_dataloader(
        args.data_root, "train",
        batch_size=args.batch_size,
        num_workers=args.workers,
        img_size=args.img_size)
    val_loader = build_dataloader(
        args.data_root, "val",
        batch_size=args.batch_size // 2,
        num_workers=args.workers,
        img_size=args.img_size)

    print(f"  Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader)}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_densefish_v13(num_classes=args.num_classes,
                                 pretrained=args.pretrained)
    model = model.to(device)
    print(f"  Parameters: {model.count_parameters()/1e6:.1f} M")

    # ── losses ────────────────────────────────────────────────────────────────
    criterion = DenseFishDetectionLoss(
        lam_box=args.lam_box,
        lam_cls=args.lam_cls,
        lam_dfl=args.lam_dfl,
        lam_rep=args.lam_rep,
        tau=args.tau,
        num_classes=args.num_classes,
        img_size=args.img_size,
        repulsion_start_epoch=args.rep_start_epoch,
    ).to(device)
    hic = HICLoss(a_max=15.0, weight=0.05).to(device)

    # ── optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ── callbacks ─────────────────────────────────────────────────────────────
    ckpt_cb = CheckpointCallback(
        save_dir=args.output_dir,
        monitor="mAP@50:95",
        mode="max")
    es_cb = EarlyStoppingCallback(patience=20, monitor="mAP@50:95", mode="max")

    # ── training loop ─────────────────────────────────────────────────────────
    best_map = 0.0
    for epoch in range(args.epochs):
        t0 = time.time()

        # lr schedule
        lr_now = cosine_lr(optimizer, epoch, args.epochs,
                            args.warmup_epochs, args.lr)

        train_metrics = train_one_epoch(
            model, criterion, hic, train_loader,
            optimizer, scaler, epoch, device)

        val_metrics = validate(model, criterion, val_loader, device)

        elapsed = time.time() - t0
        print(f"\nEpoch {epoch:3d}/{args.epochs}  "
              f"lr={lr_now:.6f}  time={elapsed:.1f}s")
        print(f"  Train: {_fmt(train_metrics)}")
        print(f"  Val  : {_fmt(val_metrics)}\n")

        # callbacks
        ckpt_cb.step(epoch, val_metrics, model, optimizer)
        if es_cb.step(val_metrics):
            print("[EarlyStopping] Stopping training.")
            break

    print(f"\n[Train] Best mAP@50:95: {ckpt_cb.best_value:.4f}  "
          f"(saved at {ckpt_cb.best_path})")


def _fmt(d: Dict) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in d.items()
                     if isinstance(v, float))


# ──────────────────────────────────────────────────────────────────────────────
# 5.  CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DenseFish-v13 Training")
    # data
    p.add_argument("--data_root",  default="data/dense_aqua")
    p.add_argument("--img_size",   type=int, default=640)
    p.add_argument("--num_classes",type=int, default=1)
    # training
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr",         type=float, default=0.001)
    p.add_argument("--momentum",   type=float, default=0.937)
    p.add_argument("--weight_decay",type=float, default=5e-4)
    p.add_argument("--warmup_epochs",type=int, default=5)
    # loss
    p.add_argument("--lam_box",    type=float, default=5.0)
    p.add_argument("--lam_cls",    type=float, default=1.0)
    p.add_argument("--lam_dfl",    type=float, default=1.5)
    p.add_argument("--lam_rep",    type=float, default=0.2)
    p.add_argument("--tau",        type=float, default=0.5)
    p.add_argument("--rep_start_epoch", type=int, default=50)
    # misc
    p.add_argument("--device",     default="cuda:0")
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--output_dir", default="outputs/run1")
    p.add_argument("--pretrained", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
