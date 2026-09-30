"""Training script for open-set person re-identification.

Supports:
- ArcFace / CosFace margin loss  (config toggle)
- Batch-hard triplet loss         (config toggle)
- PK sampler for hard-negative mining
- Mixed-precision (fp16) training
- Linear warmup → cosine / step LR decay
"""
from __future__ import annotations

import os
import random
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from models.backbone import build_backbone
from models.heads import ReIDModel
from losses.arcface import ArcFaceLoss
from losses.cosface import CosFaceLoss
from losses.softmax import SoftmaxLoss
from losses.triplet import BatchHardTripletLoss
from data.market1501 import Market1501, split_known_unknown, get_transforms
from data.sampler import PKSampler


# ── Reproducibility ──────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── LR helpers ───────────────────────────────────────────────────────

def build_scheduler(optimizer, cfg):
    sched = cfg["train"]["scheduler"]
    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["train"]["epochs"], eta_min=1e-7
        )
    if sched == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg["train"]["step_size"],
            gamma=cfg["train"]["gamma"],
        )
    raise ValueError(f"Unknown scheduler: {sched}")


def warmup_lr(optimizer, epoch: int, warmup: int, base_lr: float):
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / warmup
        for g in optimizer.param_groups:
            g["lr"] = lr


# ── One epoch ────────────────────────────────────────────────────────

def train_one_epoch(model, margin_loss, triplet_loss, loader, optimizer,
                    scaler, device, cfg, epoch):
    model.train()
    if margin_loss is not None:
        margin_loss.train()

    sum_loss = sum_m = sum_t = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False)
    for imgs, labels, _, _ in pbar:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if (labels < 0).any():
            continue

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg["train"]["fp16"]):
            emb = model(imgs)

            loss = torch.tensor(0.0, device=device)
            ml = tl = torch.tensor(0.0, device=device)

            if margin_loss is not None:
                ml = margin_loss(emb, labels)
                loss = loss + cfg["loss"]["margin_weight"] * ml

            if triplet_loss is not None:
                tl = triplet_loss(emb, labels)
                loss = loss + cfg["loss"]["triplet_weight"] * tl

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        sum_loss += loss.item()
        sum_m    += ml.item()
        sum_t    += tl.item()
        n += 1

        pbar.set_postfix(loss=f"{sum_loss/n:.4f}",
                         margin=f"{sum_m/n:.4f}",
                         triplet=f"{sum_t/n:.4f}")

    return sum_loss / max(n, 1)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser("train.py — open-set re-id training")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--resume", default=None, help="Checkpoint to resume from")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Persist config for reproducibility
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    # ── Identity split ───────────────────────────────────────────────
    known_pids, unknown_pids = split_known_unknown(
        cfg["data"]["root"],
        unknown_ratio=cfg["openset"]["unknown_ratio"],
        seed=cfg["train"]["seed"],
    )
    print(f"[split] Known IDs: {len(known_pids)} | Unknown IDs: {len(unknown_pids)}")
    torch.save(
        {"known_pids": sorted(known_pids), "unknown_pids": sorted(unknown_pids)},
        os.path.join(out_dir, "splits.pth"),
    )

    # ── Dataset + loader ─────────────────────────────────────────────
    transform = get_transforms(cfg["data"]["height"], cfg["data"]["width"],
                               is_train=True)
    ds = Market1501(cfg["data"]["root"], split="train",
                    known_pids=known_pids, unknown_pids=unknown_pids,
                    transform=transform, relabel=True)
    sampler = PKSampler(ds, p=cfg["data"]["pk_p"], k=cfg["data"]["pk_k"])
    loader = DataLoader(
        ds,
        batch_size=cfg["data"]["pk_p"] * cfg["data"]["pk_k"],
        sampler=sampler,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    num_classes = ds.num_classes
    print(f"[data]  {len(ds)} images,  {num_classes} known classes")

    # ── Model ────────────────────────────────────────────────────────
    backbone = build_backbone(cfg["model"]["backbone"],
                              cfg["model"]["pretrained"])
    model = ReIDModel(backbone, cfg["model"]["embed_dim"]).to(device)

    # ── Losses ───────────────────────────────────────────────────────
    margin_loss = None
    mt = cfg["loss"]["margin_type"]
    if mt == "arcface":
        margin_loss = ArcFaceLoss(
            cfg["model"]["embed_dim"], num_classes,
            s=cfg["loss"]["arcface_s"], m=cfg["loss"]["arcface_m"],
        ).to(device)
    elif mt == "cosface":
        margin_loss = CosFaceLoss(
            cfg["model"]["embed_dim"], num_classes,
            s=cfg["loss"]["cosface_s"], m=cfg["loss"]["cosface_m"],
        ).to(device)
    elif mt == "softmax":
        margin_loss = SoftmaxLoss(
            cfg["model"]["embed_dim"], num_classes,
        ).to(device)
    elif mt != "none":
        raise ValueError(f"Unknown margin type: {mt}")

    triplet_loss = None
    if cfg["loss"]["triplet"]:
        triplet_loss = BatchHardTripletLoss(
            margin=cfg["loss"]["triplet_margin"],
            hard_mining=cfg["loss"].get("hard_mining", True),
        ).to(device)

    # ── Optimizer ────────────────────────────────────────────────────
    params = list(model.parameters())
    if margin_loss is not None:
        params += list(margin_loss.parameters())

    optimizer = torch.optim.Adam(
        params, lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, cfg)
    scaler = GradScaler(enabled=cfg["train"]["fp16"])

    # ── Resume ───────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        if margin_loss is not None and "margin_loss" in ckpt:
            margin_loss.load_state_dict(ckpt["margin_loss"])
        print(f"[resume] Starting from epoch {start_epoch}")

    # ── Training loop ────────────────────────────────────────────────
    best_loss = float("inf")
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        warmup_lr(optimizer, epoch, cfg["train"]["warmup_epochs"],
                  cfg["train"]["lr"])

        t0 = time.time()
        loss = train_one_epoch(model, margin_loss, triplet_loss, loader,
                               optimizer, scaler, device, cfg, epoch)
        elapsed = time.time() - t0

        if epoch >= cfg["train"]["warmup_epochs"]:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:>3d}/{cfg['train']['epochs']}  "
              f"loss={loss:.4f}  lr={lr:.2e}  ({elapsed:.1f}s)")

        # Checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss,
            "config": cfg,
        }
        if margin_loss is not None:
            ckpt["margin_loss"] = margin_loss.state_dict()

        if (epoch + 1) % cfg["output"]["save_freq"] == 0:
            p = os.path.join(out_dir, f"checkpoint_ep{epoch+1}.pth")
            torch.save(ckpt, p)

        if loss < best_loss:
            best_loss = loss
            torch.save(ckpt, os.path.join(out_dir, "best.pth"))
            print(f"  -> new best (loss={loss:.4f})")

    print("\n[OK] Training complete.")


if __name__ == "__main__":
    main()
