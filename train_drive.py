#!/usr/bin/env python3
"""
DRIVE deep-supervision lambda grid training.

Examples:
  python train_drive.py --data_dir ./data/DRIVE --save_dir ./models/drive --lambda_ds 0.1 --seed 42
  python train_drive.py --data_dir ./data/DRIVE --save_dir ./models/drive --seed 42 --grid_mode
"""

import os
import json
import argparse
from datetime import datetime
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from improved_pdc_convolutions import normalize_config_string
from pdc_unet_model import create_model
from drive_preprocessing import get_data_loaders
from path_compat import apply_model_path_aliases


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum(dim=2).sum(dim=2) +
            target.sum(dim=2).sum(dim=2) +
            self.smooth
        )
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def set_random_seed(seed: int) -> None:
    print(f"Set random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_deep_supervision_loss(outputs, targets, criterion, lambda_ds: float):
    if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
        p0, p1, p2 = outputs
    else:
        p0 = outputs
        p1, p2 = None, None

    loss0 = criterion(p0, targets)
    if lambda_ds <= 0.0 or p1 is None or p2 is None:
        return loss0

    target_size = targets.shape[2:]
    p1_up = F.interpolate(p1, size=target_size, mode="bilinear", align_corners=False)
    p2_up = F.interpolate(p2, size=target_size, mode="bilinear", align_corners=False)

    loss1 = criterion(p1_up, targets)
    loss2 = criterion(p2_up, targets)
    return loss0 + lambda_ds * (loss1 + 0.5 * loss2)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch: int, lambda_ds: float, exp_name: str):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc=f"{exp_name} - Epoch {epoch}")
    for batch_idx, (images, masks) in enumerate(progress_bar):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = compute_deep_supervision_loss(outputs, masks, criterion, lambda_ds)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
        })

    return total_loss / max(1, num_batches)


def validate_epoch(model, dataloader, criterion, device, lambda_ds: float):
    """Run validation for one epoch."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = compute_deep_supervision_loss(outputs, masks, criterion, lambda_ds)
            total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


def plot_training_curves(history, save_dir: str, exp_name: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Training Loss")
    ax1.plot(epochs, history["val_loss"], label="Validation Loss")
    ax1.set_title(f"{exp_name} - Loss Curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["learning_rates"], label="Learning Rate")
    ax2.set_title(f"{exp_name} - Learning Rate")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def train_single_lambda(lambda_ds: float, args, device, train_loader, val_loader) -> None:
    lambda_str = f"{lambda_ds:.1f}".replace(".", "_")
    lambda_dir = f"lambda_{lambda_str}"
    seed_dir = f"seed{args.seed}"

    exp_root = os.path.join(args.save_dir, lambda_dir, seed_dir)
    os.makedirs(exp_root, exist_ok=True)
    checkpoint_path = os.path.join(exp_root, "best_model.pth")
    exp_name = f"lambda_{lambda_ds:.2f}_seed{args.seed}"

    print("\n######################################################################")
    print(f"Starting DRIVE experiment: {exp_name}")
    print("Description: gpdc_only + SDPM backbone + deep supervision")
    print(f"use_deep_supervision = True, lambda_ds = {lambda_ds:.2f}")
    print(f"Save directory: {exp_root}")
    print("######################################################################\n")

    if os.path.exists(checkpoint_path) and not args.overwrite:
        print(f"Checkpoint already exists: {checkpoint_path}")
        print("Skip this lambda. Use --overwrite to retrain.")
        return

    model = create_model(
        config_str=args.fixed_config,
        channels=args.fixed_channels,
        use_gpdc=True,
        use_residual=False,
        use_lmm=False,
        use_sdpm=True,
        use_deep_supervision=True,
    ).to(device)

    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rates": [],
        "experiment_info": {
            "lambda_ds": lambda_ds,
            "seed": args.seed,
            "config": args.fixed_config,
            "channels": args.fixed_channels,
            "use_gpdc": True,
            "use_residual": False,
            "use_lmm": False,
            "use_sdpm": True,
            "use_deep_supervision": True,
            "parameters": param_count,
        },
    }

    best_val_loss = float("inf")
    patience_counter = 0
    start_time = datetime.now()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, lambda_ds, exp_name
        )
        val_loss = validate_epoch(model, val_loader, criterion, device, lambda_ds)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rates"].append(current_lr)

        print(
            f"Epoch {epoch:3d}: "
            f"Train Loss = {train_loss:.6f}, "
            f"Val Loss = {val_loss:.6f}, "
            f"LR = {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "lambda_ds": lambda_ds,
                    "seed": args.seed,
                    "config": args.fixed_config,
                    "channels": args.fixed_channels,
                    "use_gpdc": True,
                    "use_residual": False,
                    "use_lmm": False,
                    "use_sdpm": True,
                    "use_deep_supervision": True,
                    "parameters": param_count,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"  -> Saved best checkpoint (Val Loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {args.patience} stagnant epochs.")
            break

    duration = datetime.now() - start_time
    history_path = os.path.join(exp_root, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    plot_training_curves(history, exp_root, exp_name)

    print(f"\nFinished experiment: {exp_name}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Duration: {duration}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History JSON: {history_path}")


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def main() -> None:
    parser = argparse.ArgumentParser(description="DRIVE deep-supervision lambda grid training")

    parser.add_argument("--data_dir", type=str, default="./data/DRIVE", help="Dataset root.")
    parser.add_argument("--save_dir", type=str, default="./models/drive", help="Checkpoint root.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--patch_size", type=int, default=48, help="Patch size.")
    parser.add_argument("--stride", type=int, default=16, help="Patch stride.")
    parser.add_argument("--num_workers", type=int, default=4, help="Data loader workers.")
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing checkpoints.")

    parser.add_argument("--lambda_ds", type=float, default=0.1, help="Single-run deep supervision weight.")
    parser.add_argument("--grid_mode", action="store_true", help="Run all lambdas in --lambda_list.")
    parser.add_argument(
        "--lambda_list",
        type=str,
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated lambda list for grid mode.",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fixed_config", type=str, default="C-[V]脳11", help="Backbone config string.")
    parser.add_argument("--fixed_channels", type=int, default=32, help="Backbone channels.")

    args = parser.parse_args()
    try:
        args.fixed_config = normalize_config_string(args.fixed_config)
    except ValueError:
        args.fixed_config = "C-[V]x11"
    apply_model_path_aliases(args, ("save_dir",))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_random_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, val_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        stride=args.stride,
        num_workers=args.num_workers,
    )

    print("\n######################################################################")
    print("DRIVE training configuration")
    print(f"- data_dir     : {args.data_dir}")
    print(f"- save_dir     : {args.save_dir}")
    print(f"- config       : {args.fixed_config}")
    print(f"- channels     : {args.fixed_channels}")
    print(f"- batch_size   : {args.batch_size}")
    print(f"- epochs       : {args.epochs}")
    print(f"- lr           : {args.learning_rate}")
    print(f"- patch_size   : {args.patch_size}")
    print(f"- stride       : {args.stride}")
    print(f"- patience     : {args.patience}")
    print(f"- seed         : {args.seed}")
    print(f"- grid_mode    : {args.grid_mode}")
    if args.grid_mode:
        print(f"- lambda_list  : {args.lambda_list}")
    else:
        print(f"- lambda_ds    : {args.lambda_ds}")
    print("######################################################################\n")

    start_time_all = datetime.now()

    if args.grid_mode:
        lambda_list = parse_float_list(args.lambda_list)
        print(f"Grid mode lambda list: {lambda_list}")
        for index, lambda_ds in enumerate(lambda_list, start=1):
            print(f"\n======== Grid progress: {index}/{len(lambda_list)}, lambda={lambda_ds:.2f} ========")
            train_single_lambda(lambda_ds, args, device, train_loader, val_loader)
    else:
        print(f"Single-run mode: lambda_ds = {args.lambda_ds:.2f}")
        train_single_lambda(args.lambda_ds, args, device, train_loader, val_loader)

    duration_all = datetime.now() - start_time_all
    print("\n============================================================")
    print("DRIVE training finished")
    print(f"Total duration: {duration_all}")
    print("============================================================")


if __name__ == "__main__":
    main()
