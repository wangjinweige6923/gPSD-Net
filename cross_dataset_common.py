#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm

import eval_drive_source_cross_dataset as eval_base
from unet_baseline_model import create_unet_baseline
from path_compat import apply_model_path_aliases

try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image
except Exception:
    Image = None


ALLOWED_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ""}
OURS_MODEL_KEY = "ds_lambda_0_2"
BASELINE_MODEL_KEY = "baseline_c32"
BASELINE_SAVE_SUBDIR = "baseline"
DEFAULT_OURS_LAMBDA_LIST = "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"

_IMG_CACHE: Dict[str, np.ndarray] = {}
_MASK_CACHE: Dict[str, np.ndarray] = {}
_FOV_CACHE: Dict[str, np.ndarray] = {}


@dataclass(frozen=True)
class DatasetProtocol:
    name: str
    train_patch_size: int
    train_stride: int
    eval_patch_size: int
    eval_stride: int
    default_threshold: float
    threshold_rule: str
    chase_mask_type: str = "1stHO"
    min_foreground_ratio: float = 0.0
    augment: bool = False
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    weight_decay: float = 0.0


DATASET_PROTOCOLS: Dict[str, DatasetProtocol] = {
    "DRIVE": DatasetProtocol(
        name="DRIVE",
        train_patch_size=48,
        train_stride=16,
        eval_patch_size=48,
        eval_stride=16,
        default_threshold=0.4,
        threshold_rule="max_se_at_base_f1",
    ),
    "STARE": DatasetProtocol(
        name="STARE",
        train_patch_size=48,
        train_stride=16,
        eval_patch_size=48,
        eval_stride=16,
        default_threshold=0.4,
        threshold_rule="max_se_at_base_f1",
    ),
    "CHASEDB1": DatasetProtocol(
        name="CHASEDB1",
        train_patch_size=96,
        train_stride=32,
        eval_patch_size=96,
        eval_stride=32,
        default_threshold=0.4,
        threshold_rule="chase",
        chase_mask_type="1stHO",
        min_foreground_ratio=0.01,
        augment=True,
        brightness_range=(0.8, 1.2),
        contrast_range=(0.8, 1.2),
        weight_decay=1e-4,
    ),
    "SCCD": DatasetProtocol(
        name="SCCD",
        train_patch_size=48,
        train_stride=16,
        eval_patch_size=48,
        eval_stride=16,
        default_threshold=0.4,
        threshold_rule="max_se_at_base_f1",
    ),
    "ISIC2018": DatasetProtocol(
        name="ISIC2018",
        train_patch_size=48,
        train_stride=16,
        eval_patch_size=48,
        eval_stride=16,
        default_threshold=0.4,
        threshold_rule="max_se_at_base_f1",
    ),
}


PRESETS: Dict[str, Dict[str, object]] = {
    "drive_chase_to_stare": {
        "display_name": "DRIVE + CHASEDB1 -> STARE",
        "sources": ["DRIVE", "CHASEDB1"],
        "target": "STARE",
    },
    "drive_stare_to_chase": {
        "display_name": "DRIVE + STARE -> CHASEDB1",
        "sources": ["DRIVE", "STARE"],
        "target": "CHASEDB1",
    },
    "stare_chase_to_drive": {
        "display_name": "STARE + CHASEDB1 -> DRIVE",
        "sources": ["STARE", "CHASEDB1"],
        "target": "DRIVE",
    },
    "sccd_to_sccd": {
        "display_name": "SCCD -> SCCD",
        "sources": ["SCCD"],
        "target": "SCCD",
        "models_dir": os.path.join(".", "models", "sccd_filtered"),
        "results_dir": os.path.join(".", "results", "sccd_filtered"),
        "artifact_prefix": "sccd_filtered",
    },
    "isic2018_to_isic2018": {
        "display_name": "ISIC2018 -> ISIC2018",
        "sources": ["ISIC2018"],
        "target": "ISIC2018",
        "models_dir": os.path.join(".", "models", "isic2018"),
        "results_dir": os.path.join(".", "results", "isic2018"),
        "artifact_prefix": "isic2018",
    },
}


PRESET_OVERRIDES: Dict[str, Dict[str, Dict[str, object]]] = {
    "stare_chase_to_drive": {
        "train": {
            "normalize_input": True,
            "source_balance": True,
        },
        "eval": {
            "baseline_search_threshold": True,
            "incremental_save": True,
        },
    },
    "drive_stare_to_chase": {
        "train": {
            "patch_size": 96,
            "stride": 32,
            "weight_decay": 1e-4,
            "val_ratio": 0.1,
            "normalize_input": True,
            "source_balance": True,
            "force_source_augment": True,
            "force_min_foreground_ratio": 0.02,
            "force_source_use_fov_mask": True,
            "force_fov_min_ratio": 0.20,
            "save_recent_checkpoints": False,
            "recent_checkpoint_keep": 0,
        },
        "eval": {
            "patch_size": 96,
            "stride": 32,
            "normalize_input": True,
            "chase_use_fov_mask": True,
            "baseline_search_threshold": True,
            "thr_min": 0.01,
            "thr_max": 0.20,
            "chase_optimize_mode": "f1",
            "checkpoint_policy": "best",
            "recent_checkpoint_keep": 0,
            "target_resize_scale": 0.5,
            "target_resize_interpolation": "area",
        },
    },
    "sccd_to_sccd": {
        "train": {
            "ours_lambda_ds": 0.0,
            "ours_lambda_list": "0.0",
            "ours_grid_mode": False,
            "val_ratio": 0.2,
            "normalize_input": True,
            "save_recent_checkpoints": False,
        },
        "eval": {
            "ours_lambda_list": "0.0",
            "baseline_thr": 0.4,
            "ours_thr": 0.4,
            "baseline_search_threshold": False,
            "ours_search_threshold": False,
            "normalize_input": True,
            "patch_size": 48,
            "stride": 16,
        },
    },
    "isic2018_to_isic2018": {
        "train": {
            "ours_lambda_ds": 0.0,
            "ours_lambda_list": "0.0",
            "ours_grid_mode": False,
            "val_ratio": 0.2,
            "normalize_input": True,
            "save_recent_checkpoints": False,
        },
        "eval": {
            "ours_lambda_list": "0.0",
            "baseline_thr": 0.4,
            "ours_thr": 0.4,
            "baseline_search_threshold": False,
            "ours_search_threshold": False,
            "normalize_input": True,
            "patch_size": 48,
            "stride": 16,
        },
    },
}


def get_preset_override(preset_name: str, stage: str, key: str, default=None):
    return PRESET_OVERRIDES.get(preset_name, {}).get(stage, {}).get(key, default)


def _fmt_lambda(lambda_ds: float) -> str:
    return f"{lambda_ds:.1f}".replace(".", "_")


def default_models_dir_for_preset(preset_name: str) -> str:
    preset = PRESETS.get(preset_name, {})
    explicit = preset.get("models_dir")
    if explicit:
        return str(explicit)
    return os.path.join(".", "models", "cross_dataset", preset_name)


def default_results_dir_for_preset(preset_name: str) -> str:
    preset = PRESETS.get(preset_name, {})
    explicit = preset.get("results_dir")
    if explicit:
        return str(explicit)
    return os.path.join(".", "results", "cross_dataset", preset_name)


def default_artifact_prefix_for_preset(preset_name: str) -> str:
    preset = PRESETS.get(preset_name, {})
    return str(preset.get("artifact_prefix", "cross_dataset"))


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_range(text: str, default: Tuple[float, float]) -> Tuple[float, float]:
    vals = parse_float_list(text) if text is not None else []
    if len(vals) != 2:
        return default
    lo, hi = vals
    return (min(lo, hi), max(lo, hi))


def safe_convert_for_json(obj):
    if isinstance(obj, dict):
        return {k: safe_convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_convert_for_json(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        try:
            return obj.to_dict()
        except Exception:
            return str(obj)
    return str(obj)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _find_subdir(root: str, candidates: Sequence[str]) -> Optional[str]:
    for cand in candidates:
        path = os.path.join(root, cand)
        if os.path.isdir(path):
            return path
    return None


def _list_files(folder: str) -> List[str]:
    path = Path(folder)
    if not path.exists():
        return []
    files = [str(x) for x in path.iterdir() if x.is_file() and x.suffix.lower() in ALLOWED_EXTS]
    files.sort()
    return files


def _read_gray(path: str) -> np.ndarray:
    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim == 3:
            img = img[:, :, 1]
        img = img.astype(np.float32)
    else:
        if Image is None:
            raise ImportError("Missing cv2 and PIL, cannot read image.")
        img = np.array(Image.open(path))
        if img.ndim == 3:
            img = img[:, :, 1]
        img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return img


def _read_mask(path: str) -> np.ndarray:
    return (_read_gray(path) > 0.5).astype(np.float32)


def _compute_fov_mask(
    img: np.ndarray,
    thresh: float = 0.05,
    blur_ksize: int = 7,
    close_ksize: int = 15,
) -> np.ndarray:
    out = img.copy()
    if cv2 is not None and blur_ksize and blur_ksize > 0:
        k = int(blur_ksize)
        if k % 2 == 0:
            k += 1
        out = cv2.GaussianBlur(out, (k, k), 0)
    mask = (out > float(thresh)).astype(np.uint8)
    if cv2 is not None and close_ksize and close_ksize > 0:
        k = int(close_ksize)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(np.float32)


def _candidate_mask_stems(image_stem: str, mask_type: str = "1stHO") -> List[str]:
    suffix = "1stHO" if str(mask_type).lower() != "2ndho" else "2ndHO"
    candidates: List[str] = []
    if "_rot" in image_stem:
        base, rot = image_stem.rsplit("_rot", 1)
        candidates.extend([
            f"{base}_rot{rot}_{suffix}",
            f"{base}_{suffix}_rot{rot}",
            f"{image_stem}_{suffix}",
        ])
    else:
        candidates.append(f"{image_stem}_{suffix}")
    candidates.append(image_stem)
    return candidates


def _pair_images_and_masks(
    dataset_name: str,
    image_dir: str,
    mask_dir: str,
    chase_mask_type: str,
) -> List[Dict[str, str]]:
    image_files = _list_files(image_dir)
    if not image_files:
        raise RuntimeError(f"No images found: {image_dir}")

    mask_map = {Path(m).stem: m for m in _list_files(mask_dir)}
    pairs: List[Dict[str, str]] = []
    missing: List[str] = []

    for image_path in image_files:
        stem = Path(image_path).stem
        mask_path = None
        if dataset_name == "CHASEDB1":
            for cand in _candidate_mask_stems(stem, chase_mask_type):
                if cand in mask_map:
                    mask_path = mask_map[cand]
                    break
        else:
            mask_path = mask_map.get(stem)

        if mask_path is None:
            same_name = os.path.join(mask_dir, Path(image_path).name)
            same_png = os.path.join(mask_dir, f"{stem}.png")
            if os.path.exists(same_name):
                mask_path = same_name
            elif os.path.exists(same_png):
                mask_path = same_png

        if mask_path is None:
            missing.append(Path(image_path).name)
            continue

        pairs.append({"source": dataset_name, "image_path": image_path, "mask_path": mask_path})

    if missing:
        print(f"[Warn] {dataset_name}: {len(missing)} images missing masks, e.g. {missing[:5]}")
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found for {dataset_name}: {image_dir}")
    return pairs


def _collect_split_pairs(dataset_name: str, split_root: str, chase_mask_type: str) -> List[Dict[str, str]]:
    image_dir = _find_subdir(split_root, ["im", "image", "images"])
    mask_dir = _find_subdir(split_root, ["label", "labels"])
    if image_dir is None or mask_dir is None:
        return []
    if not _list_files(image_dir) or not _list_files(mask_dir):
        return []
    return _pair_images_and_masks(dataset_name, image_dir, mask_dir, chase_mask_type)


def collect_train_val_pairs(
    dataset_name: str,
    data_root: str,
    seed: int,
    val_ratio: float,
    chase_mask_type: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    dataset_root = os.path.join(data_root, dataset_name)
    train_pairs_all = _collect_split_pairs(dataset_name, os.path.join(dataset_root, "train"), chase_mask_type)
    if not train_pairs_all:
        raise RuntimeError(f"{dataset_name}/train has no valid image-label pairs.")

    val_pairs = _collect_split_pairs(dataset_name, os.path.join(dataset_root, "validate"), chase_mask_type)
    if val_pairs:
        return train_pairs_all, val_pairs

    rng = np.random.RandomState(seed)
    n = len(train_pairs_all)
    order = np.arange(n)
    rng.shuffle(order)
    n_val = max(1, int(round(n * float(val_ratio))))
    val_idx = set(order[:n_val].tolist())
    train_pairs = [train_pairs_all[i] for i in range(n) if i not in val_idx]
    val_pairs = [train_pairs_all[i] for i in range(n) if i in val_idx]
    return train_pairs, val_pairs


class MultiSourcePatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pairs: Sequence[Dict[str, str]],
        patch_size: int,
        stride: int,
        source_options: Dict[str, Dict[str, object]],
        training: bool,
    ):
        self.pairs = list(pairs)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.source_options = source_options
        self.training = bool(training)
        self.index: List[Tuple[int, int, int]] = []

        for idx, record in enumerate(self.pairs):
            source = str(record["source"])
            opts = self.source_options[source]
            image_path = str(record["image_path"])
            mask_path = str(record["mask_path"])
            h, w = self._get_hw(image_path)
            if h < self.patch_size or w < self.patch_size:
                continue

            fov = None
            if bool(opts["use_fov_mask"]) and float(opts["fov_min_ratio"]) > 0.0:
                if image_path not in _IMG_CACHE:
                    _IMG_CACHE[image_path] = _read_gray(image_path)
                if image_path not in _FOV_CACHE:
                    _FOV_CACHE[image_path] = _compute_fov_mask(
                        _IMG_CACHE[image_path],
                        thresh=float(opts["fov_threshold"]),
                        blur_ksize=int(opts["fov_blur"]),
                        close_ksize=int(opts["fov_close"]),
                    )
                fov = _FOV_CACHE[image_path]

            need_prefilter = float(opts["min_foreground_ratio"]) > 0.0
            mask = _read_mask(mask_path) if need_prefilter else None

            for y in range(0, h - self.patch_size + 1, self.stride):
                for x in range(0, w - self.patch_size + 1, self.stride):
                    if fov is not None and fov[y:y + self.patch_size, x:x + self.patch_size].mean() < float(opts["fov_min_ratio"]):
                        continue
                    if mask is not None and mask[y:y + self.patch_size, x:x + self.patch_size].mean() < float(opts["min_foreground_ratio"]):
                        continue
                    self.index.append((idx, y, x))

        if not self.index:
            raise RuntimeError("Empty patch index. Check patch_size/stride and dataset contents.")

    @staticmethod
    def _get_hw(image_path: str) -> Tuple[int, int]:
        if Image is not None:
            with Image.open(image_path) as img:
                w, h = img.size
            return h, w
        if cv2 is not None:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(image_path)
            return int(img.shape[0]), int(img.shape[1])
        raise ImportError("Missing cv2 and PIL, cannot read image size.")

    def __len__(self) -> int:
        return len(self.index)

    def _apply_augmentation(
        self,
        image_patch: np.ndarray,
        mask_patch: np.ndarray,
        brightness_range: Tuple[float, float],
        contrast_range: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > 0.5:
            k = random.randint(1, 3)
            image_patch = np.rot90(image_patch, k).copy()
            mask_patch = np.rot90(mask_patch, k).copy()
        if random.random() > 0.5:
            image_patch = np.fliplr(image_patch).copy()
            mask_patch = np.fliplr(mask_patch).copy()
        if random.random() > 0.5:
            image_patch = np.flipud(image_patch).copy()
            mask_patch = np.flipud(mask_patch).copy()
        if random.random() > 0.5:
            lo, hi = brightness_range
            image_patch = np.clip(image_patch * random.uniform(lo, hi), 0.0, 1.0)
        if random.random() > 0.5:
            lo, hi = contrast_range
            alpha = random.uniform(lo, hi)
            image_patch = np.clip((image_patch - 0.5) * alpha + 0.5, 0.0, 1.0)
        return image_patch, mask_patch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pair_idx, y, x = self.index[idx]
        record = self.pairs[pair_idx]
        source = str(record["source"])
        opts = self.source_options[source]
        image_path = str(record["image_path"])
        mask_path = str(record["mask_path"])

        if image_path not in _IMG_CACHE:
            _IMG_CACHE[image_path] = _read_gray(image_path)
        if mask_path not in _MASK_CACHE:
            _MASK_CACHE[mask_path] = _read_mask(mask_path)
        if bool(opts["use_fov_mask"]) and image_path not in _FOV_CACHE:
            _FOV_CACHE[image_path] = _compute_fov_mask(
                _IMG_CACHE[image_path],
                thresh=float(opts["fov_threshold"]),
                blur_ksize=int(opts["fov_blur"]),
                close_ksize=int(opts["fov_close"]),
            )

        image = _IMG_CACHE[image_path]
        mask = _MASK_CACHE[mask_path]
        image_patch = image[y:y + self.patch_size, x:x + self.patch_size]
        mask_patch = mask[y:y + self.patch_size, x:x + self.patch_size]

        if bool(opts["use_fov_mask"]):
            fov = _FOV_CACHE.get(image_path)
            if fov is not None:
                fov_patch = fov[y:y + self.patch_size, x:x + self.patch_size]
                image_patch = image_patch * fov_patch
                mask_patch = mask_patch * fov_patch

        if self.training and bool(opts["augment"]):
            image_patch, mask_patch = self._apply_augmentation(
                image_patch,
                mask_patch,
                brightness_range=tuple(opts["brightness_range"]),
                contrast_range=tuple(opts["contrast_range"]),
            )

        if bool(opts.get("normalize_input", False)):
            image_patch = (image_patch - np.mean(image_patch)) / (np.std(image_patch) + 1e-8)

        image_t = torch.from_numpy(image_patch[None, ...].astype(np.float32))
        mask_t = torch.from_numpy(mask_patch[None, ...].astype(np.float32))
        return image_t, mask_t

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth
        )
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(pred, target) + self.dice_weight * self.dice(pred, target)


def compute_deep_supervision_loss(outputs, targets: torch.Tensor, criterion: nn.Module, lambda_ds: float) -> torch.Tensor:
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


def train_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    lambda_ds: float,
    use_deep_supervision: bool,
    epoch_idx: int,
    exp_name: str,
) -> float:
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"{exp_name} - Epoch {epoch_idx}")
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = compute_deep_supervision_loss(outputs, masks, criterion, lambda_ds) if use_deep_supervision else criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss / (batch_idx + 1):.4f}")
    return total_loss / max(1, len(dataloader))


def validate_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    lambda_ds: float,
    use_deep_supervision: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            outputs = model(images)
            loss = compute_deep_supervision_loss(outputs, masks, criterion, lambda_ds) if use_deep_supervision else criterion(outputs, masks)
            total_loss += float(loss.item())
    return total_loss / max(1, len(dataloader))


def plot_training_curves(history: Dict[str, List[float]], save_dir: str, exp_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{exp_name} - Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(epochs, history["learning_rates"], label="Learning Rate")
    axes[1].set_title(f"{exp_name} - Learning Rate")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def resolve_combo_training_protocol(sources: Sequence[str]) -> Tuple[int, int]:
    patch_size = max(DATASET_PROTOCOLS[src].train_patch_size for src in sources)
    stride = max(DATASET_PROTOCOLS[src].train_stride for src in sources)
    return int(patch_size), int(stride)


def resolve_combo_weight_decay(sources: Sequence[str]) -> float:
    return float(max(DATASET_PROTOCOLS[src].weight_decay for src in sources))


def resolve_combo_training_protocol_for_preset(preset_name: str, sources: Sequence[str]) -> Tuple[int, int]:
    patch_size, stride = resolve_combo_training_protocol(sources)
    patch_size = int(get_preset_override(preset_name, "train", "patch_size", patch_size))
    stride = int(get_preset_override(preset_name, "train", "stride", stride))
    return patch_size, stride


def resolve_combo_weight_decay_for_preset(preset_name: str, sources: Sequence[str]) -> float:
    default_weight_decay = resolve_combo_weight_decay(sources)
    return float(get_preset_override(preset_name, "train", "weight_decay", default_weight_decay))


def resolve_source_options(preset_name: str, sources: Sequence[str], args: argparse.Namespace) -> Dict[str, Dict[str, object]]:
    options: Dict[str, Dict[str, object]] = {}
    force_source_augment = bool(get_preset_override(preset_name, "train", "force_source_augment", False))
    force_min_foreground_ratio = get_preset_override(preset_name, "train", "force_min_foreground_ratio", None)
    force_source_use_fov_mask = bool(get_preset_override(preset_name, "train", "force_source_use_fov_mask", False))
    force_fov_min_ratio = get_preset_override(preset_name, "train", "force_fov_min_ratio", None)
    for source in sources:
        default_augment = bool(args.chase_augment) if source == "CHASEDB1" else False
        default_min_foreground_ratio = float(args.chase_min_foreground_ratio) if source == "CHASEDB1" else 0.0
        default_use_fov_mask = bool(args.chase_use_fov_mask) if source == "CHASEDB1" else False
        default_fov_min_ratio = float(args.fov_min_ratio) if source == "CHASEDB1" else 0.0
        options[source] = {
            "augment": True if force_source_augment else default_augment,
            "brightness_range": tuple(parse_range(args.chase_brightness_range, DATASET_PROTOCOLS[source].brightness_range)),
            "contrast_range": tuple(parse_range(args.chase_contrast_range, DATASET_PROTOCOLS[source].contrast_range)),
            "min_foreground_ratio": (
                float(force_min_foreground_ratio)
                if force_min_foreground_ratio is not None else default_min_foreground_ratio
            ),
            "use_fov_mask": True if force_source_use_fov_mask else default_use_fov_mask,
            "fov_threshold": float(args.fov_threshold),
            "fov_blur": int(args.fov_blur),
            "fov_close": int(args.fov_close),
            "fov_min_ratio": (
                float(force_fov_min_ratio)
                if force_fov_min_ratio is not None else default_fov_min_ratio
            ),
            "normalize_input": bool(getattr(args, "normalize_input", False)),
        }
    return options


def build_train_val_loaders(
    preset_name: str,
    args: argparse.Namespace,
) -> Tuple[object, object, Dict[str, Dict[str, int]], Dict[str, Dict[str, object]]]:
    preset = PRESETS[preset_name]
    sources = list(preset["sources"])
    source_options = resolve_source_options(preset_name, sources, args)
    train_pairs: List[Dict[str, str]] = []
    val_pairs: List[Dict[str, str]] = []
    split_stats: Dict[str, Dict[str, int]] = {}

    for source in sources:
        src_train_pairs, src_val_pairs = collect_train_val_pairs(
            dataset_name=source,
            data_root=args.data_root,
            seed=args.seed,
            val_ratio=args.val_ratio,
            chase_mask_type=args.chase_mask_type,
        )
        train_pairs.extend(src_train_pairs)
        val_pairs.extend(src_val_pairs)
        split_stats[source] = {"train_images": len(src_train_pairs), "val_images": len(src_val_pairs)}

    train_dataset = MultiSourcePatchDataset(train_pairs, args.patch_size, args.stride, source_options, training=True)
    val_dataset = MultiSourcePatchDataset(val_pairs, args.patch_size, args.stride, source_options, training=False)

    train_sampler = None
    if bool(getattr(args, "source_balance", False)):
        patch_counts_by_source: Counter = Counter()
        sample_weights: List[float] = []
        for pair_idx, _, _ in train_dataset.index:
            source_name = str(train_dataset.pairs[pair_idx]["source"])
            patch_counts_by_source[source_name] += 1
        for pair_idx, _, _ in train_dataset.index:
            source_name = str(train_dataset.pairs[pair_idx]["source"])
            sample_weights.append(1.0 / max(1, patch_counts_by_source[source_name]))
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(int(args.seed))
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=sampler_generator,
        )
        for source_name, count in patch_counts_by_source.items():
            split_stats.setdefault(source_name, {})
            split_stats[source_name]["train_patches"] = int(count)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    if train_sampler is None:
        train_patch_counts: Counter = Counter()
        for pair_idx, _, _ in train_dataset.index:
            source_name = str(train_dataset.pairs[pair_idx]["source"])
            train_patch_counts[source_name] += 1
        for source_name, count in train_patch_counts.items():
            split_stats.setdefault(source_name, {})
            split_stats[source_name]["train_patches"] = int(count)
    val_patch_counts: Counter = Counter()
    for pair_idx, _, _ in val_dataset.index:
        source_name = str(val_dataset.pairs[pair_idx]["source"])
        val_patch_counts[source_name] += 1
    for source_name, count in val_patch_counts.items():
        split_stats.setdefault(source_name, {})
        split_stats[source_name]["val_patches"] = int(count)
    return train_loader, val_loader, split_stats, source_options


def build_training_model(method_name: str, device: torch.device) -> Tuple[nn.Module, bool]:
    if method_name == "baseline":
        model = create_unet_baseline(base_channels=32).to(device)
        return model, False
    model = eval_base.build_model(OURS_MODEL_KEY, device)
    return model, bool(eval_base.MODEL_SPECS[OURS_MODEL_KEY]["use_deep_supervision"])


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def build_checkpoint_payload(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    best_val_loss: float,
    param_count: int,
    seed: int,
    lambda_ds: float,
    config: Dict[str, object],
    args: argparse.Namespace,
) -> Dict[str, object]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": float(best_val_loss),
        "parameters": int(param_count),
        "seed": int(seed),
        "lambda_ds": float(lambda_ds),
        "config": safe_convert_for_json(config),
        "args": safe_convert_for_json(vars(args)),
        "timestamp": datetime.now().isoformat(),
    }


def save_recent_checkpoint(recent_dir: str, epoch: int, payload: Dict[str, object], keep: int) -> str:
    os.makedirs(recent_dir, exist_ok=True)
    ckpt_path = os.path.join(recent_dir, f"epoch_{int(epoch):03d}.pth")
    torch.save(payload, ckpt_path)
    keep_n = max(0, int(keep))
    recent_files = sorted(Path(recent_dir).glob("epoch_*.pth"))
    if keep_n > 0 and len(recent_files) > keep_n:
        for old_path in recent_files[:-keep_n]:
            try:
                old_path.unlink()
            except OSError:
                pass
    return ckpt_path


def train_single_experiment(preset_name: str, method_name: str, lambda_ds: float, args: argparse.Namespace) -> None:
    preset = PRESETS[preset_name]
    display_name = str(preset["display_name"])
    set_random_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    train_loader, val_loader, split_stats, source_options = build_train_val_loaders(preset_name, args)

    if method_name == "baseline":
        exp_root = os.path.join(args.save_dir, BASELINE_SAVE_SUBDIR, f"seed{args.seed}")
        exp_name = f"baseline_seed{args.seed}"
    else:
        exp_root = os.path.join(args.save_dir, "ours", f"lambda_{_fmt_lambda(lambda_ds)}", f"seed{args.seed}")
        exp_name = f"ours_lambda_{lambda_ds:.1f}_seed{args.seed}"

    os.makedirs(exp_root, exist_ok=True)
    checkpoint_path = os.path.join(exp_root, "best_model.pth")
    recent_ckpt_dir = os.path.join(exp_root, "recent_checkpoints")
    if os.path.exists(checkpoint_path) and not args.overwrite:
        print(f"[Skip] Existing checkpoint found: {checkpoint_path}")
        return
    if bool(getattr(args, "save_recent_checkpoints", False)):
        os.makedirs(recent_ckpt_dir, exist_ok=True)
        if bool(args.overwrite):
            for old_path in Path(recent_ckpt_dir).glob("epoch_*.pth"):
                try:
                    old_path.unlink()
                except OSError:
                    pass
    elif bool(args.overwrite) and os.path.isdir(recent_ckpt_dir):
        try:
            shutil.rmtree(recent_ckpt_dir)
        except OSError:
            pass

    model, use_deep_supervision = build_training_model(method_name, device)
    param_count = count_parameters(model)
    criterion = CombinedLoss(0.5, 0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, verbose=True)

    config = {
        "preset": preset_name,
        "display_name": display_name,
        "method": method_name,
        "sources": list(preset["sources"]),
        "target": str(preset["target"]),
        "seed": int(args.seed),
        "lambda_ds": float(lambda_ds),
        "patch_size": int(args.patch_size),
        "stride": int(args.stride),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "val_ratio": float(args.val_ratio),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "normalize_input": bool(args.normalize_input),
        "source_balance": bool(args.source_balance),
        "save_recent_checkpoints": bool(args.save_recent_checkpoints),
        "recent_checkpoint_keep": int(args.recent_checkpoint_keep),
        "parameters": int(param_count),
        "source_split_stats": split_stats,
        "source_options": safe_convert_for_json(source_options),
        "model_spec": (
            {
                "display_name": "Baseline_UNet32",
                "architecture": "UNet",
                "base_channels": 32,
                "encoder_stages": 3,
                "bottleneck": 1,
                "use_deep_supervision": False,
                "use_gpdc": False,
                "use_residual": False,
                "use_lmm": False,
                "use_sdpm": False,
                "save_subdir": BASELINE_SAVE_SUBDIR,
            }
            if method_name == "baseline"
            else eval_base.MODEL_SPECS[OURS_MODEL_KEY]
        ),
    }
    with open(os.path.join(exp_root, "config.json"), "w", encoding="utf-8") as f:
        json.dump(safe_convert_for_json(config), f, indent=2, ensure_ascii=False)

    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rates": [],
        "experiment_info": safe_convert_for_json(config),
    }
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    start_time = datetime.now()

    print("=" * 100)
    print(f"[Train] {display_name}")
    print(f"[Train] method={method_name}, lambda={lambda_ds:.1f}, seed={args.seed}, device={device}")
    print(
        f"[Train] patch/stride={args.patch_size}/{args.stride}, lr={args.learning_rate}, wd={args.weight_decay}, "
        f"normalize={'yes' if args.normalize_input else 'no'}, source_balance={'yes' if args.source_balance else 'no'}"
    )
    if bool(args.save_recent_checkpoints):
        print(f"[Train] recent_checkpoints=on (keep_last={int(args.recent_checkpoint_keep)})")
    print(f"[Train] save_dir={exp_root}")
    print(f"[Train] train_patches={len(train_loader.dataset)}, val_patches={len(val_loader.dataset)}")
    for src, stats in split_stats.items():
        extra = ""
        if "train_patches" in stats or "val_patches" in stats:
            extra = (
                f", train_patches={int(stats.get('train_patches', 0))}, "
                f"val_patches={int(stats.get('val_patches', 0))}"
            )
        print(f"[Train] source={src}: train_images={stats['train_images']}, val_images={stats['val_images']}{extra}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, lambda_ds, use_deep_supervision, epoch, exp_name)
        val_loss = validate_epoch(model, val_loader, criterion, device, lambda_ds, use_deep_supervision)
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["learning_rates"].append(current_lr)
        print(f"Epoch {epoch:03d}: train={train_loss:.6f}, val={val_loss:.6f}, lr={current_lr:.2e}")

        checkpoint_payload = build_checkpoint_payload(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            best_val_loss=min(best_val_loss, float(val_loss)),
            param_count=param_count,
            seed=args.seed,
            lambda_ds=lambda_ds,
            config=config,
            args=args,
        )
        if bool(args.save_recent_checkpoints):
            save_recent_checkpoint(recent_ckpt_dir, epoch, checkpoint_payload, int(args.recent_checkpoint_keep))

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_epoch = int(epoch)
            patience_counter = 0
            checkpoint_payload["best_val_loss"] = float(best_val_loss)
            torch.save(checkpoint_payload, checkpoint_path)
            print(f"  -> saved best checkpoint (epoch={epoch}, val={best_val_loss:.6f})")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"  -> early stop: patience reached ({args.patience})")
            break

    elapsed = datetime.now() - start_time
    with open(os.path.join(exp_root, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(safe_convert_for_json(history), f, indent=2, ensure_ascii=False)
    plot_training_curves(history, exp_root, exp_name)
    summary = {
        "preset": preset_name,
        "display_name": display_name,
        "method": method_name,
        "lambda_ds": float(lambda_ds),
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "num_epochs_trained": int(len(history["train_loss"])),
        "elapsed": str(elapsed),
        "checkpoint_path": checkpoint_path,
        "recent_checkpoint_dir": recent_ckpt_dir if bool(args.save_recent_checkpoints) else None,
        "recent_checkpoint_keep": int(args.recent_checkpoint_keep) if bool(args.save_recent_checkpoints) else 0,
    }
    with open(os.path.join(exp_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(safe_convert_for_json(summary), f, indent=2, ensure_ascii=False)
    print(f"[Done] {exp_name} | best_epoch={best_epoch}, best_val={best_val_loss:.6f}, elapsed={elapsed}")


def build_train_parser(preset_name: str) -> argparse.ArgumentParser:
    preset = PRESETS[preset_name]
    sources = list(preset["sources"])
    default_patch_size, default_stride = resolve_combo_training_protocol_for_preset(preset_name, sources)
    parser = argparse.ArgumentParser(description=f"Cross-dataset training: {preset['display_name']}")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default=default_models_dir_for_preset(preset_name))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--methods",
        type=str,
        default=str(get_preset_override(preset_name, "train", "methods", "baseline,ours")),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=resolve_combo_weight_decay_for_preset(preset_name, sources))
    parser.add_argument("--patch_size", type=int, default=default_patch_size)
    parser.add_argument("--stride", type=int, default=default_stride)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val_ratio", type=float, default=float(get_preset_override(preset_name, "train", "val_ratio", 0.2)))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--ours_lambda_ds",
        type=float,
        default=float(get_preset_override(preset_name, "train", "ours_lambda_ds", 0.1)),
    )
    parser.add_argument(
        "--ours_lambda_list",
        type=str,
        default=str(get_preset_override(preset_name, "train", "ours_lambda_list", DEFAULT_OURS_LAMBDA_LIST)),
    )
    parser.add_argument("--ours_grid_mode", dest="ours_grid_mode", action="store_true")
    parser.add_argument("--no_ours_grid_mode", dest="ours_grid_mode", action="store_false")
    parser.set_defaults(ours_grid_mode=bool(get_preset_override(preset_name, "train", "ours_grid_mode", True)))
    parser.add_argument("--chase_mask_type", type=str, default="1stHO")
    parser.add_argument("--chase_min_foreground_ratio", type=float, default=0.01)
    parser.add_argument("--chase_augment", dest="chase_augment", action="store_true")
    parser.add_argument("--no_chase_augment", dest="chase_augment", action="store_false")
    parser.set_defaults(chase_augment=("CHASEDB1" in sources))
    parser.add_argument("--chase_brightness_range", type=str, default="0.8,1.2")
    parser.add_argument("--chase_contrast_range", type=str, default="0.8,1.2")
    parser.add_argument("--chase_use_fov_mask", action="store_true")
    parser.add_argument("--fov_threshold", type=float, default=0.05)
    parser.add_argument("--fov_blur", type=int, default=7)
    parser.add_argument("--fov_close", type=int, default=15)
    parser.add_argument("--fov_min_ratio", type=float, default=0.0)
    parser.add_argument("--normalize_input", dest="normalize_input", action="store_true")
    parser.add_argument("--no_normalize_input", dest="normalize_input", action="store_false")
    parser.set_defaults(normalize_input=bool(get_preset_override(preset_name, "train", "normalize_input", False)))
    parser.add_argument("--source_balance", dest="source_balance", action="store_true")
    parser.add_argument("--no_source_balance", dest="source_balance", action="store_false")
    parser.set_defaults(source_balance=bool(get_preset_override(preset_name, "train", "source_balance", False)))
    parser.add_argument("--save_recent_checkpoints", dest="save_recent_checkpoints", action="store_true")
    parser.add_argument("--no_save_recent_checkpoints", dest="save_recent_checkpoints", action="store_false")
    parser.set_defaults(
        save_recent_checkpoints=bool(get_preset_override(preset_name, "train", "save_recent_checkpoints", False))
    )
    parser.add_argument(
        "--recent_checkpoint_keep",
        type=int,
        default=int(get_preset_override(preset_name, "train", "recent_checkpoint_keep", 5)),
    )
    return parser


def run_train_preset(preset_name: str) -> None:
    args = build_train_parser(preset_name).parse_args()
    apply_model_path_aliases(args, ("save_dir",))
    os.makedirs(args.save_dir, exist_ok=True)
    methods = eval_base.parse_method_list(args.methods)
    seeds = parse_int_list(args.seeds) if args.seeds else [int(args.seed)]
    if not seeds:
        raise ValueError("seed list is empty.")

    for seed in seeds:
        args.seed = int(seed)
        if "baseline" in methods:
            train_single_experiment(preset_name, "baseline", 0.0, args)
        if "ours" in methods:
            lambda_values = parse_float_list(args.ours_lambda_list) if args.ours_grid_mode else [float(args.ours_lambda_ds)]
            if not lambda_values:
                raise ValueError("ours lambda list is empty.")
            for lambda_ds in lambda_values:
                train_single_experiment(preset_name, "ours", float(lambda_ds), args)


def resolve_eval_ckpt_path(models_dir: str, method_name: str, seed: int, lambda_ds: float) -> str:
    if method_name == "baseline":
        return os.path.join(models_dir, BASELINE_SAVE_SUBDIR, f"seed{seed}", "best_model.pth")
    return os.path.join(models_dir, "ours", f"lambda_{_fmt_lambda(lambda_ds)}", f"seed{seed}", "best_model.pth")


def resolve_recent_ckpt_dir(models_dir: str, method_name: str, seed: int, lambda_ds: float) -> str:
    if method_name == "baseline":
        return os.path.join(models_dir, BASELINE_SAVE_SUBDIR, f"seed{seed}", "recent_checkpoints")
    return os.path.join(models_dir, "ours", f"lambda_{_fmt_lambda(lambda_ds)}", f"seed{seed}", "recent_checkpoints")


def resolve_eval_candidate_ckpt_paths(
    models_dir: str,
    method_name: str,
    seed: int,
    lambda_ds: float,
    checkpoint_policy: str,
    recent_checkpoint_keep: int,
) -> List[str]:
    best_path = resolve_eval_ckpt_path(models_dir, method_name, seed, lambda_ds)
    policy = str(checkpoint_policy).lower()
    candidates: List[str] = []
    if os.path.exists(best_path):
        candidates.append(best_path)
    if policy == "best_recent_target_f1":
        recent_dir = resolve_recent_ckpt_dir(models_dir, method_name, seed, lambda_ds)
        recent_paths = sorted(Path(recent_dir).glob("epoch_*.pth"))
        keep_n = max(0, int(recent_checkpoint_keep))
        if keep_n > 0 and len(recent_paths) > keep_n:
            recent_paths = recent_paths[-keep_n:]
        candidates.extend(str(path) for path in recent_paths if path.exists())
    deduped: List[str] = []
    seen: set = set()
    for path in candidates:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def is_better_checkpoint_candidate(candidate_metrics: Dict[str, float], best_metrics: Optional[Dict[str, float]]) -> bool:
    if best_metrics is None:
        return True
    candidate_key = (
        float(candidate_metrics["F1"]),
        float(candidate_metrics["AUC"]),
        float(candidate_metrics["Se"]),
    )
    best_key = (
        float(best_metrics["F1"]),
        float(best_metrics["AUC"]),
        float(best_metrics["Se"]),
    )
    return candidate_key > best_key


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    tp, tn, fp, fn = eval_base.confusion_from_scores(labels, scores, threshold)
    m = eval_base.metrics_from_confusion(tp, tn, fp, fn)
    auc = eval_base.calc_auc_safe(labels, scores)
    pr_auc = eval_base.calc_pr_auc_safe(labels, scores)
    pred = (scores > threshold).astype(np.uint8)
    try:
        mcc = float(matthews_corrcoef(labels.astype(np.uint8), pred))
    except Exception:
        mcc = 0.0
    return {
        "F1": float(m["F1"]),
        "Se": float(m["Se"]),
        "Spe": float(m["Spe"]),
        "Acc": float(m["Acc"]),
        "Precision": float(m["Precision"]),
        "AUC": float(auc),
        "PR_AUC": float(pr_auc),
        "MCC": float(mcc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "n_pixels": int(labels.size),
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
        "Threshold": float(threshold),
    }


def search_threshold_drive_stare_rule(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold_base: float,
    thr_min: float,
    thr_max: float,
    thr_step: float,
    f1_drop_tolerance: float,
    search_mode: str,
) -> Tuple[Dict[str, float], float, Dict[str, float]]:
    base_metrics = compute_metrics(labels, scores, threshold_base)
    best_metrics = dict(base_metrics)
    best_thr = float(threshold_base)
    best_f1 = float(base_metrics["F1"])
    f1_base = float(base_metrics["F1"])
    mode = str(search_mode).lower()
    if mode not in {"max_se_at_base_f1", "best_f1"}:
        mode = "max_se_at_base_f1"
    thr = float(thr_min)
    while thr <= float(thr_max) + 1e-8:
        metrics_t = compute_metrics(labels, scores, thr)
        f1_t = float(metrics_t["F1"])
        if mode == "best_f1":
            if (f1_t > best_f1 + 1e-6) or (abs(f1_t - best_f1) <= 1e-6 and metrics_t["Se"] > best_metrics["Se"] + 1e-6):
                best_metrics = metrics_t
                best_thr = float(thr)
                best_f1 = f1_t
        elif metrics_t["F1"] >= f1_base - float(f1_drop_tolerance):
            if metrics_t["Se"] > best_metrics["Se"] + 1e-6:
                best_metrics = metrics_t
                best_thr = float(thr)
        thr += float(thr_step)
    return best_metrics, best_thr, base_metrics


def search_threshold_chase_rule(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold_base: float,
    thr_min: float,
    thr_max: float,
    thr_step: float,
    f1_drop_tolerance: float,
    optimize_mode: str,
) -> Tuple[Dict[str, float], float, Dict[str, float]]:
    base_metrics = compute_metrics(labels, scores, threshold_base)
    best_metrics = dict(base_metrics)
    best_thr = float(threshold_base)
    best_f1 = float(base_metrics["F1"])
    f1_base = float(base_metrics["F1"])
    mode = str(optimize_mode).lower()
    if mode not in {"se", "f1"}:
        mode = "se"
    thr = float(thr_min)
    while thr <= float(thr_max) + 1e-8:
        metrics_t = compute_metrics(labels, scores, thr)
        f1_t = float(metrics_t["F1"])
        se_t = float(metrics_t["Se"])
        if mode == "f1":
            if (f1_t > best_f1 + 1e-6) or (abs(f1_t - best_f1) <= 1e-6 and se_t > best_metrics["Se"] + 1e-6):
                best_metrics = metrics_t
                best_thr = float(thr)
                best_f1 = f1_t
        else:
            if f1_t >= f1_base - float(f1_drop_tolerance):
                if se_t > best_metrics["Se"] + 1e-6:
                    best_metrics = metrics_t
                    best_thr = float(thr)
                    best_f1 = f1_t
        thr += float(thr_step)
    return best_metrics, best_thr, base_metrics


def _extract_batch_tensors(batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(batch, (list, tuple)) and len(batch) == 4:
        return batch[0], batch[1], batch[3]
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1], None
    raise RuntimeError("Unexpected batch format.")


def evaluate_one_model_seed(
    target_name: str,
    method_name: str,
    model_key: str,
    ckpt_path: str,
    seed: int,
    loader,
    device: torch.device,
    patch_size: int,
    stride: int,
    threshold_base: float,
    optimize_threshold: bool,
    thr_min: float,
    thr_max: float,
    thr_step: float,
    f1_drop_tolerance: float,
    drive_stare_search_mode: str,
    chase_optimize_mode: str,
    use_fov_mask: bool,
    sweep_thresholds: Optional[List[float]] = None,
) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
    if method_name == "baseline":
        model = create_unet_baseline(base_channels=32).to(device)
        use_deep_supervision = False
    else:
        model = eval_base.build_model(model_key, device)
        use_deep_supervision = bool(eval_base.MODEL_SPECS[model_key]["use_deep_supervision"])
    model, _ = eval_base.load_checkpoint_to_model(model, ckpt_path, device)
    infer_model = eval_base.WrappedDeepSupervisionModel(model) if use_deep_supervision else model
    infer_model.eval()

    score_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{target_name}-seed{seed}", leave=False):
            images, masks, fov_masks = _extract_batch_tensors(batch)
            bsz = int(images.size(0))
            for idx in range(bsz):
                image = images[idx:idx + 1]
                mask = masks[idx:idx + 1]
                pred = eval_base.predict_full_image_aligned(model=infer_model, image=image, patch_size=patch_size, stride=stride, device=device)
                score = pred.detach().cpu().numpy()[0, 0].astype(np.float32)
                score = np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
                score = np.clip(score, 0.0, 1.0)
                label = (mask.numpy()[0, 0] > 0.5).astype(np.uint8)
                if target_name == "CHASEDB1" and use_fov_mask and fov_masks is not None:
                    fov = (fov_masks.numpy()[idx, 0] > 0.5)
                    if np.any(fov):
                        score_chunks.append(score[fov].reshape(-1))
                        label_chunks.append(label[fov].reshape(-1))
                    else:
                        score_chunks.append(score.reshape(-1))
                        label_chunks.append(label.reshape(-1))
                else:
                    score_chunks.append(score.reshape(-1))
                    label_chunks.append(label.reshape(-1))

    if not score_chunks:
        raise RuntimeError(f"No predictions produced for target={target_name}, seed={seed}")

    scores_all = np.concatenate(score_chunks, axis=0)
    labels_all = np.concatenate(label_chunks, axis=0)
    base_metrics = compute_metrics(labels_all, scores_all, threshold_base)
    metrics = dict(base_metrics)
    threshold_auto = float(threshold_base)
    threshold_rule = "fixed"

    if optimize_threshold:
        if target_name == "CHASEDB1":
            metrics, threshold_auto, base_metrics_ref = search_threshold_chase_rule(
                labels_all, scores_all, threshold_base, thr_min, thr_max, thr_step, f1_drop_tolerance, chase_optimize_mode
            )
            threshold_rule = f"chase_{str(chase_optimize_mode).lower()}"
        else:
            metrics, threshold_auto, base_metrics_ref = search_threshold_drive_stare_rule(
                labels_all, scores_all, threshold_base, thr_min, thr_max, thr_step, f1_drop_tolerance, drive_stare_search_mode
            )
            threshold_rule = str(drive_stare_search_mode)
        base_metrics = base_metrics_ref

    metrics["ThresholdBase"] = float(threshold_base)
    metrics["ThresholdAuto"] = float(threshold_auto)
    metrics["threshold_policy"] = str(threshold_rule)
    metrics["F1_base"] = float(base_metrics["F1"])
    metrics["Se_base"] = float(base_metrics["Se"])
    metrics["Spe_base"] = float(base_metrics["Spe"])
    metrics["Acc_base"] = float(base_metrics["Acc"])
    metrics["Precision_base"] = float(base_metrics["Precision"])
    sweep_df = eval_base.compute_threshold_sweep(labels_all, scores_all, sweep_thresholds) if sweep_thresholds else None

    del score_chunks, label_chunks, scores_all, labels_all, infer_model, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, sweep_df


def build_eval_parser(
    preset_name: str,
    local_eval_defaults: Optional[Dict[str, object]] = None,
) -> argparse.ArgumentParser:
    preset = PRESETS[preset_name]
    target = str(preset["target"])
    target_proto = DATASET_PROTOCOLS[target]
    parser = argparse.ArgumentParser(description=f"Cross-dataset evaluation: {preset['display_name']}")
    parser.add_argument("--models_dir", type=str, default=default_models_dir_for_preset(preset_name))
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--results_dir", type=str, default=default_results_dir_for_preset(preset_name))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument(
        "--methods",
        type=str,
        default=str(get_preset_override(preset_name, "eval", "methods", "baseline,ours")),
    )
    parser.add_argument(
        "--ours_lambda_list",
        type=str,
        default=str(get_preset_override(preset_name, "eval", "ours_lambda_list", DEFAULT_OURS_LAMBDA_LIST)),
    )
    parser.add_argument(
        "--baseline_thr",
        type=float,
        default=float(get_preset_override(preset_name, "eval", "baseline_thr", 0.5)),
    )
    parser.add_argument("--baseline_search_threshold", dest="baseline_search_threshold", action="store_true")
    parser.add_argument("--no_baseline_search_threshold", dest="baseline_search_threshold", action="store_false")
    parser.set_defaults(baseline_search_threshold=False)
    parser.add_argument("--baseline_search_mode", type=str, default="best_f1", choices=["max_se_at_base_f1", "best_f1"])
    parser.add_argument(
        "--ours_thr",
        type=float,
        default=float(get_preset_override(preset_name, "eval", "ours_thr", target_proto.default_threshold)),
    )
    parser.add_argument("--ours_search_threshold", dest="ours_search_threshold", action="store_true")
    parser.add_argument("--no_ours_search_threshold", dest="ours_search_threshold", action="store_false")
    parser.set_defaults(ours_search_threshold=True)
    parser.add_argument("--ours_search_mode", type=str, default="max_se_at_base_f1", choices=["max_se_at_base_f1", "best_f1"])
    parser.add_argument("--thr_min", type=float, default=float(get_preset_override(preset_name, "eval", "thr_min", 0.30)))
    parser.add_argument("--thr_max", type=float, default=float(get_preset_override(preset_name, "eval", "thr_max", 0.60)))
    parser.add_argument("--thr_step", type=float, default=float(get_preset_override(preset_name, "eval", "thr_step", 0.01)))
    parser.add_argument("--f1_drop_tolerance", type=float, default=0.0005)
    parser.add_argument(
        "--patch_size",
        type=int,
        default=int(get_preset_override(preset_name, "eval", "patch_size", target_proto.eval_patch_size)),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=int(get_preset_override(preset_name, "eval", "stride", target_proto.eval_stride)),
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--chase_mask_type", type=str, default="1stHO")
    parser.add_argument("--chase_use_fov_mask", action="store_true")
    parser.add_argument(
        "--chase_optimize_mode",
        type=str,
        default=str(get_preset_override(preset_name, "eval", "chase_optimize_mode", "se")),
        choices=["se", "f1"],
    )
    parser.add_argument("--fov_threshold", type=float, default=0.05)
    parser.add_argument("--fov_blur", type=int, default=7)
    parser.add_argument("--fov_close", type=int, default=15)
    parser.add_argument("--normalize_input", dest="normalize_input", action="store_true")
    parser.add_argument("--no_normalize_input", dest="normalize_input", action="store_false")
    parser.set_defaults(normalize_input=bool(get_preset_override(preset_name, "eval", "normalize_input", (target == "DRIVE"))))
    parser.add_argument(
        "--target_resize_scale",
        type=float,
        default=float(get_preset_override(preset_name, "eval", "target_resize_scale", 1.0)),
    )
    parser.add_argument(
        "--target_resize_interpolation",
        type=str,
        default=str(get_preset_override(preset_name, "eval", "target_resize_interpolation", "area")),
        choices=["area", "bilinear", "bicubic", "nearest"],
    )
    parser.add_argument("--diagnose_threshold_sweep", action="store_true")
    parser.add_argument("--sweep_start", type=float, default=0.10)
    parser.add_argument("--sweep_end", type=float, default=0.50)
    parser.add_argument("--sweep_step", type=float, default=0.05)
    parser.add_argument("--incremental_save", dest="incremental_save", action="store_true")
    parser.add_argument("--no_incremental_save", dest="incremental_save", action="store_false")
    parser.set_defaults(incremental_save=bool(get_preset_override(preset_name, "eval", "incremental_save", False)))
    parser.add_argument(
        "--checkpoint_policy",
        type=str,
        default=str(get_preset_override(preset_name, "eval", "checkpoint_policy", "best")),
        choices=["best", "best_recent_target_f1"],
    )
    parser.add_argument(
        "--recent_checkpoint_keep",
        type=int,
        default=int(get_preset_override(preset_name, "eval", "recent_checkpoint_keep", 5)),
    )
    parser.set_defaults(
        baseline_search_threshold=bool(
            get_preset_override(preset_name, "eval", "baseline_search_threshold", parser.get_default("baseline_search_threshold"))
        )
    )
    parser.set_defaults(
        ours_search_threshold=bool(
            get_preset_override(preset_name, "eval", "ours_search_threshold", parser.get_default("ours_search_threshold"))
        )
    )
    parser.set_defaults(
        chase_use_fov_mask=bool(
            get_preset_override(preset_name, "eval", "chase_use_fov_mask", parser.get_default("chase_use_fov_mask"))
        )
    )
    if local_eval_defaults:
        parser.set_defaults(**local_eval_defaults)
    return parser


def run_eval_preset(
    preset_name: str,
    local_eval_defaults: Optional[Dict[str, object]] = None,
) -> None:
    args = build_eval_parser(preset_name, local_eval_defaults=local_eval_defaults).parse_args()
    apply_model_path_aliases(args, ("models_dir",))
    preset = PRESETS[preset_name]
    target = str(preset["target"])
    display_name = str(preset["display_name"])
    artifact_prefix = default_artifact_prefix_for_preset(preset_name)
    os.makedirs(args.results_dir, exist_ok=True)

    seeds = parse_int_list(args.seeds)
    methods = eval_base.parse_method_list(args.methods)
    ours_lambdas = parse_float_list(args.ours_lambda_list)
    if "ours" in methods and not ours_lambdas:
        raise ValueError("ours lambda list is empty.")
    eval_jobs = eval_base.build_eval_jobs(methods, ours_lambdas if ours_lambdas else [0.0])
    device = torch.device("cuda" if args.device.lower() == "cuda" and torch.cuda.is_available() else "cpu")
    sweep_thresholds = eval_base.build_threshold_values(args.sweep_start, args.sweep_end, args.sweep_step) if args.diagnose_threshold_sweep else None

    loader = eval_base.get_test_loader_target(
        data_dir=os.path.join(args.data_root, target),
        target_name=target,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        chase_mask_type=args.chase_mask_type,
        normalize_input=bool(args.normalize_input),
        use_fov_mask=bool(args.chase_use_fov_mask and target == "CHASEDB1"),
        fov_threshold=args.fov_threshold,
        fov_blur=args.fov_blur,
        fov_close=args.fov_close,
        resize_scale=float(args.target_resize_scale),
        resize_interpolation=str(args.target_resize_interpolation),
    )

    meta = {
        "preset": preset_name,
        "display_name": display_name,
        "sources": list(preset["sources"]),
        "target": target,
        "models_dir": args.models_dir,
        "data_root": args.data_root,
        "results_dir": args.results_dir,
        "device_requested": args.device,
        "device_actual": str(device),
        "seeds": seeds,
        "methods": methods,
        "ours_lambda_list": ours_lambdas,
        "baseline_thr": float(args.baseline_thr),
        "baseline_search_threshold": bool(args.baseline_search_threshold),
        "baseline_search_mode": args.baseline_search_mode,
        "ours_thr": float(args.ours_thr),
        "ours_search_threshold": bool(args.ours_search_threshold),
        "ours_search_mode": args.ours_search_mode,
        "thr_min": float(args.thr_min),
        "thr_max": float(args.thr_max),
        "thr_step": float(args.thr_step),
        "f1_drop_tolerance": float(args.f1_drop_tolerance),
        "patch_size": int(args.patch_size),
        "stride": int(args.stride),
        "normalize_input": bool(args.normalize_input),
        "target_resize_scale": float(args.target_resize_scale),
        "target_resize_interpolation": str(args.target_resize_interpolation),
        "chase_mask_type": args.chase_mask_type,
        "chase_use_fov_mask": bool(args.chase_use_fov_mask),
        "chase_optimize_mode": args.chase_optimize_mode,
        "checkpoint_policy": args.checkpoint_policy,
        "recent_checkpoint_keep": int(args.recent_checkpoint_keep),
        "diagnose_threshold_sweep": bool(args.diagnose_threshold_sweep),
        "sweep_thresholds": sweep_thresholds,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(args.results_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(safe_convert_for_json(meta), f, indent=2, ensure_ascii=False)

    def persist_eval_outputs(rows_local: List[Dict[str, object]]) -> None:
        if not rows_local:
            return
        raw_df_local = pd.DataFrame(rows_local)
        raw_path_local = os.path.join(args.results_dir, f"{artifact_prefix}_raw.csv")
        raw_df_local.to_csv(raw_path_local, index=False, float_format="%.6f")

        summary_num_df_local, summary_table_df_local = eval_base.build_summary(raw_df_local)
        summary_num_path_local = os.path.join(args.results_dir, f"{artifact_prefix}_summary_numeric.csv")
        summary_table_path_local = os.path.join(args.results_dir, f"{artifact_prefix}_summary_table.csv")
        summary_num_df_local.to_csv(summary_num_path_local, index=False, float_format="%.6f")
        summary_table_df_local.to_csv(summary_table_path_local, index=False)

        best_compare_df_local = eval_base.build_best_ours_vs_baseline(summary_num_df_local)
        best_compare_path_local = os.path.join(args.results_dir, f"{artifact_prefix}_best_ours_vs_baseline.csv")
        best_compare_df_local.to_csv(best_compare_path_local, index=False, float_format="%.6f")

        with open(os.path.join(args.results_dir, f"{artifact_prefix}_raw.json"), "w", encoding="utf-8") as f_raw:
            json.dump(safe_convert_for_json(rows_local), f_raw, indent=2, ensure_ascii=False)
        with open(os.path.join(args.results_dir, f"{artifact_prefix}_summary_numeric.json"), "w", encoding="utf-8") as f_sum:
            json.dump(safe_convert_for_json(summary_num_df_local.to_dict(orient="records")), f_sum, indent=2, ensure_ascii=False)

    rows: List[Dict[str, object]] = []
    for job in eval_jobs:
        method_name = str(job["method_name"])
        method_display = str(job["method_display"])
        model_key = str(job["model_key"])
        lambda_ds = float(job["lambda_ds"]) if not np.isnan(job["lambda_ds"]) else 0.0
        threshold_base = float(args.baseline_thr if method_name == "baseline" else args.ours_thr)
        optimize_this = bool(
            (method_name == "baseline" and args.baseline_search_threshold)
            or (method_name == "ours" and args.ours_search_threshold)
        )
        drive_stare_search_mode = args.baseline_search_mode if method_name == "baseline" else args.ours_search_mode

        for seed in seeds:
            candidate_ckpt_paths = resolve_eval_candidate_ckpt_paths(
                args.models_dir,
                method_name,
                seed,
                lambda_ds,
                args.checkpoint_policy,
                int(args.recent_checkpoint_keep),
            )
            if not candidate_ckpt_paths:
                raise FileNotFoundError(
                    f"No checkpoints found for method={method_name}, seed={seed}, lambda={lambda_ds:.1f}, policy={args.checkpoint_policy}"
                )

            print("=" * 100)
            print(
                f"[Eval] {display_name} | target={target}, method={method_display}, seed={seed}, "
                f"lambda={'N/A' if method_name == 'baseline' else f'{lambda_ds:.1f}'}"
            )
            print(
                f"[Eval] threshold_base={threshold_base:.3f}, optimize={'yes' if optimize_this else 'no'}, "
                f"patch/stride={args.patch_size}/{args.stride}, resize_scale={float(args.target_resize_scale):.3f}"
            )
            if len(candidate_ckpt_paths) > 1:
                print(
                    f"[Eval] checkpoint_policy={args.checkpoint_policy}, "
                    f"candidates={len(candidate_ckpt_paths)}"
                )

            selected_metrics: Optional[Dict[str, float]] = None
            selected_sweep_df: Optional[pd.DataFrame] = None
            selected_ckpt_path: Optional[str] = None
            for cand_idx, cand_ckpt_path in enumerate(candidate_ckpt_paths, start=1):
                metrics_cand, sweep_df_cand = evaluate_one_model_seed(
                    target_name=target,
                    method_name=method_name,
                    model_key=model_key,
                    ckpt_path=cand_ckpt_path,
                    seed=seed,
                    loader=loader,
                    device=device,
                    patch_size=args.patch_size,
                    stride=args.stride,
                    threshold_base=threshold_base,
                    optimize_threshold=optimize_this,
                    thr_min=args.thr_min,
                    thr_max=args.thr_max,
                    thr_step=args.thr_step,
                    f1_drop_tolerance=args.f1_drop_tolerance,
                    drive_stare_search_mode=drive_stare_search_mode,
                    chase_optimize_mode=args.chase_optimize_mode,
                    use_fov_mask=bool(args.chase_use_fov_mask and target == "CHASEDB1"),
                    sweep_thresholds=sweep_thresholds,
                )
                if len(candidate_ckpt_paths) > 1:
                    print(
                        f"[Eval] candidate {cand_idx}/{len(candidate_ckpt_paths)} "
                        f"{Path(cand_ckpt_path).name}: F1={float(metrics_cand['F1']):.4f}, "
                        f"AUC={float(metrics_cand['AUC']):.4f}, thr={float(metrics_cand['ThresholdAuto']):.3f}"
                    )
                if is_better_checkpoint_candidate(metrics_cand, selected_metrics):
                    selected_metrics = metrics_cand
                    selected_sweep_df = sweep_df_cand
                    selected_ckpt_path = cand_ckpt_path

            metrics = selected_metrics if selected_metrics is not None else {}
            sweep_df = selected_sweep_df
            ckpt_path = str(selected_ckpt_path) if selected_ckpt_path is not None else candidate_ckpt_paths[0]
            if len(candidate_ckpt_paths) > 1:
                print(f"[Eval] selected_checkpoint={ckpt_path}")

            row = {
                "Source (Train)": " + ".join(list(preset["sources"])),
                "Target (Test)": eval_base.TARGET_DISPLAY.get(target, target),
                "Method": "Baseline" if method_name == "baseline" else f"Ours(lambda={lambda_ds:.1f})",
                "seed": int(seed),
                "checkpoint": ckpt_path,
                "checkpoint_policy": args.checkpoint_policy,
                "checkpoint_candidates": int(len(candidate_ckpt_paths)),
            }
            row.update(metrics)
            rows.append(row)
            print(
                f"[Done] F1={float(metrics['F1']):.4f}, AUC={float(metrics['AUC']):.4f}, "
                f"Se={float(metrics['Se']):.4f}, Spe={float(metrics['Spe']):.4f}, "
                f"thr={float(metrics['ThresholdAuto']):.3f}"
            )
            if bool(args.incremental_save):
                persist_eval_outputs(rows)

            if sweep_df is not None and not sweep_df.empty:
                lambda_tag = "" if method_name == "baseline" else f"_lambda_{_fmt_lambda(lambda_ds)}"
                sweep_name = f"thr_sweep_{preset_name}_{method_name}{lambda_tag}_seed{seed}.csv"
                sweep_df.to_csv(os.path.join(args.results_dir, sweep_name), index=False)

    persist_eval_outputs(rows)
    raw_df = pd.DataFrame(rows)
    summary_num_df, summary_table_df = eval_base.build_summary(raw_df)
    best_compare_df = eval_base.build_best_ours_vs_baseline(summary_num_df)

    print("\nSummary (mean +/- std over seeds):")
    print(summary_table_df.to_string(index=False))
    if not best_compare_df.empty:
        print("\nBest ours vs baseline:")
        print(best_compare_df.to_string(index=False))
