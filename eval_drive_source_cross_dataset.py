#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-dataset evaluation for DRIVE-trained checkpoints (diagnostic/aligned version).

Main fixes relative to the original script:
1) Fix corrupted config_str for Ours: C-[V]脳11
2) Support DRIVE as an evaluation target to sanity-check DRIVE->DRIVE reproduction
3) Align CHASEDB1 inference defaults with eval_chase.py:
   - patch_size=96, stride=32, mask_type=1stHO
4) Try to reuse drive_preprocessing.predict_full_image when available
5) Add optional threshold sweep diagnostics for calibration-shift analysis

Typical uses:
- Reproduce same-domain sanity check:
    python eval_drive_source_cross_dataset.py --targets DRIVE

- Cross-dataset with aligned defaults:
    python eval_drive_source_cross_dataset.py --targets STARE,CHASEDB1

- Diagnostic threshold sweep on STARE for Ours:
    python eval_drive_source_cross_dataset.py \
        --targets STARE \
        --methods ours \
        --diagnose_threshold_sweep \
        --sweep_start 0.10 --sweep_end 0.50 --sweep_step 0.05
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image
except Exception:
    Image = None

from pdc_unet_model import create_model
from path_compat import canonicalize_ours_models_subdir

official_predict_full_image = None
official_get_test_loader = None
_official_drive_helpers_loaded = False


def _load_official_drive_helpers_if_needed() -> None:
    """Lazy-load drive_preprocessing helpers only when DRIVE target is used."""
    global official_predict_full_image, official_get_test_loader, _official_drive_helpers_loaded
    if _official_drive_helpers_loaded:
        return
    _official_drive_helpers_loaded = True
    try:
        from drive_preprocessing import (
            predict_full_image as _predict_full_image,
            get_test_loader as _get_test_loader,
        )
        official_predict_full_image = _predict_full_image
        official_get_test_loader = _get_test_loader
    except Exception as e:
        official_predict_full_image = None
        official_get_test_loader = None
        print(f"[Warn] Failed to import drive_preprocessing helpers: {e}")


SOURCE_NAME = "DRIVE"
MODEL_SPECS = {
    "baseline_c32": {
        "display_name": "Baseline_C32",
        "config_str": "baseline",
        "channels": 32,
        "use_gpdc": False,
        "use_residual": False,
        "use_lmm": False,
        "use_sdpm": False,
        "use_deep_supervision": False,
        "path_type": "step12_baseline",
    },
    "ds_lambda_0_2": {
        "display_name": "SDPM+DS(lambda=0.2)",
        "config_str": "C-[V]脳11",
        "channels": 32,
        "use_gpdc": True,
        "use_residual": False,
        "use_lmm": False,
        "use_sdpm": True,
        "use_deep_supervision": True,
        "path_type": "drive_ds",
    },
}
METHOD_SPECS = {
    "baseline": {
        "display": "Baseline",
        "model_key": "baseline_c32",
    },
    "ours": {
        "display": "Ours",
        "model_key": "ds_lambda_0_2",
    },
}
TARGET_DISPLAY = {
    "DRIVE": "DRIVE",
    "STARE": "STARE",
    "CHASEDB1": "CHASE_DB1",
}
ALLOWED_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ""}


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_method_list(text: str) -> List[str]:
    items: List[str] = []
    for x in text.split(","):
        k = x.strip().lower()
        if not k:
            continue
        if k not in METHOD_SPECS:
            raise ValueError(f"Unknown method: {x}")
        items.append(k)
    if not items:
        raise ValueError("No valid methods provided.")
    return items


class WrappedDeepSupervisionModel(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        out = self.base_model(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def resolve_model_path(
    models_root: str,
    model_key: str,
    seed: int,
    lambda_ds: float,
    ours_models_subdir: str,
) -> str:
    spec = MODEL_SPECS[model_key]
    path_type = spec["path_type"]
    if path_type == "step12_baseline":
        cands = ["baseline_C32", "baseline"]
        for c in cands:
            p = os.path.join(models_root, "step12", f"seed{seed}", c, "best_model.pth")
            if os.path.exists(p):
                return p
        return os.path.join(models_root, "step12", f"seed{seed}", "baseline", "best_model.pth")
    if path_type == "drive_ds":
        lambda_str = f"{lambda_ds:.1f}".replace(".", "_")
        return os.path.join(models_root, ours_models_subdir, f"lambda_{lambda_str}", f"seed{seed}", "best_model.pth")
    raise ValueError(f"Unknown path_type: {path_type}")


def build_eval_jobs(method_order: List[str], ours_lambdas: List[float]) -> List[Dict[str, object]]:
    jobs: List[Dict[str, object]] = []
    for method_name in method_order:
        spec = METHOD_SPECS[method_name]
        if method_name == "ours":
            for lam in ours_lambdas:
                jobs.append({
                    "method_name": method_name,
                    "method_display": f"Ours(lambda={lam:.1f})",
                    "model_key": spec["model_key"],
                    "lambda_ds": float(lam),
                })
        else:
            jobs.append({
                "method_name": method_name,
                "method_display": spec["display"],
                "model_key": spec["model_key"],
                "lambda_ds": float("nan"),
            })
    return jobs


def build_model(model_key: str, device: torch.device) -> nn.Module:
    spec = MODEL_SPECS[model_key]
    model = create_model(
        config_str=spec["config_str"],
        channels=spec["channels"],
        use_gpdc=spec["use_gpdc"],
        use_residual=spec["use_residual"],
        use_lmm=spec["use_lmm"],
        use_sdpm=spec["use_sdpm"],
        use_deep_supervision=spec["use_deep_supervision"],
    ).to(device)
    return model


def load_checkpoint_to_model(model: nn.Module, ckpt_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        meta = checkpoint
    else:
        state_dict = checkpoint
        meta = {}
    model.load_state_dict(state_dict)
    model.eval()
    return model, meta


def predict_full_image_local(
    model: nn.Module,
    image: torch.Tensor,
    patch_size: int,
    stride: int,
    device: torch.device,
) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(f"Expected image shape [B,C,H,W], got {image.shape}")
    model.eval()
    image = image.to(device, non_blocking=True)
    b, _, h, w = image.shape
    if b != 1:
        raise ValueError(f"Only batch_size=1 supported, got batch={b}")

    patch_size = min(patch_size, h, w)
    prediction = torch.zeros((b, 1, h, w), dtype=torch.float32, device=image.device)
    count_map = torch.zeros((b, 1, h, w), dtype=torch.float32, device=image.device)

    ys = list(range(0, max(h - patch_size + 1, 1), stride))
    xs = list(range(0, max(w - patch_size + 1, 1), stride))
    if len(ys) == 0 or ys[-1] != h - patch_size:
        ys.append(h - patch_size)
    if len(xs) == 0 or xs[-1] != w - patch_size:
        xs.append(w - patch_size)
    ys = sorted(set(ys))
    xs = sorted(set(xs))

    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = image[:, :, y:y + patch_size, x:x + patch_size]
                patch_pred = model(patch)
                if isinstance(patch_pred, (tuple, list)):
                    patch_pred = patch_pred[0]
                if patch_pred.shape[-2:] != (patch_size, patch_size):
                    patch_pred = torch.nn.functional.interpolate(
                        patch_pred,
                        size=(patch_size, patch_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                prediction[:, :, y:y + patch_size, x:x + patch_size] += patch_pred
                count_map[:, :, y:y + patch_size, x:x + patch_size] += 1.0
    prediction = prediction / torch.clamp(count_map, min=1.0)
    return prediction


def predict_full_image_aligned(
    model: nn.Module,
    image: torch.Tensor,
    patch_size: int,
    stride: int,
    device: torch.device,
) -> torch.Tensor:
    if official_predict_full_image is not None:
        # The official helper keeps patch predictions on CPU internally.
        # Force CPU input here to avoid CPU/CUDA tensor mixing during accumulation.
        image_cpu = image.detach().cpu() if image.is_cuda else image
        return official_predict_full_image(
            model=model,
            image=image_cpu,
            patch_size=patch_size,
            stride=stride,
            device=device,
        )
    return predict_full_image_local(
        model=model,
        image=image,
        patch_size=patch_size,
        stride=stride,
        device=device,
    )


def parse_targets(text: str) -> List[str]:
    out: List[str] = []
    for item in text.split(","):
        t = item.strip()
        if not t:
            continue
        key = t.upper().replace("-", "").replace("_", "")
        if key == "DRIVE":
            out.append("DRIVE")
        elif key == "STARE":
            out.append("STARE")
        elif key in {"CHASEDB1", "CHASEDB", "CHASE"}:
            out.append("CHASEDB1")
        else:
            raise ValueError(f"Unknown target: {t}")
    if not out:
        raise ValueError("No valid targets provided.")
    return out


def _read_gray(path: str) -> np.ndarray:
    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        if img.ndim == 3:
            img = img[:, :, 1]
        img = img.astype(np.float32)
    else:
        if Image is None:
            raise ImportError("Missing both cv2 and PIL.")
        img = np.array(Image.open(path))
        if img.ndim == 3:
            img = img[:, :, 1]
        img = img.astype(np.float32)

    if img.max() > 1.0:
        img = img / 255.0
    return img


def _read_mask(path: str) -> np.ndarray:
    mask = _read_gray(path)
    return (mask > 0.5).astype(np.float32)


def _resolve_resized_hw(h: int, w: int, scale: float) -> Tuple[int, int]:
    scale = max(1e-6, float(scale))
    new_h = max(1, int(round(int(h) * scale)))
    new_w = max(1, int(round(int(w) * scale)))
    return new_h, new_w


def _resize_2d_array(arr: np.ndarray, out_h: int, out_w: int, mode: str = "area") -> np.ndarray:
    src = np.asarray(arr, dtype=np.float32)
    if src.ndim != 2:
        raise ValueError(f"Expected 2D array for resize, got shape={src.shape}")
    if src.shape == (int(out_h), int(out_w)):
        return src.astype(np.float32, copy=False)

    mode_key = str(mode or "area").lower()
    if cv2 is not None:
        interp_map = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
        }
        interp = interp_map.get(mode_key, cv2.INTER_AREA)
        if mode_key == "area" and (int(out_h) > src.shape[0] or int(out_w) > src.shape[1]):
            interp = cv2.INTER_LINEAR
        out = cv2.resize(src, (int(out_w), int(out_h)), interpolation=interp)
        return out.astype(np.float32)

    if Image is None:
        raise ImportError("Missing both cv2 and PIL for resize.")

    resample_map = {
        "nearest": Image.Resampling.NEAREST,
        "linear": Image.Resampling.BILINEAR,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "area": Image.Resampling.BOX,
    }
    resample = resample_map.get(mode_key, Image.Resampling.BOX)
    src_uint8 = np.clip(src * 255.0, 0.0, 255.0).astype(np.uint8)
    pil_img = Image.fromarray(src_uint8, mode="L")
    out = np.array(pil_img.resize((int(out_w), int(out_h)), resample=resample), dtype=np.float32)
    return out / 255.0


def _compute_fov_mask(
    img: np.ndarray,
    thresh: float = 0.05,
    blur_ksize: int = 7,
    close_ksize: int = 15,
) -> np.ndarray:
    """Compute FOV mask from grayscale image in [0,1], aligned with eval_chase.py."""
    m = img.copy()
    if cv2 is not None and blur_ksize and blur_ksize > 0:
        k = int(blur_ksize)
        if k % 2 == 0:
            k += 1
        m = cv2.GaussianBlur(m, (k, k), 0)

    mask = (m > float(thresh)).astype(np.uint8)

    if cv2 is not None and close_ksize and close_ksize > 0:
        k = int(close_ksize)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask.astype(np.float32)


def _list_files(folder: str) -> List[str]:
    p = Path(folder)
    if not p.exists():
        return []
    files = [
        str(x)
        for x in p.iterdir()
        if x.is_file() and x.suffix.lower() in ALLOWED_EXTS
    ]
    files.sort()
    return files


def _find_subdir(root: str, candidates: List[str]) -> str:
    for c in candidates:
        d = os.path.join(root, c)
        if os.path.isdir(d):
            return d
    raise RuntimeError(f"Could not find subdir in {root}, expected one of {candidates}")


def _candidate_chase_mask_stems(image_stem: str, mask_type: str = "auto") -> List[str]:
    mask_type = (mask_type or "auto").lower()
    if mask_type not in {"auto", "1stho", "2ndho"}:
        mask_type = "auto"
    if mask_type == "1stho":
        suffixes = ["1stHO"]
    elif mask_type == "2ndho":
        suffixes = ["2ndHO"]
    else:
        suffixes = ["1stHO", "2ndHO"]

    cands: List[str] = []
    if "_rot" in image_stem:
        base, rot = image_stem.rsplit("_rot", 1)
        for suf in suffixes:
            cands.append(f"{base}_rot{rot}_{suf}")
            cands.append(f"{base}_{suf}_rot{rot}")
            cands.append(f"{image_stem}_{suf}")
    else:
        for suf in suffixes:
            cands.append(f"{image_stem}_{suf}")
    cands.append(image_stem)
    return cands


def _pair_images_and_masks(
    im_dir: str,
    mask_dir: str,
    target_name: str,
    chase_mask_type: str,
) -> List[Tuple[str, str]]:
    image_files = _list_files(im_dir)
    if not image_files:
        raise RuntimeError(f"No test images found in {im_dir}")

    mask_map: Dict[str, str] = {}
    for p in _list_files(mask_dir):
        mask_map[Path(p).stem] = p

    pairs: List[Tuple[str, str]] = []
    missing: List[str] = []
    for im_path in image_files:
        stem = Path(im_path).stem
        if target_name == "CHASEDB1":
            candidates = _candidate_chase_mask_stems(stem, mask_type=chase_mask_type)
        else:
            candidates = [stem]

        mask_path = None
        for cand in candidates:
            if cand in mask_map:
                mask_path = mask_map[cand]
                break
        if mask_path is None:
            fallback = os.path.join(mask_dir, f"{stem}.png")
            if os.path.exists(fallback):
                mask_path = fallback

        if mask_path is None:
            missing.append(Path(im_path).name)
            continue
        pairs.append((im_path, mask_path))

    if missing:
        print(f"[Warn] {target_name}: {len(missing)} images without masks, e.g. {missing[:5]}")
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found for {target_name} in {im_dir} and {mask_dir}")
    return pairs


class CrossDatasetTestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        target_name: str,
        chase_mask_type: str = "auto",
        normalize_input: bool = False,
        use_fov_mask: bool = False,
        fov_threshold: float = 0.05,
        fov_blur: int = 7,
        fov_close: int = 15,
        resize_scale: float = 1.0,
        resize_interpolation: str = "area",
    ):
        test_root = os.path.join(data_dir, "test")
        # Keep DRIVE exactly aligned with legacy eval scripts: test/image + test/label.
        if target_name == "DRIVE":
            im_dir = _find_subdir(test_root, ["image", "images", "im"])
            mask_dir = _find_subdir(test_root, ["label", "labels"])
        else:
            # Keep STARE/CHASE behavior aligned with their existing dataset scripts.
            im_dir = _find_subdir(test_root, ["im", "image", "images"])
            mask_dir = _find_subdir(test_root, ["label", "labels"])
        self.target_name = str(target_name)
        self.normalize_input = bool(normalize_input)
        self.use_fov_mask = bool(use_fov_mask)
        self.fov_threshold = float(fov_threshold)
        self.fov_blur = int(fov_blur)
        self.fov_close = int(fov_close)
        self.resize_scale = float(resize_scale)
        self.resize_interpolation = str(resize_interpolation)
        self.pairs = _pair_images_and_masks(
            im_dir=im_dir,
            mask_dir=mask_dir,
            target_name=target_name,
            chase_mask_type=chase_mask_type,
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.pairs[idx]
        image_raw = _read_gray(image_path)
        mask = _read_mask(mask_path)
        if abs(self.resize_scale - 1.0) > 1e-8:
            new_h, new_w = _resolve_resized_hw(image_raw.shape[0], image_raw.shape[1], self.resize_scale)
            image_raw = _resize_2d_array(image_raw, new_h, new_w, mode=self.resize_interpolation)
            mask = _resize_2d_array(mask, new_h, new_w, mode="nearest")
            mask = (mask > 0.5).astype(np.float32)

        image = image_raw
        if self.normalize_input:
            image = (image - np.mean(image)) / (np.std(image) + 1e-8)
        if self.target_name == "CHASEDB1" and self.use_fov_mask:
            fov = _compute_fov_mask(
                image_raw,
                thresh=self.fov_threshold,
                blur_ksize=self.fov_blur,
                close_ksize=self.fov_close,
            )
        else:
            fov = np.ones_like(mask, dtype=np.float32)
        image_t = torch.from_numpy(image[None, ...].astype(np.float32))
        mask_t = torch.from_numpy(mask[None, ...].astype(np.float32))
        fov_t = torch.from_numpy(fov[None, ...].astype(np.float32))
        name = Path(image_path).name
        return image_t, mask_t, name, fov_t


def get_test_loader_target(
    data_dir: str,
    target_name: str,
    batch_size: int,
    num_workers: int,
    chase_mask_type: str,
    normalize_input: bool,
    use_fov_mask: bool,
    fov_threshold: float,
    fov_blur: int,
    fov_close: int,
    resize_scale: float = 1.0,
    resize_interpolation: str = "area",
):
    # For DRIVE, prefer the official loader to match eval_drive exactly.
    if target_name == "DRIVE":
        _load_official_drive_helpers_if_needed()
        if abs(float(resize_scale) - 1.0) > 1e-8:
            print("[Warn] DRIVE official loader bypassed because resize_scale != 1.0.")
        elif official_get_test_loader is not None:
            if not normalize_input:
                print("[Warn] DRIVE official loader always applies normalize=True; ignore normalize_input=False.")
            return official_get_test_loader(
                data_dir=data_dir,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        print("[Warn] DRIVE official loader unavailable; fallback to local loader.")

    ds = CrossDatasetTestDataset(
        data_dir=data_dir,
        target_name=target_name,
        chase_mask_type=chase_mask_type,
        normalize_input=normalize_input,
        use_fov_mask=use_fov_mask,
        fov_threshold=fov_threshold,
        fov_blur=fov_blur,
        fov_close=fov_close,
        resize_scale=resize_scale,
        resize_interpolation=resize_interpolation,
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def confusion_from_scores(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Tuple[int, int, int, int]:
    pred = (scores > threshold).astype(np.uint8)
    labels = labels.astype(np.uint8)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    return tp, tn, fp, fn


def metrics_from_confusion(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    se = tp / (tp + fn + 1e-8)
    spe = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    f1 = 2.0 * precision * se / (precision + se + 1e-8)
    return {
        "F1": float(f1),
        "Se": float(se),
        "Spe": float(spe),
        "Acc": float(acc),
        "Precision": float(precision),
    }


def calc_auc_safe(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    if len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


def calc_pr_auc_safe(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    if len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(labels, scores))
    except Exception:
        return float("nan")


def metrics_at_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    auc_value: Optional[float] = None,
    pr_auc_value: Optional[float] = None,
) -> Dict[str, float]:
    tp, tn, fp, fn = confusion_from_scores(labels, scores, threshold)
    m = metrics_from_confusion(tp, tn, fp, fn)
    m["AUC"] = float(calc_auc_safe(labels, scores) if auc_value is None else auc_value)
    m["PR_AUC"] = float(calc_pr_auc_safe(labels, scores) if pr_auc_value is None else pr_auc_value)
    m["tp"] = int(tp)
    m["tn"] = int(tn)
    m["fp"] = int(fp)
    m["fn"] = int(fn)
    m["n_pixels"] = int(labels.size)
    m["n_pos"] = int((labels == 1).sum())
    m["n_neg"] = int((labels == 0).sum())
    m["Threshold"] = float(threshold)
    return m


def get_infer_params_for_target(target_name: str, args: argparse.Namespace) -> Tuple[int, int]:
    if target_name == "CHASEDB1":
        return int(args.chase_patch_size), int(args.chase_stride)
    return int(args.patch_size), int(args.stride)


def compute_threshold_sweep(labels: np.ndarray, scores: np.ndarray, thr_values: List[float]) -> pd.DataFrame:
    rows = []
    auc = calc_auc_safe(labels, scores)
    pr_auc = calc_pr_auc_safe(labels, scores)
    for thr in thr_values:
        tp, tn, fp, fn = confusion_from_scores(labels, scores, thr)
        m = metrics_from_confusion(tp, tn, fp, fn)
        rows.append({
            "threshold": float(thr),
            "F1": m["F1"],
            "AUC": auc,
            "PR_AUC": pr_auc,
            "Se": m["Se"],
            "Spe": m["Spe"],
            "Acc": m["Acc"],
            "Precision": m["Precision"],
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        })
    return pd.DataFrame(rows)


def select_threshold_from_sweep(
    sweep_df: pd.DataFrame,
    threshold_policy: str,
    f1_drop_tolerance: float,
) -> float:
    if sweep_df is None or sweep_df.empty:
        raise ValueError("Empty threshold sweep dataframe.")

    policy = (threshold_policy or "fixed").lower()
    if policy == "fixed":
        return float(sweep_df.iloc[0]["threshold"])

    if policy == "best_f1":
        best_f1 = float(sweep_df["F1"].max())
        cands = sweep_df[sweep_df["F1"] >= best_f1 - 1e-12]
        cands = cands.sort_values(by=["Se", "threshold"], ascending=[False, True])
        return float(cands.iloc[0]["threshold"])

    if policy == "max_se_at_f1":
        best_f1 = float(sweep_df["F1"].max())
        cands = sweep_df[sweep_df["F1"] >= best_f1 - float(f1_drop_tolerance)]
        if cands.empty:
            cands = sweep_df
        cands = cands.sort_values(by=["Se", "F1", "threshold"], ascending=[False, False, True])
        return float(cands.iloc[0]["threshold"])

    raise ValueError(f"Unknown threshold_policy: {threshold_policy}")


def evaluate_one_model_seed(
    model_key: str,
    ckpt_path: str,
    seed: int,
    target_name: str,
    loader,
    device: torch.device,
    patch_size: int,
    stride: int,
    threshold: float,
    threshold_policy: str = "fixed",
    auto_thresholds: Optional[List[float]] = None,
    auto_f1_drop_tolerance: float = 0.0005,
    use_fov_mask: bool = False,
    sweep_thresholds: Optional[List[float]] = None,
) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
    model = build_model(model_key, device)
    model, _ = load_checkpoint_to_model(model, ckpt_path, device)
    infer_model = WrappedDeepSupervisionModel(model) if MODEL_SPECS[model_key]["use_deep_supervision"] else model
    infer_model.eval()

    score_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{target_name}-seed{seed}", leave=False):
            if isinstance(batch, (list, tuple)) and len(batch) == 4:
                images, masks, _, fov_masks = batch
            else:
                images, masks, _ = batch
                fov_masks = None
            # Keep image on CPU here; backend predictor handles device transfer.
            image = images
            pred = predict_full_image_aligned(
                model=infer_model,
                image=image,
                patch_size=patch_size,
                stride=stride,
                device=device,
            )
            score = pred.detach().cpu().numpy()[0, 0].astype(np.float32)
            score = np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
            score = np.clip(score, 0.0, 1.0)
            label = (masks.numpy()[0, 0] > 0.5).astype(np.uint8)
            if (
                target_name == "CHASEDB1"
                and use_fov_mask
                and fov_masks is not None
            ):
                fov = (fov_masks.numpy()[0, 0] > 0.5)
                if np.any(fov):
                    score_flat = score[fov].reshape(-1)
                    label_flat = label[fov].reshape(-1)
                else:
                    score_flat = score.reshape(-1)
                    label_flat = label.reshape(-1)
            else:
                score_flat = score.reshape(-1)
                label_flat = label.reshape(-1)

            score_chunks.append(score_flat)
            label_chunks.append(label_flat)

    if not score_chunks:
        raise RuntimeError(f"No predictions produced for target={target_name}, seed={seed}")

    scores_all = np.concatenate(score_chunks, axis=0)
    labels_all = np.concatenate(label_chunks, axis=0)

    auc = calc_auc_safe(labels_all, scores_all)
    pr_auc = calc_pr_auc_safe(labels_all, scores_all)
    base_metrics = metrics_at_threshold(labels_all, scores_all, threshold, auc_value=auc, pr_auc_value=pr_auc)
    metrics = dict(base_metrics)

    auto_thr_used = float(threshold)
    if (threshold_policy or "fixed").lower() != "fixed":
        thr_values = auto_thresholds
        if not thr_values:
            thr_values = build_threshold_values(0.05, 0.95, 0.01)
        auto_sweep_df = compute_threshold_sweep(labels_all, scores_all, thr_values)
        auto_thr_used = select_threshold_from_sweep(
            auto_sweep_df,
            threshold_policy=threshold_policy,
            f1_drop_tolerance=auto_f1_drop_tolerance,
        )
        metrics = metrics_at_threshold(
            labels_all,
            scores_all,
            auto_thr_used,
            auc_value=auc,
            pr_auc_value=pr_auc,
        )

    metrics["ThresholdBase"] = float(threshold)
    metrics["ThresholdAuto"] = float(auto_thr_used)
    metrics["threshold_policy"] = str(threshold_policy)
    metrics["F1_base"] = float(base_metrics["F1"])
    metrics["Se_base"] = float(base_metrics["Se"])
    metrics["Spe_base"] = float(base_metrics["Spe"])
    metrics["Acc_base"] = float(base_metrics["Acc"])
    metrics["Precision_base"] = float(base_metrics["Precision"])

    sweep_df = None
    if sweep_thresholds:
        sweep_df = compute_threshold_sweep(labels_all, scores_all, sweep_thresholds)

    del score_chunks, label_chunks, scores_all, labels_all, infer_model, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, sweep_df


def format_mean_std(mean: float, std: float, digits: int = 4) -> str:
    if np.isnan(mean):
        return "nan"
    if np.isnan(std):
        std = 0.0
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def build_summary(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = ["Source (Train)", "Target (Test)", "Method"]
    metric_cols = ["F1", "AUC", "PR_AUC", "Se", "Spe", "Acc"]
    rows_num: List[Dict] = []
    rows_table: List[Dict] = []

    for keys, grp in raw_df.groupby(group_cols, sort=False):
        source, target, method = keys
        row_num: Dict[str, object] = {
            "Source (Train)": source,
            "Target (Test)": target,
            "Method": method,
            "n_seed": int(len(grp)),
        }
        row_table: Dict[str, object] = {
            "Source (Train)": source,
            "Target (Test)": target,
            "Method": method,
        }
        for m in metric_cols:
            mean = float(grp[m].mean())
            std = float(grp[m].std(ddof=1)) if len(grp) > 1 else 0.0
            row_num[f"{m}_mean"] = mean
            row_num[f"{m}_std"] = std
            row_table[m] = format_mean_std(mean, std, digits=4)
        rows_num.append(row_num)
        rows_table.append(row_table)

    df_num = pd.DataFrame(rows_num)
    df_table = pd.DataFrame(rows_table)
    return df_num, df_table


def build_best_ours_vs_baseline(summary_num_df: pd.DataFrame) -> pd.DataFrame:
    if summary_num_df is None or summary_num_df.empty:
        return pd.DataFrame()

    out_rows: List[Dict[str, object]] = []
    targets = list(summary_num_df["Target (Test)"].dropna().unique())
    for target in targets:
        grp = summary_num_df[summary_num_df["Target (Test)"] == target]
        baseline = grp[grp["Method"] == "Baseline"]
        ours = grp[grp["Method"].astype(str).str.startswith("Ours(")]
        if baseline.empty or ours.empty:
            continue

        b = baseline.iloc[0]
        o = ours.sort_values(by=["F1_mean", "AUC_mean"], ascending=[False, False]).iloc[0]
        out_rows.append({
            "Source (Train)": b["Source (Train)"],
            "Target (Test)": target,
            "Baseline_Method": b["Method"],
            "Baseline_F1_mean": float(b["F1_mean"]),
            "Best_Ours_Method": o["Method"],
            "Best_Ours_F1_mean": float(o["F1_mean"]),
            "Delta_F1_mean": float(o["F1_mean"] - b["F1_mean"]),
            "Baseline_AUC_mean": float(b["AUC_mean"]),
            "Best_Ours_AUC_mean": float(o["AUC_mean"]),
            "Delta_AUC_mean": float(o["AUC_mean"] - b["AUC_mean"]),
            "Baseline_PR_AUC_mean": float(b["PR_AUC_mean"]),
            "Best_Ours_PR_AUC_mean": float(o["PR_AUC_mean"]),
            "Delta_PR_AUC_mean": float(o["PR_AUC_mean"] - b["PR_AUC_mean"]),
        })

    return pd.DataFrame(out_rows)


def build_threshold_values(start: float, end: float, step: float) -> List[float]:
    vals: List[float] = []
    cur = start
    while cur <= end + 1e-9:
        vals.append(round(cur, 6))
        cur += step
    return vals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DRIVE-source cross-dataset evaluation for baseline and ours (aligned/diagnostic version).")
    parser.add_argument("--models_root", type=str, default="./models", help="Root folder containing checkpoints.")
    parser.add_argument("--data_root", type=str, default="./data", help="Root folder containing datasets.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results/drive_source_cross_dataset",
        help="Output folder.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    parser.add_argument("--seeds", type=str, default="20,42,80", help="Comma-separated seeds.")
    parser.add_argument("--targets", type=str, default="STARE,CHASEDB1", help="Comma-separated target datasets.")
    parser.add_argument("--methods", type=str, default="baseline,ours", help="Comma-separated methods: baseline,ours")
    parser.add_argument(
        "--ours_lambda_list",
        type=str,
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="Lambda list for Ours checkpoints, e.g. 0.1,0.2,0.3",
    )
    parser.add_argument(
        "--ours_models_subdir",
        type=str,
        default="drive",
        help="Subdir under models_root for Ours lambda checkpoints.",
    )
    parser.add_argument("--baseline_thr", type=float, default=0.5, help="Threshold for baseline.")
    parser.add_argument("--ours_thr", type=float, default=0.4, help="Threshold for ours.")
    parser.add_argument(
        "--threshold_policy",
        type=str,
        default="best_f1",
        choices=["fixed", "best_f1", "max_se_at_f1"],
        help="Threshold policy: fixed uses method thresholds; others search on target scores per seed.",
    )
    parser.add_argument("--auto_thr_start", type=float, default=0.05, help="Auto-threshold search start.")
    parser.add_argument("--auto_thr_end", type=float, default=0.95, help="Auto-threshold search end.")
    parser.add_argument("--auto_thr_step", type=float, default=0.01, help="Auto-threshold search step.")
    parser.add_argument(
        "--auto_f1_drop_tolerance",
        type=float,
        default=0.0005,
        help="Only for max_se_at_f1: Se max under F1 >= best_F1 - tolerance.",
    )
    parser.add_argument("--patch_size", type=int, default=48, help="Default patch size for DRIVE/STARE.")
    parser.add_argument("--stride", type=int, default=16, help="Default stride for DRIVE/STARE.")
    parser.add_argument("--chase_patch_size", type=int, default=96, help="Patch size for CHASEDB1 (aligned with eval_chase.py).")
    parser.add_argument("--chase_stride", type=int, default=32, help="Stride for CHASEDB1 (aligned with eval_chase.py).")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for test loader.")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument(
        "--chase_mask_type",
        type=str,
        default="1stho",
        choices=["auto", "1stho", "2ndho"],
        help="Mask matching mode for CHASEDB1. Default set to 1stHO for alignment.",
    )
    parser.add_argument(
        "--normalize_mode",
        type=str,
        default="all",
        choices=["none", "drive", "all"],
        help="Input normalization mode. Default all matches DRIVE train/test normalization style.",
    )
    parser.add_argument(
        "--use_fov_mask",
        action="store_true",
        help="Apply CHASEDB1 FOV mask during metric computation (aligned with eval_chase.py).",
    )
    parser.add_argument("--fov_threshold", type=float, default=0.05, help="FOV mask intensity threshold.")
    parser.add_argument("--fov_blur", type=int, default=7, help="FOV Gaussian blur kernel size (odd, 0 disables).")
    parser.add_argument("--fov_close", type=int, default=15, help="FOV morph-close kernel size (odd, 0 disables).")
    parser.add_argument("--diagnose_threshold_sweep", action="store_true", help="Run threshold sweep diagnostics and save csv files.")
    parser.add_argument("--sweep_start", type=float, default=0.10, help="Threshold sweep start.")
    parser.add_argument("--sweep_end", type=float, default=0.50, help="Threshold sweep end.")
    parser.add_argument("--sweep_step", type=float, default=0.05, help="Threshold sweep step.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.ours_models_subdir = canonicalize_ours_models_subdir(args.ours_models_subdir)
    os.makedirs(args.results_dir, exist_ok=True)

    seeds = parse_int_list(args.seeds)
    targets = parse_targets(args.targets)
    method_order = parse_method_list(args.methods)
    if "DRIVE" in targets:
        _load_official_drive_helpers_if_needed()
    ours_lambdas = parse_float_list(args.ours_lambda_list)
    if not ours_lambdas:
        raise ValueError("ours_lambda_list is empty.")
    eval_jobs = build_eval_jobs(method_order, ours_lambdas)

    use_cuda = args.device.lower() == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    method_thresholds = {
        "baseline": float(args.baseline_thr),
        "ours": float(args.ours_thr),
    }
    auto_thresholds = None
    if args.threshold_policy != "fixed":
        auto_thresholds = build_threshold_values(args.auto_thr_start, args.auto_thr_end, args.auto_thr_step)
    sweep_thresholds = None
    if args.diagnose_threshold_sweep:
        sweep_thresholds = build_threshold_values(args.sweep_start, args.sweep_end, args.sweep_step)

    meta = {
        "source_train": SOURCE_NAME,
        "models_root": args.models_root,
        "data_root": args.data_root,
        "results_dir": args.results_dir,
        "device_requested": args.device,
        "device_actual": str(device),
        "seeds": seeds,
        "targets": targets,
        "methods": method_order,
        "ours_lambda_list": ours_lambdas,
        "ours_models_subdir": args.ours_models_subdir,
        "method_thresholds": method_thresholds,
        "threshold_policy": args.threshold_policy,
        "auto_thresholds": auto_thresholds,
        "auto_f1_drop_tolerance": float(args.auto_f1_drop_tolerance),
        "patch_size": int(args.patch_size),
        "stride": int(args.stride),
        "chase_patch_size": int(args.chase_patch_size),
        "chase_stride": int(args.chase_stride),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "chase_mask_type": args.chase_mask_type,
        "normalize_mode": args.normalize_mode,
        "use_fov_mask": bool(args.use_fov_mask),
        "fov_threshold": float(args.fov_threshold),
        "fov_blur": int(args.fov_blur),
        "fov_close": int(args.fov_close),
        "diagnose_threshold_sweep": bool(args.diagnose_threshold_sweep),
        "sweep_thresholds": sweep_thresholds,
        "official_predict_full_image_available": bool(official_predict_full_image is not None),
        "official_get_test_loader_available": bool(official_get_test_loader is not None),
    }
    with open(os.path.join(args.results_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    target_loaders = {}
    for target in targets:
        target_data_dir = os.path.join(args.data_root, target)
        if args.normalize_mode == "all":
            normalize_input = True
        elif args.normalize_mode == "drive":
            normalize_input = (target == "DRIVE")
        else:
            normalize_input = False

        loader = get_test_loader_target(
            data_dir=target_data_dir,
            target_name=target,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            chase_mask_type=args.chase_mask_type,
            normalize_input=normalize_input,
            use_fov_mask=bool(args.use_fov_mask and target == "CHASEDB1"),
            fov_threshold=args.fov_threshold,
            fov_blur=args.fov_blur,
            fov_close=args.fov_close,
        )
        target_loaders[target] = loader
        pz, st = get_infer_params_for_target(target, args)
        print(
            f"[Data] target={target}, test_images={len(loader.dataset)}, patch={pz}, "
            f"stride={st}, normalize={'yes' if normalize_input else 'no'}, "
            f"fov_mask={'yes' if (args.use_fov_mask and target == 'CHASEDB1') else 'no'}"
        )

    rows: List[Dict] = []

    for target in targets:
        patch_size, stride = get_infer_params_for_target(target, args)

        for job in eval_jobs:
            method_name = str(job["method_name"])
            model_key = str(job["model_key"])
            method_display = str(job["method_display"])
            lambda_ds_job = float(job["lambda_ds"]) if not np.isnan(job["lambda_ds"]) else float("nan")
            threshold = method_thresholds[method_name]

            for seed in seeds:
                lambda_ds_for_path = 0.0 if np.isnan(lambda_ds_job) else lambda_ds_job
                ckpt_path = resolve_model_path(
                    models_root=args.models_root,
                    model_key=model_key,
                    seed=seed,
                    lambda_ds=lambda_ds_for_path,
                    ours_models_subdir=args.ours_models_subdir,
                )
                if not os.path.exists(ckpt_path):
                    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

                print("=" * 100)
                print(
                    f"[Run] source={SOURCE_NAME}, target={target}, method={method_display}, "
                    f"seed={seed}, thr_base={threshold:.3f}, policy={args.threshold_policy}, "
                    f"patch={patch_size}, stride={stride}, "
                    f"lambda={'N/A' if np.isnan(lambda_ds_job) else f'{lambda_ds_job:.1f}'}, "
                    f"chase_mask_type={args.chase_mask_type if target == 'CHASEDB1' else 'N/A'}, "
                    f"fov_mask={'yes' if (args.use_fov_mask and target == 'CHASEDB1') else 'no'}"
                )

                metrics, sweep_df = evaluate_one_model_seed(
                    model_key=model_key,
                    ckpt_path=ckpt_path,
                    seed=seed,
                    target_name=target,
                    loader=target_loaders[target],
                    device=device,
                    patch_size=patch_size,
                    stride=stride,
                    threshold=threshold,
                    threshold_policy=args.threshold_policy,
                    auto_thresholds=auto_thresholds,
                    auto_f1_drop_tolerance=args.auto_f1_drop_tolerance,
                    use_fov_mask=bool(args.use_fov_mask and target == "CHASEDB1"),
                    sweep_thresholds=sweep_thresholds,
                )

                row = {
                    "Source (Train)": SOURCE_NAME,
                    "Target (Test)": TARGET_DISPLAY.get(target, target),
                    "Method": method_display,
                    "target_key": target,
                    "method_key": method_name,
                    "lambda_ds": lambda_ds_job,
                    "seed": int(seed),
                    "threshold": float(metrics["Threshold"]),
                    "threshold_base": float(metrics["ThresholdBase"]),
                    "threshold_auto": float(metrics["ThresholdAuto"]),
                    "threshold_policy": args.threshold_policy,
                    "patch_size": int(patch_size),
                    "stride": int(stride),
                    "checkpoint": ckpt_path,
                    "F1": metrics["F1"],
                    "AUC": metrics["AUC"],
                    "PR_AUC": metrics["PR_AUC"],
                    "Se": metrics["Se"],
                    "Spe": metrics["Spe"],
                    "Acc": metrics["Acc"],
                    "Precision": metrics["Precision"],
                    "tp": metrics["tp"],
                    "tn": metrics["tn"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "n_pos": metrics["n_pos"],
                    "n_neg": metrics["n_neg"],
                    "n_pixels": metrics["n_pixels"],
                    "F1_base": metrics["F1_base"],
                    "Se_base": metrics["Se_base"],
                    "Spe_base": metrics["Spe_base"],
                    "Acc_base": metrics["Acc_base"],
                    "Precision_base": metrics["Precision_base"],
                }
                rows.append(row)
                print(
                    f"[Done] target={target}, method={method_display}, seed={seed} | "
                    f"thr={row['threshold']:.3f} (base={row['threshold_base']:.3f}), "
                    f"F1={row['F1']:.6f}, AUC={row['AUC']:.6f}, PR_AUC={row['PR_AUC']:.6f}, "
                    f"Se={row['Se']:.6f}, Spe={row['Spe']:.6f}, Acc={row['Acc']:.6f}"
                )

                if sweep_df is not None:
                    lambda_tag = "" if np.isnan(lambda_ds_job) else f"_lambda{lambda_ds_job:.1f}".replace(".", "_")
                    sweep_name = f"thr_sweep_{SOURCE_NAME}_to_{target}_{method_name}{lambda_tag}_seed{seed}.csv"
                    sweep_path = os.path.join(args.results_dir, sweep_name)
                    sweep_df.to_csv(sweep_path, index=False, encoding="utf-8-sig", float_format="%.6f")
                    best_idx = int(sweep_df["F1"].values.argmax())
                    best_row = sweep_df.iloc[best_idx]
                    print(
                        f"[Sweep] best_F1={best_row['F1']:.6f} at thr={best_row['threshold']:.3f}; "
                        f"Se={best_row['Se']:.6f}, Spe={best_row['Spe']:.6f} | saved={sweep_name}"
                    )

    raw_df = pd.DataFrame(rows)
    raw_path = os.path.join(args.results_dir, "drive_source_cross_dataset_raw.csv")
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig", float_format="%.6f")

    summary_num_df, summary_table_df = build_summary(raw_df)
    summary_num_path = os.path.join(args.results_dir, "drive_source_cross_dataset_summary_numeric.csv")
    summary_num_df.to_csv(summary_num_path, index=False, encoding="utf-8-sig", float_format="%.6f")

    summary_table_path = os.path.join(args.results_dir, "drive_source_cross_dataset_summary_table.csv")
    summary_table_df.to_csv(summary_table_path, index=False, encoding="utf-8-sig")

    best_compare_df = build_best_ours_vs_baseline(summary_num_df)
    best_compare_path = os.path.join(args.results_dir, "drive_source_cross_dataset_best_ours_vs_baseline.csv")
    if not best_compare_df.empty:
        best_compare_df.to_csv(best_compare_path, index=False, encoding="utf-8-sig", float_format="%.6f")

    print("\n" + "=" * 100)
    print("Summary (mean +/- std over seeds):")
    print(summary_table_df.to_string(index=False))
    if not best_compare_df.empty:
        print("-" * 100)
        print("Best Ours(lambda=*) vs Baseline:")
        print(best_compare_df.to_string(index=False))
    print("-" * 100)
    print(f"Raw csv: {raw_path}")
    print(f"Numeric summary csv: {summary_num_path}")
    print(f"Table summary csv: {summary_table_path}")
    if not best_compare_df.empty:
        print(f"Best compare csv: {best_compare_path}")
    print(f"Meta json: {os.path.join(args.results_dir, 'run_meta.json')}")
    print("=" * 100)


if __name__ == "__main__":
    main()

