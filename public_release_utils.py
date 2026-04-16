#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import eval_drive_source_cross_dataset as eval_base
from unet_baseline_model import create_unet_baseline

import cross_dataset_common as core


ROOT_DIR = Path(__file__).resolve().parent
PUBLIC_PRESETS: Tuple[str, ...] = (
    "drive_stare_to_chase",
    "stare_chase_to_drive",
    "drive_chase_to_stare",
)
WEIGHTS_MANIFEST_PATH = ROOT_DIR / "weights" / "weights_manifest.json"


def ensure_public_preset(preset: str) -> str:
    preset_key = str(preset).strip()
    if preset_key not in PUBLIC_PRESETS:
        raise ValueError(
            f"Unsupported preset: {preset_key}. Supported presets: {', '.join(PUBLIC_PRESETS)}"
        )
    return preset_key


def default_public_models_dir(preset: str) -> Path:
    return ROOT_DIR / "outputs" / "models" / ensure_public_preset(preset)


def default_public_eval_dir(preset: str) -> Path:
    return ROOT_DIR / "outputs" / "eval" / ensure_public_preset(preset)


def load_weights_manifest() -> Dict[str, Dict[str, Any]]:
    if not WEIGHTS_MANIFEST_PATH.exists():
        return {}
    with WEIGHTS_MANIFEST_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("weights_manifest.json must contain a top-level object.")
    return data


def get_weight_entry(token: str) -> Optional[Dict[str, Any]]:
    token = str(token).strip()
    if not token:
        return None

    manifest = load_weights_manifest()
    if token in manifest:
        entry = dict(manifest[token])
        entry["alias"] = token
        return entry

    for alias, item in manifest.items():
        release_asset = str(item.get("release_asset", "")).strip()
        if not release_asset:
            continue
        if token == release_asset or token == Path(release_asset).stem:
            entry = dict(item)
            entry["alias"] = alias
            return entry
    return None


def resolve_weight_path(weights_arg: str) -> Tuple[Path, Optional[Dict[str, Any]]]:
    raw = str(weights_arg).strip()
    if not raw:
        raise FileNotFoundError("Empty --weights argument.")

    candidate_paths = []
    raw_path = Path(raw)
    if raw_path.is_absolute():
        candidate_paths.append(raw_path)
    else:
        candidate_paths.extend([Path.cwd() / raw_path, ROOT_DIR / raw_path])
    for path in candidate_paths:
        if path.exists():
            return path.resolve(), None

    entry = get_weight_entry(raw)
    if entry is None:
        raise FileNotFoundError(
            f"Could not resolve weights '{raw}'. Pass a checkpoint path or one of the aliases in "
            f"{WEIGHTS_MANIFEST_PATH.as_posix()}."
        )

    release_asset = str(entry["release_asset"])
    release_candidates = [
        ROOT_DIR / "weights" / release_asset,
        Path.cwd() / "weights" / release_asset,
        ROOT_DIR / release_asset,
        Path.cwd() / release_asset,
    ]
    for path in release_candidates:
        if path.exists():
            return path.resolve(), entry

    alias = entry.get("alias", raw)
    raise FileNotFoundError(
        f"Weights alias '{alias}' maps to '{release_asset}', but that file was not found locally. "
        f"Download the release asset and place it under './weights/', or pass an explicit checkpoint path."
    )


def merge_missing_defaults(namespace, defaults: Dict[str, Any], skip_keys: Optional[Sequence[str]] = None):
    skipped = set(skip_keys or [])
    for key, value in defaults.items():
        if key in skipped:
            continue
        if not hasattr(namespace, key) or getattr(namespace, key) is None:
            setattr(namespace, key, value)
    return namespace


def get_train_defaults(preset: str) -> Dict[str, Any]:
    preset = ensure_public_preset(preset)
    return vars(core.build_train_parser(preset).parse_args([])).copy()


def get_eval_defaults(preset: str) -> Dict[str, Any]:
    preset = ensure_public_preset(preset)
    return vars(core.build_eval_parser(preset).parse_args([])).copy()


def infer_method(method: Optional[str], entry: Optional[Dict[str, Any]], fallback: str = "ours") -> str:
    if method:
        return str(method).strip().lower()
    if entry and entry.get("method"):
        return str(entry["method"]).strip().lower()
    return str(fallback).strip().lower()


def infer_seed(seed: Optional[int], entry: Optional[Dict[str, Any]], fallback: int = 42) -> int:
    if seed is not None:
        return int(seed)
    if entry and entry.get("seed") is not None:
        return int(entry["seed"])
    return int(fallback)


def infer_lambda(lambda_ds: Optional[float], entry: Optional[Dict[str, Any]], fallback: float = 0.0) -> float:
    if lambda_ds is not None:
        return float(lambda_ds)
    if entry and entry.get("lambda_ds") is not None:
        return float(entry["lambda_ds"])
    return float(fallback)


def infer_threshold(
    threshold: Optional[float],
    method: str,
    entry: Optional[Dict[str, Any]],
    eval_defaults: Dict[str, Any],
) -> float:
    if threshold is not None:
        return float(threshold)
    if entry and entry.get("recommended_threshold") is not None:
        return float(entry["recommended_threshold"])
    if str(method).lower() == "baseline":
        return float(eval_defaults["baseline_thr"])
    return float(eval_defaults["ours_thr"])


def load_inference_model(method: str, ckpt_path: Path, device):
    method_key = str(method).strip().lower()
    if method_key == "baseline":
        model = create_unet_baseline(base_channels=32).to(device)
        use_deep_supervision = False
    elif method_key == "ours":
        model = eval_base.build_model(core.OURS_MODEL_KEY, device)
        use_deep_supervision = bool(eval_base.MODEL_SPECS[core.OURS_MODEL_KEY]["use_deep_supervision"])
    else:
        raise ValueError(f"Unsupported method: {method_key}")

    model, meta = eval_base.load_checkpoint_to_model(model, str(ckpt_path), device)
    infer_model = eval_base.WrappedDeepSupervisionModel(model) if use_deep_supervision else model
    infer_model.eval()
    return infer_model, meta


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(core.safe_convert_for_json(payload), f, indent=2, ensure_ascii=False)
