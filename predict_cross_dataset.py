#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import cross_dataset_common as core
import eval_drive_source_cross_dataset as eval_base
from public_release_utils import (
    PUBLIC_PRESETS,
    ensure_dir,
    ensure_public_preset,
    get_eval_defaults,
    infer_method,
    infer_threshold,
    load_inference_model,
    merge_missing_defaults,
    resolve_weight_path,
    save_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run prediction on an input directory using a published cross-dataset checkpoint.",
    )
    parser.add_argument("--preset", required=True, choices=PUBLIC_PRESETS, help="Cross-dataset preset the checkpoint belongs to.")
    parser.add_argument("--weights", required=True, help="Checkpoint path or release alias from weights/weights_manifest.json.")
    parser.add_argument("--input_dir", required=True, help="Directory containing images for prediction.")
    parser.add_argument("--output_dir", required=True, help="Directory to save probability, mask, and overlay outputs.")
    parser.add_argument("--method", choices=["baseline", "ours"], default=None, help="Model family. Defaults to alias metadata when available.")
    parser.add_argument("--device", type=str, default="cuda", help="Requested device, e.g. 'cuda' or 'cpu'.")
    parser.add_argument("--threshold", type=float, default=None, help="Binary decision threshold. Defaults to manifest recommendation or preset default.")
    parser.add_argument("--patch_size", type=int, default=None, help="Patch size used during full-image inference.")
    parser.add_argument("--stride", type=int, default=None, help="Patch stride used during full-image inference.")
    parser.add_argument("--normalize_input", dest="normalize_input", action="store_true", help="Normalize each image before inference.")
    parser.add_argument("--no_normalize_input", dest="normalize_input", action="store_false", help="Disable input normalization.")
    parser.set_defaults(normalize_input=None)
    parser.add_argument("--target_resize_scale", type=float, default=None, help="Optional target resize scale before inference.")
    parser.add_argument("--target_resize_interpolation", type=str, default=None, choices=["area", "bilinear", "bicubic", "nearest"], help="Interpolation used when resize_scale != 1.")
    parser.add_argument("--chase_use_fov_mask", dest="chase_use_fov_mask", action="store_true", help="Apply CHASEDB1 FOV masking to outputs.")
    parser.add_argument("--no_chase_use_fov_mask", dest="chase_use_fov_mask", action="store_false", help="Disable CHASEDB1 FOV masking.")
    parser.set_defaults(chase_use_fov_mask=None)
    parser.add_argument("--fov_threshold", type=float, default=None, help="FOV threshold.")
    parser.add_argument("--fov_blur", type=int, default=None, help="FOV blur kernel size.")
    parser.add_argument("--fov_close", type=int, default=None, help="FOV closing kernel size.")
    parser.add_argument("--max_images", type=int, default=None, help="Optional maximum number of images to process.")
    return parser


def _write_gray_png(path: Path, arr: np.ndarray) -> None:
    data = np.asarray(arr, dtype=np.float32)
    data = np.clip(data, 0.0, 1.0)
    img_uint8 = (data * 255.0).round().astype(np.uint8)
    if core.cv2 is not None:
        core.cv2.imwrite(str(path), img_uint8)
        return
    if core.Image is None:
        raise ImportError("Missing both cv2 and PIL; cannot write PNG files.")
    core.Image.fromarray(img_uint8, mode="L").save(path)


def _write_rgb_png(path: Path, arr: np.ndarray) -> None:
    data = np.asarray(arr, dtype=np.float32)
    data = np.clip(data, 0.0, 1.0)
    img_uint8 = (data * 255.0).round().astype(np.uint8)
    if core.cv2 is not None:
        core.cv2.imwrite(str(path), img_uint8[:, :, ::-1])
        return
    if core.Image is None:
        raise ImportError("Missing both cv2 and PIL; cannot write PNG files.")
    core.Image.fromarray(img_uint8, mode="RGB").save(path)


def _build_overlay(base_gray: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    gray = np.asarray(base_gray, dtype=np.float32)
    gray = np.clip(gray, 0.0, 1.0)
    mask = (np.asarray(binary_mask, dtype=np.float32) > 0.5).astype(np.float32)
    rgb = np.stack([gray, gray, gray], axis=-1)
    color = np.zeros_like(rgb)
    color[..., 0] = 1.0
    alpha = 0.55 * mask[..., None]
    return rgb * (1.0 - alpha) + color * alpha


def _list_input_images(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Input directory does not exist: {folder}")
    files = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in core.ALLOWED_EXTS
    ]
    files.sort()
    if not files:
        raise RuntimeError(f"No supported images found in: {folder}")
    return files


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.preset = ensure_public_preset(args.preset)

    eval_defaults = get_eval_defaults(args.preset)
    merge_missing_defaults(args, eval_defaults)

    ckpt_path, manifest_entry = resolve_weight_path(args.weights)
    method_name = infer_method(args.method, manifest_entry, fallback="ours")
    threshold = infer_threshold(args.threshold, method_name, manifest_entry, eval_defaults)
    target_name = str(core.PRESETS[args.preset]["target"])

    device = torch.device("cuda" if args.device.lower() == "cuda" and torch.cuda.is_available() else "cpu")
    infer_model, checkpoint_meta = load_inference_model(method_name, ckpt_path, device)
    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(Path(args.output_dir))

    image_paths = _list_input_images(input_dir)
    if args.max_images is not None:
        image_paths = image_paths[: max(0, int(args.max_images))]
    if not image_paths:
        raise RuntimeError("No images selected for prediction.")

    manifest_rows: List[Dict[str, object]] = []
    for image_path in image_paths:
        image_raw = core._read_gray(str(image_path))
        image_eval = image_raw

        if abs(float(args.target_resize_scale) - 1.0) > 1e-8:
            new_h, new_w = eval_base._resolve_resized_hw(
                image_raw.shape[0], image_raw.shape[1], float(args.target_resize_scale)
            )
            image_eval = eval_base._resize_2d_array(
                image_raw, new_h, new_w, mode=str(args.target_resize_interpolation)
            )

        model_input = image_eval.astype(np.float32, copy=True)
        if bool(args.normalize_input):
            model_input = (model_input - np.mean(model_input)) / (np.std(model_input) + 1e-8)

        image_tensor = torch.from_numpy(model_input[None, None, ...].astype(np.float32))
        with torch.no_grad():
            pred = eval_base.predict_full_image_aligned(
                model=infer_model,
                image=image_tensor,
                patch_size=int(args.patch_size),
                stride=int(args.stride),
                device=device,
            )
        prob = pred.detach().cpu().numpy()[0, 0].astype(np.float32)
        prob = np.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)
        prob = np.clip(prob, 0.0, 1.0)

        if target_name == "CHASEDB1" and bool(args.chase_use_fov_mask):
            fov = core._compute_fov_mask(
                image_eval,
                thresh=float(args.fov_threshold),
                blur_ksize=int(args.fov_blur),
                close_ksize=int(args.fov_close),
            )
            prob = prob * fov
        binary = (prob > float(threshold)).astype(np.float32)
        overlay = _build_overlay(image_eval, binary)

        stem = image_path.stem
        input_png = output_dir / f"{stem}_input.png"
        prob_png = output_dir / f"{stem}_prob.png"
        pred_png = output_dir / f"{stem}_pred.png"
        overlay_png = output_dir / f"{stem}_overlay.png"
        _write_gray_png(input_png, image_eval)
        _write_gray_png(prob_png, prob)
        _write_gray_png(pred_png, binary)
        _write_rgb_png(overlay_png, overlay)

        manifest_rows.append(
            {
                "input_image": str(image_path),
                "saved_input_png": str(input_png),
                "saved_prob_png": str(prob_png),
                "saved_pred_png": str(pred_png),
                "saved_overlay_png": str(overlay_png),
                "threshold": float(threshold),
                "height": int(image_eval.shape[0]),
                "width": int(image_eval.shape[1]),
            }
        )

    run_meta = {
        "preset": args.preset,
        "display_name": str(core.PRESETS[args.preset]["display_name"]),
        "target": target_name,
        "weights_argument": args.weights,
        "checkpoint": str(ckpt_path),
        "weight_alias": manifest_entry.get("alias") if manifest_entry else None,
        "release_asset": manifest_entry.get("release_asset") if manifest_entry else None,
        "method": method_name,
        "threshold": float(threshold),
        "patch_size": int(args.patch_size),
        "stride": int(args.stride),
        "normalize_input": bool(args.normalize_input),
        "target_resize_scale": float(args.target_resize_scale),
        "target_resize_interpolation": str(args.target_resize_interpolation),
        "chase_use_fov_mask": bool(args.chase_use_fov_mask),
        "device_requested": args.device,
        "device_actual": str(device),
        "checkpoint_meta_keys": sorted(list(checkpoint_meta.keys())) if isinstance(checkpoint_meta, dict) else [],
        "timestamp": datetime.now().isoformat(),
        "outputs": manifest_rows,
    }
    save_json(output_dir / "prediction_manifest.json", run_meta)

    print(f"[Predict] preset={args.preset} target={target_name} method={method_name}")
    print(f"[Predict] checkpoint={ckpt_path}")
    print(f"[Predict] output_dir={output_dir}")
    print(f"[Predict] images_processed={len(manifest_rows)} threshold={float(threshold):.3f}")


if __name__ == "__main__":
    main()
