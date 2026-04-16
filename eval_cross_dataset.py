#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

import cross_dataset_common as core
import eval_drive_source_cross_dataset as eval_base
from public_release_utils import (
    PUBLIC_PRESETS,
    default_public_eval_dir,
    ensure_dir,
    ensure_public_preset,
    get_eval_defaults,
    infer_lambda,
    infer_method,
    infer_seed,
    infer_threshold,
    merge_missing_defaults,
    resolve_weight_path,
    save_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a published cross-dataset checkpoint on the preset target dataset.",
    )
    parser.add_argument("--preset", required=True, choices=PUBLIC_PRESETS, help="Cross-dataset preset to evaluate.")
    parser.add_argument("--weights", required=True, help="Checkpoint path or release alias from weights/weights_manifest.json.")
    parser.add_argument("--method", choices=["baseline", "ours"], default=None, help="Model family. Defaults to the alias metadata when available.")
    parser.add_argument("--seed", type=int, default=None, help="Seed label written into the output summary.")
    parser.add_argument("--lambda_ds", type=float, default=None, help="Lambda label written for Ours checkpoints.")
    parser.add_argument("--data_root", type=str, default="./data", help="Dataset root containing DRIVE/STARE/CHASEDB1.")
    parser.add_argument("--results_dir", type=str, default=None, help="Output directory for csv/json summaries.")
    parser.add_argument("--device", type=str, default="cuda", help="Requested device, e.g. 'cuda' or 'cpu'.")
    parser.add_argument("--threshold", type=float, default=None, help="Fixed decision threshold. Defaults to the manifest recommendation or preset default.")
    parser.add_argument("--search_threshold", dest="search_threshold", action="store_true", help="Search a better threshold on the evaluation split.")
    parser.add_argument("--no_search_threshold", dest="search_threshold", action="store_false", help="Use the fixed threshold directly.")
    parser.set_defaults(search_threshold=None)
    parser.add_argument("--search_mode", type=str, default=None, choices=["max_se_at_base_f1", "best_f1"], help="Threshold search mode for DRIVE/STARE.")
    parser.add_argument("--thr_min", type=float, default=None, help="Threshold search lower bound.")
    parser.add_argument("--thr_max", type=float, default=None, help="Threshold search upper bound.")
    parser.add_argument("--thr_step", type=float, default=None, help="Threshold search step size.")
    parser.add_argument("--f1_drop_tolerance", type=float, default=None, help="Allowed F1 drop when maximizing sensitivity.")
    parser.add_argument("--patch_size", type=int, default=None, help="Patch size used during full-image inference.")
    parser.add_argument("--stride", type=int, default=None, help="Patch stride used during full-image inference.")
    parser.add_argument("--batch_size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--chase_mask_type", type=str, default=None, help="CHASEDB1 mask variant: 1stHO or 2ndHO.")
    parser.add_argument("--chase_use_fov_mask", dest="chase_use_fov_mask", action="store_true", help="Evaluate CHASEDB1 inside the FOV mask.")
    parser.add_argument("--no_chase_use_fov_mask", dest="chase_use_fov_mask", action="store_false", help="Disable CHASEDB1 FOV masking.")
    parser.set_defaults(chase_use_fov_mask=None)
    parser.add_argument("--chase_optimize_mode", type=str, default=None, choices=["se", "f1"], help="CHASEDB1 threshold search objective.")
    parser.add_argument("--fov_threshold", type=float, default=None, help="FOV threshold.")
    parser.add_argument("--fov_blur", type=int, default=None, help="FOV blur kernel size.")
    parser.add_argument("--fov_close", type=int, default=None, help="FOV closing kernel size.")
    parser.add_argument("--normalize_input", dest="normalize_input", action="store_true", help="Normalize target images before inference.")
    parser.add_argument("--no_normalize_input", dest="normalize_input", action="store_false", help="Disable input normalization.")
    parser.set_defaults(normalize_input=None)
    parser.add_argument("--target_resize_scale", type=float, default=None, help="Optional target resize scale before inference.")
    parser.add_argument("--target_resize_interpolation", type=str, default=None, choices=["area", "bilinear", "bicubic", "nearest"], help="Interpolation used when resize_scale != 1.")
    parser.add_argument("--diagnose_threshold_sweep", action="store_true", help="Save a threshold sweep csv for the evaluated checkpoint.")
    parser.add_argument("--sweep_start", type=float, default=0.10, help="Threshold sweep start.")
    parser.add_argument("--sweep_end", type=float, default=0.50, help="Threshold sweep end.")
    parser.add_argument("--sweep_step", type=float, default=0.05, help="Threshold sweep step.")
    parser.add_argument("--output_name", type=str, default="evaluation", help="Base filename for the written csv/json summary.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.preset = ensure_public_preset(args.preset)

    eval_defaults = get_eval_defaults(args.preset)
    merge_missing_defaults(args, eval_defaults, skip_keys=("results_dir",))
    if args.results_dir is None:
        args.results_dir = str(default_public_eval_dir(args.preset))

    ckpt_path, manifest_entry = resolve_weight_path(args.weights)
    method_name = infer_method(args.method, manifest_entry, fallback="ours")
    seed = infer_seed(args.seed, manifest_entry, fallback=42)
    lambda_ds = infer_lambda(args.lambda_ds, manifest_entry, fallback=0.0)
    threshold = infer_threshold(args.threshold, method_name, manifest_entry, eval_defaults)
    optimize_threshold = bool(args.search_threshold)
    search_mode = str(args.search_mode or (eval_defaults["baseline_search_mode"] if method_name == "baseline" else eval_defaults["ours_search_mode"]))

    preset = core.PRESETS[args.preset]
    target_name = str(preset["target"])
    results_dir = ensure_dir(Path(args.results_dir))
    device = torch.device("cuda" if args.device.lower() == "cuda" and torch.cuda.is_available() else "cpu")

    loader = eval_base.get_test_loader_target(
        data_dir=str(Path(args.data_root) / target_name),
        target_name=target_name,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        chase_mask_type=str(args.chase_mask_type),
        normalize_input=bool(args.normalize_input),
        use_fov_mask=bool(args.chase_use_fov_mask and target_name == "CHASEDB1"),
        fov_threshold=float(args.fov_threshold),
        fov_blur=int(args.fov_blur),
        fov_close=int(args.fov_close),
        resize_scale=float(args.target_resize_scale),
        resize_interpolation=str(args.target_resize_interpolation),
    )

    model_key = core.BASELINE_MODEL_KEY if method_name == "baseline" else core.OURS_MODEL_KEY
    sweep_thresholds = (
        eval_base.build_threshold_values(float(args.sweep_start), float(args.sweep_end), float(args.sweep_step))
        if bool(args.diagnose_threshold_sweep)
        else None
    )
    metrics, sweep_df = core.evaluate_one_model_seed(
        target_name=target_name,
        method_name=method_name,
        model_key=model_key,
        ckpt_path=str(ckpt_path),
        seed=seed,
        loader=loader,
        device=device,
        patch_size=int(args.patch_size),
        stride=int(args.stride),
        threshold_base=float(threshold),
        optimize_threshold=optimize_threshold,
        thr_min=float(args.thr_min),
        thr_max=float(args.thr_max),
        thr_step=float(args.thr_step),
        f1_drop_tolerance=float(args.f1_drop_tolerance),
        drive_stare_search_mode=search_mode,
        chase_optimize_mode=str(args.chase_optimize_mode),
        use_fov_mask=bool(args.chase_use_fov_mask and target_name == "CHASEDB1"),
        sweep_thresholds=sweep_thresholds,
    )

    row = {
        "Source (Train)": " + ".join(list(preset["sources"])),
        "Target (Test)": eval_base.TARGET_DISPLAY.get(target_name, target_name),
        "Method": "Baseline" if method_name == "baseline" else f"Ours(lambda={lambda_ds:.1f})",
        "preset": args.preset,
        "seed": int(seed),
        "lambda_ds": float(lambda_ds),
        "checkpoint": str(ckpt_path),
        "weight_alias": manifest_entry.get("alias") if manifest_entry else None,
        "release_asset": manifest_entry.get("release_asset") if manifest_entry else None,
    }
    row.update(metrics)

    summary_df = pd.DataFrame([row])
    summary_csv = results_dir / f"{args.output_name}_summary.csv"
    summary_json = results_dir / f"{args.output_name}_summary.json"
    summary_df.to_csv(summary_csv, index=False, float_format="%.6f")
    save_json(summary_json, {"rows": summary_df.to_dict(orient="records")})

    if sweep_df is not None and not sweep_df.empty:
        sweep_df.to_csv(results_dir / f"{args.output_name}_threshold_sweep.csv", index=False)

    run_meta = {
        "preset": args.preset,
        "display_name": str(preset["display_name"]),
        "target": target_name,
        "weights_argument": args.weights,
        "checkpoint": str(ckpt_path),
        "weight_alias": manifest_entry.get("alias") if manifest_entry else None,
        "method": method_name,
        "seed": int(seed),
        "lambda_ds": float(lambda_ds),
        "threshold_base": float(threshold),
        "search_threshold": bool(optimize_threshold),
        "search_mode": search_mode,
        "patch_size": int(args.patch_size),
        "stride": int(args.stride),
        "normalize_input": bool(args.normalize_input),
        "target_resize_scale": float(args.target_resize_scale),
        "target_resize_interpolation": str(args.target_resize_interpolation),
        "chase_use_fov_mask": bool(args.chase_use_fov_mask),
        "device_requested": args.device,
        "device_actual": str(device),
        "timestamp": datetime.now().isoformat(),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
    }
    save_json(results_dir / "run_meta.json", run_meta)

    print(f"[Eval] preset={args.preset} target={target_name} method={row['Method']}")
    print(
        f"[Eval] F1={float(metrics['F1']):.4f} AUC={float(metrics['AUC']):.4f} "
        f"Se={float(metrics['Se']):.4f} Spe={float(metrics['Spe']):.4f} "
        f"thr={float(metrics['ThresholdAuto']):.3f}"
    )
    print(f"[Eval] summary_csv={summary_csv}")
    print(f"[Eval] summary_json={summary_json}")


if __name__ == "__main__":
    main()
