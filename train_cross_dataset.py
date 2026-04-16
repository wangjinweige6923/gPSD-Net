#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

import cross_dataset_common as core
import eval_drive_source_cross_dataset as eval_base
from public_release_utils import (
    PUBLIC_PRESETS,
    default_public_models_dir,
    ensure_public_preset,
    get_train_defaults,
    merge_missing_defaults,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Public training entrypoint for the cross-dataset paper presets.",
    )
    parser.add_argument("--preset", required=True, choices=PUBLIC_PRESETS, help="Cross-dataset preset to train.")
    parser.add_argument("--data_root", type=str, default="./data", help="Dataset root containing DRIVE/STARE/CHASEDB1.")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory for checkpoints and training logs.")
    parser.add_argument("--device", type=str, default="cuda", help="Requested device, e.g. 'cuda' or 'cpu'.")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated methods: baseline,ours.")
    parser.add_argument("--seed", type=int, default=42, help="Single training seed.")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed list. Overrides --seed.")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay.")
    parser.add_argument("--patch_size", type=int, default=None, help="Patch size for source sampling.")
    parser.add_argument("--stride", type=int, default=None, help="Patch stride for source sampling.")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader worker count.")
    parser.add_argument("--patience", type=int, default=None, help="Early-stop patience.")
    parser.add_argument("--val_ratio", type=float, default=None, help="Validation split ratio per source dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing checkpoints.")
    parser.add_argument("--ours_lambda_ds", type=float, default=None, help="Single deep-supervision lambda for Ours.")
    parser.add_argument("--ours_lambda_list", type=str, default=None, help="Comma-separated lambda grid for Ours.")
    parser.add_argument("--ours_grid_mode", dest="ours_grid_mode", action="store_true", help="Train all lambdas in --ours_lambda_list.")
    parser.add_argument("--no_ours_grid_mode", dest="ours_grid_mode", action="store_false", help="Train only --ours_lambda_ds.")
    parser.set_defaults(ours_grid_mode=None)
    parser.add_argument("--chase_mask_type", type=str, default=None, help="CHASEDB1 mask variant: 1stHO or 2ndHO.")
    parser.add_argument("--chase_min_foreground_ratio", type=float, default=None, help="Minimum foreground ratio for CHASEDB1 patches.")
    parser.add_argument("--chase_augment", dest="chase_augment", action="store_true", help="Enable CHASEDB1 augmentation.")
    parser.add_argument("--no_chase_augment", dest="chase_augment", action="store_false", help="Disable CHASEDB1 augmentation.")
    parser.set_defaults(chase_augment=None)
    parser.add_argument("--chase_brightness_range", type=str, default=None, help="Brightness range 'low,high' for CHASEDB1 augmentation.")
    parser.add_argument("--chase_contrast_range", type=str, default=None, help="Contrast range 'low,high' for CHASEDB1 augmentation.")
    parser.add_argument("--chase_use_fov_mask", action="store_true", help="Use CHASEDB1 FOV masks during patch selection.")
    parser.add_argument("--fov_threshold", type=float, default=None, help="FOV threshold.")
    parser.add_argument("--fov_blur", type=int, default=None, help="FOV blur kernel size.")
    parser.add_argument("--fov_close", type=int, default=None, help="FOV closing kernel size.")
    parser.add_argument("--fov_min_ratio", type=float, default=None, help="Minimum FOV ratio required for sampled patches.")
    parser.add_argument("--normalize_input", dest="normalize_input", action="store_true", help="Normalize source images before training.")
    parser.add_argument("--no_normalize_input", dest="normalize_input", action="store_false", help="Disable input normalization.")
    parser.set_defaults(normalize_input=None)
    parser.add_argument("--source_balance", dest="source_balance", action="store_true", help="Balance patches across source datasets.")
    parser.add_argument("--no_source_balance", dest="source_balance", action="store_false", help="Disable source balancing.")
    parser.set_defaults(source_balance=None)
    parser.add_argument("--save_recent_checkpoints", dest="save_recent_checkpoints", action="store_true", help="Keep recent epoch checkpoints.")
    parser.add_argument("--no_save_recent_checkpoints", dest="save_recent_checkpoints", action="store_false", help="Save only best checkpoint.")
    parser.set_defaults(save_recent_checkpoints=None)
    parser.add_argument("--recent_checkpoint_keep", type=int, default=None, help="How many recent checkpoints to keep when enabled.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.preset = ensure_public_preset(args.preset)

    defaults = get_train_defaults(args.preset)
    merge_missing_defaults(args, defaults, skip_keys=("save_dir",))
    if args.save_dir is None:
        args.save_dir = str(default_public_models_dir(args.preset))

    methods = eval_base.parse_method_list(args.methods)
    seeds = core.parse_int_list(args.seeds) if args.seeds else [int(args.seed)]
    if not seeds:
        raise ValueError("Seed list is empty.")

    for seed in seeds:
        args.seed = int(seed)
        if "baseline" in methods:
            core.train_single_experiment(args.preset, "baseline", 0.0, args)
        if "ours" in methods:
            lambda_values = (
                core.parse_float_list(args.ours_lambda_list)
                if bool(args.ours_grid_mode)
                else [float(args.ours_lambda_ds)]
            )
            if not lambda_values:
                raise ValueError("Ours lambda list is empty.")
            for lambda_ds in lambda_values:
                core.train_single_experiment(args.preset, "ours", float(lambda_ds), args)


if __name__ == "__main__":
    main()
