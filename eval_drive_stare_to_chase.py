#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from cross_dataset_common import run_eval_preset


LOCAL_EVAL_DEFAULTS = {
    "baseline_thr": 0.5,
    "baseline_search_threshold": True,
    "baseline_search_mode": "best_f1",
    "ours_thr": 0.4,
    "ours_search_threshold": True,
    "ours_search_mode": "best_f1",
    "thr_min": 0.01,
    "thr_max": 0.8,
    "thr_step": 0.01,
    "patch_size": 96,
    "stride": 32,
    "normalize_input": True,
    "chase_use_fov_mask": True,
    "chase_optimize_mode": "f1",
    "checkpoint_policy": "best",
    "recent_checkpoint_keep": 0,
    "target_resize_scale": 0.5,
    "target_resize_interpolation": "area",
}


def main() -> None:
    run_eval_preset("drive_stare_to_chase", local_eval_defaults=LOCAL_EVAL_DEFAULTS)


if __name__ == "__main__":
    main()
