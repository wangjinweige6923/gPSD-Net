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
    "thr_min": 0.08,
    "thr_max": 0.12,
    "thr_step": 0.01,
    "incremental_save": True,
}


def main() -> None:
    run_eval_preset("stare_chase_to_drive", local_eval_defaults=LOCAL_EVAL_DEFAULTS)


if __name__ == "__main__":
    main()
