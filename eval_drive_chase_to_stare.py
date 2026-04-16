#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from cross_dataset_common import run_eval_preset


LOCAL_EVAL_DEFAULTS = {
    "baseline_thr": 0.7,
    "baseline_search_threshold": False,
    "baseline_search_mode": "best_f1",
    "ours_thr": 0.59,
    "ours_search_threshold": False,
    "ours_search_mode": "max_se_at_base_f1",
    "thr_min": 0.30,
    "thr_max": 0.60,
    "thr_step": 0.01,
}


def main() -> None:
    run_eval_preset("drive_chase_to_stare", local_eval_defaults=LOCAL_EVAL_DEFAULTS)


if __name__ == "__main__":
    main()
