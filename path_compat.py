#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from typing import Iterable, Optional


MODEL_PATH_ALIASES = {
    "models/cross_dataset_step6": "models/cross_dataset",
    "models/step6_stare": "models/stare",
    "models/step6_chase": "models/chase",
    "models/step6_sccd_filtered": "models/sccd_filtered",
    "models/step6_crack_filtered": "models/crack_filtered",
    "models/step6_isic2018": "models/isic2018",
    "models/step6": "models/drive",
}

OURS_MODELS_SUBDIR_ALIASES = {
    "step6": "drive",
}


def _replace_segment_path(path: str, old: str, new: str) -> str:
    normalized = str(path).replace("\\", "/")
    old_parts = old.split("/")
    new_parts = new.split("/")
    parts = normalized.split("/")

    for idx in range(len(parts) - len(old_parts) + 1):
        if parts[idx : idx + len(old_parts)] == old_parts:
            parts = parts[:idx] + new_parts + parts[idx + len(old_parts) :]
            break

    return "/".join(parts)


def canonicalize_models_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None

    resolved = str(path)
    for old, new in sorted(MODEL_PATH_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        updated = _replace_segment_path(resolved, old, new)
        if updated != resolved:
            resolved = updated
            break

    return os.path.normpath(resolved)


def apply_model_path_aliases(namespace, field_names: Iterable[str]):
    for field_name in field_names:
        if hasattr(namespace, field_name):
            value = getattr(namespace, field_name)
            if value:
                setattr(namespace, field_name, canonicalize_models_path(value))
    return namespace


def canonicalize_ours_models_subdir(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    return OURS_MODELS_SUBDIR_ALIASES.get(str(name), str(name))
