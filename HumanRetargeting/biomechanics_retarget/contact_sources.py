#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Optional contact sidecar loading for biomechanics trials."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .subject_profiles import SubjectProfile
except ImportError:
    from subject_profiles import SubjectProfile


def _load_np_container(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.ndim == 0:
        item = data.item()
        if isinstance(item, dict):
            return {
                key: np.asarray(value)
                for key, value in item.items()
            }
    if hasattr(data, "files"):
        return {key: np.asarray(data[key]) for key in data.files}
    raise ValueError(f"Unsupported NumPy sidecar format: {path}")


def _extract_contacts(container: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if "left_foot_contacts" in container and "right_foot_contacts" in container:
        return np.asarray(container["left_foot_contacts"]), np.asarray(
            container["right_foot_contacts"]
        )
    if "foot_contacts" in container:
        foot_contacts = np.asarray(container["foot_contacts"])
        if foot_contacts.ndim != 2 or foot_contacts.shape[1] < 2:
            raise ValueError("foot_contacts must have shape [T, >=2]")
        return foot_contacts[:, [0]], foot_contacts[:, [1]]
    raise ValueError("Contact sidecar must contain left/right foot contacts or foot_contacts")


def _normalize_contact_shape(values: np.ndarray, expected_frames: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[1] == 1:
        arr = np.repeat(arr, 2, axis=1)
    if arr.shape[0] >= expected_frames:
        return arr[:expected_frames]
    if arr.shape[0] == 0:
        return np.zeros((expected_frames, 2), dtype=float)
    padding = np.repeat(arr[-1:], expected_frames - arr.shape[0], axis=0)
    return np.concatenate([arr, padding], axis=0)


def _load_csv_contacts(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    normalized = {column.lower(): column for column in frame.columns}
    left_name = normalized.get("left_contact") or normalized.get("left_foot_contact")
    right_name = normalized.get("right_contact") or normalized.get("right_foot_contact")
    if not left_name or not right_name:
        raise ValueError(
            f"CSV contact sidecar must include left_contact/right_contact columns: {path}"
        )
    return frame[left_name].to_numpy(), frame[right_name].to_numpy()


def load_trial_contacts(
    profile: SubjectProfile,
    trial_stem: str,
    expected_frames: int,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Load optional measured contacts for one trial."""
    source = profile.contact_source
    if source == "heuristic":
        return None

    candidates: list[Path] = []
    for pattern in (profile.event_glob, profile.grf_glob):
        if not pattern:
            continue
        search_glob = pattern
        if not Path(pattern).is_absolute():
            search_glob = str((profile.input_dir / pattern).resolve())
        candidates.extend(Path(match) for match in sorted(glob.glob(search_glob)))

    # Deterministic fallback: common sidecar locations near the input directory.
    if not candidates:
        for extension in (".npz", ".npy", ".csv"):
            local = profile.input_dir / f"{trial_stem}_contacts{extension}"
            if local.exists():
                candidates.append(local)

    for path in candidates:
        if trial_stem not in path.stem:
            continue
        if path.suffix in {".npz", ".npy"}:
            left, right = _extract_contacts(_load_np_container(path))
        elif path.suffix == ".csv":
            left, right = _load_csv_contacts(path)
        else:
            continue
        return (
            _normalize_contact_shape(left, expected_frames),
            _normalize_contact_shape(right, expected_frames),
            path.name,
        )

    return None
