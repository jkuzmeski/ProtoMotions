#!/usr/bin/env python3
"""Inspect a keypoints .npy file and print its basic structure."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def inspect_file(file_path: Path) -> None:
    data = np.load(file_path, allow_pickle=True)
    print(f"File: {file_path}")
    print(f"Type: {type(data)}")
    print(f"Shape: {data.shape}")

    if data.shape == ():
        print("Data is a 0-d array (scalar). Content:")
        value = data.item()
        print(value)
        if isinstance(value, dict):
            print("Keys:", value.keys())
            for key, val in value.items():
                if isinstance(val, np.ndarray):
                    print(f"Key '{key}': shape {val.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a .npy file.")
    parser.add_argument("file_path", type=Path, help="Path to the .npy file")
    args = parser.parse_args()
    inspect_file(args.file_path)


if __name__ == "__main__":
    main()
