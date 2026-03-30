#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the Newton collection autoresearch dashboard."""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "newton_collection_autoresearch"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from protomotions.utils.newton_collection_dashboard import write_dashboard


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the autoresearch dashboard from run artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", type=pathlib.Path, default=REPO_ROOT)
    parser.add_argument("--results-dir", type=pathlib.Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = write_dashboard(
        repo_root=args.repo_root.resolve(),
        results_root=args.results_dir.resolve(),
    )
    print(args.results_dir.resolve() / "index.html")
    print(payload["generated_at"])


if __name__ == "__main__":
    main()
