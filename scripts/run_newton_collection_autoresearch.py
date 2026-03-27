#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run an autoresearch-style loop against Newton collection throughput."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import shutil
import subprocess
import sys
from typing import Iterable


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PROGRAM = REPO_ROOT / "scripts" / "newton_collection_program.md"
DEFAULT_ALLOWED_FILES = (
    "protomotions/simulator/newton/simulator.py",
    "protomotions/simulator/newton/config.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Codex-driven research loop over Newton collection throughput.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--benchmark-cmd",
        type=str,
        default=(
            "python examples/benchmark/newton_rollout_bench.py "
            "--robot-name smpl_lower_body_subject_S_GENERIC "
            "--experiment-path examples/experiments/mimic/mlp_bm_bootstrap.py "
            "--experiment-name s_generic_teacher_every_other_rollout_bench "
            "--motion-file HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/"
            "yaml_data/experiment_matrix/every_other.yaml "
            "--num-envs 1024 "
            "--benchmark-seconds 120 "
            "--json"
        ),
    )
    parser.add_argument("--program", type=pathlib.Path, default=DEFAULT_PROGRAM)
    parser.add_argument(
        "--allowed-file",
        action="append",
        default=[],
        help="Path relative to repo root. Repeat to allow more files.",
    )
    parser.add_argument("--model", type=str, default="")
    parser.add_argument(
        "--codex-bin",
        type=str,
        default="codex",
    )
    parser.add_argument(
        "--workspace",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/newton_collection_autoresearch"),
        help="Temporary workspace used for the loop.",
    )
    parser.add_argument(
        "--results-dir",
        type=pathlib.Path,
        default=REPO_ROOT / "results" / "newton_collection_autoresearch",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="samples_per_s",
        help="Benchmark metric key to maximize.",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.0,
        help="Required absolute improvement in the chosen metric to accept a change.",
    )
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Keep the temporary workspace after the run.",
    )
    return parser.parse_args()


def _run(
    cmd: list[str],
    *,
    cwd: pathlib.Path,
    env: dict[str, str] | None = None,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        capture_output=capture_output,
    )


def _git_status_entries(repo_root: pathlib.Path) -> list[tuple[str, str]]:
    status = _run(["git", "status", "--porcelain"], cwd=repo_root)
    if status.returncode != 0:
        raise RuntimeError(f"Failed to query git status:\n{status.stderr}")
    entries: list[tuple[str, str]] = []
    for line in status.stdout.splitlines():
        if not line:
            continue
        state = line[:2]
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        entries.append((state, path))
    return entries


def _prepare_workspace(repo_root: pathlib.Path, workspace: pathlib.Path) -> None:
    if workspace.exists():
        raise RuntimeError(
            f"Workspace already exists: {workspace}. Remove it or choose a new path."
        )

    worktree = _run(
        ["git", "worktree", "add", "--detach", str(workspace), "HEAD"],
        cwd=repo_root,
    )
    if worktree.returncode != 0:
        raise RuntimeError(f"Failed to create worktree:\n{worktree.stderr}")


def _sync_working_tree_changes(
    repo_root: pathlib.Path,
    workspace: pathlib.Path,
    status_entries: list[tuple[str, str]],
) -> None:
    for state, rel_path in status_entries:
        src = repo_root / rel_path
        dst = workspace / rel_path

        # If the file is deleted in the source working tree, reflect that in the workspace.
        if "D" in state and not src.exists():
            if dst.exists():
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            continue

        if not src.exists():
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _remove_workspace(repo_root: pathlib.Path, workspace: pathlib.Path) -> None:
    if not workspace.exists():
        return
    cleanup = _run(["git", "worktree", "remove", "--force", str(workspace)], cwd=repo_root)
    if cleanup.returncode != 0:
        raise RuntimeError(f"Failed to remove worktree:\n{cleanup.stderr}")


def _capture_files(root: pathlib.Path, paths: Iterable[str]) -> dict[str, bytes]:
    snapshot: dict[str, bytes] = {}
    for rel_path in paths:
        snapshot[rel_path] = (root / rel_path).read_bytes()
    return snapshot


def _restore_files(root: pathlib.Path, snapshot: dict[str, bytes]) -> None:
    for rel_path, data in snapshot.items():
        (root / rel_path).write_bytes(data)


def _git_diff_names(root: pathlib.Path) -> list[str]:
    result = _run(["git", "diff", "--name-only"], cwd=root)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to inspect diff:\n{result.stderr}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _render_prompt(
    *,
    program_path: pathlib.Path,
    benchmark_cmd: str,
    metric: str,
    baseline_result: dict[str, float],
    iteration: int,
    allowed_files: list[str],
) -> str:
    program_text = program_path.read_text(encoding="utf-8").strip()
    allowed_text = "\n".join(f"- {path}" for path in allowed_files)
    return (
        f"{program_text}\n\n"
        f"Current iteration: {iteration}\n"
        f"Benchmark command:\n{benchmark_cmd}\n\n"
        f"Metric to maximize: {metric}\n"
        f"Current best result:\n{json.dumps(baseline_result, indent=2, sort_keys=True)}\n\n"
        f"Editable files:\n{allowed_text}\n"
    )


def _run_codex_iteration(
    *,
    codex_bin: str,
    cwd: pathlib.Path,
    prompt: str,
    model: str,
    output_file: pathlib.Path,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        codex_bin,
        "exec",
        "--full-auto",
        "--sandbox",
        "workspace-write",
        "-C",
        str(cwd),
        "-o",
        str(output_file),
    ]
    if model:
        cmd.extend(["-m", model])
    cmd.append(prompt)
    return _run(cmd, cwd=cwd)


def _run_benchmark(root: pathlib.Path, benchmark_cmd: str) -> dict[str, float]:
    result = _run(["bash", "-lc", benchmark_cmd], cwd=root)
    if result.returncode != 0:
        raise RuntimeError(
            f"Benchmark command failed with code {result.returncode}:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Benchmark command produced no output.")

    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Benchmark output did not end in JSON.\nSTDOUT:\n{result.stdout}"
        ) from exc
    return payload


def _write_json(path: pathlib.Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    repo_root = REPO_ROOT
    allowed_files = args.allowed_file or list(DEFAULT_ALLOWED_FILES)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = args.results_dir / run_id
    results_dir.mkdir(parents=True, exist_ok=False)

    workspace = args.workspace
    metadata = {
        "repo_root": str(repo_root),
        "workspace": str(workspace),
        "benchmark_cmd": args.benchmark_cmd,
        "metric": args.metric,
        "allowed_files": allowed_files,
        "iterations": args.iterations,
        "model": args.model,
        "program": str(args.program),
    }
    _write_json(results_dir / "run_config.json", metadata)

    try:
        status_entries = _git_status_entries(repo_root)
        _prepare_workspace(repo_root, workspace)
        if status_entries:
            _sync_working_tree_changes(repo_root, workspace, status_entries)
        accepted_snapshot = _capture_files(workspace, allowed_files)

        best_result = _run_benchmark(workspace, args.benchmark_cmd)
        _write_json(results_dir / "baseline.json", best_result)

        history: list[dict[str, object]] = []
        for iteration in range(1, args.iterations + 1):
            prompt = _render_prompt(
                program_path=args.program,
                benchmark_cmd=args.benchmark_cmd,
                metric=args.metric,
                baseline_result=best_result,
                iteration=iteration,
                allowed_files=allowed_files,
            )
            prompt_path = results_dir / f"iteration_{iteration:02d}_prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")

            agent_summary_path = results_dir / f"iteration_{iteration:02d}_agent.txt"
            codex_result = _run_codex_iteration(
                codex_bin=args.codex_bin,
                cwd=workspace,
                prompt=prompt,
                model=args.model,
                output_file=agent_summary_path,
            )

            changed_files = _git_diff_names(workspace)
            unexpected = sorted(set(changed_files) - set(allowed_files))

            record: dict[str, object] = {
                "iteration": iteration,
                "codex_returncode": codex_result.returncode,
                "changed_files": changed_files,
                "unexpected_files": unexpected,
            }

            if codex_result.returncode != 0:
                record["status"] = "agent_failed"
                record["stderr"] = codex_result.stderr
                history.append(record)
                break

            if unexpected:
                record["status"] = "unexpected_files_changed"
                history.append(record)
                break

            try:
                benchmark_result = _run_benchmark(workspace, args.benchmark_cmd)
            except RuntimeError as exc:
                _restore_files(workspace, accepted_snapshot)
                record["status"] = "benchmark_failed"
                record["error"] = str(exc)
                history.append(record)
                continue

            record["benchmark"] = benchmark_result
            improvement = benchmark_result[args.metric] - best_result[args.metric]
            record["improvement"] = improvement

            if improvement > args.min_improvement:
                accepted_snapshot = _capture_files(workspace, allowed_files)
                best_result = benchmark_result
                record["status"] = "accepted"
                _write_json(results_dir / "best_result.json", best_result)
            else:
                _restore_files(workspace, accepted_snapshot)
                record["status"] = "rejected"

            history.append(record)
            _write_json(results_dir / "history.json", history)

        _write_json(results_dir / "history.json", history)
        _write_json(results_dir / "final_best.json", best_result)
        print(json.dumps(best_result, indent=2, sort_keys=True))
    finally:
        if not args.keep_workspace:
            try:
                _remove_workspace(repo_root, workspace)
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)


if __name__ == "__main__":
    main()
