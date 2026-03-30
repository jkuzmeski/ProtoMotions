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
from typing import Any, Iterable

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from protomotions.utils.newton_collection_dashboard import write_dashboard

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
        "--reasoning-effort",
        type=str,
        default="high",
        choices=("low", "medium", "high", "xhigh"),
        help="Reasoning effort for the top-level Codex agent.",
    )
    parser.add_argument(
        "--delegate-to-mini",
        action="store_true",
        help=(
            "Instruct Codex to keep planning/judgment on the top-level model and "
            "delegate bounded subtasks to mini subagents when available."
        ),
    )
    parser.add_argument(
        "--subagent-model",
        type=str,
        default="gpt-5.4-mini",
        help="Preferred mini model name to mention in delegation guidance.",
    )
    parser.add_argument(
        "--codex-config",
        action="append",
        default=[],
        help="Extra codex exec config overrides in key=value form. Repeat to pass multiple.",
    )
    parser.add_argument("--codex-bin", type=str, default="codex")
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
        "--branch-name",
        type=str,
        default="autoresearch",
        help="Local git branch used to track autoresearch work.",
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


def _write_text(path: pathlib.Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


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


def _git_current_branch(repo_root: pathlib.Path) -> str:
    result = _run(["git", "branch", "--show-current"], cwd=repo_root)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to query current branch:\n{result.stderr}")
    branch = result.stdout.strip()
    return branch if branch else "DETACHED"


def _git_head_commit(repo_root: pathlib.Path, ref: str = "HEAD") -> str:
    result = _run(["git", "rev-parse", ref], cwd=repo_root)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to query commit for {ref}:\n{result.stderr}")
    return result.stdout.strip()


def _ensure_branch_exists(repo_root: pathlib.Path, branch_name: str) -> None:
    existing = _run(["git", "branch", "--list", branch_name], cwd=repo_root)
    if existing.returncode != 0:
        raise RuntimeError(f"Failed to query branch {branch_name}:\n{existing.stderr}")
    if existing.stdout.strip():
        return

    create = _run(["git", "branch", branch_name, "HEAD"], cwd=repo_root)
    if create.returncode != 0:
        raise RuntimeError(f"Failed to create branch {branch_name}:\n{create.stderr}")


def _prepare_workspace(repo_root: pathlib.Path, workspace: pathlib.Path, branch_name: str) -> None:
    if workspace.exists():
        raise RuntimeError(
            f"Workspace already exists: {workspace}. Remove it or choose a new path."
        )

    worktree = _run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "worktree",
            "add",
            str(workspace),
            branch_name,
        ],
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
    cleanup = _run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "worktree",
            "remove",
            "--force",
            str(workspace),
        ],
        cwd=repo_root,
    )
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


def _git_diff_stat(root: pathlib.Path) -> str:
    result = _run(["git", "diff", "--stat", "--"], cwd=root)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to inspect diff stat:\n{result.stderr}")
    return result.stdout


def _git_diff_patch(root: pathlib.Path) -> str:
    result = _run(["git", "diff", "--", "."], cwd=root)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to inspect diff patch:\n{result.stderr}")
    return result.stdout


def _git_status_short(root: pathlib.Path) -> str:
    result = _run(["git", "status", "--short"], cwd=root)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to inspect git status:\n{result.stderr}")
    return result.stdout


def _git_has_changes(root: pathlib.Path, paths: Iterable[str]) -> bool:
    cmd = ["git", "status", "--porcelain", "--", *paths]
    result = _run(cmd, cwd=root)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to inspect pending changes:\n{result.stderr}")
    return bool(result.stdout.strip())


def _git_commit_paths(root: pathlib.Path, paths: Iterable[str], message: str) -> str:
    add_result = _run(["git", "add", "--", *paths], cwd=root)
    if add_result.returncode != 0:
        raise RuntimeError(f"Failed to stage files for commit:\n{add_result.stderr}")

    commit_result = _run(
        ["git", "-c", "core.hooksPath=/dev/null", "commit", "-m", message],
        cwd=root,
    )
    if commit_result.returncode != 0:
        raise RuntimeError(f"Failed to create commit:\n{commit_result.stderr}")

    return _git_head_commit(root)


def _classify_iteration_changes(
    *,
    baseline_dirty_files: Iterable[str],
    current_dirty_files: Iterable[str],
    allowed_files: Iterable[str],
) -> tuple[list[str], list[str]]:
    baseline_set = set(baseline_dirty_files)
    current_set = set(current_dirty_files)
    allowed_set = set(allowed_files)
    net_new_changes = sorted(current_set - baseline_set)
    unexpected = sorted(path for path in net_new_changes if path not in allowed_set)
    return net_new_changes, unexpected


def _render_prompt(
    *,
    program_path: pathlib.Path,
    benchmark_cmd: str,
    metric: str,
    baseline_result: dict[str, float],
    iteration: int,
    allowed_files: list[str],
    branch_name: str,
    delegate_to_mini: bool,
    subagent_model: str,
) -> str:
    program_text = program_path.read_text(encoding="utf-8").strip()
    allowed_text = "\n".join(f"- {path}" for path in allowed_files)
    delegation_text = ""
    if delegate_to_mini:
        delegation_text = (
            "\nDelegation policy:\n"
            "- Keep planning, coordination, and final benchmark judgment on the top-level model.\n"
            f"- When subagents are available, delegate bounded implementation or search subtasks to `{subagent_model}`.\n"
            "- Use subagents only for concrete, narrow subtasks; keep the final edit selection and commit decision local.\n"
        )
    return (
        f"{program_text}\n\n"
        f"Work only on branch: {branch_name}\n"
        f"Current iteration: {iteration}\n"
        f"Benchmark command:\n{benchmark_cmd}\n\n"
        f"Metric to maximize: {metric}\n"
        f"Current best result:\n{json.dumps(baseline_result, indent=2, sort_keys=True)}\n\n"
        f"Editable files:\n{allowed_text}\n"
        f"{delegation_text}"
    )


def _build_codex_exec_cmd(
    *,
    codex_bin: str,
    cwd: pathlib.Path,
    prompt: str,
    model: str,
    output_file: pathlib.Path,
    reasoning_effort: str,
    codex_config: list[str],
) -> list[str]:
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
        "-c",
        f"model_reasoning_effort={json.dumps(reasoning_effort)}",
    ]
    for config_entry in codex_config:
        cmd.extend(["-c", config_entry])
    if model:
        cmd.extend(["-m", model])
    cmd.append(prompt)
    return cmd


def _run_codex_iteration(
    *,
    codex_bin: str,
    cwd: pathlib.Path,
    prompt: str,
    model: str,
    output_file: pathlib.Path,
    reasoning_effort: str,
    codex_config: list[str],
) -> subprocess.CompletedProcess[str]:
    cmd = _build_codex_exec_cmd(
        codex_bin=codex_bin,
        cwd=cwd,
        prompt=prompt,
        model=model,
        output_file=output_file,
        reasoning_effort=reasoning_effort,
        codex_config=codex_config,
    )
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


def _artifact_relpath(results_dir: pathlib.Path, path: pathlib.Path) -> str:
    return str(path.relative_to(results_dir))


def _write_iteration_report(
    *,
    results_dir: pathlib.Path,
    iteration: int,
    record: dict[str, Any],
    prompt_path: pathlib.Path,
    agent_summary_path: pathlib.Path,
    diff_stat_path: pathlib.Path,
    diff_patch_path: pathlib.Path,
    status_path: pathlib.Path,
) -> pathlib.Path:
    report_path = results_dir / f"iteration_{iteration:02d}_report.md"
    changed_files = record.get("changed_files", [])
    changed_text = "\n".join(f"- `{path}`" for path in changed_files) or "- None"
    unexpected_files = record.get("unexpected_files", [])
    unexpected_text = "\n".join(f"- `{path}`" for path in unexpected_files) or "- None"

    lines = [
        f"# Iteration {iteration}",
        "",
        "## Outcome",
        f"- Status: `{record.get('status', 'unknown')}`",
        f"- Codex return code: `{record.get('codex_returncode', 'n/a')}`",
        f"- Improvement: `{record.get('improvement', 'n/a')}`",
        f"- Branch: `{record.get('branch_name', 'n/a')}`",
        f"- Commit: `{record.get('commit_sha', 'n/a')}`",
        "",
        "## Files",
        "Changed files:",
        changed_text,
        "",
        "Unexpected files:",
        unexpected_text,
        "",
        "## Benchmark",
    ]

    benchmark = record.get("benchmark")
    if isinstance(benchmark, dict):
        lines.extend([
            "```json",
            json.dumps(benchmark, indent=2, sort_keys=True),
            "```",
        ])
    else:
        lines.append("Benchmark did not complete.")
    lines.extend([
        "",
        "## Artifacts",
        f"- Prompt: `{_artifact_relpath(results_dir, prompt_path)}`",
        f"- Agent summary: `{_artifact_relpath(results_dir, agent_summary_path)}`",
        f"- Git status: `{_artifact_relpath(results_dir, status_path)}`",
        f"- Diff stat: `{_artifact_relpath(results_dir, diff_stat_path)}`",
        f"- Diff patch: `{_artifact_relpath(results_dir, diff_patch_path)}`",
        "",
    ])

    if record.get("error"):
        lines.extend([
            "## Error",
            "```text",
            str(record["error"]),
            "```",
            "",
        ])
    if record.get("stderr"):
        lines.extend([
            "## Agent STDERR",
            "```text",
            str(record["stderr"]),
            "```",
            "",
        ])

    _write_text(report_path, "\n".join(lines) + "\n")
    return report_path


def _write_run_summary(
    *,
    results_dir: pathlib.Path,
    history: list[dict[str, Any]],
    best_result: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    lines = [
        "# Newton Collection Autoresearch Summary",
        "",
        "## Run Metadata",
        "```json",
        json.dumps(metadata, indent=2, sort_keys=True),
        "```",
        "",
        "## Final Best Result",
        "```json",
        json.dumps(best_result, indent=2, sort_keys=True),
        "```",
        "",
        "## Iterations",
    ]

    if not history:
        lines.append("- No iterations were completed.")
    else:
        for record in history:
            iteration = int(record["iteration"])
            report_file = record.get("report_file", f"iteration_{iteration:02d}_report.md")
            lines.append(
                f"- Iteration {iteration}: `{record.get('status', 'unknown')}` | "
                f"improvement={record.get('improvement', 'n/a')} | "
                f"commit={record.get('commit_sha', 'n/a')} | "
                f"report=`{report_file}`"
            )

    _write_text(results_dir / "summary.md", "\n".join(lines) + "\n")


def _refresh_dashboard(repo_root: pathlib.Path, results_root: pathlib.Path) -> None:
    try:
        write_dashboard(repo_root=repo_root, results_root=results_root)
    except Exception as exc:  # pragma: no cover - dashboard refresh should not abort a run
        print(f"Failed to refresh dashboard: {exc}", file=sys.stderr)


def main() -> None:
    args = _parse_args()
    repo_root = REPO_ROOT
    allowed_files = args.allowed_file or list(DEFAULT_ALLOWED_FILES)
    selected_model = args.model or ("gpt-5.4" if args.delegate_to_mini else "")
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = args.results_dir / run_id
    results_dir.mkdir(parents=True, exist_ok=False)

    workspace = args.workspace
    base_branch = _git_current_branch(repo_root)
    base_commit = _git_head_commit(repo_root)
    metadata = {
        "repo_root": str(repo_root),
        "workspace": str(workspace),
        "benchmark_cmd": args.benchmark_cmd,
        "metric": args.metric,
        "allowed_files": allowed_files,
        "iterations": args.iterations,
        "model": selected_model,
        "reasoning_effort": args.reasoning_effort,
        "delegate_to_mini": args.delegate_to_mini,
        "subagent_model": args.subagent_model,
        "codex_config": args.codex_config,
        "program": str(args.program),
        "branch_name": args.branch_name,
        "base_branch": base_branch,
        "base_commit": base_commit,
    }
    _write_json(results_dir / "run_config.json", metadata)
    _refresh_dashboard(repo_root, args.results_dir)

    history: list[dict[str, Any]] = []
    best_result: dict[str, Any] = {}
    try:
        status_entries = _git_status_entries(repo_root)
        _ensure_branch_exists(repo_root, args.branch_name)
        metadata["branch_commit_before_run"] = _git_head_commit(repo_root, args.branch_name)
        _write_json(results_dir / "run_config.json", metadata)
        _refresh_dashboard(repo_root, args.results_dir)

        _prepare_workspace(repo_root, workspace, args.branch_name)
        if status_entries:
            _sync_working_tree_changes(repo_root, workspace, status_entries)
            if _git_has_changes(workspace, allowed_files):
                baseline_commit = _git_commit_paths(
                    workspace,
                    allowed_files,
                    "autoresearch: sync current working tree",
                )
                metadata["baseline_sync_commit"] = baseline_commit
                _write_json(results_dir / "run_config.json", metadata)
                _refresh_dashboard(repo_root, args.results_dir)
        accepted_snapshot = _capture_files(workspace, allowed_files)
        baseline_dirty_files = _git_diff_names(workspace)

        best_result = _run_benchmark(workspace, args.benchmark_cmd)
        _write_json(results_dir / "baseline.json", best_result)
        _refresh_dashboard(repo_root, args.results_dir)

        for iteration in range(1, args.iterations + 1):
            prompt = _render_prompt(
                program_path=args.program,
                benchmark_cmd=args.benchmark_cmd,
                metric=args.metric,
                baseline_result=best_result,
                iteration=iteration,
                allowed_files=allowed_files,
                branch_name=args.branch_name,
                delegate_to_mini=args.delegate_to_mini,
                subagent_model=args.subagent_model,
            )
            prompt_path = results_dir / f"iteration_{iteration:02d}_prompt.txt"
            _write_text(prompt_path, prompt)

            agent_summary_path = results_dir / f"iteration_{iteration:02d}_agent.txt"
            codex_result = _run_codex_iteration(
                codex_bin=args.codex_bin,
                cwd=workspace,
                prompt=prompt,
                model=selected_model,
                output_file=agent_summary_path,
                reasoning_effort=args.reasoning_effort,
                codex_config=args.codex_config,
            )

            workspace_dirty_files = _git_diff_names(workspace)
            changed_files, unexpected = _classify_iteration_changes(
                baseline_dirty_files=baseline_dirty_files,
                current_dirty_files=workspace_dirty_files,
                allowed_files=allowed_files,
            )
            status_text = _git_status_short(workspace)
            diff_stat_text = _git_diff_stat(workspace)
            diff_patch_text = _git_diff_patch(workspace)

            status_path = results_dir / f"iteration_{iteration:02d}_status.txt"
            diff_stat_path = results_dir / f"iteration_{iteration:02d}_diff_stat.txt"
            diff_patch_path = results_dir / f"iteration_{iteration:02d}_diff.patch"
            _write_text(status_path, status_text)
            _write_text(diff_stat_path, diff_stat_text)
            _write_text(diff_patch_path, diff_patch_text)

            record: dict[str, Any] = {
                "iteration": iteration,
                "branch_name": args.branch_name,
                "codex_returncode": codex_result.returncode,
                "changed_files": changed_files,
                "unexpected_files": unexpected,
                "status_file": _artifact_relpath(results_dir, status_path),
                "diff_stat_file": _artifact_relpath(results_dir, diff_stat_path),
                "diff_patch_file": _artifact_relpath(results_dir, diff_patch_path),
                "prompt_file": _artifact_relpath(results_dir, prompt_path),
                "agent_file": _artifact_relpath(results_dir, agent_summary_path),
            }

            if codex_result.returncode != 0:
                record["status"] = "agent_failed"
                record["stderr"] = codex_result.stderr
            elif unexpected:
                record["status"] = "unexpected_files_changed"
            else:
                try:
                    benchmark_result = _run_benchmark(workspace, args.benchmark_cmd)
                except RuntimeError as exc:
                    _restore_files(workspace, accepted_snapshot)
                    record["status"] = "benchmark_failed"
                    record["error"] = str(exc)
                else:
                    record["benchmark"] = benchmark_result
                    improvement = benchmark_result[args.metric] - best_result[args.metric]
                    record["improvement"] = improvement

                    if improvement > args.min_improvement:
                        commit_sha = _git_commit_paths(
                            workspace,
                            allowed_files,
                            f"autoresearch: iteration {iteration:02d} accepted",
                        )
                        record["commit_sha"] = commit_sha
                        best_result = benchmark_result
                        record["status"] = "accepted"
                        accepted_snapshot = _capture_files(workspace, allowed_files)
                        _write_json(results_dir / "best_result.json", best_result)
                        _write_text(results_dir / "best_diff_stat.txt", diff_stat_text)
                        _write_text(results_dir / "best_diff.patch", diff_patch_text)
                    else:
                        _restore_files(workspace, accepted_snapshot)
                        record["status"] = "rejected"

            if "improvement" not in record:
                record["improvement"] = "n/a"
            if "commit_sha" not in record:
                record["commit_sha"] = "n/a"

            report_path = _write_iteration_report(
                results_dir=results_dir,
                iteration=iteration,
                record=record,
                prompt_path=prompt_path,
                agent_summary_path=agent_summary_path,
                diff_stat_path=diff_stat_path,
                diff_patch_path=diff_patch_path,
                status_path=status_path,
            )
            record["report_file"] = _artifact_relpath(results_dir, report_path)

            history.append(record)
            _write_json(results_dir / "history.json", history)
            _write_run_summary(
                results_dir=results_dir,
                history=history,
                best_result=best_result,
                metadata=metadata,
            )
            _refresh_dashboard(repo_root, args.results_dir)

            if record["status"] in {"agent_failed", "unexpected_files_changed"}:
                break

        metadata["branch_commit_after_run"] = _git_head_commit(repo_root, args.branch_name)
        _write_json(results_dir / "run_config.json", metadata)
        _write_json(results_dir / "history.json", history)
        _write_json(results_dir / "final_best.json", best_result)
        _write_run_summary(
            results_dir=results_dir,
            history=history,
            best_result=best_result,
            metadata=metadata,
        )
        _refresh_dashboard(repo_root, args.results_dir)
        print(json.dumps(best_result, indent=2, sort_keys=True))
    finally:
        if not args.keep_workspace:
            try:
                _remove_workspace(repo_root, workspace)
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)


if __name__ == "__main__":
    main()
