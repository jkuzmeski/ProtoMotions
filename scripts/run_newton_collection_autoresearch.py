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
import time
from typing import Iterable


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PROGRAM = REPO_ROOT / "scripts" / "newton_collection_program.md"
DEFAULT_DASHBOARD_SCRIPT = REPO_ROOT / "scripts" / "newton_collection_dashboard.py"
DEFAULT_ALLOWED_FILES = (
    "protomotions/simulator/newton/simulator.py",
    "protomotions/simulator/newton/config.py",
)
HEARTBEAT_POLL_SECONDS = 10


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
        default="",
        help="Codex reasoning effort to request, for example low, medium, high, or xhigh.",
    )
    parser.add_argument(
        "--delegate-to-mini",
        action="store_true",
        help="Encourage the agent to delegate suitable side tasks to a smaller subagent.",
    )
    parser.add_argument(
        "--subagent-model",
        type=str,
        default="gpt-5.3-codex",
        help="Preferred subagent model name when delegation guidance is enabled.",
    )
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
        ["git", "-c", "core.hooksPath=/dev/null", "worktree", "add", "--detach", str(workspace), "HEAD"],
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
    delegate_to_mini: bool,
    subagent_model: str,
) -> str:
    program_text = program_path.read_text(encoding="utf-8").strip()
    allowed_text = "\n".join(f"- {path}" for path in allowed_files)
    delegation_text = ""
    if delegate_to_mini:
        delegation_text = (
            "\nDelegation guidance:\n"
            "- If there are bounded side tasks that can run in parallel, prefer delegating them.\n"
            "- Keep the main critical-path implementation local.\n"
            "- In your final note, include exactly one line starting with `Delegation summary:`.\n"
            "- If you delegated, list the subagent model and task briefly on that line.\n"
            "- If you did not delegate, write `Delegation summary: none`.\n"
        )
        if subagent_model:
            delegation_text += f"- Preferred subagent model: {subagent_model}\n"
    return (
        f"{program_text}\n\n"
        f"Current iteration: {iteration}\n"
        f"Benchmark command:\n{benchmark_cmd}\n\n"
        f"Metric to maximize: {metric}\n"
        f"Current best result:\n{json.dumps(baseline_result, indent=2, sort_keys=True)}\n\n"
        f"Editable files:\n{allowed_text}\n"
        f"{delegation_text}"
    )


def _artifact_relpath(results_dir: pathlib.Path, path: pathlib.Path) -> str:
    return str(path.relative_to(results_dir))


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _append_jsonl(path: pathlib.Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_heartbeat(
    results_dir: pathlib.Path,
    *,
    phase: str,
    status: str,
    iteration: int | None = None,
    detail: str | None = None,
    artifact: str | None = None,
    agent_output_bytes: int | None = None,
) -> None:
    payload: dict[str, object] = {
        "updated_at": _now_iso(),
        "phase": phase,
        "status": status,
    }
    if iteration is not None:
        payload["iteration"] = iteration
    if detail:
        payload["detail"] = detail
    if artifact:
        payload["artifact"] = artifact
    if agent_output_bytes is not None:
        payload["agent_output_bytes"] = agent_output_bytes
    _write_json(results_dir / "heartbeat.json", payload)


def _record_activity(
    results_dir: pathlib.Path,
    *,
    event: str,
    phase: str,
    status: str,
    iteration: int | None = None,
    detail: str | None = None,
    artifact: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "at": _now_iso(),
        "event": event,
        "phase": phase,
        "status": status,
    }
    if iteration is not None:
        payload["iteration"] = iteration
    if detail:
        payload["detail"] = detail
    if artifact:
        payload["artifact"] = artifact
    _append_jsonl(results_dir / "activity.jsonl", payload)


def _refresh_dashboard(repo_root: pathlib.Path, results_root: pathlib.Path) -> None:
    if not DEFAULT_DASHBOARD_SCRIPT.exists():
        return
    result = _run(
        [
            sys.executable,
            str(DEFAULT_DASHBOARD_SCRIPT),
            "--repo-root",
            str(repo_root),
            "--results-dir",
            str(results_root),
        ],
        cwd=repo_root,
    )
    if result.returncode != 0:
        print(
            "Warning: dashboard refresh failed:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
            file=sys.stderr,
        )


def _update_dashboard_progress(
    repo_root: pathlib.Path,
    results_root: pathlib.Path,
    results_dir: pathlib.Path,
    *,
    event: str | None = None,
    phase: str,
    status: str,
    iteration: int | None = None,
    detail: str | None = None,
    artifact: str | None = None,
    agent_output_bytes: int | None = None,
) -> None:
    _write_heartbeat(
        results_dir,
        phase=phase,
        status=status,
        iteration=iteration,
        detail=detail,
        artifact=artifact,
        agent_output_bytes=agent_output_bytes,
    )
    if event:
        _record_activity(
            results_dir,
            event=event,
            phase=phase,
            status=status,
            iteration=iteration,
            detail=detail,
            artifact=artifact,
        )
    _refresh_dashboard(repo_root, results_root)


def _run_with_heartbeat(
    cmd: list[str],
    *,
    cwd: pathlib.Path,
    repo_root: pathlib.Path,
    results_root: pathlib.Path,
    results_dir: pathlib.Path,
    phase: str,
    detail: str,
    iteration: int | None = None,
    artifact_path: pathlib.Path | None = None,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    while True:
        try:
            stdout, stderr = process.communicate(timeout=HEARTBEAT_POLL_SECONDS)
            return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            artifact_relpath = None
            artifact_bytes = None
            heartbeat_detail = detail
            if artifact_path is not None and artifact_path.exists():
                artifact_bytes = artifact_path.stat().st_size
                artifact_relpath = _artifact_relpath(results_dir, artifact_path)
                heartbeat_detail = f"{detail} | {artifact_path.name} {artifact_bytes} B"
            _update_dashboard_progress(
                repo_root,
                results_root,
                results_dir,
                phase=phase,
                status="running",
                iteration=iteration,
                detail=heartbeat_detail,
                artifact=artifact_relpath,
                agent_output_bytes=artifact_bytes,
            )


def _run_codex_iteration(
    *,
    codex_bin: str,
    cwd: pathlib.Path,
    repo_root: pathlib.Path,
    results_root: pathlib.Path,
    results_dir: pathlib.Path,
    prompt: str,
    model: str,
    reasoning_effort: str,
    iteration: int,
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
    if reasoning_effort:
        cmd.extend(["-c", f"model_reasoning_effort={reasoning_effort}"])
    cmd.append(prompt)
    return _run_with_heartbeat(
        cmd,
        cwd=cwd,
        repo_root=repo_root,
        results_root=results_root,
        results_dir=results_dir,
        phase="codex_running",
        detail="Codex iteration is still running",
        iteration=iteration,
        artifact_path=output_file,
    )


def _run_benchmark(
    root: pathlib.Path,
    benchmark_cmd: str,
    *,
    repo_root: pathlib.Path,
    results_root: pathlib.Path,
    results_dir: pathlib.Path,
    phase: str,
    detail: str,
    iteration: int | None = None,
) -> dict[str, float]:
    result = _run_with_heartbeat(
        ["bash", "-lc", benchmark_cmd],
        cwd=root,
        repo_root=repo_root,
        results_root=results_root,
        results_dir=results_dir,
        phase=phase,
        detail=detail,
        iteration=iteration,
    )
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
        "reasoning_effort": args.reasoning_effort,
        "delegate_to_mini": args.delegate_to_mini,
        "subagent_model": args.subagent_model,
        "program": str(args.program),
    }
    _write_json(results_dir / "run_config.json", metadata)
    _update_dashboard_progress(
        repo_root,
        args.results_dir,
        results_dir,
        event="run_started",
        phase="initializing",
        status="running",
        detail="Run directory created",
    )

    try:
        accepted_any = False
        status_entries = _git_status_entries(repo_root)
        _prepare_workspace(repo_root, workspace)
        _update_dashboard_progress(
            repo_root,
            args.results_dir,
            results_dir,
            event="workspace_prepared",
            phase="preparing_workspace",
            status="running",
            detail="Temporary worktree prepared",
        )
        if status_entries:
            _sync_working_tree_changes(repo_root, workspace, status_entries)
            _update_dashboard_progress(
                repo_root,
                args.results_dir,
                results_dir,
                event="workspace_synced",
                phase="preparing_workspace",
                status="running",
                detail=f"Synchronized {len(status_entries)} working tree changes",
            )
        baseline_dirty_files = _git_diff_names(workspace)
        accepted_snapshot = _capture_files(workspace, allowed_files)

        _update_dashboard_progress(
            repo_root,
            args.results_dir,
            results_dir,
            event="baseline_benchmark_started",
            phase="baseline_benchmark",
            status="running",
            detail="Measuring baseline benchmark",
        )
        best_result = _run_benchmark(
            workspace,
            args.benchmark_cmd,
            repo_root=repo_root,
            results_root=args.results_dir,
            results_dir=results_dir,
            phase="baseline_benchmark",
            detail="Baseline benchmark still running",
        )
        _write_json(results_dir / "baseline.json", best_result)
        _update_dashboard_progress(
            repo_root,
            args.results_dir,
            results_dir,
            event="baseline_benchmark_completed",
            phase="baseline_benchmark",
            status="running",
            detail=f"Baseline {args.metric}={best_result.get(args.metric, 'n/a')}",
        )

        history: list[dict[str, object]] = []
        for iteration in range(1, args.iterations + 1):
            prompt = _render_prompt(
                program_path=args.program,
                benchmark_cmd=args.benchmark_cmd,
                metric=args.metric,
                baseline_result=best_result,
                iteration=iteration,
                allowed_files=allowed_files,
                delegate_to_mini=args.delegate_to_mini,
                subagent_model=args.subagent_model,
            )
            prompt_path = results_dir / f"iteration_{iteration:02d}_prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            _update_dashboard_progress(
                repo_root,
                args.results_dir,
                results_dir,
                event="prompt_written",
                phase="iteration_prompt",
                status="running",
                iteration=iteration,
                detail=f"Prompt written for iteration {iteration:02d}",
                artifact=_artifact_relpath(results_dir, prompt_path),
            )

            agent_summary_path = results_dir / f"iteration_{iteration:02d}_agent.txt"
            _update_dashboard_progress(
                repo_root,
                args.results_dir,
                results_dir,
                event="codex_started",
                phase="codex_running",
                status="running",
                iteration=iteration,
                detail=f"Codex iteration {iteration:02d} started",
                artifact=_artifact_relpath(results_dir, agent_summary_path),
            )
            codex_result = _run_codex_iteration(
                codex_bin=args.codex_bin,
                cwd=workspace,
                repo_root=repo_root,
                results_root=args.results_dir,
                results_dir=results_dir,
                prompt=prompt,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                iteration=iteration,
                output_file=agent_summary_path,
            )
            _update_dashboard_progress(
                repo_root,
                args.results_dir,
                results_dir,
                event="codex_finished",
                phase="codex_completed",
                status="running",
                iteration=iteration,
                detail=f"Codex exited with code {codex_result.returncode}",
                artifact=_artifact_relpath(results_dir, agent_summary_path),
                agent_output_bytes=agent_summary_path.stat().st_size if agent_summary_path.exists() else None,
            )

            changed_files = _git_diff_names(workspace)
            iteration_changed_files = sorted(set(changed_files) - set(baseline_dirty_files))
            unexpected = sorted(set(iteration_changed_files) - set(allowed_files))

            record: dict[str, object] = {
                "iteration": iteration,
                "codex_returncode": codex_result.returncode,
                "changed_files": iteration_changed_files,
                "unexpected_files": unexpected,
                "agent_file": _artifact_relpath(results_dir, agent_summary_path),
            }

            if codex_result.returncode != 0:
                record["status"] = "agent_failed"
                record["stderr"] = codex_result.stderr
                history.append(record)
                _write_json(results_dir / "history.json", history)
                _update_dashboard_progress(
                    repo_root,
                    args.results_dir,
                    results_dir,
                    event="iteration_failed",
                    phase="codex_completed",
                    status="failed",
                    iteration=iteration,
                    detail="Codex returned a non-zero exit code",
                    artifact=_artifact_relpath(results_dir, agent_summary_path),
                )
                break

            if unexpected:
                record["status"] = "unexpected_files_changed"
                history.append(record)
                _write_json(results_dir / "history.json", history)
                _update_dashboard_progress(
                    repo_root,
                    args.results_dir,
                    results_dir,
                    event="iteration_failed",
                    phase="validating_changes",
                    status="failed",
                    iteration=iteration,
                    detail=f"Unexpected files changed: {', '.join(unexpected)}",
                )
                break

            _update_dashboard_progress(
                repo_root,
                args.results_dir,
                results_dir,
                event="benchmark_started",
                phase="iteration_benchmark",
                status="running",
                iteration=iteration,
                detail=f"Benchmarking iteration {iteration:02d}",
            )
            try:
                benchmark_result = _run_benchmark(
                    workspace,
                    args.benchmark_cmd,
                    repo_root=repo_root,
                    results_root=args.results_dir,
                    results_dir=results_dir,
                    phase="iteration_benchmark",
                    detail=f"Benchmark still running for iteration {iteration:02d}",
                    iteration=iteration,
                )
            except RuntimeError as exc:
                _restore_files(workspace, accepted_snapshot)
                record["status"] = "benchmark_failed"
                record["error"] = str(exc)
                history.append(record)
                _write_json(results_dir / "history.json", history)
                _update_dashboard_progress(
                    repo_root,
                    args.results_dir,
                    results_dir,
                    event="iteration_failed",
                    phase="iteration_benchmark",
                    status="failed",
                    iteration=iteration,
                    detail="Benchmark failed",
                )
                continue

            record["benchmark"] = benchmark_result
            improvement = benchmark_result[args.metric] - best_result[args.metric]
            record["improvement"] = improvement

            if improvement > args.min_improvement:
                accepted_snapshot = _capture_files(workspace, allowed_files)
                best_result = benchmark_result
                record["status"] = "accepted"
                _write_json(results_dir / "best_result.json", best_result)
                accepted_any = True
                baseline_dirty_files = _git_diff_names(workspace)
                event_name = "iteration_accepted"
                event_detail = (
                    f"Accepted iteration {iteration:02d} with delta {improvement:+.2f} "
                    f"{args.metric}"
                )
            else:
                _restore_files(workspace, accepted_snapshot)
                record["status"] = "rejected"
                baseline_dirty_files = _git_diff_names(workspace)
                event_name = "iteration_rejected"
                event_detail = (
                    f"Rejected iteration {iteration:02d} with delta {improvement:+.2f} "
                    f"{args.metric}"
                )

            history.append(record)
            _write_json(results_dir / "history.json", history)
            _update_dashboard_progress(
                repo_root,
                args.results_dir,
                results_dir,
                event=event_name,
                phase="iteration_complete",
                status="running",
                iteration=iteration,
                detail=event_detail,
                artifact=_artifact_relpath(results_dir, agent_summary_path),
            )

        _write_json(results_dir / "history.json", history)
        _write_json(results_dir / "final_best.json", best_result)
        if accepted_any:
            _restore_files(repo_root, accepted_snapshot)
            _update_dashboard_progress(
                repo_root,
                args.results_dir,
                results_dir,
                event="best_applied_to_repo",
                phase="run_complete",
                status="completed",
                detail="Applied best accepted files back to the main worktree",
            )
        final_status = "completed"
        final_detail = "Run completed successfully"
        if history:
            latest_status = str(history[-1].get("status", "completed"))
            if latest_status in {"agent_failed", "benchmark_failed", "unexpected_files_changed"}:
                final_status = "failed"
                final_detail = f"Run stopped after {latest_status}"
        _update_dashboard_progress(
            repo_root,
            args.results_dir,
            results_dir,
            event="run_finished",
            phase="run_complete",
            status=final_status,
            detail=final_detail,
        )
        print(json.dumps(best_result, indent=2, sort_keys=True))
    finally:
        if not args.keep_workspace:
            try:
                _remove_workspace(repo_root, workspace)
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)


if __name__ == "__main__":
    main()
