#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Static dashboard generation for Newton collection autoresearch runs."""

from __future__ import annotations

import datetime as dt
import html
import json
import pathlib
import subprocess
from typing import Any


REFRESH_SECONDS = 15
RECENT_ACTIVITY_LIMIT = 10
MASTER_CHANGE_LOG_JSONL = "master_change_log.jsonl"
MASTER_CHANGE_LOG_MD = "master_change_log.md"


def _run_git(repo_root: pathlib.Path, args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr}")
    return result.stdout.strip()


def _read_json(path: pathlib.Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _read_text(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float(value: Any, digits: int = 2) -> str:
    numeric = _coerce_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:,.{digits}f}"


def _format_signed_float(value: Any, digits: int = 2) -> str:
    numeric = _coerce_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:+,.{digits}f}"


def _format_commit(sha: str | None) -> str:
    if not sha or sha == "n/a":
        return "n/a"
    return sha[:12]


def _format_run_timestamp(run_id: str) -> str:
    try:
        parsed = dt.datetime.strptime(run_id, "%Y%m%d_%H%M%S")
    except ValueError:
        return run_id
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _parse_timestamp(value: Any) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def _format_event_timestamp(value: Any) -> str:
    parsed = _parse_timestamp(value)
    if parsed is None:
        return "n/a"
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _extract_delegation_summary(text: str | None) -> str | None:
    if not text:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("delegation summary:"):
            return stripped.split(":", 1)[1].strip() or "none"
    return None


def _format_age(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m ago"


def _relative_to(root: pathlib.Path, path: pathlib.Path) -> str:
    return str(path.relative_to(root))


def _branch_exists(repo_root: pathlib.Path, branch_name: str) -> bool:
    result = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"],
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )
    return result.returncode == 0


def _collect_branch(repo_root: pathlib.Path, branch_name: str) -> dict[str, Any]:
    branch_data: dict[str, Any] = {"name": branch_name, "exists": _branch_exists(repo_root, branch_name)}
    if not branch_data["exists"]:
        return branch_data

    branch_data["commit"] = _run_git(repo_root, ["rev-parse", branch_name])
    branch_data["subject"] = _run_git(repo_root, ["show", "-s", "--format=%s", branch_name])
    branch_data["authored_at"] = _run_git(repo_root, ["show", "-s", "--format=%cI", branch_name])
    return branch_data


def _collect_git_state(repo_root: pathlib.Path, tracked_branch: str, base_branch: str) -> dict[str, Any]:
    status_output = _run_git(repo_root, ["status", "--short", "--branch"])
    status_lines = [line for line in status_output.splitlines() if line.strip()]
    branch_state = {
        "current_branch": _run_git(repo_root, ["branch", "--show-current"]) or "DETACHED",
        "head_commit": _run_git(repo_root, ["rev-parse", "HEAD"]),
        "head_subject": _run_git(repo_root, ["show", "-s", "--format=%s", "HEAD"]),
        "status": status_output,
        "status_lines": status_lines[1:] if status_lines else [],
        "dirty": len(status_lines) > 1,
        "tracked_branch": _collect_branch(repo_root, tracked_branch),
        "base_branch": _collect_branch(repo_root, base_branch),
    }

    if branch_state["tracked_branch"].get("exists") and branch_state["base_branch"].get("exists"):
        counts = _run_git(repo_root, ["rev-list", "--left-right", "--count", f"{base_branch}...{tracked_branch}"])
        behind, ahead = [int(part) for part in counts.split()]
        branch_state["divergence"] = {
            "ahead": ahead,
            "behind": behind,
            "merge_base": _run_git(repo_root, ["merge-base", base_branch, tracked_branch]),
        }
    else:
        branch_state["divergence"] = None

    return branch_state


def _heartbeat_freshness(age_seconds: float | None, finalized: bool) -> str:
    if finalized:
        return "muted"
    if age_seconds is None:
        return "warn"
    if age_seconds <= 180:
        return "ok"
    if age_seconds <= 900:
        return "warn"
    return "bad"


def _collect_run(run_dir: pathlib.Path, results_root: pathlib.Path, now: dt.datetime) -> dict[str, Any]:
    config = _read_json(run_dir / "run_config.json") or {}
    history = _read_json(run_dir / "history.json") or []
    for record in history:
        agent_file = record.get("agent_file")
        if isinstance(agent_file, str):
            record["delegation_summary"] = _extract_delegation_summary(_read_text(run_dir / agent_file))
    baseline = _read_json(run_dir / "baseline.json") or {}
    best_result = (
        _read_json(run_dir / "final_best.json")
        or _read_json(run_dir / "best_result.json")
        or baseline
        or {}
    )
    heartbeat = _read_json(run_dir / "heartbeat.json") or {}
    activity = _read_jsonl(run_dir / "activity.jsonl")
    metric = config.get("metric", "samples_per_s")
    baseline_metric = _coerce_float(baseline.get(metric))
    best_metric = _coerce_float(best_result.get(metric))
    improvement = None
    if baseline_metric is not None and best_metric is not None:
        improvement = best_metric - baseline_metric

    latest_record = history[-1] if history else None
    finalized = (run_dir / "final_best.json").exists()
    heartbeat_updated_at = _parse_timestamp(heartbeat.get("updated_at"))
    heartbeat_age_seconds = None
    if heartbeat_updated_at is not None:
        heartbeat_age_seconds = max((now - heartbeat_updated_at.astimezone(now.tzinfo)).total_seconds(), 0.0)

    return {
        "run_id": run_dir.name,
        "started_at": _format_run_timestamp(run_dir.name),
        "dir_name": run_dir.name,
        "dir_link": _relative_to(results_root, run_dir),
        "config": config,
        "history": history,
        "history_count": len(history),
        "latest_status": latest_record.get("status", "pending") if latest_record else "pending",
        "latest_record": latest_record,
        "baseline": baseline,
        "best_result": best_result,
        "metric": metric,
        "baseline_metric": baseline_metric,
        "best_metric": best_metric,
        "improvement": improvement,
        "finalized": finalized,
        "heartbeat": heartbeat,
        "heartbeat_age_seconds": heartbeat_age_seconds,
        "heartbeat_age_text": _format_age(heartbeat_age_seconds),
        "heartbeat_freshness": _heartbeat_freshness(heartbeat_age_seconds, finalized),
        "activity": activity,
        "recent_activity": list(reversed(activity[-RECENT_ACTIVITY_LIMIT:])),
        "summary_link": (
            _relative_to(results_root, run_dir / "summary.md") if (run_dir / "summary.md").exists() else None
        ),
        "history_link": (
            _relative_to(results_root, run_dir / "history.json") if (run_dir / "history.json").exists() else None
        ),
        "config_link": _relative_to(results_root, run_dir / "run_config.json"),
        "heartbeat_link": (
            _relative_to(results_root, run_dir / "heartbeat.json") if (run_dir / "heartbeat.json").exists() else None
        ),
        "activity_link": (
            _relative_to(results_root, run_dir / "activity.jsonl") if (run_dir / "activity.jsonl").exists() else None
        ),
    }


def collect_dashboard_data(repo_root: pathlib.Path, results_root: pathlib.Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    results_root = results_root.resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).astimezone()

    run_dirs = sorted(
        [path for path in results_root.iterdir() if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )
    runs = [_collect_run(run_dir, results_root, now) for run_dir in run_dirs if (run_dir / "run_config.json").exists()]
    latest_run = runs[0] if runs else None
    tracked_branch = "autoresearch"
    base_branch = "main"
    if latest_run:
        tracked_branch = latest_run["config"].get("branch_name", tracked_branch)
        base_branch = latest_run["config"].get("base_branch", base_branch)

    master_log_records = _read_jsonl(results_root / MASTER_CHANGE_LOG_JSONL)
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "results_root": str(results_root),
        "tracked_branch": tracked_branch,
        "base_branch": base_branch,
        "git": _collect_git_state(repo_root, tracked_branch=tracked_branch, base_branch=base_branch),
        "runs": runs,
        "latest_run_id": latest_run["run_id"] if latest_run else None,
        "master_change_log_link": MASTER_CHANGE_LOG_MD if (results_root / MASTER_CHANGE_LOG_MD).exists() else None,
        "master_change_log_jsonl_link": (
            MASTER_CHANGE_LOG_JSONL if (results_root / MASTER_CHANGE_LOG_JSONL).exists() else None
        ),
        "master_change_log_count": len(master_log_records),
    }


def _badge_class(status: str) -> str:
    if status == "accepted":
        return "ok"
    if status in {"rejected", "pending", "running"}:
        return "warn"
    if status in {"agent_failed", "benchmark_failed", "unexpected_files_changed", "failed"}:
        return "bad"
    return "muted"


def _render_metric_pair(label: str, value: str) -> str:
    return (
        '<div class="metric">'
        f"<span>{html.escape(label)}</span>"
        f"<strong>{html.escape(value)}</strong>"
        "</div>"
    )


def _render_links(run: dict[str, Any]) -> str:
    links: list[str] = []
    if run.get("summary_link"):
        links.append(f'<a href="{html.escape(run["summary_link"])}">summary</a>')
    if run.get("history_link"):
        links.append(f'<a href="{html.escape(run["history_link"])}">history</a>')
    if run.get("heartbeat_link"):
        links.append(f'<a href="{html.escape(run["heartbeat_link"])}">heartbeat</a>')
    if run.get("activity_link"):
        links.append(f'<a href="{html.escape(run["activity_link"])}">activity</a>')
    links.append(f'<a href="{html.escape(run["config_link"])}">config</a>')
    return " · ".join(links)


def _render_iteration(record: dict[str, Any], metric: str) -> str:
    benchmark = record.get("benchmark") or {}
    metric_value = _format_float(benchmark.get(metric))
    improvement = _format_float(record.get("improvement"))
    commit = _format_commit(record.get("commit_sha"))
    changed_files = record.get("changed_files") or []
    changed_text = ", ".join(changed_files) if changed_files else "none"
    artifact_links: list[str] = []
    for key, label in (("report_file", "report"), ("agent_file", "agent")):
        if record.get(key):
            artifact_links.append(f'<a href="{html.escape(str(record[key]))}">{label}</a>')
    artifacts_html = " · ".join(artifact_links) if artifact_links else "n/a"
    delegation_summary = str(record.get("delegation_summary", "unknown"))

    return (
        '<article class="iteration">'
        f'<div class="iteration-head"><h4>Iteration {int(record["iteration"]):02d}</h4>'
        f'<span class="badge {_badge_class(str(record.get("status", "unknown")))}">'
        f'{html.escape(str(record.get("status", "unknown")))}'
        "</span></div>"
        '<div class="iteration-grid">'
        f"{_render_metric_pair(metric, metric_value)}"
        f"{_render_metric_pair('improvement', improvement)}"
        f"{_render_metric_pair('commit', commit)}"
        "</div>"
        f'<p class="iteration-meta"><strong>Changed:</strong> {html.escape(changed_text)}</p>'
        f'<p class="iteration-meta"><strong>Delegation:</strong> {html.escape(delegation_summary)}</p>'
        f'<p class="iteration-meta"><strong>Artifacts:</strong> {artifacts_html}</p>'
        "</article>"
    )


def _chart_fill_for_status(status: str) -> str:
    if status == "accepted":
        return "var(--ok)"
    if status == "rejected":
        return "var(--accent-2)"
    if status in {"agent_failed", "benchmark_failed", "unexpected_files_changed"}:
        return "var(--bad)"
    return "var(--muted)"


def _render_iteration_chart(run: dict[str, Any]) -> str:
    metric = str(run.get("metric", "samples_per_s"))
    baseline_metric = _coerce_float(run.get("baseline_metric"))
    if baseline_metric is None:
        return '<p class="empty">Iteration chart is unavailable because the baseline metric is missing.</p>'

    chart_records: list[dict[str, Any]] = []
    for record in run.get("history", []):
        benchmark = record.get("benchmark") or {}
        benchmark_metric = _coerce_float(benchmark.get(metric))
        if benchmark_metric is None:
            continue
        chart_records.append(
            {
                "iteration": int(record["iteration"]),
                "status": str(record.get("status", "unknown")),
                "metric_value": benchmark_metric,
            }
        )

    if not chart_records:
        return '<p class="empty">No benchmarked iterations are available yet.</p>'

    width = 900.0
    height = 280.0
    left = 68.0
    right = 24.0
    top = 24.0
    bottom = 54.0
    plot_width = width - left - right
    plot_height = height - top - bottom

    metric_values = [baseline_metric, *[record["metric_value"] for record in chart_records]]
    min_metric = min(metric_values)
    max_metric = max(metric_values)
    padding = max((max_metric - min_metric) * 0.08, max(abs(max_metric), 1.0) * 0.002)
    chart_min = min_metric - padding
    chart_max = max_metric + padding
    if chart_min == chart_max:
        chart_max = chart_min + 1.0
    scale_range = chart_max - chart_min
    tick_values = [chart_min + scale_range * step / 4 for step in range(5)]

    if len(chart_records) == 0:
        point_gap = 0.0
        start_x = left
    elif len(chart_records) == 1:
        point_gap = 0.0
        start_x = left + plot_width / 2
    else:
        point_gap = plot_width / len(chart_records)
        start_x = left

    svg_parts = [
        (
            f'<svg class="iteration-chart" viewBox="0 0 {int(width)} {int(height)}" '
            'role="img" aria-label="Scatter plot showing samples per second for each iteration">'
        )
    ]

    for value in tick_values:
        y = top + plot_height - ((value - chart_min) / scale_range) * plot_height
        svg_parts.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{width - right:.1f}" y2="{y:.1f}" '
            'style="stroke: rgba(24, 34, 45, 0.12); stroke-width: 1;" />'
        )
        svg_parts.append(
            f'<text x="{left - 10:.1f}" y="{y + 4:.1f}" text-anchor="end" class="chart-axis">'
            f'{html.escape(_format_float(value))}'
            "</text>"
        )

    baseline_x = left
    baseline_y = top + plot_height - ((baseline_metric - chart_min) / scale_range) * plot_height
    points: list[dict[str, Any]] = [
        {"iteration": 0, "metric_value": baseline_metric, "status": "baseline", "x": baseline_x, "y": baseline_y}
    ]
    running_best = baseline_metric
    step_segments: list[str] = [f"{baseline_x:.1f},{baseline_y:.1f}"]
    for index, record in enumerate(chart_records):
        x = left + (index + 1) * point_gap if len(chart_records) > 1 else start_x
        y = top + plot_height - ((record["metric_value"] - chart_min) / scale_range) * plot_height
        point = {
            "iteration": record["iteration"],
            "metric_value": record["metric_value"],
            "status": record["status"],
            "x": x,
            "y": y,
        }
        points.append(point)
        step_segments.append(f"{x:.1f},{top + plot_height - ((running_best - chart_min) / scale_range) * plot_height:.1f}")
        if record["status"] == "accepted" and record["metric_value"] > running_best:
            running_best = record["metric_value"]
            step_segments.append(f"{x:.1f},{top + plot_height - ((running_best - chart_min) / scale_range) * plot_height:.1f}")

    if len(step_segments) >= 2:
        svg_parts.append(
            f'<polyline points="{" ".join(step_segments)}" '
            'style="fill: none; stroke: var(--ok); stroke-width: 2.5; stroke-linecap: round; stroke-linejoin: round; opacity: 0.9;" />'
        )

    for point in points:
        if point["status"] == "baseline":
            fill = "var(--ok)"
            stroke = "var(--ok)"
            opacity = "0.92"
        else:
            fill = _chart_fill_for_status(point["status"]) if point["status"] == "accepted" else "rgba(255, 253, 248, 0.92)"
            stroke = _chart_fill_for_status(point["status"])
            opacity = "0.98" if point["status"] == "accepted" else "0.55"
        point_label = "Baseline" if point["iteration"] == 0 else f"Iteration {point['iteration']:02d}"
        x_label = "B" if point["iteration"] == 0 else f"{point['iteration']:02d}"
        svg_parts.append(
            f'<circle cx="{point["x"]:.1f}" cy="{point["y"]:.1f}" r="6.5" '
            f'style="fill: {fill}; stroke: {stroke}; stroke-width: 2.5; opacity: {opacity};" >'
            f"<title>{point_label}: {_format_float(point['metric_value'])} {html.escape(metric)} ({html.escape(point['status'])})</title>"
            "</circle>"
        )
        svg_parts.append(
            f'<text x="{point["x"]:.1f}" y="{height - 14:.1f}" text-anchor="middle" class="chart-caption">'
            f"{x_label}"
            "</text>"
        )

    svg_parts.append(
        f'<text x="{left:.1f}" y="{top - 8:.1f}" text-anchor="start" class="chart-caption">'
        f"Y axis: {html.escape(metric)}"
        "</text>"
    )
    svg_parts.append(
        f'<text x="{width - right:.1f}" y="{top - 8:.1f}" text-anchor="end" class="chart-caption">'
        "Filled points and running-best line: kept"
        "</text>"
    )
    svg_parts.append(
        f'<text x="{width / 2:.1f}" y="{height - 32:.1f}" text-anchor="middle" class="chart-caption">'
        "Iteration"
        "</text>"
    )
    svg_parts.append("</svg>")

    latest_iteration = chart_records[-1]
    best_iteration = max(chart_records, key=lambda record: record["metric_value"])
    average_metric = sum(record["metric_value"] for record in chart_records) / len(chart_records)
    return (
        '<div class="run-chart">'
        '<p class="eyebrow">Samples Per Second By Iteration</p>'
        '<div class="metric-stack">'
        f"{_render_metric_pair('baseline', _format_float(baseline_metric))}"
        f"{_render_metric_pair('best', _format_float(best_iteration['metric_value']))}"
        f"{_render_metric_pair('latest', _format_float(latest_iteration['metric_value']))}"
        f"{_render_metric_pair('charted iterations', str(len(chart_records)))}"
        f"{_render_metric_pair('average', _format_float(average_metric))}"
        "</div>"
        '<p class="iteration-meta">Every iteration benchmark is plotted, including worse ones. The solid line tracks the running best.</p>'
        f'{"".join(svg_parts)}'
        "</div>"
    )


def _render_runs_chart(runs: list[dict[str, Any]]) -> str:
    chart_runs = [run for run in reversed(runs) if _coerce_float(run.get("best_metric")) is not None]
    if not chart_runs:
        return '<p class="empty">No completed runs are available yet.</p>'

    width = 900.0
    height = 280.0
    left = 68.0
    right = 24.0
    top = 24.0
    bottom = 54.0
    plot_width = width - left - right
    plot_height = height - top - bottom

    metric = str(chart_runs[-1].get("metric", "samples_per_s"))
    metric_values = [_coerce_float(run["best_metric"]) or 0.0 for run in chart_runs]
    min_metric = min(metric_values)
    max_metric = max(metric_values)
    padding = max((max_metric - min_metric) * 0.08, max(abs(max_metric), 1.0) * 0.002)
    chart_min = min_metric - padding
    chart_max = max_metric + padding
    if chart_min == chart_max:
        chart_max = chart_min + 1.0
    scale_range = chart_max - chart_min
    tick_values = [chart_min + scale_range * step / 4 for step in range(5)]

    point_gap = plot_width / max(len(chart_runs) - 1, 1)
    step_segments: list[str] = []
    points: list[dict[str, Any]] = []
    running_best = metric_values[0]
    for index, run in enumerate(chart_runs):
        x = left + index * point_gap if len(chart_runs) > 1 else left + plot_width / 2
        y = top + plot_height - ((metric_values[index] - chart_min) / scale_range) * plot_height
        points.append(
            {
                "index": index + 1,
                "run_id": str(run["run_id"]),
                "metric_value": metric_values[index],
                "status": str(run.get("latest_status", "unknown")),
                "x": x,
                "y": y,
            }
        )
        step_segments.append(
            f"{x:.1f},{top + plot_height - ((running_best - chart_min) / scale_range) * plot_height:.1f}"
        )
        if metric_values[index] > running_best:
            running_best = metric_values[index]
            step_segments.append(
                f"{x:.1f},{top + plot_height - ((running_best - chart_min) / scale_range) * plot_height:.1f}"
            )

    svg_parts = [
        (
            f'<svg class="iteration-chart" viewBox="0 0 {int(width)} {int(height)}" '
            'role="img" aria-label="Scatter plot showing best samples per second across runs">'
        )
    ]
    for value in tick_values:
        y = top + plot_height - ((value - chart_min) / scale_range) * plot_height
        svg_parts.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{width - right:.1f}" y2="{y:.1f}" '
            'style="stroke: rgba(24, 34, 45, 0.12); stroke-width: 1;" />'
        )
        svg_parts.append(
            f'<text x="{left - 10:.1f}" y="{y + 4:.1f}" text-anchor="end" class="chart-axis">'
            f"{html.escape(_format_float(value))}"
            "</text>"
        )

    if len(step_segments) >= 2:
        svg_parts.append(
            f'<polyline points="{" ".join(step_segments)}" '
            'style="fill: none; stroke: var(--ok); stroke-width: 2.5; stroke-linecap: round; stroke-linejoin: round; opacity: 0.9;" />'
        )

    for point in points:
        fill = _chart_fill_for_status(point["status"]) if point["status"] == "accepted" else "rgba(255, 253, 248, 0.92)"
        stroke = _chart_fill_for_status(point["status"])
        opacity = "0.98" if point["status"] == "accepted" else "0.55"
        svg_parts.append(
            f'<circle cx="{point["x"]:.1f}" cy="{point["y"]:.1f}" r="6.5" '
            f'style="fill: {fill}; stroke: {stroke}; stroke-width: 2.5; opacity: {opacity};" >'
            f"<title>Run {html.escape(point['run_id'])}: {_format_float(point['metric_value'])} {html.escape(metric)} ({html.escape(point['status'])})</title>"
            "</circle>"
        )
        svg_parts.append(
            f'<text x="{point["x"]:.1f}" y="{height - 14:.1f}" text-anchor="middle" class="chart-caption">'
            f"{point['index']}"
            "</text>"
        )

    svg_parts.append(
        f'<text x="{left:.1f}" y="{top - 8:.1f}" text-anchor="start" class="chart-caption">'
        f"Y axis: {html.escape(metric)}"
        "</text>"
    )
    svg_parts.append(
        f'<text x="{width - right:.1f}" y="{top - 8:.1f}" text-anchor="end" class="chart-caption">'
        "Filled points and running-best line: kept"
        "</text>"
    )
    svg_parts.append(
        f'<text x="{width / 2:.1f}" y="{height - 32:.1f}" text-anchor="middle" class="chart-caption">'
        "Run index"
        "</text>"
    )
    svg_parts.append("</svg>")

    return (
        '<div class="run-chart">'
        '<p class="eyebrow">Across Runs</p>'
        '<div class="metric-stack">'
        f"{_render_metric_pair('runs charted', str(len(chart_runs)))}"
        f"{_render_metric_pair('best run', _format_float(max(metric_values)))}"
        f"{_render_metric_pair('latest run', _format_float(metric_values[-1]))}"
        "</div>"
        '<p class="iteration-meta">Best samples per second from each run. The solid line tracks the running best across runs.</p>'
        f'{"".join(svg_parts)}'
        "</div>"
    )


def _render_activity(run: dict[str, Any]) -> str:
    activity = run.get("recent_activity") or []
    if not activity:
        return '<p class="empty">No activity recorded yet.</p>'

    items: list[str] = []
    for event in activity:
        iteration = event.get("iteration")
        label = f"Iteration {int(iteration):02d}" if isinstance(iteration, int) else "Run"
        detail = str(event.get("detail", event.get("event", "update")))
        timestamp = _format_event_timestamp(event.get("at"))
        badge_class = _badge_class(str(event.get("status", "running")))
        artifact_html = ""
        if event.get("artifact"):
            artifact_html = f' · <a href="{html.escape(str(event["artifact"]))}">artifact</a>'
        items.append(
            '<li class="activity-item">'
            f'<span class="badge {badge_class}">{html.escape(str(event.get("phase", "update")))}</span>'
            f'<strong>{html.escape(label)}</strong>'
            f'<span>{html.escape(detail)}</span>'
            f'<span class="activity-time">{html.escape(timestamp)}{artifact_html}</span>'
            "</li>"
        )
    return '<ul class="activity-list">' + "".join(items) + "</ul>"


def _render_run_details(run: dict[str, Any], is_latest: bool) -> str:
    metric = str(run["metric"])
    heartbeat = run.get("heartbeat") or {}
    phase = str(heartbeat.get("phase", "idle"))
    heartbeat_status = str(heartbeat.get("status", "unknown"))
    heartbeat_detail = str(heartbeat.get("detail", "No current detail"))
    summary = [
        f'<span class="badge {_badge_class(str(run["latest_status"]))}">{html.escape(str(run["latest_status"]))}</span>',
        f'<span>{html.escape(run["started_at"])}</span>',
        f'<span>{html.escape(metric)} {html.escape(_format_float(run["best_metric"]))}</span>',
    ]
    iterations_html = "".join(_render_iteration(record, metric) for record in run["history"])
    if not iterations_html:
        iterations_html = '<p class="empty">No iterations recorded yet.</p>'

    heartbeat_html = (
        '<div class="metric-stack">'
        f"{_render_metric_pair('phase', phase)}"
        f"{_render_metric_pair('heartbeat', str(run['heartbeat_age_text']))}"
        f"{_render_metric_pair('heartbeat status', heartbeat_status)}"
        f"{_render_metric_pair('events', str(len(run.get('activity') or [])))}"
        f"{_render_metric_pair('delegation mode', 'enabled' if run['config'].get('delegate_to_mini') else 'off')}"
        f"{_render_metric_pair('subagent model', str(run['config'].get('subagent_model', 'n/a')))}"
        "</div>"
        f'<p class="iteration-meta"><span class="badge {run["heartbeat_freshness"]}">'
        f'{html.escape(run["heartbeat_freshness"])}'
        f"</span> {html.escape(heartbeat_detail)}</p>"
    )

    open_attr = " open" if is_latest else ""
    return (
        f"<details class=\"run\"{open_attr}>"
        f"<summary><span>Run {html.escape(run['run_id'])}</span>{''.join(summary)}</summary>"
        '<div class="run-body">'
        '<div class="run-grid">'
        f"{_render_metric_pair('baseline', _format_float(run['baseline_metric']))}"
        f"{_render_metric_pair('best', _format_float(run['best_metric']))}"
        f"{_render_metric_pair('delta', _format_float(run['improvement']))}"
        f"{_render_metric_pair('iterations', str(run['history_count']))}"
        "</div>"
        f'<p class="run-links">{_render_links(run)}</p>'
        '<div class="run-chart">'
        '<p class="eyebrow">Run Heartbeat</p>'
        f"{heartbeat_html}"
        "</div>"
        '<div class="run-chart">'
        '<p class="eyebrow">Recent Activity</p>'
        f"{_render_activity(run)}"
        "</div>"
        f"{_render_iteration_chart(run)}"
        f"{iterations_html}"
        "</div>"
        "</details>"
    )


def render_dashboard_html(payload: dict[str, Any]) -> str:
    git_state = payload["git"]
    tracked_branch = git_state["tracked_branch"]
    base_branch = git_state["base_branch"]
    divergence = git_state.get("divergence")
    latest_run = payload["runs"][0] if payload["runs"] else None
    run_cards = "".join(
        _render_run_details(run, is_latest=index == 0) for index, run in enumerate(payload["runs"])
    ) or '<p class="empty">No autoresearch runs found yet.</p>'

    if latest_run:
        latest_heartbeat = latest_run.get("heartbeat") or {}
        latest_run_html = (
            '<div class="card">'
            '<p class="eyebrow">Latest Run</p>'
            f"<h2>{html.escape(latest_run['run_id'])}</h2>"
            f"<p>{html.escape(latest_run['started_at'])}</p>"
            '<div class="metric-stack">'
            f"{_render_metric_pair(latest_run['metric'], _format_float(latest_run['best_metric']))}"
            f"{_render_metric_pair('phase', str(latest_heartbeat.get('phase', 'idle')))}"
            f"{_render_metric_pair('last heartbeat', str(latest_run['heartbeat_age_text']))}"
            "</div>"
            f'<p class="footer"><span class="badge {latest_run["heartbeat_freshness"]}">'
            f'{html.escape(str(latest_heartbeat.get("detail", "No recent heartbeat detail")))}'
            "</span></p>"
            f'<p class="run-links">{_render_links(latest_run)}</p>'
            "</div>"
        )
    else:
        latest_run_html = '<div class="card"><p class="eyebrow">Latest Run</p><h2>No runs yet</h2></div>'

    tracked_branch_name = str(tracked_branch["name"])
    divergence_html = (
        '<div class="metric-stack">'
        f"{_render_metric_pair(f'{tracked_branch_name} ahead', str(divergence['ahead']))}"
        f"{_render_metric_pair(f'{tracked_branch_name} behind', str(divergence['behind']))}"
        f"{_render_metric_pair('merge base', _format_commit(divergence['merge_base']))}"
        "</div>"
        if divergence
        else '<p class="empty">Branch divergence is unavailable.</p>'
    )
    status_lines_html = "".join(
        f"<li>{html.escape(line)}</li>" for line in git_state.get("status_lines", [])
    ) or "<li>clean</li>"
    runs_chart_html = _render_runs_chart(payload["runs"])
    master_log_links: list[str] = []
    if payload.get("master_change_log_link"):
        master_log_links.append(
            f'<a href="{html.escape(str(payload["master_change_log_link"]))}">master change log</a>'
        )
    if payload.get("master_change_log_jsonl_link"):
        master_log_links.append(
            f'<a href="{html.escape(str(payload["master_change_log_jsonl_link"]))}">master change log jsonl</a>'
        )
    master_log_html = " · ".join(master_log_links) if master_log_links else "No master changelog yet."

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="{REFRESH_SECONDS}">
  <title>Newton Autoresearch Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4efe7;
      --panel: rgba(255, 251, 245, 0.92);
      --panel-strong: #fffdf8;
      --ink: #18222d;
      --muted: #5f6c76;
      --line: rgba(24, 34, 45, 0.12);
      --accent: #0f766e;
      --accent-2: #c26d2b;
      --ok: #1f7a4d;
      --warn: #a8680f;
      --bad: #b53a2d;
      --shadow: 0 24px 60px rgba(34, 43, 53, 0.12);
      --radius: 22px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.15), transparent 34%),
        radial-gradient(circle at top right, rgba(194, 109, 43, 0.18), transparent 28%),
        linear-gradient(180deg, #f9f5ef 0%, var(--bg) 100%);
    }}
    main {{
      width: min(1200px, calc(100vw - 32px));
      margin: 24px auto 48px;
    }}
    .hero, .section, .card, .run {{
      box-shadow: var(--shadow);
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,255,255,0.86), rgba(255,248,239,0.96));
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 28px;
    }}
    .hero h1 {{
      margin: 6px 0 12px;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      font-size: clamp(2rem, 3vw, 3.4rem);
      line-height: 0.95;
      letter-spacing: -0.03em;
    }}
    .hero p, .footer, .empty, .iteration-meta, .run-links, .activity-time {{
      color: var(--muted);
    }}
    .eyebrow {{
      margin: 0;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--accent);
      font-size: 0.78rem;
      font-weight: 700;
    }}
    .grid {{
      display: grid;
      gap: 16px;
      margin-top: 18px;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    }}
    .card, .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 20px 22px;
      backdrop-filter: blur(8px);
    }}
    .metric-stack, .run-grid, .iteration-grid {{
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    }}
    .metric {{
      background: rgba(255,255,255,0.66);
      border: 1px solid rgba(24, 34, 45, 0.08);
      border-radius: 14px;
      padding: 12px 14px;
    }}
    .metric span {{
      display: block;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 0.76rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      background: rgba(24, 34, 45, 0.08);
      color: var(--ink);
    }}
    .badge.ok {{ background: rgba(31, 122, 77, 0.12); color: var(--ok); }}
    .badge.warn {{ background: rgba(168, 104, 15, 0.12); color: var(--warn); }}
    .badge.bad {{ background: rgba(181, 58, 45, 0.12); color: var(--bad); }}
    .badge.muted {{ background: rgba(24, 34, 45, 0.08); color: var(--muted); }}
    .status-list {{
      margin: 12px 0 0;
      padding-left: 18px;
    }}
    .run {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--panel-strong);
      margin-top: 14px;
      overflow: hidden;
    }}
    .run summary {{
      cursor: pointer;
      list-style: none;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      justify-content: space-between;
      align-items: center;
      padding: 16px 18px;
      font-weight: 700;
    }}
    .run summary::-webkit-details-marker {{ display: none; }}
    .run-body {{
      padding: 0 18px 18px;
      border-top: 1px solid var(--line);
    }}
    .run-chart {{
      margin-top: 18px;
      padding: 16px;
      border-radius: 16px;
      border: 1px solid rgba(24, 34, 45, 0.08);
      background: rgba(255,255,255,0.58);
    }}
    .iteration {{
      margin-top: 14px;
      border: 1px solid rgba(24, 34, 45, 0.08);
      border-radius: 16px;
      padding: 14px;
      background: rgba(248, 244, 237, 0.85);
    }}
    .iteration-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 12px;
    }}
    .iteration-chart {{
      display: block;
      width: 100%;
      height: auto;
      margin-top: 16px;
      overflow: visible;
    }}
    .chart-axis, .chart-caption {{
      fill: var(--muted);
      font-size: 12px;
      font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
    }}
    .activity-list {{
      list-style: none;
      margin: 12px 0 0;
      padding: 0;
      display: grid;
      gap: 10px;
    }}
    .activity-item {{
      display: grid;
      gap: 6px;
      padding: 12px 14px;
      border: 1px solid rgba(24, 34, 45, 0.08);
      border-radius: 14px;
      background: rgba(255,255,255,0.72);
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <p class="eyebrow">Newton Collection Autoresearch</p>
      <h1>Branch health, progress heartbeat, and iteration evidence in one place.</h1>
      <p>Generated {html.escape(payload['generated_at'])}. This page refreshes every {REFRESH_SECONDS} seconds and is rebuilt directly from git state plus the artifacts in <code>results/newton_collection_autoresearch</code>.</p>
      <div class="grid">
        <div class="card">
          <p class="eyebrow">Current Worktree</p>
          <h2>{html.escape(git_state['current_branch'])}</h2>
          <div class="metric-stack">
            {_render_metric_pair('HEAD', _format_commit(git_state['head_commit']))}
            {_render_metric_pair('dirty', 'yes' if git_state['dirty'] else 'no')}
            {_render_metric_pair('changes', str(len(git_state['status_lines'])))}
          </div>
          <p class="footer">{html.escape(git_state['head_subject'])}</p>
        </div>
        <div class="card">
          <p class="eyebrow">Tracked Branch</p>
          <h2>{html.escape(tracked_branch['name'])}</h2>
          <div class="metric-stack">
            {_render_metric_pair('exists', 'yes' if tracked_branch.get('exists') else 'no')}
            {_render_metric_pair('commit', _format_commit(tracked_branch.get('commit')))}
            {_render_metric_pair('base', base_branch['name'])}
          </div>
          <p class="footer">{html.escape(str(tracked_branch.get('subject', 'n/a')))}</p>
        </div>
        {latest_run_html}
      </div>
    </section>

    <section class="section">
      <h3>Branch Divergence</h3>
      {divergence_html}
    </section>

    <section class="section">
      <h3>Cross-Run Progress</h3>
      <p class="empty">Master changelog entries: {payload['master_change_log_count']}. {master_log_html}</p>
      {runs_chart_html}
    </section>

    <section class="section">
      <h3>Working Tree Status</h3>
      <ul class="status-list">{status_lines_html}</ul>
    </section>

    <section class="section">
      <h3>Runs</h3>
      {run_cards}
    </section>
  </main>
</body>
</html>
"""


def write_dashboard(repo_root: pathlib.Path, results_root: pathlib.Path) -> dict[str, Any]:
    payload = collect_dashboard_data(repo_root=repo_root, results_root=results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "dashboard.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (results_root / "index.html").write_text(render_dashboard_html(payload), encoding="utf-8")
    return payload
