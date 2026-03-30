import json
import subprocess
from pathlib import Path

from protomotions.utils.newton_collection_dashboard import write_dashboard


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def test_write_dashboard_includes_git_run_heartbeat_and_activity(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    _git(repo_root, "init")
    _git(repo_root, "config", "user.name", "Proto Test")
    _git(repo_root, "config", "user.email", "proto@example.com")
    _git(repo_root, "checkout", "-b", "main")

    tracked_file = repo_root / "tracked.txt"
    tracked_file.write_text("baseline\n", encoding="utf-8")
    _git(repo_root, "add", "tracked.txt")
    _git(repo_root, "commit", "-m", "initial")
    _git(repo_root, "branch", "autoresearch")

    tracked_file.write_text("baseline\nlocal change\n", encoding="utf-8")

    results_root = repo_root / "results" / "newton_collection_autoresearch"
    run_dir = results_root / "20260327_181300"
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "base_branch": "main",
                "branch_name": "autoresearch",
                "metric": "samples_per_s",
                "iterations": 2,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "baseline.json").write_text(
        json.dumps({"samples_per_s": 1000.0}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "history.json").write_text(
        json.dumps(
            [
                {
                    "iteration": 1,
                    "status": "accepted",
                    "improvement": 125.0,
                    "commit_sha": _git(repo_root, "rev-parse", "HEAD"),
                    "changed_files": ["protomotions/simulator/newton/simulator.py"],
                    "benchmark": {"samples_per_s": 1125.0},
                    "report_file": "iteration_01_report.md",
                    "agent_file": "iteration_01_agent.txt",
                }
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "heartbeat.json").write_text(
        json.dumps(
            {
                "updated_at": "2026-03-27T18:15:10+00:00",
                "phase": "codex_running",
                "status": "running",
                "iteration": 2,
                "detail": "Codex iteration is still running | iteration_02_agent.txt 4096 B",
                "artifact": "iteration_02_agent.txt",
                "agent_output_bytes": 4096,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "activity.jsonl").write_text(
        json.dumps(
            {
                "at": "2026-03-27T18:14:30+00:00",
                "event": "iteration_accepted",
                "phase": "iteration_complete",
                "status": "running",
                "iteration": 1,
                "detail": "Accepted iteration 01 with delta +125.00 samples_per_s",
                "artifact": "iteration_01_agent.txt",
            },
            sort_keys=True,
        )
        + "\n"
        + json.dumps(
            {
                "at": "2026-03-27T18:15:10+00:00",
                "event": "codex_started",
                "phase": "codex_running",
                "status": "running",
                "iteration": 2,
                "detail": "Codex iteration 02 started",
                "artifact": "iteration_02_agent.txt",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "iteration_01_agent.txt").write_text(
        "Optimized simulator path.\nDelegation summary: none\n",
        encoding="utf-8",
    )
    (run_dir / "final_best.json").write_text(
        json.dumps({"samples_per_s": 1125.0}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.md").write_text("# summary\n", encoding="utf-8")

    payload = write_dashboard(repo_root=repo_root, results_root=results_root)

    assert payload["tracked_branch"] == "autoresearch"
    assert payload["git"]["current_branch"] == "main"
    assert payload["git"]["dirty"] is True
    assert payload["runs"][0]["latest_status"] == "accepted"
    assert payload["runs"][0]["improvement"] == 125.0
    assert payload["runs"][0]["heartbeat"]["phase"] == "codex_running"

    dashboard_json = json.loads((results_root / "dashboard.json").read_text(encoding="utf-8"))
    assert dashboard_json["latest_run_id"] == "20260327_181300"

    html_output = (results_root / "index.html").read_text(encoding="utf-8")
    assert "Newton Collection Autoresearch" in html_output
    assert "Run Heartbeat" in html_output
    assert "Recent Activity" in html_output
    assert "Samples Per Second By Iteration" in html_output
    assert 'class="iteration-chart"' in html_output
    assert "Scatter plot showing samples per second for each iteration" in html_output
    assert "Codex iteration is still running" in html_output
    assert "Accepted iteration 01 with delta +125.00 samples_per_s" in html_output
    assert "Delegation:" in html_output
