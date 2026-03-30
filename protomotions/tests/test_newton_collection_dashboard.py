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


def test_write_dashboard_includes_git_and_run_state(tmp_path: Path) -> None:
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
                }
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
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

    dashboard_json = json.loads((results_root / "dashboard.json").read_text(encoding="utf-8"))
    assert dashboard_json["latest_run_id"] == "20260327_181300"

    html_output = (results_root / "index.html").read_text(encoding="utf-8")
    assert "Newton Collection Autoresearch" in html_output
    assert "Iteration Delta Vs Baseline" in html_output
    assert 'class="iteration-chart"' in html_output
    assert "Scatter plot showing each iteration delta from the run baseline" in html_output
    assert "<circle " in html_output
    assert "Each point shows iteration benchmark minus the original run baseline." in html_output
    assert "Iteration 01" in html_output
    assert "accepted" in html_output
