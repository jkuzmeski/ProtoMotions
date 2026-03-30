from pathlib import Path

from scripts.run_newton_collection_autoresearch import (
    _build_codex_exec_cmd,
    _classify_iteration_changes,
    _render_prompt,
)


def test_classify_iteration_changes_ignores_preexisting_dirty_files() -> None:
    changed_files, unexpected = _classify_iteration_changes(
        baseline_dirty_files=["scripts/run_newton_collection_autoresearch.py"],
        current_dirty_files=[
            "protomotions/simulator/newton/simulator.py",
            "scripts/run_newton_collection_autoresearch.py",
        ],
        allowed_files=[
            "protomotions/simulator/newton/simulator.py",
            "protomotions/simulator/newton/config.py",
        ],
    )

    assert changed_files == ["protomotions/simulator/newton/simulator.py"]
    assert unexpected == []


def test_build_codex_exec_cmd_adds_reasoning_effort_and_model() -> None:
    cmd = _build_codex_exec_cmd(
        codex_bin="codex",
        cwd=Path("/tmp/workspace"),
        prompt="optimize",
        model="gpt-5.4",
        output_file=Path("/tmp/last.txt"),
        reasoning_effort="high",
        codex_config=['foo.bar="baz"'],
    )

    assert cmd[:4] == ["codex", "exec", "--full-auto", "--sandbox"]
    assert '-c' in cmd
    assert 'model_reasoning_effort="high"' in cmd
    assert 'foo.bar="baz"' in cmd
    assert "-m" in cmd
    assert "gpt-5.4" in cmd
    assert cmd[-1] == "optimize"


def test_render_prompt_includes_mini_delegation_guidance(tmp_path: Path) -> None:
    prompt_path = tmp_path / "program.md"
    prompt_path.write_text("Top-level objective.", encoding="utf-8")
    prompt = _render_prompt(
        program_path=prompt_path,
        benchmark_cmd="python bench.py",
        metric="samples_per_s",
        baseline_result={"samples_per_s": 1.0},
        iteration=1,
        allowed_files=["foo.py"],
        branch_name="autoresearch",
        delegate_to_mini=True,
        subagent_model="gpt-5.4-mini",
    )

    assert "Delegation policy:" in prompt
    assert "gpt-5.4-mini" in prompt
