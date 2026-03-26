"""PyRoki retarget stage helpers."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile


def default_pyroki_python(repo_root: Path) -> Path:
    return repo_root / ".venvs" / "pyroki" / "bin" / "python"


def default_pyroki_script(repo_root: Path) -> Path:
    return repo_root / "pyroki" / "batch_retarget_to_smpl_lower_body.py"


def resolve_pyroki_runtime(
    *,
    repo_root: Path,
    retarget_python: Path | None,
    retarget_script: Path | None,
) -> tuple[Path, Path]:
    python_path = (retarget_python or default_pyroki_python(repo_root)).expanduser()
    script_path = (retarget_script or default_pyroki_script(repo_root)).expanduser()
    if not python_path.is_absolute():
        python_path = (repo_root / python_path).resolve()
    if not script_path.is_absolute():
        script_path = (repo_root / script_path).resolve()

    if not python_path.exists():
        raise FileNotFoundError(
            f"PyRoki interpreter not found: {python_path}. "
            "Expected the production runtime at ./.venvs/pyroki/bin/python."
        )
    if not script_path.exists():
        raise FileNotFoundError(
            f"PyRoki retarget script not found: {script_path}. "
            "Expected the production wrapper at ./pyroki/batch_retarget_to_smpl_lower_body.py."
        )
    return python_path, script_path


def verify_pyroki_runtime(python_path: Path) -> None:
    result = subprocess.run(
        [str(python_path), "-c", "import pyroki, jax, jaxls, yourdfpy"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "The selected PyRoki interpreter is missing required packages:\n"
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def run_pyroki_retarget_trial(
    *,
    python_path: Path,
    script_path: Path,
    keypoint_file: Path,
    retargeted_dir: Path,
    contacts_dir: Path,
    retarget_fps: int,
    retarget_urdf_path: Path,
    force: bool,
) -> tuple[Path, Path]:
    """Run the production PyRoki retargeter for one staged keypoint file."""
    output_file = retargeted_dir / f"{keypoint_file.stem}_retargeted.npz"
    contact_file = contacts_dir / f"{keypoint_file.stem}_contacts.npz"
    should_run_retarget = force or not output_file.exists()
    should_run_contacts = force or not contact_file.exists()

    if not should_run_retarget and not should_run_contacts:
        return output_file, contact_file

    with tempfile.TemporaryDirectory(prefix="pipeline-retarget-", dir="/tmp") as temp_dir:
        staged_keypoint = Path(temp_dir) / keypoint_file.name
        shutil.copy2(keypoint_file, staged_keypoint)

        common_args = [
            str(python_path),
            str(script_path),
            "--keypoints-folder-path",
            temp_dir,
            "--source-type",
            "treadmill",
            "--retarget-fps",
            str(retarget_fps),
            "--target-raw-frames",
            "-1",
            "--no-visualize",
            "--urdf-path",
            str(retarget_urdf_path),
        ]

        if should_run_retarget:
            result = subprocess.run(
                common_args + ["--output-dir", str(retargeted_dir)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Retargeting failed for {keypoint_file.name}:\n"
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )

        if should_run_contacts:
            result = subprocess.run(
                common_args
                + [
                    "--contacts-dir",
                    str(contacts_dir),
                    "--save-contacts-only",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Contact extraction failed for {keypoint_file.name}:\n"
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )

    if not output_file.exists():
        raise FileNotFoundError(f"No retargeted output created for {keypoint_file.name}")
    if not contact_file.exists():
        raise FileNotFoundError(f"No contact output created for {keypoint_file.name}")
    return output_file, contact_file
