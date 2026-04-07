# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
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
#
"""
SLURM Training Launcher for ProtoMotions
=========================================

This script provides a template for launching ProtoMotions training jobs on SLURM clusters.
It handles code synchronization, container setup, and job submission.

BEFORE USING:
-------------
1. Edit the CLUSTER CONFIGURATION section below to match your cluster setup
2. Ensure you have SSH access to your cluster login node
3. Prepare your container images (see CONTAINER SETUP below)

USAGE EXAMPLE:
--------------
python protomotions/train_slurm.py \\
    --robot-name=g1 \\
    --simulator=isaacgym \\
    --num-envs=4096 \\
    --batch-size=32768 \\
    --motion-file=data/motions/my_motion.pt \\
    --experiment-path=examples/experiments/mimic/mimic_mlp.py \\
    --experiment-name=my_experiment \\
    --user=myusername

CONTAINER SETUP:
----------------
You'll need to prepare container images with the required dependencies:
- For IsaacGym: PyTorch + IsaacGym + ProtoMotions dependencies
- For IsaacLab: Isaac Lab + ProtoMotions dependencies
- For Newton: PyTorch + Newton + Warp + ProtoMotions dependencies

Convert Docker images to Singularity/Enroot format as required by your cluster.
"""

import argparse
import datetime
import os
from pathlib import Path
import shlex
import subprocess


# =============================================================================
# CLUSTER CONFIGURATION - EDIT THIS SECTION FOR YOUR CLUSTER
# =============================================================================

# Login node hostname (e.g., "login.mycluster.edu")
CLUSTER_LOGIN_NODE = "YOUR_CLUSTER_LOGIN_NODE"

# Base directory for experiments on the cluster filesystem
CLUSTER_BASE_DIR = "/path/to/your/experiments/directory"

# Container images (Singularity .sif or Enroot .sqsh format)
CONTAINER_IMAGES = {
    "isaacgym": "/path/to/containers/isaacgym.sqsh",
    "isaaclab": "/path/to/containers/isaaclab.sqsh",
    "newton": "/path/to/containers/newton.sqsh",
}

# Python executable inside each container
PYTHON_EXECUTABLES = {
    "isaacgym": "python",
    "isaaclab": "/workspace/isaaclab/isaaclab.sh -p",  # Isaac Lab wrapper
    "newton": "python",
}

# Default SLURM account (your allocation/project)
DEFAULT_SLURM_ACCOUNT = "your_account"

# Default SLURM partitions (comma-separated)
DEFAULT_SLURM_PARTITION = "gpu"

# Filesystem mounts for container (cluster-specific)
CONTAINER_MOUNTS = "/scratch:/scratch:rw"

# =============================================================================
# END CLUSTER CONFIGURATION
# =============================================================================


def subprocess_run(cmd, ignore_err=False, **kwargs):
    """Run subprocess command and raise on error unless ignored."""
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0 and not ignore_err:
        raise Exception(f"Command failed: {cmd}")
    return result


def check_wandb_credentials(user):
    """
    Check if wandb credentials are available on the remote machine.
    Returns the API key if it needs to be passed explicitly, or None if already configured.
    """
    # Check for ~/.netrc with wandb.ai entry
    check_cmd = f"ssh {user}@{CLUSTER_LOGIN_NODE} \"grep -q 'machine api.wandb.ai' ~/.netrc 2>/dev/null && echo 'found' || echo 'not_found'\""
    result = subprocess_run(check_cmd, shell=True, capture_output=True, text=True, ignore_err=True)

    if result.stdout.strip() == "found":
        print("WANDB credentials found in ~/.netrc on remote. No API key needed.")
        return None

    # Check environment variable
    check_env_cmd = f'ssh {user}@{CLUSTER_LOGIN_NODE} "bash -l -c \'test -n \\"\\$WANDB_API_KEY\\" && echo found || echo not_found\'"'
    result = subprocess_run(check_env_cmd, shell=True, capture_output=True, text=True, ignore_err=True)

    if result.stdout.strip() == "found":
        print("WANDB_API_KEY found in remote environment.")
        return None

    # Try local environment
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        print("Using WANDB_API_KEY from local environment.")
        return wandb_api_key

    # Prompt user
    print("WANDB credentials not found. Options:")
    print("  1. Run 'wandb login' on the cluster")
    print("  2. Set WANDB_API_KEY in your cluster ~/.bashrc")
    print("  3. Enter API key now")
    wandb_api_key = input("Enter WANDB API key (or press Enter to skip): ").strip()
    return wandb_api_key if wandb_api_key else None


def create_parser():
    """Create argument parser with all training options."""
    parser = argparse.ArgumentParser(
        description="Launch ProtoMotions training on SLURM cluster",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--robot-name", type=str, required=True, help="Robot name (e.g., 'g1', 'smpl')")
    parser.add_argument("--simulator", type=str, required=True, help="Simulator (isaacgym/isaaclab/newton)")
    parser.add_argument("--num-envs", type=int, required=True, help="Number of parallel environments")
    parser.add_argument("--batch-size", type=int, required=True, help="Training batch size")
    parser.add_argument("--motion-file", type=str, required=True, help="Path to motion data file")
    parser.add_argument("--experiment-path", type=str, required=True, help="Path to experiment config")
    parser.add_argument("--experiment-name", type=str, required=True, help="Experiment name for logging")
    parser.add_argument("--user", type=str, required=True, help="Cluster username")

    # Optional arguments
    parser.add_argument("--scenes-file", type=str, default=None, help="Path to scenes file (optional)")
    parser.add_argument("--headless", default=True, help="Run headless (no GUI)")
    parser.add_argument("--training-max-steps", type=int, default=10000000000, help="Max training steps")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--use-slurm", action="store_true", default=True, help="Enable SLURM autoresume")
    parser.add_argument("--ngpu", type=int, default=1, help="GPUs per node")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--overrides", nargs="*", default=[], help="Config overrides (key=value)")
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "Additional raw CLI args appended to protomotions/train_agent.py. "
            "This option must be last because it captures all remaining tokens."
        ),
    )
    parser.add_argument(
        "--sync-paths",
        nargs="*",
        default=[],
        help=(
            "Additional local repo-relative paths to rsync into the remote experiment "
            "folder before submission, for example results/my_teacher_run"
        ),
    )

    # SLURM arguments
    parser.add_argument("-t", "--slurm-time", default="4:00:00", help="Job time limit")
    parser.add_argument("-a", "--account", default=DEFAULT_SLURM_ACCOUNT, help="SLURM account")
    parser.add_argument("-p", "--partition", default=DEFAULT_SLURM_PARTITION, help="SLURM partition")
    parser.add_argument("--array-size", type=int, default=5, help="Job array size for auto-resume")
    parser.add_argument(
        "--slurm-begin",
        default=None,
        help="Optional SBATCH --begin value, for example now+12hours",
    )
    parser.add_argument(
        "--remote-dir-name",
        default=None,
        help=(
            "Stable remote directory name under CLUSTER_BASE_DIR/<user>/ to reuse "
            "the same synced repo across related jobs. Defaults to exp-<timestamp>."
        ),
    )
    parser.add_argument("--only-upload-code", action="store_true", help="Only sync code, don't submit")

    return parser


def sync_code_to_cluster(user, exp_folder, local_repo, extra_sync_paths=None):
    """Sync local code to cluster, excluding unnecessary files."""
    exclude_patterns = [
        ".git", ".idea", "**/__pycache__", "**/*.egg-info",
        "outputs", "output/*", "results", "exps", "tmp",
        "data/smpl", "data/amass", "data/pretrained_models",
        "docs", "docs/*",
    ]
    exclude_str = " ".join([f'--exclude="{f}"' for f in exclude_patterns])

    # Create remote directory
    subprocess_run(f"ssh {user}@{CLUSTER_LOGIN_NODE} 'mkdir -p {exp_folder}'", shell=True)

    # Sync code
    rsync_cmd = f"rsync -az --partial -m --chmod=775 {exclude_str} {local_repo}/ {user}@{CLUSTER_LOGIN_NODE}:{exp_folder}/"
    print(f"Syncing code: {rsync_cmd}")
    subprocess_run(rsync_cmd, shell=True)

    extra_sync_paths = extra_sync_paths or []
    for raw_path in extra_sync_paths:
        local_path = Path(raw_path).expanduser()
        if not local_path.is_absolute():
            local_path = (local_repo / local_path).resolve()
        else:
            local_path = local_path.resolve()

        if not local_path.exists():
            raise FileNotFoundError(f"Sync path does not exist: {local_path}")

        try:
            relative_path = local_path.relative_to(local_repo)
        except ValueError as exc:
            raise ValueError(
                f"Sync path must live under the local repo root {local_repo}: {local_path}"
            ) from exc

        remote_parent = os.path.join(exp_folder, str(relative_path.parent))
        subprocess_run(
            f"ssh {user}@{CLUSTER_LOGIN_NODE} mkdir -p {shlex.quote(remote_parent)}",
            shell=True,
        )

        source = f"{local_path}/" if local_path.is_dir() else str(local_path)
        destination = f"{user}@{CLUSTER_LOGIN_NODE}:{remote_parent}/"
        rsync_extra_cmd = f"rsync -az --partial --chmod=775 {shlex.quote(source)} {shlex.quote(destination)}"
        print(f"Syncing extra path: {rsync_extra_cmd}")
        subprocess_run(rsync_extra_cmd, shell=True)


def build_job_command(args, exp_folder, python_path):
    """Build the training command to run inside the container."""
    def q(value):
        return shlex.quote(str(value))

    # Install package
    job_cmd = (
        "pip uninstall -y protomotions 2>/dev/null; "
        f"cd {q(exp_folder)}; "
        "pip install -e . --no-dependencies; "
    )

    # Add WANDB API key if needed
    if args.use_wandb:
        wandb_key = check_wandb_credentials(args.user)
        if wandb_key:
            job_cmd += f"WANDB_API_KEY={q(wandb_key)} "

    # Build training command
    job_cmd += (
        f"PYTHONUNBUFFERED=1 {python_path} -u protomotions/train_agent.py "
        f"--robot-name={q(args.robot_name)} "
        f"--simulator={q(args.simulator)} "
        f"--motion-file={q(args.motion_file)} "
        f"--ngpu={q(args.ngpu)} "
        f"--nodes={q(args.nodes)} "
        f"--training-max-steps={q(args.training_max_steps)} "
        f"--experiment-name={q(args.experiment_name)} "
        f"--experiment-path={q(args.experiment_path)} "
        f"--num-envs={q(args.num_envs)} "
        f"--batch-size={q(args.batch_size)} "
        f"--use-slurm "
    )

    if args.scenes_file:
        job_cmd += f"--scenes-file={q(args.scenes_file)} "
    if args.use_wandb:
        job_cmd += "--use-wandb "
    if args.checkpoint:
        job_cmd += f"--checkpoint={q(args.checkpoint)} "
    if args.overrides:
        job_cmd += "--overrides " + " ".join(q(value) for value in args.overrides) + " "
    if args.extra_args:
        job_cmd += " ".join(q(value) for value in args.extra_args) + " "

    return job_cmd


def generate_slurm_script(args, exp_folder, job_cmd, container_image):
    """Generate SLURM batch script content."""
    log_file = f"{exp_folder}/slurm_output.log"

    # Container run command (adjust for your cluster's container runtime)
    srun_cmd = (
        f"srun "
        f"--container-image={container_image} "
        f"--container-mounts={CONTAINER_MOUNTS} "
        f"/bin/bash -c '{job_cmd}'"
    )

    script = f"""#!/bin/bash
#SBATCH --job-name={args.experiment_name}
#SBATCH --account={args.account}
#SBATCH --partition={args.partition}
#SBATCH --nodes={args.nodes}
#SBATCH --gpus-per-node={args.ngpu}
#SBATCH --ntasks-per-node={args.ngpu}
#SBATCH --time={args.slurm_time}
#SBATCH --output={log_file}
#SBATCH --error={log_file}
#SBATCH --array=0-{args.array_size}%1
"""
    if args.slurm_begin:
        script += f"#SBATCH --begin={args.slurm_begin}\n"

    script += f"""
# Job array enables automatic resume: if job times out, next array task continues

{srun_cmd}
"""
    return script, log_file


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Setup paths
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    local_repo = Path(__file__).parent.parent
    remote_dir_name = args.remote_dir_name or f"exp-{timestamp}"
    exp_folder = os.path.join(CLUSTER_BASE_DIR, args.user, remote_dir_name)

    print(f"Local repository: {local_repo}")
    print(f"Remote experiment folder: {exp_folder}")

    # Sync code
    sync_code_to_cluster(args.user, exp_folder, local_repo, args.sync_paths)

    if args.only_upload_code:
        print("Code uploaded. Exiting (--only-upload-code).")
        return

    # Get container and python path
    container_image = CONTAINER_IMAGES.get(args.simulator)
    python_path = PYTHON_EXECUTABLES.get(args.simulator, "python")

    if not container_image:
        raise ValueError(f"No container configured for simulator: {args.simulator}")

    # Build job command
    job_cmd = build_job_command(args, exp_folder, python_path)

    # Generate SLURM script
    slurm_script, log_file = generate_slurm_script(args, exp_folder, job_cmd, container_image)

    print("\n" + "=" * 60)
    print("SLURM SCRIPT:")
    print("=" * 60)
    print(slurm_script)

    # Write and upload script
    local_script = Path(f"tmp/slurm_{timestamp}.sh")
    local_script.parent.mkdir(exist_ok=True)
    local_script.write_text(slurm_script)

    remote_script = f"{exp_folder}/submit.sh"
    subprocess_run(f"scp {local_script} {args.user}@{CLUSTER_LOGIN_NODE}:{remote_script}", shell=True)

    # Submit job
    submit_cmd = f"ssh {args.user}@{CLUSTER_LOGIN_NODE} 'chmod +x {remote_script}; sbatch {remote_script}'"
    print(f"\nSubmitting: {submit_cmd}")
    subprocess_run(submit_cmd, shell=True)

    print("\n" + "=" * 60)
    print("JOB SUBMITTED!")
    print("=" * 60)
    print(f"Monitor logs:  ssh {args.user}@{CLUSTER_LOGIN_NODE} 'tail -f {log_file}'")
    print(f"Check status:  ssh {args.user}@{CLUSTER_LOGIN_NODE} 'squeue -u {args.user}'")
    print(f"Cancel job:    ssh {args.user}@{CLUSTER_LOGIN_NODE} 'scancel <job_id>'")
    print("=" * 60)


if __name__ == "__main__":
    main()
