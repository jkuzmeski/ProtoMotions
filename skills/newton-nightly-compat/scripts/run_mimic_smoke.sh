#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_mimic_smoke.sh [options]

Run the standard ProtoMotions/Newton mimic smoke command with a fixed
motion file and experiment config, optionally capped to a target epoch count.

Options:
  --epochs N             Target epoch threshold to clear. Default: 10
  --num-envs N           Environment count. Default: 512
  --num-steps N          PPO rollout steps per epoch. Default: 32
  --batch-size N         PPO batch size. Default: 2048
  --motion-file PATH     Motion file to use.
  --experiment-path PATH Experiment config path.
  --experiment-name NAME Experiment name. Default: newton-nightly-smoke-<timestamp>
  --viewer BACKEND       Enable viewer mode with backend "viser" or "gl".
  --dry-run              Print the command without executing it.
  -h, --help             Show this help text.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

epochs=10
num_envs=512
num_steps=32
batch_size=2048
motion_file="HumanRetargeting/biomechanics_retarget/processed_data/S_GENERIC/motion_files/S02_20ms_Long.motion"
experiment_path="examples/experiments/mimic/mlp.py"
experiment_name="newton-nightly-smoke-$(date +%Y%m%d-%H%M%S)"
viewer_backend=""
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs)
      epochs="$2"
      shift 2
      ;;
    --num-envs)
      num_envs="$2"
      shift 2
      ;;
    --num-steps)
      num_steps="$2"
      shift 2
      ;;
    --batch-size)
      batch_size="$2"
      shift 2
      ;;
    --motion-file)
      motion_file="$2"
      shift 2
      ;;
    --experiment-path)
      experiment_path="$2"
      shift 2
      ;;
    --experiment-name)
      experiment_name="$2"
      shift 2
      ;;
    --viewer)
      viewer_backend="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$epochs" =~ ^[0-9]+$ && "$num_envs" =~ ^[0-9]+$ && "$num_steps" =~ ^[0-9]+$ && "$batch_size" =~ ^[0-9]+$ ]]; then
  echo "epochs, num-envs, num-steps, and batch-size must be integers" >&2
  exit 2
fi

training_max_steps=$(( epochs * num_envs * num_steps ))

cmd=(
  python -u protomotions/train_agent.py
  --robot-name smpl_lower_body_subject_S_GENERIC
  --simulator newton
  --num-envs "$num_envs"
  --batch-size "$batch_size"
  --motion-file "$motion_file"
  --experiment-path "$experiment_path"
  --training-max-steps "$training_max_steps"
  --experiment-name "$experiment_name"
)

if [[ -n "$viewer_backend" ]]; then
  cmd+=(
    --overrides
    "simulator.headless=False"
    "simulator.viewer_backend=${viewer_backend}"
  )
else
  cmd+=(
    --overrides
    "simulator.headless=True"
  )
fi

printf 'Smoke command:\n'
printf '  %q' "${cmd[@]}"
printf '\n'

if [[ "$dry_run" -eq 1 ]]; then
  exit 0
fi

exec "${cmd[@]}"
