#!/usr/bin/env bash

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required on PATH before running this script." >&2
    exit 1
fi

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"
NEWTON_REF="${NEWTON_REF:-8a2abf2}"
NEWTON_DIR="${PROTOMOTIONS_NEWTON_DIR:-$HOME/.cache/protomotions/newton-$NEWTON_REF}"

mkdir -p "$(dirname "$NEWTON_DIR")"

if [[ ! -d "$NEWTON_DIR/.git" ]]; then
    git clone https://github.com/newton-physics/newton.git "$NEWTON_DIR"
fi

git -C "$NEWTON_DIR" fetch --all --tags
git -C "$NEWTON_DIR" checkout "$NEWTON_REF"

uv venv --python "$PYTHON_VERSION" "$VENV_PATH"

# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

uv pip install --upgrade pip setuptools wheel
uv pip install --index-url https://download.pytorch.org/whl/cu130 "torch>=2.10.0"
uv pip install mujoco --pre -f https://py.mujoco.org/
uv pip install warp-lang --pre -U -f https://pypi.nvidia.com/warp-lang/
uv pip install git+https://github.com/google-deepmind/mujoco_warp.git@main
uv pip install -e "${NEWTON_DIR}[examples]"
uv pip install -e "$REPO_ROOT"
uv pip install -r "$REPO_ROOT/requirements_newton.txt"
uv pip install pre-commit pytest ruff

python - <<'PY'
import torch

print("python", ".".join(map(str, __import__("sys").version_info[:3])))
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY
