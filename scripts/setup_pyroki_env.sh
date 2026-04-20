#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Create a dedicated virtual environment for PyRoki and install the upstream repo.
#
# Usage:
#   ./scripts/setup_pyroki_env.sh [python_executable] [venv_dir] [repo_dir]
#
# Example:
#   ./scripts/setup_pyroki_env.sh python3.12 .venvs/pyroki third_party/pyroki

set -euo pipefail

PYTHON_BIN="${1:-python3}"
VENV_DIR="${2:-.venvs/pyroki}"
REPO_DIR="${3:-third_party/pyroki}"
PYROKI_REPO_URL="https://github.com/chungmin99/pyroki.git"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${ROOT_DIR}/${VENV_DIR}"
REPO_PATH="${ROOT_DIR}/${REPO_DIR}"

echo "=============================================="
echo "Setting up PyRoki environment"
echo "=============================================="
echo "Python:   ${PYTHON_BIN}"
echo "Venv dir: ${VENV_PATH}"
echo "Repo dir: ${REPO_PATH}"
echo "=============================================="

"${PYTHON_BIN}" -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [ -d "${REPO_PATH}/.git" ]; then
    git -C "${REPO_PATH}" fetch --all --tags
    git -C "${REPO_PATH}" pull --ff-only
elif [ -d "${REPO_PATH}" ]; then
    echo "Error: ${REPO_PATH} exists but is not a git checkout." >&2
    exit 1
else
    mkdir -p "$(dirname "${REPO_PATH}")"
    git clone "${PYROKI_REPO_URL}" "${REPO_PATH}"
fi

python -m pip install -e "${REPO_PATH}"
python -c "import pyroki, jax, jaxls, yourdfpy; print('PyRoki runtime OK')"

echo
echo "PyRoki setup complete."
echo "Interpreter: ${VENV_PATH}/bin/python"
