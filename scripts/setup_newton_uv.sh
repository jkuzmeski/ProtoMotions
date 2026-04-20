#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
newton_root="${NEWTON_PATH:-$(cd "${repo_root}/.." && pwd)/newton}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found on PATH." >&2
    exit 1
fi

if [[ ! -f "${newton_root}/pyproject.toml" ]]; then
    echo "Expected a sibling Newton checkout at: ${newton_root}" >&2
    echo "Set NEWTON_PATH=/absolute/path/to/newton if your fork lives elsewhere." >&2
    exit 1
fi

cd "${repo_root}"
uv sync --python 3.12 --extra newton

cat <<EOF

Environment ready.
Activate it with:
  source "${repo_root}/.venv/bin/activate"

Using Newton from:
  ${newton_root}
EOF
