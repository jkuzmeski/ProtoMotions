#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Convenience script to run the biomechanics lower-body retargeting pipeline
# with the JAX/PyRoki lower-body backend.

set -euo pipefail

usage() {
    echo "Usage: $0 <proto_python> <pyroki_python> <subject_profile> [output_dir] [--force]"
    echo ""
    echo "Arguments:"
    echo "  proto_python   Python interpreter with ProtoMotions/HumanRetargeting deps"
    echo "  pyroki_python  Python interpreter with PyRoki/JAX deps"
    echo "  subject_profile YAML subject profile"
    echo "  output_dir     Optional output directory"
    echo "  --force        Re-run stages even when outputs already exist"
}

if [ $# -lt 3 ]; then
    if [ $# -eq 1 ] && { [ "$1" = "--help" ] || [ "$1" = "-h" ]; }; then
        usage
        exit 0
    fi
    usage
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROTO_PYTHON="$1"
PYROKI_PYTHON="$2"
SUBJECT_PROFILE="$3"
shift 3

OUTPUT_DIR_OVERRIDE=""
FORCE_FLAG=0

while [ $# -gt 0 ]; do
    case "$1" in
        --force)
            FORCE_FLAG=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [ -z "$OUTPUT_DIR_OVERRIDE" ]; then
                OUTPUT_DIR_OVERRIDE="$1"
            else
                echo "Error: unexpected argument '$1'"
                usage
                exit 1
            fi
            ;;
    esac
    shift
done

if [ ! -f "$PROTO_PYTHON" ]; then
    echo "Error: ProtoMotions Python not found: $PROTO_PYTHON"
    exit 1
fi

if [ ! -f "$PYROKI_PYTHON" ]; then
    echo "Error: PyRoki Python not found: $PYROKI_PYTHON"
    exit 1
fi

if [ ! -f "$SUBJECT_PROFILE" ]; then
    echo "Error: subject profile not found: $SUBJECT_PROFILE"
    exit 1
fi

PROFILE_JSON_FILE="$(mktemp)"
trap 'rm -f "$PROFILE_JSON_FILE"' EXIT

if ! "$PROTO_PYTHON" - "$REPO_ROOT" "$SUBJECT_PROFILE" "$OUTPUT_DIR_OVERRIDE" "$PROFILE_JSON_FILE" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
profile_path = Path(sys.argv[2]).resolve()
output_override = sys.argv[3].strip()
output_json_path = Path(sys.argv[4]).resolve()

sys.path.insert(0, str(repo_root))

from HumanRetargeting.biomechanics_retarget.subject_assets import SubjectAssetBuilder
from HumanRetargeting.biomechanics_retarget.subject_profiles import load_subject_profile

profile = load_subject_profile(profile_path)
builder = SubjectAssetBuilder(
    profile=profile,
    rescale_dir=repo_root / "HumanRetargeting" / "rescale",
    assets_root=repo_root / "protomotions" / "data" / "assets",
)
assets = builder.build(force=False)

if output_override:
    output_dir = Path(output_override).resolve()
else:
    output_dir = (
        repo_root
        / "HumanRetargeting"
        / "biomechanics_retarget"
        / "processed_data"
        / profile.subject_id
    )

payload = {
    "subject_id": profile.subject_id,
    "input_dir": str(profile.input_dir),
    "output_dir": str(output_dir),
    "output_fps": str(profile.output_fps),
    "retarget_urdf": str(assets.urdf_path),
    "model_xml": str(assets.mjcf_path),
}
output_json_path.write_text(json.dumps(payload), encoding="utf-8")
PY
then
    echo "Error: failed to resolve subject profile metadata"
    exit 1
fi

mapfile -t PROFILE_INFO < <(
    "$PROTO_PYTHON" - "$PROFILE_JSON_FILE" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for key in (
    "subject_id",
    "input_dir",
    "output_dir",
    "output_fps",
    "retarget_urdf",
    "model_xml",
):
    print(payload[key])
PY
)

if [ "${#PROFILE_INFO[@]}" -ne 6 ]; then
    echo "Error: failed to resolve subject profile metadata"
    exit 1
fi

SUBJECT_ID="${PROFILE_INFO[0]}"
INPUT_DIR="${PROFILE_INFO[1]}"
OUTPUT_DIR="${PROFILE_INFO[2]}"
OUTPUT_FPS="${PROFILE_INFO[3]}"
RETARGET_URDF="${PROFILE_INFO[4]}"
MODEL_XML="${PROFILE_INFO[5]}"

KEYPOINTS_DIR="${OUTPUT_DIR}/keypoints"
RETARGETED_DIR="${OUTPUT_DIR}/retargeted_motions"
CONTACTS_DIR="${OUTPUT_DIR}/contacts"
MOTION_DIR="${OUTPUT_DIR}/motion_files"
PACKAGED_DIR="${OUTPUT_DIR}/packaged_data"

mkdir -p "$OUTPUT_DIR"

FORCE_ARGS=()
if [ "$FORCE_FLAG" -eq 1 ]; then
    FORCE_ARGS+=(--force)
fi

RETARGET_ARGS=()
if [ "$FORCE_FLAG" -eq 0 ]; then
    RETARGET_ARGS+=(--skip-existing)
fi

echo "=============================================="
echo "Biomechanics Subject Retargeting (PyRoki)"
echo "=============================================="
echo "Subject ID:          $SUBJECT_ID"
echo "ProtoMotions Python: $PROTO_PYTHON"
echo "PyRoki Python:       $PYROKI_PYTHON"
echo "Subject profile:     $SUBJECT_PROFILE"
echo "Input dir:           $INPUT_DIR"
echo "Output dir:          $OUTPUT_DIR"
echo "Output FPS:          $OUTPUT_FPS"
echo "Retarget URDF:       $RETARGET_URDF"
echo "Model XML:           $MODEL_XML"
echo "=============================================="

echo ""
echo "[Step 1/6] Transforming treadmill data to overground..."
"$PROTO_PYTHON" "$REPO_ROOT/HumanRetargeting/biomechanics_retarget/pipeline.py" \
    "$INPUT_DIR" \
    "$OUTPUT_DIR" \
    --subject-profile "$SUBJECT_PROFILE" \
    --step overground \
    "${FORCE_ARGS[@]}"

echo ""
echo "[Step 2/6] Extracting keypoints..."
"$PROTO_PYTHON" "$REPO_ROOT/HumanRetargeting/biomechanics_retarget/pipeline.py" \
    "$INPUT_DIR" \
    "$OUTPUT_DIR" \
    --subject-profile "$SUBJECT_PROFILE" \
    --step keypoints \
    "${FORCE_ARGS[@]}"

echo ""
echo "[Step 3/6] Retargeting lower-body motions with JAX/PyRoki..."
"$PYROKI_PYTHON" "$REPO_ROOT/pyroki/batch_retarget_to_smpl_lower_body_pyroki.py" \
    --keypoints-folder-path "$KEYPOINTS_DIR" \
    --output-dir "$RETARGETED_DIR" \
    --urdf-path "$RETARGET_URDF" \
    --retarget-fps "$OUTPUT_FPS" \
    --source-type treadmill \
    --no-visualize \
    "${RETARGET_ARGS[@]}"

echo ""
echo "[Step 4/6] Extracting contact labels..."
"$PYROKI_PYTHON" "$REPO_ROOT/pyroki/batch_retarget_to_smpl_lower_body_pyroki.py" \
    --keypoints-folder-path "$KEYPOINTS_DIR" \
    --contacts-dir "$CONTACTS_DIR" \
    --urdf-path "$RETARGET_URDF" \
    --retarget-fps "$OUTPUT_FPS" \
    --source-type treadmill \
    --save-contacts-only \
    "${RETARGET_ARGS[@]}"

echo ""
echo "[Step 5/6] Converting retargeted files to .motion..."
"$PROTO_PYTHON" "$REPO_ROOT/HumanRetargeting/biomechanics_retarget/pipeline.py" \
    "$INPUT_DIR" \
    "$OUTPUT_DIR" \
    --subject-profile "$SUBJECT_PROFILE" \
    --step convert \
    "${FORCE_ARGS[@]}"

echo ""
echo "[Step 6/6] Packaging MotionLib..."
"$PROTO_PYTHON" "$REPO_ROOT/HumanRetargeting/biomechanics_retarget/pipeline.py" \
    "$INPUT_DIR" \
    "$OUTPUT_DIR" \
    --subject-profile "$SUBJECT_PROFILE" \
    --step package \
    "${FORCE_ARGS[@]}"

echo ""
echo "=============================================="
echo "Retargeting complete!"
echo "=============================================="
echo "Keypoints dir:       $KEYPOINTS_DIR"
echo "Retargeted dir:      $RETARGETED_DIR"
echo "Contacts dir:        $CONTACTS_DIR"
echo "Motion dir:          $MOTION_DIR"
echo "Packaged dir:        $PACKAGED_DIR"
echo ""
