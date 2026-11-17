#!/usr/bin/env bash
# Wan Animate: animate a character, or replace a character in the driving scene.
# /wan/v2.2/animate defaults to animate; /wan/v2.2/recast defaults to replace.
# OPENFAL_ANIMATE_MODE=replace selects the replacement workflow explicitly.
set -euo pipefail

OPENFAL_URL="${OPENFAL_URL:-http://127.0.0.1:8000}"
OPENFAL_URL="${OPENFAL_URL%/}"
PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLES_DIR="${SCRIPT_DIR}/../samples"
OUT_DIR="${OPENFAL_OUTPUTS_DIR:-${SCRIPT_DIR}/outputs}"
MODE="${OPENFAL_ANIMATE_MODE:-animate}"
SEED="${OPENFAL_TEST_SEED:-20261117}"
case "${MODE}" in
  animate) ENDPOINT="animate" ;;
  replace) ENDPOINT="recast" ;;
  *) printf 'OPENFAL_ANIMATE_MODE must be animate or replace\n' >&2; exit 2 ;;
esac
if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
  printf 'OPENFAL_TEST_SEED must be a nonnegative integer\n' >&2
  exit 2
fi

resolve_sample() {
  if [[ -f "$1" ]]; then
    printf '%s\n' "$1"
  elif [[ -f "${SAMPLES_DIR}/$1" ]]; then
    printf '%s\n' "${SAMPLES_DIR}/$1"
  else
    printf 'Input file not found: %s\n' "$1" >&2
    return 1
  fi
}
if [[ "${MODE}" == "replace" ]]; then
  FACE_PATH="$(resolve_sample "${OPENFAL_REPLACEMENT_PORTRAIT:-${OPENFAL_PORTRAIT:-sample_replacement_photo.png}}")"
else
  FACE_PATH="$(resolve_sample "${OPENFAL_PORTRAIT:-sample_portrait_photo.png}")"
fi
DRIVER_PATH="$(resolve_sample "${OPENFAL_DRIVING_VIDEO:-sample_driving_portrait.mp4}")"
mkdir -p "${OUT_DIR}"

upload_file() {
  local mime response
  mime="$("${PYTHON}" -c 'import mimetypes, sys; print(mimetypes.guess_type(sys.argv[1])[0] or "application/octet-stream")' "$1")"
  if ! response="$(curl --fail-with-body -sS -X POST "${OPENFAL_URL}/v1/storage/upload" \
    -H "Content-Type: ${mime}" --data-binary "@$1")"; then
    printf '%s\n' "${response}" >&2
    return 1
  fi
  printf '%s\n' "${response}" | "${PYTHON}" -c 'import json, sys; print(json.load(sys.stdin)["url"])'
}

printf 'Uploading identity: %s\nDriving video: %s\n' "${FACE_PATH}" "${DRIVER_PATH}"
FACE_URL="$(upload_file "${FACE_PATH}")"
DRIVER_URL="$(upload_file "${DRIVER_PATH}")"
REQUEST_FILE="${OUT_DIR}/05_wan22_${MODE}.request.json"
RESPONSE_FILE="${OUT_DIR}/05_wan22_${MODE}.response.json"
OUT_FILE="${OUT_DIR}/05_wan22_${MODE}.mp4"
"${PYTHON}" - "${REQUEST_FILE}" "${FACE_URL}" "${DRIVER_URL}" "${MODE}" "${SEED}" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "prompt": "Natural expressive portrait animation, subtle head turn, warm studio lighting, high detail",
    "face_image_url": sys.argv[2],
    "driving_video_url": sys.argv[3],
    "animate_mode": sys.argv[4],
    "resolution": "480P",
    "aspect_ratio": "1:1",
    "duration": 2.0,
    "fps": 16,
    "seed": int(sys.argv[5]),
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

printf 'Generating Wan %s at %s with seed %s\n' "${MODE}" "${OPENFAL_URL}" "${SEED}"
if ! curl --fail-with-body -sS --max-time 1800 -X POST "${OPENFAL_URL}/wan/v2.2/${ENDPOINT}" \
  -H "Content-Type: application/json" --data-binary "@${REQUEST_FILE}" -o "${RESPONSE_FILE}"; then
  [[ ! -f "${RESPONSE_FILE}" ]] || cat "${RESPONSE_FILE}" >&2
  exit 1
fi
"${PYTHON}" -m json.tool "${RESPONSE_FILE}"
VID_URL="$("${PYTHON}" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["video"]["url"])' "${RESPONSE_FILE}")"
curl --fail-with-body -sS -L "${VID_URL}" -o "${OUT_FILE}"
printf 'Saved video -> %s\nRequest/response metadata -> %s\n' "${OUT_FILE}" "${OUT_DIR}"
