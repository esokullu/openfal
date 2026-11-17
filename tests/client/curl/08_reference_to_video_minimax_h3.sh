#!/usr/bin/env bash
# ==============================================================================
# 08: MiniMax-H3 33B Ref2VA Multi-Reference Omni Video + Recast / Relight
# Endpoints:
#   - POST /minimax/h3/reference-to-video
#   - POST /minimax/h3-max/recast
#   - POST /minimax/h3-max/relight
#   - POST /minimax/h3-max/insert-video
# ==============================================================================
set -euo pipefail

OPENFAL_URL="${OPENFAL_URL:-http://127.0.0.1:8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLES_DIR="${SCRIPT_DIR}/../samples"
OUT_DIR="${SCRIPT_DIR}/outputs"
mkdir -p "${OUT_DIR}"

echo "=== [08] Uploading reference images (portrait + street scene) ==="
PORTRAIT_URL="$(curl -sS -X POST "${OPENFAL_URL}/v1/storage/upload" \
  -H "Content-Type: image/png" \
  --data-binary "@${SAMPLES_DIR}/sample_portrait.png" | python3 -c "import sys, json; print(json.load(sys.stdin)['url'])")"

SCENE_URL="$(curl -sS -X POST "${OPENFAL_URL}/v1/storage/upload" \
  -H "Content-Type: image/png" \
  --data-binary "@${SAMPLES_DIR}/sample_street_scene.png" | python3 -c "import sys, json; print(json.load(sys.stdin)['url'])")"

echo "  <Picture 1> Portrait: ${PORTRAIT_URL}"
echo "  <Picture 2> Scene   : ${SCENE_URL}"

echo "=== [08] MiniMax-H3 Ref2VA Multi-Reference Video+Audio (${OPENFAL_URL}) ==="
RESP="$(curl -sS --max-time 1800 -X POST "${OPENFAL_URL}/minimax/h3/reference-to-video" \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"The character in <Picture 1> walks through the sunlit cobblestone street of <Picture 2> while a tabby cat chases a poodle in the background, warm ambient sound\",
    \"reference_image_urls\": [\"${PORTRAIT_URL}\", \"${SCENE_URL}\"],
    \"resolution\": \"480P\",
    \"aspect_ratio\": \"16:9\",
    \"duration\": 3.0,
    \"enable_audio\": true
  }")"

echo "${RESP}" | python3 -m json.tool

VID_URL="$(echo "${RESP}" | python3 -c "import sys, json; print(json.load(sys.stdin)['video']['url'])")"
OUT_FILE="${OUT_DIR}/08_minimax_h3_ref2va.mp4"
curl -sS -L "${VID_URL}" -o "${OUT_FILE}"
echo "Saved video -> ${OUT_FILE}"
