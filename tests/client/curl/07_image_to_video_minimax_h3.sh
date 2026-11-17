#!/usr/bin/env bash
# ==============================================================================
# 07: MiniMax-H3 33B FL2VA First/Last Keyframe Image-to-Video + Stereo Audio
# Endpoint: POST /minimax/h3/image-to-video (alias: /fal-ai/minimax/h3/image-to-video)
# ==============================================================================
set -euo pipefail

OPENFAL_URL="${OPENFAL_URL:-http://127.0.0.1:8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLES_DIR="${SCRIPT_DIR}/../samples"
OUT_DIR="${SCRIPT_DIR}/outputs"
mkdir -p "${OUT_DIR}"

echo "=== [07] Uploading start keyframe sample_street_scene.png ==="
START_URL="$(curl -sS -X POST "${OPENFAL_URL}/v1/storage/upload" \
  -H "Content-Type: image/png" \
  --data-binary "@${SAMPLES_DIR}/sample_street_scene.png" | python3 -c "import sys, json; print(json.load(sys.stdin)['url'])")"
echo "  Start Keyframe URL: ${START_URL}"

echo "=== [07] MiniMax-H3 FL2VA Image-to-Video+Audio (${OPENFAL_URL}) ==="
RESP="$(curl -sS --max-time 1800 -X POST "${OPENFAL_URL}/minimax/h3/image-to-video" \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"A tabby cat sprints after a playful apricot poodle along the cobblestone street at sunset, lively street ambiance and paw steps\",
    \"start_image_url\": \"${START_URL}\",
    \"resolution\": \"480P\",
    \"aspect_ratio\": \"16:9\",
    \"duration\": 3.0,
    \"enable_audio\": true
  }")"

echo "${RESP}" | python3 -m json.tool

VID_URL="$(echo "${RESP}" | python3 -c "import sys, json; print(json.load(sys.stdin)['video']['url'])")"
OUT_FILE="${OUT_DIR}/07_minimax_h3_i2v.mp4"
curl -sS -L "${VID_URL}" -o "${OUT_FILE}"
echo "Saved video -> ${OUT_FILE}"
