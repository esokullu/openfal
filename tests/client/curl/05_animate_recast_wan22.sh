#!/usr/bin/env bash
# ==============================================================================
# 05: Alibaba Wan 2.2 Animate-14B (Character Animation & Video Face Recast)
# Endpoint: POST /wan/v2.2/animate (alias: /fal-ai/wan/v2.2/recast)
# ==============================================================================
set -euo pipefail

OPENFAL_URL="${OPENFAL_URL:-http://127.0.0.1:8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLES_DIR="${SCRIPT_DIR}/../samples"
OUT_DIR="${SCRIPT_DIR}/outputs"
mkdir -p "${OUT_DIR}"

echo "=== [05] Uploading character portrait sample_portrait.png ==="
FACE_URL="$(curl -sS -X POST "${OPENFAL_URL}/v1/storage/upload" \
  -H "Content-Type: image/png" \
  --data-binary "@${SAMPLES_DIR}/${OPENFAL_PORTRAIT:-sample_portrait_photo.png}" | python3 -c "import sys, json; print(json.load(sys.stdin)['url'])")"
echo "  Character Portrait URL: ${FACE_URL}"

echo "=== [05] Wan 2.2 Animate-14B Video Character Animation (${OPENFAL_URL}) ==="
RESP="$(curl -sS --max-time 1800 -X POST "${OPENFAL_URL}/wan/v2.2/animate" \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"Natural expressive portrait animation, subtle head turn, warm studio lighting, high detail\",
    \"face_image_url\": \"${FACE_URL}\",
    \"start_image_url\": \"${FACE_URL}\",
    \"resolution\": \"480P\",
    \"aspect_ratio\": \"1:1\",
    \"duration\": 2.0,
    \"fps\": 16
  }")"

echo "${RESP}" | python3 -m json.tool

VID_URL="$(echo "${RESP}" | python3 -c "import sys, json; print(json.load(sys.stdin)['video']['url'])")"
OUT_FILE="${OUT_DIR}/05_wan22_animate.mp4"
curl -sS -L "${VID_URL}" -o "${OUT_FILE}"
echo "Saved video -> ${OUT_FILE}"
