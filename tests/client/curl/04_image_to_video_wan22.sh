#!/usr/bin/env bash
# ==============================================================================
# 04: Alibaba Wan 2.2 Two-Expert MoE (27B Total / 14B Active) Image-to-Video
# Endpoint: POST /wan/v2.2/image-to-video (alias: /fal-ai/wan/v2.2-a14b/image-to-video)
# ==============================================================================
set -euo pipefail

OPENFAL_URL="${OPENFAL_URL:-http://127.0.0.1:8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLES_DIR="${SCRIPT_DIR}/../samples"
OUT_DIR="${SCRIPT_DIR}/outputs"
mkdir -p "${OUT_DIR}"

echo "=== [04] Uploading start keyframe sample_street_scene.png ==="
START_URL="$(curl -sS -X POST "${OPENFAL_URL}/v1/storage/upload" \
  -H "Content-Type: image/png" \
  --data-binary "@${SAMPLES_DIR}/sample_street_scene.png" | python3 -c "import sys, json; print(json.load(sys.stdin)['url'])")"
echo "  Start Image URL: ${START_URL}"

echo "=== [04] Wan 2.2 MoE Image-to-Video (${OPENFAL_URL}) ==="
RESP="$(curl -sS --max-time 1800 -X POST "${OPENFAL_URL}/wan/v2.2/image-to-video" \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"Camera glides forward down the sunlit cobblestone street as a cat chases a playful poodle toward the golden sunset\",
    \"start_image_url\": \"${START_URL}\",
    \"resolution\": \"480P\",
    \"aspect_ratio\": \"16:9\",
    \"duration\": 2.0,
    \"fps\": 16
  }")"

echo "${RESP}" | python3 -m json.tool

VID_URL="$(echo "${RESP}" | python3 -c "import sys, json; print(json.load(sys.stdin)['video']['url'])")"
OUT_FILE="${OUT_DIR}/04_wan22_i2v.mp4"
curl -sS -L "${VID_URL}" -o "${OUT_FILE}"
echo "Saved video -> ${OUT_FILE}"
