#!/usr/bin/env bash
# ==============================================================================
# 06: MiniMax-H3 33B FL2VA Text-to-Video + 32kHz Native Stereo Audio
# Endpoint: POST /minimax/h3/text-to-video (and /queue/minimax/h3/text-to-video)
# ==============================================================================
set -euo pipefail

OPENFAL_URL="${OPENFAL_URL:-http://192.168.1.47:8008}"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/outputs"
mkdir -p "${OUT_DIR}"

echo "=== [06] Submitting MiniMax-H3 33B Text-to-Video+Audio via Queue (${OPENFAL_URL}) ==="
SUBMIT_JSON="$(curl -sS -X POST "${OPENFAL_URL}/queue/minimax/h3/text-to-video" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A realistic cinematic video of a tabby cat running fast behind a fluffy apricot poodle down a sunlit cobblestone street in Istanbul, historic Ottoman wooden houses and Bosphorus ferry horn in the background, footsteps and playful barking",
    "resolution": "480P",
    "aspect_ratio": "16:9",
    "duration": 3.0,
    "enable_audio": true
  }')"

echo "${SUBMIT_JSON}" | python3 -m json.tool
REQ_ID="$(echo "${SUBMIT_JSON}" | python3 -c "import sys, json; print(json.load(sys.stdin)['request_id'])")"

echo "=== [06] Streaming live SSE logs for request_id=${REQ_ID} ==="
curl -sS -N "${OPENFAL_URL}/requests/${REQ_ID}/status/stream"

echo "=== [06] Fetching completed result for request_id=${REQ_ID} ==="
RESULT_JSON="$(curl -sS "${OPENFAL_URL}/requests/${REQ_ID}")"
echo "${RESULT_JSON}" | python3 -m json.tool

VID_URL="$(echo "${RESULT_JSON}" | python3 -c "import sys, json; print(json.load(sys.stdin)['video']['url'])")"
OUT_FILE="${OUT_DIR}/06_minimax_h3_t2v.mp4"
curl -sS -L "${VID_URL}" -o "${OUT_FILE}"
echo "Saved video with stereo audio -> ${OUT_FILE}"
