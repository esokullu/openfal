#!/usr/bin/env bash
# ==============================================================================
# 03: Alibaba Wan 2.2 Two-Expert MoE (27B Total / 14B Active) Text-to-Video
# Endpoint: POST /wan/v2.2/text-to-video (alias: /fal-ai/wan/v2.2-a14b/text-to-video)
# ==============================================================================
set -euo pipefail

OPENFAL_URL="${OPENFAL_URL:-http://127.0.0.1:8000}"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/outputs"
mkdir -p "${OUT_DIR}"

echo "=== [03] Wan 2.2 MoE Text-to-Video (${OPENFAL_URL}) ==="
RESP="$(curl -sS --max-time 1800 -X POST "${OPENFAL_URL}/wan/v2.2/text-to-video" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A realistic cinematic video of an orange tabby cat running fast behind a fluffy apricot poodle down a sunlit cobblestone street in Istanbul, warm golden hour lighting",
    "resolution": "480P",
    "aspect_ratio": "16:9",
    "duration": 2.0,
    "fps": 16
  }')"

echo "${RESP}" | python3 -m json.tool

VID_URL="$(echo "${RESP}" | python3 -c "import sys, json; print(json.load(sys.stdin)['video']['url'])")"
OUT_FILE="${OUT_DIR}/03_wan22_t2v.mp4"
curl -sS -L "${VID_URL}" -o "${OUT_FILE}"
echo "Saved video -> ${OUT_FILE}"
