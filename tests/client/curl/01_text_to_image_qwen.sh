#!/usr/bin/env bash
# ==============================================================================
# 01: Alibaba Qwen-Image-2.1-Turbo (7B, 8-Step Text-to-Image)
# Endpoint: POST /qwen/image-2.1/text-to-image (alias: /fal-ai/qwen-image-2.1/turbo)
# ==============================================================================
set -euo pipefail

OPENFAL_URL="${OPENFAL_URL:-http://192.168.1.47:8008}"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/outputs"
mkdir -p "${OUT_DIR}"

echo "=== [01] Qwen-Image-2.1-Turbo Text-to-Image (${OPENFAL_URL}) ==="
RESP="$(curl -sS --max-time 600 -X POST "${OPENFAL_URL}/qwen/image-2.1/text-to-image" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A realistic photo of a tabby cat running after a fluffy apricot poodle in sunlit Istanbul cobblestone streets, Galata Tower in the distance",
    "resolution": "768P",
    "aspect_ratio": "16:9"
  }')"

echo "${RESP}" | python3 -m json.tool

IMG_URL="$(echo "${RESP}" | python3 -c "import sys, json; print(json.load(sys.stdin)['images'][0]['url'])")"
OUT_FILE="${OUT_DIR}/01_qwen_t2i.png"
curl -sS -L "${IMG_URL}" -o "${OUT_FILE}"
echo "Saved image -> ${OUT_FILE}"
