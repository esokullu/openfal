#!/usr/bin/env bash
# ==============================================================================
# 02: Alibaba Qwen-Image-2.1-Turbo Multi-Reference Image & Portrait Edit
# Endpoint: POST /qwen/image-2.1/edit (alias: /fal-ai/qwen-image-2.1/edit)
# Uploads sample_portrait.png and sample_style_ref.png via /v1/storage/upload
# ==============================================================================
set -euo pipefail

OPENFAL_URL="${OPENFAL_URL:-http://192.168.1.47:8008}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLES_DIR="${SCRIPT_DIR}/../samples"
OUT_DIR="${SCRIPT_DIR}/outputs"
mkdir -p "${OUT_DIR}"

echo "=== [02] Uploading sample reference images to ${OPENFAL_URL}/v1/storage/upload ==="
PORTRAIT_URL="$(curl -sS -X POST "${OPENFAL_URL}/v1/storage/upload" \
  -H "Content-Type: image/png" \
  --data-binary "@${SAMPLES_DIR}/sample_portrait.png" | python3 -c "import sys, json; print(json.load(sys.stdin)['url'])")"

STYLE_URL="$(curl -sS -X POST "${OPENFAL_URL}/v1/storage/upload" \
  -H "Content-Type: image/png" \
  --data-binary "@${SAMPLES_DIR}/sample_style_ref.png" | python3 -c "import sys, json; print(json.load(sys.stdin)['url'])")"

echo "  Portrait URL : ${PORTRAIT_URL}"
echo "  Style Ref URL: ${STYLE_URL}"

echo "=== [02] Qwen-Image-2.1-Turbo Multi-Reference Image Edit ==="
RESP="$(curl -sS --max-time 600 -X POST "${OPENFAL_URL}/qwen/image-2.1/edit" \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"Transform the character portrait from the first image into a neon cyberpunk detective on a rainy Istanbul street matching the lighting of the second reference image\",
    \"reference_image_urls\": [\"${PORTRAIT_URL}\", \"${STYLE_URL}\"],
    \"resolution\": \"768P\",
    \"aspect_ratio\": \"1:1\"
  }")"

echo "${RESP}" | python3 -m json.tool

IMG_URL="$(echo "${RESP}" | python3 -c "import sys, json; print(json.load(sys.stdin)['images'][0]['url'])")"
OUT_FILE="${OUT_DIR}/02_qwen_edit.png"
curl -sS -L "${IMG_URL}" -o "${OUT_FILE}"
echo "Saved edited image -> ${OUT_FILE}"
