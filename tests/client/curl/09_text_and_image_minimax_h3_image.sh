#!/usr/bin/env bash
# ==============================================================================
# 09: MiniMax-H3-Image (33B 6-Step PDD Distilled Text-to-Image & Image-to-Image)
# Endpoints:
#   - POST /minimax/h3/text-to-image
#   - POST /minimax/h3/image-to-image
# ==============================================================================
set -euo pipefail

OPENFAL_URL="${OPENFAL_URL:-http://192.168.1.47:8008}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLES_DIR="${SCRIPT_DIR}/../samples"
OUT_DIR="${SCRIPT_DIR}/outputs"
mkdir -p "${OUT_DIR}"

echo "=== [09a] MiniMax-H3-Image Text-to-Image (${OPENFAL_URL}) ==="
RESP_T2I="$(curl -sS --max-time 600 -X POST "${OPENFAL_URL}/minimax/h3/text-to-image" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A photorealistic wide shot of an orange tabby cat chasing a white poodle across an Istanbul cobblestone alley at sunset, Bosphorus in background",
    "resolution": "768P",
    "aspect_ratio": "16:9"
  }')"

echo "${RESP_T2I}" | python3 -m json.tool
IMG_T2I_URL="$(echo "${RESP_T2I}" | python3 -c "import sys, json; print(json.load(sys.stdin)['images'][0]['url'])")"
curl -sS -L "${IMG_T2I_URL}" -o "${OUT_DIR}/09a_minimax_h3_t2i.png"
echo "Saved -> ${OUT_DIR}/09a_minimax_h3_t2i.png"

echo "=== [09b] Uploading sample_portrait.png for MiniMax-H3-Image Image-to-Image ==="
PORTRAIT_URL="$(curl -sS -X POST "${OPENFAL_URL}/v1/storage/upload" \
  -H "Content-Type: image/png" \
  --data-binary "@${SAMPLES_DIR}/sample_portrait.png" | python3 -c "import sys, json; print(json.load(sys.stdin)['url'])")"

RESP_I2I="$(curl -sS --max-time 600 -X POST "${OPENFAL_URL}/minimax/h3/image-to-image" \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"Restyle the person in <Picture 1> as a cinematic oil painting with dramatic golden rim lighting\",
    \"reference_image_urls\": [\"${PORTRAIT_URL}\"],
    \"resolution\": \"768P\",
    \"aspect_ratio\": \"1:1\"
  }")"

echo "${RESP_I2I}" | python3 -m json.tool
IMG_I2I_URL="$(echo "${RESP_I2I}" | python3 -c "import sys, json; print(json.load(sys.stdin)['images'][0]['url'])")"
curl -sS -L "${IMG_I2I_URL}" -o "${OUT_DIR}/09b_minimax_h3_i2i.png"
echo "Saved -> ${OUT_DIR}/09b_minimax_h3_i2i.png"
