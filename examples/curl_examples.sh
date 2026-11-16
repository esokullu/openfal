#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${OPENFAL_URL:-http://localhost:8000}"

echo "=== 1. Check Strata 3-Tier Memory Telemetry ==="
curl -s "${BASE_URL}/v1/engine/status" | python3 -m json.tool | head -n 35

echo "=== 2. Switch Hardware Profile to 2x RTX 5060 Ti (24GB Total VRAM) ==="
curl -s -X POST "${BASE_URL}/v1/engine/profile" \
  -H "Content-Type: application/json" \
  -d '{"profile_id": "dual_5060ti_24gb"}' | python3 -m json.tool | head -n 20

echo "=== 3. Text-to-Video (fal.ai compatible) ==="
curl -s -X POST "${BASE_URL}/minimax/h3-max/text-to-video" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Aerial shot over turquoise waves crashing against volcanic cliffs, stereo ocean waves audio",
    "duration": 2,
    "resolution": "768P",
    "aspect_ratio": "16:9",
    "num_inference_steps": 4
  }' | python3 -m json.tool

echo "=== 4. Video Face Swap / Recast (Ref2VA) ==="
curl -s -X POST "${BASE_URL}/minimax/h3-max/recast" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Character speaking on a dimly lit film set",
    "reference_image_urls": ["https://example.com/face_ref.png"],
    "duration": 2,
    "resolution": "768P",
    "num_inference_steps": 4
  }' | python3 -m json.tool
