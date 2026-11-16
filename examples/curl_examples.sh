#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${OPENFAL_URL:-http://localhost:8000}"

echo "=== 1. Check Strata 3-Tier Memory Telemetry ==="
curl -s "${BASE_URL}/v1/engine/status" | python3 -m json.tool | head -n 30

echo "=== 2. [Alibaba Qwen-Image-2.1-Turbo] 7B Fast Text-to-Image ==="
curl -s -X POST "${BASE_URL}/qwen/image-2.1/text-to-image" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Minimalist glass cafe in Tokyo with glowing neon sign OPENFAL",
    "resolution": "2K",
    "aspect_ratio": "16:9",
    "num_inference_steps": 4
  }' | python3 -m json.tool

echo "=== 3. [Alibaba Wan 2.2 MoE] 27B-A14B Fast Text-to-Video ==="
curl -s -X POST "${BASE_URL}/wan/v2.2/text-to-video" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Aerial FPV shot over turquoise waves crashing against volcanic cliffs",
    "duration": 2,
    "resolution": "768P",
    "aspect_ratio": "16:9",
    "num_inference_steps": 4
  }' | python3 -m json.tool

echo "=== 4. [Alibaba Wan 2.2 Animate 14B] Fast Video Face Swap ==="
curl -s -X POST "${BASE_URL}/wan/v2.2/animate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Character speaking on a dimly lit film set",
    "face_image_url": "https://example.com/face_ref.png",
    "duration": 2,
    "resolution": "768P",
    "num_inference_steps": 4
  }' | python3 -m json.tool

echo "=== 5. [Optional MiniMax-H3] 33B Omni Video + Native Stereo Audio ==="
curl -s -X POST "${BASE_URL}/minimax/h3-max/text-to-video" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Steam train crossing a snowy bridge at night, whistle and chugging stereo audio",
    "duration": 2,
    "resolution": "768P",
    "num_inference_steps": 4
  }' | python3 -m json.tool
