#!/usr/bin/env bash
# ==============================================================================
# Run all openfal curl client suites sequentially against OPENFAL_URL
# Usage:
#   OPENFAL_URL=http://192.168.1.47:8008 ./run_all.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OPENFAL_URL="${OPENFAL_URL:-http://192.168.1.47:8008}"

for script in \
  "10_engine_telemetry_and_queue.sh" \
  "01_text_to_image_qwen.sh" \
  "02_image_edit_qwen.sh" \
  "03_text_to_video_wan22.sh" \
  "04_image_to_video_wan22.sh" \
  "05_animate_recast_wan22.sh" \
  "06_text_to_video_minimax_h3.sh" \
  "07_image_to_video_minimax_h3.sh" \
  "08_reference_to_video_minimax_h3.sh" \
  "09_text_and_image_minimax_h3_image.sh"; do
  echo ""
  echo "######################################################################"
  echo "# Running ${script}"
  echo "######################################################################"
  bash "${SCRIPT_DIR}/${script}"
done
