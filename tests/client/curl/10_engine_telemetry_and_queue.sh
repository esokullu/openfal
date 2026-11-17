#!/usr/bin/env bash
# ==============================================================================
# 10: Strata-DiT Engine Telemetry, Model Status, Hardware Profiles & Sleep/Wake
# Endpoints:
#   - GET  /health
#   - GET  /v1/endpoints
#   - GET  /v1/engine/status
#   - GET  /v1/engine/profiles
#   - GET  /v1/models/status
#   - GET  /v1/models/hardware-detect
#   - POST /v1/engine/sleep
#   - POST /v1/engine/wake
# ==============================================================================
set -euo pipefail

OPENFAL_URL="${OPENFAL_URL:-http://127.0.0.1:8000}"

echo "=== [10.1] GET /health ==="
curl -sS "${OPENFAL_URL}/health" | python3 -m json.tool

echo "=== [10.2] GET /v1/engine/status ==="
curl -sS "${OPENFAL_URL}/v1/engine/status" | python3 -m json.tool

echo "=== [10.3] GET /v1/models/hardware-detect ==="
curl -sS "${OPENFAL_URL}/v1/models/hardware-detect" | python3 -m json.tool

echo "=== [10.4] GET /v1/endpoints (count) ==="
curl -sS "${OPENFAL_URL}/v1/endpoints" | python3 -c "import sys, json; eps = json.load(sys.stdin)['endpoints']; print(f'Total supported endpoints: {len(eps)}'); [print(' -', e['id'], '->', e['backend'], '/', e['partition']) for e in eps]"

echo "=== [10.5] POST /v1/engine/sleep & POST /v1/engine/wake ==="
curl -sS -X POST "${OPENFAL_URL}/v1/engine/sleep" | python3 -c "import sys, json; d = json.load(sys.stdin); print('Sleep status:', d['status'])"
curl -sS -X POST "${OPENFAL_URL}/v1/engine/wake" | python3 -c "import sys, json; d = json.load(sys.stdin); print('Wake status :', d['status'])"
