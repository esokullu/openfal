#!/usr/bin/env bash
# ==============================================================================
# Idempotent GPU Pod Bootstrap for openfal vs ComfyUI vs vLLM-Omni
# Works on RunPod PyTorch/CUDA pods or any fresh Ubuntu/CUDA host.
#
# Usage:
#   bash benchmark/setup_pod.sh
#   bash benchmark/setup_pod.sh --skip-vllm        # setup only openfal + ComfyUI
#   bash benchmark/setup_pod.sh --skip-download    # skip openfal model downloads
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

SKIP_COMFYUI=0
SKIP_VLLM=0
SKIP_DOWNLOAD=0
for arg in "$@"; do
  case "${arg}" in
    --skip-comfyui) SKIP_COMFYUI=1 ;;
    --skip-vllm) SKIP_VLLM=1 ;;
    --skip-download) SKIP_DOWNLOAD=1 ;;
    *)
      echo "Unknown option: ${arg}" >&2
      exit 1
      ;;
  esac
done

BENCH_WORKSPACE="${BENCH_WORKSPACE:-/workspace/openfal_bench}"
OPENFAL_REPO_URL="${OPENFAL_REPO_URL:-https://github.com/esokullu/openfal.git}"
OPENFAL_BRANCH="${OPENFAL_BRANCH:-main}"
OPENFAL_DIR="${OPENFAL_DIR:-${BENCH_WORKSPACE}/openfal}"
OPENFAL_WEIGHTS_DIR="${OPENFAL_WEIGHTS_DIR:-${BENCH_WORKSPACE}/weights/MiniMax-H3}"
COMFYUI_DIR="${COMFYUI_DIR:-${BENCH_WORKSPACE}/ComfyUI}"
HF_HOME="${HF_HOME:-${BENCH_WORKSPACE}/hf_cache}"
OPENFAL_DOWNLOAD_BUNDLE="${OPENFAL_DOWNLOAD_BUNDLE:-all}"

export HF_HOME
mkdir -p "${BENCH_WORKSPACE}" "${OPENFAL_WEIGHTS_DIR}" "${HF_HOME}"

echo "=== [1/5] GPU & Python Environment ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true
python3 --version

if command -v apt-get >/dev/null 2>&1 && ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Installing system packages (git, ffmpeg, tmux)..."
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git ffmpeg tmux curl
fi

# Prefer active repo if setup_pod.sh is already running from inside a checkout
if [ -f "${SCRIPT_DIR}/../pyproject.toml" ]; then
  OPENFAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

echo "=== [2/5] Installing openfal[inference] from ${OPENFAL_DIR} ==="
if [ ! -d "${OPENFAL_DIR}/.git" ]; then
  git clone --branch "${OPENFAL_BRANCH}" "${OPENFAL_REPO_URL}" "${OPENFAL_DIR}"
fi
python3 -m pip install -q --upgrade pip
python3 -m pip install -q -e "${OPENFAL_DIR}[inference]"

if [ "${SKIP_COMFYUI}" -eq 0 ]; then
  echo "=== [3/5] Installing ComfyUI -> ${COMFYUI_DIR} ==="
  if [ ! -d "${COMFYUI_DIR}/.git" ]; then
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "${COMFYUI_DIR}"
  else
    git -C "${COMFYUI_DIR}" pull --ff-only || true
  fi
  python3 -m pip install -q -r "${COMFYUI_DIR}/requirements.txt"
fi

if [ "${SKIP_VLLM}" -eq 0 ]; then
  echo "=== [4/5] Installing vllm & vllm-omni ==="
  python3 -m pip install -q --upgrade vllm vllm-omni || {
    echo "WARNING: pip install vllm vllm-omni failed; re-run manually or pass --skip-vllm" >&2
  }
fi

if [ "${SKIP_DOWNLOAD}" -eq 0 ]; then
  echo "=== [5/5] Downloading openfal weights (${OPENFAL_DOWNLOAD_BUNDLE}, fp8_e4m3) -> ${OPENFAL_WEIGHTS_DIR} ==="
  openfal detect --weights-dir "${OPENFAL_WEIGHTS_DIR}" || true
  openfal download \
    --weights-dir "${OPENFAL_WEIGHTS_DIR}" \
    --component "${OPENFAL_DOWNLOAD_BUNDLE}" \
    --quant fp8_e4m3 \
    ${HF_TOKEN:+--hf-token "${HF_TOKEN}"}
fi

if [ "${SKIP_COMFYUI}" -eq 0 ] && [ -d "${COMFYUI_DIR}" ]; then
  echo "Symlinking shared Comfy-Org .safetensors weights -> ${COMFYUI_DIR}/models/ ..."
  mkdir -p \
    "${COMFYUI_DIR}/models/diffusion_models" \
    "${COMFYUI_DIR}/models/text_encoders" \
    "${COMFYUI_DIR}/models/vae"

  while IFS= read -r -d '' sf_file; do
    base="$(basename "${sf_file}")"
    case "${sf_file}" in
      */diffusion_models/*|*/transformer/*)
        ln -sf "${sf_file}" "${COMFYUI_DIR}/models/diffusion_models/${base}" ;;
      */text_encoders/*|*/text_encoder/*)
        ln -sf "${sf_file}" "${COMFYUI_DIR}/models/text_encoders/${base}" ;;
      */vae/*)
        ln -sf "${sf_file}" "${COMFYUI_DIR}/models/vae/${base}" ;;
      *)
        ;;
    esac
  done < <(find "${OPENFAL_WEIGHTS_DIR}" -type f -name "*.safetensors" -print0)
fi

echo ""
echo "Pod bootstrap complete!"
echo "Run sequential GPU-isolated benchmark:"
echo "  cd ${OPENFAL_DIR} && python3 -m benchmark.bench_runner --sequential-local"
