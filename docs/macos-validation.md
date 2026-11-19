# Validate local inference on macOS

This is a validation procedure, not a record of a successful Mac run. The current
development work was performed on Windows. Passing host tests there does not
establish MPS execution, Mac memory capacity, output quality or performance.
Running the Node client on a Mac against a Windows server only tests the client;
the procedure below runs the server on the Mac itself.

## Install without CUDA

Use native arm64 Python on Apple Silicon, Python 3.10+ (3.11 is a useful baseline),
and Node.js 18+. Choose a macOS version supported by the installed PyTorch wheel;
check [PyTorch's MPS requirements](https://docs.pytorch.org/docs/stable/notes/mps.html).
Wan preprocessing's [ONNX Runtime 1.23.2 arm64 wheel](https://pypi.org/project/onnxruntime/1.23.2/#files)
requires macOS 13+, and newer dependencies can impose a newer OS minimum.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
OPENFAL_BUILD_CUDA=0 python -m pip install -e ".[inference,dev]"
unset OPENFAL_SYNTHETIC_FALLBACK OPENFAL_ALLOW_SYNTHETIC_TESTS
unset PYTORCH_ENABLE_MPS_FALLBACK
export OPENFAL_WEIGHTS_DIR="$PWD/weights/MiniMax-H3"
export OPENFAL_VALIDATION_DIR="$PWD/.cache/macos-validation-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OPENFAL_VALIDATION_DIR"
```

CUDA, `nvcc`, and the optional native CUDA extension are not needed. Do not
interpret CPU execution or synthetic output as a successful MPS run. This guide
leaves PyTorch's optional unsupported-operation CPU fallback disabled so missing
Metal operations remain visible during validation.

## Capture the environment and run MPS tests

Keep the exact revision, dependency versions and hardware details with the results:

```bash
git rev-parse HEAD > "$OPENFAL_VALIDATION_DIR/revision.txt"
git diff --stat > "$OPENFAL_VALIDATION_DIR/local-changes.txt"
sw_vers > "$OPENFAL_VALIDATION_DIR/macos-version.txt"
sysctl hw.model hw.memsize > "$OPENFAL_VALIDATION_DIR/hardware.txt"
uname -m >> "$OPENFAL_VALIDATION_DIR/hardware.txt"
node --version > "$OPENFAL_VALIDATION_DIR/node-version.txt"
python -m pip freeze > "$OPENFAL_VALIDATION_DIR/python-packages.txt"
python - <<'PY'
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import torch
from openfal.runtime.common import resolve_compute_device_and_dtype

device, dtype = resolve_compute_device_and_dtype()
report = {
    "python": platform.python_version(),
    "architecture": platform.machine(),
    "macos": platform.mac_ver()[0],
    "mps_built": torch.backends.mps.is_built(),
    "mps_available": torch.backends.mps.is_available(),
    "compute_device": str(device),
    "compute_dtype": str(dtype),
    "packages": {name: importlib.metadata.version(name) for name in (
        "torch", "torchvision", "diffusers", "transformers", "accelerate", "onnxruntime", "ftfy",
    )},
}
Path(os.environ["OPENFAL_VALIDATION_DIR"], "runtime.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8",
)
print(json.dumps(report, indent=2))
assert report["architecture"] == "arm64", "Use native arm64 Python rather than Rosetta"
assert device.type == "mps", "No usable MPS device: this run cannot validate Metal inference"
PY

set -o pipefail
python -m pytest tests/test_mps_runtime.py tests/test_mps_native_components.py \
  tests/test_mps_runner_memory.py -q -rs \
  --junitxml="$OPENFAL_VALIDATION_DIR/mps-tests.xml" \
  2>&1 | tee "$OPENFAL_VALIDATION_DIR/mps-tests.log"
```

These tests cover portable quantization behavior, native model components,
inactive model memory release, and real Metal execution where available.
Inspect skip reasons: tests skipped because Apple hardware is unavailable provide
no Metal coverage. Record failures, skips and the selected compute dtype.
BF16 is selected only after an operation probe succeeds; otherwise the runtime
uses FP32. Unsupported Metal FP8 storage keeps packed weights on CPU while
transferring decoded compute tensors to MPS.

CPU and MPS allocations share unified memory. Budget for the active model's
loaded weights, text/vision encoders, VAEs, preprocessing models and working
tensors. CPU offload does not add a separate RAM pool, and FP32 increases memory
use. A 24/36/64 GB profile name does not establish that every model fits. Record
Activity Monitor memory pressure and swap before, during and after each model;
Studio's memory estimates are not measurements of this usage.

## Complete setup, then serve locally

The full client suite needs every component. Downloading all components uses
approximately 186.7 GB of disk under the current installer estimates. If the Mac
cannot accommodate the complete workload, validate selected models and label the
report as partial instead of claiming a full-suite result.

```bash
openfal download --component all --quant fp8_e4m3 --weights-dir "$OPENFAL_WEIGHTS_DIR"
openfal models --weights-dir "$OPENFAL_WEIGHTS_DIR" > "$OPENFAL_VALIDATION_DIR/models.txt"
```

For existing checkpoints, `openfal download --component all --runtime-assets-only`
repairs missing metadata and supporting models. It does not install missing DiTs.
Animate additionally needs the complete pinned CLIP, detector, pose and SAM2 asset
set described in the [setup guide](../README.md#complete-setup-before-inference).

Start the server in a second terminal from the same repository. Activate the same
virtual environment and use the same weights path. Keep the Mac plugged into
full-wattage AC power with the lid open (or clamshell with an external display)
and wrap the server with `caffeinate -dims` so idle sleep does not pause Metal
compute or trip wall-clock client timeouts. Keep its output as `server.log` in
the validation directory if collecting a shareable report.

```bash
source .venv/bin/activate
unset OPENFAL_SYNTHETIC_FALLBACK PYTORCH_ENABLE_MPS_FALLBACK
export OPENFAL_WEIGHTS_DIR="$PWD/weights/MiniMax-H3"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  caffeinate -dims openfal serve --host 127.0.0.1 --port 8008 --weights-dir "$OPENFAL_WEIGHTS_DIR"
```

The offline flags make missing model assets fail during this verification instead
of being fetched. Input uploads and local result downloads still use HTTP. Verify
that `/health` reports `synthetic_mode: false` and that generation responses report
`strata_engine.mode: "real_pytorch"`.

## Run and inspect the real client suite

In the first terminal, with the validation variables still set (`OPENFAL_TEST_TIMEOUT_MS`
is an explicit test deadline, not a promised runtime):

```bash
export OPENFAL_URL="http://127.0.0.1:8008"
export OPENFAL_OUTPUTS_DIR="$OPENFAL_VALIDATION_DIR/media"
export OPENFAL_TEST_SEED=20261117
export OPENFAL_TEST_TIMEOUT_MS=7200000
set -o pipefail
caffeinate -dims node tests/client/api/run_all.mjs 2>&1 | tee "$OPENFAL_VALIDATION_DIR/client-suite.log"
python tests/client/inspect_outputs.py "$OPENFAL_OUTPUTS_DIR"
```

The suite records individual failures and runs both Wan Animate and replacement.
Run the inspector even if the suite exits nonzero so successful artifacts and
partial failures remain reviewable. Keep `suite_results.json`, request/response
manifests, `media_checks.json`, review frames, and source videos. Play every video
and listen to MiniMax audio. Check image coherence, identity, motion, duration,
hand/face pose, replacement boundaries and continuity between Animate segments.
A multi-segment check needs a sufficiently long visible-person performance and a
request exceeding 81 frames, for example `duration: 6` at `fps: 16`. Set
`OPENFAL_DRIVING_VIDEO` to the source path and submit those duration/FPS values
through the API or Studio. The shipped client cases request two seconds; changing
the driver alone does not lengthen their requests, and the normal short portrait
fixture does not cover multiple segments.

The curl example supports the same two modes:

```bash
OPENFAL_ANIMATE_MODE=animate bash tests/client/curl/05_animate_recast_wan22.sh
OPENFAL_ANIMATE_MODE=replace bash tests/client/curl/05_animate_recast_wan22.sh
```

## What a report establishes

Report the exact Mac model, RAM, macOS/Python/PyTorch versions, compute dtype,
revision, skipped/failed tests, model assets, inference parameters, elapsed time,
observed memory pressure/swap and inspected outputs. A successful case establishes
that combination only. Actual Mac full-model inference, visual/audio quality,
large-model capacity, multi-segment replacement and sustained runtime have not been
validated by the Windows host tests. Intel Mac CPU fallback is also outside the
Apple Silicon MPS validation described here.
