# macOS Apple Silicon validation, 2026-11-18 / 2026-11-19

This records local Metal MPS validation on a single 64 GB Apple Silicon MacBook Pro
(`Mac16,6`, Apple M4 Max 40-core GPU, Metal 4, 64 GB Unified Memory), running macOS 26.0.1
(`25A362`), native arm64 Python 3.13.11, Node.js v22.22.3, PyTorch 2.15.0, Torchvision 0.29.2,
Diffusers 0.43.0, Transformers 5.23.0, Accelerate 1.16.0, ONNX Runtime 1.32.0, and ftfy 6.3.1.
It establishes results for this hardware/software combination only and does not certify smaller
unified-memory Macs (24/36/48 GB) or older macOS/PyTorch releases.

## Environment probes and automated checks

- Device probe: `compute_device = "mps"`, `compute_dtype = "torch.bfloat16"`,
  `mps_supports_bfloat16() = True`, `mps_supports_fp8(torch.float8_e4m3fn) = True`
  (packed FP8 weights stay resident on Metal at 1 byte/weight; single-block Wan 2.2
  microbenchmark on AC measured `909.7 ms` FP8 vs `876.8 ms` native BF16).
- Automated MPS suite (`tests/test_mps_runtime.py`, `tests/test_mps_native_components.py`,
  `tests/test_mps_runner_memory.py`): **73 passed** (`11.35 s`) on revision `ad6416a`.
- Expanded targeted regression suite after fixes (`test_mps_*`, `test_runtime_safety.py`,
  `test_minimax_*`, `test_wan_runtime_regressions.py`, `test_real_runtime.py`, `test_fal_api.py`):
  **116 passed, 1 skipped** (opt-in CUDA streaming test) on revision `be27516`.

## Issues found and fixed during Mac validation

1. **Missing `torchvision` dependency (`ad0d4d7`):** A fresh `.venv` following
   `docs/macos-validation.md` failed on Qwen-Image-2.1, MiniMax-H3 (`Qwen3VLVideoProcessor`),
   and Wan 2.2 Replace (`Sam2ImageProcessor`) because `pyproject.toml` omitted `torchvision`
   from `[project.optional-dependencies].inference`. Added `torchvision>=0.19.0` to the extra
   and added fail-fast checks before weight loading / ONNX pose extraction.
2. **Client sleep/socket resilience (`ad0d4d7`, `be27516`):** Clamshell sleep and a 1% battery
   hibernate exposed two client-harness gaps: timed-out jobs were not cancelled on the server,
   and a single keep-alive `write EPIPE` on wake aborted polling for a healthy running job.
   Added `PUT /requests/{id}/cancel` on timeout, transient socket retry with backoff in
   `tests/client/api/config.mjs`, and `caffeinate -dims` / AC-power / open-lid guidance.
3. **Unified-memory intra-run stage eviction (`be27516`):** Before this fix, keeping
   Qwen3-VL-32B (~25.3 GB) on CPU RAM during MiniMax-H3 Stage 2 DiT denoising (~20 GB) pushed
   `Ref2VA` into **44.3 GB of encrypted SSD swap** on 64 GB unified memory. Releasing Stage 1
   encoders before Stage 2 DiT loading on MPS (and releasing Wan's high-noise expert before
   low-noise), paired with `50/50` Metal DiT residency, kept system swap flat at **`~4.0 GB`**
   across back-to-back `FL2VA` I2V and `Ref2VA` runs.

## Real offline MPS generations (`seed=20261117`)

All generations ran locally with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`,
`OPENFAL_SYNTHETIC_FALLBACK` unset, `PYTORCH_ENABLE_MPS_FALLBACK` unset, `synthetic_mode: false`,
and `strata_engine.mode: "real_pytorch"`. Outputs were verified with `tests/client/inspect_outputs.py`
and visual frame inspection (plus stereo audio RMS checks on MiniMax video outputs).

| Endpoint | Output Spec | Active Compute | Result & Notes |
| --- | --- | ---: | --- |
| `engine_control` (`/health`, `/v1/engine/*`) | — | `0.4 s` | Passed |
| `fal-ai/qwen-image-2.1/turbo` | `1344×768` PNG, 8 steps | `66.5 s` | Passed; crisp Istanbul street scene |
| `fal-ai/qwen-image-2.1/edit` | `1024×1024` PNG, 2 refs, 8 steps | `116.5 s` | Passed; strong identity + lighting transfer |
| `fal-ai/minimax/h3/text-to-image` | `1344×768` PNG, 6 PDD steps | `141.1 s` | Passed; crisp cat/poodle Bosphorus scene |
| `fal-ai/minimax/h3/image-to-image` | `1024×1024` PNG, 1 ref, 6 PDD steps | `297.9 s` | Passed; faithful identity + oil-painting restyle |
| `fal-ai/minimax/h3/text-to-video` | `832×480`, 73f @ 24 fps, 30 steps, stereo audio | `45.1 min` (`2,705.8 s`) | Passed (`audio_rms=0.0127`); wall `7,168.8 s` due to battery sleep before AC |
| `fal-ai/minimax/h3/image-to-video` | `832×480`, 73f @ 24 fps, 30 steps, stereo audio | `53.2 min` (`3,193.5 s`) | Passed (`audio_rms=0.0680`), `50/50` Counter, ~4.0 GB swap |
| `fal-ai/wan/v2.2/recast` (replace) | `512×512`, 32f @ 16 fps, 20 steps (77f window) | `57.5 min` (`3,448.5 s`) | Passed; SAM2 CPU + Animate MPS, clean replacement |
| `fal-ai/wan/v2.2/animate` | `512×512`, 32f @ 16 fps, 20 steps (77f window) | `66.7 min` (`4,003.9 s`) | Completed on server (`openfal_vid_1bb6ad2eff.mp4`), frames verified; client hit 2h timeout during lid sleep |
| `fal-ai/wan/v2.2-a14b/image-to-video` | `848×480`, 29f @ 16 fps, 40 steps, CFG | `98.9 min` (`5,936.0 s`) | Passed; smooth motion from start keyframe |
| `fal-ai/wan/v2.2-a14b/text-to-video` | `848×480`, 29f @ 16 fps, 40 steps, CFG | `105.2 min` (`6,312.0 s`) | Passed; started on battery down to 2% before AC |
| `fal-ai/minimax/h3/reference-to-video` | `832×480`, 73f @ 24 fps, 2 image refs, 30 steps, stereo audio | `~4h 02m` | Completed on server (`openfal_vid_6f65b04427.mp4`, `audio_rms=0.0083`, `50/50` Counter, ~4.0 GB swap); exceeded 2h client deadline |
| `fal-ai/minimax/h3-max/recast` | `832×480`, 3s ref video + 2 image refs | Not run | Cancelled at 2h client timeout while queued behind `Ref2VA` |

### Why `Ref2VA` is ~4.7× slower than `FL2VA`

Upstream Diffusers' `MiniMaxH3Ref2VASetupStep` unconditionally upscales every reference image
to `reference_image_short_edge = 2048` with no area cap (`1254×1254` → `2048×2048` and
`1344×768` → `3584×2048`), expanding the single-stream sequence from ~8.7k tokens (`FL2VA`)
to ~23k+ tokens (`Ref2VA`). On the 40-core M4 Max at 30 steps, Stage 1 took `6m 23s` and
Stage 2 took `3h 55m` with zero sleep and flat ~4 GB swap. Validating `reference-to-video`
or `minimax/h3-max/recast` client-side on Apple Silicon requires raising
`OPENFAL_TEST_TIMEOUT_MS` above 2 hours (e.g., `21600000`–`28800000` ms).

## Remaining unvalidated macOS scope

- `fal-ai/minimax/h3-max/recast` (Ref2VA with a driving video + 2 image references).
- Multi-segment Wan Animate / Replace (>81 frames).
- 24 GB / 36 GB / 48 GB Apple Silicon Macs and older macOS/PyTorch builds without Metal BF16
  or native Metal FP8 storage.
- Direct listening check of MiniMax stereo audio quality (verified via container metadata and
  nonzero stereo RMS only).
