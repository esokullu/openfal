# Inference validation, 2026-11-17

This records tests on Windows with an RTX 5090 (32 GB), 256 GB host RAM,
PyTorch 2.8.0+cu128, Diffusers 0.43.0 and Transformers 5.23.0. It is not a
certification of other GPUs or Apple Silicon hardware.

## Automated checks

- Python suite with `OPENFAL_TEST_CUDA=1`: 244 passed, 20 skipped. All skips
  require actual Apple Metal hardware.
- Studio: 13 Node tests passed, including both Animate modes and failed uploads.
- Four Bash/curl integration tests are included in the Python total.
- Wheel build passed; its package includes the complete Animate asset manifest.
- Native versus staged Animate inference matches exactly across three segments,
  both modes, BF16 encoders/DiT, FP32 VAE, FP8 weights, and zero/one resident
  streamed blocks, on CPU and CUDA. This verifies adapter behavior; it does not
  establish perceptual quality for arbitrary inputs or segment sizes.

## Setup and real generation

Explicit setup installed and verified 403 supporting files (4,770,524,030 bytes):
CLIP, YOLOv10m, whole-body ViTPose-H and SAM2. An offline repeat with download
functions blocked succeeded with no missing assets. Initial real Animate and
replacement requests used `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

Animation produced 32 frames at 512x512 and 16 fps in 222.869 seconds. Final
cross-identity replacement produced 96 frames (six seconds) in 452.45 seconds,
including preprocessing. Both used 20 steps, guidance 1 and the native 77-frame
segment length. Decoded frame count, dimensions, file size and duration matched
the API responses. Reviewed first, middle and final frames show coherent faces,
blinking and expression changes. Replacement also passed inspection across the
segment boundary (frames 76–78), retaining the new character's hair and face
without the earlier silhouette halo. This is generative conditioning, not a
guarantee of pixel-exact compositing or artifact-free output on every input.

Real whole-body/face preprocessing passed on the upstream animation and
replacement examples and the fictional portrait fixture. Real SAM2 masks followed
the subject, with correct foreground/background polarity. No model downloads
occurred during those offline requests.

Supplementary cross-identity tests exposed a missing upstream preprocessing
step: the dilated subject mask must be expanded to its bounding rectangle.
A tight silhouette caused visible borders and constrained the replacement to the
old character's outline. Five real masks now match upstream augmentation exactly,
and focused mask/background regression tests pass. The corrected standard-length
multi-segment render passed visual inspection. A two-segment stress test with
17-frame internal windows still produced visible corruption after the mask fix.
Production requests therefore enforce the native 77-frame window and reject that
unsupported override before preprocessing. Output duration remains independent:
short clips are padded internally and trimmed, while long clips use multiple
native windows. The API still supports both animation and replacement.

All six regenerated regression artifacts (Qwen text/edit, MiniMax text/edit,
Wan text-to-video and MiniMax image-to-video) match the earlier validated files
byte for byte. MP4 dimensions, frame counts, durations and audio channels match
the API responses. MiniMax's stereo audio decodes and is non-silent; listening
quality has not been assessed in this run.

Local artifacts are under
`tests/client/api/outputs/wan-animate-2026-11-17/`, with request/response manifests,
`media_checks.json`, frame previews, regression hashes and supplementary runs.
The driving fixture is a fictional adult portrait generated locally by MiniMax;
its full source is 56 frames at 24 fps. The long boundary-test source explicitly
loops that fixture three times; it is not presented as a naturally continuous
performance. Its API request asks for 96 frames over six seconds.

## Apple Silicon scope

Portable tests verify BF16 capability failures and FP32 fallback, recursive
FP8/INT8 storage preservation, CPU decoding for unsupported Metal FP8 formats,
safe cleanup, and release of inactive models from unified RAM. Genuine Metal
tests also cover streaming, Qwen position encoding, MiniMax position dtype
conversion, the Wan VAE and Animate motion encoding. They were skipped here.

Actual Mac full-model inference, numerical stability, output quality, peak unified
memory, swapping and sustained performance remain unverified. CPU offload does
not add a separate memory pool on Apple Silicon. Follow the
[Mac validation procedure](macos-validation.md) on each target configuration;
do not count a Mac client talking to this Windows server as MPS validation.
