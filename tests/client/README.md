# openfal Client Test Suite (`curl` & `fal.ai` Node.js API)

End-to-end client scripts exercising representative task endpoints across Qwen-Image-2.1, Wan 2.2 and MiniMax-H3.

`node api/run_all.mjs` runs each model separately, continues after failures, writes
`api/outputs/suite_results.json`, and exits nonzero if any case fails. Generation
uses a fixed seed (`OPENFAL_TEST_SEED`, default `20261117`) and a 30-minute polling
deadline (`OPENFAL_TEST_TIMEOUT_MS`). Inspect the media as well as the exit status.
Each suite run gets a fresh timestamped output directory so a failed request cannot
reuse an older artifact. Request parameters and response metadata are saved beside
the media; `OPENFAL_OUTPUTS_DIR` selects an explicit directory when needed.
Wan Animate and character replacement exercise real pose/face-motion conditioning;
replacement also uses SAM2 background segmentation. Each request requires one
identity image and a driving video. Install the complete local supporting assets
before testing (`openfal download --component wan22_animate --runtime-assets-only`
repairs an older installation without downloading its DiT again).

The edit/reference tests use photographic portrait, lighting and street fixtures.
Wan replacement uses a different fictional identity from the driver so mask and
outline failures are visible. Override it with `OPENFAL_REPLACEMENT_PORTRAIT`;
`OPENFAL_PORTRAIT` remains the fallback override for both portrait fixtures.
The original drawings remain available via `OPENFAL_PORTRAIT`, `OPENFAL_STYLE`, and
`OPENFAL_STREET`; see `samples/README.md` for provenance and visual evaluation criteria.
MiniMax Recast requires a driving video produced by the
preceding MiniMax text-to-video test; native Ref2VA editing is generative and does
not guarantee exact preservation of the driving video's motion or scene.

Wan `wan/v2.2/animate` defaults to `animate_mode="animate"`; `wan/v2.2/recast`
defaults to `animate_mode="replace"`. Both use `face_image_url` for the identity
and `driving_video_url` for the performance (`target_video_url` and `video_url`
are accepted aliases). Native defaults are 20 steps and guidance 1. Long driving
sequences use segmented generation rather than an 81-frame total cap, and output
does not extend beyond the actual driver duration. Inspect identity, body/hand
pose, facial motion, segment continuity and replacement background boundaries.
Host tests and asset completeness checks do not establish visual quality; keep
the saved inputs, request metadata and rendered outputs for that review.
The Wan client defaults to `samples/sample_driving_portrait.mp4`; set
`OPENFAL_DRIVING_VIDEO` to a local video path to evaluate another performance.
Use a clearly visible person and review the input's pose and face visibility.

By default, all scripts target `OPENFAL_URL=http://127.0.0.1:8000`. Override with:

```bash
export OPENFAL_URL="http://localhost:8008"
```

## Directory Structure

```text
tests/client/
├── inspect_outputs.py                         # Decode saved outputs and verify response metadata
├── samples/                                    # Sample input images for I2V, Edit, Ref2VA & Recast
│   ├── sample_portrait_photo.png               # Default fictional photographic identity
│   ├── sample_replacement_photo.png            # Different identity to expose replacement-mask failures
│   ├── sample_street_photo.png                 # Default photographic street scene
│   ├── sample_style_photo.png                  # Default photographic lighting reference
│   ├── sample_driving_portrait.mp4             # Wan Animate/Replacement driving clip
│   ├── sample_portrait.png                     # Optional drawn portrait smoke test
│   ├── sample_street_scene.png                 # Optional geometric scene smoke test
│   └── sample_style_ref.png                    # Optional neon-grid smoke test
├── curl/                                       # Shell curl test scripts (outputs -> curl/outputs/)
│   ├── 01_text_to_image_qwen.sh                # POST /qwen/image-2.1/text-to-image
│   ├── 02_image_edit_qwen.sh                   # POST /qwen/image-2.1/edit (multi-reference)
│   ├── 03_text_to_video_wan22.sh               # POST /wan/v2.2/text-to-video (27B-A14B MoE)
│   ├── 04_image_to_video_wan22.sh              # POST /wan/v2.2/image-to-video (27B-A14B MoE)
│   ├── 05_animate_recast_wan22.sh              # POST /wan/v2.2/animate (Wan2.2-Animate-14B)
│   ├── 06_text_to_video_minimax_h3.sh          # POST /queue/minimax/h3/text-to-video + SSE stream
│   ├── 07_image_to_video_minimax_h3.sh         # POST /minimax/h3/image-to-video (FL2VA + stereo audio)
│   ├── 08_reference_to_video_minimax_h3.sh     # POST /minimax/h3/reference-to-video (Ref2VA omni)
│   ├── 09_text_and_image_minimax_h3_image.sh   # POST /minimax/h3/text-to-image & image-to-image
│   ├── 10_engine_telemetry_and_queue.sh        # GET /health, /v1/engine/status, /v1/models/status
│   └── run_all.sh                              # Run all curl tests sequentially
└── api/                                        # Node.js @fal-ai/client scripts (outputs -> api/outputs/)
    ├── package.json
    ├── config.mjs                              # Shared fal client + sample uploader + downloader
    ├── engine_control.mjs                      # Engine telemetry, hardware detect, sleep/wake
    ├── text_to_image.mjs                       # Qwen-Image-2.1-Turbo & MiniMax-H3-Image T2I
    ├── image_to_image_edit.mjs                 # Qwen-Image-2.1 Edit & MiniMax-H3 Image-to-Image
    ├── text_to_video.mjs                       # Wan 2.2 MoE T2V & MiniMax-H3 T2V+Audio
    ├── image_to_video.mjs                      # Wan 2.2 MoE I2V & MiniMax-H3 I2V+Audio
    ├── reference_to_video.mjs                  # MiniMax-H3 Ref2VA multi-reference video+audio
    ├── video_recast_animate.mjs                # Wan 2.2 Animate & MiniMax-H3 Recast
    └── run_all.mjs                             # Run all Node.js tests sequentially
```

## Running `curl` Scripts

```bash
cd tests/client/curl
./10_engine_telemetry_and_queue.sh
./01_text_to_image_qwen.sh
./06_text_to_video_minimax_h3.sh
OPENFAL_ANIMATE_MODE=animate ./05_animate_recast_wan22.sh
OPENFAL_ANIMATE_MODE=replace ./05_animate_recast_wan22.sh
```

The Wan curl example uploads the photographic portrait and driving clip, uses a
fixed seed (`OPENFAL_TEST_SEED`, default `20261117`), and saves both JSON manifests
and the result video. `OPENFAL_PORTRAIT` and `OPENFAL_DRIVING_VIDEO` accept a local
path or sample filename; `OPENFAL_OUTPUTS_DIR` overrides its output directory.
HTTP failures exit nonzero and show the response body (`curl --fail-with-body`
requires curl 7.76+).

## Running Node.js `fal.ai` Scripts

Works out-of-the-box with Node.js 18+ (and also supports `npm install` for `@fal-ai/client`):

```bash
cd tests/client/api
node engine_control.mjs
node text_to_image.mjs --model=qwen
node text_to_video.mjs --model=minimax
node text_to_video.mjs --model=wan
node image_to_video.mjs
node reference_to_video.mjs
node video_recast_animate.mjs
node video_recast_animate.mjs --model=wan --mode=animate
node video_recast_animate.mjs --model=wan --mode=replace
```

After a run, decode every saved output and check dimensions, frame counts,
durations, file sizes and stereo audio against its response manifest:

```bash
# From the repository root, using the Python environment with inference dependencies:
python tests/client/inspect_outputs.py tests/client/api/outputs/run-<timestamp>
```

This writes `media_checks.json` and first/middle/last video frames in `review/`.
Inspect those frames and play the videos to assess identity, motion, prompt
adherence and audio quality. A decodable file does not establish visual quality.
