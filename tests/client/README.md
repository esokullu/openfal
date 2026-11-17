# openfal Client Test Suite (`curl` & `fal.ai` Node.js API)

End-to-end client scripts exercising every engine (`Qwen-Image-2.1-Turbo`, `Wan 2.2 MoE`, and `MiniMax-H3`) and every task endpoint supported by `openfal`.

By default, all scripts target `OPENFAL_URL=http://192.168.1.47:8008`. Override with:

```bash
export OPENFAL_URL="http://localhost:8008"
```

## Directory Structure

```text
tests/client/
├── samples/                                    # Sample input images for I2V, Edit, Ref2VA & Recast
│   ├── sample_portrait.png                     # 768x768 character portrait
│   ├── sample_street_scene.png                 # 1280x720 Istanbul cobblestone sunset scene
│   └── sample_style_ref.png                    # 768x768 neon cyberpunk lighting reference
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
```

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
```
