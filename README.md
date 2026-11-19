# openfal

Local PyTorch inference for Qwen-Image-2.1, Wan 2.2 and MiniMax H3, with an HTTP API, a request queue, and a browser Studio. Model checkpoints stay on the Windows/Linux/macOS host running the server; clients can connect from another machine.

The runtime loads supported single-file FP8/INT8 checkpoints and moves model blocks between host memory and the selected compute device. Hardware profiles guide offloading. They do not establish measured memory use, multi-GPU execution, or a performance advantage over other engines.

## Why openfal

- **fal-style local API and SDK:** Exposes fal-compatible route names, `/queue/*` submission, SSE status streaming, and a `fal_client`-style Python SDK (`openfal.subscribe`, `run`, `submit`, `upload_file`) for the supported local workflows, without claiming hosted-service catalog or feature parity.
- **Turnkey workflows without node-graph wiring:** Loads supported single-file FP8/INT8 checkpoints and installs/verifies supporting conditioning assets (encoders, VAEs, and Wan Animate's CLIP/pose/SAM2 files) behind fixed HTTP routes and Studio. You trade ComfyUI's custom node-graph composability for zero-graph setup.
- **Single CLI for detection, downloads, and serving:** Combines `openfal detect`, `autosetup`/`download`, `models`, `cleanup`, and `serve` in one tool so you do not have to assemble model weights and dependencies separately before starting the server. Unlike LLM serving engines such as vLLM (designed around continuous batching and multi-GPU tensor parallelism), openfal is a serialized local PyTorch block-offloading runner for image and video diffusion workloads.

## Implemented workflows and limits

| Workflow | Endpoint | Local model |
| --- | --- | --- |
| Text-to-image | `qwen/image-2.1/text-to-image` | Qwen-Image-2.1-Turbo |
| Reference-guided image editing | `qwen/image-2.1/edit` | Qwen-Image-2.1-Turbo |
| Text-to-video | `wan/v2.2/text-to-video` | Wan2.2-T2V-A14B |
| Image-to-video | `wan/v2.2/image-to-video` | Wan2.2-I2V-A14B |
| Text/image-to-video with audio | `minimax/h3/text-to-video`, `minimax/h3/image-to-video` | MiniMax FL2VA |
| Reference-conditioned video and editing | `minimax/h3/reference-to-video`, `minimax/h3-max/recast`, `minimax/h3-max/relight`, `minimax/h3-max/insert-video` | MiniMax Ref2VA |
| Text-to-image and image editing | `minimax/h3/text-to-image`, `minimax/h3/image-to-image` | MiniMax H3-Image |
| Character animation from a driving performance | `wan/v2.2/animate` | Wan2.2-Animate-14B |
| Character replacement in a driving scene | `wan/v2.2/recast` | Wan2.2-Animate-14B |

Wan Animate uses an identity image and a driving video, with local CLIP conditioning, whole-body pose detection and face-motion crops. Replacement additionally uses SAM2 video segmentation to condition the background. These are generative workflows; identity, motion and scene fidelity still require visual inspection.

The `minimax/h3-max/*` names route to the corresponding local H3 partitions. They do not download a separate H3 Max model or provide a separately implemented hosted-service feature set. Ref2VA editing is generative: it does not guarantee exact preservation of a driving video's motion, scene, identity or timing.

The API accepts common fal-style route names and request/queue conventions. Compatibility does not imply feature parity with the hosted service. Use `GET /v1/endpoints` and `/docs` for the local route inventory and schema. Some accepted compatibility fields, including safety-checker and prompt-expansion options, do not invoke equivalent hosted services. `has_nsfw_concepts` is not a moderation guarantee.

## Measured RTX 5090 Benchmarks (32 GiB)

Sequential VRAM-isolated runs on a single NVIDIA GeForce RTX 5090 (`--warmup 1 --repeats 2 --concurrency 1 --sequential-local`; `openfal` and `ComfyUI` native on Windows, `vLLM-Omni` on the same GPU under WSL2). Full methodology, cold-start/startup latencies, and raw JSON telemetry are in [docs/benchmark-rtx-5090-2026-11-18.md](docs/benchmark-rtx-5090-2026-11-18.md) and [benchmark/results/bench_NVIDIA_GeForce_RTX_5090_full_comparison.md](benchmark/results/bench_NVIDIA_GeForce_RTX_5090_full_comparison.md).

| Scenario | Workload | `openfal` Warm (Peak VRAM) | `ComfyUI` Warm (Peak VRAM) | `vLLM-Omni` Warm (Peak VRAM) |
| --- | --- | ---: | ---: | ---: |
| `qwen21_t2i` | Qwen-Image-2.1-Turbo (`1344×768`, 8 steps) | `3.98 s` (`24.7 GB`) | `5.32 s` (`16.6 GB`) | **`2.45 s`** (`24.7 GB`) |
| `minimax_h3_t2i` | MiniMax-H3-Image (`1344×768`, 6 PDD steps) | **`3.47 s`** (`26.1 GB`) | `3.83 s` (`30.8 GB`) | `65.00 s` (`16.3 GB`) |
| `wan22_i2v_14b` | Wan 2.2 A14B I2V (`848×480`, 33f @ 16 fps, 20 steps) | `109.14 s` (`19.3 GB`) | `85.87 s` (`31.3 GB`) | **`75.58 s`** (`21.2 GB`) |
| `minimax_h3_i2v` | MiniMax-H3 FL2VA I2V + 32 kHz stereo audio (`832×480`, 20 steps) | `52.78 s` (`26.4 GB`) | **`48.34 s`** (`30.9 GB`) | `116.41 s` (`14.2 GB`)\* |

\*`openfal` and `ComfyUI` generate `73` frames (`3.04 s` @ `24 fps`); `vLLM-Omni` enforces a `4 s` minimum duration and outputs `107` frames (`4.46 s`). Server startup time: `1.8 s` (`openfal`) vs `18.6 s` (`ComfyUI`) vs `53–129 s` (`vLLM-Omni`).

## Install and start on Windows

Python 3.10+ is required. On Windows and Linux NVIDIA GPUs, install a **CUDA-enabled PyTorch build first** (otherwise `pip install -e ".[inference]"` pulls CPU-only PyTorch from PyPI on Windows):

```powershell
git clone https://github.com/esokullu/openfal.git
cd openfal

# 1. Install CUDA PyTorch first (match your NVIDIA driver; cu128 supports RTX 40/50 series):
py -3.11 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. Install openfal and its inference dependencies:
py -3.11 -m pip install -e ".[inference]"

# 3. Check the devices PyTorch can use:
py -3.11 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
openfal detect

# 4. Install the smallest supported image-generation setup (~16.1 GB):
openfal autosetup --tier minimal -y

# 5. Start the local API and Studio (binds 127.0.0.1 by default):
openfal serve --port 8008
```

To accept connections from another machine on your LAN, bind explicitly and protect the server with an API key (`Authorization: Key <key>`, `Authorization: Bearer <key>`, or `X-API-Key: <key>`):

```powershell
openfal serve --host 0.0.0.0 --port 8008 --api-key "your-local-secret"
```

If `openfal` is not on `PATH`, use `py -3.11 -m openfal.cli` in its place. Open `http://127.0.0.1:8008/` on the server machine for Studio, or `/docs` for the API explorer.

On Linux/macOS, use `python3 -m pip install -e ".[inference]"` and the same `openfal` commands. Device selection can fall back to MPS or CPU when CUDA is unavailable. MPS compute uses BF16 only after a capability probe succeeds, otherwise FP32. When Metal cannot store FP8 tensors, the packed weights remain on CPU and decoded compute tensors are transferred to MPS. Host regression tests cover these branches; actual Mac inference has not been validated. Memory requirements and performance need testing on the target Mac.

On Apple Silicon, CPU and MPS use the same physical memory pool: CPU offload does not provide additional RAM. Budget unified memory for the active model's loaded weights, encoders/VAEs, preprocessing models and working tensors. FP32 fallback increases that requirement. The 24/36/64 GB profile names do not guarantee that every model will fit; actual memory requirements have not been benchmarked on a Mac. Wan preprocessing's ONNX Runtime 1.23.2 macOS arm64 wheel requires macOS 13 or newer. For long macOS video runs, keep the Mac on AC power with the lid open (or clamshell with an external display) and start the server with `caffeinate -dims openfal serve ...` so idle sleep does not pause Metal execution.

Follow the [Mac validation guide](docs/macos-validation.md) to capture hardware and package versions, run the automated Metal checks, and evaluate the full client suite on a local Mac server.

`autosetup` offers a preview and explicit tiers:

```powershell
openfal autosetup --dry-run
openfal autosetup --tier moderate -y
openfal autosetup --tier aggressive -y
```

Automatic setup chooses supported FP8/INT8 checkpoints and uses estimated bundle sizes to select a tier. Ensure sufficient disk space, host memory and device memory for your workload. A profile name alone does not establish that a model will fit.

## Complete setup before inference

Weights default to `./weights/MiniMax-H3`. Use the same `--weights-dir` when downloading, inspecting and serving:

```powershell
openfal download --component qwen_image_21 --quant fp8_e4m3 --weights-dir D:\openfal-weights
openfal models --weights-dir D:\openfal-weights
openfal serve --weights-dir D:\openfal-weights --port 8008
```

Setup installs checkpoints, configuration files, tokenizers and shared components. Metadata download failures fail setup. Generation requests read model assets locally and never download missing model checkpoints or metadata. Reference images, video and audio supplied as URLs still require access to those URLs.

Repair supporting assets in an older installation without downloading the DiT checkpoints again:

```powershell
openfal download --component all --runtime-assets-only
openfal models
```

The repair operation installs missing configs/tokenizers, shared Wan encoder/VAE dependencies, and Animate's CLIP/pose/SAM2 models. Missing main model weights still require a normal component download. Wan 2.2 14B needs the 16-channel `wan_2.1_vae.safetensors`; the 48-channel TI2V-5B VAE is a different architecture.

An existing Animate installation needs approximately **4.77 GB (4.44 GiB)** of additional supporting assets:

| Assets | Pinned source | Files | Size |
| --- | --- | ---: | ---: |
| CLIP vision model and processor | [Wan Animate Diffusers](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B-Diffusers/tree/6f4df10861c758af86ac3c979aacc1bf5c03eff0) | 3 | 1.26 GB |
| YOLOv10m detector and ViTPose-H whole-body ONNX | [Wan Animate processing checkpoints](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/tree/cb93a225fbaf1ca100f54e79da8f994995b689b3/process_checkpoint) | 395 | 2.61 GB |
| SAM2.1 Hiera Large model and processors | [SAM2.1 Transformers checkpoint](https://huggingface.co/facebook/sam2.1-hiera-large/tree/665f8e2ad61cf5f53d65644ff27c8ee525124610) | 5 | 0.90 GB |

The packaged [asset manifest](openfal/assets/wan_animate_manifest.json) includes every pose external tensor and the expected byte size of all 403 files. Setup and runtime readiness check these files locally. Repair downloads only missing or truncated assets; it does not defer them until the first generation request. Run `openfal download --component wan22_animate --runtime-assets-only` to repair only this component.

`GET /v1/models/status` distinguishes these fields:

- `installed`: required local checkpoints and metadata are present; Animate's supporting files also match their expected sizes. This does not measure generation quality or hardware capacity.
- `missing_assets`: missing or unsupported files that prevent the asset set from being complete.
- `runtime_supported` and `runtime_unavailable_reason`: whether the component has an implemented generation path. Animate supports both animation and replacement modes.
- `weights_present`: some checkpoint files are present, including in an incomplete installation.

Structural demonstration shards do not establish inference readiness. Supported checkpoints may coexist with old unsupported files; the runtime ignores GGUF/NVFP4/W4A8 files when resolving model weights.

### Supported installer formats

Use `--quant fp8_e4m3`. Depending on the component, this selects FP8 or INT8 checkpoints from the configured Comfy-Org repositories. Recognized local single-file BF16 checkpoints remain usable where a runner supports their names and tensor layout.

INT4/NF4, GGUF, NVFP4 and W4A8 inference are not implemented. Explicit installs of these formats fail before downloads. `--source official-bf16` and `--quant bf16` also fail before downloading: the Python runners do not consume the official repository shard layouts or the experimental packer's representation of them.

`--pack-only` is an experimental local quantization tool. Its output is not a substitute for runtime checkpoints. A custom `--repo-id` must contain the supported filenames and tensor layout; it does not convert an arbitrary repository into a supported model.

### Components and bundles

Sizes below are approximate download sizes for the supported FP8/INT8 selection, not measured runtime memory requirements.

| Bundle | Contents | Approximate size |
| --- | --- | ---: |
| `qwen_image_21` | Qwen Turbo DiT, Qwen3-VL-8B encoder and image VAE | 16.1 GB |
| `wan22_animate` | Animate DiT and conditioning/preprocessing assets, with shared Wan dependencies | 21.7 GB |
| `alibaba-only` | Wan T2V, I2V, Animate assets and Qwen Image, with shared encoders/VAEs | 98.6 GB |
| `minimax-omni-compact` | Ref2VA and shared Qwen3-VL-32B/video/audio VAEs | 48.0 GB |
| `minimax-only` | FL2VA, Ref2VA, H3-Image and shared components | 88.1 GB |
| `all` | Both model families | 186.7 GB |

The compact MiniMax bundle supplies Ref2VA; it does not contain the dedicated FL2VA or H3-Image checkpoints.

```powershell
openfal download --component alibaba-only --quant fp8_e4m3
openfal download --component minimax-only --quant fp8_e4m3
openfal download --component minimax-omni-compact --quant fp8_e4m3
openfal download --component all --quant fp8_e4m3
```

Individual IDs are `wan22_t2v_moe`, `wan22_i2v_moe`, `wan22_animate`, `qwen_image_21`, `fl2va`, `ref2va`, `h3-image`, `qwen3_vl`, and `vae`. Individual MiniMax DiT installs include their shared encoder/decoder dependencies. Wan components share their UMT5 encoder and compatible VAE.

Completed Hugging Face downloads are reused where their local metadata permits. Interrupted downloads can be resumed by rerunning the same command. Inspect before removing files:

```powershell
openfal cleanup --dry-run
openfal cleanup
```

## Call the API

For a client on another machine, replace `127.0.0.1` with the server's LAN address and allow the selected port through the server firewall. `localhost` on a Mac refers to the Mac, not the Windows server.

### Generate an image

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8008/qwen/image-2.1/text-to-image" `
  -ContentType "application/json" `
  -Body '{"prompt":"A golden poodle running on an Istanbul cobblestone street","resolution":"768P","aspect_ratio":"16:9","seed":20261117}' |
  ConvertTo-Json -Depth 8
```

The response contains `images`, `seed`, `timings` and `strata_engine` metadata. Fetch the returned image URL to inspect the result. For real inference, `strata_engine.mode` is `real_pytorch`.

### Edit with a local reference

```python
import openfal

server = "http://127.0.0.1:8008"
reference = openfal.upload_file("portrait.png", base_url=server)
result = openfal.subscribe(
    "qwen/image-2.1/edit",
    arguments={
        "prompt": "Keep the person recognizable and change the lighting to warm evening light",
        "reference_image_urls": [reference],
        "resolution": "768P",
        "aspect_ratio": "1:1",
        "seed": 20261117,
    },
    base_url=server,
)
print(result["images"][0]["url"])
```

### Queue a video request

```bash
curl -X POST http://127.0.0.1:8008/queue/wan/v2.2/text-to-video \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A dog walking down a quiet street","resolution":"480P","duration":2,"seed":20261117}'
```

Submission returns `request_id`, `status_url`, `response_url` and `cancel_url`. Poll `status_url` (optionally with `?logs=1`) until `COMPLETED` or `FAILED`, then fetch `response_url` only on success. `/requests/{id}/status/stream` provides SSE updates. Both direct requests and queued generation use the same serialized model execution path.

Wan image-to-video requires a starting image. MiniMax FL2VA accepts first and/or last frame anchors. MiniMax Ref2VA accepts reference images, videos and audio; it enforces at most nine images, three videos, three audio clips and twelve references total. Audio references require a visual reference. Reference decoding failures fail the request instead of being silently ignored.

### Animate or replace a character

Upload one identity image and one driving video. Studio provides separate upload buttons for each. For an API request:

```python
import openfal

server = "http://127.0.0.1:8008"
identity = openfal.upload_file("character.png", base_url=server)
driver = openfal.upload_file("performance.mp4", base_url=server)
result = openfal.subscribe(
    "wan/v2.2/animate",
    arguments={
        "prompt": "The reference character follows the driving performance naturally",
        "face_image_url": identity,
        "driving_video_url": driver,
        "resolution": "480P",
        "duration": 3,
        "fps": 16,
        "seed": 20261117,
    },
    base_url=server,
)
print(result["video"]["url"])
```

`wan/v2.2/animate` defaults to `animate_mode="animate"`; `wan/v2.2/recast` defaults to `animate_mode="replace"`. Either route accepts an explicit `animate_mode` override. The driving video field also accepts the aliases `target_video_url` and `video_url`. Use a clear, visible person in the driver so pose and face motion can be detected. Replacement uses a tracked foreground mask and background conditioning; it does not promise pixel-exact compositing.

### Defaults and output checks

Leave `num_inference_steps` and `guidance_scale` unset to use model-specific defaults. Studio does this by default. Qwen Turbo uses eight steps; Wan MoE T2V/I2V uses forty; Wan Animate/Replacement uses twenty steps with guidance 1; MiniMax video uses thirty, while MiniMax Image follows its six-step PDD schedule. These models do not share one universal "turbo" step count.

Accepted dimensions, frame counts and durations depend on the backend and checkpoint. Wan MoE T2V/I2V supports at most 81 frames. Wan Animate/Replacement uses native 77-frame internal windows and caps output to the real driving video's available duration. Short clips are padded internally and trimmed; longer clips use multiple windows. Custom window lengths are rejected because short-window quality checks failed. Increasing resolution, duration, references or steps increases resource use and may exceed the machine's capacity. Read the returned media dimensions/duration and inspect the media itself rather than assuming an HTTP success establishes correct output.

MiniMax FL2VA accepts `start_image_url` (or `image_url`) and `end_image_url`, including a last-frame-only request. Use Ref2VA for `reference_image_urls` or face/identity references; FL2VA rejects those fields instead of ignoring them.

MiniMax recast, relighting and insertion routes use reference-conditioned generation. They are not deterministic face replacement or motion-preserving compositing tools.

## Hardware, memory and telemetry

`openfal detect` inspects eligible GPUs, system memory and free disk space. It honors `CUDA_VISIBLE_DEVICES` and avoids selecting small display GPUs such as a T400 alongside a larger compute GPU. To select a particular CUDA device before launching in PowerShell:

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
openfal serve --profile single_5090_32gb --port 8008
```

The current real PyTorch runners select logical `cuda:0` (or MPS/CPU fallback). Profiles named `dual_*` describe memory planning configurations; they do not establish that these runners execute one expert per GPU, perform tensor parallelism, or eliminate PCIe transfers.

Studio's memory/block map and `/v1/engine/status` are profile estimates. The response labels these as `telemetry_kind: "profile_estimate"`. RAM/VRAM totals, bandwidth and compute/DMA overlap are not live hardware measurements. The project does not publish validated comparative speedup or latency guarantees for every listed profile.

`POST /v1/engine/sleep` evicts loaded inference models; `POST /v1/engine/wake` permits subsequent execution. CUDA context overhead can remain, and model loading on the next request takes time. Use device monitoring to measure actual allocations. Profile changes wait for active inference and unload cached model instances before the new profile is used.

The optional `csrc/` extension can be built with `OPENFAL_BUILD_CUDA=1` and `pip install -e . --no-build-isolation` when a compatible CUDA toolkit and compiler are installed. Building it does not enable currently unsupported checkpoint formats or validate the modeled multi-GPU performance claims. It is not required for the Python inference paths.

## Development, validation and deployment scope

```powershell
py -3.11 -m pip install -e ".[inference,dev]"
py -3.11 -m pytest -q
node --test tests/client/studio_installer.test.mjs

$env:OPENFAL_URL = "http://127.0.0.1:8008"
node tests/client/api/run_all.mjs
```

The API suite runs model cases independently, retains request/response metadata and media in a fresh output directory, continues after failures, and exits nonzero if any case fails. Wan Animate and replacement tests require both identity and driving inputs plus the complete supporting asset set. See [the client test guide](tests/client/README.md) and [sample provenance](tests/client/samples/README.md) for options and evaluation criteria. Real GPU tests and visual/audio inspection are necessary in addition to unit tests; a passing installer or host test does not establish end-to-end generation quality.

The [2026-11-17 Windows validation record](docs/validation-2026-11-17.md) documents real outputs, regression checks and the remaining Apple Silicon hardware verification. The [RTX 5090 engine benchmark](docs/benchmark-rtx-5090-2026-11-18.md) (updated 2026-11-19) compares `openfal`, `ComfyUI`, and `vLLM-Omni` across text-to-image and image-to-video workloads for Qwen-Image-2.1, Wan 2.2, and MiniMax-H3.

`OPENFAL_SYNTHETIC_FALLBACK=1` explicitly enables development placeholder rendering. Leave it unset for real inference. Synthetic output is labeled in the API and Studio; missing model assets never silently enable it. `/health` reports API liveness and the synthetic-mode flag, not a complete readiness or output-quality guarantee.

The server is a local inference service, not a managed hosting platform. By default `openfal serve` binds `127.0.0.1`; pass `--api-key` (or set `OPENFAL_API_KEY`) whenever exposing `--host 0.0.0.0` on a LAN, and restrict network access to trusted clients. Its queue state is held in memory and does not survive server restarts. Validate the endpoints, model assets, input types and hardware settings your application needs before shipping it. The unsupported capabilities listed above remain unavailable even when the rest of the suite succeeds.

The application source is MIT-licensed; see [LICENSE](LICENSE). Model weights retain their upstream license terms.
