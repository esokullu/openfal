# openfal vs ComfyUI vs vLLM-Omni Benchmark Suite

Apples-to-apples latency, throughput, cold-start, and peak VRAM benchmark suite designed for **RunPod GPU pods** (and single-host Linux CUDA setups).

## Why Sequential VRAM Isolation + Matched Scenarios Matter

Comparing diffusion servers fails if engines use different checkpoints, step counts, resolutions, frame counts, or compete for GPU memory:

- **Matched parameters (`scenarios.py`)**: Every engine runs identical pixel dimensions, frame counts, denoising steps, CFG scales, seeds, and prompts:
  - `qwen21_t2i`: **1344×768**, 8 steps (`Qwen-Image-2.1-Turbo`)
  - `wan22_t2v_14b`: **848×480**, 33 frames @ 16 fps, 20 steps (`Wan 2.2 A14B MoE`)
  - `minimax_h3_t2v`: **832×480**, 73 frames @ 24 fps, 20 steps (`MiniMax-H3 FL2VA` + stereo audio)
  - `minimax_h3_t2i`: **1344×768**, 6 PDD steps (`MiniMax-H3-Image`)
- **Sequential isolation (`--sequential-local`)**: Boots one engine server at a time (`openfal serve` -> `ComfyUI` -> `vllm serve ... --omni`), waits for readiness, captures **cold 1st-run**, **warm mean / p50 / min / max**, **peak VRAM (`nvidia-smi`)**, and **mean GPU utilization**, then kills the server and waits for CUDA memory reclamation before starting the next engine.
- **Zero duplicate weights between `openfal` and `ComfyUI`**: `setup_pod.sh` downloads `Comfy-Org` `.safetensors` checkpoints once via `openfal download` and symlinks them into `ComfyUI/models/{diffusion_models,text_encoders,vae}`.
- **Fairness guard**: By default `bench_runner.py` fails loudly if `openfal` returns `strata_engine.mode != "real_pytorch"` so synthetic fallbacks never sneak into benchmark numbers.

---

## Choosing a RunPod GPU for What You Want to Test

| Regime Tested | Recommended `RUNPOD_GPU_TYPE` | `VLLM_OMNI_OFFLOAD` | What It Measures |
| --- | --- | --- | --- |
| **Tight VRAM / Async Offload** | `NVIDIA GeForce RTX 5090` (32 GB) or `NVIDIA GeForce RTX 4090` (24 GB) | `true` | `openfal` Counter/Pantry async DMA vs `ComfyUI` offload vs `vllm-omni --enable-layerwise-offload` |
| **Fits in VRAM (Kernel Speed)** | `NVIDIA A100-SXM4-80GB` or `NVIDIA H100 80GB HBM3` | `false` | Compute efficiency when models fit fully resident in VRAM |
| **Serialization vs Batching** | Any GPU above (`BENCH_CONCURRENCY="1,4"`) | match GPU | Impact of `openfal`'s `execution_lock` under concurrent traffic |

---

## Quick Start with RunPod

### 1. Configure `benchmark/.env`
```bash
cp benchmark/.env.example benchmark/.env
# Edit benchmark/.env and set RUNPOD_API_KEY=... (and optional HF_TOKEN=...)
```

### 2. Provision Pod, Bootstrap All 3 Engines, and Run
```bash
# 1. Provision pod (prints SSH command & proxy URLs, saves state to benchmark/.runpod_pod.json)
python3 -m benchmark.runpod_provision create

# 2. SSH-sync openfal + .env, install openfal + ComfyUI + vllm-omni, download & symlink weights
python3 -m benchmark.runpod_provision bootstrap

# 3. Run full sequential benchmark remotely and sync Markdown + JSON reports back to benchmark/results/
python3 -m benchmark.runpod_provision run

# Optional: run specific scenarios, targets, or concurrency sweep
python3 -m benchmark.runpod_provision run -- --targets openfal,comfyui --scenarios qwen21_t2i,wan22_t2v_14b --concurrency 1,4 --repeats 5

# 4. Stop billing as soon as results are synced
python3 -m benchmark.runpod_provision terminate
```

*(Use `python3 -m benchmark.runpod_provision stop` instead of `terminate` if you want to keep the `/workspace` volume warm between runs.)*

---

## Running Locally Against Existing Servers (e.g., Windows 5090 + Mac Client)

You can also skip RunPod and point `bench_runner.py` at running instances configured in `benchmark/.env`:

```bash
OPENFAL_URL=http://192.168.1.47:8008 \
  python3 -m benchmark.bench_runner --targets openfal --scenarios qwen21_t2i,minimax_h3_t2v --warmup 1 --repeats 3
```

Outputs are saved to `benchmark/results/bench_<gpu>_<timestamp>.md` and `.json`.
