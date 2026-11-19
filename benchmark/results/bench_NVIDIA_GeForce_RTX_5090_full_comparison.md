# RTX 5090 (32 GiB) Engine Benchmark: openfal vs ComfyUI vs vLLM-Omni

Run configuration: `GPU 0 = NVIDIA GeForce RTX 5090`, `--warmup 1 --repeats 2 --concurrency 1 --sequential-local` (each engine launched & torn down sequentially in VRAM isolation).

| Scenario | Engine | Startup (s) | Cold (s) | Warm Mean (s) | Warm p50 (s) | Req/min | Peak VRAM (GB) | Peak VRAM (MiB) | Mean GPU Util (%) | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `qwen21_t2i` | `openfal` | 1.85 | 16.23 | 3.98 | 3.98 | 15.06 | 24.7 | 25333 | 35.9% | `bench_NVIDIA_GeForce_RTX_5090_20261119_102359.json` |
| `qwen21_t2i` | `comfyui` | 18.55 | 12.88 | 5.32 | 5.32 | 11.27 | 16.6 | 17009 | 65.6% | `bench_NVIDIA_GeForce_RTX_5090_20261118_073755.json` |
| `qwen21_t2i` | `vllm_omni` | 53.40 | 5.05 | 2.45 | 2.45 | 24.49 | 24.7 | 25257 | 74.7% | `bench_NVIDIA_GeForce_RTX_5090_20261118_124042.json` |
| `minimax_h3_t2i` | `openfal` | 1.81 | 32.44 | 3.47 | 3.47 | 17.28 | 26.1 | 26711 | 19.5% | `bench_NVIDIA_GeForce_RTX_5090_20261119_102359.json` |
| `minimax_h3_t2i` | `comfyui` | 18.55 | 10.18 | 3.83 | 3.83 | 15.68 | 30.8 | 31513 | 65.3% | `bench_NVIDIA_GeForce_RTX_5090_20261118_073755.json` |
| `minimax_h3_t2i` | `vllm_omni` | 128.59 | 65.25 | 65.00 | 65.00 | 0.92 | 16.3 | 16697 | 23.2% | `bench_NVIDIA_GeForce_RTX_5090_20261118_140548.json` |
| `wan22_i2v_14b` | `openfal` | 1.86 | 127.27 | 109.14 | 109.14 | 0.55 | 19.3 | 19803 | 92.2% | `bench_NVIDIA_GeForce_RTX_5090_20261119_102359.json` |
| `wan22_i2v_14b` | `comfyui` | 18.55 | 121.74 | 85.87 | 85.87 | 0.70 | 31.3 | 32021 | 85.7% | `bench_NVIDIA_GeForce_RTX_5090_20261118_073755.json` |
| `wan22_i2v_14b` | `vllm_omni` | 107.76 | 90.18 | 75.58 | 75.58 | 0.79 | 21.2 | 21729 | 93.6% | `bench_NVIDIA_GeForce_RTX_5090_20261118_124042.json` |
| `minimax_h3_i2v` | `openfal` | 1.81 | 84.45 | 52.78 | 52.78 | 1.14 | 26.4 | 27061 | 76.2% | `bench_NVIDIA_GeForce_RTX_5090_20261119_102359.json` |
| `minimax_h3_i2v` | `comfyui` | 18.55 | 54.70 | 48.34 | 48.34 | 1.24 | 30.9 | 31635 | 90.6% | `bench_NVIDIA_GeForce_RTX_5090_20261118_073755.json` |
| `minimax_h3_i2v` | `vllm_omni` | 94.26 | 133.22 | 116.41 | 116.41 | 0.52 | 14.2 | 14503 | 69.2% | `bench_NVIDIA_GeForce_RTX_5090_20261118_134107.json` |

## Scenario Winners (Warm Mean Latency)

| Scenario | Fastest Engine | Warm Mean (s) | Speedup vs 2nd | Speedup vs 3rd |
| --- | --- | ---: | --- | --- |
| `qwen21_t2i` | `vllm_omni` | 2.45 | 1.63x vs `openfal` (3.98s) | 2.17x vs `comfyui` (5.32s) |
| `minimax_h3_t2i` | `openfal` | 3.47 | 1.10x vs `comfyui` (3.83s) | 18.72x vs `vllm_omni` (65.00s) |
| `wan22_i2v_14b` | `vllm_omni` | 75.58 | 1.14x vs `comfyui` (85.87s) | 1.44x vs `openfal` (109.14s) |
| `minimax_h3_i2v` | `comfyui` | 48.34 | 1.09x vs `openfal` (52.78s) | 2.41x vs `vllm_omni` (116.41s) |

## Notes & Precision / Parameter Differences

- `openfal` (`20261119_102359.json`): uses dynamic VRAM residency (`compute_resident_blocks`), native CUDA Tensor Core GEMMs (`torch._scaled_mm` FP8 / `torch._int_mm` INT8), cuDNN Flash Attention SDPA priority (`prefer_cudnn_sdpa`), pinned host-RAM fast H2D / O(1) zero-D2H swapping (`PinnedModuleStager`), and warm residency/conditioning caches on 32 GiB GPUs. Prior pre-optimization baseline runs (`7.00s` / `12.30s` / `182.74s` / `90.30s`) remain archived in `bench_NVIDIA_GeForce_RTX_5090_20261118_070636.json` and `bench_NVIDIA_GeForce_RTX_5090_20261118_072017.json`.
- `qwen21_t2i` (`1344x768`, 8 steps, CFG 1.0): `openfal` and `ComfyUI` run quantized `INT8 ConvRot`, while `vLLM-Omni` loads dequantized `BF16` weights quantized online to `--quantization fp8`.
- `wan22_i2v_14b` (`848x480`, 33 frames @ 16 fps, 20 steps, CFG 3.5): `openfal` and `ComfyUI` run `FP8 scaled` (boundary 0.9), while `vLLM-Omni` loads `BF16` + `--quantization fp8 --enable-cpu-offload`.
- `minimax_h3_t2i` (`1344x768`, 6 PDD steps, `min_generation_area=1024`): `openfal` and `ComfyUI` run quantized `INT8 ConvRot` resident in 32 GiB VRAM, whereas `vLLM-Omni` runs dequantized `BF16` (`40.23 GB` DiT + `51.51 GB` `Qwen3VL-32B`) with `DistributedLayerwiseOffloadBackend` (`checkpoint_mmap`, `--dlo-resident-layers 12`, `--enforce-eager`, `CUDNN_ATTN`).
- `minimax_h3_i2v` (`832x480`, 20 steps, 24 fps, coupled video+audio): `openfal` (`FP8 scaled`) and `ComfyUI` (`FP8 scaled`) generate `73` frames (`3.04s`) + `32 kHz` stereo AAC audio, while `vLLM-Omni` enforces `[4, 15]s` video duration and receives `num_frames=97`, outputting `107` frames (`4.46s`, `24 fps`) + `32 kHz` stereo AAC audio using `BF16` DiT + `DistributedLayerwiseOffloadBackend` (`--dlo-resident-layers 12`).
