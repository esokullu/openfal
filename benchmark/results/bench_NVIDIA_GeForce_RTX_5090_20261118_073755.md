# openfal vs ComfyUI vs vLLM-Omni Benchmark (NVIDIA GeForce RTX 5090)

| Scenario | Engine | Concurrency | Cold (s) | Warm Mean (s) | Warm p50 (s) | Req/min | Peak VRAM (GB) | GPU Util (%) | Mode / Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen21_t2i | comfyui | 1 | 12.88 | 5.32 | 5.32 | 11.27 | 16.6 | 66% | comfyui |
| minimax_h3_t2i | comfyui | 1 | 10.18 | 3.83 | 3.83 | 15.68 | 30.8 | 65% | comfyui |
| wan22_i2v_14b | comfyui | 1 | 121.74 | 85.87 | 85.87 | 0.70 | 31.3 | 86% | comfyui |
| minimax_h3_i2v | comfyui | 1 | 54.70 | 48.34 | 48.34 | 1.24 | 30.9 | 91% | comfyui |
