# openfal vs ComfyUI vs vLLM-Omni Benchmark (NVIDIA GeForce RTX 5090)

| Scenario | Engine | Concurrency | Cold (s) | Warm Mean (s) | Warm p50 (s) | Req/min | Peak VRAM (GB) | GPU Util (%) | Mode / Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen21_t2i | openfal | 1 | 16.23 | 3.98 | 3.98 | 15.06 | 24.7 | 36% | real_pytorch |
| minimax_h3_t2i | openfal | 1 | 32.44 | 3.47 | 3.47 | 17.28 | 26.1 | 20% | real_pytorch |
| wan22_i2v_14b | openfal | 1 | 127.27 | 109.14 | 109.14 | 0.55 | 19.3 | 92% | real_pytorch |
| minimax_h3_i2v | openfal | 1 | 84.45 | 52.78 | 52.78 | 1.14 | 26.4 | 76% | real_pytorch |
