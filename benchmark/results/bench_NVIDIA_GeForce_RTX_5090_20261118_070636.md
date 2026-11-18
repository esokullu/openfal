# openfal vs ComfyUI vs vLLM-Omni Benchmark (NVIDIA GeForce RTX 5090)

| Scenario | Engine | Concurrency | Cold (s) | Warm Mean (s) | Warm p50 (s) | Req/min | Peak VRAM (GB) | GPU Util (%) | Mode / Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen21_t2i | openfal | 1 | 52.49 | 7.00 | 7.00 | 8.57 | 12.0 | 24% | real_pytorch |
| minimax_h3_t2i | openfal | 1 | 42.03 | 12.30 | 12.30 | 4.88 | 14.8 | 33% | real_pytorch |
| wan22_i2v_14b | openfal | 1 | 206.75 | 182.74 | 182.74 | 0.33 | 14.3 | 92% | real_pytorch |
| minimax_h3_i2v | openfal | 1 | - | - | - | - | n/a | n/a | ERR: [WinError 10054] An existing connection was f |
