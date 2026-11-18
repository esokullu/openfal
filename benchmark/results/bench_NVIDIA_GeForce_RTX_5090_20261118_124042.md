# openfal vs ComfyUI vs vLLM-Omni Benchmark (NVIDIA GeForce RTX 5090)

| Scenario | Engine | Concurrency | Cold (s) | Warm Mean (s) | Warm p50 (s) | Req/min | Peak VRAM (GB) | GPU Util (%) | Mode / Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen21_t2i | vllm_omni | 1 | 5.05 | 2.45 | 2.45 | 24.49 | 24.7 | 75% | vllm_omni |
| wan22_i2v_14b | vllm_omni | 1 | 90.18 | 75.58 | 75.58 | 0.79 | 21.2 | 94% | vllm_omni |
