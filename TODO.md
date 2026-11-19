# TODO

## Pre-Launch Polish (v0.1.0)

- [ ] **Add visual proof to `README.md` and `docs/index.html`:**
  - Capture 1 crisp screenshot of the built-in Studio UI.
  - Add 2–3 compact WebP/PNG real sample outputs (Qwen-Image-2.1, MiniMax-H3-Image) and a short Wan 2.2 / MiniMax FL2VA GIF or preview.
- [ ] **Add a 10-line "Drop-in `fal.ai` → `openfal`" migration example (Python + JS/curl):**
  - Show side-by-side how an app switches between hosted `fal.ai` and local `openfal` via one `base_url` / env var.
  - Include a 5-line local-first / cloud-fallback router snippet.
- [ ] **Final repo hygiene pass & `v0.1.0` release:**
  - Run full `pytest -q` and `node --test tests/client/studio_installer.test.mjs`.
  - Audit the repo (`git grep`) to ensure no stray local paths or credentials remain.
  - Tag and publish a clean `v0.1.0` GitHub Release with concise release notes.

## Post-v0.1 Roadmap (Deferred until after public launch)

- [ ] **Pluggable execution backends (`--backend pytorch | comfyui | vllm-omni`):**
  - Promote the tested ComfyUI workflow adapters (`benchmark/comfy_workflows.py`) and vLLM-Omni adapters (`benchmark/bench_runner.py`) into opt-in server backends behind the same fal-compatible API, queue, SSE stream, Studio UI, and shared `weights/` cache.
- [ ] **Agent & OpenAI compatibility layer:**
  - Add an `openfal mcp` server for local agentic image/video generation.
  - Add an optional thin OpenAI `/v1/images/generations` compatibility shim.
- [ ] **Native Apple Silicon full-model validation:**
  - Run `docs/macos-validation.md` on real Apple Silicon hardware (36 GB+ unified memory) and publish measured latency, peak unified memory, and output verification records.
