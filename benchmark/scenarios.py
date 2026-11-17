"""Locked apples-to-apples benchmark scenario definitions across openfal, ComfyUI, and vLLM-Omni."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class BenchmarkScenario:
    """Single cross-engine workload with identical resolution, steps, frames, and prompt."""

    scenario_id: str
    modality: str  # "image" | "video"
    display_name: str
    prompt: str
    negative_prompt: str
    width: int
    height: int
    openfal_resolution: str
    openfal_aspect_ratio: str
    steps: int
    guidance_scale: float
    num_frames: int
    fps: int
    duration_sec: float
    openfal_endpoint: str
    vllm_model_id: str
    vllm_default_offload: bool
    comfy_template: str
    vram_resident_estimate_gb: float


SCENARIOS: Dict[str, BenchmarkScenario] = {
    "qwen21_t2i": BenchmarkScenario(
        scenario_id="qwen21_t2i",
        modality="image",
        display_name="Qwen-Image-2.1-Turbo T2I (1344x768, 8 steps)",
        prompt="A realistic photo of a tabby cat running fast behind a fluffy apricot poodle on a sunlit cobblestone street in Istanbul, warm late afternoon light",
        negative_prompt="",
        width=1344,
        height=768,
        openfal_resolution="768P",
        openfal_aspect_ratio="16:9",
        steps=8,
        guidance_scale=1.0,
        num_frames=1,
        fps=1,
        duration_sec=0.0,
        openfal_endpoint="qwen/image-2.1/text-to-image",
        vllm_model_id="Qwen/Qwen-Image-2.1-Turbo",
        vllm_default_offload=False,
        comfy_template="qwen21_t2i",
        vram_resident_estimate_gb=18.0,
    ),
    "wan22_t2v_14b": BenchmarkScenario(
        scenario_id="wan22_t2v_14b",
        modality="video",
        display_name="Wan 2.2 A14B T2V (848x480, 33 frames @ 16fps, 20 steps)",
        prompt="A realistic cinematic video of an orange tabby cat running fast behind a fluffy apricot poodle down a sunlit cobblestone street in Istanbul, warm golden hour lighting",
        negative_prompt="low quality, blurry, watermark, text, distorted",
        width=848,
        height=480,
        openfal_resolution="480P",
        openfal_aspect_ratio="16:9",
        steps=20,
        guidance_scale=4.0,
        num_frames=33,
        fps=16,
        duration_sec=2.1,
        openfal_endpoint="wan/v2.2-a14b/text-to-video",
        vllm_model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        vllm_default_offload=True,
        comfy_template="wan22_t2v_14b",
        vram_resident_estimate_gb=36.0,
    ),
    "minimax_h3_t2v": BenchmarkScenario(
        scenario_id="minimax_h3_t2v",
        modality="video",
        display_name="MiniMax-H3 FL2VA Joint Video+Audio (832x480, 73f @ 24fps, 20 steps)",
        prompt="A realistic cinematic video of a tabby cat running fast behind a fluffy apricot poodle down a sunlit cobblestone street in Istanbul, historic wooden houses, playful barking and footsteps",
        negative_prompt="",
        width=832,
        height=480,
        openfal_resolution="480P",
        openfal_aspect_ratio="16:9",
        steps=20,
        guidance_scale=1.0,
        num_frames=73,
        fps=24,
        duration_sec=3.0,
        openfal_endpoint="minimax/h3/text-to-video",
        vllm_model_id="MiniMaxAI/MiniMax-H3",
        vllm_default_offload=True,
        comfy_template="minimax_h3_t2v",
        vram_resident_estimate_gb=48.0,
    ),
    "minimax_h3_t2i": BenchmarkScenario(
        scenario_id="minimax_h3_t2i",
        modality="image",
        display_name="MiniMax-H3-Image T2I (1344x768, 6 PDD steps)",
        prompt="Studio photograph of a tabby cat chasing an apricot miniature poodle past an Istanbul cafe, dramatic late sunlight, crisp cobblestone texture",
        negative_prompt="",
        width=1344,
        height=768,
        openfal_resolution="768P",
        openfal_aspect_ratio="16:9",
        steps=6,
        guidance_scale=1.0,
        num_frames=1,
        fps=1,
        duration_sec=0.0,
        openfal_endpoint="minimax/h3/text-to-image",
        vllm_model_id="MiniMaxAI/MiniMax-H3-Image",
        vllm_default_offload=True,
        comfy_template="minimax_h3_t2i",
        vram_resident_estimate_gb=42.0,
    ),
}
