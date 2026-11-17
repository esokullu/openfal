"""Shared setup/runtime manifest for local model metadata.

Downloading belongs to the installer. Inference only reads these assets, so both
paths must agree on filenames (including the upstream processor naming).
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConfigAssets:
    repo_id: str
    subdir: str
    required_files: tuple[str, ...]
    allow_patterns: tuple[str, ...]

    def missing(self, weights_dir: Path) -> list[str]:
        root = weights_dir / self.subdir
        return [name for name in self.required_files
                if not (root / name).is_file() or (root / name).stat().st_size == 0]


_WAN_FILES = (
    "model_index.json", "scheduler/scheduler_config.json", "transformer/config.json",
    "vae/config.json", "text_encoder/config.json", "tokenizer/tokenizer_config.json",
    "tokenizer/spiece.model",
)
_WAN_PATTERNS = (
    "model_index.json", "scheduler/*.json", "transformer/*.json", "transformer_2/*.json",
    "vae/*.json", "text_encoder/*.json", "tokenizer/*",
)
HF_CONFIGS = {
    "qwen_image_21": ConfigAssets(
        "Qwen/Qwen-Image-2.1-Turbo", "Qwen-Image-2.1-Turbo/hf_configs",
        ("model_index.json", "scheduler/scheduler_config.json", "transformer/config.json",
         "vae/config.json", "text_encoder/config.json", "processor/processor_config.json",
         "processor/tokenizer.json", "processor/tokenizer_config.json", "processor/chat_template.jinja"),
        ("model_index.json", "scheduler/*.json", "transformer/*.json", "vae/*.json",
         "text_encoder/*.json", "processor/*"),
    ),
    "minimax_h3": ConfigAssets(
        "MiniMaxAI/MiniMax-H3", "minimax_h3_hf_configs",
        ("modular_model_index.json", "scheduler/scheduler_config.json", "audio_scheduler/scheduler_config.json",
         "transformer/config.json", "transformer_ref/config.json", "vae/config.json", "audio_vae/config.json",
         "text_encoder/config.json", "tokenizer/tokenizer_config.json", "tokenizer/tokenizer.json",
         "processor/preprocessor_config.json", "processor/video_preprocessor_config.json",
         "processor/tokenizer.json", "processor/tokenizer_config.json", "processor/chat_template.json"),
        ("modular_model_index.json", "scheduler/*.json", "audio_scheduler/*.json", "transformer/*.json",
         "transformer_ref/*.json", "vae/*.json", "audio_vae/*.json", "text_encoder/*.json",
         "tokenizer/*", "processor/*"),
    ),
    "minimax_h3_image": ConfigAssets(
        "MiniMaxAI/MiniMax-H3-Image", "minimax_h3_image_hf_configs",
        ("modular_model_index.json", "scheduler/scheduler_config.json",
         "transformer_diffusers/config.json", "vae_diffusers/config.json"),
        ("modular_model_index.json", "scheduler/*.json", "transformer_diffusers/*.json", "vae_diffusers/*.json"),
    ),
}
for _key, _variant in (
    ("wan22_t2v_moe", "Wan2.2-T2V-A14B"),
    ("wan22_i2v_moe", "Wan2.2-I2V-A14B"),
    ("wan22_animate", "Wan2.2-Animate-14B"),
):
    HF_CONFIGS[_key] = ConfigAssets(
        f"Wan-AI/{_variant}-Diffusers", f"{_variant}/hf_configs",
        _WAN_FILES + (() if _key == "wan22_animate" else ("transformer_2/config.json",)), _WAN_PATTERNS,
    )


def config_keys_for_component(component_id: str) -> tuple[str, ...]:
    if component_id == "h3-image":
        return ("minimax_h3", "minimax_h3_image")
    if component_id in ("fl2va", "ref2va", "qwen3_vl", "vae"):
        return ("minimax_h3",)
    return (component_id,)


# Each tuple names one required checkpoint, with supported filename alternatives.
# Structural demo shards and one half of a MoE pair are not an installation.
WEIGHT_GROUPS = {
    "qwen_image_21": (
        ("*qwen_image_2.1_turbo*.safetensors", "qwen_image_2.1_int8_convrot.safetensors", "qwen_image_2.1_bf16.safetensors"),
        ("*qwen3vl_8b*.safetensors",),
        ("*qwen_image_2.1_vae*.safetensors", "vae/diffusion_pytorch_model.safetensors"),
    ),
    "wan22_t2v_moe": (("*t2v*high_noise*.safetensors",), ("*t2v*low_noise*.safetensors",)),
    "wan22_i2v_moe": (("*i2v*high_noise*.safetensors",), ("*i2v*low_noise*.safetensors",)),
    "wan22_animate": (("*wan2.2_animate*.safetensors",),),
    "fl2va": (("*minimax_h3_fl2va*.safetensors",),),
    "ref2va": (("*minimax_h3_ref2va*.safetensors",),),
    "h3-image": (("*minimax_h3_image_pruned*.safetensors", "diffusion_models/minimax_h3_image*.safetensors"), ("*minimax_h3_image_vae*.safetensors",)),
    "qwen3_vl": (("*qwen3vl_32b*.safetensors",),),
    "vae": (("*minimax_h3_video_vae*.safetensors",), ("*minimax_h3_audio_vae*.safetensors",)),
}
COMPONENT_DEPENDENCIES = {"fl2va": ("qwen3_vl", "vae"), "ref2va": ("qwen3_vl", "vae"), "h3-image": ("qwen3_vl",)}
WAN_SHARED_FILES = (
    "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "split_files/vae/wan_2.1_vae.safetensors",
)


def unsupported_checkpoint_format(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(".gguf") or "nvfp4" in lowered or "w4a8" in lowered


def find_weight(root: Path, patterns: tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        for path in root.rglob(pattern):
            if ("strata_blocks" not in path.parts and not unsupported_checkpoint_format(path.name)
                    and path.is_file() and path.stat().st_size):
                return path
    return None
