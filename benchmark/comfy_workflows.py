"""ComfyUI API-format graph builders matching official `Comfy-Org` workflow templates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from benchmark.scenarios import BenchmarkScenario

QWEN21_TURBO_SIGMAS = "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0"

DEFAULT_COMFY_FILES: Dict[str, Dict[str, str]] = {
    "qwen21_t2i": {
        "unet": "qwen_image_2.1_turbo_int8_convrot.safetensors",
        "clip": "qwen3vl_8b_int8_convrot.safetensors",
        "vae": "qwen_image_2.1_vae_bf16.safetensors",
    },
    "minimax_h3_t2i": {
        "unet": "minimax_h3_image_pruned_int8_convrot.safetensors",
        "clip": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        "vae": "minimax_h3_image_vae_fp16.safetensors",
    },
    "wan22_i2v_14b": {
        "unet_high": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        "unet_low": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        "clip": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "vae": "wan_2.1_vae.safetensors",
        "image": "sample_street_photo.png",
    },
    "minimax_h3_i2v": {
        "unet": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
        "clip": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        "video_vae": "minimax_h3_video_vae_int8_convrot.safetensors",
        "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
        "image": "sample_street_photo.png",
    },
    "wan22_t2v_14b": {
        "unet_high": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        "unet_low": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
        "clip": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "vae": "wan_2.1_vae.safetensors",
    },
    "minimax_h3_t2v": {
        "unet": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
        "clip": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        "video_vae": "minimax_h3_video_vae_int8_convrot.safetensors",
        "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
    },
}


def build_comfy_prompt(
    scenario: BenchmarkScenario,
    seed: int,
    file_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build a ComfyUI `POST /prompt` API graph with identical geometry, seed, and steps."""
    files = dict(DEFAULT_COMFY_FILES[scenario.comfy_template])
    if scenario.input_image_path and "image" in files:
        files["image"] = Path(scenario.input_image_path).name
    if file_overrides:
        files.update(file_overrides)

    if scenario.comfy_template == "qwen21_t2i":
        return {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": files["unet"], "weight_dtype": "default"}},
            "2": {
                "class_type": "QwenImage21Cache",
                "inputs": {"model": ["1", 0], "device": "auto", "dtype": "default"},
            },
            "3": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": files["clip"], "type": "qwen_image", "device": "default"},
            },
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": files["vae"]}},
            "5": {
                "class_type": "TextEncodeQwenImage21",
                "inputs": {
                    "clip": ["3", 0],
                    "prompt": scenario.prompt,
                    "negative_prompt": scenario.negative_prompt,
                    "resolution": 1024,
                },
            },
            "6": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": scenario.width, "height": scenario.height, "batch_size": 1},
            },
            "7": {
                "class_type": "CFGGuider",
                "inputs": {
                    "model": ["2", 0],
                    "positive": ["5", 0],
                    "negative": ["5", 1],
                    "cfg": float(scenario.guidance_scale),
                },
            },
            "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "9": {"class_type": "ManualSigmas", "inputs": {"sigmas": QWEN21_TURBO_SIGMAS}},
            "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
            "11": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["10", 0],
                    "guider": ["7", 0],
                    "sampler": ["8", 0],
                    "sigmas": ["9", 0],
                    "latent_image": ["6", 0],
                },
            },
            "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["4", 0]}},
            "13": {
                "class_type": "SaveImage",
                "inputs": {"images": ["12", 0], "filename_prefix": f"bench_{scenario.scenario_id}"},
            },
        }

    if scenario.comfy_template == "wan22_t2v_14b":
        flow_shift = 5.0 if max(scenario.width, scenario.height) >= 1200 else 3.0
        split_step = max(1, int(scenario.steps) // 2)
        return {
            "1": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": files["clip"], "type": "wan", "device": "default"},
            },
            "2": {"class_type": "VAELoader", "inputs": {"vae_name": files["vae"]}},
            "3": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": files["unet_high"], "weight_dtype": "default"},
            },
            "4": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": files["unet_low"], "weight_dtype": "default"},
            },
            "5": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["3", 0], "shift": flow_shift}},
            "6": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["4", 0], "shift": flow_shift}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 0], "text": scenario.prompt}},
            "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 0], "text": scenario.negative_prompt}},
            "9": {
                "class_type": "EmptyHunyuanLatentVideo",
                "inputs": {
                    "width": scenario.width,
                    "height": scenario.height,
                    "length": int(scenario.num_frames),
                    "batch_size": 1,
                },
            },
            "10": {
                "class_type": "KSamplerAdvanced",
                "inputs": {
                    "model": ["5", 0],
                    "positive": ["7", 0],
                    "negative": ["8", 0],
                    "latent_image": ["9", 0],
                    "add_noise": "enable",
                    "noise_seed": int(seed),
                    "steps": int(scenario.steps),
                    "cfg": float(scenario.guidance_scale),
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "start_at_step": 0,
                    "end_at_step": split_step,
                    "return_with_leftover_noise": "enable",
                },
            },
            "11": {
                "class_type": "KSamplerAdvanced",
                "inputs": {
                    "model": ["6", 0],
                    "positive": ["7", 0],
                    "negative": ["8", 0],
                    "latent_image": ["10", 0],
                    "add_noise": "disable",
                    "noise_seed": int(seed),
                    "steps": int(scenario.steps),
                    "cfg": float(scenario.guidance_scale),
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "start_at_step": split_step,
                    "end_at_step": int(scenario.steps),
                    "return_with_leftover_noise": "disable",
                },
            },
            "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["2", 0]}},
            "13": {"class_type": "CreateVideo", "inputs": {"images": ["12", 0], "fps": float(scenario.fps)}},
            "14": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["13", 0],
                    "filename_prefix": f"bench_{scenario.scenario_id}",
                    "format": "mp4",
                    "format.codec": "auto",
                },
            },
        }

    if scenario.comfy_template == "wan22_i2v_14b":
        flow_shift = 5.0
        split_step = max(1, int(scenario.steps) // 2)
        return {
            "1": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": files["clip"], "type": "wan", "device": "default"},
            },
            "2": {"class_type": "VAELoader", "inputs": {"vae_name": files["vae"]}},
            "3": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": files["unet_high"], "weight_dtype": "default"},
            },
            "4": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": files["unet_low"], "weight_dtype": "default"},
            },
            "5": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["3", 0], "shift": flow_shift}},
            "6": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["4", 0], "shift": flow_shift}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 0], "text": scenario.prompt}},
            "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 0], "text": scenario.negative_prompt}},
            "9": {"class_type": "LoadImage", "inputs": {"image": files["image"]}},
            "10": {
                "class_type": "WanImageToVideo",
                "inputs": {
                    "positive": ["7", 0],
                    "negative": ["8", 0],
                    "vae": ["2", 0],
                    "width": scenario.width,
                    "height": scenario.height,
                    "length": int(scenario.num_frames),
                    "batch_size": 1,
                    "start_image": ["9", 0],
                },
            },
            "11": {
                "class_type": "KSamplerAdvanced",
                "inputs": {
                    "model": ["5", 0],
                    "positive": ["10", 0],
                    "negative": ["10", 1],
                    "latent_image": ["10", 2],
                    "add_noise": "enable",
                    "noise_seed": int(seed),
                    "steps": int(scenario.steps),
                    "cfg": float(scenario.guidance_scale),
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "start_at_step": 0,
                    "end_at_step": split_step,
                    "return_with_leftover_noise": "enable",
                },
            },
            "12": {
                "class_type": "KSamplerAdvanced",
                "inputs": {
                    "model": ["6", 0],
                    "positive": ["10", 0],
                    "negative": ["10", 1],
                    "latent_image": ["11", 0],
                    "add_noise": "disable",
                    "noise_seed": int(seed),
                    "steps": int(scenario.steps),
                    "cfg": float(scenario.guidance_scale),
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "start_at_step": split_step,
                    "end_at_step": int(scenario.steps),
                    "return_with_leftover_noise": "disable",
                },
            },
            "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["2", 0]}},
            "14": {"class_type": "CreateVideo", "inputs": {"images": ["13", 0], "fps": float(scenario.fps)}},
            "15": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["14", 0],
                    "filename_prefix": f"bench_{scenario.scenario_id}",
                    "format": "mp4",
                    "format.codec": "auto",
                },
            },
        }

    if scenario.comfy_template in ("minimax_h3_t2v", "minimax_h3_i2v"):
        cond_inputs: Dict[str, Any] = {
            "clip": ["2", 0],
            "vae": ["3", 0],
            "prompt": scenario.prompt,
            "width": scenario.width,
            "height": scenario.height,
            "length": int(scenario.num_frames),
        }
        graph: Dict[str, Dict[str, Any]] = {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": files["unet"], "weight_dtype": "default"}},
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": files["clip"], "type": "minimax", "device": "default"},
            },
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": files["video_vae"]}},
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": files["audio_vae"]}},
        }
        if scenario.comfy_template == "minimax_h3_i2v":
            graph["15"] = {"class_type": "LoadImage", "inputs": {"image": files["image"]}}
            cond_inputs["first_frame"] = ["15", 0]
        graph.update(
            {
                "5": {"class_type": "MiniMaxH3ImageToVideo", "inputs": cond_inputs},
                "6": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["5", 0]}},
                "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
                "8": {
                    "class_type": "BasicScheduler",
                    "inputs": {
                        "model": ["1", 0],
                        "scheduler": "simple",
                        "steps": int(scenario.steps),
                        "denoise": 1.0,
                    },
                },
                "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
                "10": {
                    "class_type": "SamplerCustomAdvanced",
                    "inputs": {
                        "noise": ["9", 0],
                        "guider": ["6", 0],
                        "sampler": ["7", 0],
                        "sigmas": ["8", 0],
                        "latent_image": ["5", 1],
                    },
                },
                "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
                "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}},
                "13": {
                    "class_type": "CreateVideo",
                    "inputs": {"images": ["11", 0], "audio": ["12", 0], "fps": float(scenario.fps)},
                },
                "14": {
                    "class_type": "SaveVideo",
                    "inputs": {
                        "video": ["13", 0],
                        "filename_prefix": f"bench_{scenario.scenario_id}",
                        "format": "mp4",
                        "format.codec": "auto",
                    },
                },
            }
        )
        return graph

    if scenario.comfy_template == "minimax_h3_t2i":
        return {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": files["unet"], "weight_dtype": "default"}},
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": files["clip"], "type": "minimax", "device": "default"},
            },
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": files["vae"]}},
            "4": {
                "class_type": "MiniMaxH3ImageConditioning",
                "inputs": {
                    "clip": ["2", 0],
                    "vae": ["3", 0],
                    "prompt": scenario.prompt,
                    "width": scenario.width,
                    "height": scenario.height,
                },
            },
            "5": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["4", 0]}},
            "6": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "7": {
                "class_type": "BasicScheduler",
                "inputs": {"model": ["1", 0], "scheduler": "simple", "steps": int(scenario.steps), "denoise": 1.0},
            },
            "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
            "9": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["8", 0],
                    "guider": ["5", 0],
                    "sampler": ["6", 0],
                    "sigmas": ["7", 0],
                    "latent_image": ["4", 1],
                },
            },
            "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
            "11": {
                "class_type": "SaveImage",
                "inputs": {"images": ["10", 0], "filename_prefix": f"bench_{scenario.scenario_id}"},
            },
        }

    raise ValueError(f"Unsupported ComfyUI benchmark template: {scenario.comfy_template}")
