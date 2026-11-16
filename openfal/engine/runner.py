"""
Unified Multi-Backend Inference Runner (`OpenFalRunner`) orchestrating the 3-stage Strata-DiT
pipeline across:

1. Primary Fast MoE Stack (Alibaba Wan 2.2 MoE 27B-A14B + Qwen-Image-2.1-Turbo 7B):
   - `wan/v2.2/text-to-video` & `wan/v2.2-a14b/text-to-video` (`Wan2.2-T2V-A14B` 2-Expert MoE)
   - `wan/v2.2/image-to-video` & `wan/v2.2-a14b/image-to-video` (`Wan2.2-I2V-A14B` 2-Expert MoE)
   - `wan/v2.2/animate` & `wan/v2.2/recast` (`Wan2.2-Animate-14B` Video Face Swap & Character Replacement)
   - `qwen/image-2.1/text-to-image` & `qwen-image-2.1/turbo` (`Qwen-Image-2.1-Turbo` 7B 8-Step T2I)
   - `qwen/image-2.1/edit` & `qwen-image-2.1/edit` (`Qwen-Image-2.1` 10-Reference Image & Face Edit)

2. Optional Omni + Native Stereo Audio Stack (`MiniMax-H3` 33B Dense DiT):
   - `minimax/h3/text-to-video` & `minimax/h3-max/text-to-video` (`FL2VA` partition)
   - `minimax/h3/image-to-video` & `minimax/h3-max/image-to-video` (`FL2VA` partition)
   - `minimax/h3/reference-to-video` & `minimax/h3-max/reference-to-video` (`Ref2VA` partition)
   - `minimax/h3-max/recast`, `relight`, `insert-video` (`Ref2VA` partition)
   - `minimax/h3/text-to-image` & `minimax/h3/image-to-image` (`H3-Image` partition)
"""

from __future__ import annotations

import base64
import random
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from openfal.config import EngineConfig
from openfal.engine.downloader import StrataModelManager
from openfal.engine.memory_hierarchy import StrataMemoryHierarchy
from openfal.engine.text_cache import TextEncoderCache
from openfal.models.face_swap import ReferenceFaceSwapConditioner
from openfal.models.hybrid_vae import TiledHybridViTVAE
from openfal.models.minimax_h3 import MiniMaxH3DiT
from openfal.models.qwen3_vl_encoder import StreamedQwen3VLEncoder
from openfal.models.qwen_image import QwenImage21DiT
from openfal.models.wan22_moe import Wan22MoEDiT


class OpenFalRunner:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.memory = StrataMemoryHierarchy(config)
        self.model_manager = StrataModelManager(config.weights_dir)
        self.text_cache = TextEncoderCache(config.cache_dir)
        self.qwen_encoder = StreamedQwen3VLEncoder(self.memory, self.text_cache)
        self.wan_moe_dit = Wan22MoEDiT(self.memory, config.weights_dir)
        self.qwen_image_dit = QwenImage21DiT(self.memory, config.weights_dir)
        self.dit = MiniMaxH3DiT(self.memory, config.weights_dir)
        self.face_conditioner = ReferenceFaceSwapConditioner(upload_dir=config.output_dir)
        self.vae = TiledHybridViTVAE(
            memory=self.memory,
            output_dir=config.output_dir,
            face_conditioner=self.face_conditioner,
        )

    @staticmethod
    def resolve_backend_and_partition(endpoint_id: str) -> tuple[str, str]:
        ep = endpoint_id.lower()
        if ep.startswith("qwen"):
            return "qwen_image_21", "Qwen-Image-2.1"
        if ep.startswith("wan"):
            if any(k in ep for k in ("animate", "recast", "face")):
                return "wan22_moe", "Wan2.2-Animate-14B"
            if "image-to-video" in ep or "i2v" in ep:
                return "wan22_moe", "Wan2.2-I2V-A14B"
            return "wan22_moe", "Wan2.2-T2V-A14B"
        # MiniMax-H3 optional backend
        if any(k in ep for k in ("text-to-image", "image-to-image", "h3-image", "/edit")):
            return "minimax_h3", "H3-Image"
        if any(k in ep for k in ("reference-to-video", "recast", "face-swap", "relight", "insert-video", "styles/")):
            return "minimax_h3", "Ref2VA"
        return "minimax_h3", "FL2VA"

    @classmethod
    def resolve_partition_for_endpoint(cls, endpoint_id: str) -> str:
        return cls.resolve_backend_and_partition(endpoint_id)[1]

    @staticmethod
    def is_image_endpoint(endpoint_id: str) -> bool:
        ep = endpoint_id.lower()
        return ep.startswith("qwen") or any(k in ep for k in ("text-to-image", "image-to-image", "h3-image", "/edit"))

    def execute(
        self,
        endpoint_id: str,
        payload: Dict[str, Any],
        base_url: str = "http://localhost:8000",
        on_log: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        prompt: str = payload.get("prompt") or "Cinematic high-definition shot, natural lighting"
        resolution: str = payload.get("resolution") or "768P"
        aspect_ratio: str = payload.get("aspect_ratio") or "16:9"
        duration: float = float(payload.get("duration") or 5.0)
        fps: int = int(payload.get("fps") or 16)
        num_steps: int = int(payload.get("num_inference_steps") or self.config.default_num_inference_steps)
        guidance_scale: float = float(payload.get("guidance_scale") or 4.5)
        expansion_mode: str = payload.get("prompt_expansion_mode") or "balanced"
        sync_mode: bool = bool(payload.get("sync_mode", False))
        seed: int = int(payload.get("seed") if payload.get("seed") is not None else random.randint(1, 2_147_483_647))

        ref_images: List[str] = list(payload.get("reference_image_urls") or [])
        for single_key in ("start_image_url", "image_url", "face_image_url", "source_face_url", "end_image_url"):
            val = payload.get(single_key)
            if val and val not in ref_images:
                ref_images.append(val)

        ref_videos: List[str] = list(payload.get("reference_video_urls") or [])
        for vkey in ("video_url", "target_video_url"):
            if payload.get(vkey) and payload[vkey] not in ref_videos:
                ref_videos.append(payload[vkey])

        ref_audios: List[str] = list(payload.get("reference_audio_urls") or [])
        all_refs = ref_images + ref_videos + ref_audios

        backend_family, partition = self.resolve_backend_and_partition(endpoint_id)
        is_image = self.is_image_endpoint(endpoint_id)
        width, height = self.dit.resolve_dimensions(resolution, aspect_ratio)

        # Stage 1: Streamed Text/Multimodal Encoding + Immediate VRAM Eviction
        ctx_embedding = self.qwen_encoder.encode(
            prompt=prompt,
            reference_urls=all_refs,
            task=partition.lower(),
            expansion_mode=expansion_mode,
            on_log=on_log,
        )

        start_img = None
        if payload.get("start_image_url") or payload.get("image_url"):
            start_img = self.face_conditioner.load_image_from_url_or_path(
                payload.get("start_image_url") or payload.get("image_url")
            )

        face_img = None
        identity_meta = {}
        if (
            partition in ("Ref2VA", "Wan2.2-Animate-14B", "Qwen-Image-2.1")
            or payload.get("face_image_url")
            or payload.get("source_face_url")
            or ref_images
        ):
            face_img, identity_meta = self.face_conditioner.extract_identity_descriptor(
                reference_image_urls=ref_images,
                source_face_url=payload.get("face_image_url") or payload.get("source_face_url"),
                on_log=on_log,
            )

        # Stage 2: Execute selected backend (Wan 2.2 MoE vs Qwen-Image-2.1 vs MiniMax-H3)
        num_frames = 1 if is_image else max(16, int(duration * fps))
        if backend_family == "qwen_image_21":
            latents = self.qwen_image_dit.sample_latents(
                context_embedding=ctx_embedding,
                width=width,
                height=height,
                num_steps=num_steps,
                seed=seed,
                guidance_scale=guidance_scale,
                num_references=len(ref_images),
                on_log=on_log,
            )
        elif backend_family == "wan22_moe":
            latents = self.wan_moe_dit.sample_latents(
                variant=partition,
                context_embedding=ctx_embedding,
                width=width,
                height=height,
                num_frames=num_frames,
                num_steps=num_steps,
                seed=seed,
                guidance_scale=guidance_scale,
                on_log=on_log,
            )
        else:
            latents = self.dit.sample_latents(
                partition=partition,
                context_embedding=ctx_embedding,
                width=width,
                height=height,
                num_frames=num_frames,
                num_steps=num_steps,
                seed=seed,
                guidance_scale=guidance_scale,
                on_log=on_log,
            )

        # Stage 3: Tiled VAE Decode (+ Stereo Audio Muxing for video)
        short_id = uuid.uuid4().hex[:10]
        base_url_clean = base_url.rstrip("/")

        if is_image:
            num_images = max(1, min(4, int(payload.get("num_images") or 1)))
            images_out = []
            for idx in range(num_images):
                fname = f"openfal_img_{short_id}_{idx}.png"
                out_path = self.vae.decode_image(
                    latents=latents,
                    prompt=prompt,
                    width=width,
                    height=height,
                    seed=seed + idx,
                    task_name=endpoint_id,
                    output_filename=fname,
                    start_image=start_img,
                    face_ref=face_img,
                    identity_meta=identity_meta,
                    on_log=on_log if idx == 0 else None,
                )
                file_bytes = out_path.read_bytes()
                url_val = (
                    "data:image/png;base64," + base64.b64encode(file_bytes).decode("ascii")
                    if sync_mode
                    else f"{base_url_clean}/files/{fname}"
                )
                images_out.append(
                    {
                        "url": url_val,
                        "file_name": fname,
                        "file_size": len(file_bytes),
                        "content_type": "image/png",
                        "width": width,
                        "height": height,
                    }
                )
            elapsed = round(time.perf_counter() - t0, 3)
            return {
                "images": images_out,
                "seed": seed,
                "prompt": prompt,
                "timings": {"inference": elapsed},
                "has_nsfw_concepts": [False] * len(images_out),
                "strata_engine": {
                    "backend": backend_family,
                    "partition": partition,
                    "profile_id": self.config.profile_id,
                    "vram_counter_blocks": self.config.profile.resident_vram_blocks,
                    "ram_pantry_blocks": self.config.profile.total_dit_blocks - self.config.profile.resident_vram_blocks,
                },
            }

        fname = f"openfal_vid_{short_id}.mp4"
        out_path = self.vae.decode_video_with_stereo_audio(
            latents=latents,
            prompt=prompt,
            width=width,
            height=height,
            duration_sec=duration,
            fps=fps,
            seed=seed,
            task_name=endpoint_id,
            output_filename=fname,
            start_image=start_img,
            face_ref=face_img,
            identity_meta=identity_meta,
            enable_audio=bool(payload.get("enable_audio", self.config.enable_native_stereo_audio)),
            on_log=on_log,
        )
        file_bytes = out_path.read_bytes()
        url_val = (
            "data:video/mp4;base64," + base64.b64encode(file_bytes).decode("ascii")
            if sync_mode
            else f"{base_url_clean}/files/{fname}"
        )

        elapsed = round(time.perf_counter() - t0, 3)
        return {
            "video": {
                "url": url_val,
                "file_name": fname,
                "file_size": len(file_bytes),
                "content_type": "video/mp4",
                "width": width,
                "height": height,
                "duration": duration,
                "fps": fps,
                "stereo_audio": True,
            },
            "seed": seed,
            "prompt": prompt,
            "timings": {"inference": elapsed},
            "strata_engine": {
                "backend": backend_family,
                "partition": partition,
                "profile_id": self.config.profile_id,
                "vram_counter_blocks": self.config.profile.resident_vram_blocks,
                "ram_pantry_blocks": self.config.profile.total_dit_blocks - self.config.profile.resident_vram_blocks,
            },
        }
