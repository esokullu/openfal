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
import threading
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
        self._real_qwen_runner = None
        self._real_wan_runner = None
        self._real_minimax_runner = None
        self._active_real_family: Optional[str] = None
        # Schedulers, module residency and the memory hierarchy are shared per runner.
        self.execution_lock = threading.RLock()

    def evict(self) -> None:
        """Release real GPU allocations as well as the telemetry state."""
        for runtime in (self._real_qwen_runner, self._real_wan_runner, self._real_minimax_runner):
            if runtime is not None:
                runtime.evict()
        self._active_real_family = None

    def _evict_other_backends(self, next_family: str) -> None:
        if self._active_real_family and self._active_real_family != next_family:
            if self._active_real_family == "qwen_image_21" and self._real_qwen_runner is not None:
                self._real_qwen_runner.evict()
            elif self._active_real_family == "wan22_moe" and self._real_wan_runner is not None:
                self._real_wan_runner.evict()
            elif self._active_real_family == "minimax_h3" and self._real_minimax_runner is not None:
                self._real_minimax_runner.evict()
        self._active_real_family = next_family

    def _load_reference_images(self, urls: List[str], limit: int) -> list:
        if len(urls) > limit:
            raise ValueError(f"This endpoint supports at most {limit} reference images")
        images = []
        for index, url in enumerate(urls, 1):
            loaded = self.face_conditioner.load_image_from_url_or_path(url)
            if loaded is None:
                raise ValueError(f"Could not load reference image {index}; check its URL or uploaded file")
            images.append(loaded)
        return images

    def _try_execute_real(
        self,
        backend_family: str,
        partition: str,
        endpoint_id: str,
        payload: Dict[str, Any],
        prompt: str,
        width: int,
        height: int,
        duration: float,
        fps: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        ref_images: List[str],
        base_url: str,
        sync_mode: bool,
        on_log: Optional[Callable[[str], None]],
        t0: float,
    ) -> Optional[Dict[str, Any]]:
        from openfal.runtime.common import use_synthetic_fallback

        if use_synthetic_fallback():
            return None

        short_id = uuid.uuid4().hex[:10]
        base_url_clean = base_url.rstrip("/")

        if backend_family == "qwen_image_21":
            from openfal.runtime.qwen_image_runner import RealQwenImage21Runner

            if self._real_qwen_runner is None:
                self._real_qwen_runner = RealQwenImage21Runner(self.config, self.memory)
            if not self._real_qwen_runner.has_weights():
                return None

            self._evict_other_backends("qwen_image_21")
            ref_pil = self._load_reference_images(ref_images, 10)

            num_images = max(1, min(4, int(payload.get("num_images") or 1)))
            images_out = []
            for idx in range(num_images):
                fname = f"openfal_img_{short_id}_{idx}.png"
                out_path = self.config.output_dir / fname
                img = self._real_qwen_runner.generate(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_steps=num_steps,
                    seed=seed + idx,
                    guidance_scale=guidance_scale,
                    negative_prompt=payload.get("negative_prompt"),
                    reference_images=ref_pil or None,
                    output_path=out_path,
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
                        "width": img.size[0],
                        "height": img.size[1],
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
                    "mode": "real_pytorch",
                    "profile_id": self.config.profile_id,
                    "vram_counter_blocks": self.config.profile.resident_vram_blocks,
                    "ram_pantry_blocks": self.config.profile.total_dit_blocks - self.config.profile.resident_vram_blocks,
                },
            }

        if backend_family == "wan22_moe":
            from openfal.runtime.wan22_runner import RealWan22Runner

            if self._real_wan_runner is None:
                self._real_wan_runner = RealWan22Runner(self.config, self.memory)
            if not self._real_wan_runner.has_weights_for(partition):
                return None

            self._evict_other_backends("wan22_moe")
            start_url = (
                payload.get("start_image_url")
                or payload.get("image_url")
                or payload.get("face_image_url")
                or payload.get("source_face_url")
                or (ref_images[0] if ref_images else None)
            )
            start_img = self.face_conditioner.load_image_from_url_or_path(start_url) if start_url else None
            if start_url and start_img is None:
                raise ValueError("Could not load starting image")
            fname = f"openfal_vid_{short_id}.mp4"
            out_path = self.config.output_dir / fname
            width, height = max(256, width // 16 * 16), max(256, height // 16 * 16)
            num_frames = max(17, min(81, ((max(17, int(duration * fps)) - 1) // 4) * 4 + 1))
            self._real_wan_runner.generate_video(
                variant=partition,
                prompt=prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps,
                num_steps=num_steps,
                seed=seed,
                guidance_scale=guidance_scale,
                guidance_scale_2=payload.get("guidance_scale_2") if payload.get("guidance_scale_2") is not None else (
                    3.0 if partition == "Wan2.2-T2V-A14B" and payload.get("guidance_scale") is None else None
                ),
                negative_prompt=payload.get("negative_prompt") or "",
                start_image=start_img,
                output_path=out_path,
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
                    "duration": num_frames / fps,
                    "num_frames": num_frames,
                    "fps": fps,
                    "stereo_audio": False,
                },
                "seed": seed,
                "prompt": prompt,
                "timings": {"inference": elapsed},
                "strata_engine": {
                    "backend": backend_family,
                    "partition": partition,
                    "mode": "real_pytorch",
                    "profile_id": self.config.profile_id,
                    "vram_counter_blocks": self.config.profile.resident_vram_blocks,
                    "ram_pantry_blocks": self.config.profile.total_dit_blocks - self.config.profile.resident_vram_blocks,
                },
            }

        if backend_family == "minimax_h3":
            from openfal.runtime.minimax_h3_runner import RealMiniMaxH3Runner

            if self._real_minimax_runner is None:
                self._real_minimax_runner = RealMiniMaxH3Runner(self.config, self.memory)
            if not self._real_minimax_runner.has_weights_for(partition):
                return None

            self._evict_other_backends("minimax_h3")
            ref_pil = self._load_reference_images(ref_images, 16 if partition == "H3-Image" else 9)

            if partition == "H3-Image":
                num_images = max(1, min(4, int(payload.get("num_images") or 1)))
                images_out = []
                for idx in range(num_images):
                    fname = f"openfal_img_{short_id}_{idx}.png"
                    out_path = self.config.output_dir / fname
                    img = self._real_minimax_runner.generate_image(
                        prompt=prompt,
                        width=width,
                        height=height,
                        seed=seed + idx,
                        reference_images=ref_pil or None,
                        output_path=out_path,
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
                            "width": img.size[0],
                            "height": img.size[1],
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
                        "mode": "real_pytorch",
                        "profile_id": self.config.profile_id,
                        "vram_counter_blocks": self.config.profile.resident_vram_blocks,
                        "ram_pantry_blocks": self.config.profile.total_dit_blocks - self.config.profile.resident_vram_blocks,
                    },
                }

            start_url = payload.get("start_image_url") or payload.get("image_url")
            end_url = payload.get("end_image_url")
            decoded_images = dict(zip(ref_images, ref_pil))
            start_img = decoded_images[start_url] if start_url else None
            end_img = decoded_images[end_url] if end_url else None
            ref_vids = list(payload.get("reference_video_urls") or [])
            for vk in ("video_url", "target_video_url"):
                if payload.get(vk) and payload[vk] not in ref_vids:
                    ref_vids.append(payload[vk])
            ref_auds = list(payload.get("reference_audio_urls") or [])

            fname = f"openfal_vid_{short_id}.mp4"
            out_path = self.config.output_dir / fname
            enable_audio = bool(payload.get("enable_audio", self.config.enable_native_stereo_audio))
            requested_frames = max(22, int(round(duration * 24)))
            num_frames = ((requested_frames - 5 + 16) // 17) * 17 + 5
            width, height = max(256, round(width / 32) * 32), max(256, round(height / 32) * 32)
            self._real_minimax_runner.generate_video(
                partition=partition,
                prompt=prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_steps=num_steps,
                seed=seed,
                start_image=start_img,
                end_image=end_img,
                reference_images=ref_pil or None,
                reference_videos=ref_vids or None,
                reference_audios=ref_auds or None,
                output_path=out_path,
                enable_audio=enable_audio,
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
                    "duration": num_frames / 24,
                    "num_frames": num_frames,
                    "fps": 24,
                    "stereo_audio": enable_audio,
                },
                "seed": seed,
                "prompt": prompt,
                "timings": {"inference": elapsed},
                "strata_engine": {
                    "backend": backend_family,
                    "partition": partition,
                    "mode": "real_pytorch",
                    "profile_id": self.config.profile_id,
                    "vram_counter_blocks": self.config.profile.resident_vram_blocks,
                    "ram_pantry_blocks": self.config.profile.total_dit_blocks - self.config.profile.resident_vram_blocks,
                },
            }

        return None

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

    @staticmethod
    def resolve_backend_defaults(backend_family: str, partition: str) -> tuple[int, float]:
        """Return native `(num_inference_steps, guidance_scale)` defaults per model family."""
        if backend_family == "qwen_image_21":
            return 8, 1.0
        if backend_family == "wan22_moe":
            if "animate" in partition.lower():
                return 20, 4.0
            return 40, 3.5 if partition == "Wan2.2-I2V-A14B" else 4.0
        # MiniMax-H3 checkpoints are guidance-distilled; base FL2VA/Ref2VA use ~30 steps
        if partition == "H3-Image":
            return 6, 1.0
        return 30, 1.0

    def execute(
        self,
        endpoint_id: str,
        payload: Dict[str, Any],
        base_url: str = "http://localhost:8000",
        on_log: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        with self.execution_lock:
            try:
                return self._execute(endpoint_id, payload, base_url, on_log)
            except Exception:
                try:
                    self.evict()
                except Exception:
                    # Preserve the inference error, but discard partially loaded runners.
                    self._real_qwen_runner = None
                    self._real_wan_runner = None
                    self._real_minimax_runner = None
                    self._active_real_family = None
                self.memory.finish_generation()
                raise

    def _execute(
        self,
        endpoint_id: str,
        payload: Dict[str, Any],
        base_url: str = "http://localhost:8000",
        on_log: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        backend_family, partition = self.resolve_backend_and_partition(endpoint_id)
        is_image = self.is_image_endpoint(endpoint_id)
        default_steps, default_guidance = self.resolve_backend_defaults(backend_family, partition)

        prompt: str = payload.get("prompt") or "Cinematic high-definition shot, natural lighting"
        resolution: str = payload.get("resolution") or "768P"
        aspect_ratio: str = payload.get("aspect_ratio") or "16:9"
        duration: float = float(payload.get("duration") or 5.0)
        fps: int = int(payload.get("fps") or 16)
        num_steps: int = (
            int(payload["num_inference_steps"])
            if payload.get("num_inference_steps") is not None
            else default_steps
        )
        guidance_scale: float = (
            float(payload["guidance_scale"])
            if payload.get("guidance_scale") is not None
            else default_guidance
        )
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

        from openfal.runtime.common import use_synthetic_fallback

        if not use_synthetic_fallback():
            if backend_family == "wan22_moe" and int(duration * fps) > 81:
                raise ValueError("Wan currently supports at most 81 frames per request; lower duration or fps")
            if backend_family == "minimax_h3" and partition == "FL2VA":
                if any(payload.get(key) for key in ("reference_image_urls", "face_image_url", "source_face_url")):
                    raise ValueError(
                        "MiniMax FL2VA accepts start_image_url (or image_url) and end_image_url; "
                        "reference_image_urls and face references require a reference-to-video endpoint"
                    )
                if (
                    payload.get("start_image_url")
                    and payload.get("image_url")
                    and payload["start_image_url"] != payload["image_url"]
                ):
                    raise ValueError("start_image_url and its image_url alias must identify the same starting frame")
            if endpoint_id.endswith(("/edit", "/image-to-image", "/image-to-video")) and not ref_images:
                raise ValueError("This endpoint requires an input image")
            if backend_family == "minimax_h3" and endpoint_id.endswith(("/recast", "/relight", "/insert-video")) and not ref_videos:
                raise ValueError("This video editing endpoint requires video_url or target_video_url")
            if backend_family == "minimax_h3" and partition != "Ref2VA" and (ref_videos or ref_audios):
                raise ValueError("Video/audio references require a MiniMax reference-to-video endpoint")

        width, height = self.dit.resolve_dimensions(resolution, aspect_ratio)

        # If real PyTorch `.safetensors` checkpoints exist on disk, run genuine neural inference
        # (raising an actionable RuntimeError if inference deps are missing rather than faking).
        real_result = self._try_execute_real(
            backend_family=backend_family,
            partition=partition,
            endpoint_id=endpoint_id,
            payload=payload,
            prompt=prompt,
            width=width,
            height=height,
            duration=duration,
            fps=fps,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            ref_images=ref_images,
            base_url=base_url,
            sync_mode=sync_mode,
            on_log=on_log,
            t0=t0,
        )
        if real_result is not None:
            return real_result

        if not use_synthetic_fallback():
            component = {
                "Qwen-Image-2.1": "qwen_image_21",
                "Wan2.2-T2V-A14B": "wan22_t2v_moe",
                "Wan2.2-I2V-A14B": "wan22_i2v_moe",
                "Wan2.2-Animate-14B": "wan22_animate",
                "FL2VA": "fl2va", "Ref2VA": "ref2va", "H3-Image": "h3-image",
            }.get(partition, "all")
            raise RuntimeError(
                f"No usable model weights for {partition}. "
                f"Run: openfal download --component {component}. "
                "Synthetic rendering is only available with OPENFAL_SYNTHETIC_FALLBACK=1."
            )

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
                    "mode": "synthetic",
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
                "mode": "synthetic",
                "profile_id": self.config.profile_id,
                "vram_counter_blocks": self.config.profile.resident_vram_blocks,
                "ram_pantry_blocks": self.config.profile.total_dit_blocks - self.config.profile.resident_vram_blocks,
            },
        }
