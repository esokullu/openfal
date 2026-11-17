"""Native Wan Animate animation/replacement with staged model residency."""

from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path
from types import MethodType
from typing import Any, Callable

import torch
from PIL import Image

from openfal.config import EngineConfig
from openfal.engine.memory_hierarchy import StrataMemoryHierarchy
from openfal.runtime.common import (
    clear_device_cache,
    encode_frames_to_mp4,
    resolve_compute_device_and_dtype,
)
from openfal.runtime.wan22_runner import (
    RealWan22Runner,
    WAN_DEFAULT_NEGATIVE_PROMPT,
    _materialize_remaining_meta,
    _validate_checkpoint_keys,
)


def _make_scheduler(config_path: Path):
    """Keep native Wan's sigma grid when the pipeline resets between segments."""
    import numpy as np
    from diffusers import UniPCMultistepScheduler

    class NativeWanAnimateScheduler(UniPCMultistepScheduler):
        def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None):
            if sigmas is None:
                if num_inference_steps is None:
                    raise ValueError("Wan Animate requires num_inference_steps or sigmas")
                maximum = float(np.float32(1.0 - 1.0 / self.config.num_train_timesteps))
                sigmas = np.linspace(maximum, 0.0, num_inference_steps + 1)[:-1]
            return super().set_timesteps(device=device, sigmas=sigmas, mu=mu)

    return NativeWanAnimateScheduler.from_pretrained(
        str(config_path), local_files_only=True, flow_shift=5.0,
    )


def _make_pipeline(**components):
    from diffusers import WanAnimatePipeline

    class StagedWanAnimatePipeline(WanAnimatePipeline):
        @property
        def _execution_device(self):
            # Models alternate between CPU and accelerator; the target stays fixed.
            return self._strata_device

    return StagedWanAnimatePipeline(**components)


def _motion_forward_with_cpu_basis(self, face_image, channel_dim=1):
    """Native motion decomposition using a precomputed CPU QR basis on Metal."""
    if face_image.shape[-2:] != (self.size, self.size):
        raise ValueError(f"Wan Animate face crops must be {self.size}x{self.size}")
    value = self.conv_in(face_image, channel_dim)
    for block in self.res_blocks:
        value = block(value, channel_dim)
    value = self.conv_out(value, channel_dim).squeeze(-1).squeeze(-1)
    for linear in self.motion_network:
        value = linear(value, channel_dim=channel_dim)
    basis = self._openfal_motion_basis.to(device=value.device)
    decomposed = torch.matmul(torch.diag_embed(value.float()), basis.T)
    return decomposed.sum(dim=1).to(dtype=value.dtype)


def _install_mps_motion_basis(motion_encoder):
    # QR has no kernel on older Metal runtimes. The basis depends only on frozen
    # weights, so this exact small CPU calculation is done once during loading.
    weight = (motion_encoder.motion_synthesis_weight.detach().cpu() + 1e-8).float()
    motion_encoder.register_buffer("_openfal_motion_basis", torch.linalg.qr(weight)[0], persistent=False)
    motion_encoder.forward = MethodType(_motion_forward_with_cpu_basis, motion_encoder)


class _AnimateStages:
    """Offload complete encoders/VAE around the block-streamed transformer."""

    def __init__(self, shared_runner, pipeline, transformer, offloader, device, on_log=None):
        self.shared = shared_runner
        self.pipeline = pipeline
        self.transformer = transformer
        self.offloader = offloader
        self.device = device
        self.on_log = on_log
        self.current = None

    def release(self):
        current, self.current = self.current, None
        if current == "transformer":
            self.shared._deactivate_expert(self.transformer, self.offloader, self.device)
        elif current is not None:
            model = getattr(self.pipeline, current)
            if current == "vae" and hasattr(model, "clear_cache"):
                model.clear_cache()
            model.to("cpu")
            if current == "text_encoder":
                self.shared.memory.evict_qwen3_vl_stage()
            clear_device_cache(self.device)

    def activate(self, name):
        if self.current == name:
            return
        self.release()
        # Track the component before moving it so failed allocations also clean up.
        self.current = name
        if name == "transformer":
            self.shared._activate_expert(self.transformer, self.offloader, self.device)
        else:
            getattr(self.pipeline, name).to(self.device)
            if name == "text_encoder":
                self.shared.memory.begin_qwen3_vl_stage(cache_hit=False)
            elif name == "vae":
                self.shared.memory.begin_vae_stage()
        if self.on_log:
            label = {"text_encoder": "UMT5 prompt encoding", "image_encoder": "CLIP reference encoding",
                     "vae": "VAE conditioning/decode", "transformer": "pose/face-conditioned denoising"}[name]
            self.on_log(f"[Wan Animate] {label}")


@contextmanager
def _managed_pipeline_calls(pipeline, stages):
    """Scope method wrappers to one call; always restore shared model methods."""
    previous = []
    absent = object()

    def wrap(owner, name, stage, release_after=False):
        previous.append((owner, name, owner.__dict__.get(name, absent)))
        original = getattr(owner, name)

        def managed(*args, **kwargs):
            stages.activate(stage)
            try:
                return original(*args, **kwargs)
            finally:
                if release_after:
                    stages.release()

        setattr(owner, name, managed)

    handle = None
    try:
        wrap(pipeline, "encode_prompt", "text_encoder", release_after=True)
        wrap(pipeline, "encode_image", "image_encoder", release_after=True)
        wrap(pipeline.vae, "encode", "vae")
        wrap(pipeline.vae, "decode", "vae")
        handle = pipeline.transformer.register_forward_pre_hook(lambda *_: stages.activate("transformer"))
        yield
    finally:
        if handle is not None:
            handle.remove()
        for owner, name, original in reversed(previous):
            if original is absent:
                delattr(owner, name)
            else:
                setattr(owner, name, original)
        stages.release()


def _validate_conditioning(image, pose_video, face_video, background_video, mask_video,
                           width, height, fps, mode, segment_frame_length, overlap):
    if not isinstance(image, Image.Image):
        raise ValueError("Wan Animate requires a reference image")
    if mode not in ("animate", "replace"):
        raise ValueError("Wan Animate mode must be 'animate' or 'replace'")
    if width < 16 or height < 16 or width % 16 or height % 16:
        raise ValueError("Wan Animate width and height must be positive multiples of 16")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Wan Animate fps must be positive and finite")
    if overlap not in (1, 5):
        raise ValueError("Wan Animate previous-segment conditioning must use 1 or 5 frames")
    if segment_frame_length != 77:
        raise ValueError("Wan Animate segment length must be 77; use duration to request a shorter video")
    videos = {"pose_video": pose_video, "face_video": face_video}
    if mode == "replace":
        videos.update(background_video=background_video, mask_video=mask_video)
    elif background_video is not None or mask_video is not None:
        raise ValueError("Background and mask videos are only used in replace mode")
    for name, frames in videos.items():
        if not isinstance(frames, list) or not frames or not all(isinstance(frame, Image.Image) for frame in frames):
            raise ValueError(f"Wan Animate {name} must contain PIL image frames")
        if len(frames) != len(pose_video):
            raise ValueError("Wan Animate conditioning videos must have equal frame counts")
    if len(pose_video) <= overlap:
        raise ValueError("Wan Animate driving video must have more frames than the segment overlap")


class RealWanAnimateRunner:
    """Execute real character animation or masked replacement from motion inputs."""

    def __init__(self, config: EngineConfig, memory: StrataMemoryHierarchy,
                 shared_runner: RealWan22Runner | None = None):
        self.config = config
        self.memory = memory
        self._wan = shared_runner or RealWan22Runner(config, memory)
        self._pipe = None
        self._offloader = None
        self._stages = None

    def has_weights(self) -> bool:
        return self._wan.has_weights_for("Wan2.2-Animate-14B")

    def has_weights_for(self, variant="Wan2.2-Animate-14B") -> bool:
        return variant == "Wan2.2-Animate-14B" and self.has_weights()

    def _ensure_loaded(self, on_log=None):
        if self._pipe is not None:
            return
        from transformers import CLIPImageProcessor, CLIPVisionModel

        cfg_dir = self._wan._ensure_configs("Wan2.2-Animate-14B", on_log=on_log)
        path, _ = self._wan._resolve_expert_paths("Wan2.2-Animate-14B")
        if path is None:
            raise RuntimeError("Wan Animate weights missing. Run: openfal download --component wan22_animate")
        self._wan._load_shared_text_encoder(cfg_dir, on_log=on_log)
        self._wan._load_shared_vae(cfg_dir, on_log=on_log)
        transformer, self._offloader = self._wan._get_expert(
            "Wan2.2-Animate-14B", path, cfg_dir, on_log=on_log,
        )
        device, dtype = resolve_compute_device_and_dtype()
        if device.type == "mps":
            _install_mps_motion_basis(transformer.motion_encoder)
        clip, loading = CLIPVisionModel.from_pretrained(
            str(cfg_dir / "image_encoder"), local_files_only=True, dtype=dtype,
            output_loading_info=True,
        )
        _validate_checkpoint_keys(loading["missing_keys"], loading["unexpected_keys"], "Wan Animate CLIP")
        if loading.get("mismatched_keys") or loading.get("error_msgs"):
            raise RuntimeError(f"Wan Animate CLIP checkpoint is incompatible: {loading}")
        _materialize_remaining_meta(clip, dtype)
        clip.eval()
        image_processor = CLIPImageProcessor.from_pretrained(
            str(cfg_dir / "image_processor"), local_files_only=True,
        )
        self._pipe = _make_pipeline(
            tokenizer=self._wan._tokenizer, text_encoder=self._wan._text_encoder,
            vae=self._wan._vae, scheduler=_make_scheduler(cfg_dir / "scheduler"),
            image_processor=image_processor, image_encoder=clip, transformer=transformer,
        )
        self._pipe._strata_device = device
        self._pipe.set_progress_bar_config(disable=True)

    @torch.inference_mode()
    def generate_video(
        self, *, image: Image.Image, pose_video: list[Image.Image], face_video: list[Image.Image],
        prompt: str, width: int, height: int, fps: float, seed: int, output_path: Path,
        num_steps: int = 20, mode: str = "animate",
        background_video: list[Image.Image] | None = None, mask_video: list[Image.Image] | None = None,
        segment_frame_length: int = 77, prev_segment_conditioning_frames: int = 1,
        motion_encode_batch_size: int = 1, guidance_scale: float = 1.0, negative_prompt: str = "",
        on_log: Callable[[str], None] | None = None,
    ) -> Path:
        _validate_conditioning(
            image, pose_video, face_video, background_video, mask_video, width, height, fps,
            mode, segment_frame_length, prev_segment_conditioning_frames,
        )
        if num_steps < 1 or motion_encode_batch_size < 1 or not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("Wan Animate requires positive steps/batch size and finite nonnegative guidance")
        self._ensure_loaded(on_log=on_log)
        if self.memory.sleeping:
            self.memory.wake()
        self.memory.switch_partition("Wan2.2-Animate-14B")
        device, _ = resolve_compute_device_and_dtype()
        generator = torch.Generator(device="cpu" if device.type == "mps" else device).manual_seed(int(seed))
        pipe = self._pipe
        pipe._strata_device = device
        stages = _AnimateStages(self._wan, pipe, pipe.transformer, self._offloader, device, on_log)
        self._stages = stages
        effective_length = segment_frame_length - prev_segment_conditioning_frames
        segments = math.ceil((len(pose_video) - prev_segment_conditioning_frames) / effective_length)
        completed_steps = 0

        def progress(_pipeline, _index, _timestep, callback_kwargs):
            nonlocal completed_steps
            completed_steps += 1
            if on_log and (completed_steps == 1 or completed_steps % 5 == 0 or completed_steps == segments * num_steps):
                on_log(f"[Wan Animate] Denoising step {completed_steps}/{segments * num_steps}")
            return callback_kwargs

        try:
            with _managed_pipeline_calls(pipe, stages):
                result = pipe(
                    image=image.convert("RGB"), pose_video=pose_video, face_video=face_video,
                    background_video=background_video, mask_video=mask_video,
                    prompt=prompt, negative_prompt=negative_prompt or WAN_DEFAULT_NEGATIVE_PROMPT,
                    height=height, width=width, mode=mode,
                    segment_frame_length=segment_frame_length,
                    prev_segment_conditioning_frames=prev_segment_conditioning_frames,
                    motion_encode_batch_size=motion_encode_batch_size,
                    num_inference_steps=num_steps, guidance_scale=guidance_scale, generator=generator,
                    output_type="pil", callback_on_step_end=progress,
                )
            frames = result.frames[0]
            if len(frames) != len(pose_video):
                raise RuntimeError(f"Wan Animate returned {len(frames)} frames for {len(pose_video)} conditioning frames")
            if any(frame.size != (width, height) for frame in frames):
                raise RuntimeError("Wan Animate output dimensions do not match the requested canvas")
            return encode_frames_to_mp4(frames, Path(output_path), fps=fps)
        finally:
            self._stages = None
            self.memory.finish_generation()

    def evict(self):
        if self._stages is not None:
            self._stages.release()
        self._wan.evict()
        if self._pipe is not None:
            self._pipe.image_encoder.to("cpu")
        device, _ = resolve_compute_device_and_dtype()
        clear_device_cache(device)
