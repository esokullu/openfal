"""
Shared utilities for real PyTorch inference runners:
- Dependency verification (fail-fast with actionable pip instructions if weights exist)
- Auto-fetching tiny HF config/tokenizer JSON files (zero giant weight downloads)
- Device & dtype resolution (CUDA BF16 / MPS BF16-FP16 / CPU)
- Checkpoint loading (`safetensors.torch.load_file`) + file discovery helpers
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as safetensors_load_file

from openfal.config import EngineConfig


def check_inference_dependencies(packages: Iterable[str] = ("diffusers", "transformers", "accelerate")) -> None:
    """
    Ensure PyTorch neural inference dependencies are importable.

    Called only when real `.safetensors` weights are present on disk so CPU unit tests
    without downloaded weights do not require heavyweight ML libraries, while real runs
    never silently fall back to synthetic placeholders.
    """
    missing: List[str] = []
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise RuntimeError(
            "Real model weights are present on disk, but required inference packages are missing: "
            f"{', '.join(missing)}.\n"
            "Install them cleanly into this Python environment with:\n"
            '  pip install -e ".[inference]"   (or: pip install "diffusers>=0.43.0" "transformers>=5.3.0" '
            '"accelerate>=1.0.0" "sentencepiece>=0.2.0" "protobuf>=4.25.0" "einops>=0.7.0")'
        )

    # Guard against stale/ABI-mismatched optional torchaudio wheels (e.g. after upgrading torch for Blackwell)
    # triggering an OSError when transformers imports audio_utils for vision/text pipelines.
    try:
        import torchaudio  # noqa: F401
    except OSError:
        import transformers.utils as _tf_utils
        import transformers.utils.import_utils as _tf_import_utils

        _tf_utils.is_torchaudio_available = lambda: False
        _tf_import_utils.is_torchaudio_available = lambda: False
    except ImportError:
        pass


def resolve_compute_device_and_dtype() -> Tuple[torch.device, torch.dtype]:
    """Select the active compute device (`cuda`, `mps`, or `cpu`) and optimal native dtype."""
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.device("cuda", 0), dtype
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.bfloat16
    return torch.device("cpu"), torch.float32


def clear_device_cache(device: Optional[torch.device] = None) -> None:
    """Release cached allocator blocks back to VRAM/Unified Memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def ensure_hf_configs(
    repo_id: str,
    target_dir: Path,
    required_files: Iterable[str],
    allow_patterns: Iterable[str],
    subfolder: Optional[str] = None,
    on_log: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Ensure lightweight Hugging Face config/tokenizer files exist in `target_dir` without
    downloading any `.safetensors` checkpoints.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    missing = [rel for rel in required_files if not (target_dir / rel).exists()]
    if missing:
        if on_log:
            on_log(
                f"[HF-Config] Fetching {len(missing)} missing config/tokenizer metadata file(s) "
                f"from '{repo_id}' -> {target_dir}..."
            )
        patterns = list(allow_patterns)
        if subfolder:
            patterns = [f"{subfolder.strip('/')}/{p.lstrip('/')}" for p in patterns]
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            allow_patterns=patterns,
            ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.ckpt", "assets/*"],
        )
        # If snapshot fetched under subfolder/, promote or expose cleanly if needed
        if subfolder:
            sub_path = target_dir / subfolder.strip("/")
            return sub_path
    return (target_dir / subfolder.strip("/")) if subfolder else target_dir


def find_first_safetensors(root: Path, candidates: Iterable[str]) -> Optional[Path]:
    """Return the first existing safetensors file matching candidate relative paths or filename globs."""
    if not root.exists():
        return None
    for rel in candidates:
        p = root / rel
        if p.is_file():
            return p
        # Fallback match anywhere inside root by basename
        basename = Path(rel).name
        hits = list(root.rglob(basename))
        if hits:
            return hits[0]
    return None


def find_matching_safetensors(root: Path, substrings: Iterable[str]) -> Optional[Path]:
    """Find a `.safetensors` file inside `root` whose name contains all lowercase `substrings`."""
    if not root.exists():
        return None
    subs = [s.lower() for s in substrings]
    for p in sorted(root.rglob("*.safetensors")):
        name = p.name.lower()
        if all(s in name for s in subs):
            return p
    return None


def load_safetensors_state_dict(path: Path, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load a `.safetensors` checkpoint cleanly onto `device` (CPU by default for Strata staging)."""
    return safetensors_load_file(str(path), device=device)


def compute_resident_blocks(
    config: EngineConfig,
    total_blocks: int,
    estimated_model_vram_gb: float,
) -> int:
    """
    Compute how many Transformer blocks stay permanently resident in the GPU VRAM Counter
    while the rest stream asynchronously from Pinned Host RAM Pantry.
    """
    profile = config.profile
    if total_blocks <= 0:
        return 0
    # Leave ~6 GB VRAM headroom for KV cache, activations, and VAE staging
    usable_vram_gb = max(4.0, profile.total_vram_gb - 6.0)
    if estimated_model_vram_gb <= usable_vram_gb:
        return total_blocks

    # Scale proportionally from calibrated 50-block hardware profile
    ratio = profile.resident_vram_blocks / max(1, profile.total_dit_blocks)
    # Also respect memory fit ratio
    fit_ratio = usable_vram_gb / max(0.1, estimated_model_vram_gb)
    effective_ratio = min(ratio, fit_ratio) if ratio < 1.0 else fit_ratio
    resident = max(2, min(total_blocks, int(total_blocks * effective_ratio)))
    return resident


def use_synthetic_fallback() -> bool:
    """Explicit opt-in override for synthetic test rendering even if weights exist."""
    return os.environ.get("OPENFAL_SYNTHETIC_FALLBACK", "0") == "1"


def save_audio_wav(wav_path: Path, waveform: "Any", sample_rate: int = 32000) -> None:
    """Save float waveform `(channels, samples)` or `(samples, channels)` in [-1, 1] as 16-bit PCM WAV."""
    import wave
    import numpy as np

    if isinstance(waveform, torch.Tensor):
        arr = waveform.detach().float().cpu().numpy()
    else:
        arr = np.asarray(waveform, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim == 2 and arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
        arr = arr.T

    pcm16 = np.clip(arr * 32767.0, -32768.0, 32767.0).astype(np.int16)
    n_channels = pcm16.shape[1]
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())


def encode_frames_to_mp4(
    frames: List[Any],
    out_path: Path,
    fps: int = 16,
    audio_waveform: Optional[Any] = None,
    audio_sample_rate: int = 32000,
) -> Path:
    """Encode a list of PIL Images (and optional audio waveform) into an H.264/AAC MP4 container."""
    import shutil
    import subprocess
    import tempfile

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        try:
            import imageio_ffmpeg  # type: ignore

            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_bin = None
    if ffmpeg_bin:
        with tempfile.TemporaryDirectory(prefix="openfal_mp4_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            for idx, frame in enumerate(frames):
                frame.save(tmp_dir / f"frame_{idx:05d}.png")
            wav_path = tmp_dir / "audio.wav"
            has_audio = False
            if audio_waveform is not None:
                save_audio_wav(wav_path, audio_waveform, sample_rate=audio_sample_rate)
                has_audio = wav_path.exists()

            cmd = [
                ffmpeg_bin,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(tmp_dir / "frame_%05d.png"),
            ]
            if has_audio:
                cmd.extend(["-i", str(wav_path), "-c:a", "aac", "-b:a", "192k", "-shortest"])
            cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(out_path),
                ]
            )
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_path

    # Fallback using diffusers' built-in exporter when ffmpeg isn't on PATH
    from diffusers.utils import export_to_video

    export_to_video(frames, str(out_path), fps=fps)
    return out_path
