"""
Pydantic Schemas matching fal.ai's MiniMax H3 & H3 Max API specification.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ResolutionEnum(str, Enum):
    R_480P = "480P"
    R_768P = "768P"
    R_1080P = "1080P"
    R_2K = "2K"
    R_4K = "4K"


class AspectRatioEnum(str, Enum):
    ADAPTIVE = "adaptive"
    AR_21_9 = "21:9"
    AR_16_9 = "16:9"
    AR_4_3 = "4:3"
    AR_1_1 = "1:1"
    AR_3_4 = "3:4"
    AR_9_16 = "9:16"


class PromptExpansionMode(str, Enum):
    BALANCED = "balanced"
    FAST = "fast"
    QUALITY = "quality"
    NONE = "none"


class MiniMaxH3Input(BaseModel):
    prompt: str = Field(
        default="Cinematic drone shot rolling over a misty mountain ridge at dawn, sweeping motion, wind howling.",
        description="Text prompt describing the shot, action, visual style, and stereo soundscape cues.",
    )
    duration: float = Field(default=5.0, ge=1.0, le=15.0, description="Video length in seconds (5 to 15s).")
    resolution: ResolutionEnum = Field(default=ResolutionEnum.R_768P)
    aspect_ratio: AspectRatioEnum = Field(default=AspectRatioEnum.AR_16_9)
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility.")
    fps: int = Field(default=16, ge=8, le=60)
    num_inference_steps: int = Field(default=8, ge=1, le=50)
    guidance_scale: float = Field(default=4.5, ge=1.0, le=15.0)
    prompt_expansion_mode: PromptExpansionMode = Field(default=PromptExpansionMode.BALANCED)
    enable_safety_checker: bool = Field(default=True)
    enable_audio: bool = Field(default=True, description="Synthesize synchronized native stereo audio.")
    sync_mode: bool = Field(default=False, description="Return inline base64 data URI instead of local URL.")
    num_images: int = Field(default=1, ge=1, le=4, description="Number of images for H3-Image endpoints.")

    # Multimodal & Reference Conditioning (I2V, Reference-to-Video, Video Face Swap / Recast)
    start_image_url: Optional[str] = Field(default=None, description="Opening frame URL or data URI for I2V.")
    end_image_url: Optional[str] = Field(default=None, description="Optional closing frame URL for FL2VA.")
    image_url: Optional[str] = Field(default=None, description="Alias for start_image_url or edit input image.")
    face_image_url: Optional[str] = Field(
        default=None,
        description="Reference face/identity image URL or data URI for video/image face swap (recast).",
    )
    source_face_url: Optional[str] = Field(default=None, description="Alias for face_image_url.")
    target_video_url: Optional[str] = Field(
        default=None,
        description="Target video URL for minimax/h3-max/recast (video face swap) or relight.",
    )
    video_url: Optional[str] = Field(default=None, description="Alias for target_video_url.")
    reference_image_urls: List[str] = Field(
        default_factory=list,
        description="Up to 9 reference images for subject/face consistency and omni-reference conditioning.",
    )
    reference_video_urls: List[str] = Field(
        default_factory=list,
        description="Up to 3 reference video clips for camera/motion transfer.",
    )
    reference_audio_urls: List[str] = Field(
        default_factory=list,
        description="Up to 3 reference audio clips for voice/timbre matching.",
    )

    model_config = {"extra": "allow"}


class FalLogEntry(BaseModel):
    message: str
    level: str = "INFO"
    source: str = "openfal-strata"
    timestamp: str


class FalQueueStatusResponse(BaseModel):
    status: str  # "IN_QUEUE", "IN_PROGRESS", "COMPLETED", "FAILED"
    request_id: str
    response_url: str
    status_url: str
    cancel_url: str
    queue_position: Optional[int] = None
    logs: Optional[List[FalLogEntry]] = None
    metrics: Optional[Dict[str, Any]] = None


class ProfileSwitchRequest(BaseModel):
    profile_id: str = Field(..., description="Hardware profile ID (e.g., dual_5060ti_32gb, single_5090_32gb)")
    resident_vram_blocks: Optional[int] = Field(
        default=None,
        ge=2,
        le=50,
        description="Optional override for number of MiniMax-H3 DiT blocks pinned in VRAM Counter.",
    )
