"""
openfal — Strata-Inspired 3-Tier Offload Inference Engine & fal.ai-Compatible Server
for Wan 2.2 MoE, Qwen-Image-2.1 & MiniMax H3 (Video, Image, Omni-Ref & Face Swap) on Consumer GPUs.

License: MIT
"""

__version__ = "0.1.0"
__license__ = "MIT"

from openfal.config import EngineConfig, HardwareProfile, HARDWARE_PROFILES, get_default_config
from openfal.client.sdk import subscribe, run, submit, status, result, upload_file, OpenFalClient

__all__ = [
    "__version__",
    "__license__",
    "EngineConfig",
    "HardwareProfile",
    "HARDWARE_PROFILES",
    "get_default_config",
    "subscribe",
    "run",
    "submit",
    "status",
    "result",
    "upload_file",
    "OpenFalClient",
]
