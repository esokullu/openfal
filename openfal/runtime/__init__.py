"""
Real PyTorch / Diffusers / Transformers inference runtime with Strata 3-Tier Offloading,
ConvRot (INT8 Hadamard), and Scaled FP8 (E4M3) layers.
"""

from openfal.runtime.quant import (
    StrataBlockOffloadManager,
    StrataConvRotEmbedding,
    StrataConvRotLinear,
    StrataFP8ScaledLinear,
    build_hadamard_matrix,
    quantized_module_swap_from_state_dict,
)

__all__ = [
    "StrataBlockOffloadManager",
    "StrataConvRotEmbedding",
    "StrataConvRotLinear",
    "StrataFP8ScaledLinear",
    "build_hadamard_matrix",
    "quantized_module_swap_from_state_dict",
]
