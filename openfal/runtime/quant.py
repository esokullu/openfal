"""
MIT-licensed PyTorch quantized linear/embedding layers and Strata 3-Tier asynchronous
block streaming offloader.

Supports:
1. `int8_convrot`: Regular Hadamard activation/embedding rotation (`group_size=256`) paired with
   `torch.int8` weights and per-output-channel (or scalar) `float32` scales — matching the
   pre-quantized `Comfy-Org` checkpoints (`Qwen-Image-2.1`, `Qwen3-VL`, `MiniMax-H3`) while
   keeping weights 8-bit in GPU VRAM and pinned CPU RAM.
2. `fp8_scaled`: `torch.float8_e4m3fn` weights paired with `weight_scale` / `scale_weight`
   `float32` scaling factors (`Wan 2.2`, `MiniMax-H3`).
3. `StrataBlockOffloadManager`: Keeps `resident_vram_blocks` permanently on GPU (Counter) and
   streams remaining DiT blocks from read-only pinned host CPU RAM (Pantry) via async CUDA
   H2D `Stream` prefetching — evicting by restoring O(1) pinned CPU tensor handles with zero
   D2H copy overhead.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_HADAMARD_CACHE: Dict[Tuple[int, str, str], torch.Tensor] = {}


def build_hadamard_matrix(
    group_size: int = 256,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Construct the normalized regular orthogonal Hadamard matrix H_G used by ConvRot.

    For powers of 4 (including group_size=256), matches `comfy_kitchen` via Kronecker
    powers of the regular H_4 block:
        [[ 1,  1,  1, -1],
         [ 1,  1, -1,  1],
         [ 1, -1,  1,  1],
         [-1,  1,  1,  1]] / sqrt(G)
    so activations/outliers spread uniformly across rows while remaining symmetric orthogonal
    (H_G^T == H_G and H_G @ H_G^T == I_G).
    """
    if group_size <= 0 or (group_size & (group_size - 1)) != 0:
        raise ValueError(f"group_size must be a positive power of 2, got {group_size}")

    dev = torch.device(device) if device is not None else torch.device("cpu")
    key = (group_size, str(dev), str(dtype))
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached

    if group_size >= 4 and int(round(math.log(group_size, 4))) ** 2 >= 0 and (4 ** int(round(math.log(group_size, 4))) == group_size):
        h4 = torch.tensor(
            [[1.0, 1.0, 1.0, -1.0], [1.0, 1.0, -1.0, 1.0], [1.0, -1.0, 1.0, 1.0], [-1.0, 1.0, 1.0, 1.0]],
            dtype=torch.float32,
        )
        cur = h4
        size = 4
        while size < group_size:
            cur = torch.kron(cur, h4)
            size *= 4
        cur = cur / math.sqrt(float(group_size))
    else:
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        h2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float32) * inv_sqrt2
        cur = h2
        size = 2
        while size < group_size:
            cur = torch.kron(h2, cur)
            size *= 2

    out = cur.to(device=dev, dtype=dtype).contiguous()
    _HADAMARD_CACHE[key] = out
    return out


def apply_group_hadamard(x: torch.Tensor, group_size: int = 256) -> torch.Tensor:
    """Apply block-diagonal orthogonal Hadamard transform along the last dimension."""
    if group_size <= 1 or x.shape[-1] % group_size != 0:
        return x
    h = build_hadamard_matrix(group_size=group_size, device=x.device, dtype=x.dtype)
    orig_shape = x.shape
    return (x.reshape(-1, group_size) @ h).reshape(orig_shape)


def _norm_scale_tensor(scale: Optional[torch.Tensor], out_features: int) -> torch.Tensor:
    if scale is None:
        return torch.ones((1, 1), dtype=torch.float32)
    s = scale.detach().to(dtype=torch.float32)
    if s.ndim == 0:
        return s.reshape(1, 1)
    if s.ndim == 1:
        if s.numel() == 1:
            return s.reshape(1, 1)
        if s.numel() == out_features:
            return s.reshape(out_features, 1)
    return s.reshape(out_features, -1) if s.shape[0] == out_features else s


class StrataConvRotLinear(nn.Module):
    """
    8-bit INT8 weight-resident Linear layer with optional ConvRot activation rotation.

    Parameters remain stored as `torch.int8` (1 byte/weight) in CPU RAM or GPU VRAM and are
    scaled to the activation dtype on the fly during forward computation.
    """

    def __init__(
        self,
        weight_i8: torch.Tensor,
        weight_scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        convrot: bool = True,
        group_size: int = 256,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if weight_i8.ndim != 2:
            raise ValueError(f"Expected 2D weight tensor, got shape {tuple(weight_i8.shape)}")
        self.out_features, self.in_features = weight_i8.shape
        self.convrot = bool(convrot and (self.in_features % group_size == 0))
        self.group_size = int(group_size)
        self._dtype_anchor = nn.Parameter(torch.empty(0, dtype=compute_dtype), requires_grad=False)

        self.register_buffer("weight_i8", weight_i8.detach().to(torch.int8).contiguous(), persistent=True)
        self.register_buffer(
            "weight_scale",
            _norm_scale_tensor(weight_scale, self.out_features).contiguous(),
            persistent=True,
        )
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

    def dequantized_weight(self, dtype: torch.dtype) -> torch.Tensor:
        w = self.weight_i8.to(dtype=dtype) * self.weight_scale.to(device=self.weight_i8.device, dtype=dtype)
        if self.convrot:
            # Since W_rot = W_orig @ H_G and H_G is symmetric orthogonal, W_orig = W_rot @ H_G
            w = apply_group_hadamard(w, group_size=self.group_size)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Rotate input x along K -> compute dense INT8*scale matmul in compute dtype without permanently
        # rotating W (equivalent because (x H_G) (W_rot)^T = x H_G H_G^T W_orig^T = x W_orig^T).
        x_in = apply_group_hadamard(x, group_size=self.group_size) if self.convrot else x
        w_deq = self.weight_i8.to(device=x.device, dtype=x.dtype) * self.weight_scale.to(
            device=x.device, dtype=x.dtype
        )
        b = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return F.linear(x_in, w_deq, b)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, convrot={self.convrot}, group_size={self.group_size}"
        )


class StrataConvRotEmbedding(nn.Module):
    """8-bit INT8 Embedding table supporting ConvRot unrotation on looked-up rows."""

    def __init__(
        self,
        weight_i8: torch.Tensor,
        weight_scale: Optional[torch.Tensor] = None,
        convrot: bool = True,
        group_size: int = 256,
        padding_idx: Optional[int] = None,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_embeddings, self.embedding_dim = weight_i8.shape
        self.convrot = bool(convrot and (self.embedding_dim % group_size == 0))
        self.group_size = int(group_size)
        self.padding_idx = padding_idx
        self.target_dtype: torch.dtype = compute_dtype
        self._dtype_anchor = nn.Parameter(torch.empty(0, dtype=compute_dtype), requires_grad=False)

        self.register_buffer("weight_i8", weight_i8.detach().to(torch.int8).contiguous(), persistent=True)
        self.register_buffer(
            "weight_scale",
            _norm_scale_tensor(weight_scale, self.num_embeddings).contiguous(),
            persistent=True,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        rows_i8 = F.embedding(input_ids, self.weight_i8, padding_idx=self.padding_idx)
        compute_dtype = self.target_dtype
        if self.weight_scale.shape[0] == self.num_embeddings:
            scales = F.embedding(input_ids, self.weight_scale)
        else:
            scales = self.weight_scale
        rows = rows_i8.to(dtype=compute_dtype) * scales.to(device=rows_i8.device, dtype=compute_dtype)
        if self.convrot:
            rows = apply_group_hadamard(rows, group_size=self.group_size)
        return rows


class StrataFP8ScaledLinear(nn.Module):
    """
    1-byte-per-weight FP8 E4M3 Linear layer with `weight_scale` / `scale_weight` support.

    Compatible with `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` (`scale_weight`) and
    `Comfy-Org/MiniMax-H3` (`weight_scale`).
    """

    def __init__(
        self,
        weight_fp8: torch.Tensor,
        weight_scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if weight_fp8.ndim != 2:
            raise ValueError(f"Expected 2D FP8 weight tensor, got shape {tuple(weight_fp8.shape)}")
        self.out_features, self.in_features = weight_fp8.shape
        self._dtype_anchor = nn.Parameter(torch.empty(0, dtype=compute_dtype), requires_grad=False)
        self.register_buffer("weight_fp8", weight_fp8.detach().contiguous(), persistent=True)
        self.register_buffer(
            "weight_scale",
            _norm_scale_tensor(weight_scale, self.out_features).contiguous(),
            persistent=True,
        )
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

    def to(self, *args, **kwargs):
        # MPS currently lacks native float8_e4m3fn storage buffers on some PyTorch builds;
        # keep `weight_fp8` safely on CPU when targeting MPS and cast on demand in `forward`.
        target_device = kwargs.get("device")
        if target_device is None and args:
            for a in args:
                if isinstance(a, (torch.device, str)):
                    target_device = torch.device(a)
                    break
        if (
            target_device is not None
            and torch.device(target_device).type == "mps"
            and self.weight_fp8.dtype == torch.float8_e4m3fn
        ):
            self.weight_scale = self.weight_scale.to(device=target_device)
            if self.bias is not None:
                self.bias = nn.Parameter(self.bias.detach().to(device=target_device), requires_grad=False)
            return self
        return super().to(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_src = self.weight_fp8
        if w_src.device != x.device:
            w_deq = w_src.to(dtype=x.dtype).to(device=x.device)
        else:
            w_deq = w_src.to(dtype=x.dtype)
        w_deq = w_deq * self.weight_scale.to(device=x.device, dtype=x.dtype)
        b = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w_deq, b)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, dtype={self.weight_fp8.dtype}"
        )


def parse_comfy_quant_meta(raw: Optional[torch.Tensor]) -> Dict[str, Any]:
    """Decode Comfy's uint8 UTF-8 JSON tensor metadata (`*.comfy_quant`)."""
    if raw is None:
        return {}
    try:
        arr = raw.detach().cpu().to(torch.uint8).numpy().tobytes()
        return json.loads(arr.decode("utf-8"))
    except Exception:
        return {}


def _get_parent_and_attr(root: nn.Module, dotted_name: str) -> Tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    cur: nn.Module = root
    for p in parts[:-1]:
        cur = getattr(cur, p)
    return cur, parts[-1]


def quantized_module_swap_from_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    compute_dtype: torch.dtype = torch.bfloat16,
    embedding_names: Iterable[str] = (),
) -> Tuple[List[str], List[str]]:
    """
    Replace `nn.Linear` (and optionally `nn.Embedding`) modules inside `model` with
    `StrataConvRotLinear` / `StrataFP8ScaledLinear` / `StrataConvRotEmbedding` wherever
    the checkpoint `state_dict` stores quantized weights, and load all remaining unquantized
    parameters directly into `model`.

    Returns `(missing_keys, unexpected_keys)` relative to remaining modules.
    """
    consumed: set[str] = set()
    handled_missing: set[str] = set()
    emb_set = set(embedding_names)
    swapped = 0

    module_map = dict(model.named_modules())
    for mod_name, mod in module_map.items():
        w_key = f"{mod_name}.weight" if mod_name else "weight"
        if w_key not in state_dict:
            continue
        w_tensor = state_dict[w_key]
        scale_key = None
        for candidate in (f"{mod_name}.weight_scale", f"{mod_name}.scale_weight"):
            if candidate in state_dict:
                scale_key = candidate
                break
        qmeta_key = f"{mod_name}.comfy_quant"
        qmeta = parse_comfy_quant_meta(state_dict.get(qmeta_key))
        is_i8 = w_tensor.dtype == torch.int8
        is_fp8 = w_tensor.dtype in (
            getattr(torch, "float8_e4m3fn", torch.int8),
            getattr(torch, "float8_e5m2", torch.int8),
        ) and w_tensor.dtype != torch.int8

        b_key = f"{mod_name}.bias"
        b_tensor = state_dict.get(b_key)
        s_tensor = state_dict.get(scale_key) if scale_key else None

        is_convrot = bool(qmeta.get("convrot", is_i8))
        group_size = int(qmeta.get("convrot_groupsize", 256))

        if (isinstance(mod, nn.Embedding) or mod_name in emb_set) and is_i8:
            parent, attr = _get_parent_and_attr(model, mod_name)
            rep = StrataConvRotEmbedding(
                weight_i8=w_tensor,
                weight_scale=s_tensor,
                convrot=is_convrot,
                group_size=group_size,
                padding_idx=getattr(mod, "padding_idx", None),
                compute_dtype=compute_dtype,
            )
            setattr(parent, attr, rep)
            consumed.update(k for k in (w_key, scale_key, qmeta_key) if k)
            swapped += 1
        elif isinstance(mod, nn.Linear) and is_i8:
            parent, attr = _get_parent_and_attr(model, mod_name)
            rep = StrataConvRotLinear(
                weight_i8=w_tensor,
                weight_scale=s_tensor,
                bias=b_tensor.to(compute_dtype) if b_tensor is not None else None,
                convrot=is_convrot,
                group_size=group_size,
                compute_dtype=compute_dtype,
            )
            setattr(parent, attr, rep)
            consumed.update(k for k in (w_key, scale_key, qmeta_key, b_key) if k and k in state_dict)
            if b_key in state_dict:
                handled_missing.add(b_key)
            swapped += 1
        elif isinstance(mod, nn.Linear) and is_fp8:
            parent, attr = _get_parent_and_attr(model, mod_name)
            rep = StrataFP8ScaledLinear(
                weight_fp8=w_tensor,
                weight_scale=s_tensor,
                bias=b_tensor.to(compute_dtype) if b_tensor is not None else None,
                compute_dtype=compute_dtype,
            )
            setattr(parent, attr, rep)
            for k in (
                w_key,
                scale_key,
                qmeta_key,
                b_key,
                f"{mod_name}.scale_input",
                f"{mod_name}.input_scale",
            ):
                if k and k in state_dict:
                    consumed.add(k)
            if b_key in state_dict:
                handled_missing.add(b_key)
            swapped += 1

    # Load all remaining unquantized tensors (norms, RoPE tables, biases, BF16/FP16 layers, etc.)
    keep_fp32 = list(getattr(model, "_keep_in_fp32_modules", None) or [])
    if "quant_conv" in keep_fp32 and "decoder.proj_in" not in keep_fp32:
        keep_fp32.append("decoder.proj_in")

    remainder: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k in consumed:
            continue
        if k.endswith((".comfy_quant", ".config", ".scale_input", ".input_scale", "scaled_fp8")):
            continue
        if v.is_floating_point():
            target_dt = torch.float32 if any(pat in k for pat in keep_fp32) else compute_dtype
            remainder[k] = v.to(dtype=target_dt)
        else:
            remainder[k] = v

    res = model.load_state_dict(remainder, strict=False, assign=True)
    # Filter out keys owned by swapped modules from missing_keys if assign didn't touch them
    missing = [
        k
        for k in res.missing_keys
        if k not in handled_missing and not k.endswith((".weight_i8", ".weight_fp8", ".weight_scale", "._dtype_anchor"))
    ]
    return missing, list(res.unexpected_keys)


class StrataBlockOffloadManager:
    """
    Strata 3-Tier Async Block Streamer for real PyTorch Transformer blocks (`nn.ModuleList`).

    - Tier 1 (VRAM Counter): Blocks `[0 .. resident_vram_blocks - 1]` stay permanently on `device`.
    - Tier 2 (Host RAM Pantry): Blocks `[resident_vram_blocks .. N - 1]` stay backed by read-only
      pinned CPU tensors (`pin_memory=True` on CUDA) occupying 0 VRAM at rest.
      Before each Pantry block executes:
        * Wait for its async H2D prefetch event (or load synchronously if not prefetched).
        * Enqueue async H2D prefetch for `block_idx + 1` on a dedicated CUDA transfer stream.
      After each Pantry block finishes:
        * Restore the module's tensor references directly to the read-only pinned CPU backing
          buffers in O(1) microseconds (zero D2H copy back over PCIe!).
    """

    def __init__(
        self,
        blocks: nn.ModuleList,
        device: torch.device,
        resident_vram_blocks: int,
        pin_memory: bool = True,
        on_block_step: Optional[Callable[[int, Optional[int], float], None]] = None,
    ):
        self.blocks = blocks
        self.total_blocks = len(blocks)
        self.device = torch.device(device)
        self.resident_vram_blocks = max(0, min(int(resident_vram_blocks), self.total_blocks))
        self.on_block_step = on_block_step

        self._use_cuda_stream = self.device.type == "cuda" and torch.cuda.is_available()
        self._prefetch_stream: Optional[Any] = (
            torch.cuda.Stream(device=self.device) if self._use_cuda_stream else None
        )
        self._pinned_cpu_backing: Dict[int, Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = {}
        self._staged_gpu_tensors: Dict[int, Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Any]] = {}
        self._handles: List[Any] = []
        self._pin_cpu = bool(pin_memory and self._use_cuda_stream)

    def prepare(self) -> None:
        """Place Counter blocks on GPU and pin Pantry blocks in host CPU RAM."""
        for idx, block in enumerate(self.blocks):
            if idx < self.resident_vram_blocks:
                block.to(self.device)
                continue

            # Ensure Pantry blocks sit on CPU first (and optionally pin floating/int8 tensors)
            block.to("cpu")
            p_back: Dict[str, torch.Tensor] = {}
            b_back: Dict[str, torch.Tensor] = {}
            for name, param in block.named_parameters(recurse=True):
                t = param.data.detach().contiguous()
                if self._pin_cpu and not t.is_pinned():
                    try:
                        t = t.pin_memory()
                    except RuntimeError:
                        pass
                param.data = t
                p_back[name] = t
            for name, buf in block.named_buffers(recurse=True):
                if buf is None:
                    continue
                t = buf.detach().contiguous()
                # Keep float8 CPU backing unpinned/safe or pinned when supported
                if self._pin_cpu and not t.is_pinned():
                    try:
                        t = t.pin_memory()
                    except RuntimeError:
                        pass
                self._assign_nested_buffer(block, name, t)
                b_back[name] = t

            self._pinned_cpu_backing[idx] = (p_back, b_back)

    @staticmethod
    def _assign_nested_parameter(root: nn.Module, dotted: str, tensor: torch.Tensor) -> None:
        parent, attr = _get_parent_and_attr(root, dotted)
        param = getattr(parent, attr)
        param.data = tensor

    @staticmethod
    def _assign_nested_buffer(root: nn.Module, dotted: str, tensor: torch.Tensor) -> None:
        parent, attr = _get_parent_and_attr(root, dotted)
        parent._buffers[attr] = tensor

    def _start_prefetch(self, idx: int) -> None:
        if idx < self.resident_vram_blocks or idx >= self.total_blocks:
            return
        if idx in self._staged_gpu_tensors:
            return
        p_back, b_back = self._pinned_cpu_backing[idx]
        p_gpu: Dict[str, torch.Tensor] = {}
        b_gpu: Dict[str, torch.Tensor] = {}

        if self._prefetch_stream is not None:
            evt = torch.cuda.Event()
            with torch.cuda.stream(self._prefetch_stream):
                for name, t in p_back.items():
                    p_gpu[name] = t.to(self.device, non_blocking=True)
                for name, t in b_back.items():
                    if self.device.type == "mps" and t.dtype == torch.float8_e4m3fn:
                        b_gpu[name] = t
                    else:
                        b_gpu[name] = t.to(self.device, non_blocking=True)
                evt.record(self._prefetch_stream)
            self._staged_gpu_tensors[idx] = (p_gpu, b_gpu, evt)
        else:
            for name, t in p_back.items():
                p_gpu[name] = t.to(self.device)
            for name, t in b_back.items():
                if self.device.type == "mps" and t.dtype == torch.float8_e4m3fn:
                    b_gpu[name] = t
                else:
                    b_gpu[name] = t.to(self.device)
            self._staged_gpu_tensors[idx] = (p_gpu, b_gpu, None)

    def _activate_block(self, idx: int) -> float:
        t0 = time.perf_counter()
        block = self.blocks[idx]
        if idx < self.resident_vram_blocks:
            return 0.0
        if idx not in self._staged_gpu_tensors:
            self._start_prefetch(idx)
        p_gpu, b_gpu, evt = self._staged_gpu_tensors.pop(idx)
        if evt is not None:
            torch.cuda.current_stream(self.device).wait_event(evt)
        for name, t in p_gpu.items():
            self._assign_nested_parameter(block, name, t)
        for name, t in b_gpu.items():
            self._assign_nested_buffer(block, name, t)
        return (time.perf_counter() - t0) * 1000.0

    def _evict_block(self, idx: int) -> None:
        if idx < self.resident_vram_blocks:
            return
        block = self.blocks[idx]
        p_back, b_back = self._pinned_cpu_backing[idx]
        for name, t in p_back.items():
            self._assign_nested_parameter(block, name, t)
        for name, t in b_back.items():
            self._assign_nested_buffer(block, name, t)

    def attach_hooks(self) -> None:
        self.detach_hooks()
        if self.resident_vram_blocks >= self.total_blocks:
            return

        for idx, block in enumerate(self.blocks):
            def _make_pre(block_idx: int):
                def _pre_hook(_mod, _inputs):
                    dma_ms = self._activate_block(block_idx)
                    next_idx = block_idx + 1
                    prefetch_idx = next_idx if next_idx < self.total_blocks else None
                    if prefetch_idx is not None and prefetch_idx >= self.resident_vram_blocks:
                        self._start_prefetch(prefetch_idx)
                    if self.on_block_step is not None:
                        self.on_block_step(block_idx, prefetch_idx, dma_ms)
                return _pre_hook

            def _make_post(block_idx: int):
                def _post_hook(_mod, _inputs, _outputs):
                    self._evict_block(block_idx)
                return _post_hook

            self._handles.append(block.register_forward_pre_hook(_make_pre(idx)))
            self._handles.append(block.register_forward_hook(_make_post(idx)))

    def detach_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._staged_gpu_tensors.clear()

    def release_all_to_cpu(self) -> None:
        """Evict all Counter & Pantry blocks from GPU cleanly (preserving pinned CPU handles)."""
        self.detach_hooks()
        for idx in range(self.resident_vram_blocks):
            self.blocks[idx].to("cpu")
        for idx in range(self.resident_vram_blocks, self.total_blocks):
            self._evict_block(idx)
        if self._use_cuda_stream:
            torch.cuda.empty_cache()
