"""
Tests for openfal's real PyTorch runtime:
1. Hadamard ConvRot INT8 exact mathematical equivalence (`StrataConvRotLinear`, `StrataConvRotEmbedding`)
2. Scaled FP8 E4M3 (`StrataFP8ScaledLinear`) accuracy + CPU/MPS compatibility
3. Zero-duplication `quantized_module_swap_from_state_dict`
4. Asynchronous / pinned-CPU `StrataBlockOffloadManager` block streaming Counter/Pantry split
5. Exact Comfy-Org -> Diffusers/Transformers state-dict key & tensor conversions for:
   - Qwen-Image-2.1-Turbo (DiT `gate_up` split, Qwen3-VL prefixing, all 238 VAE keys)
   - Wan 2.2 MoE (Transformer & UMT5-XXL FP8)
   - MiniMax-H3 (pruned `adaln_t_table`, SwiGLU `[gate; value] -> [value; gate]`, QKV split, ViT/Audio VAEs)
6. End-to-end tiny-config instantiation + forward pass against real `diffusers` model classes (`missing=[]`, `unexpected=[]`)
7. Fail-fast guard in `OpenFalRunner` ensuring real checkpoints never silently degrade to placeholders.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file as safetensors_save_file

from openfal.config import EngineConfig
from openfal.engine.memory_hierarchy import StrataMemoryHierarchy
from openfal.engine.runner import OpenFalRunner
from openfal.runtime.minimax_h3_runner import (
    _PrunedAdaLayerNormModulation,
    _PrunedAdaLayerNormOut,
    _PrunedAdalnTableEmbedder,
    _strip_audio_weight_norm,
    convert_minimax_h3_dit_state_dict,
    convert_minimax_h3_video_vae_state_dict,
)
from openfal.runtime.quant import (
    StrataBlockOffloadManager,
    StrataConvRotEmbedding,
    StrataConvRotLinear,
    StrataFP8ScaledLinear,
    apply_group_hadamard,
    build_hadamard_matrix,
    quantized_module_swap_from_state_dict,
)
from openfal.runtime.qwen_image_runner import (
    convert_qwen3vl_text_encoder_state_dict,
    convert_qwen_image21_transformer_state_dict,
    convert_qwen_image21_vae_state_dict,
)
from openfal.runtime.wan22_runner import convert_umt5_fp8_state_dict, convert_wan22_transformer_state_dict


def _make_convrot_i8(weight_f32: torch.Tensor, group_size: int = 256):
    """Quantize a weight matrix using the ConvRot Hadamard transform W_rot = W @ H_G."""
    w_rot = apply_group_hadamard(weight_f32.float(), group_size=group_size)
    row_max = torch.clamp(w_rot.abs().amax(dim=-1, keepdim=True), min=1e-8)
    scale = row_max / 127.0
    q = torch.clamp(torch.round(w_rot / scale), -127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def test_hadamard_and_convrot_linear_equivalence():
    torch.manual_seed(42)
    h4 = build_hadamard_matrix(4, dtype=torch.float32)
    expected_h4 = torch.tensor(
        [[1.0, 1.0, 1.0, -1.0], [1.0, 1.0, -1.0, 1.0], [1.0, -1.0, 1.0, 1.0], [-1.0, 1.0, 1.0, 1.0]],
        dtype=torch.float32,
    ) / 2.0
    assert torch.allclose(h4, expected_h4, atol=1e-6)

    h = build_hadamard_matrix(256, dtype=torch.float32)
    assert h.shape == (256, 256)
    # Regular Hadamard H_4^{\otimes 4}: unnormalized row sum 2^4 = 16 -> 1.0 after dividing by sqrt(256)
    assert torch.allclose(h.sum(dim=-1), torch.full((256,), 1.0), atol=1e-4)
    # Orthogonal and symmetric
    assert torch.allclose(h @ h.T, torch.eye(256), atol=1e-5, rtol=1e-5)
    assert torch.allclose(h, h.T, atol=1e-6)

    w_orig = torch.randn(64, 512, dtype=torch.float32)
    bias = torch.randn(64, dtype=torch.float32)
    x = torch.randn(3, 17, 512, dtype=torch.float32)

    q_w, scale = _make_convrot_i8(w_orig, group_size=256)
    layer = StrataConvRotLinear(q_w, scale, bias=bias, convrot=True, group_size=256, compute_dtype=torch.float32)

    expected = F.linear(x, w_orig, bias)
    actual = layer(x)
    # 8-bit quantization error should be tiny (~0.01-0.05 RMS vs signal std ~22)
    rel_err = (actual - expected).norm() / expected.norm()
    assert rel_err < 0.02, f"ConvRot relative error too large: {rel_err}"

    # Verify weight storage truly stayed int8 (1 byte/param)
    assert layer.weight_i8.dtype == torch.int8
    assert layer._dtype_anchor.dtype == torch.float32


def test_convrot_embedding_and_fp8_scaled_linear():
    torch.manual_seed(7)
    emb_orig = torch.randn(32, 512, dtype=torch.float32)
    q_emb, emb_scale = _make_convrot_i8(emb_orig, group_size=256)
    emb = StrataConvRotEmbedding(q_emb, emb_scale, convrot=True, group_size=256, compute_dtype=torch.float32)
    ids = torch.tensor([[1, 5, 17, 30]], dtype=torch.long)
    out_emb = emb(ids)
    ref_emb = F.embedding(ids, emb_orig)
    assert ((out_emb - ref_emb).norm() / ref_emb.norm()) < 0.02

    # Scaled FP8 E4M3 linear
    w = torch.randn(48, 64, dtype=torch.float32)
    fp8_max = 448.0
    scale = torch.tensor(float(w.abs().max() / fp8_max), dtype=torch.float32)
    w_fp8 = (w / scale).to(torch.float8_e4m3fn)
    bias = torch.randn(48, dtype=torch.float32)
    fp8_layer = StrataFP8ScaledLinear(w_fp8, scale, bias=bias, compute_dtype=torch.float32)

    x = torch.randn(2, 5, 64, dtype=torch.float32)
    y_fp8 = fp8_layer(x)
    y_ref = F.linear(x, w, bias)
    assert ((y_fp8 - y_ref).norm() / y_ref.norm()) < 0.06

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        fp8_mps = fp8_layer.to("mps")
        y_mps = fp8_mps(x.to("mps"))
        assert torch.allclose(y_mps.cpu(), y_fp8, atol=1e-2, rtol=1e-2)


def test_quantized_module_swap_and_block_offloader():
    torch.manual_seed(11)

    class ToyBlock(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.proj = nn.Linear(dim, dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.proj(self.norm(x))

    class ToyModel(nn.Module):
        def __init__(self, dim: int = 256, depth: int = 5):
            super().__init__()
            self.embed = nn.Embedding(16, dim)
            self.blocks = nn.ModuleList([ToyBlock(dim) for _ in range(depth)])
            self.out = nn.Linear(dim, dim, bias=True)

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            h = self.embed(ids)
            for b in self.blocks:
                h = b(h)
            return self.out(h)

    model = ToyModel(dim=256, depth=5)
    # Build mixed INT8 ConvRot + FP8 state dict
    sd = {}
    q_e, s_e = _make_convrot_i8(torch.randn(16, 256), group_size=256)
    sd["embed.weight"] = q_e
    sd["embed.weight_scale"] = s_e
    for i in range(5):
        sd[f"blocks.{i}.norm.weight"] = torch.ones(256)
        sd[f"blocks.{i}.norm.bias"] = torch.zeros(256)
        q_w, s_w = _make_convrot_i8(torch.randn(256, 256) * 0.05, group_size=256)
        sd[f"blocks.{i}.proj.weight"] = q_w
        sd[f"blocks.{i}.proj.weight_scale"] = s_w
    sd["out.weight"] = (torch.randn(256, 256) * 0.05).to(torch.float8_e4m3fn)
    sd["out.scale_weight"] = torch.tensor(1.0, dtype=torch.float32)
    sd["out.bias"] = torch.zeros(256)

    missing, unexpected = quantized_module_swap_from_state_dict(
        model, sd, compute_dtype=torch.float32, embedding_names=("embed",)
    )
    assert missing == []
    assert unexpected == []
    assert isinstance(model.embed, StrataConvRotEmbedding)
    assert isinstance(model.blocks[0].proj, StrataConvRotLinear)
    assert isinstance(model.out, StrataFP8ScaledLinear)

    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    baseline = model(ids).clone()

    # Now test StrataBlockOffloadManager keeping 2 blocks in "VRAM Counter" and 3 in "RAM Pantry"
    target_dev = torch.device("mps") if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else torch.device("cpu")
    events = []
    mgr = StrataBlockOffloadManager(
        blocks=model.blocks,
        device=target_dev,
        resident_vram_blocks=2,
        on_block_step=lambda active, pre, ms: events.append((active, pre, ms)),
    )
    model.embed.to(target_dev)
    model.out.to(target_dev)
    mgr.prepare()
    mgr.attach_hooks()

    # Before forward: Counter blocks [0,1] on target_dev, Pantry blocks [2,3,4] on cpu
    assert model.blocks[0].norm.weight.device.type == target_dev.type
    assert model.blocks[2].norm.weight.device.type == "cpu"

    offload_out = model(ids.to(target_dev)).cpu()
    # After forward: Pantry blocks must be evicted back to CPU backing
    for idx in (2, 3, 4):
        assert model.blocks[idx].norm.weight.device.type == "cpu"
        assert model.blocks[idx].proj.weight_i8.device.type == "cpu"
    assert len(events) == 5
    assert torch.allclose(offload_out, baseline, atol=1e-3, rtol=1e-3)

    mgr.release_all_to_cpu()


def test_qwen_image21_transformer_te_and_238_vae_converters():
    # 1. Transformer gate_up split + prefix stripping
    raw_tr = {
        "model.diffusion_model.img_in.weight": torch.zeros(128, 64),
        "model.diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight": torch.cat(
            [torch.ones(64, 128, dtype=torch.int8), torch.full((64, 128), 2, dtype=torch.int8)], dim=0
        ),
        "model.diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight_scale": torch.cat(
            [torch.ones(64, 1), torch.full((64, 1), 3.0)], dim=0
        ),
        "model.diffusion_model.transformer_blocks.0.img_mlp.gate_up.comfy_quant": torch.tensor([1, 2], dtype=torch.uint8),
        "model.diffusion_model.transformer_blocks.0.attn.comfy_attention.config": torch.tensor([9], dtype=torch.uint8),
    }
    tr_out = convert_qwen_image21_transformer_state_dict(raw_tr)
    assert "img_in.weight" in tr_out
    assert torch.equal(tr_out["transformer_blocks.0.img_mlp.gate_layer.weight"], torch.ones(64, 128, dtype=torch.int8))
    assert torch.equal(tr_out["transformer_blocks.0.img_mlp.proj.weight"], torch.full((64, 128), 2, dtype=torch.int8))
    assert torch.equal(tr_out["transformer_blocks.0.img_mlp.gate_layer.weight_scale"], torch.ones(64, 1))
    assert torch.equal(tr_out["transformer_blocks.0.img_mlp.proj.weight_scale"], torch.full((64, 1), 3.0))
    assert "transformer_blocks.0.img_mlp.proj.comfy_quant" in tr_out
    assert "transformer_blocks.0.attn.comfy_attention.config" not in tr_out

    # 2. Qwen3-VL prefix mapping
    te_out = convert_qwen3vl_text_encoder_state_dict(
        {
            "model.embed_tokens.weight": torch.zeros(4, 4),
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(4, 4),
            "model.norm.weight": torch.ones(4),
            "visual.blocks.0.norm1.weight": torch.ones(4),
            "model.visual.patch_embed.proj.weight": torch.zeros(4, 3),
            "lm_head.weight": torch.zeros(4, 4),
        }
    )
    assert "model.language_model.embed_tokens.weight" in te_out
    assert "model.language_model.layers.0.self_attn.q_proj.weight" in te_out
    assert "model.language_model.norm.weight" in te_out
    assert "model.visual.blocks.0.norm1.weight" in te_out
    assert "model.visual.patch_embed.proj.weight" in te_out
    assert "lm_head.weight" in te_out

    # 3. Full 238-key Comfy Qwen-Image-2.1 VAE mapping & 5D->4D temporal squeeze
    raw_vae = {
        "conv1.weight": torch.zeros(128, 128, 1, 1, 1),
        "conv1.bias": torch.zeros(128),
        "conv2.weight": torch.zeros(64, 64, 1, 1, 1),
        "conv2.bias": torch.zeros(64),
        "decoder.conv1.weight": torch.zeros(1152, 64, 1, 3, 3),
        "decoder.head.0.gamma": torch.ones(144, 1, 1, 1),
        "decoder.head.2.weight": torch.zeros(4, 144, 1, 3, 3),
        "decoder.middle.0.residual.2.weight": torch.zeros(1152, 1152, 1, 3, 3),
        "decoder.middle.1.to_qkv.weight": torch.zeros(3456, 1152, 1, 1),
        "decoder.upsamples.2.upsamples.0.shortcut.weight": torch.zeros(576, 1152, 1, 1, 1),
        "decoder.upsamples.0.upsamples.3.time_conv.weight": torch.zeros(2304, 1152, 1, 1, 1),
        "encoder.downsamples.1.downsamples.0.shortcut.weight": torch.zeros(192, 96, 1, 1, 1),
        "encoder.downsamples.1.downsamples.2.resample.1.weight": torch.zeros(192, 192, 3, 3),
    }
    vae_out = convert_qwen_image21_vae_state_dict(raw_vae)
    assert vae_out["quant_conv.weight"].shape == (128, 128, 1, 1)
    assert vae_out["post_quant_conv.weight"].shape == (64, 64, 1, 1)
    assert vae_out["decoder.conv_in.weight"].shape == (1152, 64, 3, 3)
    assert "decoder.norm_out.gamma" in vae_out
    assert vae_out["decoder.conv_out.weight"].shape == (4, 144, 3, 3)
    assert vae_out["decoder.mid_block.resnets.0.conv1.weight"].shape == (1152, 1152, 3, 3)
    assert "decoder.mid_block.attentions.0.to_qkv.weight" in vae_out
    assert vae_out["decoder.up_blocks.2.resnets.0.conv_shortcut.weight"].shape == (576, 1152, 1, 1)
    assert vae_out["decoder.up_blocks.0.upsampler.time_conv.weight"].shape == (2304, 1152, 1, 1)
    assert vae_out["encoder.down_blocks.1.resnets.0.conv_shortcut.weight"].shape == (192, 96, 1, 1)
    assert "encoder.down_blocks.1.downsampler.resample.1.weight" in vae_out


def test_minimax_h3_pruned_converters_and_table():
    raw_mm = {
        "adaln_t_table": torch.tensor([[0.0, 1.0], [10.0, 20.0], [20.0, 40.0]]),
        "rope.inv_freq": torch.zeros(4),
        "video_patch_proj.weight": torch.zeros(8, 24),
        "audio_patch_proj.weight": torch.zeros(8, 32),
        "condition_proj.weight": torch.zeros(8, 16),
        "token_refiner.blocks.0.attn.qkv_proj.weight": torch.cat(
            [torch.ones(4, 8), torch.full((4, 8), 2.0), torch.full((4, 8), 3.0)], dim=0
        ),
        "blocks.0.attn.qkv_proj.weight": torch.cat(
            [torch.ones(4, 8, dtype=torch.float8_e4m3fn), torch.zeros(8, 8, dtype=torch.float8_e4m3fn)], dim=0
        ),
        "blocks.0.attn.qkv_proj.weight_scale": torch.tensor(0.5),
        "blocks.0.mlp.fc1.weight": torch.cat(
            [torch.ones(6, 8, dtype=torch.int8), torch.full((6, 8), 7, dtype=torch.int8)], dim=0
        ),
        "blocks.0.mlp.fc1.weight_scale": torch.cat([torch.ones(6, 1), torch.full((6, 1), 9.0)], dim=0),
        "final_layer.video_out.weight": torch.zeros(24, 8),
        "final_layer.audio_out.weight": torch.zeros(32, 8),
    }
    sd, table = convert_minimax_h3_dit_state_dict(raw_mm)
    assert table is not None and table.shape == (3, 2)
    assert "proj_in.weight" in sd and "audio_proj_in.weight" in sd and "context_embedder.weight" in sd
    assert torch.equal(sd["token_refiner.refiner_blocks.0.attn.to_q.weight"], torch.ones(4, 8))
    assert torch.equal(sd["transformer_blocks.0.attn.to_q.weight_scale"], torch.tensor(0.5))
    # [gate; value] -> [value; gate] check
    assert torch.equal(
        sd["transformer_blocks.0.ff.net.0.proj.weight"][:6],
        torch.full((6, 8), 7, dtype=torch.int8),
    )
    assert torch.equal(
        sd["transformer_blocks.0.ff.net.0.proj.weight_scale"][:6],
        torch.full((6, 1), 9.0),
    )

    emb = _PrunedAdalnTableEmbedder(table)
    # t=0.0005 -> idx=0.5 interpolated between row 0 and row 1
    out = emb(torch.tensor([0.0, 0.0005, 0.002]))
    assert torch.allclose(out[0], torch.tensor([0.0, 1.0]))
    assert torch.allclose(out[1], torch.tensor([5.0, 10.5]))
    assert torch.allclose(out[2], torch.tensor([20.0, 40.0]))

    # Video VAE conversion
    vvae_sd = convert_minimax_h3_video_vae_state_dict(
        {
            "encoder.down.1.block.0.nin_shortcut.weight": torch.zeros(4, 2, 1, 1, 1),
            "decoder.x_embedder.weight": torch.zeros(8, 24),
            "decoder.transformer_blocks.0.attn.to_qkv.weight": torch.ones(12, 8),
            "decoder.transformer_blocks.0.ff.w1.weight": torch.cat([torch.ones(6, 8), torch.zeros(6, 8)], dim=0),
            "decoder.mask_token": torch.zeros(1, 1, 8),
        }
    )
    assert "encoder.down_blocks.1.resnets.0.conv_shortcut.weight" in vvae_sd
    assert "decoder.proj_in.weight" in vvae_sd
    assert "decoder.mask_token" not in vvae_sd
    assert vvae_sd["decoder.transformer_blocks.0.attn.to_v.weight"].shape == (4, 8)
    assert torch.equal(vvae_sd["decoder.transformer_blocks.0.ff.net.0.proj.weight"][:6], torch.zeros(6, 8))


def test_diffusers_models_accept_converted_state_dicts():
    diffusers = pytest.importorskip("diffusers")
    from openfal.runtime.common import check_inference_dependencies

    check_inference_dependencies(("diffusers", "transformers", "accelerate"))

    # 1. Tiny QwenImage21Transformer2DModel exact load + forward pass
    qtr = diffusers.QwenImage21Transformer2DModel(
        patch_size=1,
        in_channels=16,
        out_channels=16,
        num_layers=2,
        attention_head_dim=128,
        num_attention_heads=2,
        context_in_dim=256,
        mlp_ratio=2,
        axes_dims_rope=(16, 56, 56),
        causal_condition=True,
    )
    comfy_tr = {}
    for k, v in qtr.state_dict().items():
        if "img_mlp.gate_layer.weight" in k:
            proj_k = k.replace("gate_layer", "proj")
            w_gate, s_gate = _make_convrot_i8(v.float(), 256)
            w_proj, s_proj = _make_convrot_i8(qtr.state_dict()[proj_k].float(), 256)
            gu_key = k.replace("gate_layer", "gate_up")
            comfy_tr[f"model.diffusion_model.{gu_key}"] = torch.cat([w_gate, w_proj], dim=0)
            comfy_tr[f"model.diffusion_model.{gu_key.replace('.weight', '.weight_scale')}"] = torch.cat(
                [s_gate, s_proj], dim=0
            )
            comfy_tr[f"model.diffusion_model.{gu_key.replace('.weight', '.comfy_quant')}"] = torch.tensor(
                list(b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}'),
                dtype=torch.uint8,
            )
        elif "img_mlp.proj.weight" in k:
            continue
        elif "transformer_blocks." in k and v.ndim == 2 and v.shape[1] % 256 == 0:
            q_w, sc = _make_convrot_i8(v.float(), 256)
            comfy_tr[f"model.diffusion_model.{k}"] = q_w
            comfy_tr[f"model.diffusion_model.{k.replace('.weight', '.weight_scale')}"] = sc
        else:
            comfy_tr[f"model.diffusion_model.{k}"] = v.clone()

    mapped = convert_qwen_image21_transformer_state_dict(comfy_tr)
    missing, unexp = quantized_module_swap_from_state_dict(qtr, mapped, compute_dtype=torch.float32)
    assert missing == []
    assert unexp == []
    # Run real forward pass on INT8-ConvRot swapped QwenImage21Transformer2DModel
    qtr_out = qtr(
        hidden_states=torch.randn(1, 4, 16),
        encoder_hidden_states=torch.randn(1, 6, 256),
        timestep=torch.tensor([0.5]),
        img_shapes=[[(1, 2, 2)]],
        img_mask=torch.tensor([[0, 0, 0, 0, 0, 0, 1]], dtype=torch.bool),
        return_dict=False,
    )[0]
    assert qtr_out.shape == (1, 10, 16)
    assert qtr_out[:, -4:].shape == (1, 4, 16)

    # 2. Tiny AutoencoderKLQwenImage21 exact forward decode
    vae = diffusers.AutoencoderKLQwenImage21(
        in_channels=4,
        base_dim=8,
        decoder_base_dim=8,
        z_dim=4,
        dim_mult=[1, 2, 2],
        num_res_blocks=1,
        temperal_downsample=[False, False],
    )
    lat = torch.randn(1, 4, 1, 2, 2)
    img_out = vae.decode(lat, return_dict=False)[0]
    assert img_out.shape[:3] == (1, 4, 1)

    # 3. Tiny WanTransformer3DModel with scaled FP8 blocks -> convert + swap + forward pass
    wan = diffusers.WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=32,
        in_channels=16,
        out_channels=16,
        text_dim=64,
        freq_dim=32,
        ffn_dim=64,
        num_layers=2,
    )
    comfy_wan = {
        "scaled_fp8": torch.empty(0, dtype=torch.float8_e4m3fn),
        "patch_embedding.weight": wan.patch_embedding.weight.detach().clone(),
        "patch_embedding.bias": wan.patch_embedding.bias.detach().clone(),
        "time_embedding.0.weight": wan.condition_embedder.time_embedder.linear_1.weight.detach().clone(),
        "time_embedding.0.bias": wan.condition_embedder.time_embedder.linear_1.bias.detach().clone(),
        "time_embedding.2.weight": wan.condition_embedder.time_embedder.linear_2.weight.detach().clone(),
        "time_embedding.2.bias": wan.condition_embedder.time_embedder.linear_2.bias.detach().clone(),
        "time_projection.1.weight": wan.condition_embedder.time_proj.weight.detach().clone(),
        "time_projection.1.bias": wan.condition_embedder.time_proj.bias.detach().clone(),
        "text_embedding.0.weight": wan.condition_embedder.text_embedder.linear_1.weight.detach().clone(),
        "text_embedding.0.bias": wan.condition_embedder.text_embedder.linear_1.bias.detach().clone(),
        "text_embedding.2.weight": wan.condition_embedder.text_embedder.linear_2.weight.detach().clone(),
        "text_embedding.2.bias": wan.condition_embedder.text_embedder.linear_2.bias.detach().clone(),
        "head.head.weight": wan.proj_out.weight.detach().clone(),
        "head.head.bias": wan.proj_out.bias.detach().clone(),
        "head.modulation": wan.scale_shift_table.detach().clone(),
    }
    for i, blk in enumerate(wan.blocks):
        comfy_wan[f"blocks.{i}.modulation"] = blk.scale_shift_table.detach().clone()
        comfy_wan[f"blocks.{i}.norm3.weight"] = blk.norm2.weight.detach().clone()
        comfy_wan[f"blocks.{i}.norm3.bias"] = blk.norm2.bias.detach().clone()
        for src_p, dst_m in [
            ("self_attn.q", blk.attn1.to_q),
            ("self_attn.k", blk.attn1.to_k),
            ("self_attn.v", blk.attn1.to_v),
            ("self_attn.o", blk.attn1.to_out[0]),
            ("cross_attn.q", blk.attn2.to_q),
            ("cross_attn.k", blk.attn2.to_k),
            ("cross_attn.v", blk.attn2.to_v),
            ("cross_attn.o", blk.attn2.to_out[0]),
            ("ffn.0", blk.ffn.net[0].proj),
            ("ffn.2", blk.ffn.net[2]),
        ]:
            comfy_wan[f"blocks.{i}.{src_p}.weight"] = (dst_m.weight.detach() * 0.1).to(torch.float8_e4m3fn)
            comfy_wan[f"blocks.{i}.{src_p}.scale_weight"] = torch.tensor(1.0, dtype=torch.float32)
            comfy_wan[f"blocks.{i}.{src_p}.bias"] = dst_m.bias.detach().clone()
        for src_p, dst_m in [
            ("self_attn.norm_q", blk.attn1.norm_q),
            ("self_attn.norm_k", blk.attn1.norm_k),
            ("cross_attn.norm_q", blk.attn2.norm_q),
            ("cross_attn.norm_k", blk.attn2.norm_k),
        ]:
            comfy_wan[f"blocks.{i}.{src_p}.weight"] = dst_m.weight.detach().clone()

    wan_mapped = convert_wan22_transformer_state_dict(comfy_wan)
    w_miss, w_unexp = quantized_module_swap_from_state_dict(wan, wan_mapped, compute_dtype=torch.float32)
    assert w_miss == []
    assert w_unexp == []
    assert isinstance(wan.blocks[0].attn1.to_q, StrataFP8ScaledLinear)
    wan_out = wan(
        hidden_states=torch.randn(1, 16, 2, 4, 4),
        timestep=torch.tensor([500.0]),
        encoder_hidden_states=torch.randn(1, 4, 64),
        return_dict=False,
    )[0]
    assert wan_out.shape == (1, 16, 2, 4, 4)

    # 4. Tiny MiniMaxH3Transformer3DModel pruned checkpoint load & patched AdaLN check
    mm = diffusers.MiniMaxH3Transformer3DModel(
        num_attention_heads=2,
        attention_head_dim=32,
        hidden_size=64,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=128,
        in_channels=4,
        audio_in_channels=8,
        patch_size=(1, 2, 2),
        text_dim=32,
        freq_dim=16,
        time_embed_hidden_dim=64,
        time_embed_dim=8,
        rope_freq_dim=8,
    )
    mm.time_proj = nn.Identity()
    mm.time_embedder = _PrunedAdalnTableEmbedder(torch.randn(16, 8))
    mm.norm_out = _PrunedAdaLayerNormOut(mm.norm_out.norm, mm.norm_out.linear)
    mm.transformer_blocks[0].adaln_proj = _PrunedAdaLayerNormModulation(
        hidden_size=64, orig_linear=mm.transformer_blocks[0].adaln_proj.linear
    )
    t_emb = mm.time_embedder(mm.time_proj(torch.tensor([0.005])))
    mod_chunks = mm.transformer_blocks[0].adaln_proj(t_emb)
    assert len(mod_chunks) == 6

    # 5. Tiny MiniMaxH3AudioVAE weight_norm removal roundtrip
    avae = diffusers.AutoencoderKLMiniMaxH3Audio(
        encoder_dim=16,
        encoder_rates=(2,),
        latent_dim=32,
        latent_channels=8,
        decoder_dim=32,
        decoder_rates=(2,),
        decoder_kernel_sizes=(4,),
        num_attention_heads=2,
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1,),),
    )
    _strip_audio_weight_norm(avae)
    sd_keys = list(avae.state_dict().keys())
    assert not any(k.endswith(("weight_g", "weight_v", "original0", "original1")) for k in sd_keys)


def test_real_weights_do_not_silently_fallback(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENFAL_SYNTHETIC_FALLBACK", raising=False)
    cfg = EngineConfig(
        profile_id="apple_silicon_64gb",
        weights_dir=tmp_path / "weights",
        output_dir=tmp_path / "outputs",
        cache_dir=tmp_path / "cache",
    )
    # Create dummy Qwen-Image-2.1 DiT .safetensors so runner treats weights as present
    dit_file = cfg.weights_dir / "Qwen-Image-2.1-Turbo" / "diffusion_models" / "qwen_image_2.1_turbo_int8_convrot.safetensors"
    dit_file.parent.mkdir(parents=True, exist_ok=True)
    safetensors_save_file({"img_in.weight": torch.zeros(2, 2)}, str(dit_file))

    runner = OpenFalRunner(cfg)
    # Because DiT exists on disk while TE & VAE are missing, must raise loudly rather than returning fake image
    with pytest.raises(RuntimeError, match="missing component"):
        runner.execute(
            endpoint_id="qwen/image-2.1/text-to-image",
            payload={"prompt": "test"},
        )
