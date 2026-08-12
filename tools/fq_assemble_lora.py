#!/usr/bin/env python3
"""fq_assemble_lora — Encode MSRT residual cartridges as LoRA-compatible adapters.

Given a BF16 (or EXL3) source checkpoint and a cartridge recipe, this tool:
  1. Quantizes the base tier (K2 or K3) into standard EXL3 trellis format
  2. Computes residuals, rescales, and quantizes each residual stage
  3. Emits two outputs:
     - A base checkpoint (standard EXL3 safetensors, loads normally in vLLM)
     - One or more cartridge adapters (safetensors with per-stage trellis
       tensors, loadable as LoRA adapters via vLLM's add_lora API)

The cartridge adapter is NOT a low-rank LoRA — it contains full-rank trellis-
quantized residual weights. The vLLM EXL3 LoRA wrapper (Exl3LoRAMoMethod)
applies them by running additional exl3_gemm passes and summing with rescaling.

MSRT (Multi-Stage Rescaled Trellis) is described in:
  research/fungible-quant/poc/V50-LOW-BITRATE-MSRT.md
  research/fungible-quant/MSRT-CARTRIDGE-FEASIBILITY-AND-PLAN.md

Cartridge recipe format (fq-cartridge/1):

  {
    "schema": "fq-cartridge/1",
    "base_k": 2,
    "stages": [
      {"k": 1, "label": "res1", "experts": "all"},
      {"k": 2, "label": "res2", "experts": [0, 1, 10, 11, ...]}
    ],
    "moe_layers": [3, 4, 5, ...]
  }

Usage:
  python tools/fq_assemble_lora.py encode \\
    --source ./bf16-checkpoint \\
    --recipe recipes/fruit-k2-k3k4-cart.json \\
    --out ./output \\
    --encoder-source /opt/fruit-pip/exllamav3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any

import torch

# ── Constants ──────────────────────────────────────────────────────────────

TOOL_VERSION = "fq_assemble_lora/1"
CARTRIDGE_SCHEMA = "fq-cartridge/1"
ADAPTER_CONFIG_SCHEMA = "fq-cartridge-adapter/1"
HADAMARD_BLOCK = 128

# Expert tensor name pattern — matches both BF16 (.weight) and EXL3 (.rank0.trellis)
EXPERT_RE_PATTERN = (
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(?:rank\d+\.)?(?:weight|trellis)$"
)

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


# ── EXL3 Encoder Bootstrap ────────────────────────────────────────────────

def bootstrap_encoder(encoder_source: str) -> tuple[Any, ...]:
    """Load the EXL3 encoder without importing all of exllamav3.

    Returns (ext, get_hadamard_dt, tensor_core_perm, tensor_core_perm_i,
             quantize_tiles, codebook_scale).
    """
    import importlib.util
    import types

    pkg_root = Path(encoder_source)
    pkg = types.ModuleType("exllamav3")
    pkg.__path__ = [str(pkg_root)]
    sys.modules["exllamav3"] = pkg

    for sub in ["util", "modules", "modules.quant", "modules.quant.exl3_lib"]:
        full = f"exllamav3.{sub}"
        m = types.ModuleType(full)
        m.__path__ = [str(pkg_root / sub.replace(".", "/"))]
        sys.modules[full] = m

    # Stub progress/memory to avoid flash_attn dependency
    _stub = types.ModuleType("exllamav3.util.progress")
    class _DPB:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def update(self, *a): pass
        def new_task(self, *a, **kw): pass
    _stub.ProgressBar = _DPB
    sys.modules["exllamav3.util.progress"] = _stub

    _stub = types.ModuleType("exllamav3.util.memory")
    _stub.free_mem = lambda: None
    _stub.list_gpu_tensors = lambda: []
    sys.modules["exllamav3.util.memory"] = _stub

    _stub = types.ModuleType("exllamav3.util")
    _stub.__path__ = [str(pkg_root / "util")]
    _stub.cuda_sync_active = lambda *a, **kw: torch.cuda.synchronize()
    sys.modules["exllamav3.util"] = _stub

    _stub = types.ModuleType("exllamav3.util.tensor")
    _stub.save_tensor_image = lambda *a, **kw: None
    sys.modules["exllamav3.util.tensor"] = _stub

    # Load the extension
    spec = importlib.util.spec_from_file_location(
        "exllamav3.ext", str(pkg_root / "ext.py"))
    ext_mod = importlib.util.module_from_spec(spec)
    sys.modules["exllamav3.ext"] = ext_mod
    spec.loader.exec_module(ext_mod)

    # Load Hadamard
    spec = importlib.util.spec_from_file_location(
        "exllamav3.util.hadamard", str(pkg_root / "util" / "hadamard.py"))
    had_mod = importlib.util.module_from_spec(spec)
    sys.modules["exllamav3.util.hadamard"] = had_mod
    spec.loader.exec_module(had_mod)

    # Load quantize module
    quant_path = pkg_root / "modules" / "quant" / "exl3_lib" / "quantize.py"
    spec = importlib.util.spec_from_file_location(
        "exllamav3.modules.quant.exl3_lib.quantize", str(quant_path))
    quant_mod = importlib.util.module_from_spec(spec)
    sys.modules["exllamav3.modules.quant.exl3_lib.quantize"] = quant_mod
    spec.loader.exec_module(quant_mod)

    return (ext_mod.exllamav3_ext, had_mod.get_hadamard_dt,
            quant_mod.tensor_core_perm, quant_mod.tensor_core_perm_i,
            quant_mod.quantize_tiles, quant_mod.codebook_scale)


# ── Quantization Primitives ────────────────────────────────────────────────

def block_rms(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """RMS along a dimension."""
    return x.square().mean(dim=dim, keepdim=keepdim).sqrt()


def regularize(
    w: torch.Tensor,
    device: torch.device,
    ghd: Any,
    cbs: float,
    had_k: int = HADAMARD_BLOCK,
    had_n: int = HADAMARD_BLOCK,
    seed: int = 0,
) -> torch.Tensor:
    """Apply Hadamard regularization (in-place transform, returns new tensor).

    This matches the EXL3 regularize() used in the PoC scripts (v35-v52).
    The Hadamard is orthogonal, so MSE in regularized space = MSE in original.
    """
    k, n = w.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    su = (torch.randn(k, generator=g).sign() + 1e-5).sign().float().to(device)
    sv = (torch.randn(n, generator=g).sign() + 1e-5).sign().float().to(device)

    out_scales = block_rms(w, dim=0, keepdim=True)
    mean = out_scales.mean().item()
    if mean > 1e-30:
        out_scales = out_scales / mean
    sv = (sv * out_scales + 1e-10).float()
    w = (w / sv).contiguous()

    had_n_mat = ghd(had_n, device, torch.float, 1.0 / math.sqrt(had_n))
    w = (w.view(k, n // had_n, had_n) @ had_n_mat).view(k, n).contiguous()

    in_scales = block_rms(w, dim=1, keepdim=True).clamp(min=1e-30)
    su = (su.unsqueeze(1) * in_scales / (-cbs) + 1e-10).float()
    w = (w / su).contiguous()

    had_k_mat = ghd(had_k, device, torch.float, 1.0 / math.sqrt(had_k))
    w = (had_k_mat @ w.view(k // had_k, had_k, n)).view(k, n).contiguous()
    return w


def quantize_trellis(
    data: torch.Tensor,
    K: int,
    device: torch.device,
    tcp: Any,
    tcpi: Any,
    qtf: Any,
) -> torch.Tensor:
    """Quantize a 2D tensor with EXL3 trellis at K bits.

    Returns the dequantized (reconstructed) tensor, NOT the packed indices.
    The trellis tiles are 16×16, processed row-by-row in blocks of 16.
    """
    k, n = data.shape
    tiles_n = n // 16
    weight_q = torch.zeros_like(data)
    qa = {"K": K, "mcg": True}
    perm = tcp(device)
    perm_i = tcpi(device)

    for bi in range(0, k, 16):
        rows = data[bi:bi + 16]
        tiles = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)
        tiles = tiles[:, perm].contiguous()
        quant_w, _ = qtf(tiles, qa)
        quant_w = quant_w[:, perm_i].reshape(tiles_n, 16, 16).permute(1, 0, 2).reshape(16, n)
        weight_q[bi:bi + 16] = quant_w

    return weight_q


def quantize_trellis_packed(
    data: torch.Tensor,
    K: int,
    device: torch.device,
    tcp: Any,
    tcpi: Any,
    qtf: Any,
    ext: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize and return BOTH reconstructed values and packed trellis indices.

    Returns (reconstructed_float, packed_trellis_int16) where packed_trellis
    has shape (k // 16, n // 16, K * 16) dtype int16 — the EXL3 storage format.

    The quantize_tiles function returns raw Viterbi path indices of shape
    (n_tiles, 256) — one int16 per weight element. ext.pack_trellis compresses
    these to (n_tiles, K * 16) packed indices, which is what the EXL3 checkpoint
    and vLLM loader expect (validated at exl3.py:2099-2102).
    """
    k, n = data.shape
    tiles_n = n // 16
    weight_q = torch.zeros_like(data)
    # Raw (unpacked) indices: (tiles_k, tiles_n, 256) int16
    raw_indices = torch.zeros(k // 16, tiles_n, 256, dtype=torch.int16, device=device)
    qa = {"K": K, "mcg": True}
    perm = tcp(device)
    perm_i = tcpi(device)

    for bi in range(0, k, 16):
        tk = bi // 16
        rows = data[bi:bi + 16]
        tiles = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)
        tiles = tiles[:, perm].contiguous()
        quant_w, quant_idx = qtf(tiles, qa)
        # Store raw indices — quant_idx has shape (n_tiles, 256)
        raw_indices[tk] = quant_idx
        # Reconstruct
        quant_w = quant_w[:, perm_i].reshape(tiles_n, 16, 16).permute(1, 0, 2).reshape(16, n)
        weight_q[bi:bi + 16] = quant_w

    # Pack the raw indices to EXL3 format: (tiles_k, tiles_n, K * 16) int16
    if ext is not None and hasattr(ext, "pack_trellis"):
        packed_shape = (k // 16, tiles_n, 256 * K // 16)
        packed = torch.zeros(packed_shape, dtype=torch.int16, device=device)
        ext.pack_trellis(packed, raw_indices.contiguous(), K)
    else:
        # Fallback: store raw indices (unpacked) — for testing without ext
        packed = raw_indices

    return weight_q, packed


def rescaled_trellis_quantize(
    base_q: torch.Tensor,
    residual: torch.Tensor,
    K_res: int,
    device: torch.device,
    tcp: Any, tcpi: Any, qtf: Any,
    cbs: float,
    ext: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Rescale residual to match codebook range, quantize, return (recon, packed, scale).

    The rescaling is the key MSRT innovation (v35 breakthrough):
      scale = |codebook_scale| / RMS(residual)
      quantized = trellis(residual * scale) / scale
    """
    residual_rms = residual.square().mean().sqrt().item()
    if residual_rms < 1e-12:
        packed_w = 256 * K_res // 16 if ext else 256
        return base_q, torch.zeros(
            residual.shape[0] // 16, residual.shape[1] // 16, packed_w,
            dtype=torch.int16, device=device), 1.0

    scale = abs(cbs) / residual_rms
    scaled = residual * scale
    recon_packed, packed = quantize_trellis_packed(scaled, K_res, device, tcp, tcpi, qtf, ext)
    recon = base_q + recon_packed / scale
    return recon, packed, scale


# ── Hadamard Vectors ──────────────────────────────────────────────────────

def compute_hadamard_vectors(
    w: torch.Tensor,
    device: torch.device,
    ghd: Any,
    cbs: float,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Compute the suh and svh vectors that EXL3 stores alongside trellis.

    These are the per-block sign vectors and scales from regularize().
    The exact format must match what exl3_gemm expects.
    """
    k, n = w.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    su = (torch.randn(k, generator=g).sign() + 1e-5).sign().float().to(device)
    sv = (torch.randn(n, generator=g).sign() + 1e-5).sign().float().to(device)

    out_scales = block_rms(w, dim=0, keepdim=True)
    mean = out_scales.mean().item()
    if mean > 1e-30:
        out_scales = out_scales / mean
    sv_full = (sv * out_scales + 1e-10).float()

    # After column Hadamard
    w_col = (w / sv_full).contiguous()
    had_n_mat = ghd(HADAMARD_BLOCK, device, torch.float, 1.0 / math.sqrt(HADAMARD_BLOCK))
    w_col = (w_col.view(k, n // HADAMARD_BLOCK, HADAMARD_BLOCK) @ had_n_mat).view(k, n).contiguous()

    in_scales = block_rms(w_col, dim=1, keepdim=True).clamp(min=1e-30)
    su_full = (su.unsqueeze(1) * in_scales / (-cbs) + 1e-10).float()

    return {"suh": su_full.squeeze().contiguous(), "svh": sv_full.squeeze().contiguous()}


# ── Encoding Pipeline ─────────────────────────────────────────────────────

def encode_expert_msrt(
    w_bf16: torch.Tensor,
    base_k: int,
    stages: list[dict[str, Any]],
    device: torch.device,
    ghd: Any, tcp: Any, tcpi: Any, qtf: Any,
    cbs: float,
    ext: Any = None,
) -> dict[str, Any]:
    """Encode one expert weight matrix with MSRT.

    Returns dict with:
      - base: {trellis, suh, svh} for the base tier
      - stages: list of {trellis, suh, svh, scale} for each residual stage
      - mse: reconstruction MSE vs original
    """
    w_reg = regularize(w_bf16, device, ghd, cbs)

    # Base tier
    base_recon, base_packed = quantize_trellis_packed(w_reg, base_k, device, tcp, tcpi, qtf, ext)
    had_vectors = compute_hadamard_vectors(w_bf16, device, ghd, cbs)

    result = {
        "base": {
            "trellis": base_packed.cpu(),
            "suh": had_vectors["suh"].cpu(),
            "svh": had_vectors["svh"].cpu(),
        },
        "stages": [],
    }

    current_recon = base_recon

    for stage in stages:
        residual = w_reg - current_recon
        recon, packed, scale = rescaled_trellis_quantize(
            current_recon, residual, stage["k"], device, tcp, tcpi, qtf, cbs, ext)
        result["stages"].append({
            "trellis": packed.cpu(),
            "suh": had_vectors["suh"].cpu(),  # Same Hadamard vectors
            "svh": had_vectors["svh"].cpu(),
            "scale": scale,
        })
        current_recon = recon

    result["mse"] = (w_reg - current_recon).pow(2).mean().item()
    return result


# ── Safetensors Output ────────────────────────────────────────────────────

def save_safetensors(tensors: dict[str, torch.Tensor], path: Path) -> None:
    """Save tensors as safetensors file."""
    from safetensors.torch import save_file
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))


def write_base_checkpoint(
    source_dir: Path,
    out_dir: Path,
    layer_results: dict[int, dict[int, dict[str, Any]]],
    moe_layers: list[int],
    tp: int = 1,
) -> None:
    """Write the base K checkpoint in standard EXL3 format.

    The base checkpoint has the same structure as a normal EXL3 quant:
    - config.json with hybrid_tr3_tail
    - tier_bitmap.json
    - model-layer-*.safetensors with trellis, suh, svh, mcg tensors
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy non-layer files from source
    import shutil
    for f in source_dir.iterdir():
        if f.is_file() and not f.name.startswith("model-layer-"):
            shutil.copy2(f, out_dir / f.name)

    # Write base tensors into layer shards
    for layer in moe_layers:
        if layer not in layer_results:
            continue
        tensors = {}
        for exp_id, exp_data in layer_results[layer].items():
            base = exp_data["base"]
            for proj_idx, proj in enumerate(PROJECTIONS):
                rank = 0  # TP=1 for Fruit model
                prefix = f"model.layers.{layer}.mlp.experts.{exp_id}.{proj}.rank{rank}"
                tensors[f"{prefix}.trellis"] = base["trellis"]
                tensors[f"{prefix}.suh"] = base["suh"]
                tensors[f"{prefix}.svh"] = base["svh"]
                # mcg sentinel
                import hashlib
                mcg_val = 0xCBAC1FED
                tensors[f"{prefix}.mcg"] = torch.tensor([mcg_val], dtype=torch.int32)

        shard_path = out_dir / f"model-layer-{layer:03d}.safetensors"
        save_safetensors(tensors, shard_path)
        print(f"  Base layer {layer}: {len(tensors)} tensors -> {shard_path.name}", flush=True)


def write_cartridge_adapter(
    out_dir: Path,
    layer_results: dict[int, dict[int, dict[str, Any]]],
    stage_idx: int,
    stage_label: str,
    moe_layers: list[int],
    expert_filter: dict[int, list[int]] | None = None,
) -> Path:
    """Write one cartridge stage as a LoRA-compatible safetensors adapter.

    If expert_filter is provided, only the specified experts per layer are included.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}

    for layer in moe_layers:
        if layer not in layer_results:
            continue
        for exp_id, exp_data in layer_results[layer].items():
            if expert_filter and layer in expert_filter:
                if exp_id not in expert_filter[layer]:
                    continue
            if stage_idx >= len(exp_data["stages"]):
                continue
            stage = exp_data["stages"][stage_idx]
            for proj in PROJECTIONS:
                rank = 0
                prefix = f"model.layers.{layer}.mlp.experts.{exp_id}.{proj}.rank{rank}"
                tensors[f"{prefix}.trellis_{stage_label}"] = stage["trellis"]
                tensors[f"{prefix}.suh_{stage_label}"] = stage["suh"]
                tensors[f"{prefix}.svh_{stage_label}"] = stage["svh"]
                tensors[f"{prefix}.scale_{stage_label}"] = torch.tensor([stage["scale"]], dtype=torch.float32)

    adapter_path = out_dir / f"cartridge_{stage_label}.safetensors"
    save_safetensors(tensors, adapter_path)

    # Write adapter_config.json
    config = {
        "schema": ADAPTER_CONFIG_SCHEMA,
        "stage_label": stage_label,
        "stage_k": stage_idx,
        "num_tensors": len(tensors),
        "tool_version": TOOL_VERSION,
    }
    (out_dir / f"cartridge_{stage_label}_config.json").write_text(
        json.dumps(config, indent=2) + "\n")

    print(f"  Cartridge '{stage_label}': {len(tensors)} tensors -> {adapter_path.name}", flush=True)
    return adapter_path


# ── Main Encode Command ───────────────────────────────────────────────────

def cmd_encode(args) -> int:
    """Encode a BF16 checkpoint into base K + cartridge adapters."""
    device = torch.device(args.device)
    print(f"Device: {device}  GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Bootstrap encoder
    ext, ghd, tcp, tcpi, qtf, cbs = bootstrap_encoder(args.encoder_source)
    print(f"codebook_scale = {cbs}", flush=True)

    # Load recipe
    recipe = json.loads(args.recipe.read_text())
    base_k = recipe["base_k"]
    stages = recipe["stages"]
    moe_layers = recipe.get("moe_layers", [])
    if not moe_layers:
        # Auto-detect from source config
        cfg_path = args.source / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            tail = cfg.get("hybrid_tr3_tail", {})
            if "moe_layers" in tail:
                moe_layers = list(range(tail["moe_layers"][0], tail["moe_layers"][1] + 1))
            else:
                moe_layers = list(range(
                    cfg.get("first_k_dense_replace", 3),
                    cfg.get("num_hidden_layers", 13)))
    print(f"Base K={base_k}, {len(stages)} cartridge stages, "
          f"MoE layers {moe_layers[0]}-{moe_layers[-1]} ({len(moe_layers)} layers)", flush=True)

    # Parse expert filters
    expert_filters = []
    hot_experts = recipe.get("hot_experts", list(range(96)))  # default: first 96
    for stage in stages:
        exp_spec = stage["experts"]
        if exp_spec == "all":
            expert_filters.append(None)
        elif isinstance(exp_spec, str) and exp_spec in ("hot96", "hot"):
            expert_filters.append({l: hot_experts for l in moe_layers})
        elif isinstance(exp_spec, list):
            expert_filters.append({l: exp_spec for l in moe_layers})
        else:
            expert_filters.append(None)

    # Load source weights and encode
    from safetensors import safe_open
    layer_results: dict[int, dict[int, dict[str, Any]]] = {}
    total_mse = 0.0
    n_experts_total = 0

    for layer in moe_layers:
        shard_path = args.source / f"model-layer-{layer:03d}.safetensors"
        if not shard_path.exists():
            # Try other shard naming patterns
            shard_path = args.source / f"model-layer-{layer:04d}.safetensors"
            if not shard_path.exists():
                print(f"  Layer {layer}: shard not found, skipping", flush=True)
                continue

        print(f"\nEncoding layer {layer}...", flush=True)
        layer_results[layer] = {}

        # Load expert weights from shard
        with safe_open(str(shard_path), framework="pt") as f:
            keys = list(f.keys())
            # Find expert keys for this layer
            import re
            expert_pattern = re.compile(EXPERT_RE_PATTERN)
            expert_weights: dict[int, dict[str, torch.Tensor]] = {}
            for key in keys:
                m = expert_pattern.match(key)
                if m:
                    l, e, proj = int(m.group(1)), int(m.group(2)), m.group(3)
                    if l == layer:
                        if e not in expert_weights:
                            expert_weights[e] = {}
                        expert_weights[e][proj] = f.get_tensor(key).float()

            print(f"  Found {len(expert_weights)} experts", flush=True)

            for exp_id in sorted(expert_weights.keys()):
                for proj in PROJECTIONS:
                    if proj not in expert_weights[exp_id]:
                        continue
                    w = expert_weights[exp_id][proj].to(device)
                    result = encode_expert_msrt(
                        w, base_k, stages, device, ghd, tcp, tcpi, qtf, cbs, ext)
                    del w
                    torch.cuda.empty_cache()

                    # Store under (exp_id, proj) — flatten for writing
                    key = (exp_id, proj)
                    if exp_id not in layer_results[layer]:
                        layer_results[layer][exp_id] = {"base": {}, "stages": [[] for _ in stages], "mse": {}}
                    layer_results[layer][exp_id]["base"][proj] = result["base"]
                    for si, stage_result in enumerate(result["stages"]):
                        layer_results[layer][exp_id]["stages"][si].append(stage_result)
                    layer_results[layer][exp_id]["mse"][proj] = result["mse"]
                    total_mse += result["mse"]
                    n_experts_total += 1

        # Print layer summary
        layer_mses = []
        for exp_data in layer_results[layer].values():
            layer_mses.extend(exp_data["mse"].values())
        avg = sum(layer_mses) / len(layer_mses) if layer_mses else 0
        print(f"  Layer {layer}: avg MSE = {avg:.4e} ({len(layer_mses)} projections)", flush=True)

    print(f"\nOverall avg MSE: {total_mse / max(n_experts_total, 1):.4e}", flush=True)

    # Write outputs
    out_dir = Path(args.out)
    print(f"\nWriting outputs to {out_dir}...", flush=True)

    # Restructure for writing: group by (layer, expert) -> {base: {trellis, suh, svh}, stages: [...]}
    # The layer_results is already structured, but we need to reorganize for the writers
    write_results: dict[int, dict[int, dict[str, Any]]] = {}
    for layer, experts in layer_results.items():
        write_results[layer] = {}
        for exp_id, exp_data in experts.items():
            # Merge projections into single base/stages
            base_tensors = {}
            for proj in PROJECTIONS:
                if proj in exp_data["base"]:
                    for tname, tval in exp_data["base"][proj].items():
                        base_tensors[f"{proj}_{tname}"] = tval

            stage_list = []
            for si in range(len(stages)):
                stage_tensors = {}
                for proj in PROJECTIONS:
                    if si < len(exp_data["stages"]) and proj == PROJECTIONS[0]:
                        # Only need one copy of suh/svh per expert
                        pass
                if si < len(exp_data["stages"]):
                    for proj_idx, proj in enumerate(PROJECTIONS):
                        if proj_idx < len(exp_data["stages"][si]):
                            stage = exp_data["stages"][si][proj_idx]
                            stage_tensors[f"{proj}"] = stage
                stage_list.append(stage_tensors)

            write_results[layer][exp_id] = {
                "base": base_tensors,
                "stages": stage_list,
                "mse": exp_data["mse"],
            }

    # Write base checkpoint
    print("\nWriting base checkpoint...", flush=True)
    base_dir = out_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    # Simplified: write all experts for each layer into one shard
    for layer in moe_layers:
        if layer not in write_results:
            continue
        tensors = {}
        for exp_id, exp_data in write_results[layer].items():
            for proj in PROJECTIONS:
                if f"{proj}_trellis" not in exp_data["base"]:
                    continue
                rank = 0
                prefix = f"model.layers.{layer}.mlp.experts.{exp_id}.{proj}.rank{rank}"
                tensors[f"{prefix}.trellis"] = exp_data["base"][f"{proj}_trellis"]
                tensors[f"{prefix}.suh"] = exp_data["base"][f"{proj}_suh"]
                tensors[f"{prefix}.svh"] = exp_data["base"][f"{proj}_svh"]
                tensors[f"{prefix}.mcg"] = torch.tensor([0xCBAC1FED], dtype=torch.uint32).view(torch.int32)
        if tensors:
            save_safetensors(tensors, base_dir / f"model-layer-{layer:03d}.safetensors")
            print(f"  Layer {layer}: {len(tensors)} base tensors", flush=True)

    # Copy non-layer files from source
    import shutil
    for f in args.source.iterdir():
        if f.is_file() and not f.name.startswith("model-layer-"):
            shutil.copy2(f, base_dir / f.name)

    # Write cartridge adapters
    print("\nWriting cartridge adapters...", flush=True)
    cart_dir = out_dir / "cartridges"

    for si, stage in enumerate(stages):
        label = stage["label"]
        expert_filter = expert_filters[si]
        tensors = {}

        for layer in moe_layers:
            if layer not in write_results:
                continue
            for exp_id, exp_data in write_results[layer].items():
                if expert_filter and layer in expert_filter:
                    if exp_id not in expert_filter[layer]:
                        continue
                if si >= len(exp_data["stages"]):
                    continue
                stage_data = exp_data["stages"][si]
                for proj in PROJECTIONS:
                    if proj not in stage_data:
                        continue
                    rank = 0
                    prefix = f"model.layers.{layer}.mlp.experts.{exp_id}.{proj}.rank{rank}"
                    s = stage_data[proj]
                    tensors[f"{prefix}.trellis_{label}"] = s["trellis"]
                    tensors[f"{prefix}.suh_{label}"] = s["suh"]
                    tensors[f"{prefix}.svh_{label}"] = s["svh"]
                    tensors[f"{prefix}.scale_{label}"] = torch.tensor([s["scale"]], dtype=torch.float32)

        adapter_path = cart_dir / f"cartridge_{label}.safetensors"
        save_safetensors(tensors, adapter_path)
        config = {
            "schema": ADAPTER_CONFIG_SCHEMA,
            "stage_label": label,
            "stage_k": stage["k"],
            "num_tensors": len(tensors),
            "tool_version": TOOL_VERSION,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (cart_dir / f"cartridge_{label}_config.json").write_text(
            json.dumps(config, indent=2) + "\n")
        print(f"  Cartridge '{label}': {len(tensors)} tensors -> {adapter_path.name}", flush=True)

    # Write summary
    summary = {
        "tool": TOOL_VERSION,
        "base_k": base_k,
        "stages": [{"k": s["k"], "label": s["label"],
                     "experts": s["experts"]} for s in stages],
        "moe_layers": moe_layers,
        "overall_mse": total_mse / max(n_experts_total, 1),
        "n_experts_encoded": n_experts_total,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "encoding_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nDone! Output: {out_dir}", flush=True)
    print(f"  Overall MSE: {summary['overall_mse']:.4e}", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="Encode BF16 → base K + cartridge adapters")
    enc.add_argument("--source", required=True, type=Path,
                     help="Source BF16 checkpoint directory")
    enc.add_argument("--recipe", required=True, type=Path,
                     help="Cartridge recipe JSON (fq-cartridge/1)")
    enc.add_argument("--out", required=True, type=Path,
                     help="Output directory (base/ and cartridges/ subdirs)")
    enc.add_argument("--encoder-source", required=True, type=Path,
                     help="Path to exllamav3 Python package")
    enc.add_argument("--device", default="cuda:0")

    args = p.parse_args(argv)
    if args.command == "encode":
        return cmd_encode(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
