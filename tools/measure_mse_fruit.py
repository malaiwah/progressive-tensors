#!/usr/bin/env python3
"""Measure weight-level MSE between Fruit model quantization variants.

Compares:
1. BF16 original (reference)
2. SIQ quant (160 K3 + 96 K4 per layer)
3. MSRT K2 base (all experts at K2)
4. MSRT K2+K1trsc (all experts at K3-equivalent)
5. MSRT K2+K1trsc+K2trsc (96 hot experts at K4-equivalent)

All measurements in regularized (Hadamard) space, which equals original
space since Hadamard is orthogonal.
"""
from __future__ import annotations
import json, math, os, sys, types, importlib.util, gc
from pathlib import Path
import torch

EXL3_PKG = "/opt/fruit-pip/exllamav3"

def _bootstrap():
    pkg = types.ModuleType("exllamav3"); pkg.__path__ = [EXL3_PKG]; sys.modules["exllamav3"] = pkg
    for sub in ["util", "modules", "modules.quant", "modules.quant.exl3_lib"]:
        full = f"exllamav3.{sub}"; m = types.ModuleType(full)
        m.__path__ = [f"{EXL3_PKG}/{sub.replace('.', '/')}"]; sys.modules[full] = m
    class _DPB:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def update(self, *a): pass
        def new_task(self, *a, **kw): pass
    _s = types.ModuleType("exllamav3.util.progress"); _s.ProgressBar = _DPB; sys.modules["exllamav3.util.progress"] = _s
    _s = types.ModuleType("exllamav3.util.memory"); _s.free_mem = lambda: None; _s.list_gpu_tensors = lambda: []; sys.modules["exllamav3.util.memory"] = _s
    _s = types.ModuleType("exllamav3.util"); _s.__path__ = [f"{EXL3_PKG}/util"]; _s.cuda_sync_active = lambda *a, **kw: torch.cuda.synchronize(); sys.modules["exllamav3.util"] = _s
    _s = types.ModuleType("exllamav3.util.tensor"); _s.save_tensor_image = lambda *a, **kw: None; sys.modules["exllamav3.util.tensor"] = _s
    spec = importlib.util.spec_from_file_location("exllamav3.ext", f"{EXL3_PKG}/ext.py")
    m = importlib.util.module_from_spec(spec); sys.modules["exllamav3.ext"] = m; spec.loader.exec_module(m)
    ext = m.exllamav3_ext
    spec = importlib.util.spec_from_file_location("exllamav3.util.hadamard", f"{EXL3_PKG}/util/hadamard.py")
    m = importlib.util.module_from_spec(spec); sys.modules["exllamav3.util.hadamard"] = m; spec.loader.exec_module(m)
    ghd = m.get_hadamard_dt
    spec = importlib.util.spec_from_file_location("exllamav3.modules.quant.exl3_lib.quantize", f"{EXL3_PKG}/modules/quant/exl3_lib/quantize.py")
    m = importlib.util.module_from_spec(spec); sys.modules["exllamav3.modules.quant.exl3_lib.quantize"] = m; spec.loader.exec_module(m)
    return ext, ghd, m.tensor_core_perm, m.tensor_core_perm_i, m.quantize_tiles, m.codebook_scale

def block_rms(x, dim, keepdim=False):
    return x.square().mean(dim=dim, keepdim=keepdim).sqrt()

def regularize(w, device, ghd, cbs, had_k=128, had_n=128, seed=0):
    k, n = w.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    su = (torch.randn(k, generator=g).sign() + 1e-5).sign().float().to(device)
    sv = (torch.randn(n, generator=g).sign() + 1e-5).sign().float().to(device)
    out_scales = block_rms(w, dim=0, keepdim=True)
    mean = out_scales.mean().item()
    if mean > 1e-30: out_scales = out_scales / mean
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

def quantize_trellis_raw(data, K, device, tcp, tcpi, qtf):
    k, n = data.shape; tiles_n = n // 16; weight_q = torch.zeros_like(data)
    qa = {"K": K, "mcg": True}
    perm = tcp(device); perm_i = tcpi(device)
    for bi in range(0, k, 16):
        rows = data[bi:bi+16]
        tiles = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)
        tiles = tiles[:, perm].contiguous()
        quant_w, _ = qtf(tiles, qa)
        quant_w = quant_w[:, perm_i].reshape(tiles_n, 16, 16).permute(1, 0, 2).reshape(16, n)
        weight_q[bi:bi+16] = quant_w
    return weight_q

def rescaled_trellis(base_q, residual, K_res, device, tcp, tcpi, qtf, cbs):
    residual_rms = residual.square().mean().sqrt().item()
    if residual_rms < 1e-12: return base_q
    scale = abs(cbs) / residual_rms
    scaled = residual * scale
    quant = quantize_trellis_raw(scaled, K_res, device, tcp, tcpi, qtf)
    return base_q + quant / scale

def run_measurement(bf16_path, siq_path, device, ghd, tcp, tcpi, qtf, cbs):
    """Measure weight-level MSE for all configurations."""
    from safetensors import safe_open
    import re

    # SIQ tier_bitmap: 160 K3 + 96 K4 per layer
    tier_path = Path(siq_path) / "tier_bitmap.json"
    tier_bitmap = json.loads(tier_path.read_text()) if tier_path.exists() else {}

    # Load SIQ trellis weights and compare to BF16
    moe_layers = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    results = {}

    for layer in moe_layers[:3]:  # First 3 layers for speed
        bf16_shard = bf16_path / f"model-layer-{layer:03d}.safetensors"
        siq_shard = siq_path / f"model-layer-{layer:03d}.safetensors"

        if not bf16_shard.exists():
            print(f"  Layer {layer}: bf16 shard not found", flush=True)
            continue

        print(f"\n=== Layer {layer} ===", flush=True)

        # Load BF16 expert weights
        bf16_experts = {}
        with safe_open(str(bf16_shard), framework="pt") as f:
            for key in f.keys():
                if f"layers.{layer}.mlp.experts." in key and key.endswith(".weight"):
                    parts = key.split(".")
                    eid = int(parts[5])
                    proj = parts[6]
                    if eid not in bf16_experts: bf16_experts[eid] = {}
                    bf16_experts[eid][proj] = f.get_tensor(key).float()

        # Load SIQ tier info
        k_list = tier_bitmap.get(str(layer), {}).get("k", [3] * 256)

        # Measure each config
        n_experts = min(10, len(bf16_experts))  # First 10 experts for speed
        configs_mse = {name: [] for name in [
            "K3_all", "K4_all", "SIQ_mixed", "MSRT_K2", "MSRT_K2_K1trsc", "MSRT_K2_K1trsc_K2trsc"]}

        for eid in sorted(bf16_experts.keys())[:n_experts]:
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                if proj not in bf16_experts[eid]:
                    continue
                w = bf16_experts[eid][proj].to(device)
                w_reg = regularize(w, device, ghd, cbs)
                del w

                # K3
                q_k3 = quantize_trellis_raw(w_reg, 3, device, tcp, tcpi, qtf)
                configs_mse["K3_all"].append((w_reg - q_k3).pow(2).mean().item())

                # K4
                q_k4 = quantize_trellis_raw(w_reg, 4, device, tcp, tcpi, qtf)
                configs_mse["K4_all"].append((w_reg - q_k4).pow(2).mean().item())

                # SIQ mixed: K3 or K4 depending on tier
                siq_k = k_list[eid] if eid < len(k_list) else 3
                q_siq = quantize_trellis_raw(w_reg, siq_k, device, tcp, tcpi, qtf)
                configs_mse["SIQ_mixed"].append((w_reg - q_siq).pow(2).mean().item())

                # MSRT K2 base
                q_k2 = quantize_trellis_raw(w_reg, 2, device, tcp, tcpi, qtf)
                configs_mse["MSRT_K2"].append((w_reg - q_k2).pow(2).mean().item())

                # MSRT K2 + K1trsc (K3-equivalent)
                r2 = w_reg - q_k2
                q_msrt3 = rescaled_trellis(q_k2, r2, 1, device, tcp, tcpi, qtf, cbs)
                configs_mse["MSRT_K2_K1trsc"].append((w_reg - q_msrt3).pow(2).mean().item())

                # MSRT K2 + K1trsc + K2trsc (K4-equivalent, for hot experts)
                r3 = w_reg - q_msrt3
                q_msrt4 = rescaled_trellis(q_msrt3, r3, 2, device, tcp, tcpi, qtf, cbs)
                configs_mse["MSRT_K2_K1trsc_K2trsc"].append((w_reg - q_msrt4).pow(2).mean().item())

                del w_reg, q_k3, q_k4, q_siq, q_k2, q_msrt3, q_msrt4
                torch.cuda.empty_cache()

        # Print results
        print(f"  {'Config':<30} {'avg MSE':>12} {'min':>12} {'max':>12}", flush=True)
        print(f"  {'-'*70}", flush=True)
        layer_results = {}
        for name, mses in configs_mse.items():
            avg = sum(mses) / len(mses) if mses else 0
            mn = min(mses) if mses else 0
            mx = max(mses) if mses else 0
            print(f"  {name:<30} {avg:>12.4e} {mn:>12.4e} {mx:>12.4e}", flush=True)
            layer_results[name] = {"avg": avg, "min": mn, "max": mx, "n": len(mses)}

        results[f"layer{layer}"] = layer_results

    return results

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16-path", required=True, type=Path)
    ap.add_argument("--siq-path", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="/tmp/poc_residual/mse_results.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Device: {dev}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    ext, ghd, tcp, tcpi, qtf, cbs = _bootstrap()
    print(f"codebook_scale = {cbs}", flush=True)
    results = run_measurement(args.bf16_path, args.siq_path, dev, ghd, tcp, tcpi, qtf, cbs)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {args.out}", flush=True)

if __name__ == "__main__":
    main()
