#!/usr/bin/env python3
"""Assemble combined MSRT cartridges from individual stage files.

Since vLLM supports only 1 LoRA per request, we need single adapter files that
contain all stages for a given recipe:

1. cart_k3like.safetensors: K1trsc for ALL 256 experts (K2→K3, 3bpw total)
2. cart_k3k4like.safetensors: K1trsc for all + K2trsc for 96 hot (K2→K3/K4, 3.375bpw)
   - 160 non-hot experts: only K1trsc stage (K3-equivalent)
   - 96 hot experts: K1trsc + K2trsc stages (K4-equivalent)
"""
import json
import sys
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import torch

def combine_cartridges(
    res1_path: Path,   # K1trsc for all experts
    res2_path: Path,   # K2trsc for hot experts
    hot_experts: list[int],
    out_path: Path,
    label1: str = "res1",
    label2: str = "res2",
):
    """Combine two stage files into a single adapter with both stages.

    For hot experts: both res1 and res2 tensors are included
    For non-hot experts: only res1 tensors are included
    """
    tensors = {}
    
    # Load res1 (K1trsc) for ALL experts
    with safe_open(str(res1_path), framework="pt") as f:
        keys = list(f.keys())
        for key in keys:
            tensors[key] = f.get_tensor(key)
    
    print(f"Loaded {len(tensors)} tensors from res1 (K1trsc, all experts)", flush=True)
    
    # Load res2 (K2trsc) only for hot experts
    hot_set = set(hot_experts)
    res2_count = 0
    with safe_open(str(res2_path), framework="pt") as f:
        for key in f.keys():
            # Parse expert ID from key: model.layers.{L}.mlp.experts.{E}.{proj}.rank{R}.trellis_{label}
            parts = key.split(".")
            expert_id = int(parts[5])
            if expert_id in hot_set:
                tensors[key] = f.get_tensor(key)
                res2_count += 1
    
    print(f"Loaded {res2_count} tensors from res2 (K2trsc, {len(hot_set)} hot experts)", flush=True)
    print(f"Combined: {len(tensors)} total tensors", flush=True)
    
    # Write combined adapter
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path))
    print(f"Saved combined cartridge: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    
    # Write config
    config = {
        "schema": "fq-cartridge-adapter/1",
        "stages": [
            {"k": 1, "label": label1, "experts": "all"},
            {"k": 2, "label": label2, "experts": hot_experts},
        ],
        "description": "Combined K3/K4-like cartridge matching SIQ 160K3+96K4 allocation",
        "num_tensors": len(tensors),
    }
    config_path = out_path.with_suffix(".config.json")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return config

def copy_k3only_cartridge(
    res1_path: Path,
    out_path: Path,
    label: str = "res1",
):
    """Create a K3-equivalent-only cartridge (just res1, all experts)."""
    tensors = {}
    with safe_open(str(res1_path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path))
    print(f"Saved K3-like cartridge: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    
    config = {
        "schema": "fq-cartridge-adapter/1",
        "stages": [{"k": 1, "label": label, "experts": "all"}],
        "description": "K3-equivalent cartridge (K2+K1trsc, all experts, 3bpw)",
        "num_tensors": len(tensors),
    }
    config_path = out_path.with_suffix(".config.json")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return config

if __name__ == "__main__":
    cart_dir = Path("/tmp/poc_residual/fruit_msrt_output/cartridges")
    out_dir = cart_dir
    
    # Hot experts: first 96 (matching SIQ's K4 tier)
    hot_experts = list(range(96))
    
    # 1. K3-like only (base + K1trsc = 3bpw on all experts)
    print("=== Assembling K3-like cartridge (all experts, K1trsc only) ===", flush=True)
    copy_k3only_cartridge(
        cart_dir / "cartridge_res1.safetensors",
        out_dir / "cart_k3like.safetensors",
    )
    
    # 2. K3/K4-like combined (matches SIQ 160K3 + 96K4)
    print("\n=== Assembling K3/K4-like combined cartridge (160×K3 + 96×K4) ===", flush=True)
    combine_cartridges(
        cart_dir / "cartridge_res1.safetensors",
        cart_dir / "cartridge_res2.safetensors",
        hot_experts,
        out_dir / "cart_k3k4like.safetensors",
    )
    
    print("\nDone! Combined cartridges:", flush=True)
    print(f"  cart_k3like.safetensors  (K3-equivalent, all 256 experts)", flush=True)
    print(f"  cart_k3k4like.safetensors (K3/K4 mix, matches SIQ 160K3+96K4)", flush=True)
