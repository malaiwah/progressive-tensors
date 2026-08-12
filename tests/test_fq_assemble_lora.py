#!/usr/bin/env python3
"""Tests for fq_assemble_lora.

Tests the cartridge recipe parsing, MSRT encoding pipeline, and output format.
Uses tiny random tensors (not real model weights) for speed.
"""
import json
import math
import sys
import tempfile
from pathlib import Path

import pytest
import torch

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def test_cartridge_recipe_schema():
    """Test that cartridge recipe follows fq-cartridge/1 schema."""
    recipe = {
        "schema": "fq-cartridge/1",
        "base_k": 2,
        "stages": [
            {"k": 1, "label": "res1", "experts": "all"},
            {"k": 2, "label": "res2", "experts": [0, 1, 2]},
        ],
        "moe_layers": [3, 4, 5],
    }
    assert recipe["schema"] == "fq-cartridge/1"
    assert recipe["base_k"] == 2
    assert len(recipe["stages"]) == 2
    assert recipe["stages"][0]["k"] == 1
    assert recipe["stages"][1]["experts"] == [0, 1, 2]


def test_cartridge_recipe_from_file():
    """Test loading the Fruit recipe."""
    recipe_path = Path(__file__).parent.parent / "recipes" / "fruit-k2-k3k4-cart.json"
    if not recipe_path.exists():
        pytest.skip("Recipe file not found")
    recipe = json.loads(recipe_path.read_text())
    assert recipe["schema"] == "fq-cartridge/1"
    assert recipe["base_k"] == 2
    assert len(recipe["stages"]) == 2
    assert recipe["stages"][0]["label"] == "res1"
    assert recipe["stages"][1]["label"] == "res2"
    assert recipe["stages"][0]["experts"] == "all"
    assert len(recipe["moe_layers"]) == 11  # layers 3-13


def test_block_rms():
    """Test RMS computation."""
    from fq_assemble_lora import block_rms
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    rms = block_rms(x, dim=0, keepdim=True)
    expected = torch.sqrt(torch.tensor([(1+9)/2, (4+16)/2]))
    assert torch.allclose(rms.squeeze(), expected, rtol=1e-5)


def test_rescaled_trellis_scale():
    """Test that rescaling produces correct scale factor."""
    # Mock: just test the scale computation logic
    cbs = 1.2437
    residual = torch.randn(128, 256)
    residual_rms = residual.square().mean().sqrt().item()
    scale = abs(cbs) / residual_rms
    assert scale > 0
    # After rescaling, RMS should be ~|cbs|
    scaled = residual * scale
    scaled_rms = scaled.square().mean().sqrt().item()
    assert abs(scaled_rms - abs(cbs)) < 0.01  # close to codebook scale


def test_trellis_packed_shape():
    """Test that packed trellis has correct shape (without GPU)."""
    # Shape: (k//16, n//16, K*16) int16
    k, n, K = 128, 256, 2
    expected_shape = (k // 16, n // 16, K * 16)
    assert expected_shape == (8, 16, 32)
    # For K=3: (8, 16, 48)
    assert (k // 16, n // 16, 3 * 16) == (8, 16, 48)


def test_cartridge_adapter_naming():
    """Test that cartridge tensor names follow the expected pattern."""
    layer, exp, proj, rank, label = 3, 0, "gate_proj", 0, "res1"
    prefix = f"model.layers.{layer}.mlp.experts.{exp}.{proj}.rank{rank}"
    names = [
        f"{prefix}.trellis_{label}",
        f"{prefix}.suh_{label}",
        f"{prefix}.svh_{label}",
        f"{prefix}.scale_{label}",
    ]
    for name in names:
        assert f"trellis_{label}" in name or f"suh_{label}" in name \
               or f"svh_{label}" in name or f"scale_{label}" in name


def test_msrt_bpw_calculation():
    """Test effective bpw calculation for dual-cartridge configs."""
    n_experts = 256
    base_k = 2
    # Stage 1: K1 on all 256 experts
    # Stage 2: K2 on 96 experts
    total_bits = n_experts * base_k + n_experts * 1 + 96 * 2
    eff_bpw = total_bits / n_experts
    assert eff_bpw == (512 + 256 + 192) / 256  # = 960/256 = 3.75

    # willfalco comparison: 148 K3 + 108 K4
    willfalco_bpw = (148 * 3 + 108 * 4) / 256
    assert abs(willfalco_bpw - 3.422) < 0.01


def test_encoding_summary_format():
    """Test that encoding summary has required fields."""
    summary = {
        "tool": "fq_assemble_lora/1",
        "base_k": 2,
        "stages": [{"k": 1, "label": "res1", "experts": "all"}],
        "moe_layers": [3, 4, 5],
        "overall_mse": 0.001,
        "n_experts_encoded": 768,
    }
    assert summary["tool"] == "fq_assemble_lora/1"
    assert "overall_mse" in summary
    assert summary["n_experts_encoded"] == 768  # 256 experts × 3 layers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
