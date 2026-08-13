#!/usr/bin/env python3
"""Encode -> decode parity for MSRT cartridges, against the real EXL3 kernels.

Everything else in the suite runs the encoder with a stand-in quantizer, which
proves the orchestration but says nothing about whether the *packed* trellis a
cartridge ships actually reconstructs the reconstruction the encoder measured.
That is the claim the runtime depends on:

    reconstruct(base) + sum(reconstruct(stage) / stage_scale)  ==  encoder MSE

This test proves it with the trusted exllamav3 build: ``ext.reconstruct`` is
the same dequantization kernel the EXL3 runtime uses, and the Hadamard/scale
restoration is done with upstream's own ``preapply_had_*`` helpers rather than
this repo's ``inverse_regularize``, so a disagreement between the two inverse
paths also fails here.

Needs a real GPU build. Point ``FQ_ENCODER_SOURCE`` at an exllamav3 package
with ``exllamav3_ext`` importable:

    FQ_ENCODER_SOURCE=/opt/exllamav3-python/exllamav3 \\
      PYTHONPATH=/opt/exllamav3 pytest tests/test_msrt_decode_parity.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import fq_assemble_lora as lora

ENCODER_SOURCE = os.environ.get("FQ_ENCODER_SOURCE")
QUANTIZE_MODULE = "exllamav3.modules.quant.exl3_lib.quantize"

BASES = [{"label": "k2", "k": 2}, {"label": "k3", "k": 3}]
STAGES = [
    {"label": "k2r1", "k": 1, "parent": "k2", "experts": "all"},
    {"label": "k2r1r1", "k": 1, "parent": "k2r1", "experts": "all"},
    {"label": "k2r2", "k": 2, "parent": "k2", "experts": "all"},
    {"label": "k2r2r1", "k": 1, "parent": "k2r2", "experts": "all"},
    {"label": "k3r1", "k": 1, "parent": "k3", "experts": "all"},
    {"label": "k3r1r1", "k": 1, "parent": "k3r1", "experts": "all"},
    {"label": "k3r2", "k": 2, "parent": "k3", "experts": "all"},
]
BITS = {node["label"]: node["k"] for node in [*BASES, *STAGES]}
CHAINS = {
    "k2": ("k2", []),
    "k2r1": ("k2", ["k2r1"]),
    "k2r1r1": ("k2", ["k2r1", "k2r1r1"]),
    "k2r2": ("k2", ["k2r2"]),
    "k2r2r1": ("k2", ["k2r2", "k2r2r1"]),
    "k3": ("k3", []),
    "k3r1": ("k3", ["k3r1"]),
    "k3r1r1": ("k3", ["k3r1", "k3r1r1"]),
    "k3r2": ("k3", ["k3r2"]),
}


def _dequantize(node, bits, quantize_module, ext, device):
    """Reconstruct one node in original weight space, upstream's way."""
    import torch

    suh, svh = node["suh"].to(device), node["svh"].to(device)
    weight = torch.empty(
        (suh.numel(), svh.numel()), dtype=torch.half, device=device)
    ext.reconstruct(weight, node["trellis"].to(device), bits, True, False)
    weight = quantize_module.preapply_had_l(weight, lora.HADAMARD_BLOCK)
    weight = weight * suh.unsqueeze(1)
    weight = quantize_module.preapply_had_r(weight, lora.HADAMARD_BLOCK)
    return (weight * svh.unsqueeze(0)).float()


@pytest.fixture(scope="module")
def decoded():
    """Encode one random matrix into the requested graph, then decode it."""
    torch = pytest.importorskip("torch")
    if not ENCODER_SOURCE:
        pytest.skip("set FQ_ENCODER_SOURCE to an exllamav3 package")
    if not torch.cuda.is_available():
        pytest.skip("the EXL3 trellis encoder needs CUDA")

    device = torch.device("cuda:0")
    generator = torch.Generator().manual_seed(11)
    weight = (torch.randn(512, 256, generator=generator) * 0.02).to(device)
    with lora.bootstrap_encoder(Path(ENCODER_SOURCE)) as enc:
        quantize_module = sys.modules[QUANTIZE_MODULE]
        nodes = lora.encode_matrix_dag(weight, BASES, STAGES, device, enc)
        sums = {}
        for label, (base, chain) in CHAINS.items():
            total = _dequantize(
                nodes[base], BITS[base], quantize_module, enc.ext, device)
            for stage in chain:
                node = nodes[stage]
                total = total + _dequantize(
                    node, BITS[stage], quantize_module, enc.ext, device
                ) / node["scale"]
            sums[label] = (weight.float() - total).square().mean().item()
    return weight, nodes, sums


@pytest.mark.parametrize("label", sorted(CHAINS))
def test_runtime_sum_reproduces_the_encoder_reconstruction(decoded, label):
    _weight, nodes, sums = decoded
    reported = nodes[label]["mse"]
    assert sums[label] == pytest.approx(reported, rel=0.05), (
        f"{label}: decoded MSE {sums[label]:.6e} != reported {reported:.6e}; "
        f"the shipped trellis does not reconstruct what the encoder measured")


def test_deeper_chains_reduce_error(decoded):
    _weight, nodes, _sums = decoded
    assert nodes["k2"]["mse"] > nodes["k2r1"]["mse"] > nodes["k2r1r1"]["mse"]
    assert nodes["k2"]["mse"] > nodes["k2r2"]["mse"] > nodes["k2r2r1"]["mse"]
    assert nodes["k3"]["mse"] > nodes["k3r1"]["mse"] > nodes["k3r1r1"]["mse"]
    assert nodes["k3"]["mse"] > nodes["k3r2"]["mse"]
    # A 5 bpw MSRT stack must beat the 3 bpw single tier it is compared against.
    assert nodes["k2r2r1"]["mse"] < nodes["k3"]["mse"]


def test_packed_trellis_geometry_matches_declared_bitrates(decoded):
    import torch

    weight, nodes, _sums = decoded
    k, n = weight.shape
    for label, node in nodes.items():
        assert tuple(node["trellis"].shape) == (k // 16, n // 16, BITS[label] * 16)
        assert node["trellis"].dtype is torch.int16
        assert (node["scale"] is None) == (label in {"k2", "k3"})


def test_zero_residual_stage_decodes_to_a_no_op(decoded):
    """A converged expert must not receive a fabricated correction.

    An all-zero packed trellis decodes to a *nonzero* codebook value, so the
    encoder floors the residual RMS instead of shipping zeros. Note that a zero
    weight does not reach this path: the trellis reconstructs zeros as codebook
    values, leaving an ordinary residual. Only an exactly-converged parent does,
    so the floor is exercised directly here, on the real kernel.
    """
    import torch

    _weight, _nodes, _sums = decoded
    device = torch.device("cuda:0")
    generator = torch.Generator().manual_seed(5)
    parent = (torch.randn(512, 256, generator=generator) * 0.02).to(device)
    with lora.bootstrap_encoder(Path(ENCODER_SOURCE)) as enc:
        quantize_module = sys.modules[QUANTIZE_MODULE]
        recon, packed, scale = lora.rescaled_trellis_quantize(
            parent, torch.zeros_like(parent), 1, enc)
        assert scale == pytest.approx(abs(enc.cbs) / lora.MIN_RESIDUAL_RMS)
        assert packed.any(), "a real trellis never encodes as all-zero indices"
        correction = _dequantize(
            {"trellis": packed,
             "suh": torch.ones(512, dtype=torch.float16),
             "svh": torch.ones(256, dtype=torch.float16)},
            1, quantize_module, enc.ext, device) / scale
    assert torch.isfinite(correction).all()
    assert correction.abs().max().item() < 1e-6, (
        "an exhausted residual must decode to nothing, not to a codebook value")
    assert torch.equal(recon, parent + torch.zeros_like(parent)) or (
        (recon - parent).abs().max().item() < 1e-6)


@pytest.mark.skipif(not os.environ.get("FQ_PARITY_SOURCE"),
                    reason="set FQ_PARITY_SOURCE to a per-layer BF16 checkpoint")
def test_published_campaign_decodes_to_its_measured_reconstruction(tmp_path):
    """The whole artifact path, on real weights, through the real kernel.

    Encode a two-block campaign from a real BF16 checkpoint, finalize it,
    combine one product under a pinned signer, then decode the *combined
    adapter* plus the *published base shard* and compare against the source
    weights. This is the claim a consumer relies on, checked end to end rather
    than on in-memory tensors.
    """
    import json
    from types import SimpleNamespace

    import torch
    from safetensors import safe_open

    sys.path.insert(0, str(Path(__file__).parent))
    import fq_combine_cartridges as combine

    source = Path(os.environ["FQ_PARITY_SOURCE"])
    layer = int(os.environ.get("FQ_PARITY_LAYER", "3"))
    campaign = tmp_path / "campaign"
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({
        "schema": lora.CARTRIDGE_SCHEMA,
        "bases": [{"label": "k2", "k": 2}],
        "stages": [{"label": "r1", "k": 1, "parent": "k2", "experts": "all"},
                   {"label": "r2", "k": 2, "parent": "r1", "experts": "all"}],
        "assemblies": [{"label": "k5like", "base": "k2", "chain": ["r1", "r2"]}],
        "moe_layers": [layer],
    }))
    args = SimpleNamespace(
        source=source, recipe=recipe, out=campaign, block_size=2,
        encoder_source=Path(ENCODER_SOURCE), device="cuda:0", devices=None,
        layers=None, shard_index=0, shard_count=1, force=False,
        tile_batch=lora.TILE_BATCH, sign_key=tmp_path / "sign.key",
        base_model="local/parity", base_revision="a" * 40,
        source_digests=None)
    assert lora.cmd_skeleton(args) == 0
    # One block is enough to prove the contract and keeps the test minutes long.
    args.shard_count = max(
        1, len(lora.expert_blocks(
            list(lora.SourceCheckpoint(source).experts(layer)), 2)))
    assert lora.cmd_encode(args) == 0
    args.shard_count = 1
    assert lora.cmd_encode(args) == 0
    assert lora.cmd_finalize(args) == 0

    key = json.loads(
        (campaign / "campaign_summary.json").read_text()
    )["provenance"]["signer_pubkey"]
    adapter = tmp_path / "k5like"
    assert combine.combine(SimpleNamespace(
        root=campaign, assembly="k5like", out=adapter, experts="0,1",
        layers=None, force=False, trust_key=key,
        base=campaign / "base" / "k2", insecure_unsigned=False)) == 0

    device = torch.device("cuda:0")
    base_shard = campaign / "base" / "k2" / lora.block_name(layer, 0)
    combined = adapter / lora.block_name(layer, 0)
    with lora.bootstrap_encoder(Path(ENCODER_SOURCE)) as enc:
        quantize_module = sys.modules[QUANTIZE_MODULE]
        with safe_open(str(base_shard), framework="pt") as base, \
                safe_open(str(combined), framework="pt") as stages, \
                lora.SourceCheckpoint(source) as checkpoint:
            experts = checkpoint.experts(layer)
            shards = checkpoint.layer_keys(layer)
            worst = 0.0
            for expert in (0, 1):
                for projection in lora.PROJECTIONS:
                    prefix = (f"model.layers.{layer}.mlp.experts.{expert}."
                              f"{projection}.rank0")
                    key_name = experts[expert][projection]
                    reference = checkpoint.tensor(
                        shards[key_name], key_name).to(device).float().T
                    total = _dequantize(
                        {"trellis": base.get_tensor(f"{prefix}.trellis"),
                         "suh": base.get_tensor(f"{prefix}.suh"),
                         "svh": base.get_tensor(f"{prefix}.svh")},
                        2, quantize_module, enc.ext, device)
                    for label, bits in (("r1", 1), ("r2", 2)):
                        scale = float(stages.get_tensor(f"{prefix}.scale_{label}"))
                        total = total + _dequantize(
                            {"trellis": stages.get_tensor(
                                f"{prefix}.trellis_{label}"),
                             "suh": stages.get_tensor(f"{prefix}.suh_{label}"),
                             "svh": stages.get_tensor(f"{prefix}.svh_{label}")},
                            bits, quantize_module, enc.ext, device) / scale
                    mse = (reference - total).square().mean().item()
                    reported = json.loads(
                        lora.read_attestation(
                            campaign / "stages" / "r2",
                            lora.block_name(layer, 0)
                        )["cartridge"]["mean_mse_original_space"].__repr__())
                    worst = max(worst, mse / reported)
    assert 0.2 < worst < 5.0, (
        f"published 5 bpw product decodes to {worst:.2f}x the MSE the campaign "
        f"attested for its deepest stage")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
