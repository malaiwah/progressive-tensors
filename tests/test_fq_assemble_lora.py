#!/usr/bin/env python3
"""Behavioral tests for the MSRT cartridge encoder."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import fq_assemble_lora as lora

RECIPE = Path(__file__).parent.parent / "recipes" / "fruit-k2-k3k4-cart.json"


def write_recipe(path: Path, **overrides) -> Path:
    recipe = {
        "schema": "fq-cartridge/1",
        "base_k": 2,
        "stages": [{"k": 1, "label": "res1", "experts": "all"}],
        "moe_layers": [3],
    }
    recipe.update(overrides)
    path.write_text(json.dumps(recipe))
    return path


def expert_key(layer: int, expert: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"


def complete_keys(layer: int = 3, expert: int = 0) -> list[str]:
    return [expert_key(layer, expert, projection) for projection in lora.PROJECTIONS]


def test_fruit_recipe_is_valid_and_has_claimed_bitrate():
    recipe = lora.load_recipe(RECIPE)
    assert recipe["base_k"] == 2
    assert recipe["stages"][1]["k"] == 1
    assert len(recipe["stages"][1]["experts"]) == 96
    assert lora.effective_bpw(recipe, 256) == pytest.approx(3.375)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema": "fq-cartridge/2"}, "schema must be"),
        ({"base_k": 0}, "base_k must be"),
        ({"moe_layers": []}, "moe_layers"),
        ({"stages": [{"k": 1, "label": "../../escape", "experts": "all"}]},
         "label must match"),
        ({"stages": [{"k": 1, "label": "x", "experts": "hot96"}]},
         "experts must be"),
        ({"stages": [
            {"k": 1, "label": "same", "experts": "all"},
            {"k": 1, "label": "same", "experts": [0]},
        ]}, "duplicate label"),
    ],
)
def test_recipe_validation_rejects_unsafe_or_ambiguous_input(
    tmp_path: Path, overrides: dict, message: str
):
    path = write_recipe(tmp_path / "recipe.json", **overrides)
    with pytest.raises((lora.CartridgeError, ValueError), match=message):
        lora.load_recipe(path)


def test_stage_selection_supports_disjoint_expert_sets():
    stages = [
        {"k": 1, "label": "cold", "experts": [1]},
        {"k": 2, "label": "hot", "experts": [2]},
    ]
    assert [stage["label"] for stage in lora.selected_stages(stages, 1)] == ["cold"]
    assert [stage["label"] for stage in lora.selected_stages(stages, 2)] == ["hot"]
    assert lora.selected_stages(stages, 3) == []


def test_source_key_partition_preserves_router_norm_and_attention():
    keys = complete_keys() + [
        "model.layers.3.mlp.gate.weight",
        "model.layers.3.self_attn.q_proj.weight",
        "model.layers.3.input_layernorm.weight",
    ]
    experts = lora.inspect_source_layer(keys, 3)
    assert set(experts[0]) == set(lora.PROJECTIONS)
    assert lora.preserved_source_keys(keys) == keys[3:]


def test_missing_projection_fails_before_encoding():
    keys = complete_keys()
    keys.remove(expert_key(3, 0, "gate_proj"))
    with pytest.raises(lora.CartridgeError, match="projections"):
        lora.inspect_source_layer(keys, 3)


def test_foreign_layer_in_per_layer_shard_is_rejected():
    keys = complete_keys(3, 0) + complete_keys(4, 0)
    with pytest.raises(lora.CartridgeError, match="also contains"):
        lora.inspect_source_layer(keys, 3)


def test_unsupported_source_layout_and_empty_layers_fail(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model-00001-of-00002.safetensors").write_bytes(b"not-used")
    with pytest.raises(lora.CartridgeError, match="not supported"):
        lora.resolve_layer_shards(source, [3])
    with pytest.raises(lora.CartridgeError, match="moe_layers"):
        lora.load_recipe(write_recipe(tmp_path / "empty.json", moe_layers=[]))


def test_checkpoint_copy_preserves_unselected_shards_and_drops_stale_metadata(
    tmp_path: Path,
):
    source = tmp_path / "source"
    output = tmp_path / "base"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model-layer-002.safetensors").write_bytes(b"dense")
    (source / "model-layer-003.safetensors").write_bytes(b"selected")
    (source / "model.safetensors.index.json").write_text("stale")
    (source / "MANIFEST.sha256").write_text("stale")
    lora.copy_source_checkpoint(source, output, {"model-layer-003.safetensors"})
    assert (output / "model-layer-002.safetensors").read_bytes() == b"dense"
    assert not (output / "model-layer-003.safetensors").exists()
    assert not (output / "model.safetensors.index.json").exists()
    assert not (output / "MANIFEST.sha256").exists()


def identity_hadamard(size, device, dtype, scale):
    torch = pytest.importorskip("torch")
    return torch.eye(size, device=device, dtype=dtype)


def test_zero_weight_serialized_scales_are_finite_and_round_trip():
    torch = pytest.importorskip("torch")
    if lora.torch is None:
        pytest.skip("fq_assemble_lora imported without torch")
    weight = torch.zeros(128, 128)
    regularized, suh, svh = lora.regularize_with_vectors(
        weight, torch.device("cpu"), identity_hadamard, 1.2437)
    assert torch.isfinite(suh).all() and torch.isfinite(svh).all()
    assert (suh != 0).all() and (svh != 0).all()
    restored = lora.inverse_regularize(
        regularized, suh, svh, torch.device("cpu"), identity_hadamard)
    assert torch.equal(restored, weight)


class FakeExtension:
    def pack_trellis(self, packed, raw, bits):
        packed.zero_()


def identity_permutation(device):
    torch = pytest.importorskip("torch")
    return torch.arange(256, device=device)


def identity_quantizer(tiles, options):
    torch = pytest.importorskip("torch")
    return tiles.clone(), torch.zeros_like(tiles, dtype=torch.int16)


def test_encoder_keys_stages_by_label_and_emits_packed_shape():
    torch = pytest.importorskip("torch")
    if lora.torch is None:
        pytest.skip("fq_assemble_lora imported without torch")
    weight = torch.zeros(128, 128)
    result = lora.encode_expert_msrt(
        weight, 2,
        [{"k": 1, "label": "only", "experts": "all"}],
        torch.device("cpu"), identity_hadamard,
        identity_permutation, identity_permutation, identity_quantizer,
        1.2437, FakeExtension(),
    )
    assert set(result["stages"]) == {"only"}
    assert result["base"]["trellis"].shape == (8, 8, 32)
    assert result["stages"]["only"]["trellis"].shape == (8, 8, 16)
    assert result["mse"] == 0.0


def test_packed_quantization_refuses_missing_runtime_packer():
    torch = pytest.importorskip("torch")
    if lora.torch is None:
        pytest.skip("fq_assemble_lora imported without torch")
    with pytest.raises(RuntimeError, match="pack_trellis"):
        lora.quantize_trellis_packed(
            torch.zeros(128, 128), 2, torch.device("cpu"),
            identity_permutation, identity_permutation, identity_quantizer, None)


def test_base_metadata_is_regenerated_from_written_shards(tmp_path: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    if lora.torch is None:
        pytest.skip("fq_assemble_lora imported without torch")
    from safetensors.torch import save_file

    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    save_file({"kept": torch.ones(1)}, str(base / "model-layer-002.safetensors"))
    save_file({"quant": torch.ones(1)}, str(base / "model-layer-003.safetensors"))
    (base / "MANIFEST.sha256").write_text("stale")
    lora.write_base_metadata(base, 2, [3], {3: 1})

    config = json.loads((base / "config.json").read_text())
    index = json.loads((base / "model.safetensors.index.json").read_text())
    manifest = (base / "MANIFEST.sha256").read_text()
    assert config["quantization_config"]["quant_method"] == "exl3"
    assert config["hybrid_tr3_tail"]["bits"] == 2.0
    assert index["weight_map"] == {
        "kept": "model-layer-002.safetensors",
        "quant": "model-layer-003.safetensors",
    }
    assert "stale" not in manifest
    assert "model-layer-002.safetensors" in manifest
