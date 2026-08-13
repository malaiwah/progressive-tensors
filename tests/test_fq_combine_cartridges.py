"""Validation tests for fq_combine_cartridges."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import fq_assemble_lora as lora
import fq_combine_cartridges as combine


class TensorShape:
    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape)


def stage_tensors(label="res1", k=1):
    prefix = "model.layers.3.mlp.experts.0"
    tensors = {}
    for projection in lora.PROJECTIONS:
        root = f"{prefix}.{projection}.rank0"
        tensors[f"{root}.trellis_{label}"] = TensorShape((8, 8, k * 16))
        tensors[f"{root}.suh_{label}"] = TensorShape((128,))
        tensors[f"{root}.svh_{label}"] = TensorShape((128,))
        tensors[f"{root}.scale_{label}"] = TensorShape(())
    return tensors


def test_stage_tensor_validation_requires_complete_components_and_k_geometry():
    stage = {"label": "res1", "k": 1, "experts": [0]}
    combine.validate_stage_tensors(stage_tensors(), stage, "stage.safetensors")

    missing = stage_tensors()
    missing.pop(next(key for key in missing if ".scale_res1" in key))
    with pytest.raises(lora.CartridgeError, match="components"):
        combine.validate_stage_tensors(missing, stage, "stage.safetensors")

    wrong_k = stage_tensors(k=2)
    with pytest.raises(lora.CartridgeError, match="expected last dimension 16"):
        combine.validate_stage_tensors(wrong_k, stage, "stage.safetensors")


def write_config(path: Path, stage: dict, **overrides):
    config = {
        "schema": "fq-cartridge-adapter/1",
        "format": "exl3-msrt-full-rank",
        "standard_lora_compatible": False,
        "base_k": 2,
        "base_manifest_sha256": "a" * 64,
        "stages": [stage],
        "shards": [f"{stage['label']}/model-layer-003.safetensors"],
    }
    config.update(overrides)
    path.write_text(json.dumps(config))


def test_stage_config_rejects_path_traversal_and_base_mismatch(tmp_path: Path):
    stage = {"label": "res1", "k": 1, "experts": [0]}
    path = tmp_path / "adapter_config.json"
    write_config(path, stage)
    _, identity = combine.load_stage_config(path, stage, None)
    assert identity == (2, "a" * 64)

    write_config(path, stage, shards=["../escape.safetensors"])
    with pytest.raises(lora.CartridgeError, match="safe relative paths"):
        combine.load_stage_config(path, stage, None)

    write_config(path, stage, base_manifest_sha256="b" * 64)
    with pytest.raises(lora.CartridgeError, match="base checkpoint identity"):
        combine.load_stage_config(path, stage, (2, "a" * 64))
