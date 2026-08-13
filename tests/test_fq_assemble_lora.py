#!/usr/bin/env python3
"""Behavioral tests for the MSRT cartridge encoder."""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import fq_assemble_lora as lora

RECIPES = Path(__file__).parent.parent / "recipes"
GLM_RECIPE = RECIPES / "glm52-k2k3-dag.json"
FRUIT_RECIPE = RECIPES / "fruit-k2-k3k4-cart.json"


# ── Recipes ───────────────────────────────────────────────────────────────

def write_recipe(path: Path, **overrides) -> Path:
    recipe = {
        "schema": "fq-cartridge/2",
        "bases": [{"label": "k2", "k": 2}],
        "stages": [{"label": "res1", "k": 1, "parent": "k2", "experts": "all"}],
        "assemblies": [{"label": "k3like", "base": "k2", "chain": ["res1"]}],
        "moe_layers": [3],
    }
    recipe.update(overrides)
    path.write_text(json.dumps(recipe))
    return path


def test_requested_glm_graph_encodes_nine_products_in_nine_passes():
    recipe = lora.load_recipe(GLM_RECIPE)
    assert len(recipe["moe_layers"]) == 76, "GLM-5.2 routes experts in layers 3-78"
    assert len(lora.node_index(recipe)) == 9
    bitrates = {
        assembly["label"]: lora.assembly_bpw(recipe, assembly, 256)
        for assembly in recipe["assemblies"]
    }
    assert bitrates == {
        "k2": 2.0, "k2-k3like": 3.0, "k2-k4like-stepped": 4.0,
        "k2-k4like-direct": 4.0, "k2-k5like": 5.0, "k3": 3.0,
        "k3-k4like": 4.0, "k3-k5like-stepped": 5.0, "k3-k5like-direct": 5.0,
    }
    # Nine products spanning 35 bits would cost 35 passes encoded separately;
    # sharing parents emits 14 bits in 9 passes.
    assert lora.encoded_bits_per_weight(recipe, 256) == 14.0
    assert sum(bitrates.values()) == 35.0


def test_partial_stage_reports_its_real_mixed_bitrate():
    recipe = lora.load_recipe(FRUIT_RECIPE)
    proxy = next(a for a in recipe["assemblies"] if a["label"] == "siq-proxy")
    assert lora.assembly_bpw(recipe, proxy, 256) == pytest.approx(3.375)
    assert lora.encoded_bits_per_weight(recipe, 256) == pytest.approx(3.375)


@pytest.mark.parametrize("overrides,message", [
    ({"schema": "fq-cartridge/1"}, "schema must be"),
    ({"bases": []}, "non-empty"),
    ({"bases": [{"label": "k2", "k": 9}]}, "1..6"),
    ({"bases": [{"label": "k2", "k": 2, "experts": [0]}]}, "bases cover all"),
    ({"bases": [{"label": "k2", "k": 2}, {"label": "k2", "k": 3}]}, "duplicate"),
    ({"stages": [{"label": "res1", "k": 1, "parent": "nope", "experts": "all"}]},
     "is not a base or an earlier stage"),
    ({"stages": [
        {"label": "a", "k": 1, "parent": "b", "experts": "all"},
        {"label": "b", "k": 1, "parent": "k2", "experts": "all"}]},
     "parent-first"),
    ({"stages": [
        {"label": "res1", "k": 1, "parent": "k2", "experts": [0, 1]},
        {"label": "res2", "k": 1, "parent": "res1", "experts": [1, 2]}]},
     "subset of parent"),
    ({"stages": [{"label": "res1", "k": 1, "parent": "k2", "experts": [0, 0]}]},
     "unique non-negative"),
    ({"moe_layers": []}, "moe_layers"),
    ({"moe_layers": [1, 1]}, "moe_layers"),
    ({"assemblies": [{"label": "x", "base": "nope", "chain": []}]},
     "is not declared"),
    ({"assemblies": [{"label": "x", "base": "k2", "chain": ["ghost"]}]},
     "declared stage labels"),
])
def test_recipe_validation_rejects_unsafe_or_ambiguous_graphs(
    tmp_path: Path, overrides: dict, message: str
):
    path = write_recipe(tmp_path / "recipe.json", **overrides)
    with pytest.raises((lora.CartridgeError, ValueError), match=message):
        lora.load_recipe(path)


def test_assembly_chain_must_be_one_path_through_the_graph(tmp_path: Path):
    path = write_recipe(
        tmp_path / "recipe.json",
        stages=[
            {"label": "a", "k": 1, "parent": "k2", "experts": "all"},
            {"label": "b", "k": 1, "parent": "k2", "experts": "all"},
        ],
        assemblies=[{"label": "both", "base": "k2", "chain": ["a", "b"]}])
    with pytest.raises(lora.CartridgeError, match="one path"):
        lora.load_recipe(path)


def test_stage_selection_is_per_expert():
    stages = [
        {"label": "hot", "k": 1, "parent": "k2", "experts": [0, 2]},
        {"label": "all", "k": 1, "parent": "k2", "experts": "all"},
    ]
    assert [s["label"] for s in lora.stages_for_expert(stages, 0)] == ["hot", "all"]
    assert [s["label"] for s in lora.stages_for_expert(stages, 1)] == ["all"]


# ── Work partitioning ─────────────────────────────────────────────────────

def test_expert_blocks_are_stable_and_cover_every_expert():
    blocks = lora.expert_blocks([5, 1, 4, 3, 2, 0], 4)
    assert blocks == [[0, 1, 2, 3], [4, 5]]
    assert lora.block_name(3, 7) == "model-layer-003-b007.safetensors"


def test_blocks_only_emit_nodes_that_cover_one_of_their_experts():
    recipe = lora.load_recipe(FRUIT_RECIPE)
    assert set(lora.block_outputs(recipe, [0, 1])) == {"k2", "res1", "res2"}
    # res2 covers experts 0-95 only, so a block of later experts skips it.
    assert set(lora.block_outputs(recipe, [200, 201])) == {"k2", "res1"}


def test_layer_selection_rejects_layers_outside_the_recipe():
    recipe = lora.load_recipe(FRUIT_RECIPE)
    assert lora.selected_layers(recipe, "3-5,13") == [3, 4, 5, 13]
    assert lora.selected_layers(recipe, None) == recipe["moe_layers"]
    with pytest.raises(lora.CartridgeError, match="non-recipe layers"):
        lora.selected_layers(recipe, "3,99")


# ── Source checkpoint ─────────────────────────────────────────────────────

def expert_key(layer: int, expert: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"


def complete_keys(layer: int = 3, expert: int = 0) -> list[str]:
    return [expert_key(layer, expert, p) for p in lora.PROJECTIONS]


def test_missing_projection_fails_before_encoding():
    keys = complete_keys()
    keys.remove(expert_key(3, 0, "gate_proj"))
    with pytest.raises(lora.CartridgeError, match="projections"):
        lora.inspect_source_layer(keys, 3)


def test_foreign_layer_in_per_layer_shard_is_rejected():
    keys = complete_keys(3, 0) + complete_keys(4, 0)
    with pytest.raises(lora.CartridgeError, match="also contains"):
        lora.inspect_source_layer(keys, 3)


def build_source(root: Path, layers=(3, 4), experts=4, size=128, extra=True):
    """A minimal per-layer BF16 checkpoint the encoder accepts."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({
        "num_hidden_layers": max(layers) + 1,
        "n_routed_experts": experts,
        "hidden_size": size,
        "moe_intermediate_size": size,
    }))
    (root / "tokenizer_config.json").write_text("{}")
    generator = torch.Generator().manual_seed(7)
    for layer in layers:
        tensors = {}
        for expert in range(experts):
            for projection in lora.PROJECTIONS:
                tensors[expert_key(layer, expert, projection)] = torch.randn(
                    size, size, generator=generator).to(torch.bfloat16)
        if extra:
            tensors[f"model.layers.{layer}.mlp.gate.weight"] = torch.randn(
                experts, size, generator=generator).to(torch.bfloat16)
            tensors[f"model.layers.{layer}.input_layernorm.weight"] = torch.ones(
                size, dtype=torch.bfloat16)
        save_file(tensors, str(root / f"model-layer-{layer:03d}.safetensors"))
    return root


def test_source_reader_resolves_per_layer_experts(tmp_path: Path):
    root = build_source(tmp_path / "src")
    with lora.SourceCheckpoint(root) as source:
        assert source.layout == lora.SourceCheckpoint.PER_LAYER
        assert sorted(source.experts(3)) == [0, 1, 2, 3]
        assert source.experts(3)[2]["up_proj"] == expert_key(3, 2, "up_proj")


def test_source_reader_resolves_indexed_shards(tmp_path: Path):
    root = build_source(tmp_path / "src")
    weight_map = {}
    for layer in (3, 4):
        shard = f"model-{layer:05d}-of-00002.safetensors"
        (root / f"model-layer-{layer:03d}.safetensors").rename(root / shard)
        for expert in range(4):
            for projection in lora.PROJECTIONS:
                weight_map[expert_key(layer, expert, projection)] = shard
        weight_map[f"model.layers.{layer}.mlp.gate.weight"] = shard
        weight_map[f"model.layers.{layer}.input_layernorm.weight"] = shard
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}))
    with lora.SourceCheckpoint(root) as source:
        assert source.layout == lora.SourceCheckpoint.INDEXED
        assert sorted(source.experts(4)) == [0, 1, 2, 3]
        assert source.absent_shards == []


def test_skeleton_keeps_router_and_norms_and_drops_quantized_experts(tmp_path: Path):
    root = build_source(tmp_path / "src", layers=(3, 4))
    recipe = lora.load_recipe(write_recipe(tmp_path / "r.json", moe_layers=[3]))
    with lora.SourceCheckpoint(root) as source:
        grouped = lora.skeleton_keys(source, recipe["moe_layers"])
    kept = {key for keys in grouped.values() for key in keys}
    assert expert_key(3, 0, "gate_proj") not in kept, "layer 3 is quantized"
    assert expert_key(4, 0, "gate_proj") in kept, "layer 4 is outside the recipe"
    assert "model.layers.3.mlp.gate.weight" in kept
    assert "model.layers.3.input_layernorm.weight" in kept


# ── Quantization primitives ───────────────────────────────────────────────

def identity_hadamard(size, device, dtype, scale):
    torch = pytest.importorskip("torch")
    return torch.eye(size, device=device, dtype=dtype)


def test_zero_weight_serialized_scales_are_finite_and_round_trip():
    torch = pytest.importorskip("torch")
    device = torch.device("cpu")
    weight = torch.zeros(lora.HADAMARD_BLOCK, lora.HADAMARD_BLOCK)
    regularized, suh, svh = lora.regularize_with_vectors(
        weight, device, identity_hadamard, 1.24371088)
    assert torch.isfinite(suh).all() and (suh != 0).all()
    assert torch.isfinite(svh).all() and (svh != 0).all()
    restored = lora.inverse_regularize(
        regularized, suh, svh, device, identity_hadamard)
    assert torch.equal(restored, weight)


def test_zero_residual_stage_still_ships_a_real_trellis():
    """A fabricated all-zero trellis would decode to a nonzero codebook value."""
    torch = pytest.importorskip("torch")
    enc = fake_encoder()
    zero = torch.zeros(128, 128)
    recon, packed, scale = lora.rescaled_trellis_quantize(zero, zero, 2, enc)
    assert scale == pytest.approx(abs(enc.cbs) / lora.MIN_RESIDUAL_RMS)
    assert packed.shape == (8, 8, 32)
    assert torch.isfinite(recon).all()


@pytest.mark.parametrize("bits", [0, 7, 8, True, 2.0])
def test_runtime_unsupported_bitrates_are_refused(bits):
    with pytest.raises(ValueError, match="1..6"):
        lora._validate_k(bits)


class FakeExtension:
    """Records pack_trellis calls; the real packer needs the CUDA extension."""

    def __init__(self):
        self.calls = []

    def pack_trellis(self, packed, raw, bits):
        self.calls.append((tuple(packed.shape), tuple(raw.shape), bits))
        packed.zero_()


def identity_permutation(device):
    torch = pytest.importorskip("torch")
    return torch.arange(256, device=device)


def rounding_quantizer(tiles, options):
    """Lossy but deterministic stand-in for the Viterbi trellis encoder."""
    torch = pytest.importorskip("torch")
    step = 2.0 ** -options["K"]
    return (tiles / step).round() * step, torch.zeros_like(
        tiles, dtype=torch.int16)


FAKE_IDENTITY = {
    "encoder": "exllamav3",
    "encoder_bundle": "/fake/exllamav3",
    "encoder_sha256": "e" * 64,
    "encoder_files": {"ext.py": "f" * 64},
}


def fake_encoder(ext=None) -> lora.Encoder:
    return lora.Encoder(
        ext=ext or FakeExtension(), ghd=identity_hadamard,
        tcp=identity_permutation, tcpi=identity_permutation,
        qtf=rounding_quantizer, cbs=1.24371088, identity=FAKE_IDENTITY)


def test_dag_encoder_emits_one_node_per_graph_node_and_shares_parents():
    torch = pytest.importorskip("torch")
    ext = FakeExtension()
    enc = fake_encoder(ext)
    weight = torch.randn(128, 256, generator=torch.Generator().manual_seed(3))
    bases = [{"label": "k2", "k": 2}, {"label": "k3", "k": 3}]
    stages = [
        {"label": "k2r1", "k": 1, "parent": "k2", "experts": "all"},
        {"label": "k2r1r1", "k": 1, "parent": "k2r1", "experts": "all"},
        {"label": "k2r2", "k": 2, "parent": "k2", "experts": "all"},
    ]
    nodes = lora.encode_matrix_dag(
        weight, bases, stages, torch.device("cpu"), enc)
    assert set(nodes) == {"k2", "k3", "k2r1", "k2r1r1", "k2r2"}
    assert len(ext.calls) == 5, "one quantization pass per node, not per product"
    # Every stage improves on the parent it names, and a deeper chain is better
    # than its own prefix.
    assert nodes["k2r1"]["regularized_mse"] < nodes["k2"]["regularized_mse"]
    assert nodes["k2r1r1"]["regularized_mse"] < nodes["k2r1"]["regularized_mse"]
    assert nodes["k2r2"]["regularized_mse"] < nodes["k2"]["regularized_mse"]
    assert nodes["k3"]["regularized_mse"] < nodes["k2"]["regularized_mse"]
    for label, node in nodes.items():
        assert node["trellis"].shape == (8, 16, 16 * (
            {"k2": 2, "k3": 3, "k2r1": 1, "k2r1r1": 1, "k2r2": 2}[label]))
        assert (node["scale"] is None) == (label in {"k2", "k3"})


def test_dag_encoder_refuses_a_stage_whose_parent_is_absent():
    torch = pytest.importorskip("torch")
    with pytest.raises(lora.CartridgeError, match="was not encoded"):
        lora.encode_matrix_dag(
            torch.zeros(128, 128), [{"label": "k2", "k": 2}],
            [{"label": "orphan", "k": 1, "parent": "ghost", "experts": "all"}],
            torch.device("cpu"), fake_encoder())


def test_emitted_tensor_names_are_the_runtime_key_contract():
    """The runtime finds weights by name; the key set is the contract.

    `Exl3LoraCartridge` matches `...rank<N>.trellis_<label>` plus
    `suh_`/`svh_`/`scale_` companions, and the base tier is the unsuffixed
    quartet. Dropping one component leaves a shard that still validates as
    safetensors and still decodes trellis indices, but reconstructs nothing.
    """
    torch = pytest.importorskip("torch")
    node = {"trellis": torch.zeros(8, 8, 16, dtype=torch.int16),
            "suh": torch.ones(128, dtype=torch.float16),
            "svh": torch.ones(128, dtype=torch.float16),
            "scale": 2.0}
    prefix = "model.layers.3.mlp.experts.7.gate_proj.rank0"
    assert set(lora.base_tensor_names(3, 7, "gate_proj", node)) == {
        f"{prefix}.trellis", f"{prefix}.suh", f"{prefix}.svh", f"{prefix}.mcg"}
    stage = lora.stage_tensor_names(3, 7, "gate_proj", "res1", node)
    assert set(stage) == {
        f"{prefix}.trellis_res1", f"{prefix}.suh_res1",
        f"{prefix}.svh_res1", f"{prefix}.scale_res1"}
    assert stage[f"{prefix}.scale_res1"].dtype is torch.float32

def test_packed_quantization_refuses_missing_runtime_packer():
    torch = pytest.importorskip("torch")
    enc = fake_encoder()._replace(ext=None)
    with pytest.raises(RuntimeError, match="pack_trellis"):
        lora.quantize_trellis_packed(torch.zeros(128, 128), 2, enc)


# ── Campaign output discipline ────────────────────────────────────────────

def test_committed_shard_is_resumable_and_recipe_drift_is_fatal(tmp_path: Path):
    torch = pytest.importorskip("torch")
    directory = tmp_path / "campaign" / "stages" / "res1"
    expect = {"schema": lora.BLOCK_SCHEMA, "recipe_sha256": "a" * 64,
              "label": "res1", "layer": "3", "block": "0", "experts": "0,1"}
    groups = [("0", [("model.layers.3.mlp.experts.0.gate_proj.rank0.scale_res1",
                      torch.ones(()))])]
    assert not lora.shard_is_complete(directory, "s.safetensors", expect)
    lora.write_shard(groups, directory, "s.safetensors", dict(expect))
    assert lora.shard_is_complete(directory, "s.safetensors", expect)
    assert lora.read_digest(directory, "s.safetensors")
    with pytest.raises(lora.CartridgeError, match="different recipe"):
        lora.shard_is_complete(directory, "s.safetensors",
                               {**expect, "recipe_sha256": "b" * 64})


def test_written_shard_is_readable_and_expert_contiguous(tmp_path: Path):
    """Our own serializer must stay a valid safetensors file."""
    torch = pytest.importorskip("torch")
    safe_open = pytest.importorskip("safetensors").safe_open
    directory = tmp_path / "node"
    groups = []
    for expert in (0, 1):
        prefix = f"model.layers.3.mlp.experts.{expert}.gate_proj.rank0"
        groups.append((str(expert), [
            (f"{prefix}.trellis", torch.ones(2, 2, 32, dtype=torch.int16)),
            (f"{prefix}.suh", torch.ones(128, dtype=torch.float16)),
            (f"{prefix}.svh", torch.ones(128, dtype=torch.float16)),
            (f"{prefix}.mcg", torch.tensor(7, dtype=torch.int32)),
        ]))
    target, sha, body, spans, digests = lora.write_grouped_shard(
        groups, directory, "s.safetensors", {"schema": lora.BLOCK_SCHEMA})
    with safe_open(str(target), framework="pt") as handle:
        assert len(list(handle.keys())) == 8
        assert handle.metadata()["schema"] == lora.BLOCK_SCHEMA
        assert handle.get_tensor(
            "model.layers.3.mlp.experts.1.gate_proj.rank0.mcg").item() == 7
    # Each expert is one byte range, and re-reading the file reproduces every
    # digest the attestation would have claimed.
    assert spans["0"][1] == spans["1"][0]
    again_sha, again_body, again_digests = lora.verify_shard(
        target, group_kind="expert")
    assert (again_sha, again_body, again_digests) == (sha, body, digests)


def test_uncommitted_shard_without_a_digest_is_re_encoded(tmp_path: Path):
    torch = pytest.importorskip("torch")
    directory = tmp_path / "campaign" / "stages" / "res1"
    expect = {"schema": lora.BLOCK_SCHEMA}
    lora.save_shard_tensors({"x": torch.ones(2)}, directory, "s.safetensors",
                            dict(expect))
    assert (directory / "s.safetensors").is_file()
    assert not lora.shard_is_complete(directory, "s.safetensors", expect)


def test_campaign_directory_refuses_a_different_graph_or_source(tmp_path: Path):
    recipe = write_recipe(tmp_path / "r.json")
    out = tmp_path / "out"
    identity = {"base_model": "acme/proxy", "base_revision": SOURCE_REVISION}
    with lora.SourceCheckpoint(build_source(tmp_path / "src")) as source:
        lora.prepare_out_dir(out, recipe_path=recipe, recipe_sha="a" * 64,
                             source=source, block_size=32, force=False,
                             **identity)
        recorded = json.loads((out / lora.SENTINEL).read_text())
        assert recorded["base_revision"] == SOURCE_REVISION
        assert recorded["topology_id"] == source.topology_id

        # Re-attaching with the same identity is the normal resume path.
        lora.prepare_out_dir(out, recipe_path=recipe, recipe_sha="a" * 64,
                             source=source, block_size=32, force=False,
                             **identity)

        # Identity drift is fatal with or without --force: converting a
        # campaign in place would leave the previous layout's shards behind.
        for force in (False, True):
            for drifted in ({"recipe_sha": "b" * 64},
                            {"block_size": 64},
                            {"base_revision": "c" * 40}):
                kwargs = {"recipe_sha": "a" * 64, "block_size": 32, **identity}
                kwargs.update(drifted)
                with pytest.raises(lora.CartridgeError,
                                   match="does not match this run"):
                    lora.prepare_out_dir(
                        out, recipe_path=recipe, source=source, force=force,
                        **kwargs)




def test_plan_names_the_shards_no_layer_window_would_stage(
    tmp_path: Path, capsys
):
    """Staging from `layers` alone misses whole-model tensors.

    Embeddings, `lm_head` and the dense layers live in shards that hold no
    routed expert, so they appear in no layer's shard list. An operator who
    staged strictly from the per-layer plan would only find out when `finalize`
    refused to publish a base, a whole campaign later.
    """
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    source = tmp_path / "src"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({
        "num_hidden_layers": 5, "n_routed_experts": 2,
        "hidden_size": 8, "moe_intermediate_size": 8,
    }))
    weight_map, generator = {}, torch.Generator().manual_seed(3)
    for index, layer in enumerate((3, 4), start=1):
        shard = f"model-{index:05d}-of-00003.safetensors"
        tensors = {}
        for expert in range(2):
            for projection in lora.PROJECTIONS:
                key = expert_key(layer, expert, projection)
                tensors[key] = torch.randn(
                    8, 8, generator=generator).to(torch.bfloat16)
        save_file(tensors, str(source / shard))
        weight_map.update({key: shard for key in tensors})
    orphan = "model-00003-of-00003.safetensors"
    whole_model = {
        "model.embed_tokens.weight": torch.ones(4, 8, dtype=torch.bfloat16),
        "lm_head.weight": torch.ones(4, 8, dtype=torch.bfloat16),
    }
    save_file(whole_model, str(source / orphan))
    weight_map.update({key: orphan for key in whole_model})
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}))

    recipe = write_recipe(tmp_path / "r.json", **DAG_RECIPE)
    assert lora.cmd_plan(SimpleNamespace(
        source=source, recipe=recipe, layers=None, block_size=2,
        out_plan=None)) == 0
    plan = json.loads(capsys.readouterr().out)

    assert orphan in plan["skeleton_only_shards"]
    staged = set(plan["skeleton_only_shards"])
    for entry in plan["layers"]:
        staged |= set(entry["shards"])
    # Every shard the skeleton pass needs is reachable from one plan.
    with lora.SourceCheckpoint(source) as opened:
        needed = set(lora.skeleton_keys(opened, [3, 4]))
    assert not needed - staged, f"skeleton needs unstaged shards: {needed - staged}"


def test_a_crash_between_two_node_commits_leaves_the_block_incomplete(
    tmp_path: Path, monkeypatch
):
    """A half-rewritten block must never look finished.

    A block is one consistency unit: a residual is only valid against the parent
    bytes published beside it. If markers were retracted per node as each was
    written, a crash after the base and before its stage would leave the base
    with a fresh marker and the stage with its old, still-valid one -- so the
    next run would skip the block and ship a residual built against a different
    reconstruction. Only finalize's parent-digest check would notice, a campaign
    later.
    """
    pytest.importorskip("torch")
    source = build_source(tmp_path / "src", layers=(3,), experts=2)
    recipe = write_recipe(tmp_path / "recipe.json", **DAG_RECIPE)
    out = tmp_path / "campaign"
    monkeypatch.setattr(lora, "bootstrap_encoder", _fake_bootstrap)
    assert lora.cmd_encode(encode_args(source, recipe, out, layers="3")) == 0

    name = lora.block_name(3, 0)
    labels = ("k2", "k2r1")
    recipe_doc = lora.load_recipe(recipe)
    before = {label: lora.read_digest(
        lora.node_dir(out, recipe_doc, label), name) for label in labels}
    assert all(before.values())

    # Crash the second write of a forced re-encode.
    real_write_shard, seen = lora.write_shard, []

    def crash_after_first(tensors, directory, shard, metadata, **kwargs):
        seen.append(directory.name)
        if len(seen) > 1:
            raise KeyboardInterrupt("preempted between two node commits")
        return real_write_shard(tensors, directory, shard, metadata, **kwargs)

    monkeypatch.setattr(lora, "write_shard", crash_after_first)
    with pytest.raises(KeyboardInterrupt):
        lora.cmd_encode(encode_args(source, recipe, out, layers="3", force=True))

    # The node that was being written when the process died must not carry a
    # marker, so the block cannot read as complete and cannot be skipped.
    monkeypatch.setattr(lora, "write_shard", real_write_shard)
    assert lora.read_digest(lora.node_dir(out, recipe_doc, labels[1]),
                            name) is None
    with lora.SourceCheckpoint(source) as opened:
        item = lora.build_work_list(opened, recipe_doc, [3], 2)[0]
        assert not lora.block_is_complete(
            out, recipe_doc, lora.sha256_file(recipe), opened,
            SOURCE_REVISION, item, 2)

    # A plain resume therefore rewrites the whole block, and the stage it writes
    # names the base digest that shipped beside it in the same pass.
    assert lora.cmd_encode(encode_args(source, recipe, out, layers="3")) == 0
    base_dir = lora.node_dir(out, recipe_doc, labels[0])
    for label in labels:
        assert lora.read_digest(lora.node_dir(out, recipe_doc, label), name)
    stage = lora.read_attestation(lora.node_dir(out, recipe_doc, labels[1]), name)
    assert stage["parents"][0]["sha256"] == lora.read_digest(base_dir, name)



def test_one_campaign_directory_takes_one_launcher(tmp_path: Path):
    """Per-block claims cannot police two launchers; the campaign lock does.

    A launcher clears leftover block claims before forking, which is only safe
    if no other launcher can be alive. Without an exclusive campaign claim, a
    second launcher would delete the first one's live claims and both fleets
    would quantize the same blocks -- and the first to finish a block would then
    unlink the second's replacement claim.
    """
    out = tmp_path / "campaign"
    with lora.campaign_lock(out, what="encode") as held:
        assert held is True
        with pytest.raises(lora.CartridgeError, match="takes one encode"):
            with lora.campaign_lock(out, what="encode"):
                pass
        # A worker spawned by the holder must not deadlock against its parent.
        os.environ[lora.CAMPAIGN_LOCK_ENV] = "1"
        try:
            with lora.campaign_lock(out, what="encode") as inherited:
                assert inherited is False
        finally:
            del os.environ[lora.CAMPAIGN_LOCK_ENV]
    # The kernel drops the flock when the holder exits, so a crashed campaign
    # never needs a manual unlock.
    with lora.campaign_lock(out, what="finalize") as after:
        assert after is True


def test_a_block_claim_is_ownership_not_a_leftover_file(tmp_path: Path):
    """A crashed worker must not park a block forever, or hand it out twice.

    Ownership is an flock, so the kernel frees it however the owner dies. That
    is what removes the need for a launcher to decide whether someone else's
    claim is stale -- the decision that let a second launcher delete a live one.
    """
    out = tmp_path / "campaign"
    with lora.block_claim(out, 3, 0) as owned:
        assert owned is True
        with lora.block_claim(out, 3, 0) as again:
            assert again is False          # a live owner keeps the block
        with lora.campaign_lock(out, what="encode"):
            with pytest.raises(lora.CartridgeError, match="takes one finalize"):
                with lora.campaign_lock(out, what="finalize"):
                    pass
    # The file survives on purpose: unlinking a path another process already
    # opened would give the same block to two owners.
    assert (out / "locks" / "layer-003-b000.lock").is_file()
    with lora.block_claim(out, 3, 0) as reclaimed:
        assert reclaimed is True           # released, so claimable again

def test_publication_and_encoding_are_mutually_exclusive(tmp_path: Path):
    """A launcher's lock dies with the launcher; its workers do not.

    `finalize` re-reads and signs the whole campaign, so it must not overlap a
    worker still rewriting blocks -- including one orphaned by a killed launcher,
    which no longer holds the campaign lock. Workers hold this file shared for
    their lifetime; finalize takes it exclusively.
    """
    out = tmp_path / "campaign"
    with lora.worker_presence(out):
        with pytest.raises(lora.CartridgeError, match="workers are still"):
            with lora.no_workers_running(out):
                pass


# ── End to end, on CPU, with a stand-in quantizer ─────────────────────────

@contextmanager
def _fake_bootstrap(_source, tile_batch=lora.TILE_BATCH):
    yield fake_encoder()._replace(tile_batch=tile_batch)


SOURCE_REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"


def encode_args(source: Path, recipe: Path, out: Path, **overrides):
    args = SimpleNamespace(
        source=source, recipe=recipe, out=out, block_size=2,
        encoder_source=Path("/nonexistent"), device="cpu", devices=None,
        layers=None, shard_index=0, shard_count=1, force=False,
        tile_batch=lora.TILE_BATCH, sign_key=source.parent / "sign.key",
        base_model="acme/proxy", base_revision=SOURCE_REVISION,
        source_digests=None)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


DAG_RECIPE = {
    "bases": [{"label": "k2", "k": 2}, {"label": "k3", "k": 3}],
    "stages": [
        {"label": "k2r1", "k": 1, "parent": "k2", "experts": "all"},
        {"label": "hot", "k": 1, "parent": "k2r1", "experts": [0, 1]},
    ],
    "assemblies": [
        {"label": "k3like", "base": "k2", "chain": ["k2r1"]},
        {"label": "hot-k4like", "base": "k2", "chain": ["k2r1", "hot"]},
    ],
    "moe_layers": [3, 4],
}


@pytest.fixture
def campaign(tmp_path: Path, monkeypatch):
    """A finalized four-block campaign encoded with the stand-in quantizer."""
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    source = build_source(tmp_path / "src", layers=(3, 4), experts=4)
    recipe = write_recipe(tmp_path / "recipe.json", **DAG_RECIPE)
    out = tmp_path / "campaign"
    monkeypatch.setattr(lora, "bootstrap_encoder", _fake_bootstrap)
    assert lora.cmd_skeleton(encode_args(source, recipe, out, force=False)) == 0
    assert lora.cmd_encode(encode_args(source, recipe, out)) == 0
    assert lora.cmd_finalize(encode_args(source, recipe, out)) == 0
    return SimpleNamespace(root=out, source=source, recipe=recipe)


def test_two_encodes_of_one_block_produce_identical_bytes(
    tmp_path: Path, monkeypatch
):
    """The signed reproducibility claim, tested as bytes.

    Every attestation declares a `determinism_scope` and asserts that
    re-encoding inside it reproduces the fragment. That is only true if nothing
    volatile reaches the hashed shard: a `created_utc` in the safetensors header
    would make two encodes differ whenever they cross a second boundary. The
    wall clock belongs in the attestation, which is not part of the fragment.
    """
    pytest.importorskip("torch")
    source = build_source(tmp_path / "src", layers=(3, 4), experts=2)
    recipe = write_recipe(tmp_path / "recipe.json", **DAG_RECIPE)
    monkeypatch.setattr(lora, "bootstrap_encoder", _fake_bootstrap)

    shards = []
    for run in ("a", "b"):
        out = tmp_path / run
        assert lora.cmd_encode(encode_args(source, recipe, out)) == 0
        shards.append({
            path.relative_to(out): path.read_bytes()
            for path in sorted(out.rglob("model-*.safetensors"))
        })
        time.sleep(1.1)  # cross a wall-clock second between the two encodes

    first, second = shards
    assert first and set(first) == set(second)
    differing = sorted(str(name) for name in first if first[name] != second[name])
    assert not differing, f"encode is not byte-reproducible: {differing}"
    # The digest sidecars must agree too, since they are what resume trusts.
    for name in first:
        directory = (tmp_path / "a" / name).parent
        assert (lora.read_digest(directory, name.name)
                == lora.read_digest((tmp_path / "b" / name).parent, name.name))


def test_finalize_is_resumable_after_it_linked_the_skeleton(
    campaign, monkeypatch
):
    """Finalize re-reads and re-hashes the whole campaign, so it can be cut off.

    On a real campaign that pass is terabytes long and minutes wide. Publishing
    a base means hardlinking every skeleton shard into it, so the second run
    sees files in `base/<label>/` that the recipe's block list does not name.
    Rejecting those would turn an interrupted finalize into a lost campaign.
    """
    monkeypatch.setattr(lora, "bootstrap_encoder", _fake_bootstrap)
    base = campaign.root / "base" / "k2"
    skeleton = {path.name for path
                in (campaign.root / "skeleton").glob("model-*.safetensors")}
    linked = skeleton & {path.name for path in base.glob("model-*.safetensors")}
    assert linked == skeleton, "finalize must publish the skeleton into the base"

    args = encode_args(campaign.source, campaign.recipe, campaign.root)
    assert lora.cmd_finalize(args) == 0
    assert lora.cmd_finalize(args) == 0
    # A shard from a different block layout is still refused.
    (base / lora.block_name(3, 9)).write_bytes(
        (base / lora.block_name(3, 0)).read_bytes())
    with pytest.raises(lora.CartridgeError, match="does not describe"):
        lora.cmd_finalize(args)


def test_campaign_encodes_finalizes_and_resumes(tmp_path: Path, monkeypatch):
    pytest.importorskip("torch")
    source = build_source(tmp_path / "src", layers=(3, 4), experts=4)
    recipe = write_recipe(
        tmp_path / "recipe.json",
        bases=[{"label": "k2", "k": 2}, {"label": "k3", "k": 3}],
        stages=[
            {"label": "k2r1", "k": 1, "parent": "k2", "experts": "all"},
            {"label": "hot", "k": 1, "parent": "k2r1", "experts": [0, 1]},
        ],
        assemblies=[
            {"label": "k3like", "base": "k2", "chain": ["k2r1"]},
            {"label": "hot-k4like", "base": "k2", "chain": ["k2r1", "hot"]},
        ],
        moe_layers=[3, 4])
    out = tmp_path / "campaign"
    monkeypatch.setattr(lora, "bootstrap_encoder", _fake_bootstrap)

    assert lora.cmd_skeleton(encode_args(source, recipe, out, force=False)) == 0
    assert lora.cmd_encode(encode_args(source, recipe, out)) == 0
    assert lora.cmd_finalize(encode_args(source, recipe, out)) == 0

    # Two complete base checkpoints, each with its own loader metadata.
    for label, bits in (("k2", 2.0), ("k3", 3.0)):
        base = out / "base" / label
        config = json.loads((base / "config.json").read_text())
        assert config["quantization_config"]["bits"] == bits
        assert config["hybrid_tr3_tail"]["moe_layers"] == [3, 4]
        index = json.loads((base / "model.safetensors.index.json").read_text())
        assert ("model.layers.3.mlp.experts.0.gate_proj.rank0.trellis"
                in index["weight_map"])
        assert "model.layers.3.mlp.gate.weight" in index["weight_map"], (
            "the router must be present in every base checkpoint")
        manifest = (base / "MANIFEST.sha256").read_text()
        assert "model-layer-003-b000.safetensors" in manifest
        assert json.loads((base / "tier_bitmap.json").read_text())["3"]["k"] == [
            int(bits)] * 4

    # The partial stage only wrote the blocks that hold its experts.
    hot_blocks = sorted(p.name for p in (out / "stages" / "hot").glob("*.safetensors"))
    assert hot_blocks == ["model-layer-003-b000.safetensors",
                          "model-layer-004-b000.safetensors"]
    plan = json.loads((out / "assemblies" / "hot-k4like" / "assembly.json").read_text())
    assert plan["schema"] == lora.ASSEMBLY_SCHEMA
    assert plan["bits_per_weight"] == pytest.approx(2 + 1 + 2 / 4)
    assert {entry["label"] for entry in plan["stage_shards"]} == {"k2r1", "hot"}

    summary = json.loads((out / "campaign_summary.json").read_text())
    assert summary["encoded_bits_per_weight"] == pytest.approx(2 + 3 + 1 + 0.5)

    # A second pass re-reads nothing and rewrites nothing.
    marker = out / "base" / "k2" / "model-layer-003-b000.safetensors"
    before = marker.stat().st_mtime_ns
    assert lora.cmd_encode(encode_args(source, recipe, out)) == 0
    assert marker.stat().st_mtime_ns == before


def test_finalize_names_the_blocks_that_are_still_missing(tmp_path: Path, monkeypatch):
    pytest.importorskip("torch")
    source = build_source(tmp_path / "src", layers=(3,), experts=4)
    recipe = write_recipe(tmp_path / "recipe.json", moe_layers=[3])
    out = tmp_path / "campaign"
    monkeypatch.setattr(lora, "bootstrap_encoder", _fake_bootstrap)
    lora.cmd_skeleton(encode_args(source, recipe, out, force=False))
    lora.cmd_encode(encode_args(source, recipe, out, shard_index=0, shard_count=2))
    with pytest.raises(lora.CartridgeError, match="outputs are missing"):
        lora.cmd_finalize(encode_args(source, recipe, out))
    lora.cmd_encode(encode_args(source, recipe, out, shard_index=1, shard_count=2))
    assert lora.cmd_finalize(encode_args(source, recipe, out)) == 0


def test_workers_split_disjoint_blocks(tmp_path: Path):
    source = build_source(tmp_path / "src", layers=(3, 4), experts=4)
    recipe = lora.load_recipe(write_recipe(tmp_path / "r.json", moe_layers=[3, 4]))
    with lora.SourceCheckpoint(source) as checkpoint:
        work = lora.build_work_list(checkpoint, recipe, [3, 4], 2)
    assert len(work) == 4
    shards = [work[i::2] for i in range(2)]
    assert not {(w["layer"], w["block"]) for w in shards[0]} & {
        (w["layer"], w["block"]) for w in shards[1]}
    assert sum(len(s) for s in shards) == len(work)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_resume_after_a_partial_block_rewrites_the_whole_graph(
    tmp_path: Path, monkeypatch
):
    """A residual is only valid against the parent bytes that ship with it.

    Retract one stage of a finished block, exactly as an interrupted write
    would leave it, and resume. Rewriting only the missing stage would leave it
    correcting a reconstruction that nobody re-derived, so the whole block --
    base included -- must be re-encoded as one unit.
    """
    pytest.importorskip("torch")
    source = build_source(tmp_path / "src", layers=(3,), experts=2)
    recipe = write_recipe(tmp_path / "recipe.json", moe_layers=[3])
    out = tmp_path / "campaign"
    monkeypatch.setattr(lora, "bootstrap_encoder", _fake_bootstrap)
    args = encode_args(source, recipe, out, block_size=2)
    assert lora.cmd_skeleton(args) == 0
    assert lora.cmd_encode(args) == 0

    name = "model-layer-003-b000.safetensors"
    base_shard = out / "base" / "k2" / name
    stage_shard = out / "stages" / "res1" / name
    before = (base_shard.stat().st_mtime_ns, stage_shard.stat().st_mtime_ns)
    lora.digest_path(stage_shard.parent, name).unlink()

    assert lora.cmd_encode(args) == 0
    after = (base_shard.stat().st_mtime_ns, stage_shard.stat().st_mtime_ns)
    assert after[0] != before[0], (
        "the base must be re-encoded with its stage, or the published pair "
        "could come from two different encoder runs")
    assert after[1] != before[1]
    # And the pair that ships is internally consistent: the stage names the
    # exact parent bytes beside it.
    assert lora.cmd_finalize(args) == 0
    payload = lora.read_attestation(stage_shard.parent, name)
    assert payload["parents"][0]["sha256"] == lora.read_digest(
        base_shard.parent, name)


def test_finalize_rejects_shards_from_another_block_layout(
    tmp_path: Path, monkeypatch
):
    pytest.importorskip("torch")
    source = build_source(tmp_path / "src", layers=(3,), experts=4)
    recipe = write_recipe(tmp_path / "recipe.json", moe_layers=[3])
    out = tmp_path / "campaign"
    monkeypatch.setattr(lora, "bootstrap_encoder", _fake_bootstrap)
    args = encode_args(source, recipe, out, block_size=2)
    lora.cmd_skeleton(args)
    lora.cmd_encode(args)
    # A shard a previous block layout would have left behind.
    stale = out / "stages" / "res1" / "model-layer-003-b009.safetensors"
    stale.write_bytes(
        (out / "stages" / "res1" / "model-layer-003-b000.safetensors").read_bytes())
    with pytest.raises(lora.CartridgeError, match="does not describe"):
        lora.cmd_finalize(args)


def test_finalize_rejects_a_block_copied_over_another(tmp_path: Path, monkeypatch):
    """Duplicating a block would publish one expert range twice."""
    pytest.importorskip("torch")
    source = build_source(tmp_path / "src", layers=(3,), experts=4)
    recipe = write_recipe(tmp_path / "recipe.json", moe_layers=[3])
    out = tmp_path / "campaign"
    monkeypatch.setattr(lora, "bootstrap_encoder", _fake_bootstrap)
    args = encode_args(source, recipe, out, block_size=2)
    lora.cmd_skeleton(args)
    lora.cmd_encode(args)
    node = out / "stages" / "res1"
    for suffix, folder in (("", node), (".sha256", node / "digests"),
                           (".jsonl", node / "attestations")):
        first = folder / f"model-layer-003-b000.safetensors{suffix}"
        second = folder / f"model-layer-003-b001.safetensors{suffix}"
        second.write_bytes(first.read_bytes())
    with pytest.raises(lora.CartridgeError, match="missing|failed verification"):
        lora.cmd_finalize(args)


def test_block_claim_admits_one_writer(tmp_path: Path):
    with lora.block_claim(tmp_path, 3, 0) as first:
        assert first
        with lora.block_claim(tmp_path, 3, 0) as second:
            assert not second
    with lora.block_claim(tmp_path, 3, 0) as again:
        assert again
