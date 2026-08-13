"""Validation tests for fq_combine_cartridges."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import fq_assemble_lora as lora
import fq_combine_cartridges as combine

pytest_plugins = ("test_fq_assemble_lora",)


def stage_tensors(label="res1", k=1, layer=3, experts=(0,)):
    torch = pytest.importorskip("torch")
    tensors = {}
    for expert in experts:
        prefix = f"model.layers.{layer}.mlp.experts.{expert}"
        for projection in lora.PROJECTIONS:
            root = f"{prefix}.{projection}.rank0"
            tensors[f"{root}.trellis_{label}"] = torch.zeros(
                8, 8, k * 16, dtype=torch.int16)
            tensors[f"{root}.suh_{label}"] = torch.ones(128, dtype=torch.float16)
            tensors[f"{root}.svh_{label}"] = torch.ones(128, dtype=torch.float16)
            tensors[f"{root}.scale_{label}"] = torch.tensor(
                2.0, dtype=torch.float32)
    return tensors


def test_stage_tensor_validation_requires_complete_components_and_k_geometry():
    stage = {"label": "res1", "k": 1, "experts": [0]}
    combine.validate_stage_tensors(
        stage_tensors(), stage, {0}, 3, "stage.safetensors")

    missing = stage_tensors()
    missing.pop(next(key for key in missing if ".scale_res1" in key))
    with pytest.raises(lora.CartridgeError, match="components"):
        combine.validate_stage_tensors(
            missing, stage, {0}, 3, "stage.safetensors")

    with pytest.raises(lora.CartridgeError, match="last dimension is 16"):
        combine.validate_stage_tensors(
            stage_tensors(k=2), stage, {0}, 3, "stage.safetensors")

    with pytest.raises(lora.CartridgeError, match="carries label"):
        combine.validate_stage_tensors(
            stage_tensors(label="other"), stage, {0}, 3, "stage.safetensors")

    with pytest.raises(lora.CartridgeError, match="belongs to layer"):
        combine.validate_stage_tensors(
            stage_tensors(layer=4), stage, {0}, 3, "stage.safetensors")

    with pytest.raises(lora.CartridgeError, match="!= selected"):
        combine.validate_stage_tensors(
            stage_tensors(experts=(0, 1)), stage, {0}, 3, "stage.safetensors")


@pytest.mark.parametrize("value,expected", [
    ("0-3", {0, 1, 2, 3}),
    ("5", {5}),
    ("0-2,7", {0, 1, 2, 7}),
])
def test_id_selection_parses_ranges(value, expected):
    assert combine.parse_ids(value) == expected


@pytest.mark.parametrize("value", ["3-1", "-4", "", "x"])
def test_id_selection_rejects_nonsense(value):
    with pytest.raises((lora.CartridgeError, ValueError)):
        combine.parse_ids(value)


def mutate_plan(root: Path, label: str, **overrides) -> None:
    path = root / "assemblies" / label / "assembly.json"
    plan = json.loads(path.read_text())
    plan.update(overrides)
    path.write_text(json.dumps(plan))


def signer_of(campaign) -> str:
    summary = json.loads((campaign.root / "campaign_summary.json").read_text())
    return summary["provenance"]["signer_pubkey"]


@pytest.mark.parametrize("overrides,message", [
    ({"schema": "fq-cartridge-adapter/1"}, "schema must be"),
    ({"format": "peft-lora"}, "unsupported format"),
    ({"standard_lora_compatible": True}, "must be false"),
    ({"base": {"label": "k2", "k": 2, "manifest_sha256": "short"}},
     "invalid base checkpoint identity"),
    ({"chain": [{"label": "hot", "k": 1, "parent": "k2r1", "experts": [0, 1]}]},
     "not one path"),
])
def test_assembly_plan_validation_rejects_broken_plans(
    campaign, overrides, message
):
    mutate_plan(campaign.root, "hot-k4like", **overrides)
    with pytest.raises(lora.CartridgeError, match=message):
        combine.load_assembly(campaign.root, "hot-k4like", trust=None)


def test_assembly_plan_rejects_path_traversal(campaign):
    plan = json.loads(
        (campaign.root / "assemblies" / "k3like" / "assembly.json").read_text())
    plan["stage_shards"][0]["path"] = "../../etc/passwd.safetensors"
    mutate_plan(campaign.root, "k3like", stage_shards=plan["stage_shards"])
    with pytest.raises(lora.CartridgeError, match="campaign-relative"):
        combine.load_assembly(campaign.root, "k3like", trust=None)


def combine_args(root: Path, assembly: str, out: Path, **overrides):
    args = SimpleNamespace(root=root, assembly=assembly, out=out,
                           experts=None, layers=None, force=False,
                           trust_key=None, insecure_unsigned=True,
                           base=None)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def adapter_of(out: Path) -> dict:
    return json.loads((out / "adapter_config.json").read_text())


def test_combiner_requires_an_explicit_trust_decision(campaign, tmp_path: Path):
    with pytest.raises(lora.CartridgeError, match="--trust-key"):
        combine.combine(combine_args(
            campaign.root, "k3like", tmp_path / "a", insecure_unsigned=False))


def test_combiner_verifies_the_signed_plan_and_every_fragment(
    campaign, tmp_path: Path
):
    key = signer_of(campaign)
    out = tmp_path / "trusted"
    assert combine.combine(combine_args(
        campaign.root, "hot-k4like", out,
        trust_key=key, insecure_unsigned=False)) == 0
    assert adapter_of(out)["verified_signer"] == key


def test_combiner_rejects_another_signer(campaign, tmp_path: Path):
    with pytest.raises(lora.CartridgeError, match="not the pinned"):
        combine.combine(combine_args(
            campaign.root, "k3like", tmp_path / "b",
            trust_key="c" * 64, insecure_unsigned=False))


def test_combiner_detects_a_tampered_signed_plan(campaign, tmp_path: Path):
    signed = campaign.root / "assemblies" / "k3like" / "assembly.jsonl"
    envelope = json.loads(signed.read_text())
    payload = json.loads(base64.b64decode(envelope["payload"]))
    payload["mcg_multiplier"] = 1
    envelope["payload"] = base64.b64encode(
        json.dumps(payload).encode()).decode()
    signed.write_text(json.dumps(envelope) + "\n")
    with pytest.raises(lora.CartridgeError, match="unusable|signature"):
        combine.combine(combine_args(
            campaign.root, "k3like", tmp_path / "c",
            trust_key=signer_of(campaign), insecure_unsigned=False))


def test_combiner_refuses_an_altered_runtime_contract(campaign, tmp_path: Path):
    mutate_plan(campaign.root, "k3like", mcg_multiplier=1)
    with pytest.raises(lora.CartridgeError, match="mcg_multiplier"):
        combine.load_assembly(campaign.root, "k3like", trust=None)


def test_combiner_refuses_a_base_only_product(campaign, tmp_path: Path):
    mutate_plan(campaign.root, "k3like", chain=[])
    with pytest.raises(lora.CartridgeError, match="not a cartridge"):
        combine.load_assembly(campaign.root, "k3like", trust=None)


def test_combiner_refuses_duplicate_or_foreign_shard_entries(campaign):
    plan = json.loads(
        (campaign.root / "assemblies" / "k3like" / "assembly.json").read_text())
    doubled = plan["stage_shards"] + [plan["stage_shards"][0]]
    mutate_plan(campaign.root, "k3like", stage_shards=doubled)
    with pytest.raises(lora.CartridgeError, match="listed twice"):
        combine.load_assembly(campaign.root, "k3like", trust=None)

    foreign = [dict(plan["stage_shards"][0],
                    path="stages/other/model-layer-003-b000.safetensors")]
    mutate_plan(campaign.root, "k3like", stage_shards=foreign)
    with pytest.raises(lora.CartridgeError, match="does not live under"):
        combine.load_assembly(campaign.root, "k3like", trust=None)


def test_combiner_refuses_experts_the_assembly_does_not_cover(
    campaign, tmp_path: Path
):
    with pytest.raises(lora.CartridgeError, match="not covered by assembly"):
        combine.combine(combine_args(
            campaign.root, "hot-k4like", tmp_path / "d", experts="1,99"))


def test_tiered_assembly_states_which_experts_reach_which_stage(
    campaign, tmp_path: Path
):
    """A mixed product covers different experts at different depths."""
    out = tmp_path / "tiered"
    assert combine.combine(combine_args(
        campaign.root, "hot-k4like", out)) == 0
    config = adapter_of(out)
    assert config["coverage"]["k2r1"] == {"3": [0, 1, 2, 3], "4": [0, 1, 2, 3]}
    assert config["coverage"]["hot"] == {"3": [0, 1], "4": [0, 1]}


def test_combiner_rejects_a_stage_from_another_campaign(
    campaign, tmp_path: Path, monkeypatch
):
    """Correct shape, valid signature, wrong reconstruction.

    A stage encoded against a different base is the one failure that every
    digest and signature check can miss, so the chain edge is what catches it.
    """
    import test_fq_assemble_lora as helpers

    other = tmp_path / "other"
    source = helpers.build_source(tmp_path / "src2", layers=(3, 4), experts=4)
    recipe = helpers.write_recipe(tmp_path / "other.json", **helpers.DAG_RECIPE)
    monkeypatch.setattr(lora, "bootstrap_encoder", helpers._fake_bootstrap)
    args = helpers.encode_args(source, recipe, other, base_revision="d" * 40)
    assert lora.cmd_skeleton(args) == 0
    assert lora.cmd_encode(args) == 0
    assert lora.cmd_finalize(args) == 0

    name = "model-layer-003-b000.safetensors"
    for folder in ("", "digests", "attestations"):
        suffix = {"": "", "digests": ".sha256", "attestations": ".jsonl"}[folder]
        src = other / "stages" / "k2r1" / folder / f"{name}{suffix}"
        dst = campaign.root / "stages" / "k2r1" / folder / f"{name}{suffix}"
        dst.write_bytes(src.read_bytes())
    with pytest.raises(lora.CartridgeError, match="sha256|parent|campaign"):
        combine.combine(combine_args(
            campaign.root, "k3like", tmp_path / "swapped"))


def test_narrowed_adapter_states_per_layer_coverage(campaign, tmp_path: Path):
    out = tmp_path / "hot"
    assert combine.combine(combine_args(
        campaign.root, "hot-k4like", out, experts="1")) == 0
    config = adapter_of(out)
    assert config["selected_experts"] == [1]
    assert config["selected_layers"] == [3, 4]
    assert config["coverage"]["hot"] == {"3": [1], "4": [1]}


def test_combiner_verifies_the_base_checkpoint_it_binds_to(
    campaign, tmp_path: Path
):
    """A cartridge applied to the wrong base corrects other weights."""
    key = signer_of(campaign)
    good = campaign.root / "base" / "k2"
    assert combine.combine(combine_args(
        campaign.root, "k3like", tmp_path / "bound", base=good,
        trust_key=key, insecure_unsigned=False)) == 0

    wrong = campaign.root / "base" / "k3"
    with pytest.raises(lora.CartridgeError, match="not the .* this cartridge"):
        combine.combine(combine_args(
            campaign.root, "k3like", tmp_path / "unbound", base=wrong,
            trust_key=key, insecure_unsigned=False))


def test_combiner_merges_every_stage_of_the_chain(campaign, tmp_path: Path):
    from safetensors import safe_open

    out = tmp_path / "adapter"
    assert combine.combine(combine_args(campaign.root, "hot-k4like", out)) == 0
    config = adapter_of(out)
    assert config["schema"] == lora.ADAPTER_CONFIG_SCHEMA
    assert [stage["label"] for stage in config["chain"]] == ["k2r1", "hot"]
    assert config["selected_experts"] == [0, 1, 2, 3]
    assert "stage_shards" not in config, "the emitted adapter is self-contained"
    labels = set()
    for shard in config["shards"]:
        assert (out / shard).is_file()
        with safe_open(str(out / shard), framework="pt") as handle:
            for key in handle.keys():
                labels.add(key.rsplit("_", 1)[1])
    assert labels == {"k2r1", "hot"}


def test_combiner_narrows_a_full_stage_to_selected_experts(campaign, tmp_path: Path):
    from safetensors import safe_open

    out = tmp_path / "hot-only"
    assert combine.combine(
        combine_args(campaign.root, "k3like", out, experts="1,3")) == 0
    config = adapter_of(out)
    assert config["selected_experts"] == [1, 3]
    seen = set()
    for shard in config["shards"]:
        with safe_open(str(out / shard), framework="pt") as handle:
            experts = {int(key.split(".experts.")[1].split(".")[0])
                       for key in handle.keys()}
        assert experts <= {1, 3}, "unselected experts must not be paid for"
        seen |= experts
    assert seen == {1, 3}
    full = tmp_path / "full"
    assert combine.combine(combine_args(campaign.root, "k3like", full)) == 0
    assert adapter_of(full)["num_tensors"] == 2 * config["num_tensors"]


def test_combiner_restricts_to_selected_layers(campaign, tmp_path: Path):
    out = tmp_path / "layer3"
    assert combine.combine(
        combine_args(campaign.root, "k3like", out, layers="3")) == 0
    shards = adapter_of(out)["shards"]
    assert shards and all("layer-003" in name for name in shards)


def test_combiner_refuses_an_empty_selection(campaign, tmp_path: Path):
    with pytest.raises(lora.CartridgeError, match="matched no experts"):
        combine.combine(combine_args(
            campaign.root, "k3like", tmp_path / "none", experts="99"))


def test_combiner_refuses_altered_stage_bytes(campaign, tmp_path: Path):
    plan = json.loads(
        (campaign.root / "assemblies" / "k3like" / "assembly.json").read_text())
    victim = campaign.root / plan["stage_shards"][0]["path"]
    payload = bytearray(victim.read_bytes())
    payload[-1] ^= 0xFF
    victim.write_bytes(bytes(payload))
    with pytest.raises(lora.CartridgeError, match="refusing to combine altered"):
        combine.combine(combine_args(campaign.root, "k3like", tmp_path / "bad"))


def test_combiner_reports_a_missing_stage_shard(campaign, tmp_path: Path):
    plan = json.loads(
        (campaign.root / "assemblies" / "k3like" / "assembly.json").read_text())
    (campaign.root / plan["stage_shards"][0]["path"]).unlink()
    with pytest.raises(lora.CartridgeError, match="missing stage shard"):
        combine.combine(combine_args(campaign.root, "k3like", tmp_path / "gone"))


@pytest.mark.parametrize("component,replacement,message", [
    ("trellis", lambda torch: torch.zeros(8, 8, 16, dtype=torch.float16),
     "3-D int16 trellis"),
    ("suh", lambda torch: torch.ones(128, dtype=torch.float32),
     "1-D float16 vector"),
    ("suh", lambda torch: torch.ones(64, dtype=torch.float16),
     "expected 128"),
    ("scale", lambda torch: torch.tensor(0.0, dtype=torch.float32),
     "finite and positive"),
    ("scale", lambda torch: torch.tensor(-1.0, dtype=torch.float32),
     "finite and positive"),
])
def test_stage_tensor_validation_rejects_unusable_payloads(
    component, replacement, message
):
    """A signed but corrupt shard must not reach the runtime."""
    torch = pytest.importorskip("torch")
    stage = {"label": "res1", "k": 1, "experts": [0]}
    tensors = stage_tensors()
    key = next(k for k in tensors if k.endswith(f".{component}_res1"))
    tensors[key] = replacement(torch)
    with pytest.raises(lora.CartridgeError, match=message):
        combine.validate_stage_tensors(
            tensors, stage, {0}, 3, "stage.safetensors")


def test_combiner_rejects_a_traversing_assembly_label(campaign, tmp_path: Path):
    with pytest.raises(lora.CartridgeError, match="must match"):
        combine.load_assembly(campaign.root, "../../etc", trust=None)


def test_combiner_refuses_layers_the_assembly_does_not_cover(
    campaign, tmp_path: Path
):
    with pytest.raises(lora.CartridgeError, match="layers .* not"):
        combine.combine(combine_args(
            campaign.root, "k3like", tmp_path / "e", layers="3,999"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
