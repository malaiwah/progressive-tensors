"""The schemas in schemas/ must describe what the tools actually emit.

Two layers of checking:

1. the schemas are themselves valid JSON Schema (draft 2020-12);
2. documents produced right here by fq_repack, fq_fetch and fq_release
   validate against them — so a tool that changes its output without
   updating its schema fails CI, which is the only way a schema stays true.

Real published artifacts (~/fq-segments, ~/fq-0c, ~/fq-primed) are validated
too when they happen to be present; they are absent on CI machines and the
tests skip rather than pretend.
"""
import base64
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

jsonschema = pytest.importorskip("jsonschema")

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import fq_assemble_lora  # noqa: F401  (imported for its emitted documents)
import fq_fetch  # imported for its emitted documents
import fq_release
from test_fq_fetch import (
    REV,
    build_source,
    trust_root,
    write_policy,
)
from test_fq_repack import LAYERS

pytest_plugins = ("test_fq_fetch", "test_fq_assemble_lora")


def _schemas_dir() -> Path | None:
    """Nearest ancestor holding schemas/ (see test_fq_trust._repo_root: the
    research-branch mirror carries the tools without the schema files)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "schemas" / "fq-manifest-1.schema.json").exists():
            return parent / "schemas"
    return None


SCHEMAS = _schemas_dir()
if SCHEMAS is None:
    pytest.skip("schemas/ is not in this checkout", allow_module_level=True)
REAL_TREES = [Path.home() / "fq-segments" / "GLM-5.2-EXL3-FQ",
              Path.home() / "fq-0c" / "fruit-segments",
              Path.home() / "fq-primed" / "segments-336"]


def load(name: str) -> dict:
    return json.loads((SCHEMAS / f"{name}.schema.json").read_text())


def validator(name: str, pointer: str = None):
    schema = load(name)
    if pointer:
        schema = {**schema, "$ref": pointer}
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def check(name: str, doc, pointer: str = None, label: str = ""):
    errors = sorted(validator(name, pointer).iter_errors(doc), key=lambda e: e.path)
    assert not errors, (f"{label or name} does not match {name}: " +
                        "; ".join(f"{list(e.path)}: {e.message}" for e in errors[:5]))


def segment_metadata(path: Path) -> dict:
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(hlen))["__metadata__"]


# ------------------------------------------------------------------ hygiene

def test_every_schema_is_valid_json_schema():
    files = sorted(SCHEMAS.glob("*.schema.json"))
    assert files, "no schemas found"
    for f in files:
        schema = json.loads(f.read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        assert schema["$id"].endswith(f.name), f.name
        assert schema.get("title") and schema.get("description")


def test_sparse_per_k_coverage_preserves_legacy_and_exact_contracts():
    """Legacy v1 extrema remain valid; new coverage is explicitly versioned."""
    exact_layers = [*range(3, 11), *range(35, 51)]
    legacy = {
        "schema": "fq-manifest/1",
        "predicate": "repack-of",
        "layout": "rank_sliced_tp4",
        "k_variants": [5],
        "tensor_index": "index-k5.json",
        "per_k": {"5": {"index": "index-k5.json", "layers": [3, 50]}},
    }
    check("fq-manifest-1", legacy)
    assert "layer_coverage" not in legacy["per_k"]["5"]
    legacy_with_historical_domain = json.loads(json.dumps(legacy))
    legacy_with_historical_domain["per_k"]["5"]["layers"] = [-1, 50]
    check("fq-manifest-1", legacy_with_historical_domain)

    exact = json.loads(json.dumps(legacy))
    exact["per_k"]["5"]["layer_coverage"] = {
        "schema": "fq-layer-coverage/1",
        "layers": exact_layers,
    }
    check("fq-manifest-1", exact)
    assert 11 not in exact["per_k"]["5"]["layer_coverage"]["layers"]
    assert 34 not in exact["per_k"]["5"]["layer_coverage"]["layers"]


# ---------------------------------------------------- freshly emitted output

@pytest.fixture()
def emitted(tmp_path, served):
    """A segment repo, a fetched subset of it, and a signed release."""
    repo, snap, pub = build_source(tmp_path, "pub")
    fq_release.main(["build", "--dir", str(repo), "--release", "schemas 0.1.0",
                     "--repo", "test/pub", "--revision", REV,
                     "--sign-key", str(tmp_path / "pub.key")])
    served["mount"]("test/pub", repo)
    policy = write_policy(tmp_path / "recipe.json", {LAYERS[0]: [3, 4, 3, 4]})
    out = tmp_path / "fetched"
    assert fq_fetch.main([
        "--policy", str(policy), "--out", str(out), "--source", f"test/pub@{REV}",
        "--trust-signer", pub, "--trust-root", str(trust_root(tmp_path, pub))]) == 0
    return repo, out, policy


def test_repack_output_matches_the_schemas(emitted):
    repo, out, policy = emitted
    check("fq-manifest-1", json.loads((repo / "fq-manifest.json").read_text()),
          label="fq_repack manifest")
    for k in (3, 4):
        check("fq-segment-index-1", json.loads((repo / f"index-k{k}.json").read_text()),
              label=f"fq_repack index-k{k}")
    for seg in sorted(repo.glob("layer-*.safetensors")):
        check("fq-segment-1", segment_metadata(seg), label=seg.name)
    for att in sorted((repo / "attestations").glob("*.jsonl")):
        for n, line in enumerate(att.read_text().splitlines(), 1):
            envelope = json.loads(line)
            check("fq-attestation-1", envelope, label=f"{att.name}:{n} envelope")
            import base64
            check("fq-attestation-1", json.loads(base64.b64decode(envelope["payload"])),
                  pointer="#/$defs/payload", label=f"{att.name}:{n} payload")


def test_fetch_output_matches_the_schemas(emitted):
    repo, out, policy = emitted
    check("fq-policy-2", json.loads(policy.read_text()), label="recipe")
    check("fq-manifest-1", json.loads((out / "fq-manifest.json").read_text()),
          label="fq_fetch manifest")
    for idx in sorted(out.glob("index-k*.json")):
        check("fq-segment-index-1", json.loads(idx.read_text()), label=idx.name)
    for seg in sorted(out.glob("layer-*.safetensors")):
        check("fq-segment-1", segment_metadata(seg), label=seg.name)


def test_release_output_matches_the_schema(emitted):
    import base64

    repo, out, policy = emitted
    envelope = json.loads((repo / "fq-release.json").read_text())
    check("fq-release-1", envelope, label="release envelope")
    check("fq-release-1", json.loads(base64.b64decode(envelope["payload"])),
          pointer="#/$defs/payload", label="release payload")


@pytest.mark.parametrize("name", [
    "glm52-k2k3-dag", "fruit-k2k3-dag", "fruit-k2-k3k4-cart",
])
def test_published_cartridge_recipes_match_schema(name):
    recipe = Path(__file__).parent.parent / "recipes" / f"{name}.json"
    check("fq-cartridge-2", json.loads(recipe.read_text()), label=recipe.name)


def test_campaign_assembly_plan_and_adapter_match_their_schemas(
    campaign, tmp_path
):
    import fq_combine_cartridges as combine

    for label in ("k3like", "hot-k4like"):
        plan = campaign.root / "assemblies" / label / "assembly.json"
        check("fq-cartridge-assembly-1", json.loads(plan.read_text()),
              label=f"{label}/assembly.json")

    out = tmp_path / "adapter"
    combine.combine(SimpleNamespace(
        root=campaign.root, assembly="hot-k4like", out=out, experts=None,
        layers=None, force=False, insecure_unsigned=False, base=None,
        trust_key=json.loads(
            (campaign.root / "campaign_summary.json").read_text()
        )["provenance"]["signer_pubkey"]))
    check("fq-cartridge-adapter-2",
          json.loads((out / "adapter_config.json").read_text()),
          label="adapter_config.json")


def test_campaign_attestations_match_the_schema_and_the_verifier(campaign):
    """Every line the campaign signs must satisfy the schema it claims.

    Expert fragments carry `expert_sha256`; a skeleton shard carries no routed
    expert, so it declares `kind: "skeleton"` and maps its tensors instead.
    Both profiles are checked here against `fq-attestation/1` *and* against the
    shared verifier in fq_verify, so a campaign cannot emit provenance that its
    own trust tooling would reject.
    """
    import fq_verify

    seen = set()
    for node in ("base/k2", "stages/k2r1", "stages/hot", "skeleton"):
        directory = campaign.root / node
        for line in sorted((directory / "attestations").glob("*.jsonl")):
            payload = json.loads(base64.b64decode(
                json.loads(line.read_text())["payload"]))
            check("fq-attestation-1", payload,
                  pointer="#/$defs/payload", label=f"{node}/{line.name}")
            fq_verify.validate_attestation_payload(
                payload, where=f"{node}/{line.name}")
            seen.add(payload["predicate"])
            if payload.get("kind") == "skeleton":
                assert payload["tensor_sha256"]
                assert "expert_sha256" not in payload
            else:
                assert payload["expert_sha256"]
                assert payload["materials"]["encoder_sha256"]
                assert payload["determinism_scope"]
    assert seen == {"encode-of", "repack-of"}


# ----------------------------------------------- documents that must NOT pass

@pytest.mark.parametrize("doc,name,why", [
    ({"schema": "fq-policy/1", "bits_per_expert": {"3": [3]}}, "fq-policy-2",
     "wrong schema version"),
    ({"schema": "fq-policy/2", "bits_per_expert": {"three": [3]}}, "fq-policy-2",
     "layer key is not a number"),
    ({"schema": "fq-policy/2", "bits_per_expert": {"3": ["3"]}}, "fq-policy-2",
     "bit-width is not an integer"),
    ({"schema": "fq-manifest/1", "predicate": "invented-of", "layout": "x",
      "k_variants": [3], "tensor_index": "index-k3.json"}, "fq-manifest-1",
     "unknown predicate"),
    ({"schema": "fq-manifest/1", "predicate": "repack-of", "layout": "x",
      "k_variants": [3]}, "fq-manifest-1", "no index named"),
    ({"3": {"file": "layer-003.k3.safetensors", "sha256": "0" * 63,
            "size": 1, "body_offset": 8, "experts": {"0": [0, 1]}}},
     "fq-segment-index-1", "sha256 is the wrong length"),
    ({"payload": "e30=", "signature": "AA==", "keyid": "nope"},
     "fq-attestation-1", "keyid is not a key"),
    ({"schema": "fq-manifest/1", "predicate": "repack-of",
      "layout": "rank_sliced_tp4", "k_variants": [5],
      "tensor_index": "index-k5.json",
      "per_k": {"5": {"layers": [3, 50], "layer_coverage": {
          "schema": "fq-layer-coverage/1", "layers": [3, 3]}}}},
     "fq-manifest-1", "duplicate exact per-K layer"),
])
def test_malformed_documents_are_rejected(doc, name, why):
    assert list(validator(name).iter_errors(doc)), f"schema accepted {why}"


# ------------------------------------------------- real artifacts, if present

@pytest.mark.parametrize("tree", REAL_TREES, ids=lambda p: p.name)
def test_real_published_trees_match_the_schemas(tree):
    if not tree.exists():
        pytest.skip(f"{tree} not present (published artifacts are not in CI)")
    import base64

    manifest = tree / "fq-manifest.json"
    if manifest.exists():
        check("fq-manifest-1", json.loads(manifest.read_text()), label=str(manifest))
    for idx in sorted(tree.glob("index-k*.json")):
        check("fq-segment-index-1", json.loads(idx.read_text()), label=str(idx))
    for seg in sorted(tree.glob("layer-*.safetensors"))[:4]:
        check("fq-segment-1", segment_metadata(seg), label=str(seg))
    for att in sorted((tree / "attestations").glob("*.jsonl"))[:4]:
        for n, line in enumerate(att.read_text().splitlines(), 1):
            envelope = json.loads(line)
            check("fq-attestation-1", envelope, label=f"{att}:{n}")
            check("fq-attestation-1", json.loads(base64.b64decode(envelope["payload"])),
                  pointer="#/$defs/payload", label=f"{att}:{n} payload")
