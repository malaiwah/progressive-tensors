#!/usr/bin/env python3
"""Combine one published MSRT assembly into a self-contained cartridge adapter.

An encode campaign publishes atomic stage segments, a signed
``fq-cartridge/1``-style attestation per fragment, and a signed
``fq-cartridge-assembly/1`` plan per product. This tool turns one plan into the
artifact an EXL3 MSRT runtime loads: every stage of the chain merged per shard,
optionally narrowed to a subset of experts or layers, with a self-contained
``fq-cartridge-adapter/2`` contract.

Nothing here trusts the campaign directory. The plan's signature is checked
first, then every stage shard is re-hashed and matched against its own signed
attestation, and the runtime constants in the emitted contract are re-derived
from this tool rather than copied out of the plan -- a tampered
``mcg_multiplier`` would otherwise decode every weight wrongly while validating
against the schema.

Narrowing is the point of the format. A consumer who only wants the hottest 96
experts upgraded pays for 96 experts of residual, not 256, and the emitted
contract records exactly which experts of which layers the runtime will find.

Usage:
  python tools/fq_combine_cartridges.py \\
    --root ./campaign --assembly k2-k4like-direct --out ./k4like-hot96 \\
    --experts 0-95 --trust-key <64-hex ed25519 public key>
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
import fq_assemble_lora as lora
from fq_assemble import AssemblyError, StagedOutput, check_out_dir, sha256_file

TOOL_VERSION = "fq_combine_cartridges/2"
STAGE_KEY_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.rank(\d+)\."
    r"(trellis|suh|svh|scale)_([A-Za-z0-9_-]{1,32})$"
)
COMPONENTS = {"trellis", "suh", "svh", "scale"}
BLOCK_NAME_RE = re.compile(r"^model-layer-(\d{3})-b(\d{3})\.safetensors$")


def parse_ids(value: str | None) -> set[int] | None:
    """Parse ``0-95,128,200`` into an explicit id set."""
    if value is None:
        return None
    ids: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            start, stop = int(lo), int(hi)
            if start > stop:
                raise lora.CartridgeError(f"empty id range {part!r}")
            ids.update(range(start, stop + 1))
        else:
            ids.add(int(part))
    if not ids or any(value < 0 for value in ids):
        raise lora.CartridgeError(f"{value!r} is not a non-negative id list")
    return ids


def resolve_trust(value: str | None) -> str | None:
    """A pinned ed25519 public key, as hex or a path holding hex."""
    if value is None:
        return None
    candidate = Path(value).expanduser()
    text = candidate.read_text().strip() if candidate.is_file() else value.strip()
    if len(text) != 64 or any(c not in "0123456789abcdefABCDEF" for c in text):
        raise lora.CartridgeError(
            f"--trust-key must be a 64-hex ed25519 public key, got {text!r}")
    return text.lower()


def safe_relative(value: Any, *, where: str) -> Path:
    relative = Path(str(value))
    if (not isinstance(value, str) or relative.is_absolute()
            or ".." in relative.parts or not str(value).endswith(
                (".safetensors", ".jsonl"))):
        raise lora.CartridgeError(
            f"{where}: {value!r} must be a safe campaign-relative path")
    return relative


def under_root(root: Path, relative: str, *, where: str) -> Path:
    """Resolve a campaign-relative path and prove it stayed inside the campaign.

    The plan is signed, but a symlink planted in the tree could still redirect
    a listed path outside it, so resolution is checked rather than trusted.
    """
    resolved = (root / relative).resolve()
    if resolved != root and root not in resolved.parents:
        raise lora.CartridgeError(
            f"{where} {relative!r} resolves outside {root}")
    return resolved


def load_assembly(
    root: Path, label: str, *, trust: str | None
) -> tuple[dict[str, Any], Path]:
    """Load one published assembly plan, verified under a pinned key."""
    if not lora.LABEL_RE.fullmatch(label):
        raise lora.CartridgeError(
            f"--assembly {label!r} must match {lora.LABEL_RE.pattern}; it names "
            f"a directory inside the campaign")
    directory = root / "assemblies" / label
    signed = directory / "assembly.jsonl"
    if trust is None:
        plan = json.loads((directory / "assembly.json").read_text())
        keyid = None
    else:
        payload = lora.read_signed_line(signed)
        plan, keyid = payload, payload.pop("_keyid")
        if keyid != trust:
            raise lora.CartridgeError(
                f"{signed}: signed by {keyid[:16]}, not the pinned "
                f"{trust[:16]}")
    if not isinstance(plan, dict) or plan.get("schema") != lora.ASSEMBLY_SCHEMA:
        raise lora.CartridgeError(
            f"{signed}: schema must be {lora.ASSEMBLY_SCHEMA!r}")
    if plan.get("format") != "exl3-msrt-full-rank":
        raise lora.CartridgeError(f"{signed}: unsupported format "
                                  f"{plan.get('format')!r}")
    if plan.get("standard_lora_compatible") is not False:
        raise lora.CartridgeError(f"{signed}: standard_lora_compatible must be false")
    if plan.get("assembly") != label:
        raise lora.CartridgeError(
            f"{signed}: plan describes {plan.get('assembly')!r}, not {label!r}")
    for field, expected in (("codebook", "mcg"),
                            ("mcg_ownership", "adapter-config"),
                            ("mcg_multiplier", lora.MCG_MULTIPLIER),
                            ("scale_shape", []),
                            ("paths_relative_to", "campaign root")):
        if plan.get(field) != expected:
            raise lora.CartridgeError(
                f"{signed}: {field} is {plan.get(field)!r}, expected "
                f"{expected!r}; refusing to republish an altered runtime "
                f"contract")
    base = plan.get("base")
    if (not isinstance(base, dict)
            or not isinstance(base.get("manifest_sha256"), str)
            or len(base["manifest_sha256"]) != 64
            or not lora.LABEL_RE.fullmatch(str(base.get("label")))):
        raise lora.CartridgeError(f"{signed}: invalid base checkpoint identity")
    lora._validate_k(base.get("k"), who="base k")
    chain = plan.get("chain")
    if not isinstance(chain, list) or not chain:
        raise lora.CartridgeError(
            f"{signed}: this product applies no residual stage, so it is a base "
            f"checkpoint, not a cartridge; load base/{base.get('label')} "
            f"directly")
    expected_parent = base["label"]
    seen: set[str] = set()
    for stage in chain:
        if not isinstance(stage, dict):
            raise lora.CartridgeError(f"{signed}: chain entries must be objects")
        stage_label = str(stage.get("label"))
        if not lora.LABEL_RE.fullmatch(stage_label) or stage_label in seen:
            raise lora.CartridgeError(
                f"{signed}: invalid or duplicated stage label {stage_label!r}")
        seen.add(stage_label)
        lora._validate_k(stage.get("k"), who=f"stage {stage_label!r} k")
        if stage.get("parent") != expected_parent:
            raise lora.CartridgeError(
                f"{signed}: stage {stage_label!r} corrects "
                f"{stage.get('parent')!r}, not {expected_parent!r}; the plan is "
                f"not one path through the recipe graph")
        experts = stage.get("experts")
        if experts != "all" and (
                not isinstance(experts, list) or not experts
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 0
                       for v in experts)
                or len(set(experts)) != len(experts)):
            raise lora.CartridgeError(
                f"{signed}: stage {stage_label!r} has an invalid expert set")
        expected_parent = stage_label

    campaign = plan.get("campaign")
    if (not isinstance(campaign, dict)
            or not isinstance(campaign.get("recipe_sha256"), str)
            or not isinstance(campaign.get("base_revision"), str)
            or not isinstance(campaign.get("signer_pubkey"), str)):
        raise lora.CartridgeError(f"{signed}: plan lacks a campaign identity")
    if trust is not None and campaign["signer_pubkey"] != trust:
        raise lora.CartridgeError(
            f"{signed}: campaign names signer {campaign['signer_pubkey'][:16]}, "
            f"not the pinned {trust[:16]}")
    stage_shards = plan.get("stage_shards")
    if not isinstance(stage_shards, list) or not stage_shards:
        raise lora.CartridgeError(f"{signed}: stage_shards must be a non-empty list")
    parent_of = {stage["label"]: stage["parent"] for stage in chain}
    unique: set[tuple[str, str]] = set()
    for entry in stage_shards:
        if not isinstance(entry, dict) or entry.get("label") not in seen:
            raise lora.CartridgeError(f"{signed}: malformed entry {entry!r}")
        path = safe_relative(entry.get("path"), where=str(signed))
        safe_relative(entry.get("attestation"), where=str(signed))
        if not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64:
            raise lora.CartridgeError(f"{signed}: entry {entry!r} lacks a sha256")
        experts = entry.get("experts")
        if (not isinstance(experts, list) or not experts
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 0
                       for v in experts)
                or len(set(experts)) != len(experts)):
            raise lora.CartridgeError(
                f"{signed}: entry {entry['path']!r} has an invalid expert list")
        if (not isinstance(entry.get("layer"), int)
                or not isinstance(entry.get("block"), int)):
            raise lora.CartridgeError(
                f"{signed}: entry {entry['path']!r} lacks (layer, block)")
        if (not isinstance(entry.get("parent_sha256"), str)
                or len(entry["parent_sha256"]) != 64
                or entry.get("parent_label") != parent_of[entry["label"]]):
            raise lora.CartridgeError(
                f"{signed}: entry {entry['path']!r} does not name the parent "
                f"fragment its residual corrects")
        match = BLOCK_NAME_RE.fullmatch(path.name)
        if match is None:
            raise lora.CartridgeError(
                f"{signed}: {path.name!r} is not a campaign block name")
        if (int(match.group(1)), int(match.group(2))) != (entry["layer"],
                                                          entry["block"]):
            raise lora.CartridgeError(
                f"{signed}: {path.name!r} disagrees with its declared "
                f"(layer, block)")
        if path.parts[:2] != ("stages", entry["label"]):
            raise lora.CartridgeError(
                f"{signed}: {entry['path']!r} does not live under "
                f"stages/{entry['label']}")
        key = (entry["label"], path.name)
        if key in unique:
            raise lora.CartridgeError(
                f"{signed}: {entry['label']}/{path.name} is listed twice")
        unique.add(key)
    by_label: dict[tuple[str, str], str] = {
        (entry["label"], Path(entry["path"]).name): entry["sha256"]
        for entry in stage_shards
    }
    for entry in stage_shards:
        parent = entry["parent_label"]
        if parent == base["label"]:
            continue  # bound to the base checkpoint, verified via its manifest
        published = by_label.get((parent, Path(entry["path"]).name))
        if published != entry["parent_sha256"]:
            raise lora.CartridgeError(
                f"{signed}: {entry['path']} corrects {parent} bytes "
                f"{entry['parent_sha256'][:16]}, but the plan publishes "
                f"{published if published is None else published[:16]}; the "
                f"chain is not internally consistent")
    return plan, signed


def read_block(
    root: Path, entry: dict[str, Any], *, trust: str | None,
    campaign: dict[str, Any],
) -> dict[str, str]:
    """Verify one *selected* stage shard against its digest and attestation.

    Only shards the selection actually needs are required to be present, so a
    consumer who wants 96 experts of one layer never has to download, keep or
    hash the rest of the campaign.
    """
    source = under_root(root, entry["path"], where="stage shard")
    if not source.is_file():
        raise lora.CartridgeError(
            f"missing stage shard {entry['path']} (needed by this selection)")
    sha, _body, spans = lora.verify_shard(source, group_kind="expert")
    if sha != entry["sha256"]:
        raise lora.CartridgeError(
            f"{entry['path']}: sha256 {sha} != published {entry['sha256']}; "
            f"refusing to combine altered bytes")
    header, _ = lora.read_header(source)
    meta = header.get("__metadata__") or {}
    if meta.get("schema") != lora.BLOCK_SCHEMA:
        raise lora.CartridgeError(
            f"{source}: metadata schema is {meta.get('schema')!r}, expected "
            f"{lora.BLOCK_SCHEMA!r}; this is not an encoded MSRT block")
    for field in ("label", "layer", "block", "covered_experts"):
        if field not in meta:
            raise lora.CartridgeError(f"{source}: block metadata lacks {field}")
    if meta["label"] != entry["label"]:
        raise lora.CartridgeError(
            f"{source}: metadata label {meta['label']!r} != plan "
            f"{entry['label']!r}")
    match = BLOCK_NAME_RE.fullmatch(Path(entry["path"]).name)
    if (int(match.group(1)), int(match.group(2))) != (int(meta["layer"]),
                                                      int(meta["block"])):
        raise lora.CartridgeError(
            f"{source}: file name disagrees with its recorded (layer, block)")
    if trust is not None:
        payload = lora.read_signed_line(
            under_root(root, entry["attestation"], where="attestation"))
        if payload.pop("_keyid") != trust:
            raise lora.CartridgeError(
                f"{entry['attestation']}: not signed by the pinned key")
        fragment = payload.get("fragment") or {}
        if (fragment.get("sha256") != sha
                or fragment.get("file") != Path(entry["path"]).name
                or payload.get("expert_sha256") != spans):
            raise lora.CartridgeError(
                f"{entry['attestation']}: does not attest these bytes")
        if payload.get("predicate") != "encode-of":
            raise lora.CartridgeError(
                f"{entry['attestation']}: predicate is not encode-of")
        claim = (payload.get("parents") or [None])[0]
        if (not isinstance(claim, dict)
                or claim.get("label") != entry["parent_label"]
                or claim.get("sha256") != entry["parent_sha256"]):
            raise lora.CartridgeError(
                f"{entry['attestation']}: attests parent {claim!r}, not the "
                f"{entry['parent_label']} bytes the plan names")
        for field, value in (("recipe_sha256", campaign["recipe_sha256"]),
                             ("base_revision", campaign["base_revision"])):
            if payload.get(field) != value:
                raise lora.CartridgeError(
                    f"{entry['attestation']}: {field} disagrees with the "
                    f"campaign identity in the signed plan")
    covered = {int(value) for value in meta["covered_experts"].split(",") if value}
    if covered != set(entry["experts"]):
        raise lora.CartridgeError(
            f"{source}: carries experts {sorted(covered)} but the signed plan "
            f"promises {sorted(entry['experts'])}")
    return meta


def validate_stage_tensors(
    tensors: dict[str, Any], stage: dict[str, Any], expected: set[int],
    layer: int, shard: str,
) -> None:
    """Require exact components, K geometry, projection, and expert coverage."""
    import torch

    label = stage["label"]
    groups: dict[tuple[int, int, str, int], set[str]] = {}
    observed: set[int] = set()
    shapes: dict[tuple[int, str], dict[str, Any]] = {}
    for key, tensor in tensors.items():
        match = STAGE_KEY_RE.fullmatch(key)
        if not match:
            raise lora.CartridgeError(f"{shard}: unexpected tensor key {key!r}")
        key_layer, expert, projection, rank, component, key_label = (
            int(match.group(1)), int(match.group(2)), match.group(3),
            int(match.group(4)), match.group(5), match.group(6))
        if key_label != label:
            raise lora.CartridgeError(
                f"{shard}: tensor {key!r} carries label {key_label!r}, "
                f"expected {label!r}")
        if key_layer != layer:
            raise lora.CartridgeError(
                f"{shard}: tensor {key!r} belongs to layer {key_layer}, "
                f"expected {layer}")
        if rank != 0:
            raise lora.CartridgeError(
                f"{shard}: {key} is rank {rank}; this format is rank0 only")
        groups.setdefault((key_layer, expert, projection, rank), set()).add(component)
        observed.add(expert)
        shapes.setdefault((expert, projection), {})[component] = tensor
        if component == "trellis":
            if (tensor.ndim != 3 or tensor.shape[-1] != stage["k"] * 16
                    or tensor.dtype is not torch.int16):
                raise lora.CartridgeError(
                    f"{shard}: {key} is {tuple(tensor.shape)} {tensor.dtype}, "
                    f"expected a 3-D int16 trellis whose last dimension is "
                    f"{stage['k'] * 16}")
        elif component in {"suh", "svh"}:
            if tensor.ndim != 1 or tensor.dtype is not torch.float16:
                raise lora.CartridgeError(
                    f"{shard}: {key} must be a 1-D float16 vector, got "
                    f"{tuple(tensor.shape)} {tensor.dtype}")
        elif component == "scale":
            if tensor.ndim != 0 or tensor.dtype is not torch.float32:
                raise lora.CartridgeError(
                    f"{shard}: {key} must be a float32 scalar, got "
                    f"{tuple(tensor.shape)} {tensor.dtype}")
            value = float(tensor)
            if not math.isfinite(value) or value <= 0:
                raise lora.CartridgeError(
                    f"{shard}: {key} is {value}; a residual rescale factor must "
                    f"be finite and positive or the runtime inverts the stage")
    if observed != expected:
        raise lora.CartridgeError(
            f"{shard}: experts {sorted(observed)} != selected {sorted(expected)}")
    for group, components in groups.items():
        if components != COMPONENTS:
            raise lora.CartridgeError(
                f"{shard}: {group} components {sorted(components)} != "
                f"{sorted(COMPONENTS)}")
    # The Hadamard vectors have to match the trellis geometry they invert, or
    # the runtime silently reconstructs a differently shaped matrix.
    for (expert, projection), parts in shapes.items():
        trellis = parts.get("trellis")
        if trellis is None:
            continue
        for component, axis in (("suh", 0), ("svh", 1)):
            vector = parts.get(component)
            if vector is not None and vector.numel() != trellis.shape[axis] * 16:
                raise lora.CartridgeError(
                    f"{shard}: expert {expert} {projection} {component} has "
                    f"{vector.numel()} entries, expected "
                    f"{trellis.shape[axis] * 16} for trellis "
                    f"{tuple(trellis.shape)}")
    ranks = {group[3] for group in groups}
    for expert in expected:
        missing = {
            (expert, projection, rank)
            for projection in lora.PROJECTIONS for rank in ranks
            if (layer, expert, projection, rank) not in groups
        }
        if missing:
            raise lora.CartridgeError(
                f"{shard}: incomplete projection/rank coverage for "
                f"{sorted(missing)[:3]}")


def combine(args) -> int:
    lora.require_quant_dependencies()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise lora.CartridgeError(f"--root {root} is not a directory")
    trust = resolve_trust(args.trust_key)
    if trust is None and not args.insecure_unsigned:
        raise lora.CartridgeError(
            "pass --trust-key <64-hex public key> to check the campaign's "
            "signatures, or --insecure-unsigned to combine unverified bytes")
    plan, plan_path = load_assembly(root, args.assembly, trust=trust)
    out = check_out_dir(args.out, source=root, policy=plan_path)
    wanted_experts = parse_ids(args.experts)
    wanted_layers = parse_ids(args.layers)
    stage_by_label = {stage["label"]: stage for stage in plan["chain"]}

    from safetensors import safe_open

    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in plan["stage_shards"]:
        grouped.setdefault(Path(entry["path"]).name, []).append(entry)

    staged = StagedOutput(out, args.force)
    work = staged.begin()
    shards: list[str] = []
    tensor_count = 0
    coverage: dict[str, dict[str, list[int]]] = {}
    requested_seen: set[int] = set()
    try:
        for basename in sorted(grouped):
            combined: dict[str, Any] = {}
            for entry in sorted(grouped[basename], key=lambda e: e["label"]):
                layer = entry["layer"]
                if wanted_layers is not None and layer not in wanted_layers:
                    continue
                keep = (set(entry["experts"]) if wanted_experts is None
                        else set(entry["experts"]) & wanted_experts)
                if not keep:
                    continue
                read_block(root, entry, trust=trust,
                           campaign=plan["campaign"])
                source = under_root(
                    root, entry["path"], where="stage shard")
                with safe_open(str(source), framework="pt") as handle:
                    tensors = {}
                    for key in handle.keys():
                        match = STAGE_KEY_RE.fullmatch(key)
                        if match is None:
                            raise lora.CartridgeError(
                                f"{source}: unexpected tensor key {key!r}")
                        if int(match.group(2)) in keep:
                            tensors[key] = handle.get_tensor(key)
                validate_stage_tensors(
                    tensors, stage_by_label[entry["label"]], keep, layer,
                    str(source))
                collisions = set(combined) & set(tensors)
                if collisions:
                    raise lora.CartridgeError(
                        f"{source}: duplicate tensor keys {sorted(collisions)[:3]}")
                combined.update(tensors)
                by_layer = coverage.setdefault(entry["label"], {})
                by_layer[str(layer)] = sorted(
                    set(by_layer.get(str(layer), [])) | keep)
                requested_seen |= keep
            if not combined:
                continue
            lora.save_shard_tensors(combined, work, basename)
            shards.append(basename)
            tensor_count += len(combined)
        if not shards:
            raise lora.CartridgeError(
                "the selection matched no experts; nothing to combine")
        if wanted_experts is not None and wanted_experts - requested_seen:
            raise lora.CartridgeError(
                f"experts {sorted(wanted_experts - requested_seen)[:8]} are not "
                f"covered by assembly {args.assembly!r}; the emitted contract "
                f"would silently promise less than you asked for")
        if wanted_experts is not None:
            # An expert present in one selected layer must not mask its absence
            # from another: the request was for those experts everywhere.
            per_layer: dict[int, set[int]] = {}
            for by_layer in coverage.values():
                for layer_key, ids in by_layer.items():
                    per_layer.setdefault(int(layer_key), set()).update(ids)
            for layer_id, present in sorted(per_layer.items()):
                gap = wanted_experts - present
                if gap:
                    raise lora.CartridgeError(
                        f"layer {layer_id} does not carry experts "
                        f"{sorted(gap)[:8]} in assembly {args.assembly!r}")
        # Every stage of the chain must survive the narrowing, or the product
        # is not the product its own chain describes.
        for stage in plan["chain"]:
            if not coverage.get(stage["label"]):
                raise lora.CartridgeError(
                    f"stage {stage['label']!r} contributed no tensors after "
                    f"narrowing; this selection cannot form {args.assembly!r}")
        layers_seen = sorted({int(layer) for by_layer in coverage.values()
                              for layer in by_layer})
        if wanted_layers is not None and wanted_layers - set(layers_seen):
            raise lora.CartridgeError(
                f"layers {sorted(wanted_layers - set(layers_seen))} are not "
                f"covered by assembly {args.assembly!r}")
        config = {
            "schema": lora.ADAPTER_CONFIG_SCHEMA,
            "assembly": plan["assembly"],
            "base": plan["base"],
            "chain": plan["chain"],
            # Re-derived here, never copied from the plan: a tampered codebook
            # or multiplier would decode every weight wrongly.
            "format": "exl3-msrt-full-rank",
            "standard_lora_compatible": False,
            "runtime_operation": lora.RUNTIME_OPERATION,
            "codebook": "mcg",
            "mcg_multiplier": lora.MCG_MULTIPLIER,
            "mcg_ownership": "adapter-config",
            "scale_shape": [],
            "shards": shards,
            "num_tensors": tensor_count,
            "selected_experts": sorted(requested_seen),
            "selected_layers": layers_seen,
            "coverage": coverage,
            "verified_signer": trust,
            "campaign": plan["campaign"],
            "source_assembly": {
                "path": plan_path.relative_to(root).as_posix(),
                "sha256": sha256_file(plan_path),
            },
            "tool_version": TOOL_VERSION,
            "created_utc": lora.now_utc(),
        }
        (work / "adapter_config.json").write_text(
            json.dumps(config, indent=2) + "\n")
        staged.commit()
    except Exception:
        staged.abort()
        raise
    print(f"Combined {len(shards)} shards, {tensor_count} tensors, "
          f"{len(requested_seen)} experts across {len(layers_seen)} layers "
          f"into {out}", flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="Campaign root written by fq_assemble_lora")
    parser.add_argument("--assembly", required=True,
                        help="Assembly label published under assemblies/")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--experts",
                        help="Restrict to these expert ids, e.g. 0-95,200")
    parser.add_argument("--layers", help="Restrict to these layers, e.g. 3-40")
    parser.add_argument("--trust-key",
                        help="Pinned ed25519 public key (hex or file) that must "
                             "have signed the plan and every fragment")
    parser.add_argument("--insecure-unsigned", action="store_true",
                        help="Combine without checking any signature")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        return combine(args)
    except (AssemblyError, lora.CartridgeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
