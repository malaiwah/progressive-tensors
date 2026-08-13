#!/usr/bin/env python3
"""Combine one published MSRT assembly into a self-contained cartridge adapter.

An encode campaign publishes atomic stage segments, a signed
``fq-attestation/1`` record per fragment, and a signed
``fq-cartridge-assembly/2`` plan per product. This tool turns one plan into the
artifact an EXL3 MSRT runtime loads: every stage of the chain merged per shard,
optionally narrowed to a subset of experts or layers, with a self-contained
``fq-cartridge-adapter/3`` contract.

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
import datetime
import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
import fq_assemble_lora as lora
from fq_assemble import AssemblyError, StagedOutput, check_out_dir, sha256_file

TOOL_VERSION = "fq_combine_cartridges/3"
STAGE_KEY_RE = re.compile(
    r"^model\.layers\.([0-9]+)\.mlp\.experts\.([0-9]+)\."
    r"(gate_proj|up_proj|down_proj)\.rank([0-9]+)\."
    r"(trellis|scale)_([A-Za-z0-9_-]{1,32})$"
)
COMPONENTS = {"trellis", "scale"}
BLOCK_NAME_RE = re.compile(r"^model-layer-([0-9]{3})-b([0-9]{3})\.safetensors$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_ADAPTER_CONFIG_BYTES = 16 * 1024 * 1024
MAX_ADAPTER_SHARD_BYTES = 64 * 1024**3
MAX_ADAPTER_TOTAL_BYTES = 1024**4
MAX_STAGE_SOURCE_BYTES = 64 * 1024**3
MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
ASSEMBLY_FIELDS = {
    "schema",
    "assembly",
    "base",
    "chain",
    "format",
    "runtime_profile",
    "rotation_ownership",
    "standard_lora_compatible",
    "runtime_operation",
    "codebook",
    "mcg_multiplier",
    "mcg_ownership",
    "scale_shape",
    "tensor_parallel",
    "shards",
    "stage_shards",
    "paths_relative_to",
    "bits_per_weight",
    "num_tensors",
    "tool_version",
    "created_utc",
    "campaign",
}
CAMPAIGN_FIELDS = {
    "recipe_sha256",
    "base_model",
    "base_revision",
    "encoder_sha256",
    "signer_pubkey",
    "block_size",
    "moe_layers",
}
STAGE_SHARD_FIELDS = {
    "label",
    "path",
    "sha256",
    "attestation",
    "layer",
    "block",
    "experts",
    "parent_label",
    "parent_sha256",
}
BASE_TAIL_FIELDS = {
    "format",
    "runtime_profile",
    "bits",
    "codebook",
    "moe_layers",
    "moe_layer_coverage",
    "tensor_schema",
    "tensor_parallel",
    "mcg_multiplier",
    "compatibility_sha256",
    "compatibility_by_layer",
    "experts_per_layer",
}
BASE_TENSOR_SCHEMA = (
    "model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}"
)


def valid_rfc3339(value: Any) -> bool:
    if (
        not isinstance(value, str)
        or re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt ][0-9]{2}:[0-9]{2}:[0-9]{2}"
            r"(?:\.[0-9]+)?(?:[Zz]|[+-][0-9]{2}:[0-9]{2})",
            value,
        )
        is None
    ):
        return False
    try:
        normalized = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
        parsed = datetime.datetime.fromisoformat(normalized)
    except ValueError:
        return False
    return parsed.tzinfo is not None


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def valid_label(value: Any) -> bool:
    return isinstance(value, str) and lora.LABEL_RE.fullmatch(value) is not None


def read_bounded_safetensors_header(path: Path) -> dict[str, Any]:
    """Read a campaign header without trusting its declared allocation size."""
    try:
        with open(path, "rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise lora.CartridgeError(f"{path}: truncated safetensors prefix")
            header_size = int.from_bytes(prefix, "little")
            if not 0 < header_size <= MAX_SAFETENSORS_HEADER_BYTES:
                raise lora.CartridgeError(
                    f"{path}: safetensors header is not within the "
                    f"{MAX_SAFETENSORS_HEADER_BYTES}-byte limit"
                )
            raw_header = handle.read(header_size)
            if len(raw_header) != header_size:
                raise lora.CartridgeError(f"{path}: truncated safetensors header")
        header = json.loads(raw_header)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise lora.CartridgeError(f"{path}: invalid safetensors header") from exc
    if not isinstance(header, dict):
        raise lora.CartridgeError(f"{path}: safetensors header must be an object")
    return header


def tensor_parallel_contract(layout: str, world_size: int) -> dict[str, Any]:
    """Return the closed adapter/3 TP contract, including TP=1 rank intent."""
    if layout not in {"full", "rank-sharded"}:
        raise lora.CartridgeError(f"unsupported tensor-parallel layout {layout!r}")
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size < 1
    ):
        raise lora.CartridgeError("tensor-parallel world size must be positive")
    if layout == "full" and world_size != 1:
        raise lora.CartridgeError(
            "full adapter storage has one logical rank; use rank-sharded to "
            "materialize a fixed tensor-parallel world size"
        )
    return {
        "layout": layout,
        "world_size": world_size,
        "ranks": list(range(world_size)),
        "axis_by_projection": dict(lora.TP_AXIS_BY_PROJECTION),
    }


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
            f"--trust-key must be a 64-hex ed25519 public key, got {text!r}"
        )
    return text.lower()


def safe_relative(value: Any, *, where: str) -> Path:
    relative = Path(str(value))
    if (
        not isinstance(value, str)
        or relative.is_absolute()
        or ".." in relative.parts
        or not str(value).endswith((".safetensors", ".jsonl"))
    ):
        raise lora.CartridgeError(
            f"{where}: {value!r} must be a safe campaign-relative path"
        )
    return relative


def under_root(root: Path, relative: str, *, where: str) -> Path:
    """Resolve a campaign-relative path and prove it stayed inside the campaign.

    The plan is signed, but a symlink planted in the tree could still redirect
    a listed path outside it, so resolution is checked rather than trusted.
    """
    resolved = (root / relative).resolve()
    if resolved != root and root not in resolved.parents:
        raise lora.CartridgeError(f"{where} {relative!r} resolves outside {root}")
    return resolved


def load_assembly(
    root: Path, label: str, *, trust: str | None
) -> tuple[dict[str, Any], Path]:
    """Load one published assembly plan, verified under a pinned key."""
    if not lora.LABEL_RE.fullmatch(label):
        raise lora.CartridgeError(
            f"--assembly {label!r} must match {lora.LABEL_RE.pattern}; it names "
            f"a directory inside the campaign"
        )
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
                f"{signed}: signed by {keyid[:16]}, not the pinned {trust[:16]}"
            )
    if not isinstance(plan, dict) or plan.get("schema") != lora.ASSEMBLY_SCHEMA:
        raise lora.CartridgeError(f"{signed}: schema must be {lora.ASSEMBLY_SCHEMA!r}")
    if set(plan) != ASSEMBLY_FIELDS:
        raise lora.CartridgeError(
            f"{signed}: closed {lora.ASSEMBLY_SCHEMA} fields differ; "
            f"missing={sorted(ASSEMBLY_FIELDS - set(plan))}, "
            f"unexpected={sorted(set(plan) - ASSEMBLY_FIELDS)}"
        )
    if plan.get("format") != "exl3-msrt-packed":
        raise lora.CartridgeError(
            f"{signed}: unsupported format {plan.get('format')!r}"
        )
    if plan.get("standard_lora_compatible") is not False:
        raise lora.CartridgeError(f"{signed}: standard_lora_compatible must be false")
    if plan.get("assembly") != label:
        raise lora.CartridgeError(
            f"{signed}: plan describes {plan.get('assembly')!r}, not {label!r}"
        )
    for field, expected in (
        ("runtime_profile", lora.RUNTIME_PROFILE),
        ("rotation_ownership", "base"),
        ("codebook", "mcg"),
        ("mcg_ownership", "adapter-config"),
        ("mcg_multiplier", lora.MCG_MULTIPLIER),
        ("scale_shape", []),
        ("paths_relative_to", "campaign root"),
    ):
        if plan.get(field) != expected:
            raise lora.CartridgeError(
                f"{signed}: {field} is {plan.get(field)!r}, expected "
                f"{expected!r}; refusing to republish an altered runtime "
                f"contract"
            )
    base = plan.get("base")
    if (
        not isinstance(base, dict)
        or set(base)
        != {
            "label",
            "k",
            "manifest_sha256",
            "compatibility_sha256",
            "compatibility_by_layer",
        }
        or not valid_sha256(base.get("manifest_sha256"))
        or not valid_sha256(base.get("compatibility_sha256"))
        or not valid_label(base.get("label"))
    ):
        raise lora.CartridgeError(f"{signed}: invalid base checkpoint identity")
    compatibility_by_layer = base.get("compatibility_by_layer")
    if (
        not isinstance(compatibility_by_layer, dict)
        or not compatibility_by_layer
        or any(
            not isinstance(layer, str)
            or re.fullmatch(r"(0|[1-9][0-9]*)", layer) is None
            or not valid_sha256(digest)
            for layer, digest in compatibility_by_layer.items()
        )
    ):
        raise lora.CartridgeError(f"{signed}: invalid per-layer base identities")
    lora._validate_base_k(base.get("k"), who="base k")
    tensor_parallel = plan.get("tensor_parallel")
    if tensor_parallel != tensor_parallel_contract("full", 1):
        raise lora.CartridgeError(
            f"{signed}: tensor_parallel must describe the campaign's full "
            "logical rank0 artifact"
        )
    bits_per_weight = plan.get("bits_per_weight")
    if (
        isinstance(bits_per_weight, bool)
        or not isinstance(bits_per_weight, (int, float))
        or not math.isfinite(float(bits_per_weight))
        or bits_per_weight <= 0
        or isinstance(plan.get("num_tensors"), bool)
        or not isinstance(plan.get("num_tensors"), int)
        or plan["num_tensors"] < 1
        or not isinstance(plan.get("tool_version"), str)
        or re.fullmatch(r"fq_[a-z_]+/[0-9]+", plan["tool_version"]) is None
        or not valid_rfc3339(plan.get("created_utc"))
    ):
        raise lora.CartridgeError(f"{signed}: invalid assembly accounting metadata")
    chain = plan.get("chain")
    if not isinstance(chain, list) or not chain:
        raise lora.CartridgeError(
            f"{signed}: this product applies no residual stage, so it is a base "
            f"checkpoint, not a cartridge; load base/{base.get('label')} "
            f"directly"
        )
    expected_parent = base["label"]
    parent_experts: str | list[int] = "all"
    seen: set[str] = set()
    for stage in chain:
        if not isinstance(stage, dict) or set(stage) != {
            "label",
            "k",
            "parent",
            "experts",
        }:
            raise lora.CartridgeError(f"{signed}: chain entries must be objects")
        stage_label = str(stage.get("label"))
        if not lora.LABEL_RE.fullmatch(stage_label) or stage_label in seen:
            raise lora.CartridgeError(
                f"{signed}: invalid or duplicated stage label {stage_label!r}"
            )
        seen.add(stage_label)
        lora._validate_k(stage.get("k"), who=f"stage {stage_label!r} k")
        if stage.get("parent") != expected_parent:
            raise lora.CartridgeError(
                f"{signed}: stage {stage_label!r} corrects "
                f"{stage.get('parent')!r}, not {expected_parent!r}; the plan is "
                f"not one path through the recipe graph"
            )
        experts = stage.get("experts")
        if experts != "all" and (
            not isinstance(experts, list)
            or not experts
            or any(
                isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in experts
            )
            or len(set(experts)) != len(experts)
        ):
            raise lora.CartridgeError(
                f"{signed}: stage {stage_label!r} has an invalid expert set"
            )
        if not lora._covers(parent_experts, experts):
            raise lora.CartridgeError(
                f"{signed}: stage {stage_label!r} expert coverage is not a "
                "subset of its parent"
            )
        parent_experts = experts
        expected_parent = stage_label

    campaign = plan.get("campaign")
    if (
        not isinstance(campaign, dict)
        or set(campaign) != CAMPAIGN_FIELDS
        or not valid_sha256(campaign.get("recipe_sha256"))
        or not isinstance(campaign.get("base_model"), str)
        or not campaign["base_model"]
        or not isinstance(campaign.get("base_revision"), str)
        or not campaign["base_revision"]
        or not valid_sha256(campaign.get("signer_pubkey"))
        or (
            campaign.get("encoder_sha256") is not None
            and not valid_sha256(campaign["encoder_sha256"])
        )
        or isinstance(campaign.get("block_size"), bool)
        or not isinstance(campaign.get("block_size"), int)
        or campaign["block_size"] < 1
        or not isinstance(campaign.get("moe_layers"), list)
        or not campaign["moe_layers"]
        or any(
            isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
            for layer in campaign["moe_layers"]
        )
        or campaign["moe_layers"] != sorted(set(campaign["moe_layers"]))
        or set(compatibility_by_layer)
        != {str(layer) for layer in campaign.get("moe_layers", [])}
    ):
        raise lora.CartridgeError(f"{signed}: plan lacks a campaign identity")
    if trust is not None and campaign["signer_pubkey"] != trust:
        raise lora.CartridgeError(
            f"{signed}: campaign names signer {campaign['signer_pubkey'][:16]}, "
            f"not the pinned {trust[:16]}"
        )
    stage_shards = plan.get("stage_shards")
    if not isinstance(stage_shards, list) or not stage_shards:
        raise lora.CartridgeError(f"{signed}: stage_shards must be a non-empty list")
    parent_of = {stage["label"]: stage["parent"] for stage in chain}
    unique: set[tuple[str, str]] = set()
    for entry in stage_shards:
        if (
            not isinstance(entry, dict)
            or set(entry) != STAGE_SHARD_FIELDS
            or entry.get("label") not in seen
        ):
            raise lora.CartridgeError(f"{signed}: malformed entry {entry!r}")
        path = safe_relative(entry.get("path"), where=str(signed))
        safe_relative(entry.get("attestation"), where=str(signed))
        if not valid_sha256(entry.get("sha256")):
            raise lora.CartridgeError(f"{signed}: entry {entry!r} lacks a sha256")
        experts = entry.get("experts")
        if (
            not isinstance(experts, list)
            or not experts
            or any(
                isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in experts
            )
            or len(set(experts)) != len(experts)
        ):
            raise lora.CartridgeError(
                f"{signed}: entry {entry['path']!r} has an invalid expert list"
            )
        if (
            isinstance(entry.get("layer"), bool)
            or not isinstance(entry.get("layer"), int)
            or not 0 <= entry["layer"] <= lora.MAX_LAYER
            or isinstance(entry.get("block"), bool)
            or not isinstance(entry.get("block"), int)
            or entry["block"] < 0
        ):
            raise lora.CartridgeError(
                f"{signed}: entry {entry['path']!r} lacks (layer, block)"
            )
        if (
            not valid_sha256(entry.get("parent_sha256"))
            or entry.get("parent_label") != parent_of[entry["label"]]
        ):
            raise lora.CartridgeError(
                f"{signed}: entry {entry['path']!r} does not name the parent "
                f"fragment its residual corrects"
            )
        match = BLOCK_NAME_RE.fullmatch(path.name)
        if match is None:
            raise lora.CartridgeError(
                f"{signed}: {path.name!r} is not a campaign block name"
            )
        if (int(match.group(1)), int(match.group(2))) != (
            entry["layer"],
            entry["block"],
        ):
            raise lora.CartridgeError(
                f"{signed}: {path.name!r} disagrees with its declared (layer, block)"
            )
        if path.parts[:2] != ("stages", entry["label"]):
            raise lora.CartridgeError(
                f"{signed}: {entry['path']!r} does not live under "
                f"stages/{entry['label']}"
            )
        key = (entry["label"], path.name)
        if key in unique:
            raise lora.CartridgeError(
                f"{signed}: {entry['label']}/{path.name} is listed twice"
            )
        unique.add(key)
    published_paths = [entry["path"] for entry in stage_shards]
    if plan.get("shards") != published_paths or len(set(published_paths)) != len(
        published_paths
    ):
        raise lora.CartridgeError(
            f"{signed}: shards must exactly list stage_shards paths in order"
        )
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
                f"chain is not internally consistent"
            )
    header_tensor_count = 0
    available_stage_shards = 0
    for entry in stage_shards:
        source = under_root(root, entry["path"], where="stage shard")
        if not source.is_file():
            continue  # Sparse campaign downloads are allowed until selection.
        available_stage_shards += 1
        header = read_bounded_safetensors_header(source)
        header.pop("__metadata__", None)
        header_tensor_count += len(header)
    if (
        available_stage_shards == len(stage_shards)
        and header_tensor_count != plan["num_tensors"]
    ):
        raise lora.CartridgeError(
            f"{signed}: num_tensors={plan['num_tensors']} but the published "
            f"stage shard headers contain {header_tensor_count}"
        )
    return plan, signed


def stage_regular_file(source: Path, destination: Path) -> tuple[str, int]:
    """Copy one source inode into private storage while hashing that same read."""
    requested = source.absolute()
    resolved = source.resolve()
    if requested != resolved or source.is_symlink():
        raise lora.CartridgeError(
            f"{source}: campaign stage paths must not use symlinks"
        )
    source_fd = destination_fd = None
    try:
        source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(source_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or not 0 < before.st_size <= MAX_STAGE_SOURCE_BYTES
        ):
            raise lora.CartridgeError(
                f"{source}: must be a non-empty regular file at most "
                f"{MAX_STAGE_SOURCE_BYTES} bytes"
            )
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(source_fd, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise OSError("short write while staging campaign shard")
                view = view[written:]
            copied += len(chunk)
        after = os.fstat(source_fd)
        if copied != before.st_size or (before.st_dev, before.st_ino) != (
            after.st_dev,
            after.st_ino,
        ):
            raise lora.CartridgeError(f"{source}: changed while it was staged")
        os.fsync(destination_fd)
        return digest.hexdigest(), copied
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        if source_fd is not None:
            os.close(source_fd)


def read_block(
    root: Path,
    entry: dict[str, Any],
    staging_dir: Path,
    *,
    trust: str | None,
    campaign: dict[str, Any],
) -> tuple[dict[str, str], Path]:
    """Verify one *selected* stage shard against its digest and attestation.

    Only shards the selection actually needs are required to be present, so a
    consumer who wants 96 experts of one layer never has to download, keep or
    hash the rest of the campaign.
    """
    source = under_root(root, entry["path"], where="stage shard")
    requested_source = root / entry["path"]
    if not source.is_file():
        raise lora.CartridgeError(
            f"missing stage shard {entry['path']} (needed by this selection)"
        )
    token = hashlib.sha256(entry["path"].encode("utf-8")).hexdigest()[:20]
    staged = staging_dir / f".verified-source-{token}.safetensors"
    copied_sha, _copied_size = stage_regular_file(requested_source, staged)
    if copied_sha != entry["sha256"]:
        staged.unlink(missing_ok=True)
        raise lora.CartridgeError(
            f"{entry['path']}: sha256 {copied_sha} != published "
            f"{entry['sha256']}; refusing to combine altered bytes"
        )
    read_bounded_safetensors_header(staged)
    sha, _body, spans = lora.verify_shard(staged, group_kind="expert")
    if sha != entry["sha256"]:
        staged.unlink(missing_ok=True)
        raise lora.CartridgeError(
            f"{entry['path']}: sha256 {sha} != published {entry['sha256']}; "
            f"refusing to combine altered bytes"
        )
    header, _ = lora.read_header(staged)
    meta = header.get("__metadata__") or {}
    if meta.get("schema") != lora.BLOCK_SCHEMA:
        raise lora.CartridgeError(
            f"{source}: metadata schema is {meta.get('schema')!r}, expected "
            f"{lora.BLOCK_SCHEMA!r}; this is not an encoded MSRT block"
        )
    for field in ("label", "layer", "block", "covered_experts"):
        if field not in meta:
            raise lora.CartridgeError(f"{source}: block metadata lacks {field}")
    if meta["label"] != entry["label"]:
        raise lora.CartridgeError(
            f"{source}: metadata label {meta['label']!r} != plan {entry['label']!r}"
        )
    match = BLOCK_NAME_RE.fullmatch(Path(entry["path"]).name)
    if (int(match.group(1)), int(match.group(2))) != (
        int(meta["layer"]),
        int(meta["block"]),
    ):
        raise lora.CartridgeError(
            f"{source}: file name disagrees with its recorded (layer, block)"
        )
    if trust is not None:
        payload = lora.read_signed_line(
            under_root(root, entry["attestation"], where="attestation")
        )
        if payload.pop("_keyid") != trust:
            raise lora.CartridgeError(
                f"{entry['attestation']}: not signed by the pinned key"
            )
        fragment = payload.get("fragment") or {}
        if (
            fragment.get("sha256") != sha
            or fragment.get("file") != Path(entry["path"]).name
            or payload.get("expert_sha256") != spans
        ):
            raise lora.CartridgeError(
                f"{entry['attestation']}: does not attest these bytes"
            )
        if payload.get("predicate") != "encode-of":
            raise lora.CartridgeError(
                f"{entry['attestation']}: predicate is not encode-of"
            )
        claim = (payload.get("parents") or [None])[0]
        if (
            not isinstance(claim, dict)
            or claim.get("label") != entry["parent_label"]
            or claim.get("sha256") != entry["parent_sha256"]
        ):
            raise lora.CartridgeError(
                f"{entry['attestation']}: attests parent {claim!r}, not the "
                f"{entry['parent_label']} bytes the plan names"
            )
        for field, value in (
            ("recipe_sha256", campaign["recipe_sha256"]),
            ("base_revision", campaign["base_revision"]),
        ):
            if payload.get(field) != value:
                raise lora.CartridgeError(
                    f"{entry['attestation']}: {field} disagrees with the "
                    f"campaign identity in the signed plan"
                )
    covered = {int(value) for value in meta["covered_experts"].split(",") if value}
    if covered != set(entry["experts"]):
        raise lora.CartridgeError(
            f"{source}: carries experts {sorted(covered)} but the signed plan "
            f"promises {sorted(entry['experts'])}"
        )
    return meta, staged


def validate_stage_tensors(
    tensors: dict[str, Any],
    stage: dict[str, Any],
    expected: set[int],
    layer: int,
    shard: str,
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
            int(match.group(1)),
            int(match.group(2)),
            match.group(3),
            int(match.group(4)),
            match.group(5),
            match.group(6),
        )
        if key_label != label:
            raise lora.CartridgeError(
                f"{shard}: tensor {key!r} carries label {key_label!r}, "
                f"expected {label!r}"
            )
        if key_layer != layer:
            raise lora.CartridgeError(
                f"{shard}: tensor {key!r} belongs to layer {key_layer}, "
                f"expected {layer}"
            )
        if rank != 0:
            raise lora.CartridgeError(
                f"{shard}: {key} is rank {rank}; this format is rank0 only"
            )
        groups.setdefault((key_layer, expert, projection, rank), set()).add(component)
        observed.add(expert)
        shapes.setdefault((expert, projection), {})[component] = tensor
        if component == "trellis":
            if (
                tensor.ndim != 3
                or tensor.shape[-1] != stage["k"] * 16
                or tensor.dtype is not torch.int16
            ):
                raise lora.CartridgeError(
                    f"{shard}: {key} is {tuple(tensor.shape)} {tensor.dtype}, "
                    f"expected a 3-D int16 trellis whose last dimension is "
                    f"{stage['k'] * 16}"
                )
        elif component == "scale":
            if tensor.ndim != 0 or tensor.dtype is not torch.float32:
                raise lora.CartridgeError(
                    f"{shard}: {key} must be a float32 scalar, got "
                    f"{tuple(tensor.shape)} {tensor.dtype}"
                )
            value = float(tensor)
            if not math.isfinite(value) or value <= 0:
                raise lora.CartridgeError(
                    f"{shard}: {key} is {value}; a residual rescale factor must "
                    f"be finite and positive or the runtime inverts the stage"
                )
    if observed != expected:
        raise lora.CartridgeError(
            f"{shard}: experts {sorted(observed)} != selected {sorted(expected)}"
        )
    for group, components in groups.items():
        if components != COMPONENTS:
            raise lora.CartridgeError(
                f"{shard}: {group} components {sorted(components)} != "
                f"{sorted(COMPONENTS)}"
            )
    ranks = {group[3] for group in groups}
    for expert in expected:
        missing = {
            (expert, projection, rank)
            for projection in lora.PROJECTIONS
            for rank in ranks
            if (layer, expert, projection, rank) not in groups
        }
        if missing:
            raise lora.CartridgeError(
                f"{shard}: incomplete projection/rank coverage for "
                f"{sorted(missing)[:3]}"
            )


def partition_stage_tensors(
    tensors: dict[str, Any], *, layout: str, world_size: int, shard: str
) -> dict[str, Any]:
    """Materialize the manifest-declared full or fixed-rank TP namespace."""
    if layout == "full":
        tensor_parallel_contract(layout, world_size)
        return tensors
    tensor_parallel_contract(layout, world_size)
    partitioned: dict[str, Any] = {}
    tile_alignment = lora.HADAMARD_BLOCK // 16
    for key, tensor in tensors.items():
        match = STAGE_KEY_RE.fullmatch(key)
        if match is None or int(match.group(4)) != 0:
            raise lora.CartridgeError(
                f"{shard}: rank sharding requires a full logical rank0 source"
            )
        projection, component = match.group(3), match.group(5)
        dim = 0 if projection == "down_proj" else 1
        if component == "trellis":
            tiles = tensor.shape[dim]
            if tiles % (world_size * tile_alignment):
                axis = lora.TP_AXIS_BY_PROJECTION[projection]
                raise lora.CartridgeError(
                    f"{shard}: {key} {axis} size {tiles * 16} cannot be split "
                    f"into {world_size} {lora.HADAMARD_BLOCK}-aligned TP ranks"
                )
            per_rank = tiles // world_size
        for rank in range(world_size):
            start, stop = match.span(4)
            ranked_key = f"{key[:start]}{rank}{key[stop:]}"
            if component == "trellis":
                value = tensor.narrow(dim, rank * per_rank, per_rank).contiguous()
            else:
                # Scales are per logical projection/stage and therefore
                # replicated; they are never sliced or inverted by TP.
                value = tensor.clone()
            partitioned[ranked_key] = value
    return partitioned


def verify_base(
    base_dir: Path | None, plan: dict[str, Any], root: Path, *, trust: str | None
) -> None:
    """Verify the complete physical base and its TP-invariant logical identity."""
    del root
    if base_dir is None:
        return
    base_dir = base_dir.expanduser().resolve()
    manifest = base_dir / "MANIFEST.sha256"
    try:
        descriptor = os.open(manifest, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_MANIFEST_BYTES:
                raise lora.CartridgeError(
                    f"{manifest}: must be a regular file at most "
                    f"{MAX_MANIFEST_BYTES} bytes"
                )
            manifest_bytes = bytearray()
            while len(manifest_bytes) <= MAX_MANIFEST_BYTES:
                chunk = os.read(
                    descriptor,
                    min(1 << 20, MAX_MANIFEST_BYTES + 1 - len(manifest_bytes)),
                )
                if not chunk:
                    break
                manifest_bytes.extend(chunk)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise lora.CartridgeError(
            f"--base {base_dir} has no safe readable MANIFEST.sha256"
        ) from exc
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise lora.CartridgeError(f"{manifest}: exceeds the manifest size limit")
    actual = hashlib.sha256(manifest_bytes).hexdigest()
    if actual != plan["base"]["manifest_sha256"]:
        raise lora.CartridgeError(
            f"--base {base_dir} manifest {actual[:16]} is not the "
            f"{plan['base']['manifest_sha256'][:16]} this cartridge was built "
            f"against; loading it would apply residuals to other weights"
        )

    try:
        manifest_text = bytes(manifest_bytes).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise lora.CartridgeError(f"{manifest}: is not UTF-8") from exc
    entries: dict[str, str] = {}
    for number, line in enumerate(manifest_text.splitlines(), 1):
        digest, separator, relative_text = line.partition("  ")
        relative = Path(relative_text)
        if (
            not separator
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or not relative_text
            or relative.is_absolute()
            or ".." in relative.parts
            or relative_text in entries
        ):
            raise lora.CartridgeError(
                f"{manifest}:{number}: invalid sha256sum manifest entry"
            )
        entries[relative_text] = digest
    present = {
        path.relative_to(base_dir).as_posix()
        for path in base_dir.rglob("*")
        if path.is_file() and path != manifest
    }
    symlinks = [
        path.relative_to(base_dir).as_posix()
        for path in base_dir.rglob("*")
        if path.is_symlink()
    ]
    if symlinks:
        raise lora.CartridgeError(
            f"{manifest}: base contains symlinks, e.g. {sorted(symlinks)[:5]}"
        )
    if present != set(entries):
        raise lora.CartridgeError(
            f"{manifest}: physical base files differ; "
            f"missing={sorted(set(entries) - present)[:5]}, "
            f"unlisted={sorted(present - set(entries))[:5]}"
        )

    logical_records: list[dict[str, Any]] = []
    verified_blocks: dict[str, lora.VerifiedShard] = {}
    for relative_text, expected_sha in sorted(entries.items()):
        target = (base_dir / relative_text).resolve()
        if base_dir != target and base_dir not in target.parents:
            raise lora.CartridgeError(
                f"{manifest}: {relative_text!r} resolves outside the base"
            )
        if not target.is_file():
            raise lora.CartridgeError(f"{manifest}: missing {relative_text}")
        if (base_dir / relative_text).is_symlink():
            raise lora.CartridgeError(f"{manifest}: {relative_text} is a symlink")
        if BLOCK_NAME_RE.fullmatch(relative_text):
            read_bounded_safetensors_header(target)
            verified = lora.verify_shard_details(target, group_kind="expert")
            found_sha = verified.sha256
            verified_blocks[relative_text] = verified
            logical_records.extend(verified.tensors)
        else:
            found_sha = sha256_file(target)
        if found_sha != expected_sha:
            raise lora.CartridgeError(
                f"{target}: sha256 {found_sha[:16]} does not match MANIFEST.sha256"
            )

    try:
        base_config = json.loads((base_dir / "config.json").read_text())
        tail = base_config["hybrid_tr3_tail"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise lora.CartridgeError(
            f"--base {base_dir} has invalid EXL3 metadata"
        ) from exc
    bits = tail.get("bits") if isinstance(tail, dict) else None
    base_bits_match = (
        not isinstance(bits, bool)
        and isinstance(bits, (int, float))
        and math.isfinite(float(bits))
        and float(bits) == plan["base"]["k"]
    )
    if (
        not isinstance(tail, dict)
        or set(tail) != BASE_TAIL_FIELDS
        or tail.get("format") != "exl3-trellis"
        or tail.get("runtime_profile") != lora.BASE_RUNTIME_PROFILE
        or tail.get("codebook") != "mcg"
        or tail.get("tensor_schema") != BASE_TENSOR_SCHEMA
        or tail.get("tensor_parallel") != lora.base_tensor_parallel_contract()
        or tail.get("mcg_multiplier") != lora.MCG_MULTIPLIER
        or tail.get("compatibility_sha256") != plan["base"]["compatibility_sha256"]
        or not base_bits_match
        or isinstance(tail.get("experts_per_layer"), bool)
        or not isinstance(tail.get("experts_per_layer"), int)
        or tail["experts_per_layer"] < 1
        or not isinstance(tail.get("moe_layer_coverage"), list)
        or not tail["moe_layer_coverage"]
        or any(
            isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
            for layer in tail["moe_layer_coverage"]
        )
        or tail["moe_layer_coverage"] != sorted(set(tail["moe_layer_coverage"]))
        or tail.get("moe_layers")
        != [min(tail["moe_layer_coverage"]), max(tail["moe_layer_coverage"])]
        or tail["moe_layer_coverage"]
        != sorted(int(layer) for layer in plan["base"]["compatibility_by_layer"])
    ):
        raise lora.CartridgeError(
            f"--base {base_dir} metadata does not match the assembly base contract"
        )
    compatibility, compatibility_by_layer = lora.base_compatibility_identity(
        plan["base"]["k"], logical_records
    )
    if compatibility != plan["base"]["compatibility_sha256"]:
        raise lora.CartridgeError(
            f"--base {base_dir} logical tensor identity {compatibility[:16]} is not "
            f"the {plan['base']['compatibility_sha256'][:16]} this cartridge targets"
        )
    if (
        compatibility_by_layer != plan["base"]["compatibility_by_layer"]
        or tail.get("compatibility_by_layer") != compatibility_by_layer
    ):
        raise lora.CartridgeError(
            f"--base {base_dir} per-layer logical identities do not match the "
            "assembly and checkpoint metadata"
        )

    for name, verified in sorted(verified_blocks.items()):
        published = lora.read_digest(base_dir, name)
        if published != verified.sha256:
            raise lora.CartridgeError(
                f"--base {base_dir} has no valid committed digest for {name}"
            )
        payload = lora.read_attestation(base_dir, name)
        signer = payload.pop("_keyid")
        fragment = payload.get("fragment") or {}
        if (
            fragment.get("file") != name
            or fragment.get("sha256") != verified.sha256
            or fragment.get("size") != (base_dir / name).stat().st_size
            or fragment.get("body_offset") != verified.body_offset
            or payload.get("expert_sha256") != verified.group_sha256
        ):
            raise lora.CartridgeError(
                f"{base_dir}/{name}: base attestation does not describe loaded bytes"
            )
        if trust is not None and signer != trust:
            raise lora.CartridgeError(
                f"{base_dir}/{name}: base provenance is not signed by the pinned key"
            )

    first = plan["chain"][0]["label"]
    for entry in plan["stage_shards"]:
        if entry["label"] != first:
            continue
        name = Path(entry["path"]).name
        verified = verified_blocks.get(name)
        if verified is None or verified.sha256 != entry["parent_sha256"]:
            found = None if verified is None else verified.sha256[:16]
            raise lora.CartridgeError(
                f"{entry['path']} corrects base bytes "
                f"{entry['parent_sha256'][:16]}, but the loaded base has {found}"
            )


def combine(args) -> int:
    lora.require_quant_dependencies()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise lora.CartridgeError(f"--root {root} is not a directory")
    trust = resolve_trust(args.trust_key)
    if trust is None and not args.insecure_unsigned:
        raise lora.CartridgeError(
            "pass --trust-key <64-hex public key> to check the campaign's "
            "signatures, or --insecure-unsigned to combine unverified bytes"
        )
    plan, plan_path = load_assembly(root, args.assembly, trust=trust)
    verify_base(args.base, plan, root, trust=trust)
    tp_layout = getattr(args, "tp_layout", "full")
    tp_world_size = getattr(args, "tp_world_size", 1)
    tensor_parallel = tensor_parallel_contract(tp_layout, tp_world_size)
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
    shards: list[dict[str, Any]] = []
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
                keep = (
                    set(entry["experts"])
                    if wanted_experts is None
                    else set(entry["experts"]) & wanted_experts
                )
                if not keep:
                    continue
                _meta, verified_source = read_block(
                    root,
                    entry,
                    work,
                    trust=trust,
                    campaign=plan["campaign"],
                )
                source_name = entry["path"]
                try:
                    with safe_open(str(verified_source), framework="pt") as handle:
                        tensors = {}
                        for key in tuple(handle.keys()):
                            match = STAGE_KEY_RE.fullmatch(key)
                            if match is None:
                                raise lora.CartridgeError(
                                    f"{source_name}: unexpected tensor key {key!r}"
                                )
                            if int(match.group(2)) in keep:
                                tensors[key] = handle.get_tensor(key)
                    validate_stage_tensors(
                        tensors,
                        stage_by_label[entry["label"]],
                        keep,
                        layer,
                        source_name,
                    )
                    tensors = partition_stage_tensors(
                        tensors,
                        layout=tp_layout,
                        world_size=tp_world_size,
                        shard=source_name,
                    )
                finally:
                    verified_source.unlink(missing_ok=True)
                collisions = set(combined) & set(tensors)
                if collisions:
                    raise lora.CartridgeError(
                        f"{source_name}: duplicate tensor keys {sorted(collisions)[:3]}"
                    )
                combined.update(tensors)
                by_layer = coverage.setdefault(entry["label"], {})
                by_layer[str(layer)] = sorted(set(by_layer.get(str(layer), [])) | keep)
                requested_seen |= keep
            if not combined:
                continue
            lora.save_shard_tensors(combined, work, basename)
            shard_size = (work / basename).stat().st_size
            if shard_size > MAX_ADAPTER_SHARD_BYTES:
                raise lora.CartridgeError(
                    f"{basename}: {shard_size} bytes exceeds the runtime's "
                    f"{MAX_ADAPTER_SHARD_BYTES}-byte shard limit"
                )
            shards.append(
                {
                    "path": basename,
                    "size": shard_size,
                    "sha256": sha256_file(work / basename),
                }
            )
            tensor_count += len(combined)
        if not shards:
            raise lora.CartridgeError(
                "the selection matched no experts; nothing to combine"
            )
        total_shard_bytes = sum(shard["size"] for shard in shards)
        if total_shard_bytes > MAX_ADAPTER_TOTAL_BYTES:
            raise lora.CartridgeError(
                f"adapter shards total {total_shard_bytes} bytes, exceeding the "
                f"runtime's {MAX_ADAPTER_TOTAL_BYTES}-byte limit"
            )
        if wanted_experts is not None and wanted_experts - requested_seen:
            raise lora.CartridgeError(
                f"experts {sorted(wanted_experts - requested_seen)[:8]} are not "
                f"covered by assembly {args.assembly!r}; the emitted contract "
                f"would silently promise less than you asked for"
            )
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
                        f"{sorted(gap)[:8]} in assembly {args.assembly!r}"
                    )
        # Every stage of the chain must survive the narrowing, or the product
        # is not the product its own chain describes.
        for stage in plan["chain"]:
            if not coverage.get(stage["label"]):
                raise lora.CartridgeError(
                    f"stage {stage['label']!r} contributed no tensors after "
                    f"narrowing; this selection cannot form {args.assembly!r}"
                )
        # Coverage is part of the executable chain contract, not descriptive
        # metadata: a child residual cannot exist where its parent residual is
        # absent. Re-check it after layer/expert narrowing and shard reads.
        for parent_stage, child_stage in zip(plan["chain"], plan["chain"][1:]):
            parent_layers = coverage[parent_stage["label"]]
            child_layers = coverage[child_stage["label"]]
            for layer_id, child_experts in child_layers.items():
                if not set(child_experts) <= set(parent_layers.get(layer_id, [])):
                    raise lora.CartridgeError(
                        f"stage {child_stage['label']!r} coverage at layer "
                        f"{layer_id} is not a subset of parent "
                        f"{parent_stage['label']!r}"
                    )
        layers_seen = sorted(
            {int(layer) for by_layer in coverage.values() for layer in by_layer}
        )
        if wanted_layers is not None and wanted_layers - set(layers_seen):
            raise lora.CartridgeError(
                f"layers {sorted(wanted_layers - set(layers_seen))} are not "
                f"covered by assembly {args.assembly!r}"
            )
        adapter_chain = []
        for stage in plan["chain"]:
            emitted = sorted(
                {
                    expert
                    for experts in coverage[stage["label"]].values()
                    for expert in experts
                }
            )
            adapter_chain.append({**stage, "experts": emitted})
        config = {
            "schema": lora.ADAPTER_CONFIG_SCHEMA,
            "assembly": plan["assembly"],
            # Physical MANIFEST pinning belongs to the signed assembly and
            # producer-side --base verification. Runtime compatibility is the
            # logical byte identity it can actually compare to loaded weights.
            "base": {
                "label": plan["base"]["label"],
                "k": plan["base"]["k"],
                "compatibility_sha256": plan["base"]["compatibility_sha256"],
                "compatibility_by_layer": {
                    str(layer): plan["base"]["compatibility_by_layer"][str(layer)]
                    for layer in layers_seen
                },
            },
            # Assembly plans describe recipe intent; adapter manifests
            # describe the exact narrowed artifact. Keep this stage expert
            # union in lockstep with coverage so the runtime never has to
            # reinterpret "all" relative to an external model.
            "chain": adapter_chain,
            # Re-derived here, never copied from the plan: a tampered codebook
            # or multiplier would decode every weight wrongly.
            "format": "exl3-msrt-packed",
            "runtime_profile": lora.RUNTIME_PROFILE,
            "rotation_ownership": "base",
            "standard_lora_compatible": False,
            "runtime_operation": lora.RUNTIME_OPERATION,
            "codebook": "mcg",
            "mcg_multiplier": lora.MCG_MULTIPLIER,
            "mcg_ownership": "adapter-config",
            "scale_shape": [],
            "tensor_parallel": tensor_parallel,
            "shards": shards,
            "num_tensors": tensor_count,
            "selected_experts": sorted(requested_seen),
            "selected_layers": layers_seen,
            "coverage": coverage,
            # Provenance from the producer-side verification transaction. The
            # runtime checks shard hashes but does not authenticate this key.
            "producer_verified_signer": trust,
            "campaign": plan["campaign"],
            "source_assembly": {
                "path": plan_path.relative_to(root).as_posix(),
                "sha256": sha256_file(plan_path),
            },
            "tool_version": TOOL_VERSION,
            "created_utc": lora.now_utc(),
        }
        config_bytes = (json.dumps(config, indent=2) + "\n").encode("utf-8")
        if len(config_bytes) > MAX_ADAPTER_CONFIG_BYTES:
            raise lora.CartridgeError(
                f"adapter_config.json is {len(config_bytes)} bytes, exceeding the "
                f"runtime's {MAX_ADAPTER_CONFIG_BYTES}-byte limit"
            )
        (work / "adapter_config.json").write_bytes(config_bytes)
        staged.commit()
    except Exception:
        staged.abort()
        raise
    print(
        f"Combined {len(shards)} shards, {tensor_count} tensors, "
        f"{len(requested_seen)} experts across {len(layers_seen)} layers "
        f"into {out}",
        flush=True,
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Campaign root written by fq_assemble_lora",
    )
    parser.add_argument(
        "--assembly", required=True, help="Assembly label published under assemblies/"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--experts", help="Restrict to these expert ids, e.g. 0-95,200")
    parser.add_argument("--layers", help="Restrict to these layers, e.g. 3-40")
    parser.add_argument(
        "--tp-layout",
        choices=("full", "rank-sharded"),
        default="full",
        help=(
            "Store one full logical rank for runtime slicing, or materialize "
            "a fixed rank-sharded tensor namespace"
        ),
    )
    parser.add_argument(
        "--tp-world-size",
        type=int,
        default=1,
        help="Physical TP rank count for --tp-layout rank-sharded (including 1)",
    )
    parser.add_argument(
        "--base",
        type=Path,
        help="Base checkpoint directory to verify this "
        "cartridge against (manifest digest and the first "
        "chain edge)",
    )
    parser.add_argument(
        "--trust-key",
        help="Pinned ed25519 public key (hex or file) that must "
        "have signed the plan and every fragment",
    )
    parser.add_argument(
        "--insecure-unsigned",
        action="store_true",
        help="Combine without checking any signature",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        return combine(args)
    except (AssemblyError, lora.CartridgeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
