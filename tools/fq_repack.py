#!/usr/bin/env python3
"""fq_repack — repack a per-layer-sharded EXL3 quant into Progressive Tensors segments.

Scales up poc/poc_slice.py per implementation/10 & 11: for each source layer
shard, extract the routed-expert tensors verbatim (predicate: repack-of, no
re-encode) into a per-layer segment file (pure safetensors) whose body is
per-expert contiguous and therefore range-readable per expert, plus a signed
attestation line and a per-expert byte-range index.

v1 layout keeps the source's rank-sliced granularity verbatim
(layout=rank_sliced_tp4); unsharding is a later, T4-verified upgrade.

Multi-K safety (fq-repack-state/2)
----------------------------------
One output dir holds many Ks — a K3 run, then a K4 run, then a K3 rerun, all
into the same segments dir.  State is therefore keyed by
{source fingerprint, K, layer}, never by layer alone: a K4 run can never see a
K3 layer as "done" (the v1 bug that made index-k4.json point at K3 bytes), and
every index-kK.json is rebuilt from the state entries belonging to THAT K —
one K's run never rewrites another K's index.  fq-manifest.json is likewise
merged, not overwritten: `k_variants` lists every K present and `per_k[K]`
carries that K's index file, layer coverage, segment count and provenance.
An output dir whose manifest describes a different base_model/layout is
refused outright; re-repacking a K from a different source repo/revision
needs --allow-provenance-change.

Source binding: the source shard's sha256 is COMPUTED here (never taken on
faith from the source repo's MANIFEST.sha256); when that manifest exists its
value is cross-checked and any disagreement is a hard failure.

Resumable per {source, K, layer} via <out>/state.json; optional incremental
HF publish.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mmap
import os
import re
import struct
import sys
import time
from pathlib import Path

PROJ_ORDER = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}
COMP_ORDER = {"trellis": 0, "suh": 1, "svh": 2, "mcg": 3, "mul1": 3}
EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+_proj)\.rank(\d+)\.(\w+)$"
)
SEGMENT_SCHEMA = "fq-segment/1"
ATTESTATION_SCHEMA = "fq-attestation/1"
MANIFEST_SCHEMA = "fq-manifest/1"
STATE_SCHEMA = "fq-repack-state/2"
TOOL_VERSION = "fq_repack/2"
DEFAULT_LAYOUT = "rank_sliced_tp4"
SEG_NAME_RE = re.compile(r"^layer-(\d+)\.k(\d+)\.safetensors$")
TP4_RANKS = frozenset(range(4))
TP4_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
TP4_CORE_COMPONENTS = frozenset(("trellis", "suh", "svh"))


class ProvenanceError(RuntimeError):
    """Refusal to mix incompatible provenance in one output family."""


class SourceLayoutError(ProvenanceError):
    """The source cannot be represented by the selected segment layout."""


def read_header(path: Path) -> tuple[dict, int]:
    """Return (header_dict, body_offset) of a safetensors file."""
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    return hdr, 8 + hlen

def _source_tail(snapshot: Path) -> dict:
    """Return the optional rank-sliced config block, rejecting malformed JSON."""
    path = Path(snapshot) / "config.json"
    if not path.exists():
        return {}
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SourceLayoutError(f"{path}: cannot read source topology ({e})") from e
    if not isinstance(config, dict):
        raise SourceLayoutError(f"{path}: source config must be a JSON object")
    tail = config.get("hybrid_tr3_tail")
    if tail is None:
        return {}
    if not isinstance(tail, dict):
        raise SourceLayoutError(
            f"{path}: hybrid_tr3_tail must be an object when present")
    return tail


def _required_tp4_components(tail: dict) -> frozenset[str]:
    """The codebook marker is configured, not a hard-coded rotation-layout key."""
    components = set(TP4_CORE_COMPONENTS)
    schema = tail.get("tensor_schema")
    if isinstance(schema, str):
        groups = re.findall(r"\{([^{}]+)\}", schema)
        if groups:
            components.update(part for part in groups[-1].split("|") if part)
    codebook = tail.get("codebook")
    if codebook is not None:
        if not isinstance(codebook, str) or not codebook:
            raise SourceLayoutError(
                "config.json: hybrid_tr3_tail.codebook must name the codebook "
                "tensor component")
        components.add(codebook)
    else:
        # The published layout's legacy config does not need to spell this out.
        components.add("mcg")
    unsupported = components - COMP_ORDER.keys()
    if unsupported:
        raise SourceLayoutError(
            "config.json: rank_sliced_tp4 does not support routed component(s) "
            f"{sorted(unsupported)}")
    return frozenset(components)


def _tensor_shape(tensor: dict, where: str) -> list[int]:
    shape = tensor.get("shape") if isinstance(tensor, dict) else None
    if not isinstance(shape, list) or any(
            isinstance(axis, bool) or not isinstance(axis, int) or axis <= 0
            for axis in shape):
        raise SourceLayoutError(f"{where}: tensor shape must be positive integer axes")
    return shape


def _numel(shape: list[int]) -> int:
    result = 1
    for axis in shape:
        result *= axis
    return result


def _routed_groups(header: dict, *, layer: int, who: str,
                   components: frozenset[str]) -> dict:
    groups: dict[int, dict[str, dict[int, dict[str, dict]]]] = {}
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        match = EXPERT_RE.match(name)
        if not match:
            continue
        named_layer, expert, projection, rank, component = match.groups()
        if int(named_layer) != layer:
            raise SourceLayoutError(
                f"{who}: routed tensor {name!r} names layer {named_layer}, "
                f"not source shard layer {layer}")
        if projection not in TP4_PROJECTIONS:
            raise SourceLayoutError(
                f"{who}: unsupported routed projection {projection!r} in {name!r}")
        if component not in components:
            raise SourceLayoutError(
                f"{who}: unsupported routed component {component!r} in {name!r}")
        groups.setdefault(int(expert), {}).setdefault(projection, {}).setdefault(
            int(rank), {})[component] = tensor
    return groups


def _validate_tp4_groups(groups: dict, *, experts: set[int], components: frozenset[str],
                         layer: int, who: str) -> None:
    for expert in sorted(experts):
        projections = groups.get(expert)
        if projections is None:
            raise SourceLayoutError(
                f"{who}: layer {layer} routed expert {expert} is absent; "
                "rank_sliced_tp4 requires ranks {0, 1, 2, 3}")
        for projection in TP4_PROJECTIONS:
            ranks_by_component = projections.get(projection, {})
            for component in sorted(components):
                ranks = {rank for rank, values in ranks_by_component.items()
                         if component in values}
                if ranks != TP4_RANKS:
                    raise SourceLayoutError(
                        f"{who}: layer {layer} expert {expert} {projection}."
                        f"{component} has ranks {sorted(ranks)}; "
                        "rank_sliced_tp4 is TP4 only and requires exactly "
                        "{0, 1, 2, 3} (TP1/TP2 cannot be repacked; TP8/TP16 "
                        "are unimplemented)")
            for rank in TP4_RANKS:
                values = ranks_by_component[rank]
                trellis = values["trellis"]
                trellis_shape = _tensor_shape(
                    trellis, f"{who}: layer {layer} expert {expert} "
                    f"{projection}.rank{rank}.trellis")
                if trellis.get("dtype") != "I16" or len(trellis_shape) != 3:
                    raise SourceLayoutError(
                        f"{who}: layer {layer} expert {expert} {projection}."
                        f"rank{rank}.trellis must be I16 with [in_tiles, "
                        "out_tiles, 16*K] shape")
                if trellis_shape[2] % 16:
                    raise SourceLayoutError(
                        f"{who}: layer {layer} expert {expert} {projection}."
                        f"rank{rank}.trellis K axis {trellis_shape[2]} is not "
                        "a multiple of 16")
                suh = _numel(_tensor_shape(
                    values["suh"], f"{who}: layer {layer} expert {expert} "
                    f"{projection}.rank{rank}.suh"))
                svh = _numel(_tensor_shape(
                    values["svh"], f"{who}: layer {layer} expert {expert} "
                    f"{projection}.rank{rank}.svh"))
                if suh != trellis_shape[0] * 16 or svh != trellis_shape[1] * 16:
                    raise SourceLayoutError(
                        f"{who}: layer {layer} expert {expert} {projection}."
                        f"rank{rank} rotation axes do not match trellis "
                        f"[{trellis_shape[0]}, {trellis_shape[1]}, 16*K] "
                        f"(suh={suh}, svh={svh})")


def _compare_tp4_topology(source: dict, segment: dict, *, layer: int, expert: int,
                          projection: str, rank: int, component: str) -> None:
    src_shape = _tensor_shape(
        source, f"source layer {layer} expert {expert} {projection}.rank{rank}.{component}")
    seg_shape = _tensor_shape(
        segment, f"segment layer {layer} expert {expert} {projection}.rank{rank}.{component}")
    if source.get("dtype") != segment.get("dtype"):
        raise SourceLayoutError(
            f"segment layer {layer} expert {expert} {projection}.rank{rank}."
            f"{component} dtype {segment.get('dtype')!r} does not match source "
            f"{source.get('dtype')!r}")
    # K changes the final trellis axis.  Its tile axes still determine TP
    # topology, whereas comparing the full shape would incorrectly reject K4.
    if component == "trellis":
        matches = src_shape[:2] == seg_shape[:2]
    else:
        matches = src_shape == seg_shape
    if not matches:
        raise SourceLayoutError(
            f"segment layer {layer} expert {expert} {projection}.rank{rank}."
            f"{component} topology {seg_shape} does not match source {src_shape}"
            + (" on the non-K trellis axes" if component == "trellis" else ""))


def preflight_source_layout(snapshot: Path, *, layout: str | None,
                            targets: dict[int, set[int] | None],
                            segment_headers: dict[tuple[int, int], dict] | None = None,
                            segment_targets: dict[tuple[int, int], set[int]] | None = None
                            ) -> None:
    """Fail before hashing/staging when a selected layout cannot represent source.

    Layouts are deliberately dispatched: TP4 facts apply only to
    ``rank_sliced_tp4`` and do not constrain dense, MTP, or future layouts.
    A missing ``hybrid_tr3_tail.tp`` is accepted only after exact routed-tensor
    structure proves TP4; DCP fields are never treated as TP metadata.
    """
    if layout != DEFAULT_LAYOUT:
        return
    tail = _source_tail(snapshot)
    if "tp" in tail:
        tp = tail["tp"]
        if isinstance(tp, bool) or not isinstance(tp, int) or tp != 4:
            raise SourceLayoutError(
                "config.json: hybrid_tr3_tail.tp must be integer 4 for "
                "rank_sliced_tp4; TP1/TP2 cannot be repacked and TP8/TP16 "
                "are unimplemented")
    components = _required_tp4_components(tail)
    source_groups: dict[int, dict] = {}
    for shard in sorted(Path(snapshot).glob("model-layer-*.safetensors")):
        try:
            layer = int(shard.stem.split("-")[-1])
        except ValueError as e:
            raise SourceLayoutError(f"cannot determine layer from {shard.name}") from e
        requested = targets.get(layer)
        if layer not in targets:
            continue
        try:
            header, _ = read_header(shard)
        except (OSError, struct.error, json.JSONDecodeError) as e:
            raise SourceLayoutError(f"{shard}: cannot read source header ({e})") from e
        groups = _routed_groups(header, layer=layer, who=f"source {shard.name}",
                                components=components)
        if not groups:
            # Dense/MTP shards are intentionally outside this routed layout.
            continue
        experts = set(groups) if requested is None else set(requested)
        _validate_tp4_groups(groups, experts=experts, components=components,
                             layer=layer, who=f"source {shard.name}")
        source_groups[layer] = groups
    if segment_headers is None:
        return
    for key, header in segment_headers.items():
        layer, _k = key
        wanted = (segment_targets or {}).get(key, set())
        if not wanted:
            continue
        source = source_groups.get(layer)
        if source is None:
            # The source layer was dense/MTP and did not route experts.
            continue
        segment = _routed_groups(header, layer=layer, who=f"segment layer {layer}",
                                 components=components)
        _validate_tp4_groups(segment, experts=set(wanted), components=components,
                             layer=layer, who=f"segment layer {layer}")
        for expert in wanted:
            for projection in TP4_PROJECTIONS:
                for rank in TP4_RANKS:
                    for component in components:
                        _compare_tp4_topology(
                            source[expert][projection][rank][component],
                            segment[expert][projection][rank][component],
                            layer=layer, expert=expert, projection=projection,
                            rank=rank, component=component)


def expert_key(name: str):
    m = EXPERT_RE.match(name)
    if not m:
        return None
    layer, expert, proj, rank, comp = m.groups()
    if proj not in PROJ_ORDER or comp not in COMP_ORDER:
        raise ValueError(f"unknown proj/comp in tensor name: {name}")
    return (int(expert), PROJ_ORDER[proj], int(rank), COMP_ORDER[comp])


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Signer:
    """ed25519 signer; key file holds the 32-byte seed, created on demand."""

    def __init__(self, key_path: Path):
        from nacl.signing import SigningKey

        key_path = Path(key_path)
        if key_path.exists():
            seed = key_path.read_bytes()
            if len(seed) != 32:
                raise ValueError(f"{key_path}: expected 32-byte ed25519 seed")
            self.key = SigningKey(seed)
        else:
            self.key = SigningKey.generate()
            key_path.parent.mkdir(parents=True, exist_ok=True)
            key_path.touch(mode=0o600)
            key_path.write_bytes(bytes(self.key))
        self.pub_hex = self.key.verify_key.encode().hex()

    def sign_line(self, payload: dict) -> str:
        raw = canonical_json(payload)
        sig = self.key.sign(raw).signature
        return json.dumps(
            {
                "payload": base64.b64encode(raw).decode(),
                "signature": base64.b64encode(sig).decode(),
                "keyid": self.pub_hex,
            },
            separators=(",", ":"),
        )


def load_source_shas(snapshot: Path) -> dict[str, str]:
    """Parse the source repo's MANIFEST.sha256 (sha256sum format) if present.

    Advisory only: every shard this tool reads is hashed locally and the two
    values are cross-checked (see verify_source_shard).
    """
    out = {}
    mf = snapshot / "MANIFEST.sha256"
    if mf.exists():
        for line in mf.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                out[parts[1]] = parts[0]
    return out


def sha256_file(path: Path, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_source_shard(shard: Path, declared: str | None) -> str:
    """Hash the source shard locally; cross-check any declared digest.

    The source repo's MANIFEST.sha256 is untrusted metadata that ships next to
    the bytes it describes — a shard swapped after the manifest was written
    (or a manifest edited after the shard) must not be silently attested as
    the pinned source.  Disagreement is fatal.
    """
    computed = sha256_file(shard)
    if declared and declared != computed:
        raise ProvenanceError(
            f"{shard.name}: source MANIFEST.sha256 declares {declared} but the "
            f"local file hashes {computed} — refusing to attest a source whose "
            f"bytes disagree with its manifest")
    return computed


# --------------------------------------------------------------- state (v2)

def source_fingerprint(repo: str, revision: str, base_model: str,
                       layout: str = DEFAULT_LAYOUT) -> str:
    """Stable short id of one source identity: repo@revision for one
    base_model in one layout.  State keys start with this so segments from
    two different sources can share an output dir without colliding."""
    return hashlib.sha256(canonical_json({
        "base_model": base_model, "layout": layout,
        "repo": repo, "revision": revision})).hexdigest()[:16]


def new_state() -> dict:
    return {"schema": STATE_SCHEMA, "sources": {}}


def migrate_state_v1(raw: dict, legacy: dict) -> dict:
    """Lift a layer-keyed fq-repack-state/1 file into the {source, K, layer}
    shape.  Each v1 entry names the segment file it produced, so its K is
    recoverable from the file name; entries whose file name is unusable are
    dropped (the layer is simply repacked again)."""
    state = new_state()
    src = state["sources"].setdefault(legacy["fp"], {
        "repo": legacy["repo"], "revision": legacy["revision"],
        "base_model": legacy["base_model"], "layout": legacy["layout"], "k": {}})
    for layer_s, done in (raw.get("layers") or {}).items():
        entry = dict(done)
        m = SEG_NAME_RE.match((entry.get("index") or {}).get("file", ""))
        if not m:
            continue
        entry["migrated_from"] = "fq-repack-state/1"
        src["k"].setdefault(str(int(m.group(2))), {"layers": {}})[
            "layers"][str(int(layer_s))] = entry
    state["migrated_from"] = "fq-repack-state/1"
    return state


def load_state(path: Path, legacy: dict) -> dict:
    if not path.exists():
        return new_state()
    raw = json.loads(path.read_text())
    if raw.get("schema") == STATE_SCHEMA:
        raw.setdefault("sources", {})
        return raw
    if "layers" in raw:
        return migrate_state_v1(raw, legacy)
    raise ProvenanceError(
        f"{path}: unrecognised state schema {raw.get('schema')!r}")


def state_source(state: dict, fp: str, meta: dict) -> dict:
    src = state["sources"].setdefault(fp, {**meta, "k": {}})
    src.update(meta)  # keep the human-readable fields fresh
    src.setdefault("k", {})
    return src


def get_entry(state: dict, fp: str, k: int, layer: int) -> dict | None:
    return (state.get("sources", {}).get(fp, {}).get("k", {})
            .get(str(k), {}).get("layers", {}).get(str(layer)))


def put_entry(state: dict, fp: str, k: int, layer: int, entry: dict,
              meta: dict) -> None:
    """Record one {source, K, layer} result and drop any entry for the same
    (K, layer) under a DIFFERENT source: the segment file it described has
    just been overwritten, so keeping it would publish a stale digest."""
    src = state_source(state, fp, meta)
    src["k"].setdefault(str(k), {"layers": {}})["layers"][str(layer)] = entry
    for other_fp, other in state["sources"].items():
        if other_fp == fp:
            continue
        layers = other.get("k", {}).get(str(k), {}).get("layers", {})
        if layers.pop(str(layer), None) is not None:
            print(f"  note: dropped stale K{k} layer {layer} entry from source "
                  f"{other_fp} (its segment file was just replaced)", flush=True)


def iter_entries(state: dict):
    """(fp, source_meta, k, layer, entry) for every done entry in the state."""
    for fp, src in sorted(state.get("sources", {}).items()):
        for k_s, kd in sorted(src.get("k", {}).items(), key=lambda kv: int(kv[0])):
            for layer_s, entry in sorted(kd.get("layers", {}).items(),
                                         key=lambda kv: int(kv[0])):
                if entry.get("status") == "done":
                    yield fp, src, int(k_s), int(layer_s), entry


def index_for_k(state: dict, k: int) -> dict:
    """Rebuild index-kK.json purely from the K's own state entries."""
    merged = {}
    for _fp, _src, kk, layer, entry in iter_entries(state):
        if kk != k:
            continue
        info = entry.get("index") or {}
        if not SEG_NAME_RE.match(info.get("file", "")) or \
                not info["file"].endswith(f".k{k}.safetensors"):
            raise ProvenanceError(
                f"state corruption: K{k} layer {layer} points at "
                f"{info.get('file')!r}")
        merged[str(layer)] = info
    return dict(sorted(merged.items(), key=lambda kv: int(kv[0])))


def build_manifest(state: dict, existing: dict, *, base_model: str, layout: str,
                   revision: str, signer_pubkey: str, out: Path) -> dict:
    """Merge a MULTI-K manifest from the state (never last-writer-wins).

    per_k[K] carries that K's index file, layer coverage, segment count and
    provenance; k_variants lists every K present.  Entries for Ks that are not
    in the state but whose index file still exists are preserved verbatim.
    """
    per_k: dict[str, dict] = {}
    signers, sources, all_layers = set(), set(), []
    num_experts = 0
    for _fp, src, k, layer, entry in iter_entries(state):
        info = entry.get("index") or {}
        ent = per_k.setdefault(str(k), {
            "index": f"index-k{k}.json", "predicate": "repack-of",
            "layers": [layer, layer], "segment_count": 0, "num_experts": 0,
            "sources": [], "signers": []})
        ent["layers"] = [min(ent["layers"][0], layer), max(ent["layers"][1], layer)]
        ent["segment_count"] += 1
        ent["num_experts"] = max(ent["num_experts"], len(info.get("experts", {})))
        who = {"repo": src.get("repo"), "revision": src.get("revision")}
        if who not in ent["sources"]:
            ent["sources"].append(who)
        pub = entry.get("signer_pubkey")
        if pub and pub not in ent["signers"]:
            ent["signers"].append(pub)
        if pub:
            signers.add(pub)
        if src.get("repo"):
            sources.add(src["repo"])
        all_layers.append(layer)
        num_experts = max(num_experts, len(info.get("experts", {})))
    for ent in per_k.values():
        ent["provenance"] = "; ".join(
            f"repack-of {s['repo']}@{(s['revision'] or '')[:8]}"
            for s in ent["sources"])
        if len(ent["sources"]) == 1:
            ent["source_repo"] = ent["sources"][0]["repo"]
            ent["source_revision"] = ent["sources"][0]["revision"]
    for k_s, ent in (existing.get("per_k") or {}).items():
        if k_s not in per_k and (out / f"index-k{k_s}.json").exists():
            per_k[k_s] = ent  # a K this run knows nothing about: keep as-is
            if ent.get("source_repo"):
                sources.add(ent["source_repo"])
    ks = sorted(int(k) for k in per_k)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "base_model": base_model,
        "revision": revision,
        "k_variants": ks,
        "per_k": {str(k): per_k[str(k)] for k in ks},
        "hessian_id": None,
        "predicate": "repack-of",
        "layout": layout,
        "sources": sorted(sources),
        "moe_layers": [min(all_layers), max(all_layers)] if all_layers else [],
        "num_experts": num_experts,
        "tensor_indexes": {str(k): f"index-k{k}.json" for k in ks},
        "signer_pubkey": signer_pubkey,
        "signers": sorted(signers),
        "note": ("Multi-K manifest: per_k[K] holds that K's index, coverage and "
                 "provenance; a run for one K never rewrites another K's index "
                 "or entry. Top-level revision is the most recent run's source "
                 "revision — per_k[K].source_revision is authoritative."),
    }
    return manifest


def check_output_family(out: Path, *, base_model: str, layout: str, repo: str,
                        revision: str, k: int, allow_change: bool) -> dict:
    """Refuse an output dir that already describes a different family.

    base_model / layout mismatches are fatal (use a fresh dir).  Re-repacking
    a K that is already present from a DIFFERENT source repo/revision
    overwrites that K's segments, so it needs --allow-provenance-change.
    Returns the existing manifest ({} when there is none).
    """
    mpath = out / "fq-manifest.json"
    if not mpath.exists():
        return {}
    existing = json.loads(mpath.read_text())
    for field, want in (("base_model", base_model), ("layout", layout)):
        have = existing.get(field)
        if have is not None and have != want:
            raise ProvenanceError(
                f"{out}: existing family declares {field}={have!r} but this run "
                f"is {want!r} — refusing to mix families in one output dir")
    prev = (existing.get("per_k") or {}).get(str(k)) or {}
    prev_src = (prev.get("source_repo"), prev.get("source_revision"))
    if all(prev_src) and prev_src != (repo, revision) and not allow_change:
        raise ProvenanceError(
            f"{out}: K{k} here was repacked from {prev_src[0]}@{prev_src[1]}, "
            f"this run is {repo}@{revision} — rerunning overwrites that K's "
            f"segments; pass --allow-provenance-change to accept")
    return existing


def resumable(entry: dict | None, seg_path: Path, seg_name: str, *,
              signer_pubkey: str, declared_source_sha: str | None,
              recheck: bool) -> tuple[bool, str]:
    """(ok, reason) — is this {source, K, layer} entry safe to skip?"""
    if not entry or entry.get("status") != "done":
        return False, "not done"
    info = entry.get("index") or {}
    if info.get("file") != seg_name:
        return False, f"index names {info.get('file')!r}"
    if not seg_path.exists():
        return False, "segment file missing"
    if info.get("size") and seg_path.stat().st_size != info["size"]:
        return False, "segment size changed"
    if entry.get("signer_pubkey") and entry["signer_pubkey"] != signer_pubkey:
        return False, "signed by a different key"
    if (declared_source_sha and entry.get("source_file_sha256")
            and declared_source_sha != entry["source_file_sha256"]):
        return False, "source shard changed"
    if recheck and sha256_file(seg_path) != info.get("sha256"):
        return False, "segment sha256 mismatch (--recheck)"
    return True, "done"


def repack_layer(
    shard_path: Path,
    seg_path: Path,
    *,
    k: int,
    layer: int,
    meta_common: dict,
    source_file_sha256: str | None,
) -> dict:
    """Write one segment file; return {segment sha, size, per-expert info}."""
    hdr, body_off = read_header(shard_path)
    hdr.pop("__metadata__", None)
    keyed = [(expert_key(n), n) for n in hdr]
    keyed = [(kk, n) for kk, n in keyed if kk is not None]
    if not keyed:
        return {}
    keyed.sort()

    seg_tensors, per_expert, digests = {}, {}, {}
    off = 0
    for kk, name in keyed:
        a, b = hdr[name]["data_offsets"]
        size = b - a
        seg_tensors[name] = {
            "dtype": hdr[name]["dtype"],
            "shape": hdr[name]["shape"],
            "data_offsets": [off, off + size],
        }
        eid = kk[0]
        if eid not in per_expert:
            per_expert[eid] = [off, off]
            digests[eid] = hashlib.sha256()
        if per_expert[eid][1] != off:
            raise AssertionError(f"expert {eid} tensors not contiguous at {name}")
        per_expert[eid][1] = off + size
        off += size

    meta = {
        "fq_schema": SEGMENT_SCHEMA,
        "predicate": "repack-of",
        "k": str(k),
        "layer": str(layer),
        "layout": DEFAULT_LAYOUT,
        "num_experts": str(len(per_expert)),
        "source_file": shard_path.name,
        **{kx: str(v) for kx, v in meta_common.items()},
    }
    if source_file_sha256:
        meta["source_file_sha256"] = source_file_sha256
    header = {"__metadata__": meta, **seg_tensors}
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)

    seg_sha = hashlib.sha256()
    tmp = seg_path.with_suffix(".part")
    with open(shard_path, "rb") as src, open(tmp, "wb") as dst:
        mm = mmap.mmap(src.fileno(), 0, access=mmap.ACCESS_READ)
        prefix = struct.pack("<Q", len(hj)) + hj
        dst.write(prefix)
        seg_sha.update(prefix)
        for kk, name in keyed:
            a, b = hdr[name]["data_offsets"]
            data = mm[body_off + a : body_off + b]
            dst.write(data)
            seg_sha.update(data)
            digests[kk[0]].update(data)
        mm.close()
    os.replace(tmp, seg_path)

    return {
        "sha256": seg_sha.hexdigest(),
        "size": seg_path.stat().st_size,
        "body_offset": 8 + len(hj),
        # digest of the 8-byte length prefix + header JSON, i.e. exactly the
        # bytes a consumer must trust before planning ranged reads from it
        "header_sha256": hashlib.sha256(prefix).hexdigest(),
        "experts": {str(e): rng for e, rng in sorted(per_expert.items())},
        "expert_sha256": {str(e): d.hexdigest() for e, d in sorted(digests.items())},
    }


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    os.replace(tmp, path)


class Publisher:
    """Incremental HF uploader; no-op when repo_id is None."""

    def __init__(self, repo_id: str | None, private: bool):
        self.repo_id = repo_id
        if repo_id:
            from huggingface_hub import HfApi

            self.api = HfApi()
            self.api.create_repo(repo_id, private=private, exist_ok=True)

    def upload(self, local: Path, remote: str) -> None:
        if self.repo_id:
            self.api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=remote,
                repo_id=self.repo_id,
                commit_message=f"fq_repack: {remote}",
            )


def write_if_changed(path: Path, obj) -> bool:
    """atomic_write_json, reporting whether the file's bytes actually moved."""
    before = path.read_bytes() if path.exists() else None
    atomic_write_json(path, obj)
    return path.read_bytes() != before


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--source-repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--layers", default=None, help="e.g. 3-78 or 3,4,5; default: all with experts")
    p.add_argument("--sign-key", type=Path, default=Path.home() / ".fq_keys/fq_signing.key")
    p.add_argument("--publish", default=None, help="HF repo id for incremental upload")
    p.add_argument("--private", action="store_true", default=True)
    p.add_argument("--public", dest="private", action="store_false")
    p.add_argument(
        "--allow-provenance-change", action="store_true",
        help="accept re-repacking a K already present in this dir from a "
             "different source repo/revision (overwrites that K's segments)")
    p.add_argument(
        "--recheck", action="store_true",
        help="re-hash already-done segments before skipping them on resume")
    args = p.parse_args(argv)

    out, k, layout = args.out, args.k, DEFAULT_LAYOUT
    shards = sorted(args.snapshot.glob("model-layer-*.safetensors"))
    wanted = None
    if args.layers:
        wanted = set()
        for part in args.layers.split(","):
            if "-" in part:
                a, b = part.split("-")
                wanted.update(range(int(a), int(b) + 1))
            else:
                wanted.add(int(part))
    targets = {
        int(shard.stem.split("-")[-1]): None
        for shard in shards
        if wanted is None or int(shard.stem.split("-")[-1]) in wanted
    }
    # This runs before output creation, source hashing, signing, or a segment
    # write: an incompatible TP source must never gain a trusted TP4 label.
    preflight_source_layout(args.snapshot, layout=layout, targets=targets)

    (out / "attestations").mkdir(parents=True, exist_ok=True)
    existing = check_output_family(
        out, base_model=args.base_model, layout=layout, repo=args.source_repo,
        revision=args.revision, k=k, allow_change=args.allow_provenance_change)

    fp = source_fingerprint(args.source_repo, args.revision, args.base_model, layout)
    src_meta = {"repo": args.source_repo, "revision": args.revision,
                "base_model": args.base_model, "layout": layout}
    legacy = dict(src_meta)
    if existing:
        # a v1 state in this dir belongs to whatever the existing manifest
        # describes, not necessarily to this run's source
        legacy = {
            "repo": (existing.get("sources") or [args.source_repo])[0],
            "revision": existing.get("revision") or args.revision,
            "base_model": existing.get("base_model") or args.base_model,
            "layout": existing.get("layout") or layout,
        }
    legacy["fp"] = source_fingerprint(
        legacy["repo"], legacy["revision"], legacy["base_model"], legacy["layout"])

    state_path = out / "state.json"
    state = load_state(state_path, legacy)
    state_source(state, fp, src_meta)
    signer = Signer(args.sign_key)
    source_shas = load_source_shas(args.snapshot)
    publisher = Publisher(args.publish, args.private)
    meta_common = {
        "base_model": args.base_model,
        "source_repo": args.source_repo,
        "source_revision": args.revision,
    }
    for shard in shards:
        layer = int(shard.stem.split("-")[-1])
        if wanted is not None and layer not in wanted:
            continue
        seg_name = f"layer-{layer:03d}.k{k}.safetensors"
        seg_path = out / seg_name
        att_path = out / "attestations" / f"layer-{layer:03d}.k{k}.jsonl"
        prior = get_entry(state, fp, k, layer)
        ok, why = resumable(prior, seg_path, seg_name,
                            signer_pubkey=signer.pub_hex,
                            declared_source_sha=source_shas.get(shard.name),
                            recheck=args.recheck)
        if ok:
            print(f"layer {layer:3d}: skip (done, k{k}, "
                  f"{prior['index']['sha256'][:12]})", flush=True)
            continue
        if prior:
            print(f"layer {layer:3d}: redoing k{k} ({why})", flush=True)

        t0 = time.time()
        source_sha = verify_source_shard(shard, source_shas.get(shard.name))
        info = repack_layer(
            shard,
            seg_path,
            k=k,
            layer=layer,
            meta_common=meta_common,
            source_file_sha256=source_sha,
        )
        if not info:
            print(f"layer {layer:3d}: no expert tensors, skipped", flush=True)
            continue

        payload = {
            "schema": ATTESTATION_SCHEMA,
            "predicate": "repack-of",
            "fragment": {
                "file": seg_name, "sha256": info["sha256"],
                "size": info["size"],
                # Signed header digest + body offset: lets a consumer
                # authenticate the tensor layout it plans from with ONE
                # ranged read, instead of pulling the whole fragment to
                # re-derive it (fq_fetch --header-trust attested).
                "header_sha256": info["header_sha256"],
                "body_offset": info["body_offset"],
            },
            "materials": {
                "repo": args.source_repo,
                "revision": args.revision,
                "file": shard.name,
                "file_sha256": source_sha,
                "file_sha256_source": ("computed+manifest"
                                       if source_shas.get(shard.name) else "computed"),
            },
            "layer": layer,
            "k": k,
            "layout": layout,
            "base_model": args.base_model,
            "num_experts": len(info["expert_sha256"]),
            "expert_sha256": info["expert_sha256"],
            "tool": {"name": "fq_repack", "version": TOOL_VERSION},
            "created_utc": now_utc(),
        }
        att_path.write_text(signer.sign_line(payload) + "\n")

        entry = {
            "status": "done",
            "index": {
                "file": seg_name,
                "sha256": info["sha256"],
                "size": info["size"],
                "body_offset": info["body_offset"],
                "experts": info["experts"],
            },
            "source_file": shard.name,
            "source_file_sha256": source_sha,
            "signer_pubkey": signer.pub_hex,
            "tool_version": TOOL_VERSION,
            "created_utc": payload["created_utc"],
        }
        publisher.upload(seg_path, seg_name)
        publisher.upload(att_path, f"attestations/{att_path.name}")
        put_entry(state, fp, k, layer, entry, src_meta)
        atomic_write_json(state_path, state)
        print(
            f"layer {layer:3d}: {info['size']/1e9:.2f} GB sha {info['sha256'][:12]} "
            f"({time.time()-t0:.1f}s)",
            flush=True,
        )

    ks_present = sorted({kk for _f, _s, kk, _l, _e in iter_entries(state)})
    for kk in ks_present:
        idx_path = out / f"index-k{kk}.json"
        if write_if_changed(idx_path, index_for_k(state, kk)):
            publisher.upload(idx_path, idx_path.name)
    manifest = build_manifest(
        state, existing, base_model=args.base_model, layout=layout,
        revision=args.revision, signer_pubkey=signer.pub_hex, out=out)
    mpath = out / "fq-manifest.json"
    if write_if_changed(mpath, manifest):
        publisher.upload(mpath, mpath.name)
    total = sum(len(index_for_k(state, kk)) for kk in ks_present)
    print(f"done: {total} segments across K{ks_present} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
