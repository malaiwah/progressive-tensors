#!/usr/bin/env python3
"""fq_repack — repack a per-layer-sharded EXL3 quant into Progressive Tensors segments.

Scales up poc/poc_slice.py per implementation/10 & 11: for each source layer
shard, extract the routed-expert tensors verbatim (predicate: repack-of, no
re-encode) into a per-layer segment file (pure safetensors) whose body is
per-expert contiguous and therefore range-readable per expert, plus a signed
attestation line and a per-expert byte-range index.

v1 layout keeps the source's rank-sliced granularity verbatim
(layout=rank_sliced_tp4); unsharding is a later, T4-verified upgrade.

Resumable per layer via <out>/state.json; optional incremental HF publish.
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
COMP_ORDER = {"trellis": 0, "suh": 1, "svh": 2, "mcg": 3}
EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+_proj)\.rank(\d+)\.(\w+)$"
)
SEGMENT_SCHEMA = "fq-segment/1"
ATTESTATION_SCHEMA = "fq-attestation/1"
MANIFEST_SCHEMA = "fq-manifest/1"


def read_header(path: Path) -> tuple[dict, int]:
    """Return (header_dict, body_offset) of a safetensors file."""
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    return hdr, 8 + hlen


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
    """Parse the source repo's MANIFEST.sha256 (sha256sum format) if present."""
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
        "layout": "rank_sliced_tp4",
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
    args = p.parse_args(argv)

    out, k = args.out, args.k
    (out / "attestations").mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"layers": {}}
    signer = Signer(args.sign_key)
    source_shas = load_source_shas(args.snapshot)
    publisher = Publisher(args.publish, args.private)

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

    meta_common = {
        "base_model": args.base_model,
        "source_repo": args.source_repo,
        "source_revision": args.revision,
    }
    index, moe_layers = {}, []
    for shard in shards:
        layer = int(shard.stem.split("-")[-1])
        if wanted is not None and layer not in wanted:
            continue
        seg_name = f"layer-{layer:03d}.k{k}.safetensors"
        seg_path = out / seg_name
        att_path = out / "attestations" / f"layer-{layer:03d}.k{k}.jsonl"
        done = state["layers"].get(str(layer), {})
        if done.get("status") == "done" and seg_path.exists():
            index[str(layer)] = done["index"]
            moe_layers.append(layer)
            print(f"layer {layer:3d}: skip (done, {done['index']['sha256'][:12]})", flush=True)
            continue

        t0 = time.time()
        info = repack_layer(
            shard,
            seg_path,
            k=k,
            layer=layer,
            meta_common=meta_common,
            source_file_sha256=source_shas.get(shard.name),
        )
        if not info:
            print(f"layer {layer:3d}: no expert tensors, skipped", flush=True)
            continue
        moe_layers.append(layer)

        payload = {
            "schema": ATTESTATION_SCHEMA,
            "predicate": "repack-of",
            "fragment": {"file": seg_name, "sha256": info["sha256"], "size": info["size"]},
            "materials": {
                "repo": args.source_repo,
                "revision": args.revision,
                "file": shard.name,
                "file_sha256": source_shas.get(shard.name),
            },
            "expert_sha256": info["expert_sha256"],
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        att_path.write_text(signer.sign_line(payload) + "\n")

        entry = {
            "file": seg_name,
            "sha256": info["sha256"],
            "size": info["size"],
            "body_offset": info["body_offset"],
            "experts": info["experts"],
        }
        index[str(layer)] = entry
        publisher.upload(seg_path, seg_name)
        publisher.upload(att_path, f"attestations/{att_path.name}")
        state["layers"][str(layer)] = {"status": "done", "index": entry}
        atomic_write_json(state_path, state)
        print(
            f"layer {layer:3d}: {info['size']/1e9:.2f} GB sha {info['sha256'][:12]} "
            f"({time.time()-t0:.1f}s)",
            flush=True,
        )

    atomic_write_json(out / f"index-k{k}.json", index)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "base_model": args.base_model,
        "revision": args.revision,
        "k_variants": [k],
        "hessian_id": None,
        "predicate": "repack-of",
        "layout": "rank_sliced_tp4",
        "sources": [args.source_repo],
        "moe_layers": [min(moe_layers), max(moe_layers)] if moe_layers else [],
        "num_experts": max(
            (len(v["experts"]) for v in index.values()), default=0
        ),
        "tensor_index": f"index-k{k}.json",
        "signer_pubkey": signer.pub_hex,
    }
    atomic_write_json(out / "fq-manifest.json", manifest)
    publisher.upload(out / f"index-k{k}.json", f"index-k{k}.json")
    publisher.upload(out / "fq-manifest.json", "fq-manifest.json")
    print(f"done: {len(index)} layers -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
