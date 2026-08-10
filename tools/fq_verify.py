#!/usr/bin/env python3
"""fq_verify — prove Progressive Tensors reconstructions, byte-level and numeric.

Two modes, one per proof rung (see runs/0c-campaign/ATTESTATION-V2.md):

--identity  Byte-level reconstruction proof, at the strongest granularity the
            family supports (auto-detected, or forced with --check):

  local    repack-of family + --source snapshot: stream-reassemble every MoE
           shard from the segments (expert bytes) + the source checkpoint
           (header template and non-expert bytes, exactly as fq_assemble
           writes them) and compare the assembled sha256 against the source
           shard sha256 (MANIFEST.sha256 when present, else hashed).
           Whole-shard byte-identity.  A sampled attestation pass re-hashes
           per-expert segment spans against the signed fq-attestation lines.

  remote   repack-of family primed from a remote repo (fq_prime output, no
           local snapshot): full local integrity first (segment sha256 vs
           index + signed attestation, per-expert span sha256 vs attestation),
           then for sampled experts a FRESH ranged re-read of the pinned
           source (fresh header fetch, fresh payload fetch — the tool never
           trusts fq_prime's caches) and a per-tensor byte comparison against
           the segment content.  Shared profile fragments are re-fetched and
           compared in full.  Fragment-granularity byte-identity: whole-shard
           identity is not applicable when only a layer window was primed.

  derived  derived-from family (fq_prime --expand output): re-derive every
           expanded expert from its parent shared-h segment + shared profile
           (replicating the layer's shared suh/svh rows into the per-expert
           slots) and byte-compare the full unit, expert by expert, tensor by
           tensor.  Full coverage, local.  Parent segment/profile sha256 are
           re-hashed and checked against the pins in the expanded metadata,
           chaining this proof to the parent family's remote identity.

--similarity  Numeric equivalence evidence: dequantize corresponding experts
           from two or more families with the reference exllamav3 reconstruct
           (GPU; run under the GG env) and report cosine similarity, relative
           Frobenius error and max |diff| for every family pair — and against
           the BF16 ground truth when --bf16 points at the base checkpoint.
           Families whose fragments are byte-identical inputs (e.g. shared-h
           vs its expanded view) must decode bitwise-equal; the report flags
           bitwise equality per pair.  This is the evidence line for the
           equivalence-of predicate: byte-identity where it should hold,
           bounded numeric similarity where it should not.

Outputs: human-readable progress on stdout, plus --json (machine report) and
--md (markdown table) files.

Examples:
  # end-to-end shard identity, K3 family vs local brandonmusic snapshot
  fq_verify.py --identity --segments ~/fq-segments/GLM-5.2-EXL3-FQ \
      --source <snapshot-dir> --json id-k3.json --md id-k3.md

  # fragment identity of a primed family vs fresh HF ranged reads
  fq_verify.py --identity --segments ~/fq-primed/segments-336 --sample 3

  # expansion re-derivation (full coverage, local)
  fq_verify.py --identity --segments ~/fq-primed/segments-342/expanded \
      --parent ~/fq-primed/segments-342/shared-h

  # cross-family numeric similarity on GPU (gg env)
  CUDA_VISIBLE_DEVICES=4 gg-run.sh python fq_verify.py --similarity \
      --family bm-k3=~/fq-segments/GLM-5.2-EXL3-FQ,k=3 \
      --family 342-k4=~/fq-primed/segments-342/expanded,k=4 \
      --bf16 <bf16-snapshot> --layers 3-10 --experts 24
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mmap
import random
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fq_repack import EXPERT_RE, PROJ_ORDER, expert_key, read_header  # noqa: E402
from fq_assemble import SegmentReader  # noqa: E402
import fq_prime  # noqa: E402  (Transport, SHARED_RE, parse_int_set)

CHUNK = 1 << 24
PROJS = ("gate_proj", "up_proj", "down_proj")


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def span_sha256(path: Path, lo: int, hi: int) -> str:
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        digest = hashlib.sha256(mm[lo:hi]).hexdigest()
        mm.close()
    return digest


def parse_layers(spec: str | None) -> list[int] | None:
    return fq_prime.parse_int_set(spec) if spec else None


# ------------------------------------------------------------ family model

def load_indexes(fam: Path) -> dict[int, dict[str, dict]]:
    """{k: {layer_str: index entry}} from index-k*.json; falls back to
    state.json (fq_prime keeps per-layer entries there before the final
    index write) so partially primed families are verifiable too."""
    out: dict[int, dict[str, dict]] = {}
    for p in sorted(fam.glob("index-k*.json")):
        out[int(p.stem.split("-k")[1])] = json.loads(p.read_text())
    if out:
        return out
    which = "expanded" if fam.name == "expanded" else "index"
    for state_path in (fam / "state.json", fam.parent / "state.json"):
        if not state_path.exists():
            continue
        state = json.loads(state_path.read_text())
        for layer_s, st in state.get("layers", {}).items():
            if st.get("status") != "done":
                continue
            for k_s, entry in st.get(which, {}).items():
                if (fam / entry["file"]).exists():
                    out.setdefault(int(k_s), {})[layer_s] = entry
        if out:
            print(f"note: no index-k*.json under {fam}; "
                  f"using {state_path} ({which})", flush=True)
            break
    return out


def load_shared_index(fam: Path) -> dict[str, dict]:
    p = fam / "index-shared.json"
    if p.exists():
        return json.loads(p.read_text())
    out = {}
    for state_path in (fam / "state.json", fam.parent / "state.json"):
        if not state_path.exists():
            continue
        state = json.loads(state_path.read_text())
        for layer_s, st in state.get("layers", {}).items():
            if st.get("status") == "done" and "profile" in st:
                if (fam / st["profile"]["file"]).exists():
                    out[layer_s] = st["profile"]
        break
    return out


def segment_meta(fam: Path, entry: dict) -> dict:
    return read_header(fam / entry["file"])[0].get("__metadata__", {})


def load_attestation(fam: Path, stem: str, pubkey_hex: str | None):
    """(payload, sig_state) for attestations/<stem>.jsonl; sig_state is
    'verified', 'BAD', 'unverified' (no pynacl / no pubkey) or None (missing
    file)."""
    p = fam / "attestations" / f"{stem}.jsonl"
    if not p.exists():
        return None, None
    line = json.loads(p.read_text())
    payload = json.loads(base64.b64decode(line["payload"]))
    state = "unverified"
    if pubkey_hex:
        try:
            from nacl.signing import VerifyKey
            try:
                VerifyKey(bytes.fromhex(pubkey_hex)).verify(
                    base64.b64decode(line["payload"]),
                    base64.b64decode(line["signature"]))
                state = "verified"
            except Exception:  # noqa: BLE001 — any failure is a bad signature
                state = "BAD"
        except ImportError:
            pass
    return payload, state


# --------------------------------------------------------- identity: local

def stream_assemble_sha(src_shard: Path, readers: dict[int, SegmentReader],
                        k_for_expert) -> dict:
    """sha256 of the shard fq_assemble would write for this policy, computed
    from the same byte sources (expert tensors from segments, header template
    and everything else from the source shard) without materializing it."""
    hdr, body_off = read_header(src_shard)
    meta = hdr.pop("__metadata__", None)
    ordered = sorted(hdr.items(), key=lambda kv: kv[1]["data_offsets"][0])
    header = dict(hdr) if meta is None else {"__metadata__": meta, **hdr}
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    h = hashlib.sha256()
    h.update(struct.pack("<Q", len(hj)))
    h.update(hj)
    stats = {"expert_tensors": 0, "other_tensors": 0,
             "bytes_from_segments": 0, "bytes_from_source": len(hj) + 8}
    with open(src_shard, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for name, t in ordered:
            want = t["data_offsets"][1] - t["data_offsets"][0]
            m = EXPERT_RE.match(name)
            if m:
                data = readers[k_for_expert(int(m.group(2)))].tensor_bytes(name)
                if len(data) != want:
                    raise AssertionError(
                        f"{name}: segment holds {len(data)} bytes, source slot "
                        f"{want} — mixed-size policy, whole-shard identity "
                        f"does not apply")
                h.update(data)
                stats["expert_tensors"] += 1
                stats["bytes_from_segments"] += want
            else:
                a, b = t["data_offsets"]
                for off in range(body_off + a, body_off + b, CHUNK):
                    h.update(mm[off:min(off + CHUNK, body_off + b)])
                stats["other_tensors"] += 1
                stats["bytes_from_source"] += want
        mm.close()
    return {"sha256": h.hexdigest(), **stats}


def cmd_identity_local(args) -> tuple[int, dict]:
    fam, src = args.segments, args.source
    indexes = load_indexes(fam)
    if not indexes:
        print(f"no indexes under {fam}", flush=True)
        return 2, {}
    manifest_path = fam / "fq-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    source_shas = {}
    mf = src / "MANIFEST.sha256"
    if mf.exists():
        for line in mf.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                source_shas[parts[1]] = parts[0]
    wanted = parse_layers(args.layers)
    layer_set = sorted({int(L) for idx in indexes.values() for L in idx})
    if wanted is not None:
        layer_set = [L for L in layer_set if L in wanted]

    rows, failures = [], 0
    for layer in layer_set:
        shard_name = f"model-layer-{layer:03d}.safetensors"
        src_shard = src / shard_name
        readers, k_of = {}, {}
        for k, idx in indexes.items():
            entry = idx.get(str(layer))
            if entry is None:
                continue
            readers[k] = SegmentReader(fam / entry["file"])
            for eid in entry["experts"]:
                if int(eid) in k_of:
                    raise ValueError(f"expert {eid} indexed under two Ks, layer {layer}")
                k_of[int(eid)] = k
        t0 = time.time()
        try:
            got = stream_assemble_sha(src_shard, readers, lambda e: k_of[e])
        finally:
            for r in readers.values():
                r.close()
        want_sha = source_shas.get(shard_name) or file_sha256(src_shard)
        ok = got["sha256"] == want_sha
        failures += 0 if ok else 1
        rows.append({"layer": layer, "shard": shard_name, "match": ok,
                     "assembled_sha256": got["sha256"], "source_sha256": want_sha,
                     "source_sha_from": "MANIFEST.sha256" if shard_name in source_shas
                     else "hashed", **{k: got[k] for k in (
                         "expert_tensors", "other_tensors",
                         "bytes_from_segments", "bytes_from_source")}})
        print(f"layer {layer:3d}: {'MATCH' if ok else 'MISMATCH'} "
              f"{got['sha256'][:16]} vs {want_sha[:16]} "
              f"({got['bytes_from_segments'] / 1e9:.2f}GB from segments, "
              f"{got['bytes_from_source'] / 1e9:.2f}GB source pass-through, "
              f"{time.time() - t0:.0f}s)", flush=True)

    # sampled attestation pass: per-expert span digests vs signed lines
    att_rows = []
    rng = random.Random(args.seed)
    att_layers = (layer_set if args.attest == "all"
                  else rng.sample(layer_set, min(int(args.attest), len(layer_set))))
    for layer in sorted(att_layers):
        for k, idx in indexes.items():
            entry = idx.get(str(layer))
            if entry is None:
                continue
            payload, sig = load_attestation(
                fam, f"layer-{layer:03d}.k{k}", manifest.get("signer_pubkey"))
            if payload is None:
                att_rows.append({"layer": layer, "k": k, "status": "no-attestation"})
                continue
            seg = fam / entry["file"]
            frag_ok = file_sha256(seg) == payload["fragment"]["sha256"]
            n_bad = 0
            for eid, (lo, hi) in entry["experts"].items():
                got = span_sha256(seg, entry["body_offset"] + lo,
                                  entry["body_offset"] + hi)
                if payload.get("expert_sha256", {}).get(str(eid)) != got:
                    n_bad += 1
            ok = frag_ok and n_bad == 0 and sig != "BAD"
            failures += 0 if ok else 1
            att_rows.append({"layer": layer, "k": k, "signature": sig,
                             "fragment_sha_match": frag_ok,
                             "experts_checked": len(entry["experts"]),
                             "expert_sha_mismatches": n_bad})
            print(f"attestation layer {layer:3d} k{k}: sig={sig} fragment="
                  f"{'ok' if frag_ok else 'MISMATCH'} experts="
                  f"{len(entry['experts'])} bad={n_bad}", flush=True)

    report = {
        "mode": "identity", "check": "local", "created_utc": now_utc(),
        "segments": str(fam), "source": str(src),
        "family_manifest": {k: manifest.get(k) for k in
                            ("predicate", "layout", "sources", "revision",
                             "source_revision", "signer_pubkey")},
        "layers": rows, "attestation_sample": att_rows,
        "summary": {"shards": len(rows),
                    "shards_matched": sum(r["match"] for r in rows),
                    "bytes_from_segments": sum(r["bytes_from_segments"] for r in rows),
                    "bytes_from_source": sum(r["bytes_from_source"] for r in rows),
                    "failures": failures},
        "note": ("expert bytes come from segments; safetensors header and "
                 "non-expert (attention/gate/shared-experts/norm) bytes come "
                 "from the source shard, exactly as fq_assemble writes them; "
                 "dense shards (layers 0-2, embed, head) are source "
                 "pass-through and are not claimed as reconstructed"),
    }
    return (1 if failures else 0), report


# -------------------------------------------------------- identity: remote

def fetch_fresh_header(transport, url: str):
    d, _ = transport.get_range(url, 0, 8)
    hlen = struct.unpack("<Q", d)[0]
    hj, _ = transport.get_range(url, 8, 8 + hlen)
    return json.loads(hj), 8 + hlen


def cmd_identity_remote(args) -> tuple[int, dict]:
    fam = args.segments
    indexes = load_indexes(fam)
    if not indexes:
        print(f"no indexes under {fam}", flush=True)
        return 2, {}
    manifest_path = fam / "fq-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    pubkey = manifest.get("signer_pubkey")
    wanted = parse_layers(args.layers)
    transport = fq_prime.Transport(pace=args.pace)
    rng = random.Random(args.seed)
    failures = 0

    # ---- phase 1: full local integrity (segment files vs index + attestation)
    local_rows = []
    for k, idx in sorted(indexes.items()):
        for layer_s, entry in sorted(idx.items(), key=lambda kv: int(kv[0])):
            layer = int(layer_s)
            if wanted is not None and layer not in wanted:
                continue
            seg = fam / entry["file"]
            payload, sig = load_attestation(fam, f"layer-{layer:03d}.k{k}", pubkey)
            seg_sha = file_sha256(seg)
            index_ok = seg_sha == entry["sha256"]
            att_ok = payload is not None and seg_sha == payload["fragment"]["sha256"]
            n_bad = 0
            for eid, (lo, hi) in entry["experts"].items():
                got = span_sha256(seg, entry["body_offset"] + lo,
                                  entry["body_offset"] + hi)
                if payload and payload.get("expert_sha256", {}).get(str(eid)) != got:
                    n_bad += 1
            ok = index_ok and att_ok and n_bad == 0 and sig != "BAD"
            failures += 0 if ok else 1
            local_rows.append({"layer": layer, "k": k, "file": entry["file"],
                               "index_sha_match": index_ok,
                               "attestation_sha_match": att_ok, "signature": sig,
                               "experts": len(entry["experts"]),
                               "expert_sha_mismatches": n_bad})
            print(f"local  layer {layer:3d} k{k}: index={'ok' if index_ok else 'BAD'} "
                  f"attestation={'ok' if att_ok else 'BAD'} sig={sig} "
                  f"expert-mismatches={n_bad}/{len(entry['experts'])}", flush=True)

    # ---- phase 2: fresh ranged re-reads of the pinned source
    remote_rows = []
    hdr_cache: dict[str, tuple[dict, int]] = {}
    for k, idx in sorted(indexes.items()):
        for layer_s, entry in sorted(idx.items(), key=lambda kv: int(kv[0])):
            layer = int(layer_s)
            if wanted is not None and layer not in wanted:
                continue
            seg_path = fam / entry["file"]
            meta = segment_meta(fam, entry)
            if meta.get("predicate") != "repack-of":
                print(f"skip remote check of {entry['file']}: predicate "
                      f"{meta.get('predicate')} (use --check derived)", flush=True)
                continue
            url = (f"https://huggingface.co/{meta['source_repo']}/resolve/"
                   f"{meta['source_revision']}/{meta['source_file']}")
            if url not in hdr_cache:
                hdr_cache[url] = fetch_fresh_header(transport, url)
            hdr, body_off = hdr_cache[url]
            reader = SegmentReader(seg_path)
            try:
                eids = sorted(entry["experts"], key=int)
                picks = (eids if args.sample == "all"
                         else rng.sample(eids, min(int(args.sample), len(eids))))
                for eid_s in sorted(picks, key=int):
                    names = sorted(
                        (n for n in hdr if EXPERT_RE.match(n)
                         and int(EXPERT_RE.match(n).group(2)) == int(eid_s)
                         and int(EXPERT_RE.match(n).group(1)) == layer),
                        key=expert_key)
                    lo = min(hdr[n]["data_offsets"][0] for n in names)
                    hi = max(hdr[n]["data_offsets"][1] for n in names)
                    blob, _ = transport.get_range(url, body_off + lo, body_off + hi)
                    n_bad = 0
                    for n in names:
                        a, b = hdr[n]["data_offsets"]
                        if blob[a - lo:b - lo] != reader.tensor_bytes(n):
                            n_bad += 1
                    slo, shi = entry["experts"][eid_s]
                    seg_sha = span_sha256(seg_path, entry["body_offset"] + slo,
                                          entry["body_offset"] + shi)
                    canon = hashlib.sha256()
                    for n in names:
                        a, b = hdr[n]["data_offsets"]
                        canon.update(blob[a - lo:b - lo])
                    ok = n_bad == 0 and canon.hexdigest() == seg_sha
                    failures += 0 if ok else 1
                    remote_rows.append({
                        "layer": layer, "k": k, "expert": int(eid_s),
                        "tensors": len(names), "bytes": hi - lo,
                        "byte_equal": n_bad == 0,
                        "source_sha256": canon.hexdigest(),
                        "segment_sha256": seg_sha})
                    print(f"remote layer {layer:3d} k{k} expert {eid_s:>3}: "
                          f"{'BYTES EQUAL' if ok else 'MISMATCH'} "
                          f"({(hi - lo) / 1e6:.1f}MB re-fetched)", flush=True)
            finally:
                reader.close()

    # ---- phase 3: shared profiles, re-fetched and compared IN FULL
    profile_rows = []
    shared_idx = load_shared_index(fam)
    for layer_s, entry in sorted(shared_idx.items(), key=lambda kv: int(kv[0])):
        layer = int(layer_s)
        if wanted is not None and layer not in wanted:
            continue
        prof_path = fam / entry["file"]
        meta = read_header(prof_path)[0].get("__metadata__", {})
        url = (f"https://huggingface.co/{meta['source_repo']}/resolve/"
               f"{meta['source_revision']}/{meta['source_file']}")
        if url not in hdr_cache:
            hdr_cache[url] = fetch_fresh_header(transport, url)
        hdr, body_off = hdr_cache[url]
        reader = SegmentReader(prof_path)
        try:
            n_bad = 0
            for n in reader.hdr:
                a, b = hdr[n]["data_offsets"]
                blob, _ = transport.get_range(url, body_off + a, body_off + b)
                if blob != reader.tensor_bytes(n):
                    n_bad += 1
            ok = n_bad == 0
            failures += 0 if ok else 1
            profile_rows.append({"layer": layer, "file": entry["file"],
                                 "tensors": len(reader.hdr),
                                 "tensor_mismatches": n_bad})
            print(f"profile layer {layer:3d}: "
                  f"{'BYTES EQUAL' if ok else 'MISMATCH'} "
                  f"({len(reader.hdr)} shared rows, full fetch)", flush=True)
        finally:
            reader.close()

    report = {
        "mode": "identity", "check": "remote", "created_utc": now_utc(),
        "segments": str(fam),
        "family_manifest": {k: manifest.get(k) for k in
                            ("predicate", "layout", "sources", "revision",
                             "source_revision", "signer_pubkey")},
        "local_integrity": local_rows,
        "remote_fragments": remote_rows,
        "shared_profiles": profile_rows,
        "transport": transport.stats(),
        "sample": args.sample, "seed": args.seed,
        "summary": {
            "segments_checked": len(local_rows),
            "experts_hashed_locally": sum(r["experts"] for r in local_rows),
            "experts_refetched": len(remote_rows),
            "experts_byte_equal": sum(r["byte_equal"] for r in remote_rows),
            "profiles_refetched": len(profile_rows),
            "profiles_byte_equal": sum(
                r["tensor_mismatches"] == 0 for r in profile_rows),
            "bytes_refetched": transport.stats()["bytes"],
            "failures": failures},
    }
    return (1 if failures else 0), report


# ------------------------------------------------------- identity: derived

def cmd_identity_derived(args) -> tuple[int, dict]:
    fam = args.segments
    indexes = load_indexes(fam)
    if not indexes:
        print(f"no indexes under {fam}", flush=True)
        return 2, {}
    manifest_path = fam / "fq-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    parent = args.parent
    if parent is None and manifest.get("parent_family"):
        parent = fam.parent.parent / manifest["parent_family"]
    if parent is None:
        parent = fam.parent / "shared-h"
    wanted = parse_layers(args.layers)
    pubkey = manifest.get("signer_pubkey")
    parent_sha_cache: dict[Path, str] = {}

    def parent_sha(p: Path) -> str:
        if p not in parent_sha_cache:
            parent_sha_cache[p] = file_sha256(p)
        return parent_sha_cache[p]

    rows, failures = [], 0
    for k, idx in sorted(indexes.items()):
        for layer_s, entry in sorted(idx.items(), key=lambda kv: int(kv[0])):
            layer = int(layer_s)
            if wanted is not None and layer not in wanted:
                continue
            seg_path = fam / entry["file"]
            meta = segment_meta(fam, entry)
            if meta.get("predicate") != "derived-from":
                print(f"skip {entry['file']}: predicate {meta.get('predicate')}",
                      flush=True)
                continue
            pseg_path = parent / meta["parent_segment"]
            prof_path = parent / meta["parent_profile"]
            pin_seg_ok = parent_sha(pseg_path) == meta["parent_segment_sha256"]
            pin_prof_ok = parent_sha(prof_path) == meta["parent_profile_sha256"]
            payload, sig = load_attestation(fam, f"layer-{layer:03d}.k{k}", pubkey)

            exp = SegmentReader(seg_path)
            par = SegmentReader(pseg_path)
            prof = SegmentReader(prof_path)
            try:
                n_verbatim = n_replicated = n_bad = 0
                b_verbatim = b_replicated = 0
                for name in exp.hdr:
                    m = EXPERT_RE.match(name)
                    _, _, proj, rank, comp = m.groups()
                    got = exp.tensor_bytes(name)
                    if name in par.hdr:
                        want = par.tensor_bytes(name)
                        n_verbatim += 1
                        b_verbatim += len(got)
                    else:
                        shared_comp = "suh" if proj != "down_proj" else "svh"
                        if comp != shared_comp:
                            n_bad += 1
                            continue
                        pname = (f"model.layers.{layer}.mlp.experts.shared_h."
                                 f"{proj}.rank{rank}.{comp}")
                        want = prof.tensor_bytes(pname)
                        n_replicated += 1
                        b_replicated += len(got)
                    if got != want:
                        n_bad += 1
                ok = (n_bad == 0 and pin_seg_ok and pin_prof_ok
                      and sig != "BAD")
                failures += 0 if ok else 1
                rows.append({
                    "layer": layer, "k": k, "file": entry["file"],
                    "experts": len(entry["experts"]),
                    "verbatim_tensors": n_verbatim,
                    "replicated_tensors": n_replicated,
                    "verbatim_bytes": b_verbatim,
                    "replicated_bytes": b_replicated,
                    "tensor_mismatches": n_bad,
                    "parent_segment_sha_match": pin_seg_ok,
                    "parent_profile_sha_match": pin_prof_ok,
                    "signature": sig})
                print(f"derived layer {layer:3d} k{k}: "
                      f"{'EXACT' if ok else 'MISMATCH'} "
                      f"{len(entry['experts'])} experts, "
                      f"{n_verbatim} verbatim + {n_replicated} replicated "
                      f"tensors, parents "
                      f"{'pinned-ok' if pin_seg_ok and pin_prof_ok else 'PIN-BAD'}",
                      flush=True)
            finally:
                exp.close()
                par.close()
                prof.close()

    report = {
        "mode": "identity", "check": "derived", "created_utc": now_utc(),
        "segments": str(fam), "parent": str(parent),
        "family_manifest": {k: manifest.get(k) for k in
                            ("predicate", "layout", "derived_rule",
                             "parent_family", "signer_pubkey")},
        "layers": rows,
        "summary": {
            "segments_checked": len(rows),
            "experts_checked": sum(r["experts"] for r in rows),
            "verbatim_bytes": sum(r["verbatim_bytes"] for r in rows),
            "replicated_bytes": sum(r["replicated_bytes"] for r in rows),
            "failures": failures},
        "note": ("every expanded expert re-derived from parent shared-h "
                 "segment + shared profile and byte-compared in full; parent "
                 "files re-hashed against the sha256 pins in the expanded "
                 "segment metadata — chain this to the parent family's "
                 "--check remote run for source-level identity"),
    }
    return (1 if failures else 0), report


# ------------------------------------------------------------- similarity

def compute_metrics(a, b) -> dict:
    """cosine / relative Frobenius error / max |diff| between two same-shape
    tensors (torch tensors or numpy arrays); bitwise equality on raw dtypes."""
    try:
        import torch
        is_torch = isinstance(a, torch.Tensor)
    except ImportError:
        is_torch = False
    if is_torch:
        import torch
        equal = bool(torch.equal(a, b))
        af, bf = a.double().flatten(), b.double().flatten()
        cos = torch.nn.functional.cosine_similarity(af, bf, dim=0).item()
        rel = ((af - bf).norm() / bf.norm()).item()
        mad = (af - bf).abs().max().item()
    else:
        import numpy as np
        a = np.asarray(a)
        b = np.asarray(b)
        equal = a.dtype == b.dtype and bool(np.array_equal(
            a.view(np.uint8) if a.dtype.kind not in "fc" else a,
            b.view(np.uint8) if b.dtype.kind not in "fc" else b))
        af = a.astype(np.float64).ravel()
        bf = b.astype(np.float64).ravel()
        cos = float(af @ bf / (np.linalg.norm(af) * np.linalg.norm(bf)))
        rel = float(np.linalg.norm(af - bf) / np.linalg.norm(bf))
        mad = float(np.abs(af - bf).max())
    return {"cosine": cos, "rel_frob": rel, "max_abs": mad,
            "bitwise_equal": equal}


DTYPE_MAP = {"F16": "float16", "BF16": "bfloat16", "I16": "int16", "I32": "int32"}


class FamilyView:
    """Read-side view of one segment family for similarity checks."""

    def __init__(self, label: str, path: Path, ks: list[int] | None = None):
        self.label, self.path = label, path
        indexes = load_indexes(path)
        self.indexes = ({k: v for k, v in indexes.items() if k in ks}
                        if ks else indexes)
        if not self.indexes:
            raise FileNotFoundError(f"{label}: no usable index under {path}")
        self.shared = load_shared_index(path)
        self._readers: dict[Path, SegmentReader] = {}
        any_entry = next(iter(next(iter(self.indexes.values())).values()))
        self.layout = segment_meta(path, any_entry).get("layout")

    def reader(self, file: str) -> SegmentReader:
        p = self.path / file
        if p not in self._readers:
            self._readers[p] = SegmentReader(p)
        return self._readers[p]

    def experts(self, layer: int) -> dict[int, int]:
        out = {}
        for k, idx in self.indexes.items():
            entry = idx.get(str(layer))
            if entry:
                for eid in entry["experts"]:
                    out[int(eid)] = k
        return out

    def _tensor(self, reader: SegmentReader, name: str):
        import torch
        t = reader.hdr[name]
        raw = reader.tensor_bytes(name)
        x = torch.frombuffer(bytearray(raw), dtype=getattr(torch, DTYPE_MAP[t["dtype"]]))
        return x.reshape(t["shape"]) if t["shape"] else x.reshape(())

    def load_slot(self, layer: int, eid: int, proj: str, rank: int) -> dict:
        k = self.experts(layer)[eid]
        entry = self.indexes[k][str(layer)]
        seg = self.reader(entry["file"])
        pre = f"model.layers.{layer}.mlp.experts.{eid}.{proj}.rank{rank}."
        out = {"k": k}
        for comp in ("trellis", "suh", "svh", "mcg"):
            name = pre + comp
            if name in seg.hdr:
                out[comp] = self._tensor(seg, name)
        missing = {"suh", "svh"} - set(out)
        if missing:  # shared_h_v1: H-side row lives in the layer profile
            prof_entry = self.shared.get(str(layer))
            if prof_entry is None:
                raise KeyError(f"{self.label}: no shared profile for layer {layer}")
            prof = self.reader(prof_entry["file"])
            comp = missing.pop()
            pname = f"model.layers.{layer}.mlp.experts.shared_h.{proj}.rank{rank}.{comp}"
            out[comp] = self._tensor(prof, pname)
        return out

    def close(self):
        for r in self._readers.values():
            r.close()
        self._readers.clear()


class BF16View:
    """Ground-truth reader over the base model's HF snapshot."""

    def __init__(self, snapshot: Path):
        self.snapshot = snapshot
        self.weight_map = json.loads(
            (snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        self._headers: dict[str, tuple[dict, int]] = {}

    def weight(self, layer: int, eid: int, proj: str):
        import torch
        name = f"model.layers.{layer}.mlp.experts.{eid}.{proj}.weight"
        shard = self.weight_map[name]
        if shard not in self._headers:
            self._headers[shard] = read_header(self.snapshot / shard)
        hdr, body = self._headers[shard]
        t = hdr[name]
        a, b = t["data_offsets"]
        with open(self.snapshot / shard, "rb") as f:
            f.seek(body + a)
            raw = f.read(b - a)
        x = torch.frombuffer(bytearray(raw), dtype=getattr(torch, DTYPE_MAP[t["dtype"]]))
        return x.reshape(t["shape"])


def decode_proj(slots: list[dict], proj: str, device: str):
    """Dequantize one (expert, projection): 4 rank slices through the
    reference exllamav3 reconstruct, concatenated and oriented to the HF
    weight layout ([out_features, in_features])."""
    import torch
    from exllamav3.modules.quant.exl3 import LinearEXL3
    ws = []
    for slot in slots:
        suh = slot["suh"].to(device)
        svh = slot["svh"].to(device)
        trellis = slot["trellis"].to(device)
        mcg = slot["mcg"].to(device)
        lin = LinearEXL3(None, suh.numel(), svh.numel(),
                         suh=suh, svh=svh, trellis=trellis, mcg=mcg)
        ws.append(lin.get_weight_tensor())  # [in, out] fp16
    cat_dim = 0 if proj == "down_proj" else 1
    return torch.cat(ws, dim=cat_dim).T.contiguous()


def parse_family_spec(spec: str) -> tuple[str, Path, list[int] | None]:
    label, rest = spec.split("=", 1)
    parts = rest.split(",")
    path = Path(parts[0]).expanduser()
    ks = None
    for p in parts[1:]:
        key, val = p.split("=")
        if key == "k":
            ks = [int(x) for x in val.split("+")]
        else:
            raise ValueError(f"unknown family option {key}")
    return label, path, ks


def cmd_similarity(args) -> tuple[int, dict]:
    import torch
    device = args.device
    families = []
    for spec in args.family:
        label, path, ks = parse_family_spec(spec)
        families.append(FamilyView(label, path, ks))
    bf16 = BF16View(args.bf16) if args.bf16 else None
    layers = parse_layers(args.layers) or [3]
    rng = random.Random(args.seed)

    # sample: experts available in EVERY family at that layer, spread evenly
    samples: list[tuple[int, int]] = []
    per_layer = {}
    for layer in layers:
        avail = None
        for fam in families:
            eids = set(fam.experts(layer))
            avail = eids if avail is None else (avail & eids)
        per_layer[layer] = sorted(avail or ())
    quota, extra = divmod(args.experts, len(layers))
    for i, layer in enumerate(layers):
        n = quota + (1 if i < extra else 0)
        pool = per_layer[layer]
        if len(pool) < n:
            print(f"layer {layer}: only {len(pool)} experts present in every "
                  f"family (wanted {n})", flush=True)
            n = len(pool)
        samples.extend((layer, e) for e in sorted(rng.sample(pool, n)))

    sample_rows = []
    for layer, eid in samples:
        row = {"layer": layer, "expert": eid,
               "k": {f.label: f.experts(layer)[eid] for f in families},
               "projections": {}}
        decoded = {}
        for proj in PROJS:
            per_fam = {}
            for fam in families:
                slots = [fam.load_slot(layer, eid, proj, r) for r in range(4)]
                per_fam[fam.label] = decode_proj(slots, proj, device)
            decoded[proj] = per_fam
            comps = {}
            if bf16 is not None:
                ref = bf16.weight(layer, eid, proj).to(device)
                for fam in families:
                    comps[f"{fam.label} vs bf16"] = compute_metrics(
                        per_fam[fam.label].float(), ref.float())
            for i, fa in enumerate(families):
                for fb in families[i + 1:]:
                    comps[f"{fa.label} vs {fb.label}"] = compute_metrics(
                        per_fam[fa.label], per_fam[fb.label])
            row["projections"][proj] = comps
        sample_rows.append(row)
        brief = row["projections"]["gate_proj"]
        print(f"layer {layer:3d} expert {eid:3d}: " + "  ".join(
            f"{k}: cos={v['cosine']:.5f}" for k, v in brief.items()), flush=True)

    # aggregate per (pair, projection)
    agg: dict[str, dict[str, dict]] = {}
    for row in sample_rows:
        for proj, comps in row["projections"].items():
            for pair, m in comps.items():
                a = agg.setdefault(pair, {}).setdefault(proj, {
                    "n": 0, "cos_sum": 0.0, "cos_min": 1.0, "rel_sum": 0.0,
                    "rel_max": 0.0, "mad_max": 0.0, "bitwise_equal": True})
                a["n"] += 1
                a["cos_sum"] += m["cosine"]
                a["cos_min"] = min(a["cos_min"], m["cosine"])
                a["rel_sum"] += m["rel_frob"]
                a["rel_max"] = max(a["rel_max"], m["rel_frob"])
                a["mad_max"] = max(a["mad_max"], m["max_abs"])
                a["bitwise_equal"] &= m["bitwise_equal"]
    aggregates = {}
    for pair, per_proj in agg.items():
        aggregates[pair] = {}
        for proj, a in per_proj.items():
            aggregates[pair][proj] = {
                "n": a["n"], "cosine_mean": a["cos_sum"] / a["n"],
                "cosine_min": a["cos_min"],
                "rel_frob_mean": a["rel_sum"] / a["n"],
                "rel_frob_max": a["rel_max"], "max_abs_max": a["mad_max"],
                "bitwise_equal_all": a["bitwise_equal"]}

    for fam in families:
        fam.close()
    report = {
        "mode": "similarity", "created_utc": now_utc(),
        "families": [{"label": f.label, "path": str(f.path),
                      "layout": f.layout, "ks": sorted(f.indexes)}
                     for f in families],
        "bf16": str(args.bf16) if args.bf16 else None,
        "layers": layers, "experts_sampled": len(samples), "seed": args.seed,
        "decode": ("exllamav3 LinearEXL3.get_weight_tensor (ext.reconstruct + "
                   "H128 + diag(suh/svh)), fp16, oriented to HF layout"),
        "samples": sample_rows,
        "aggregates": aggregates,
    }
    return 0, report


# ---------------------------------------------------------------- reports

def write_reports(report: dict, json_path: Path | None, md_path: Path | None):
    if json_path:
        json_path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        print(f"json report -> {json_path}", flush=True)
    if md_path:
        md_path.write_text(render_md(report))
        print(f"markdown report -> {md_path}", flush=True)


def render_md(report: dict) -> str:
    lines = []
    if report["mode"] == "identity":
        check = report["check"]
        lines.append(f"### Identity ({check}) — {report['segments']}")
        lines.append("")
        if check == "local":
            s = report["summary"]
            lines.append("| layer | shard | match | bytes from segments | "
                         "bytes from source | assembled sha256 |")
            lines.append("|---|---|---|---|---|---|")
            for r in report["layers"]:
                lines.append(
                    f"| {r['layer']} | {r['shard']} | "
                    f"{'PASS' if r['match'] else 'FAIL'} | "
                    f"{r['bytes_from_segments']:,} | {r['bytes_from_source']:,} | "
                    f"`{r['assembled_sha256'][:16]}…` |")
            lines.append("")
            lines.append(f"**{s['shards_matched']}/{s['shards']} shards sha256-"
                         f"identical** ({s['bytes_from_segments'] / 1e9:.1f} GB "
                         f"of expert bytes reassembled from segments).")
        elif check == "remote":
            s = report["summary"]
            lines.append("| layer | k | local integrity | experts re-fetched | "
                         "byte-equal |")
            lines.append("|---|---|---|---|---|")
            per = {}
            for r in report["remote_fragments"]:
                key = (r["layer"], r["k"])
                per.setdefault(key, []).append(r)
            for lr in report["local_integrity"]:
                key = (lr["layer"], lr["k"])
                rem = per.get(key, [])
                ok_local = (lr["index_sha_match"] and lr["attestation_sha_match"]
                            and lr["expert_sha_mismatches"] == 0)
                lines.append(
                    f"| {lr['layer']} | {lr['k']} | "
                    f"{'PASS' if ok_local else 'FAIL'} "
                    f"({lr['experts']} experts hashed, sig {lr['signature']}) | "
                    f"{len(rem)} | "
                    f"{sum(r['byte_equal'] for r in rem)}/{len(rem)} |")
            lines.append("")
            msg = (f"**{s['experts_byte_equal']}/{s['experts_refetched']} sampled "
                   f"experts byte-identical to freshly range-read source bytes** "
                   f"({s['bytes_refetched'] / 1e6:.0f} MB re-fetched)")
            if s["profiles_refetched"]:
                msg += (f"; {s['profiles_byte_equal']}/{s['profiles_refetched']} "
                        f"shared profiles byte-identical (fetched in full)")
            lines.append(msg + ".")
        elif check == "derived":
            s = report["summary"]
            lines.append("| layer | k | experts | verbatim tensors | "
                         "replicated tensors | mismatches | parents pinned |")
            lines.append("|---|---|---|---|---|---|---|")
            for r in report["layers"]:
                pins = r["parent_segment_sha_match"] and r["parent_profile_sha_match"]
                lines.append(
                    f"| {r['layer']} | {r['k']} | {r['experts']} | "
                    f"{r['verbatim_tensors']} | {r['replicated_tensors']} | "
                    f"{r['tensor_mismatches']} | {'PASS' if pins else 'FAIL'} |")
            lines.append("")
            lines.append(
                f"**{s['experts_checked']} expanded experts re-derived and "
                f"byte-compared in full** ({s['verbatim_bytes'] / 1e9:.1f} GB "
                f"verbatim + {s['replicated_bytes'] / 1e9:.2f} GB replicated "
                f"rows), {s['failures']} failures.")
    else:
        lines.append("### Similarity — dequantized expert weights")
        lines.append("")
        lines.append(f"{report['experts_sampled']} experts over layers "
                     f"{report['layers'][0]}–{report['layers'][-1]}, decode: "
                     f"{report['decode']}.")
        lines.append("")
        lines.append("| pair | projection | n | cos (mean / min) | "
                     "relF (mean / max) | max abs | bitwise |")
        lines.append("|---|---|---|---|---|---|---|")
        for pair in sorted(report["aggregates"]):
            for proj in PROJS:
                a = report["aggregates"][pair].get(proj)
                if not a:
                    continue
                lines.append(
                    f"| {pair} | {proj} | {a['n']} | "
                    f"{a['cosine_mean']:.5f} / {a['cosine_min']:.5f} | "
                    f"{a['rel_frob_mean']:.4f} / {a['rel_frob_max']:.4f} | "
                    f"{a['max_abs_max']:.4f} | "
                    f"{'EQUAL' if a['bitwise_equal_all'] else '—'} |")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------- cli

def detect_check(segments: Path, source: Path | None) -> str:
    if source is not None:
        return "local"
    manifest_path = segments / "fq-manifest.json"
    predicate = None
    if manifest_path.exists():
        predicate = json.loads(manifest_path.read_text()).get("predicate")
    if predicate is None:
        indexes = load_indexes(segments)
        if indexes:
            entry = next(iter(next(iter(indexes.values())).values()))
            predicate = segment_meta(segments, entry).get("predicate")
    return "derived" if predicate == "derived-from" else "remote"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--identity", action="store_true")
    mode.add_argument("--similarity", action="store_true")
    p.add_argument("--segments", type=Path, help="family dir (identity)")
    p.add_argument("--source", type=Path,
                   help="local source snapshot -> whole-shard identity")
    p.add_argument("--parent", type=Path,
                   help="parent shared-h family dir (derived check)")
    p.add_argument("--check", choices=("auto", "local", "remote", "derived"),
                   default="auto")
    p.add_argument("--layers", default=None, help="e.g. 3-10")
    p.add_argument("--sample", default="3",
                   help="experts re-fetched per (layer,k) in remote check, "
                        "or 'all'")
    p.add_argument("--attest", default="2",
                   help="layers spot-checked against attestations in local "
                        "check, or 'all'")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--pace", type=float, default=1.0,
                   help="min seconds between HTTP requests (remote check)")
    p.add_argument("--family", action="append", default=[],
                   help="similarity family: label=path[,k=3+4]")
    p.add_argument("--bf16", type=Path, default=None,
                   help="BF16 base-model snapshot for ground-truth comparison")
    p.add_argument("--experts", type=int, default=24,
                   help="total experts to sample (similarity)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--md", type=Path, default=None)
    args = p.parse_args(argv)

    if args.similarity:
        if not args.family:
            p.error("--similarity requires at least one --family")
        rc, report = cmd_similarity(args)
    else:
        if args.segments is None:
            p.error("--identity requires --segments")
        check = args.check
        if check == "auto":
            check = detect_check(args.segments, args.source)
            print(f"identity check: {check}", flush=True)
        if check == "local":
            if args.source is None:
                p.error("--check local requires --source")
            rc, report = cmd_identity_local(args)
        elif check == "remote":
            rc, report = cmd_identity_remote(args)
        else:
            rc, report = cmd_identity_derived(args)
    if report:
        write_reports(report, args.json, args.md)
        if not args.md:
            print(render_md(report), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
