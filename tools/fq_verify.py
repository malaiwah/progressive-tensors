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

  fetched  offline full provenance for a range-fetched subset: verify the
           local derived-from line under --trust-signer, then every copied
           publisher line under separately repeated --upstream-trust-signer
           pins.  It binds files, indexes, expert spans, immutable source
           revisions and header-planning evidence.  Unsafe header planning is
           rejected unless --accept-unsafe-header-plan is explicit.

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

Trust.  Every attestation this tool reads is verified through fq_trust
against a signer the CALLER chose: --trust-signer <fingerprint> pinned out
of band, or a key listed active in keys/FINGERPRINTS (git, not the artifact
download).  The family's own fq-manifest.json is never the authority — an
attacker who rewrites a family rewrites that field too.  There is no
"unverified but probably fine" state: a missing attestation, an
unverifiable line, a placeholder signature, or a missing per-expert digest
is a FAILURE and the process exits non-zero.  --insecure-skip-signatures
still prints the byte-level report, but such a run can never exit 0.

Outputs: human-readable progress on stdout, plus --json (machine report) and
--md (markdown table) files.

Examples:
  # end-to-end shard identity, K3 family vs local brandonmusic snapshot
  fq_verify.py --identity --segments ~/fq-segments/GLM-5.2-EXL3-FQ \
      --source <snapshot-dir> --trust-signer <fingerprint> \
      --json id-k3.json --md id-k3.md

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
import fq_trust  # noqa: E402
from fq_trust import TrustError  # noqa: E402


def validate_attestation_payload(payload: dict, *, where: str) -> None:
    """Validate the fq-attestation/1 payload contract without a dev dependency."""
    def fail(field: str) -> None:
        raise TrustError(f"{where}: invalid fq-attestation/1 {field}")

    if not isinstance(payload, dict):
        fail("payload")
    if payload.get("schema") != "fq-attestation/1":
        fail("schema")
    if payload.get("predicate") not in {
            "repack-of", "derived-from", "encode-of", "equivalence-of"}:
        fail("predicate")
    fragment = payload.get("fragment")
    if (not isinstance(fragment, dict) or
            not isinstance(fragment.get("file"), str) or
            not fragment["file"] or not is_sha256(fragment.get("sha256")) or
            ("size" in fragment and
             (not isinstance(fragment["size"], int) or fragment["size"] < 1))):
        fail("fragment")
    experts = payload.get("expert_sha256")
    if (not isinstance(experts, dict) or not experts or
            any(not isinstance(eid, str) or not eid.isdigit() or
                not is_sha256(digest) for eid, digest in experts.items())):
        fail("expert_sha256")
    created = payload.get("created_utc")
    if (not isinstance(created, str) or len(created) != 20 or
            created[4:5] != "-" or created[7:8] != "-" or
            created[10:11] != "T" or created[13:14] != ":" or
            created[16:17] != ":" or created[19:20] != "Z" or
            not (created[:4] + created[5:7] + created[8:10] +
                 created[11:13] + created[14:16] + created[17:19]).isdigit()):
        fail("created_utc")
    for field in ("parents", "dependencies"):
        values = payload.get(field)
        if values is not None and (
                not isinstance(values, list) or any(
                    not isinstance(value, dict) or
                    not isinstance(value.get("file"), str) or
                    not value["file"] or not is_sha256(value.get("sha256"))
                    for value in values)):
            fail(field)
    derivation = payload.get("derivation")
    if derivation is not None and (
            not isinstance(derivation, dict) or
            not isinstance(derivation.get("rule"), str) or
            not derivation["rule"]):
        fail("derivation")



def validate_shared_profile_payload(payload: dict, *, where: str) -> None:
    """Validate legacy shared-profile attestations against the common contract.

    fq-attestation/1 predates profiles and requires expert_sha256; profiles
    instead have a per-tensor map.  Validate every shared field through the
    schema with a synthetic structural expert digest, then require the real
    profile-specific map below.
    """
    normalized = dict(payload)
    normalized.setdefault("expert_sha256", {"0": "0" * 64})
    validate_attestation_payload(normalized, where=where)
    digests = payload.get("tensor_sha256")
    if not isinstance(digests, dict) or not digests:
        raise TrustError(f"{where}: shared_profile lacks tensor_sha256")
    if any(not isinstance(name, str) or not isinstance(digest, str) or
           len(digest) != 64 for name, digest in digests.items()):
        raise TrustError(f"{where}: shared_profile has invalid tensor_sha256")

def header_sha256(path: Path, body_offset: int) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read(body_offset)).hexdigest()


def fragment_binding(path: Path, entry: dict, payload: dict) -> dict:
    """Bind current bytes and signed claims; absence is a mismatch."""
    fragment = payload.get("fragment") or {}
    size = path.stat().st_size
    sha = file_sha256(path)
    body_offset = entry.get("body_offset")
    signed_body_offset = fragment.get("body_offset")
    body_ok = (signed_body_offset is None or
               signed_body_offset == body_offset)
    signed_header = fragment.get("header_sha256")
    header_ok = (signed_header is None or
                 (body_ok and isinstance(body_offset, int) and
                  header_sha256(path, body_offset) == signed_header))
    return {
        "index_sha_match": sha == entry.get("sha256"),
        "index_size_match": size == entry.get("size"),
        "attestation_sha_match": sha == fragment.get("sha256"),
        "attestation_size_match": size == fragment.get("size"),
        "fragment_file_match": path.name == fragment.get("file"),
        "body_offset_match": body_ok,
        "header_sha_match": header_ok,
    }


def is_sha256(value) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value))


def binding_ok(binding: dict) -> bool:
    return all(binding.values())
CHUNK = 1 << 24
PROJS = ("gate_proj", "up_proj", "down_proj")


def is_immutable_revision(value) -> bool:
    """A resolved Git/HF object ID, never a mutable branch or tag name."""
    return (isinstance(value, str) and len(value) >= 40 and
            all(c in "0123456789abcdef" for c in value))


def evidence_source_slug(repo: str, revision: str, subdir: str) -> str | None:
    """Canonical, traversal-free name of a copied upstream evidence directory."""
    if (not isinstance(subdir, str) or subdir.startswith("/") or
            any(part in ("", ".", "..") for part in subdir.split("/")
                if subdir)):
        return None
    suffix = f":{subdir.replace('/', '__')}" if subdir else ""
    return f"{repo.lower().replace('/', '__')}@{revision[:12]}{suffix}"

# load_attestation() states.  Only VERIFIED is evidence; NOT_CHECKED exists
# solely because the operator passed --insecure-skip-signatures and asked to
# be told nothing is proven.
ATT_VERIFIED = "verified"
ATT_NOT_CHECKED = "NOT CHECKED"
ATT_MISSING = "MISSING"


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


def make_verifier(args, manifest: dict) -> fq_trust.Verifier:
    """The trust decision for this run, from the CLI and the trust root.

    Never from the artifact: the family's own fq-manifest.json names a
    signer, and an attacker who rewrites the family rewrites that field
    too.  fq_trust resolves --trust-signer against keys/FINGERPRINTS (git,
    not the download) and fails closed when the caller pinned nothing and
    the trust root cannot vouch for the claimed key.
    """
    return fq_trust.Verifier.from_args(args, manifest=manifest)


def load_attestation(fam: Path, stem: str, verifier: fq_trust.Verifier,
                     fragment_file: str | None = None, *,
                     shared_profile: bool = False):
    """(payload, state) for attestations/<stem>.jsonl.

    The file is JSON Lines and is treated as such: every line is checked,
    and the returned payload is merged from the lines that a TRUSTED key
    signed and that name this fragment.  Decoding is never mistaken for
    verifying, and the signer never comes from the artifact.

    state is ATT_VERIFIED when at least one trusted line covered the
    fragment, ATT_NOT_CHECKED on the --insecure-skip-signatures rung,
    ATT_MISSING when the file is absent, and "BAD: <reason>" when lines
    exist but none of them can be trusted.  Every state other than
    ATT_VERIFIED (and ATT_NOT_CHECKED, which the operator asked for
    explicitly) is a verification FAILURE at the call site.
    """
    p = fam / "attestations" / f"{stem}.jsonl"
    if not p.exists():
        return None, ATT_MISSING
    merged: dict = {}
    reasons: list[str] = []
    for n, raw in enumerate(p.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        try:
            payload = verifier.verify_envelope(json.loads(raw),
                                               where=f"{p.name}:{n}")
            (validate_shared_profile_payload if shared_profile
             else validate_attestation_payload)(payload, where=f"{p.name}:{n}")
        except (TrustError, json.JSONDecodeError, TypeError) as e:
            reasons.append(f"line {n}: {e}")
            continue
        frag = payload.get("fragment") or {}
        if fragment_file and frag.get("file") not in (None, fragment_file):
            continue
        if not merged:
            merged = dict(payload)
            merged["expert_sha256"] = dict(payload.get("expert_sha256") or {})
        else:
            merged["expert_sha256"].update(payload.get("expert_sha256") or {})
            merged.setdefault("fragment", frag)
    if not merged:
        detail = "; ".join(reasons) or "no signed line names this fragment"
        return None, f"BAD: {detail}"
    state = (ATT_NOT_CHECKED if verifier.rung == fq_trust.RUNG_NONE
             else ATT_VERIFIED)
    return merged, state



def load_attestation_path(path: Path, verifier: fq_trust.Verifier,
                          fragment_file: str) -> tuple[dict | None, str]:
    """Merge every verified copied publisher line that covers one fragment."""
    if not path.exists():
        return None, ATT_MISSING
    merged: dict | None = None
    reasons = []
    for n, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        try:
            payload = verifier.verify_envelope(json.loads(raw),
                                               where=f"{path.name}:{n}")
            validate_attestation_payload(payload, where=f"{path.name}:{n}")
        except (TrustError, json.JSONDecodeError, TypeError) as e:
            reasons.append(f"line {n}: {e}")
            continue
        if (payload.get("fragment") or {}).get("file") != fragment_file:
            continue
        if merged is None:
            merged = dict(payload)
            merged["expert_sha256"] = dict(payload["expert_sha256"])
        elif any(merged.get(field) != payload.get(field)
                 for field in ("fragment", "predicate", "materials")):
            reasons.append(f"line {n}: conflicting signed fragment claims")
            continue
        else:
            merged["expert_sha256"].update(payload["expert_sha256"])
    if merged is None:
        return None, f"BAD: {'; '.join(reasons) or 'no signed line names this fragment'}"
    state = ATT_NOT_CHECKED if verifier.rung == fq_trust.RUNG_NONE else ATT_VERIFIED
    return merged, state

def attestation_ok(state: str | None) -> bool:
    """Only a positive verification counts.  MISSING, BAD and (on the
    explicit --insecure-skip-signatures rung) NOT CHECKED are all failures
    of the proof; the flag chooses whether the run continues, not whether
    the evidence exists."""
    return state == ATT_VERIFIED


def trust_report(verifier: fq_trust.Verifier) -> dict:
    return {"rung": verifier.rung, "signer": verifier.fingerprint,
            "key_id": verifier.key_id,
            "signatures_verified": verifier.checked,
            "summary": verifier.summary()}


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
    verifier = make_verifier(args, manifest)
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
                fam, f"layer-{layer:03d}.k{k}", verifier, entry["file"])
            if payload is None:
                # No trusted attestation is not "nothing to check": it is a
                # failed check.  An unsigned fragment proves nothing.
                failures += 1
                att_rows.append({"layer": layer, "k": k, "signature": sig,
                                 "status": "no trusted attestation"})
                print(f"attestation layer {layer:3d} k{k}: {sig}", flush=True)
                continue
            seg = fam / entry["file"]
            frag_ok = file_sha256(seg) == payload.get("fragment", {}).get("sha256")
            n_bad = 0
            for eid, (lo, hi) in entry["experts"].items():
                got = span_sha256(seg, entry["body_offset"] + lo,
                                  entry["body_offset"] + hi)
                # a MISSING digest is a mismatch: None never equals a sha
                if payload.get("expert_sha256", {}).get(str(eid)) != got:
                    n_bad += 1
            ok = frag_ok and n_bad == 0 and attestation_ok(sig)
            failures += 0 if ok else 1
            att_rows.append({"layer": layer, "k": k, "signature": sig,
                             "fragment_sha_match": frag_ok,
                             "experts_checked": len(entry["experts"]),
                             "expert_sha_mismatches": n_bad})
            print(f"attestation layer {layer:3d} k{k}: sig={sig} fragment="
                  f"{'ok' if frag_ok else 'MISMATCH'} experts="
                  f"{len(entry['experts'])} bad={n_bad}", flush=True)

    print(verifier.summary(), flush=True)
    report = {
        "mode": "identity", "check": "local", "created_utc": now_utc(),
        "segments": str(fam), "source": str(src),
        "family_manifest": {k: manifest.get(k) for k in
                            ("predicate", "layout", "sources", "revision",
                             "source_revision", "signer_pubkey")},
        "trust": trust_report(verifier),
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
    verifier = make_verifier(args, manifest)
    wanted = parse_layers(args.layers)
    transport = fq_prime.Transport(pace=args.pace)
    rng = random.Random(args.seed)
    failures = 0

    # ---- phase 1: full local integrity (segment files vs index + attestation)
    local_rows = []
    local_payloads: dict[tuple[int, int], dict] = {}
    for k, idx in sorted(indexes.items()):
        for layer_s, entry in sorted(idx.items(), key=lambda kv: int(kv[0])):
            layer = int(layer_s)
            if wanted is not None and layer not in wanted:
                continue
            seg = fam / entry["file"]
            payload, sig = load_attestation(fam, f"layer-{layer:03d}.k{k}",
                                            verifier, entry["file"])
            binding = fragment_binding(seg, entry, payload or {})
            n_bad = 0
            # No payload means no per-expert digests to check against: that is
            # len(experts) unproven spans, not zero mismatches.
            for eid, (lo, hi) in entry["experts"].items():
                got = span_sha256(seg, entry["body_offset"] + lo,
                                  entry["body_offset"] + hi)
                if (payload or {}).get("expert_sha256", {}).get(str(eid)) != got:
                    n_bad += 1
            ok = bool(binding) and binding_ok(binding) and n_bad == 0 and attestation_ok(sig)
            failures += 0 if ok else 1
            if payload is not None:
                local_payloads[layer, k] = payload
            local_rows.append({"layer": layer, "k": k, "file": entry["file"],
                               **binding, "signature": sig,
                               "experts": len(entry["experts"]),
                               "expert_sha_mismatches": n_bad})
            print(f"local  layer {layer:3d} k{k}: index="
                  f"{'ok' if binding.get('index_sha_match') else 'BAD'} "
                  f"attestation={'ok' if binding.get('attestation_sha_match') else 'BAD'} "
                  f"sig={sig} expert-mismatches={n_bad}/{len(entry['experts'])}",
                  flush=True)

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
            payload = local_payloads.get((layer, k), {})
            materials = payload.get("materials") or {}
            meta = segment_meta(fam, entry)
            signed_source_ok = (
                payload.get("predicate") == "repack-of" and
                all(isinstance(materials.get(field), str) and materials[field]
                    for field in ("repo", "revision", "file")) and
                all(meta.get(f"source_{field}") == materials[field]
                    for field in ("repo", "revision", "file")))
            if not signed_source_ok:
                failures += 1
                remote_rows.append({"layer": layer, "k": k, "file": entry["file"],
                                    "signed_source_match": False})
                print(f"remote layer {layer:3d} k{k}: signed source binding BAD",
                      flush=True)
                continue
            url = (f"https://huggingface.co/{materials['repo']}/resolve/"
                   f"{materials['revision']}/{materials['file']}")
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

    # ---- phase 3: authenticate shared profiles, then re-fetch them in full.
    profile_rows = []
    shared_idx = load_shared_index(fam)
    for layer_s, entry in sorted(shared_idx.items(), key=lambda kv: int(kv[0])):
        layer = int(layer_s)
        if wanted is not None and layer not in wanted:
            continue
        prof_path = fam / entry["file"]
        payload, sig = load_attestation(
            fam, f"layer-{layer:03d}.shared", verifier, entry["file"],
            shared_profile=True)
        binding = (fragment_binding(prof_path, entry, payload)
                   if payload is not None else {})
        meta = read_header(prof_path)[0].get("__metadata__", {})
        materials = (payload or {}).get("materials") or {}
        signed_source_ok = (
            (payload or {}).get("predicate") == "repack-of" and
            (payload or {}).get("kind") == "shared_profile" and
            all(isinstance(materials.get(field), str) and materials[field]
                for field in ("repo", "revision", "file")) and
            all(meta.get(f"source_{field}") == materials[field]
                for field in ("repo", "revision", "file")))
        deps = [
            dep for (dep_layer, _), expert_payload in local_payloads.items()
            if dep_layer == layer
            for dep in expert_payload.get("dependencies", [])
            if dep.get("role") == "shared_profile"
        ]
        dependency_ok = (bool(deps) and all(
            dep.get("file") == entry["file"] and dep.get("sha256") == entry.get("sha256")
            for dep in deps))
        reader = SegmentReader(prof_path)
        try:
            tensor_digests = (payload or {}).get("tensor_sha256") or {}
            tensor_bad = sum(
                tensor_digests.get(name) != hashlib.sha256(reader.tensor_bytes(name)).hexdigest()
                for name in reader.hdr)
            local_ok = (bool(binding) and binding_ok(binding) and
                        attestation_ok(sig) and signed_source_ok and
                        dependency_ok and tensor_bad == 0)
            n_bad = 0
            if signed_source_ok:
                url = (f"https://huggingface.co/{materials['repo']}/resolve/"
                       f"{materials['revision']}/{materials['file']}")
                if url not in hdr_cache:
                    hdr_cache[url] = fetch_fresh_header(transport, url)
                hdr, body_off = hdr_cache[url]
                for name in reader.hdr:
                    if name not in hdr:
                        n_bad += 1
                        continue
                    a, b = hdr[name]["data_offsets"]
                    blob, _ = transport.get_range(url, body_off + a, body_off + b)
                    if blob != reader.tensor_bytes(name):
                        n_bad += 1
            else:
                n_bad = len(reader.hdr)
            ok = local_ok and n_bad == 0
            failures += 0 if ok else 1
            profile_rows.append({
                "layer": layer, "file": entry["file"], **binding,
                "signature": sig, "signed_source_match": signed_source_ok,
                "dependency_match": dependency_ok,
                "tensors": len(reader.hdr), "tensor_sha_mismatches": tensor_bad,
                "tensor_mismatches": n_bad})
            print(f"profile layer {layer:3d}: "
                  f"{'BYTES EQUAL' if ok else 'MISMATCH'} sig={sig} "
                  f"deps={'ok' if dependency_ok else 'BAD'} "
                  f"({len(reader.hdr)} shared rows, full fetch)", flush=True)
        finally:
            reader.close()

    print(verifier.summary(), flush=True)
    report = {
        "mode": "identity", "check": "remote", "created_utc": now_utc(),
        "segments": str(fam),
        "family_manifest": {k: manifest.get(k) for k in
                            ("predicate", "layout", "sources", "revision",
                             "source_revision", "signer_pubkey")},
        "trust": trust_report(verifier),
        "local_integrity": local_rows,
        "remote_fragments": remote_rows,
        "shared_profiles": profile_rows,
        "transport": transport.stats(),
        "sample": args.sample, "seed": args.seed,
        "summary": {
            "segments_checked": len(local_rows),
            "experts_hashed_locally": sum(r["experts"] for r in local_rows),
            "experts_refetched": sum("expert" in r for r in remote_rows),
            "experts_byte_equal": sum(r.get("byte_equal", False) for r in remote_rows),
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
    verifier = make_verifier(args, manifest)
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
            payload, sig = load_attestation(fam, f"layer-{layer:03d}.k{k}",
                                            verifier, entry["file"])
            parents = (payload or {}).get("parents") or []
            signed_seg = next((p for p in parents
                               if p.get("role") == "expert_segment"), None)
            signed_prof = next((p for p in parents
                                if p.get("role") == "shared_profile"), None)
            derivation = (payload or {}).get("derivation") or {}
            signed_parents_ok = (
                (payload or {}).get("predicate") == "derived-from" and
                signed_seg is not None and signed_prof is not None and
                all(isinstance(p.get(field), str) and p[field]
                    for p in (signed_seg, signed_prof)
                    for field in ("file", "sha256")) and
                isinstance(derivation.get("rule"), str) and derivation["rule"])
            metadata_ok = signed_parents_ok and all((
                meta.get("predicate") == payload.get("predicate"),
                meta.get("parent_segment") == signed_seg["file"],
                meta.get("parent_segment_sha256") == signed_seg["sha256"],
                meta.get("parent_profile") == signed_prof["file"],
                meta.get("parent_profile_sha256") == signed_prof["sha256"],
                meta.get("derived_rule") == derivation["rule"],
            ))
            binding = (fragment_binding(seg_path, entry, payload)
                       if payload is not None else {})
            pseg_path = parent / signed_seg["file"] if signed_parents_ok else None
            prof_path = parent / signed_prof["file"] if signed_parents_ok else None
            pin_seg_ok = (pseg_path is not None and pseg_path.exists() and
                          parent_sha(pseg_path) == signed_seg["sha256"])
            pin_prof_ok = (prof_path is not None and prof_path.exists() and
                           parent_sha(prof_path) == signed_prof["sha256"])
            if not pin_seg_ok or not pin_prof_ok:
                failures += 1
                rows.append({
                    "layer": layer, "k": k, "file": entry["file"], **binding,
                    "experts": len(entry["experts"]), "verbatim_tensors": 0,
                    "replicated_tensors": 0, "verbatim_bytes": 0,
                    "replicated_bytes": 0, "tensor_mismatches": len(entry["experts"]),
                    "parent_segment_sha_match": pin_seg_ok,
                    "parent_profile_sha_match": pin_prof_ok,
                    "metadata_matches_attestation": metadata_ok,
                    "signature": sig})
                print(f"derived layer {layer:3d} k{k}: MISMATCH parents PIN-BAD",
                      flush=True)
                continue
            exp = SegmentReader(seg_path)
            par = SegmentReader(pseg_path)
            prof = SegmentReader(prof_path)
            try:
                n_verbatim = n_replicated = n_bad = 0
                b_verbatim = b_replicated = 0
                for name in exp.hdr:
                    m = EXPERT_RE.match(name)
                    if m is None:
                        n_bad += 1
                        continue
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
                digest_bad = sum(
                    (payload or {}).get("expert_sha256", {}).get(str(eid)) !=
                    span_sha256(seg_path, entry["body_offset"] + lo,
                                entry["body_offset"] + hi)
                    for eid, (lo, hi) in entry["experts"].items())
                ok = (n_bad == 0 and digest_bad == 0 and bool(binding) and
                      binding_ok(binding) and metadata_ok and pin_seg_ok and
                      pin_prof_ok and attestation_ok(sig))
                failures += 0 if ok else 1
                rows.append({
                    "layer": layer, "k": k, "file": entry["file"], **binding,
                    "experts": len(entry["experts"]),
                    "verbatim_tensors": n_verbatim,
                    "replicated_tensors": n_replicated,
                    "verbatim_bytes": b_verbatim,
                    "replicated_bytes": b_replicated,
                    "tensor_mismatches": n_bad,
                    "expert_sha_mismatches": digest_bad,
                    "parent_segment_sha_match": pin_seg_ok,
                    "parent_profile_sha_match": pin_prof_ok,
                    "metadata_matches_attestation": metadata_ok,
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

    print(verifier.summary(), flush=True)
    report = {
        "mode": "identity", "check": "derived", "created_utc": now_utc(),
        "segments": str(fam), "parent": str(parent),
        "family_manifest": {k: manifest.get(k) for k in
                            ("predicate", "layout", "derived_rule",
                             "parent_family", "signer_pubkey")},
        "trust": trust_report(verifier),
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



# ------------------------------------------------------- identity: fetched

def cmd_identity_fetched(args) -> tuple[int, dict]:
    """Verify a locally signed range subset and copied publisher evidence."""
    fam = args.segments
    indexes = load_indexes(fam)
    if not indexes:
        print(f"no indexes under {fam}", flush=True)
        return 2, {}
    manifest_path = fam / "fq-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    verifier = make_verifier(args, manifest)
    if not args.upstream_trust_signer:
        raise TrustError("--check fetched requires --upstream-trust-signer for "
                         "each publisher key pinned independently of the local key")
    upstream = {
        fq_trust.Verifier.resolve(trust_signer=spec, trust_root=args.trust_root):
        spec for spec in args.upstream_trust_signer
    }
    # The dictionary above keys by verifier identity, not artifact claims.
    by_key = {v.fingerprint: v for v in upstream}
    wanted = parse_layers(args.layers)
    failures, unsafe = 0, 0
    rows = []
    for k, idx in sorted(indexes.items()):
        for layer_s, entry in sorted(idx.items(), key=lambda kv: int(kv[0])):
            layer = int(layer_s)
            if wanted is not None and layer not in wanted:
                continue
            seg = fam / entry["file"]
            payload, sig = load_attestation(fam, f"layer-{layer:03d}.k{k}",
                                            verifier, entry["file"])
            binding = fragment_binding(seg, entry, payload or {})
            digest_bad = sum(
                (payload or {}).get("expert_sha256", {}).get(str(eid)) !=
                span_sha256(seg, entry["body_offset"] + lo,
                            entry["body_offset"] + hi)
                for eid, (lo, hi) in entry["experts"].items())
            parents = (payload or {}).get("parents") or []
            provenance_ok = ((payload or {}).get("predicate") == "derived-from" and
                             ((payload or {}).get("derivation") or {}).get("rule") ==
                             "range_subset_v1")
            parent_rows = []
            covered_experts: list[str] = []
            for parent_record in parents:
                repo, revision = parent_record.get("repo"), parent_record.get("revision")
                parent_file = parent_record.get("file")
                keyid = parent_record.get("keyid")
                subdir = parent_record.get("subdir")
                evidence_source = parent_record.get("evidence_source")
                claimed_experts = parent_record.get("experts")
                valid_experts = (isinstance(claimed_experts, list) and
                                 bool(claimed_experts) and
                                 all(str(eid) in entry["experts"]
                                     for eid in claimed_experts))
                if valid_experts:
                    covered_experts.extend(str(eid) for eid in claimed_experts)
                expected_source = (evidence_source_slug(repo, revision, subdir)
                                   if isinstance(repo, str) and
                                   is_immutable_revision(revision) else None)
                valid_parent = (valid_experts and bool(expected_source) and
                                evidence_source == expected_source and
                                isinstance(parent_file, str) and bool(parent_file) and
                                is_sha256(keyid))
                parent_verifier = by_key.get(keyid) if valid_parent else None
                copied = (fam / "attestations" / evidence_source /
                          f"layer-{layer:03d}.k{k}.jsonl"
                          if parent_verifier is not None else Path(""))
                upstream_payload, upstream_sig = (
                    load_attestation_path(copied, parent_verifier, parent_file)
                    if parent_verifier is not None else
                    (None, "BAD: publisher key is not independently pinned"))
                upstream_frag = (upstream_payload or {}).get("fragment") or {}
                materials = (upstream_payload or {}).get("materials") or {}
                header_method = parent_record.get("header_authentication")
                claimed_header = parent_record.get("header_sha256")
                publisher_header = upstream_frag.get("header_sha256")
                if header_method == "attested-header-digest":
                    header_evidence = (
                        is_sha256(claimed_header) and
                        is_sha256(publisher_header) and
                        claimed_header == publisher_header and
                        isinstance(upstream_frag.get("body_offset"), int))
                elif header_method == "full-fragment":
                    # The parent fragment SHA is verified immediately below.
                    header_evidence = True
                else:
                    header_evidence = False
                claims_ok = valid_experts and all(
                    (upstream_payload or {}).get("expert_sha256", {}).get(str(eid)) ==
                    (payload or {}).get("expert_sha256", {}).get(str(eid))
                    for eid in claimed_experts)
                material_file = materials.get("file")
                terminal_source_ok = (
                    parent_record.get("role") == "source_fragment" and
                    (upstream_payload or {}).get("predicate") == "repack-of" and
                    isinstance(material_file, str) and bool(material_file))
                release_claim = parent_record.get("signed_release_manifest", False)
                release_ok = release_claim is False
                release = ("not-release-covered" if release_ok else
                           "claimed-release-not-available-offline")
                parent_ok = (
                    valid_parent and upstream_payload is not None and
                    attestation_ok(upstream_sig) and terminal_source_ok and
                    upstream_frag.get("sha256") == parent_record.get("sha256") and
                    upstream_frag.get("size") == parent_record.get("size") and
                    claims_ok and release_ok)
                authenticated = header_evidence and parent_ok
                explicit_unsafe = (
                    header_method == "NONE (--header-trust unsafe)" and
                    parent_record.get("header_authenticated") is False and parent_ok)
                header_status = ("authenticated" if authenticated else
                                 "unsafe" if explicit_unsafe else "invalid")
                if header_status == "unsafe":
                    unsafe += 1
                parent_rows.append({
                    "repo": repo, "revision": revision, "file": parent_file,
                    "signature": upstream_sig, "match": parent_ok,
                    "experts": claimed_experts, "header_status": header_status,
                    "header_authenticated": authenticated,
                    "release_coverage": release})
            expected_experts = set(entry["experts"])
            coverage_ok = (bool(parent_rows) and set(covered_experts) == expected_experts and
                           len(covered_experts) == len(set(covered_experts)))
            parents_ok = coverage_ok and all(p["match"] for p in parent_rows)
            headers_ok = parents_ok and all(
                p["header_status"] == "authenticated" or
                (p["header_status"] == "unsafe" and args.accept_unsafe_header_plan)
                for p in parent_rows)
            ok = (bool(binding) and binding_ok(binding) and digest_bad == 0 and
                  provenance_ok and parents_ok and headers_ok and
                  attestation_ok(sig))
            failures += 0 if ok else 1
            rows.append({
                "layer": layer, "k": k, "file": entry["file"], **binding,
                "signature": sig, "expert_sha_mismatches": digest_bad,
                "provenance_match": provenance_ok, "parents": parent_rows,
                "parent_expert_coverage": coverage_ok,
                "header_plan_authenticated": headers_ok, "match": ok})
            print(f"fetched layer {layer:3d} k{k}: "
                  f"{'CHAIN VERIFIED' if ok else 'CHAIN MISMATCH'} "
                  f"parents={len(parent_rows)} header-plan="
                  f"{'authenticated' if headers_ok else 'UNSAFE'}", flush=True)
    if unsafe:
        policy = "accepted by explicit --accept-unsafe-header-plan" if args.accept_unsafe_header_plan else "REJECTED"
        print(f"unsafe header-planning evidence: {unsafe} parent record(s), {policy}",
              file=sys.stderr, flush=True)
    report = {
        "mode": "identity", "check": "fetched", "created_utc": now_utc(),
        "segments": str(fam), "trust": trust_report(verifier), "layers": rows,
        "summary": {"segments_checked": len(rows),
                    "segments_verified": sum(r["match"] for r in rows),
                    "unsafe_header_records": unsafe, "failures": failures},
        "note": ("fetched checks the local last hop plus copied, independently "
                 "pinned publisher attestations; release coverage is reported "
                 "as unavailable unless a future fetched tree preserves the "
                 "publisher's signed release manifest."),
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
                    f"{sum(r.get('byte_equal', False) for r in rem)}/{len(rem)} |")
            lines.append("")
            msg = (f"**{s['experts_byte_equal']}/{s['experts_refetched']} sampled "
                   f"experts byte-identical to freshly range-read source bytes** "
                   f"({s['bytes_refetched'] / 1e6:.0f} MB re-fetched)")
            if s["profiles_refetched"]:
                msg += (f"; {s['profiles_byte_equal']}/{s['profiles_refetched']} "
                        f"shared profiles byte-identical (fetched in full)")
            lines.append(msg + ".")
        elif check == "fetched":
            s = report["summary"]
            lines.append("| layer | k | local hop | publisher parents | header plan |")
            lines.append("|---|---|---|---|---|")
            for r in report["layers"]:
                lines.append(
                    f"| {r['layer']} | {r['k']} | "
                    f"{'PASS' if r['match'] else 'FAIL'} | "
                    f"{sum(p['match'] for p in r['parents'])}/{len(r['parents'])} | "
                    f"{'authenticated' if r['header_plan_authenticated'] else 'UNSAFE'} |")
            lines.append("")
            lines.append(
                f"**{s['segments_verified']}/{s['segments_checked']} fetched "
                f"segments have a locally pinned hop and independently pinned "
                f"publisher parents**; {s['unsafe_header_records']} unsafe "
                f"header-planning record(s), {s['failures']} failures.")
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
    # fq_fetch writes this alongside each locally signed range subset.  It is
    # only a mode selector, never a trust input; --check fetched still requires
    # independently pinned publisher attestations.
    if (segments / "fq-fetch-report.json").exists():
        return "fetched"
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
    p.add_argument("--check", choices=("auto", "local", "remote", "derived", "fetched"),
                   default="auto")
    p.add_argument("--upstream-trust-signer", action="append", default=[],
                   help="publisher signer pin for --check fetched; repeat for "
                        "independent upstream publishers")
    p.add_argument("--accept-unsafe-header-plan", action="store_true",
                   help="allow fetched chains whose signed parent evidence says "
                        "the layout header was unauthenticated")
    p.add_argument("--sample", default="3",
                   help="experts re-fetched per (layer,k) in remote check, "
                        "or 'all'")
    p.add_argument("--layers", default=None, help="e.g. 3-10")
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
    fq_trust.add_trust_arguments(p)
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
        try:
            if check == "local":
                if args.source is None:
                    p.error("--check local requires --source")
                rc, report = cmd_identity_local(args)
            elif check == "remote":
                rc, report = cmd_identity_remote(args)
            elif check == "fetched":
                rc, report = cmd_identity_fetched(args)
            else:
                rc, report = cmd_identity_derived(args)
        except TrustError as e:
            print(f"TRUST FAILURE: {e}", file=sys.stderr)
            return 1
        if args.insecure_skip_signatures:
            print("--insecure-skip-signatures: no signature was checked, so "
                  "this run proves nothing about provenance and exits "
                  "non-zero by definition", file=sys.stderr, flush=True)
    if report:
        write_reports(report, args.json, args.md)
        if not args.md:
            print(render_md(report), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
