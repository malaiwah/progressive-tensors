#!/usr/bin/env python3
"""fq_prime — community-quant priming: extract per-expert K3/K4 fragments from
published EXL3 mixed quants on Hugging Face into Progressive Tensors segments
via ranged HTTP reads (no full shard downloads).

Given repo+revision, a layer range, and a K filter, fq_prime:

1. fetches each layer shard's safetensors header by ranged GET (8-byte length
   prefix, then the JSON),
2. identifies routed-expert fragments and their K from the trellis geometry
   (last dim = 16*K) and validates the per-expert tensor-set signature against
   the detected rotation layout (shared_h_v1 vs per_expert_v1),
3. coalesces the needed byte ranges (strictly adjacent runs, capped chunks)
   and ranged-GETs only those bytes, paced to be polite to the HF account,
4. writes fq-segment/1 files (per-layer per-K, per-expert contiguous, tensors
   in fq_repack canonical order) plus index + signed fq-attestation/1 lines
   (v1 schema with extra provenance payload keys per the v2 direction).

For shared_h_v1 sources (e.g. GLM-5.2-EXL3-TR3-3.42bpw) BOTH families are
emitted:

  (a) shared-h family, verbatim (out/shared-h/): expert segments with layout
      tag shared_h_v1 + per-layer shared profile fragments (the 12 shared
      H-side rows), predicate repack-of.  An expert fragment is decodable only
      with its layer's profile fragment; the family manifest pins that
      dependency.
  (b) expanded family (out/expanded/, with --expand): the per-expert view —
      each layer's shared suh rows are replicated into every expert's
      {gate,up}_proj.rank{R}.suh slot and the shared svh rows into
      down_proj.rank{R}.svh, producing 48-tensor units with the exact
      per_expert_v1 name/shape signature (layout tag rank_sliced_tp4,
      predicate derived-from with parent refs to the shared fragments).  The
      expansion is a pure replication; trellis/mcg/expert-local bytes stay
      byte-identical (quant-342-layout-report.md, class (a)).

Per-expert-source repos (e.g. the 3.36bpw per_expert_v1 quant) get a single
family directly under out/, layout rank_sliced_tp4, predicate repack-of.

Signing reuses fq_repack's Signer (ed25519 seed at ~/.fq_keys); segments are
readable by fq_assemble.SegmentReader and consumable by fq_assemble's
mixed-size reindex path.  No uploads: this tool never publishes.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mmap
import os
import random
import re
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fq_repack import (  # noqa: E402
    ATTESTATION_SCHEMA,
    EXPERT_RE,
    MANIFEST_SCHEMA,
    PROJ_ORDER,
    SEGMENT_SCHEMA,
    Signer,
    atomic_write_json,
    expert_key,
    read_header,
)

SHARED_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.shared_h\.(\w+_proj)\.rank(\d+)\.(suh|svh)$"
)
DERIVED_RULE = "shared_h_expand_v1"
# segment-file layout tag per detected source rotation layout: per-expert
# sources keep the brandonmusic-family tag rank_sliced_tp4 (fq_repack v1)
SEGMENT_LAYOUT = {"shared_h_v1": "shared_h_v1", "per_expert_v1": "rank_sliced_tp4"}
USER_AGENT = "fq-prime/1 (progressive-tensors research)"
EXPECTED_COMPS = {
    "shared_h_v1": {
        "gate_proj": {"trellis", "svh", "mcg"},
        "up_proj": {"trellis", "svh", "mcg"},
        "down_proj": {"trellis", "suh", "mcg"},
    },
    "per_expert_v1": {
        p: {"trellis", "suh", "svh", "mcg"}
        for p in ("gate_proj", "up_proj", "down_proj")
    },
}
# select provenance keys copied from the source config's hybrid_tr3_tail into
# attestation payloads (fq-attestation/2 direction: determinism scope)
SOURCE_CONFIG_KEYS = (
    "rotation_layout", "producer_version", "exllamav3_version",
    "mcg_multiplier", "expert_bpw_mean", "bits",
)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------- transport

class _AuthStripRedirect(urllib.request.HTTPRedirectHandler):
    """Drop Authorization when a redirect leaves the original host (HF ->
    CDN presigned URLs reject foreign auth headers)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            old_host = urllib.parse.urlsplit(req.full_url).netloc
            if urllib.parse.urlsplit(newurl).netloc != old_host:
                new.headers.pop("Authorization", None)
        return new


_OPENER = urllib.request.build_opener(_AuthStripRedirect)


def _request(url: str, start: int | None, end: int | None):
    headers = {"User-Agent": USER_AGENT}
    if start is not None:
        headers["Range"] = f"bytes={start}-" if end is None else f"bytes={start}-{end - 1}"
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok and url.startswith("https://huggingface.co/"):
        headers["Authorization"] = f"Bearer {tok}"
    return urllib.request.Request(url, headers=headers)


def http_get_range(url: str, start: int, end: int, timeout: float = 600.0):
    """GET bytes [start, end) of url.  Returns (data, total_size_or_None).
    Module-level so tests can monkeypatch all remote IO."""
    with _OPENER.open(_request(url, start, end), timeout=timeout) as r:
        data = r.read()
        total = None
        cr = r.headers.get("Content-Range", "")
        if "/" in cr and cr.rsplit("/", 1)[1].isdigit():
            total = int(cr.rsplit("/", 1)[1])
    if len(data) != end - start:
        raise IOError(f"range {start}-{end} of {url}: got {len(data)} bytes")
    return data, total


def http_get_full(url: str, timeout: float = 600.0) -> bytes:
    """GET a whole (small) file.  Module-level so tests can monkeypatch."""
    with _OPENER.open(_request(url, None, None), timeout=timeout) as r:
        return r.read()


class Transport:
    """Paced, retrying remote reader; counts successful requests and bytes."""

    RETRYABLE = {408, 425, 429, 500, 502, 503, 504}

    def __init__(self, pace: float = 1.0, retries: int = 6):
        self.pace, self.retries = pace, retries
        self.requests = 0
        self.bytes = 0
        self._next = 0.0

    def _wait(self):
        now = time.monotonic()
        if now < self._next:
            time.sleep(self._next - now)
        self._next = time.monotonic() + self.pace

    def _retrying(self, fn):
        last = None
        for attempt in range(self.retries):
            self._wait()
            try:
                return fn()
            except Exception as e:  # noqa: BLE001 — classified below
                code = getattr(e, "code", None)
                if code is not None and code not in self.RETRYABLE:
                    raise
                last = e
                delay = min(60.0, 2.0 ** (attempt + 1))
                hdrs = getattr(e, "headers", None)
                ra = hdrs.get("Retry-After") if hdrs else None
                if ra and str(ra).isdigit():
                    delay = max(delay, float(ra))
                print(f"  transport: {e} — retry in {delay:.0f}s", flush=True)
                time.sleep(delay)
        raise last

    def get_range(self, url: str, start: int, end: int):
        data, total = self._retrying(lambda: http_get_range(url, start, end))
        self.requests += 1
        self.bytes += len(data)
        return data, total

    def get_full(self, url: str) -> bytes:
        data = self._retrying(lambda: http_get_full(url))
        self.requests += 1
        self.bytes += len(data)
        return data

    def stats(self) -> dict:
        return {"requests": self.requests, "bytes": self.bytes}


# ------------------------------------------------------------------ source

class Source:
    """One pinned HF repo, reached only through ranged reads."""

    def __init__(self, repo: str, revision: str, base_model: str,
                 transport: Transport, cache_dir: Path):
        self.repo, self.revision, self.base_model = repo, revision, base_model
        self.t = transport
        self.cache = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.file_shas: dict[str, str] = {}
        self.config: dict = {}
        self.tier_bitmap: dict = {}
        self.corpus_sha256: str | None = None

    def url(self, name: str) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{name}"

    def _small_file(self, name: str) -> bytes | None:
        p = self.cache / name
        if p.exists():
            return p.read_bytes()
        try:
            data = self.t.get_full(self.url(name))
        except Exception as e:  # noqa: BLE001 — optional file
            print(f"  source: {name} unavailable ({getattr(e, 'code', e)})", flush=True)
            return None
        p.write_bytes(data)
        return data

    def load_metadata(self) -> None:
        mf = self._small_file("MANIFEST.sha256")
        if mf:
            for line in mf.decode().splitlines():
                parts = line.split()
                if len(parts) == 2:
                    self.file_shas[parts[1]] = parts[0]
        cf = self._small_file("config.json")
        self.config = json.loads(cf) if cf else {}
        tb = self._small_file("tier_bitmap.json")
        self.tier_bitmap = json.loads(tb) if tb else {}
        cal = self._small_file("calibration_manifest.json")
        if cal:
            self.corpus_sha256 = json.loads(cal).get("corpus_sha256")

    def shard_name(self, layer: int) -> str:
        return f"model-layer-{layer:03d}.safetensors"

    def shard_header(self, layer: int):
        """(header, body_offset, total_size, url) — cached under source-meta."""
        name = self.shard_name(layer)
        url = self.url(name)
        cache = self.cache / f"hdr-{name}.json"
        if cache.exists():
            obj = json.loads(cache.read_text())
            return obj["header"], obj["body_offset"], obj["total_size"], url
        d, total = self.t.get_range(url, 0, 8)
        hlen = struct.unpack("<Q", d)[0]
        hj, _ = self.t.get_range(url, 8, 8 + hlen)
        hdr = json.loads(hj)
        cache.write_text(json.dumps(
            {"header": hdr, "body_offset": 8 + hlen, "total_size": total}))
        return hdr, 8 + hlen, total, url

    def config_provenance(self) -> dict:
        tail = self.config.get("hybrid_tr3_tail") or {}
        return {k: tail[k] for k in SOURCE_CONFIG_KEYS if k in tail}


# ------------------------------------------------------------- layer model

def analyze_layer(hdr: dict, layer: int):
    """(experts, shared, k_of, layout): expert name lists, shared tensors,
    per-expert K from trellis last dims, and the detected rotation layout.
    Validates the per-expert tensor-set signature and intra-expert K purity."""
    experts: dict[int, list[str]] = {}
    shared: dict[str, dict] = {}
    for name, t in hdr.items():
        if name == "__metadata__":
            continue
        m = EXPERT_RE.match(name)
        if m:
            if int(m.group(1)) != layer:
                raise ValueError(f"foreign layer tensor in shard: {name}")
            experts.setdefault(int(m.group(2)), []).append(name)
            continue
        s = SHARED_RE.match(name)
        if s:
            if int(s.group(1)) != layer:
                raise ValueError(f"foreign layer shared tensor: {name}")
            shared[name] = t
    layout = "shared_h_v1" if shared else "per_expert_v1"
    expected = EXPECTED_COMPS[layout]
    k_of: dict[int, int] = {}
    for eid, names in experts.items():
        by_slot: dict[tuple[str, int], set[str]] = {}
        ks = set()
        for n in names:
            _, _, proj, rank, comp = EXPERT_RE.match(n).groups()
            by_slot.setdefault((proj, int(rank)), set()).add(comp)
            if comp == "trellis":
                last = hdr[n]["shape"][-1]
                if last % 16:
                    raise ValueError(f"{n}: trellis last dim {last} not 16*K")
                ks.add(last // 16)
        for (proj, rank), comps in by_slot.items():
            if comps != expected[proj]:
                raise ValueError(
                    f"expert {eid} {proj}.rank{rank}: comps {sorted(comps)} "
                    f"!= {layout} signature {sorted(expected[proj])}")
        if len(ks) != 1:
            raise ValueError(f"expert {eid}: mixed K within unit: {sorted(ks)}")
        k_of[eid] = ks.pop()
    if shared:
        ranks = {int(EXPERT_RE.match(n).group(4))
                 for ns in experts.values() for n in ns}
        if len(shared) != 3 * len(ranks):
            raise ValueError(
                f"layer {layer}: {len(shared)} shared rows, expected {3 * len(ranks)}")
    return experts, shared, k_of, layout


def shared_sort_key(name: str):
    _, proj, rank, comp = SHARED_RE.match(name).groups()
    return (PROJ_ORDER[proj], int(rank), comp)


def plan_segment(hdr: dict, experts: dict[int, list[str]], eids: list[int]):
    """Canonical (fq_repack-ordered) segment plan for the given experts:
    (ordered_names, seg_tensors, per_expert_ranges, body_size)."""
    ordered: list[str] = []
    for eid in sorted(eids):
        ordered.extend(sorted(experts[eid], key=expert_key))
    seg_tensors: dict[str, dict] = {}
    per_expert: dict[int, list[int]] = {}
    off = 0
    for name in ordered:
        t = hdr[name]
        size = t["data_offsets"][1] - t["data_offsets"][0]
        seg_tensors[name] = {"dtype": t["dtype"], "shape": t["shape"],
                             "data_offsets": [off, off + size]}
        eid = int(EXPERT_RE.match(name).group(2))
        per_expert.setdefault(eid, [off, off])[1] = off + size
        off += size
    return ordered, seg_tensors, per_expert, off


# ---------------------------------------------------------------- segments

class SegmentWriter:
    """Preallocated fq-segment/1 writer: header up front, payload via pwrite
    at precomputed offsets (source arrival order need not match canonical
    output order), digests computed in one local read-back pass."""

    def __init__(self, path: Path, meta: dict, seg_tensors: dict, body_size: int):
        header = {"__metadata__": {k: str(v) for k, v in meta.items()}, **seg_tensors}
        hj = json.dumps(header, separators=(",", ":")).encode()
        hj += b" " * ((8 - len(hj) % 8) % 8)
        self.body_off = 8 + len(hj)
        self.path = path
        self.tmp = path.with_suffix(".part")
        with open(self.tmp, "wb") as f:
            f.write(struct.pack("<Q", len(hj)))
            f.write(hj)
            f.truncate(self.body_off + body_size)
        self.f = open(self.tmp, "r+b")
        self.body_size = body_size
        self.written = 0

    def pwrite(self, off: int, data: bytes) -> None:
        os.pwrite(self.f.fileno(), data, self.body_off + off)
        self.written += len(data)

    def finalize(self, ranges: dict) -> dict:
        """ranges: key -> [lo, hi) body offsets.  Returns index info with the
        whole-file sha256 and one sha256 per key (expert id or tensor name)."""
        if self.written != self.body_size:
            raise AssertionError(
                f"{self.path.name}: wrote {self.written} of {self.body_size} body bytes")
        self.f.close()
        whole = hashlib.sha256()
        with open(self.tmp, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                whole.update(block)
        per_key = {}
        with open(self.tmp, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            for key, (lo, hi) in ranges.items():
                per_key[str(key)] = hashlib.sha256(
                    mm[self.body_off + lo:self.body_off + hi]).hexdigest()
            mm.close()
        os.replace(self.tmp, self.path)
        return {
            "sha256": whole.hexdigest(),
            "size": self.path.stat().st_size,
            "body_offset": self.body_off,
            "ranges": {str(k): list(v) for k, v in ranges.items()},
            "sha256_by_key": per_key,
        }


def coalesce_pieces(pieces: list, max_chunk: int) -> list[list]:
    """Group (src_a, src_b, writer, out_off) pieces into ranged-GET chunks:
    only strictly adjacent pieces merge (no gap bytes are ever fetched) and a
    chunk never exceeds max_chunk unless a single piece does."""
    pieces = sorted(pieces, key=lambda p: p[0])
    for prev, cur in zip(pieces, pieces[1:]):
        if cur[0] < prev[1]:
            raise AssertionError(f"overlapping source ranges: {prev[:2]} vs {cur[:2]}")
    chunks: list[list] = []
    for piece in pieces:
        if (chunks and piece[0] == chunks[-1][-1][1]
                and piece[1] - chunks[-1][0][0] <= max_chunk):
            chunks[-1].append(piece)
        else:
            chunks.append([piece])
    return chunks


def execute_fetch(transport: Transport, url: str, body_off: int,
                  pieces: list, max_chunk: int) -> int:
    """Ranged-GET the coalesced chunks and pwrite each piece into its
    segment writer.  Returns the number of range requests made."""
    chunks = coalesce_pieces(pieces, max_chunk)
    for chunk in chunks:
        a, b = chunk[0][0], chunk[-1][1]
        data, _ = transport.get_range(url, body_off + a, body_off + b)
        for sa, sb, writer, out_off in chunk:
            writer.pwrite(out_off, data[sa - a:sb - a])
    return len(chunks)


# ----------------------------------------------------------------- priming

def parse_int_set(spec: str) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def seg_meta(source: Source, layer: int, layout: str, num_experts: int,
             predicate: str = "repack-of", **extra) -> dict:
    meta = {
        "fq_schema": SEGMENT_SCHEMA,
        "predicate": predicate,
        "layer": layer,
        "layout": layout,
        "num_experts": num_experts,
        "source_file": source.shard_name(layer),
        "source_repo": source.repo,
        "source_revision": source.revision,
        "base_model": source.base_model,
    }
    sha = source.file_shas.get(source.shard_name(layer))
    if sha:
        meta["source_file_sha256"] = sha
    meta.update(extra)
    return meta


def base_attestation(source: Source, layer: int, layout: str, predicate: str,
                     fragment_file: str, info: dict) -> dict:
    return {
        "schema": ATTESTATION_SCHEMA,
        "predicate": predicate,
        "fragment": {"file": fragment_file, "sha256": info["sha256"],
                     "size": info["size"]},
        "materials": {
            "repo": source.repo,
            "revision": source.revision,
            "file": source.shard_name(layer),
            "file_sha256": source.file_shas.get(source.shard_name(layer)),
        },
        "layout": layout,
        "base_model": source.base_model,
        "calibration_corpus_sha256": source.corpus_sha256,
        "source_config": source.config_provenance(),
        "created_utc": now_utc(),
    }


def tier_bitmap_check(source: Source, layer: int, k_of: dict[int, int]) -> str:
    tb = source.tier_bitmap.get(str(layer), {})
    declared = tb.get("k")
    if not isinstance(declared, list):
        return "no-tier-bitmap"
    if len(declared) != len(k_of):
        return f"MISMATCH(len {len(declared)} vs {len(k_of)})"
    bad = sum(1 for e, k in k_of.items() if declared[e] != k)
    return "agrees" if bad == 0 else f"MISMATCH({bad} experts)"


def prime_layer(source: Source, fam_dir: Path, layer: int, wanted_ks: list[int],
                signer: Signer, max_chunk: int, dry_run: bool):
    """Fetch and write one layer's segments (+ shared profile when present).
    Returns (layer_state_entry, layout) or (None, layout) on dry-run."""
    hdr, body_off, total_size, url = source.shard_header(layer)
    experts, shared, k_of, layout = analyze_layer(hdr, layer)
    tag = SEGMENT_LAYOUT[layout]
    ks_present = sorted({k for k in k_of.values()})
    ks = [k for k in wanted_ks if k in ks_present]
    agree = tier_bitmap_check(source, layer, k_of)
    counts = {k: sum(1 for v in k_of.values() if v == k) for k in ks_present}
    need = sum(
        hdr[n]["data_offsets"][1] - hdr[n]["data_offsets"][0]
        for eid, names in experts.items() if k_of[eid] in ks for n in names)
    need += sum(t["data_offsets"][1] - t["data_offsets"][0] for t in shared.values())
    print(f"layer {layer:3d}: layout={layout} experts={counts} "
          f"tier_bitmap={agree} need={need / 1e9:.2f}GB "
          f"(shard {total_size / 1e9:.2f}GB)", flush=True)
    if dry_run:
        return None, layout

    entry: dict = {"status": "done", "ks": ks, "layout": layout,
                   "shard_size": total_size, "k_counts": {str(k): counts[k] for k in ks_present},
                   "index": {}}
    att_dir = fam_dir / "attestations"
    att_dir.mkdir(parents=True, exist_ok=True)
    profile_info = None
    profile_name = f"layer-{layer:03d}.shared.safetensors"

    # phase 1 — shared profile fragment (its sha is pinned into expert meta)
    if shared:
        names = sorted(shared, key=shared_sort_key)
        seg_tensors, ranges, off = {}, {}, 0
        pieces = []
        for n in names:
            t = shared[n]
            size = t["data_offsets"][1] - t["data_offsets"][0]
            seg_tensors[n] = {"dtype": t["dtype"], "shape": t["shape"],
                              "data_offsets": [off, off + size]}
            ranges[n] = [off, off + size]
            off += size
        meta = seg_meta(source, layer, layout, len(experts),
                        kind="shared_profile", num_tensors=len(names))
        w = SegmentWriter(fam_dir / profile_name, meta, seg_tensors, off)
        for n in names:
            a, b = shared[n]["data_offsets"]
            pieces.append((a, b, w, seg_tensors[n]["data_offsets"][0]))
        execute_fetch(source.t, url, body_off, pieces, max_chunk)
        profile_info = w.finalize(ranges)
        payload = base_attestation(source, layer, layout, "repack-of",
                                   profile_name, profile_info)
        payload["kind"] = "shared_profile"
        payload["tensor_sha256"] = profile_info["sha256_by_key"]
        payload["source_spans"] = {
            n: [body_off + shared[n]["data_offsets"][0],
                body_off + shared[n]["data_offsets"][1]] for n in names}
        (att_dir / f"layer-{layer:03d}.shared.jsonl").write_text(
            signer.sign_line(payload) + "\n")
        entry["profile"] = {
            "file": profile_name, "sha256": profile_info["sha256"],
            "size": profile_info["size"], "body_offset": profile_info["body_offset"],
            "tensors": profile_info["ranges"],
        }

    # phase 2 — expert segments for all wanted Ks, one coalesced sweep
    writers: dict[int, tuple[SegmentWriter, dict, dict, str]] = {}
    pieces = []
    for k in ks:
        eids = [e for e, kk in k_of.items() if kk == k]
        ordered, seg_tensors, per_expert, body_size = plan_segment(hdr, experts, eids)
        extra = {"k": k}
        if profile_info is not None:
            extra["shared_profile_file"] = profile_name
            extra["shared_profile_sha256"] = profile_info["sha256"]
        meta = seg_meta(source, layer, tag, len(eids), **extra)
        seg_name = f"layer-{layer:03d}.k{k}.safetensors"
        w = SegmentWriter(fam_dir / seg_name, meta, seg_tensors, body_size)
        writers[k] = (w, per_expert, seg_tensors, seg_name)
        for name in ordered:
            a, b = hdr[name]["data_offsets"]
            pieces.append((a, b, w, seg_tensors[name]["data_offsets"][0]))
    n_req = execute_fetch(source.t, url, body_off, pieces, max_chunk)

    for k in ks:
        w, per_expert, seg_tensors, seg_name = writers[k]
        info = w.finalize(per_expert)
        payload = base_attestation(source, layer, tag, "repack-of", seg_name, info)
        payload["k"] = k
        payload["expert_sha256"] = info["sha256_by_key"]
        payload["expert_k"] = {str(e): k_of[e] for e in sorted(per_expert)}
        payload["source_spans"] = {
            str(e): [body_off + min(hdr[n]["data_offsets"][0] for n in experts[e]),
                     body_off + max(hdr[n]["data_offsets"][1] for n in experts[e])]
            for e in sorted(per_expert)}
        if profile_info is not None:
            payload["dependencies"] = [{
                "file": profile_name, "sha256": profile_info["sha256"],
                "role": "shared_profile"}]
        (att_dir / f"layer-{layer:03d}.k{k}.jsonl").write_text(
            signer.sign_line(payload) + "\n")
        entry["index"][str(k)] = {
            "file": seg_name, "sha256": info["sha256"], "size": info["size"],
            "body_offset": info["body_offset"], "experts": info["ranges"],
        }
    entry["range_requests"] = n_req
    return entry, layout


# --------------------------------------------------------------- expansion

def expand_layer(shared_dir: Path, exp_dir: Path, layer: int, k: int,
                 signer: Signer, layer_state: dict) -> dict:
    """Materialize the per-expert (rank_sliced_tp4) view of one shared-h
    segment by replicating the layer's shared rows into every expert.
    Pure local derivation: no network."""
    from fq_assemble import SegmentReader  # reuse, avoid duplicate reader

    seg_name = f"layer-{layer:03d}.k{k}.safetensors"
    seg_path = shared_dir / seg_name
    prof_path = shared_dir / layer_state["profile"]["file"]
    seg_meta_d = read_header(seg_path)[0]["__metadata__"]
    seg = SegmentReader(seg_path)
    prof = SegmentReader(prof_path)
    try:
        prof_by_slot = {}
        for pname in prof.hdr:
            _, proj, rank, comp = SHARED_RE.match(pname).groups()
            prof_by_slot[(proj, int(rank))] = pname
        experts: dict[int, list[str]] = {}
        for name in seg.hdr:
            experts.setdefault(int(EXPERT_RE.match(name).group(2)), []).append(name)

        plan: list[tuple[str, object, str]] = []  # (out_name, reader, src_name)
        for eid in sorted(experts):
            unit = [(n, seg, n) for n in experts[eid]]
            for (proj, rank), pname in prof_by_slot.items():
                comp = "suh" if proj in ("gate_proj", "up_proj") else "svh"
                out_name = (f"model.layers.{layer}.mlp.experts.{eid}."
                            f"{proj}.rank{rank}.{comp}")
                unit.append((out_name, prof, pname))
            unit.sort(key=lambda item: expert_key(item[0]))
            plan.extend(unit)

        seg_tensors, per_expert = {}, {}
        off = 0
        for out_name, reader, src_name in plan:
            t = reader.hdr[src_name]
            size = t["data_offsets"][1] - t["data_offsets"][0]
            seg_tensors[out_name] = {"dtype": t["dtype"], "shape": t["shape"],
                                     "data_offsets": [off, off + size]}
            eid = int(EXPERT_RE.match(out_name).group(2))
            per_expert.setdefault(eid, [off, off])[1] = off + size
            off += size

        meta = {
            "fq_schema": SEGMENT_SCHEMA,
            "predicate": "derived-from",
            "k": k,
            "layer": layer,
            "layout": "rank_sliced_tp4",
            "num_experts": len(experts),
            "source_file": seg_meta_d["source_file"],
            "source_repo": seg_meta_d["source_repo"],
            "source_revision": seg_meta_d["source_revision"],
            "base_model": seg_meta_d["base_model"],
            "derived_rule": DERIVED_RULE,
            "parent_segment": seg_name,
            "parent_segment_sha256": layer_state["index"][str(k)]["sha256"],
            "parent_profile": layer_state["profile"]["file"],
            "parent_profile_sha256": layer_state["profile"]["sha256"],
        }
        if "source_file_sha256" in seg_meta_d:
            meta["source_file_sha256"] = seg_meta_d["source_file_sha256"]

        w = SegmentWriter(exp_dir / seg_name, meta, seg_tensors, off)
        for out_name, reader, src_name in plan:
            w.pwrite(seg_tensors[out_name]["data_offsets"][0],
                     reader.tensor_bytes(src_name))
        info = w.finalize(per_expert)
    finally:
        seg.close()
        prof.close()

    att_dir = exp_dir / "attestations"
    att_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": ATTESTATION_SCHEMA,
        "predicate": "derived-from",
        "fragment": {"file": seg_name, "sha256": info["sha256"], "size": info["size"]},
        "materials": {
            "repo": seg_meta_d["source_repo"],
            "revision": seg_meta_d["source_revision"],
            "file": seg_meta_d["source_file"],
            "file_sha256": seg_meta_d.get("source_file_sha256"),
        },
        "parents": [
            {"file": seg_name, "sha256": layer_state["index"][str(k)]["sha256"],
             "role": "expert_segment", "family": "shared-h"},
            {"file": layer_state["profile"]["file"],
             "sha256": layer_state["profile"]["sha256"],
             "role": "shared_profile", "family": "shared-h"},
        ],
        "derivation": {
            "rule": DERIVED_RULE,
            "description": ("replicate the layer's shared_h suh rows into each "
                            "expert's {gate,up}_proj.rank{R}.suh and shared svh "
                            "into down_proj.rank{R}.svh; trellis/mcg/expert-local "
                            "vectors byte-identical (exact expansion, "
                            "quant-342-layout-report.md class (a))"),
        },
        "layout": "rank_sliced_tp4",
        "k": k,
        "expert_sha256": info["sha256_by_key"],
        "created_utc": now_utc(),
    }
    (att_dir / f"layer-{layer:03d}.k{k}.jsonl").write_text(
        signer.sign_line(payload) + "\n")
    return {
        "file": seg_name, "sha256": info["sha256"], "size": info["size"],
        "body_offset": info["body_offset"], "experts": info["ranges"],
    }


# --------------------------------------------------------------- manifests

def write_family_outputs(fam_dir: Path, layers_state: dict, *, which: str,
                         layout: str, predicate: str, source: Source,
                         signer: Signer, extra: dict | None = None) -> None:
    """Rebuild index-k*.json (+index-shared.json) and fq-manifest.json for a
    family from the accumulated state.  which: 'index' or 'expanded'."""
    ks = sorted({int(k) for st in layers_state.values()
                 for k in st.get(which, {})})
    for k in ks:
        idx = {L: st[which][str(k)] for L, st in sorted(
            layers_state.items(), key=lambda kv: int(kv[0]))
            if str(k) in st.get(which, {})}
        atomic_write_json(fam_dir / f"index-k{k}.json", idx)
    deps = {}
    if which == "index":
        deps = {L: {"file": st["profile"]["file"], "sha256": st["profile"]["sha256"]}
                for L, st in sorted(layers_state.items(), key=lambda kv: int(kv[0]))
                if "profile" in st}
    if deps:
        atomic_write_json(fam_dir / "index-shared.json", {
            L: st["profile"] for L, st in sorted(
                layers_state.items(), key=lambda kv: int(kv[0]))
            if "profile" in st})
    layers = sorted(int(L) for L in layers_state)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "base_model": source.base_model,
        "revision": source.revision,
        "k_variants": ks,
        "hessian_id": None,
        "predicate": predicate,
        "layout": layout,
        "sources": [source.repo],
        "source_revision": source.revision,
        "moe_layers": [min(layers), max(layers)] if layers else [],
        # per-layer expert count summed across that layer's K files
        "num_experts": max(
            (sum(len(e["experts"]) for e in st.get(which, {}).values())
             for st in layers_state.values()), default=0),
        "tensor_indexes": {str(k): f"index-k{k}.json" for k in ks},
        "calibration_corpus_sha256": source.corpus_sha256,
        "signer_pubkey": signer.pub_hex,
    }
    if deps:
        manifest["dependencies"] = {"shared_profile_per_layer": deps,
                                    "shared_index": "index-shared.json"}
        manifest["note"] = ("expert fragments are decodable only together with "
                            "their layer's shared profile fragment")
    if extra:
        manifest.update(extra)
    atomic_write_json(fam_dir / "fq-manifest.json", manifest)


def update_root_manifest(root: Path, entry: dict) -> None:
    mpath = root / "fq-manifest.json"
    m = (json.loads(mpath.read_text()) if mpath.exists()
         else {"schema": MANIFEST_SCHEMA, "kind": "multi-source", "families": []})
    fams = [f for f in m.get("families", []) if f.get("path") != entry["path"]]
    fams.append(entry)
    m["families"] = sorted(fams, key=lambda f: f["path"])
    m["updated_utc"] = now_utc()
    atomic_write_json(mpath, m)


# ------------------------------------------------------------------ prime

def cmd_prime(args) -> int:
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    transport = Transport(pace=args.pace)
    source = Source(args.repo, args.revision, args.base_model, transport,
                    out / "source-meta")
    source.load_metadata()
    signer = Signer(args.sign_key)
    wanted_ks = parse_int_set(args.k)
    layers = parse_int_set(args.layers)

    state_path = out / "state.json"
    state = (json.loads(state_path.read_text()) if state_path.exists()
             else {"layers": {}, "transport": {"requests": 0, "bytes": 0}})
    layout_seen: str | None = None

    for layer in layers:
        st = state["layers"].get(str(layer), {})
        done = (st.get("status") == "done" and st.get("ks") == wanted_ks
                and all((out / _fam_name(st["layout"]) / e["file"]).exists()
                        for e in st.get("index", {}).values()))
        need_expand = args.expand and st.get("layout") == "shared_h_v1"
        if done and (not need_expand or st.get("expanded")):
            layout_seen = st["layout"]
            print(f"layer {layer:3d}: skip (done)", flush=True)
            continue
        t0 = time.time()
        req0, bytes0 = transport.requests, transport.bytes
        if done:  # primed but expansion missing: local-only pass
            entry, layout = st, st["layout"]
        else:
            fam_dir = out  # provisional; real dir depends on layout
            hdr_probe = None
            # we must know the layout to choose the family dir; peek header
            hdr_probe, _, _, _ = source.shard_header(layer)
            has_shared = any(SHARED_RE.match(n) for n in hdr_probe)
            fam_dir = out / _fam_name("shared_h_v1" if has_shared else "per_expert_v1")
            if has_shared or fam_dir != out:
                fam_dir.mkdir(parents=True, exist_ok=True)
            entry, layout = prime_layer(source, fam_dir, layer, wanted_ks,
                                        signer, args.chunk_mb << 20, args.dry_run)
            if entry is None:  # dry run
                layout_seen = layout
                continue
        if layout_seen is None:
            layout_seen = layout
        elif layout_seen != layout:
            raise RuntimeError(f"layout flip across layers: {layout_seen} vs {layout}")

        if args.expand and layout == "shared_h_v1":
            exp_dir = out / "expanded"
            exp_dir.mkdir(parents=True, exist_ok=True)
            entry["expanded"] = {}
            for k_s in entry["index"]:
                entry["expanded"][k_s] = expand_layer(
                    out / _fam_name(layout), exp_dir, layer, int(k_s),
                    signer, entry)
        state["layers"][str(layer)] = entry
        state["transport"]["requests"] += transport.requests - req0
        state["transport"]["bytes"] += transport.bytes - bytes0
        atomic_write_json(state_path, state)
        fetched = transport.bytes - bytes0
        print(f"layer {layer:3d}: primed {fetched / 1e9:.2f}GB in "
              f"{transport.requests - req0} requests ({time.time() - t0:.0f}s)",
              flush=True)

    if args.dry_run:
        print("dry-run complete (no bytes written)", flush=True)
        return 0
    if not state["layers"]:
        print("nothing primed", flush=True)
        return 0

    layout = layout_seen
    tag = SEGMENT_LAYOUT[layout]
    fam_dir = out / _fam_name(layout)
    predicate = "repack-of"
    write_family_outputs(fam_dir, state["layers"], which="index",
                         layout=tag, predicate=predicate,
                         source=source, signer=signer)
    root = out.parent
    rel = fam_dir.relative_to(root)
    fam_entry = {
        "path": str(rel), "layout": tag, "predicate": predicate,
        "source": {"repo": source.repo, "revision": source.revision},
        "k_variants": sorted({int(k) for st in state["layers"].values()
                              for k in st.get("index", {})}),
        "moe_layers": [min(int(L) for L in state["layers"]),
                       max(int(L) for L in state["layers"])],
    }
    update_root_manifest(root, fam_entry)
    if args.expand and layout == "shared_h_v1":
        exp_dir = out / "expanded"
        write_family_outputs(
            exp_dir, state["layers"], which="expanded",
            layout="rank_sliced_tp4", predicate="derived-from",
            source=source, signer=signer,
            extra={"derived_rule": DERIVED_RULE,
                   "parent_family": str(fam_dir.relative_to(root))})
        update_root_manifest(root, {
            "path": str(exp_dir.relative_to(root)), "layout": "rank_sliced_tp4",
            "predicate": "derived-from",
            "source": {"repo": source.repo, "revision": source.revision},
            "k_variants": fam_entry["k_variants"],
            "moe_layers": fam_entry["moe_layers"],
            "derived_rule": DERIVED_RULE,
            "parent_family": str(rel),
        })
    tr = state["transport"]
    full = sum(st.get("shard_size") or 0 for st in state["layers"].values())
    print(f"done: {len(state['layers'])} layers; fetched {tr['bytes'] / 1e9:.2f}GB "
          f"in {tr['requests']} range requests "
          f"(full-download counterfactual {full / 1e9:.2f}GB)", flush=True)
    return 0


def _fam_name(layout: str) -> Path:
    """Family subdir relative to --out: shared-h/ for shared_h_v1 sources,
    the out dir itself for per-expert sources."""
    return Path("shared-h") if layout == "shared_h_v1" else Path(".")


# -------------------------------------------------------------- spot-check

def cmd_spot_check(args) -> int:
    """Transport spot-check: for N random experts of a primed family,
    independently re-fetch the source bytes (fresh header + payload range)
    and compare sha256 of the canonical-order concatenation against the
    segment content and the signed attestation."""
    fam: Path = args.dir
    transport = Transport(pace=args.pace)
    rng = random.Random(args.seed)
    samples = []
    for idx_path in sorted(fam.glob("index-k*.json")):
        k = int(idx_path.stem.split("-k")[1])
        idx = json.loads(idx_path.read_text())
        for layer_s, entry in idx.items():
            for eid in entry["experts"]:
                samples.append((int(layer_s), k, int(eid), entry))
    if not samples:
        print(f"no indexed experts under {fam}")
        return 1
    picks = rng.sample(samples, min(args.n, len(samples)))
    failures = 0
    for layer, k, eid, entry in picks:
        seg_path = fam / entry["file"]
        meta = read_header(seg_path)[0]["__metadata__"]
        if meta.get("predicate") != "repack-of":
            print(f"refusing spot-check of {seg_path.name}: predicate "
                  f"{meta.get('predicate')} is not repack-of")
            return 2
        url = (f"https://huggingface.co/{meta['source_repo']}/resolve/"
               f"{meta['source_revision']}/{meta['source_file']}")
        d, _ = transport.get_range(url, 0, 8)
        hlen = struct.unpack("<Q", d)[0]
        hj, _ = transport.get_range(url, 8, 8 + hlen)
        hdr = json.loads(hj)
        body_off = 8 + hlen
        names = sorted(
            (n for n in hdr if EXPERT_RE.match(n)
             and int(EXPERT_RE.match(n).group(2)) == eid
             and int(EXPERT_RE.match(n).group(1)) == layer),
            key=expert_key)
        lo = min(hdr[n]["data_offsets"][0] for n in names)
        hi = max(hdr[n]["data_offsets"][1] for n in names)
        blob, _ = transport.get_range(url, body_off + lo, body_off + hi)
        canon = hashlib.sha256()
        for n in names:
            a, b = hdr[n]["data_offsets"]
            canon.update(blob[a - lo:b - lo])
        src_sha = canon.hexdigest()

        slo, shi = entry["experts"][str(eid)]
        with open(seg_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            seg_sha = hashlib.sha256(
                mm[entry["body_offset"] + slo:entry["body_offset"] + shi]).hexdigest()
            mm.close()
        att_path = fam / "attestations" / f"layer-{layer:03d}.k{k}.jsonl"
        att_sha = None
        if att_path.exists():
            line = json.loads(att_path.read_text())
            payload = json.loads(base64.b64decode(line["payload"]))
            att_sha = payload.get("expert_sha256", {}).get(str(eid))
        ok = src_sha == seg_sha and (att_sha is None or att_sha == src_sha)
        failures += 0 if ok else 1
        print(f"layer {layer:3d} k{k} expert {eid:3d}: "
              f"{'OK' if ok else 'MISMATCH'} source={src_sha[:16]} "
              f"segment={seg_sha[:16]} attestation={(att_sha or '-')[:16]}",
              flush=True)
    print(f"spot-check: {len(picks) - failures}/{len(picks)} OK "
          f"({transport.bytes / 1e6:.1f}MB re-fetched)", flush=True)
    return 1 if failures else 0


# -------------------------------------------------------------------- cli

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("prime", help="fetch fragments into segments")
    pr.add_argument("--repo", required=True)
    pr.add_argument("--revision", required=True)
    pr.add_argument("--layers", required=True, help="e.g. 3-10 or 3,5,7")
    pr.add_argument("--k", default="3,4", help="K filter, e.g. 3,4 or 4")
    pr.add_argument("--out", required=True, type=Path,
                    help="source subtree dir, e.g. ~/fq-primed/segments-342")
    pr.add_argument("--base-model", default="zai-org/GLM-5.2")
    pr.add_argument("--sign-key", type=Path,
                    default=Path.home() / ".fq_keys/fq_signing.key")
    pr.add_argument("--expand", action="store_true",
                    help="also emit the per-expert expanded family (shared-h sources)")
    pr.add_argument("--pace", type=float, default=1.0,
                    help="min seconds between HTTP request starts")
    pr.add_argument("--chunk-mb", type=int, default=128,
                    help="max coalesced range-GET size")
    pr.add_argument("--dry-run", action="store_true")
    pr.set_defaults(fn=cmd_prime)

    sc = sub.add_parser("spot-check", help="re-fetch random experts, compare sha256")
    sc.add_argument("--dir", required=True, type=Path, help="family dir")
    sc.add_argument("--n", type=int, default=3)
    sc.add_argument("--seed", type=int, default=None)
    sc.add_argument("--pace", type=float, default=1.0)
    sc.set_defaults(fn=cmd_spot_check)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
