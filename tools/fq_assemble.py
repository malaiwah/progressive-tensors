#!/usr/bin/env python3
"""fq_assemble — Progressive Tensors segments + policy → bootable checkpoint.

The M0 gate tool (04-milestones.md): given per-layer per-K segment files
(fq_repack output or downloaded), a bits_per_expert policy JSON, and the
source checkpoint (for dense/attention/shared tensors and header order),
emit per-layer shards in the original GG rank-sliced layout.

Byte-identity property: assembling the all-K3 policy from K3 segments
reproduces the source shards byte-for-byte (same header, same order) —
tested in test_fq_assemble.py. Mixed policies swap in K4 expert bytes from
K4 segments as they become available.

Verification is mandatory (fail-closed)
---------------------------------------
Segments are other people's bytes, fetched from a public repo: assembling
them into a checkpoint you are about to run is a trust decision, so this tool
refuses to make it implicitly.  Every run must either pin the expected
signer(s) with --trust-signer/--trust-file or opt out loudly with --insecure.
For every (layer, K) segment consumed:

  * its safetensors header is structurally validated BEFORE any tensor read
    (offsets inside the file, spans matching dtype*shape, no overlaps);
  * its attestation line is verified: ed25519 signature under a PINNED signer
    fingerprint, predicate inside the allowed set (--allow-predicate, default
    repack-of/encode-of/derived-from);
  * the signed fragment sha256 is recomputed from the segment file's actual
    bytes, and the signed per-expert digests are recomputed for exactly the
    experts this policy consumes (one streaming pass computes both);
  * compatibility fields (layer, k, layout, base_model, num_experts) are
    cross-checked against the family manifest, the policy and each other.

Because the signed fragment digest covers the whole segment file — header
included — a verified fragment digest also authenticates the segment's
__metadata__, which is what makes the layout/base_model checks meaningful.

Pinning is by ed25519 fingerprint, the same string the project trust root
publishes (keys/FINGERPRINTS) and that appears as `keyid` in attestation
lines: --trust-file reads that file's format directly and never trusts a
record marked revoked.  Take the fingerprint from the trust root, not from
the artifact download you are trying to verify.

Mixed-size support: trellis tensors are [in/16, out/16, 16*K] i16, so K4
rows are 4/3 the bytes of K3 and the source header cannot be reused as a
byte template. When any expert tensor's segment size differs from its
source slot, assemble_layer switches to a reindex path automatically: the
source header's TENSOR ORDER is kept (non-expert tensors stay byte-exact
from the source shard) but the header is rebuilt with dtype/shape taken
from the SEGMENT header for expert tensors and fresh data_offsets.

Output safety: --out is resolved (symlinks refused) and checked before
anything is read or written.  It may not be the filesystem root, your home
directory, this checkout, a git working tree, an upstream model cache, or a
path that overlaps --source/--segments/--policy in either direction.  The
directory must be empty, or --force, which REPLACES it — and only ever a
directory carrying the `.fq-assembly` marker this tool writes, so --force
can never eat a checkpoint fq_assemble did not create.  Assembly happens in
a sibling staging dir that is swapped in with two renames at the end, so an
interrupted run leaves the previous output intact rather than half-purged.

Output integrity: model.safetensors.index.json and MANIFEST.sha256 are
ALWAYS recomputed from the bytes actually written, every loader-visible copy
of the bit allocation (config.json hybrid_tr3_tail, quantization_config,
tier_bitmap.json) is updated so they cannot contradict each other, and an
`assembly-of` provenance record (fq-assembly.json — recipe sha, per-segment
shas, per-output-shard shas, tool version, UTC; per runs/0c-campaign/
ATTESTATION-V2.md) is emitted next to the checkpoint.

Reflink mode (--reflink): expert-tensor regions are written with
os.copy_file_range from the segment file's fd instead of read()+write().
On reflink-capable filesystems (XFS with reflink=1, btrfs) the kernel can
then share extents between the segment files and the assembled shards
instead of storing the same bytes twice.  Honest caveats: (1) extent
sharing only actually happens when the kernel/filesystem chooses to
reflink — copy_file_range may silently perform a plain (server-side)
copy; (2) real savings depend on 4K-block alignment of identical regions
at BOTH the source and destination offsets — MB-scale fragments usually
share most interior blocks, but alignment is not guaranteed; (3) output
byte-identity is ALWAYS preserved regardless — the mode changes how bytes
move, never what they are: every region falls back automatically to the
ordinary copy path when copy_file_range is unavailable or fails (EXDEV,
EOPNOTSUPP, ENOSYS, cross-filesystem, partial copy, ...); (4) this is a
local-disk-space optimization only — it has no effect on HF/remote
transfer or storage.  Implementation choice: only expert tensors (whose
bytes come from segment files) use copy_file_range; the safetensors
header, non-expert tensors, and dense pass-through shards keep the
ordinary mmap/read+write path since they come from the source shard, not
the segment files.  os.copy_file_range is resolved once into the module
attribute _COPY_FILE_RANGE (None where the platform lacks it), which is
also the seam the portability tests patch.

Mixed metadata (GG loader contract, exl3.py _configure_rank_sliced /
_load_rank_sliced_bitrates): config.json hybrid_tr3_tail must carry
bits="mixed", k_values=[3,4,...], and bits_per_expert as a
"file.json:field" reference; the referenced JSON maps str(layer) -> dict
whose field holds one int bitrate per expert.  We emit the map into
tier_bitmap.json under the field "bits_per_expert" and regenerate
model.safetensors.index.json + MANIFEST.sha256 for the new shard sizes.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mmap
import os
import shutil
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fq_repack import (  # noqa: E402
    EXPERT_RE,
    Signer,
    SourceLayoutError,
    atomic_write_json,
    expert_key,
    preflight_source_layout,
    read_header,
    sha256_file,
)
from fq_trust import TrustError, TrustRoot, Verifier as TrustVerifier  # noqa: E402

TOOL_VERSION = "fq_assemble/2"
ATTESTATION_SCHEMA = "fq-attestation/2"
ASSEMBLY_RECORD = "fq-assembly.json"
# Written into every checkpoint this tool produces.  --force will only ever
# delete a directory that carries it: it is the difference between "a previous
# fq_assemble output" and "the model dir the user spent a week curating".
SENTINEL = ".fq-assembly"
SENTINEL_SCHEMA = "fq-assembly-sentinel/1"
DEFAULT_ALLOWED_PREDICATES = ("repack-of", "encode-of", "derived-from")
CHUNK = 1 << 24
MAX_HEADER_BYTES = 1 << 26  # real segment headers are ~1 MB; refuse absurdity
DTYPE_ITEMSIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}
# resolved once so tests (and platforms without the syscall) have one seam
_COPY_FILE_RANGE = getattr(os, "copy_file_range", None)


class AssemblyError(RuntimeError):
    """Assembly refused — the inputs or the output dir are not acceptable."""


class VerificationError(AssemblyError):
    """A segment could not be authenticated against a pinned signer."""


class SegmentIntegrityError(VerificationError):
    """A segment file's own bytes are not internally consistent."""


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _as_int(value):
    """Segment metadata values are strings (safetensors metadata is str->str);
    attestation payloads carry real ints.  Accept both, ignore the rest."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except ValueError:
        return None


# ------------------------------------------------- strict header validation

def validate_safetensors_view(view, size: int, who: str) -> tuple[dict, dict, int]:
    """(tensors, metadata, body_offset) — strict structural check of a mapped
    safetensors file.

    Runs BEFORE any tensor byte is used.  A segment file is untrusted input;
    slicing a header nobody bounds-checked is how a hostile or truncated
    header turns into an out-of-range read, or into silently serving a
    neighbouring tensor's bytes under the wrong name.  Every offset is checked
    against the real file size, every span against its dtype/shape, and every
    span against the others for overlap.
    """
    if size < 8:
        raise SegmentIntegrityError(f"{who}: file too small ({size} bytes)")
    hlen = struct.unpack("<Q", view[0:8])[0]
    if hlen == 0 or hlen > MAX_HEADER_BYTES or 8 + hlen > size:
        raise SegmentIntegrityError(
            f"{who}: header length {hlen} does not fit in a {size}-byte file")
    try:
        hdr = json.loads(view[8:8 + hlen])
    except json.JSONDecodeError as e:
        raise SegmentIntegrityError(f"{who}: header is not valid JSON ({e})")
    if not isinstance(hdr, dict):
        raise SegmentIntegrityError(f"{who}: header is not a JSON object")
    meta = hdr.pop("__metadata__", {}) or {}
    if not isinstance(meta, dict):
        raise SegmentIntegrityError(f"{who}: __metadata__ is not an object")
    body_off = 8 + hlen
    body = size - body_off
    spans = []
    for name, t in hdr.items():
        if not isinstance(t, dict):
            raise SegmentIntegrityError(f"{who}: {name}: entry is not an object")
        dtype, shape, offs = t.get("dtype"), t.get("shape"), t.get("data_offsets")
        itemsize = DTYPE_ITEMSIZE.get(dtype)
        if itemsize is None:
            raise SegmentIntegrityError(f"{who}: {name}: unknown dtype {dtype!r}")
        if not isinstance(shape, list) or any(
                isinstance(d, bool) or not isinstance(d, int) or d < 0 for d in shape):
            raise SegmentIntegrityError(f"{who}: {name}: bad shape {shape!r}")
        if not isinstance(offs, list) or len(offs) != 2 or any(
                isinstance(v, bool) or not isinstance(v, int) for v in offs):
            raise SegmentIntegrityError(f"{who}: {name}: bad data_offsets {offs!r}")
        a, b = offs
        if not 0 <= a <= b <= body:
            raise SegmentIntegrityError(
                f"{who}: {name}: data_offsets [{a}, {b}] outside the "
                f"{body}-byte body")
        want = itemsize
        for d in shape:
            want *= d
        if b - a != want:
            raise SegmentIntegrityError(
                f"{who}: {name}: {b - a} bytes for {dtype}{shape} "
                f"(dtype/shape imply {want})")
        spans.append((a, b, name))
    spans.sort()
    for (a1, b1, n1), (a2, b2, n2) in zip(spans, spans[1:]):
        if a2 < b1:
            raise SegmentIntegrityError(
                f"{who}: {n1} [{a1}, {b1}) overlaps {n2} [{a2}, {b2})")
    return hdr, meta, body_off


def validate_safetensors_file(path: Path) -> tuple[dict, dict, int, int]:
    """validate_safetensors_view for a path: (tensors, meta, body_off, size)."""
    size = path.stat().st_size
    if size < 8:
        raise SegmentIntegrityError(f"{path.name}: file too small ({size} bytes)")
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            hdr, meta, body_off = validate_safetensors_view(mm, size, path.name)
    return hdr, meta, body_off, size


def load_segment_index(seg_dir: Path, k: int) -> dict:
    p = seg_dir / f"index-k{k}.json"
    return json.loads(p.read_text()) if p.exists() else {}


class SegmentReader:
    """Random access to expert tensors inside one segment file.

    The header is strictly validated on open from the SAME mapping the tensor
    reads (and the verifier's digests) use, so what was checked is what is
    read: a segment file replaced on disk after verification cannot slip in
    through a second open().
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.f = open(self.path, "rb")
        self.size = os.fstat(self.f.fileno()).st_size
        if self.size < 8:
            self.f.close()
            raise SegmentIntegrityError(
                f"{self.path.name}: file too small ({self.size} bytes)")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            self.hdr, self.meta, self.body = validate_safetensors_view(
                self.mm, self.size, self.path.name)
        except Exception:
            self.close()
            raise

    def tensor_bytes(self, name: str) -> bytes:
        a, b = self.hdr[name]["data_offsets"]
        return self.mm[self.body + a: self.body + b]

    def tensor_range(self, name: str) -> tuple[int, int, int]:
        """(fd, absolute_file_offset, length) of one tensor's bytes — the
        fd-level view of tensor_bytes, for os.copy_file_range."""
        a, b = self.hdr[name]["data_offsets"]
        return self.f.fileno(), self.body + a, b - a

    def close(self):
        self.mm.close()
        self.f.close()


# ------------------------------------------------------------- verification

def expert_spans(hdr: dict, body_off: int) -> dict[int, tuple[int, int]]:
    """Absolute [lo, hi) file range of each expert's contiguous tensor run.

    Per-expert contiguity is the segment format's core promise (it is what
    makes a single ranged read fetch one expert); a segment that breaks it
    cannot be checked against the signed per-expert digests, so it is
    rejected rather than silently hashed a different way.
    """
    groups: dict[int, list[str]] = {}
    for name in hdr:
        m = EXPERT_RE.match(name)
        if m:
            groups.setdefault(int(m.group(2)), []).append(name)
    spans = {}
    for eid, names in groups.items():
        names.sort(key=expert_key)
        lo = cur = hdr[names[0]]["data_offsets"][0]
        for n in names:
            a, b = hdr[n]["data_offsets"]
            if a != cur:
                raise SegmentIntegrityError(
                    f"expert {eid}: {n} is not contiguous with its siblings "
                    f"(starts at {a}, expected {cur})")
            cur = b
        spans[eid] = (body_off + lo, body_off + cur)
    return spans


def digest_view_and_spans(view, size: int, spans: dict[str, tuple[int, int]]
                          ) -> tuple[str, dict[str, str]]:
    """One pass over a mapped file: whole-file sha256 plus one sha256 per
    named absolute [lo, hi) span."""
    whole = hashlib.sha256()
    hashers = {key: hashlib.sha256() for key in spans}
    for pos in range(0, size, CHUNK):
        end = min(pos + CHUNK, size)
        block = view[pos:end]
        whole.update(block)
        for key, (lo, hi) in spans.items():
            if lo < end and pos < hi:
                hashers[key].update(block[max(lo, pos) - pos: min(hi, end) - pos])
    return whole.hexdigest(), {k: h.hexdigest() for k, h in hashers.items()}


def digest_file_and_spans(path: Path, spans: dict[str, tuple[int, int]]
                          ) -> tuple[str, dict[str, str]]:
    """digest_view_and_spans for a path (tests, one-off checks)."""
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return digest_view_and_spans(mm, path.stat().st_size, spans)


def _resolve_explicit_signer(token: str, trust_root: Path | None) -> str:
    """Resolve one explicit pin with the shared trust-root contract.

    Preserve assembler's legacy ``ed25519:`` spelling and bare local ``.pub``
    filenames while handing their canonical values to fq_trust.  A trust-root
    key id always wins over a same-named local file.
    """
    spec = token.strip()
    root = None
    if spec.lower().startswith("ed25519:"):
        try:
            root = TrustRoot.load(trust_root)
        except TrustError as e:
            raise VerificationError(str(e)) from e
        if not any(record["key_id"] == spec for record in root.records):
            spec = spec.split(":", 1)[1]
    pub_path = Path(spec)
    if (pub_path.suffix.lower() == ".pub" and pub_path.is_file()
            and os.sep not in spec):
        if root is None:
            try:
                root = TrustRoot.load(trust_root)
            except TrustError as e:
                raise VerificationError(str(e)) from e
        if not any(record["key_id"] == spec for record in root.records):
            spec = f".{os.sep}{spec}"
    try:
        verifier = TrustVerifier.resolve(
            trust_signer=spec, trust_root=trust_root)
    except TrustError as e:
        raise VerificationError(str(e)) from e
    assert verifier.fingerprint is not None
    return verifier.fingerprint


def load_trusted_signers(values, files, *, trust_root: Path | None = None) -> list[str]:
    """Pinned ed25519 fingerprints from --trust-signer and --trust-file.

    Explicit signer values use fq_trust's resolution contract: a full
    fingerprint, an unambiguous >=16-hex prefix, a trust-root key id, or a
    .pub path.  This remains an explicit pin: every accepted attestation must
    verify under one of these exact resolved keys.  A --trust-file line may be
    a bare fingerprint or a project trust-root record
    (`<fingerprint> <key-id> <status> ...`); records marked revoked are never
    trusted.
    """
    out = []
    for raw in values or []:
        try:
            out.append(_resolve_explicit_signer(raw, trust_root))
            continue
        except VerificationError as single_error:
            tokens = raw.replace(",", " ").split()
            if len(tokens) <= 1:
                raise
            try:
                root = TrustRoot.load(trust_root)
            except TrustError:
                raise single_error
            if any(record["key_id"] == raw.strip() for record in root.records):
                raise single_error
        out.extend(_resolve_explicit_signer(tok, trust_root) for tok in tokens)
    for path in files or []:
        path = Path(path)
        if not path.exists():
            raise VerificationError(f"--trust-file {path} does not exist")
        for line in path.read_text().splitlines():
            fields = line.split("#", 1)[0].split()
            if not fields:
                continue
            if len(fields) >= 3 and fields[2].lower() == "revoked":
                print(f"note: {path.name}: skipping revoked signer "
                      f"{fields[0][:16]}…", file=sys.stderr)
                continue
            out.append(_resolve_explicit_signer(fields[0], trust_root))
    return sorted(set(out))


def verify_attestation(path: Path, trusted: set[str], allowed: set[str]
                       ) -> tuple[dict, str]:
    """(payload, keyid) of the first attestation line signed by a pinned key.

    A line whose keyid IS pinned but whose signature does not verify is fatal
    (that is tampering, not an unrelated signer); lines from unpinned keys are
    skipped, and running out of lines is fatal too.
    """
    if not path.exists():
        raise VerificationError(
            f"no attestation at {path} — every segment must ship a signed "
            f"fq-attestation line to be assembled")
    try:
        from nacl.signing import VerifyKey
    except ImportError:
        raise VerificationError(
            "pynacl is required to verify segment attestations "
            "(pip install pynacl); --insecure skips verification entirely")
    seen = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            payload_b64, sig_b64 = obj["payload"], obj["signature"]
            keyid = str(obj["keyid"]).strip().lower()
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise VerificationError(f"{path.name}:{lineno}: malformed attestation ({e})")
        seen.append(keyid[:16])
        if keyid not in trusted:
            continue
        raw = base64.b64decode(payload_b64)
        try:
            VerifyKey(bytes.fromhex(keyid)).verify(raw, base64.b64decode(sig_b64))
        except Exception:  # noqa: BLE001 — any failure here is a bad signature
            raise VerificationError(
                f"{path.name}:{lineno}: ed25519 signature does not verify under "
                f"pinned signer {keyid[:16]}… — the attestation was tampered with")
        payload = json.loads(raw)
        predicate = payload.get("predicate")
        if predicate not in allowed:
            raise VerificationError(
                f"{path.name}:{lineno}: predicate {predicate!r} is not allowed "
                f"(allowed: {sorted(allowed)}) — pass --allow-predicate to widen")
        return payload, keyid
    raise VerificationError(
        f"{path.name}: no attestation line signed by a pinned signer "
        f"(saw keyids {seen or ['<none>']}, pinned {sorted(trusted)[:4]})")


class SegmentVerifier:
    """Authenticates every (layer, K) segment an assembly consumes."""

    def __init__(self, seg_dir: Path, *, trusted, allowed, manifest, policy,
                 insecure: bool):
        self.dir = Path(seg_dir)
        self.trusted = set(trusted)
        self.allowed = set(allowed)
        self.manifest = manifest or {}
        self.insecure = insecure
        self.expect: dict[str, tuple[str, str]] = {}
        for field in ("base_model", "layout"):
            for origin, obj in (("the family manifest", self.manifest),
                                ("the policy", policy or {})):
                value = obj.get(field)
                if value is None:
                    continue
                have = self.expect.get(field)
                if have is not None and have[0] != value:
                    raise VerificationError(
                        f"{field} mismatch before any segment was read: "
                        f"{have[1]} says {have[0]!r}, {origin} says {value!r}")
                self.expect.setdefault(field, (value, origin))
        self.records: dict[tuple[int, int], dict] = {}

    def verify(self, layer: int, k: int, reader: SegmentReader, experts) -> dict:
        """Validate + authenticate one OPEN segment for the experts consumed.

        Everything is computed from the reader's mapping — the same bytes the
        assembler will copy out — so there is no verify-then-reopen window.
        """
        seg_path = reader.path
        experts = sorted(set(int(e) for e in experts))
        cached = self.records.get((layer, k))
        if cached is not None and set(experts) <= set(cached["experts_checked"]):
            return cached
        hdr, meta, body_off, size = reader.hdr, reader.meta, reader.body, reader.size
        all_spans = expert_spans(hdr, body_off)
        missing = [e for e in experts if e not in all_spans]
        if missing:
            raise VerificationError(
                f"{seg_path.name}: the policy routes experts {missing} to K{k} "
                f"but the segment does not contain them")
        spans = {str(e): all_spans[e] for e in experts}
        file_sha, expert_sha = digest_view_and_spans(reader.mm, size, spans)
        record = {
            "file": seg_path.name, "sha256": file_sha, "size": size,
            "layer": layer, "k": k, "experts": len(spans),
            "experts_checked": experts,
        }
        payload = None
        if self.insecure:
            record.update(predicate=None, keyid=None, attestation=None,
                          verified=False)
        else:
            att = self.dir / "attestations" / f"{seg_path.stem}.jsonl"
            payload, keyid = verify_attestation(att, self.trusted, self.allowed)
            frag = payload.get("fragment") or {}
            if frag.get("file") != seg_path.name:
                raise VerificationError(
                    f"{seg_path.name}: attestation describes fragment "
                    f"{frag.get('file')!r}")
            if frag.get("sha256") != file_sha:
                raise VerificationError(
                    f"{seg_path.name}: content sha256 {file_sha[:16]}… does not "
                    f"match the signed fragment digest "
                    f"{str(frag.get('sha256'))[:16]}… — the segment changed "
                    f"after it was attested")
            if frag.get("size") is not None and frag["size"] != size:
                raise VerificationError(
                    f"{seg_path.name}: signed size {frag['size']} != {size}")
            digests = payload.get("expert_sha256") or {}
            for e_s in spans:
                want = digests.get(e_s)
                if want is None:
                    raise VerificationError(
                        f"{seg_path.name}: attestation carries no digest for "
                        f"expert {e_s} (consumed by this policy)")
                if want != expert_sha[e_s]:
                    raise VerificationError(
                        f"{seg_path.name}: expert {e_s} digest "
                        f"{expert_sha[e_s][:16]}… != signed {want[:16]}…")
            record.update(predicate=payload.get("predicate"), keyid=keyid,
                          attestation=f"attestations/{att.name}", verified=True)
        self._check_compat(seg_path.name, meta, payload, hdr, layer, k)
        self.records[(layer, k)] = record
        return record

    def _check_compat(self, name, meta, payload, hdr, layer, k) -> None:
        got = _as_int(meta.get("layer"))
        if got is not None and got != layer:
            raise VerificationError(
                f"{name}: segment declares layer {got}, assembling layer {layer}")
        got = _as_int(meta.get("k"))
        if got is not None and got != k:
            raise VerificationError(
                f"{name}: segment declares k={got}, policy routes it as K{k}")
        held = len({int(EXPERT_RE.match(n).group(2)) for n in hdr
                    if EXPERT_RE.match(n)})
        declared = _as_int(meta.get("num_experts"))
        if declared is not None and declared != held:
            raise VerificationError(
                f"{name}: metadata says num_experts={declared}, header holds {held}")
        family_n = _as_int(self.manifest.get("num_experts"))
        if family_n is not None and held > family_n:
            raise VerificationError(
                f"{name}: holds {held} experts, the family manifest declares "
                f"at most {family_n}")
        for field in ("layout", "base_model"):
            value = meta.get(field)
            if value is None:
                continue
            have = self.expect.get(field)
            if have is None:
                self.expect[field] = (value, f"segment {name}")
            elif have[0] != value:
                raise VerificationError(
                    f"{name}: {field}={value!r} but {have[1]} pins {have[0]!r}")
        if not payload:
            return
        for field, want in (("layer", layer), ("k", k)):
            got = _as_int(payload.get(field))
            if got is not None and got != want:
                raise VerificationError(
                    f"{name}: attestation says {field}={got}, expected {want}")
        for field in ("layout", "base_model"):
            got, have = payload.get(field), meta.get(field)
            if got is not None and have is not None and got != have:
                raise VerificationError(
                    f"{name}: attestation {field}={got!r} contradicts segment "
                    f"metadata {have!r}")


# ------------------------------------------------------------------ writing

def _copy_file_range_region(out, src_fd: int, src_off: int, length: int) -> bool:
    """Write length bytes from src_fd[src_off:] at out's current position
    via os.copy_file_range (server-side copy; XFS/btrfs may share extents).

    Returns True when the whole region was copied and out's position was
    advanced past it.  Returns False — with out's position restored to the
    start of the region — when the syscall is unavailable, refused (EXDEV,
    EOPNOTSUPP, ENOSYS, EINVAL, ...), or stalls, including after partial
    progress; the caller then rewrites the SAME region through the ordinary
    read()+write() path, so the output bytes are identical either way.
    """
    cfr = _COPY_FILE_RANGE
    if cfr is None:
        return False
    out.flush()
    dst_fd = out.fileno()
    dst_off = out.tell()
    done = 0
    while done < length:
        try:
            n = cfr(src_fd, dst_fd, length - done,
                    src_off + done, dst_off + done)
        except OSError:
            n = 0
        if n <= 0:  # error or unexpected EOF: caller rewrites the region
            out.seek(dst_off)
            return False
        done += n
    out.seek(dst_off + length)
    return True


def _needs_reindex(hdr: dict, bits_for_expert, readers: dict[int, SegmentReader]) -> bool:
    """True when any expert tensor's segment size differs from its source slot."""
    for name, t in hdr.items():
        m = EXPERT_RE.match(name)
        if not m:
            continue
        k = bits_for_expert(int(m.group(2)))
        seg = readers[k].hdr.get(name)
        if seg is None:
            raise KeyError(f"segment k{k} is missing tensor {name}")
        if (seg["data_offsets"][1] - seg["data_offsets"][0]
                != t["data_offsets"][1] - t["data_offsets"][0]):
            return True
    return False


def assemble_layer(
    src_shard: Path,
    out_shard: Path,
    layer: int,
    bits_for_expert,           # callable expert_id -> K
    readers: dict[int, SegmentReader],
    reflink: bool = False,
) -> dict:
    """Write one output shard.

    Uniform-K (all segment sizes match the source slots): the source header
    is reused verbatim as a byte template — output is byte-identical to the
    source when the policy matches the source K.  Mixed K: reindex path —
    same tensor ORDER as the source, but the header is rebuilt with
    dtype/shape from the segment headers for expert tensors and fresh
    contiguous data_offsets.

    reflink=True: expert-tensor regions go through os.copy_file_range with
    per-region fallback to the ordinary path (see module docstring); the
    returned counts dict then carries "reflinked"/"copied" region tallies
    alongside the per-K tensor counts.  Output bytes are identical either
    way.
    """
    hdr, body_off = read_header(src_shard)
    meta = hdr.pop("__metadata__", None)
    counts = {}
    reindex = _needs_reindex(hdr, bits_for_expert, readers)
    with open(src_shard, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        ordered = sorted(hdr.items(), key=lambda kv: kv[1]["data_offsets"][0])
        if reindex:
            rebuilt, off = {}, 0
            for name, t in ordered:
                m = EXPERT_RE.match(name)
                if m:
                    seg = readers[bits_for_expert(int(m.group(2)))].hdr[name]
                    dtype, shape = seg["dtype"], seg["shape"]
                    size = seg["data_offsets"][1] - seg["data_offsets"][0]
                else:
                    dtype, shape = t["dtype"], t["shape"]
                    size = t["data_offsets"][1] - t["data_offsets"][0]
                rebuilt[name] = {"dtype": dtype, "shape": shape,
                                 "data_offsets": [off, off + size]}
                off += size
            ordered = [(n, rebuilt[n]) for n, _ in ordered]
            header = rebuilt if meta is None else {"__metadata__": meta, **rebuilt}
        else:
            header = dict(hdr) if meta is None else {"__metadata__": meta, **hdr}
        hj = json.dumps(header, separators=(",", ":")).encode()
        hj += b" " * ((8 - len(hj) % 8) % 8)
        tmp = out_shard.with_suffix(".part")
        with open(tmp, "wb") as out:
            out.write(struct.pack("<Q", len(hj)))
            out.write(hj)
            for name, t in ordered:
                want = t["data_offsets"][1] - t["data_offsets"][0]
                m = EXPERT_RE.match(name)
                if m:
                    expert = int(m.group(2))
                    k = bits_for_expert(expert)
                    counts[k] = counts.get(k, 0) + 1
                    if reflink:
                        src_fd, src_off, size = readers[k].tensor_range(name)
                        if size != want:
                            raise AssertionError(
                                f"{name}: got {size} bytes, header slot {want}")
                        if _copy_file_range_region(out, src_fd, src_off, size):
                            counts["reflinked"] = counts.get("reflinked", 0) + 1
                            continue
                        counts["copied"] = counts.get("copied", 0) + 1
                    data = readers[k].tensor_bytes(name)
                else:
                    src_t = hdr[name]
                    a, b = src_t["data_offsets"]
                    data = mm[body_off + a: body_off + b]
                if len(data) != want:
                    raise AssertionError(
                        f"{name}: got {len(data)} bytes, header slot {want}")
                out.write(data)
        mm.close()
    os.replace(tmp, out_shard)
    return counts


BITS_PER_EXPERT_REF = "tier_bitmap.json:bits_per_expert"


def emit_mixed_metadata(out_dir: Path, bpe: dict, source: Path) -> None:
    """Synchronize every loader-visible copy of the bit allocation.

    tier_bitmap.json always records what was actually assembled (so a bitmap
    inherited from a previous, differently-mixed build cannot survive), and
    config.json is UPDATED — never setdefault'd — so quantization_config.bits
    cannot keep saying 3.0 next to a mixed hybrid_tr3_tail.  Uniform policy:
    hybrid_tr3_tail.bits is the single K and the mixed fields are dropped;
    mixed policy: bits="mixed", k_values, and the bits_per_expert
    "file.json:field" reference per the GG loader contract.
    """
    ks = sorted({int(k) for bits in bpe.values() for k in bits})
    if not ks:
        return
    mixed = len(ks) > 1
    bitmap_name, field = BITS_PER_EXPERT_REF.rsplit(":", 1)
    src_bitmap = source / bitmap_name
    # mixed policies need the map (the config references it); uniform policies
    # only correct a bitmap the source already shipped — never invent a file
    # the checkpoint did not have
    if mixed or src_bitmap.exists():
        bitmap = json.loads(src_bitmap.read_text()) if src_bitmap.exists() else {}
        for layer_s, bits in bpe.items():
            bitmap.setdefault(layer_s, {})[field] = [int(b) for b in bits]
        (out_dir / bitmap_name).write_text(json.dumps(bitmap, indent=1) + "\n")

    cfg_path = out_dir / "config.json"
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text())
    tail = cfg.get("hybrid_tr3_tail")
    if not isinstance(tail, dict):
        return
    before = json.dumps(cfg, sort_keys=True)
    if mixed:
        tail["bits"] = "mixed"
        tail["k_values"] = ks
        tail["bits_per_expert"] = BITS_PER_EXPERT_REF
    else:
        tail["bits"] = float(ks[0])
        for key in ("k_values", "bits_per_expert"):
            tail.pop(key, None)
    qc = cfg.get("quantization_config")
    if not isinstance(qc, dict) and mixed:
        # vLLM resolves the quant method from hf_config.quantization_config
        # (weight_utils.get_quant_config) BEFORE Exl3Config.maybe_update_config
        # reads hybrid_tr3_tail; without this stub it demands a
        # quantization_config.json file the rank-sliced format does not ship.
        qc = {}
        cfg["quantization_config"] = qc
    if isinstance(qc, dict):
        qc["quant_method"] = "exl3"
        qc["bits"] = "mixed" if mixed else float(ks[0])
        qc.setdefault("codebook", tail.get("codebook", "mcg"))
        qc.setdefault("version", tail.get("exllamav3_version", "rank-sliced"))
    if json.dumps(cfg, sort_keys=True) != before:
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")


def regenerate_shard_index(out_dir: Path) -> None:
    """Rebuild model.safetensors.index.json from the output shards."""
    weight_map, total = {}, 0
    for shard in sorted(out_dir.glob("model-*.safetensors")):
        hdr, _ = read_header(shard)
        hdr.pop("__metadata__", None)
        for name, t in hdr.items():
            weight_map[name] = shard.name
            total += t["data_offsets"][1] - t["data_offsets"][0]
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=1, sort_keys=True) + "\n")


def hash_tree(out_dir: Path, skip=()) -> dict[str, str]:
    """{relative path: sha256} for every file under out_dir."""
    shas = {}
    for f in sorted(out_dir.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(out_dir).as_posix()
        if rel in skip:
            continue
        shas[rel] = sha256_file(f)
    return shas


def regenerate_manifest(out_dir: Path, known: dict[str, str] | None = None) -> dict:
    """Rewrite MANIFEST.sha256 (sha256sum format) for the output files.

    Always called after an assembly: a same-size payload change leaves file
    sizes untouched, so anything short of rehashing publishes a stale digest.
    """
    known = known or {}
    shas = {}
    for f in sorted(out_dir.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(out_dir).as_posix()
        if rel == "MANIFEST.sha256":
            continue
        shas[rel] = known.get(rel) or sha256_file(f)
    (out_dir / "MANIFEST.sha256").write_text(
        "\n".join(f"{shas[rel]}  {rel}" for rel in sorted(shas)) + "\n")
    return shas


# ------------------------------------------------------------ output safety

def _resolved(p: Path) -> Path:
    """Absolute, symlink-free path — the only form the safety checks compare.

    Computed with strict=False so a not-yet-created --out still resolves.
    """
    return Path(p).expanduser().resolve()


def _is_parent_or_equal(a: Path, b: Path) -> bool:
    return a == b or a in b.parents


def _overlaps(a: Path, b: Path) -> bool:
    """True when deleting `a` could delete part of `b`, or vice versa."""
    return _is_parent_or_equal(a, b) or _is_parent_or_equal(b, a)


def _repo_paths() -> list[Path]:
    """This checkout: the tools dir and the git root above it, if any."""
    here = Path(__file__).resolve().parent
    out = [here]
    for cand in [here, *here.parents]:
        if (cand / ".git").exists():
            out.append(cand)
            break
    else:
        out.append(here.parent)
    return out


def _cache_paths() -> list[Path]:
    """Upstream model caches — never a legal assembly target."""
    out = []
    for var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE",
                "TRANSFORMERS_CACHE", "XDG_CACHE_HOME"):
        value = os.environ.get(var)
        if value:
            out.append(_resolved(Path(value)))
    home = Path.home()
    out += [_resolved(home / ".cache" / "huggingface"),
            _resolved(home / ".cache")]
    return out


def check_out_dir(out: Path, *, source: Path | None = None,
                  segments: Path | None = None,
                  policy: Path | None = None) -> Path:
    """Resolve --out and refuse every destination whose purge would be a loss.

    `--force` deletes recursively, so the argument that reaches it has to be
    proven safe BEFORE anything runs: a typo'd `--out .`, an `--out ~`, or an
    `--out <the source checkpoint>` must be an error message, never a deleted
    model.  Refusals here are absolute — no flag turns them off, because
    there is no legitimate assembly whose target is your home directory.

    Returns the resolved output path.  Raises AssemblyError with the reason.
    """
    given = Path(out).expanduser()
    if given.is_symlink():
        raise AssemblyError(
            f"--out {given} is a symlink (-> {os.readlink(given)}): refusing. "
            f"Purging through a symlink deletes the target's contents, which "
            f"is never what the link was for. Pass the real path.")
    real = _resolved(given)
    home = _resolved(Path.home())

    if real == Path(real.anchor):
        raise AssemblyError(f"--out {real} is the filesystem root: refusing.")
    if real == home:
        raise AssemblyError(
            f"--out {real} is your home directory: refusing. Assemble into a "
            f"subdirectory (e.g. {home / 'my-checkpoint'}).")
    if _is_parent_or_equal(real, home):
        raise AssemblyError(
            f"--out {real} contains your home directory ({home}): refusing.")
    for repo in _repo_paths():
        if _is_parent_or_equal(real, repo):
            raise AssemblyError(
                f"--out {real} is (or contains) the Progressive Tensors "
                f"checkout at {repo}: refusing to assemble over the tools.")
    if (real / ".git").exists():
        raise AssemblyError(
            f"--out {real} looks like a git checkout (it has a .git): "
            f"refusing. Assemble into a directory that is not a repository.")
    for cache in _cache_paths():
        if _overlaps(real, cache):
            raise AssemblyError(
                f"--out {real} overlaps the upstream model cache {cache}: "
                f"refusing. Assembling there would put a derived checkpoint "
                f"inside the cache it was derived from — and --force would "
                f"delete the cache.")
    for flag, value in (("--source", source), ("--segments", segments)):
        if value is None:
            continue
        other = _resolved(value)
        if _overlaps(real, other):
            raise AssemblyError(
                f"--out {real} overlaps {flag} {other}: refusing. The output "
                f"directory is replaced wholesale with --force and rebuilt "
                f"from these inputs; it can never be the same tree as one of "
                f"them, or contain it, or live inside it.")
    if policy is not None:
        # A recipe sitting next to the output dir is normal; a recipe INSIDE
        # it would be deleted by the very --force that reads it.
        pol = _resolved(policy)
        if _is_parent_or_equal(real, pol):
            raise AssemblyError(
                f"--out {real} contains the policy file {pol}: refusing, "
                f"--force would delete the recipe this run is reading.")
    if real.exists() and not real.is_dir():
        raise AssemblyError(f"{real} exists and is not a directory")
    return real


def write_sentinel(out: Path) -> None:
    """Mark a directory as fq_assemble's own output (see SENTINEL)."""
    (out / SENTINEL).write_text(json.dumps({
        "schema": SENTINEL_SCHEMA,
        "tool": TOOL_VERSION,
        "created_utc": now_utc(),
        "note": ("written by fq_assemble; --force will only purge a directory "
                 "that carries this file"),
    }, indent=1, sort_keys=True) + "\n")


def _has_sentinel(out: Path) -> bool:
    p = out / SENTINEL
    if not p.is_file() or p.is_symlink():
        return False
    try:
        return json.loads(p.read_text()).get("schema") == SENTINEL_SCHEMA
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False


class StagedOutput:
    """Assemble into a sibling staging dir, then swap it into place.

    Nothing is ever written into --out itself and nothing is deleted from it
    until a complete checkpoint exists next to it.  The swap is two renames
    (out -> trash, staging -> out), so the destination is always either the
    previous complete checkpoint or the new one — an interrupt can leave a
    stale `.<name>.fq-staging` dir behind, never a half-purged output.
    """

    def __init__(self, out: Path, force: bool):
        self.out = out
        self.force = force
        self.staging = out.parent / f".{out.name}.fq-staging"
        self.trash = out.parent / f".{out.name}.fq-trash"
        self.committed = False
        self.replaced = False

    def begin(self) -> Path:
        """Validate the destination, create the staging dir, return it."""
        if self.out.exists():
            entries = sorted(p.name for p in self.out.iterdir())
            if entries and not self.force:
                raise AssemblyError(
                    f"{self.out} is not empty ({len(entries)} entries, e.g. "
                    f"{entries[:3]}): assembling here would merge this recipe "
                    f"into whatever is already there. Use a fresh --out, or "
                    f"--force to replace it.")
            if entries and not _has_sentinel(self.out):
                raise AssemblyError(
                    f"refusing --force on {self.out}: it holds {len(entries)} "
                    f"entries (e.g. {entries[:3]}) but no {SENTINEL} marker, "
                    f"so fq_assemble did not create it. --force deletes the "
                    f"whole directory tree; it will only do that to its own "
                    f"previous output. If you really mean to replace this "
                    f"directory, move or delete it yourself first.")
            self.replaced = bool(entries)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        for leftover in (self.staging, self.trash):
            if leftover.exists():
                print(f"note: removing leftover {leftover} from an "
                      f"interrupted run", flush=True)
                shutil.rmtree(leftover)
        self.staging.mkdir()
        write_sentinel(self.staging)
        return self.staging

    def commit(self) -> None:
        """Swap the staging dir into place; then drop the previous output."""
        moved = False
        if self.out.exists():
            os.rename(self.out, self.trash)
            moved = True
        try:
            os.rename(self.staging, self.out)
        except OSError:
            if moved:  # put the old checkpoint back before re-raising
                os.rename(self.trash, self.out)
            raise
        self.committed = True
        if moved:
            shutil.rmtree(self.trash, ignore_errors=True)
            if self.replaced:
                print(f"--force: replaced the previous assembly in {self.out}",
                      flush=True)

    def abort(self) -> None:
        """Drop the staging dir; the destination was never touched."""
        if not self.committed and self.staging.exists():
            shutil.rmtree(self.staging, ignore_errors=True)


def build_assembly_record(*, policy_path: Path, policy: dict, segments: Path,
                          source: Path, seg_records: list[dict],
                          products: dict[str, str], verification: dict,
                          assembled_layers: list[int]) -> dict:
    """The `assembly-of` provenance record (ATTESTATION-V2.md rung).

    Recipe sha + per-segment shas + per-output-shard shas + tool version +
    UTC: enough for a third party to re-run the same assembly and compare, or
    to tell which segment bytes a given checkpoint was built from.
    """
    src_manifest = source / "MANIFEST.sha256"
    return {
        "schema": ATTESTATION_SCHEMA,
        "predicate": "assembly-of",
        "created_utc": now_utc(),
        "tool": {"name": "fq_assemble", "version": TOOL_VERSION},
        "recipe": {
            "file": policy_path.name,
            "sha256": sha256_file(policy_path),
            "schema": policy.get("schema"),
            "k_values": sorted({int(k) for bits in
                                policy.get("bits_per_expert", {}).values()
                                for k in bits}),
            "layers": sorted(int(L) for L in policy.get("bits_per_expert", {})),
            # --layers can assemble a subset of the recipe: say which
            "layers_assembled": sorted(assembled_layers),
        },
        "materials": {
            "segments": {
                "dir": segments.name,
                "fragments": sorted(
                    ({k: v for k, v in r.items() if k != "experts_checked"}
                     for r in seg_records),
                    key=lambda r: (r["layer"], r["k"])),
            },
            "source": {
                "dir": source.name,
                "manifest_sha256": (sha256_file(src_manifest)
                                    if src_manifest.exists() else None),
            },
        },
        "products": [{"file": name, "sha256": products[name]}
                     for name in sorted(products)],
        "verification": verification,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--segments", required=True, type=Path, help="segment dir (repack output)")
    p.add_argument("--source", required=True, type=Path, help="source checkpoint snapshot")
    p.add_argument("--policy", required=True, type=Path, help="fq-policy JSON (bits_per_expert)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--layers", default=None)
    p.add_argument(
        "--force", action="store_true",
        help=f"replace a non-empty output dir (never merges).  Only a "
             f"directory carrying the {SENTINEL} marker written by a previous "
             f"fq_assemble run can be replaced; anything else is refused so a "
             f"mistyped --out cannot delete your data.")
    p.add_argument(
        "--sign-key", type=Path, default=None,
        help="ed25519 seed used to also emit a signed assembly-of attestation "
             "line next to the checkpoint (attestations/assembly-of.jsonl)")
    trust = p.add_argument_group("verification (mandatory unless --insecure)")
    trust.add_argument(
        "--trust-signer", action="append", metavar="FINGERPRINT",
        help="pin an expected ed25519 signer (repeatable): a 64-hex "
             "fingerprint, an unambiguous >=16-hex prefix, a key id from "
             "keys/FINGERPRINTS, or a .pub path. Signatures must validate "
             "under exactly this key; confirm it out of band before pinning")
    trust.add_argument(
        "--trust-root", type=Path, metavar="PATH",
        help="keys/FINGERPRINTS (or keys/ dir) used to resolve --trust-signer "
             "prefixes and key ids (default: shipped root or $FQ_TRUST_ROOT)")
    trust.add_argument(
        "--trust-file", action="append", type=Path, metavar="PATH",
        help="file of pinned signer fingerprints, one per line ('#' "
             "comments); also reads the project trust-root format "
             "(`<fingerprint> <key-id> <status> ...`, e.g. keys/FINGERPRINTS), "
             "skipping revoked records")
    trust.add_argument(
        "--allow-predicate", action="append", metavar="NAME",
        help=f"attestation predicates accepted for segments "
             f"(default: {', '.join(DEFAULT_ALLOWED_PREDICATES)})")
    trust.add_argument(
        "--insecure", "--insecure-skip-signatures", action="store_true",
        dest="insecure",
        help="assemble WITHOUT verifying segment attestations — local "
             "development only")
    p.add_argument(
        "--reflink", action="store_true",
        help="write expert-tensor bytes from segment files with "
             "os.copy_file_range so reflink-capable filesystems (XFS with "
             "reflink=1, btrfs) can share extents with the segments instead "
             "of storing the bytes twice. Caveats: the kernel/filesystem "
             "decides whether extents are actually shared — copy_file_range "
             "may silently do a plain copy; savings depend on 4K-block "
             "alignment of identical regions at both source and destination "
             "offsets (MB-scale fragments usually share most interior "
             "blocks, but alignment is not guaranteed); output bytes are "
             "ALWAYS identical to the normal path (automatic per-region "
             "fallback to read()+write() on EXDEV/EOPNOTSUPP/etc.); saves "
             "local disk space only — no effect on HF/remote.")
    args = p.parse_args(argv)

    # Before anything is read, written or deleted: prove --out is a legal
    # destination.  --force purges recursively, so this check is what stands
    # between a mistyped path and somebody's model collection.
    args.out = check_out_dir(args.out, source=args.source,
                             segments=args.segments, policy=args.policy)

    policy = json.loads(args.policy.read_text())
    bpe = policy["bits_per_expert"]
    trusted = load_trusted_signers(args.trust_signer, args.trust_file,
                                   trust_root=args.trust_root)
    allowed = set(args.allow_predicate or DEFAULT_ALLOWED_PREDICATES)
    if not trusted and not args.insecure:
        raise VerificationError(
            "refusing to assemble unverified segments. Pin the signer(s) you "
            "expect with --trust-signer <fingerprint> (64-hex, an "
            "unambiguous >=16-hex prefix, a key id, or a .pub path; "
            "repeatable) or --trust-file <path>; the segment family's "
            "fq-manifest.json publishes its signer as \"signer_pubkey\", so "
            "confirm that fingerprint out of band and pin it. --insecure "
            "assembles without any verification and is for local development "
            "only.")
    if args.insecure:
        print("!" * 72 + "\n"
              "!! --insecure: segment attestations are NOT being verified.\n"
              "!! Assembled bytes are unauthenticated: anyone who can write to\n"
              f"!! {args.segments} can put arbitrary weights in this checkpoint.\n"
              "!! Never use --insecure for anything you will serve.\n"
              + "!" * 72, file=sys.stderr, flush=True)

    man_path = args.segments / "fq-manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    verifier = SegmentVerifier(args.segments, trusted=trusted, allowed=allowed,
                               manifest=manifest, policy=policy,
                               insecure=args.insecure)

    wanted = None
    if args.layers:
        wanted = set()
        for part in args.layers.split(","):
            if "-" in part:
                a, b = part.split("-")
                wanted.update(range(int(a), int(b) + 1))
            else:
                wanted.add(int(part))

    jobs = []
    for src_shard in sorted(args.source.glob("model-layer-*.safetensors")):
        layer = int(src_shard.stem.split("-")[-1])
        if wanted is not None and layer not in wanted:
            continue
        jobs.append((src_shard, layer, bpe.get(str(layer))))

    layout = policy.get("layout") or manifest.get("layout")
    source_targets = {
        layer: set(range(len(lb)))
        for _src, layer, lb in jobs
        if lb is not None
    }
    try:
        # Source-only evidence is checked before a segment is even opened.
        preflight_source_layout(args.source, layout=layout, targets=source_targets)
    except SourceLayoutError as e:
        raise AssemblyError(str(e)) from e

    # Every segment is opened and authenticated BEFORE the output dir is
    # touched: a bad fragment (or a bad trust configuration) must never cost
    # the operator a previous checkpoint via --force, and the readers stay
    # open so the assembler copies exactly the bytes that were verified.
    readers: dict[tuple[int, int], SegmentReader] = {}
    total, seg_records = {}, []
    staged = StagedOutput(args.out, args.force)
    try:
        segment_targets = {}
        for _src, layer, lb in jobs:
            for k in sorted(set(lb or ())):
                seg = args.segments / f"layer-{layer:03d}.k{k}.safetensors"
                if not seg.exists():
                    raise FileNotFoundError(
                        f"missing segment {seg} for layer {layer}")
                readers[(layer, k)] = SegmentReader(seg)
                segment_targets[(layer, k)] = {
                    e for e, assigned_k in enumerate(lb) if assigned_k == k}
        try:
            # Segment headers are enough to compare TP-bearing axes; no
            # fragment bytes have been hashed or staged at this point.
            preflight_source_layout(
                args.source, layout=layout, targets=source_targets,
                segment_headers={key: reader.hdr for key, reader in readers.items()},
                segment_targets=segment_targets)
        except SourceLayoutError as e:
            raise AssemblyError(str(e)) from e
        for _src, layer, lb in jobs:
            for k in sorted(set(lb or ())):
                seg_records.append(verifier.verify(
                    layer, k, readers[(layer, k)],
                    [e for e, assigned_k in enumerate(lb) if assigned_k == k]))
        if args.insecure:
            who = "NOT VERIFIED (--insecure)"
        else:
            keyids = sorted({r["keyid"][:16] + "…" for r in seg_records})
            who = "signed by " + ", ".join(keyids) if keyids else "none needed"
        print(f"segments: {len(seg_records)} checked, {who}", flush=True)

        # Everything below writes into the staging dir; --out is untouched
        # until the final swap, so a crash here costs nothing but the staging
        # dir.  (Note the destination is validated inside begin(), before a
        # single byte moves.)
        work = staged.begin()
        for src_shard, layer, lb in jobs:
            if lb is None:
                # dense layer (or out-of-policy): copy through
                shutil.copyfile(src_shard, work / src_shard.name)
                continue
            counts = assemble_layer(
                src_shard, work / src_shard.name, layer, lambda e: lb[e],
                {k: readers[(layer, k)] for k in set(lb)}, reflink=args.reflink)
            total[layer] = counts
            print(f"layer {layer:3d}: {counts}", flush=True)

        for r in readers.values():
            r.close()
        readers.clear()

        for extra in args.source.iterdir():
            if extra.is_file() and not extra.name.startswith("model-layer-"):
                shutil.copyfile(extra, work / extra.name)
        emit_mixed_metadata(work, bpe, args.source)

        # integrity metadata is ALWAYS rebuilt from the bytes just written: a
        # same-size payload change keeps every file size intact, so a manifest
        # that is only refreshed "when something looks different" is a manifest
        # that can lie.
        regenerate_shard_index(work)
        shas = hash_tree(work, skip=("MANIFEST.sha256", ASSEMBLY_RECORD))
        record = build_assembly_record(
            policy_path=args.policy, policy=policy, segments=args.segments,
            source=args.source, seg_records=seg_records,
            assembled_layers=list(total),
            products={n: s for n, s in shas.items()
                      if n.startswith("model-") and n.endswith(".safetensors")},
            verification={
                "mode": "INSECURE (unverified)" if args.insecure else "verified",
                "trusted_signers": sorted(trusted),
                "allowed_predicates": sorted(allowed),
                "segments_verified": sum(1 for r in seg_records
                                         if r.get("verified")),
            })
        atomic_write_json(work / ASSEMBLY_RECORD, record)
        if args.sign_key:
            att_dir = work / "attestations"
            att_dir.mkdir(exist_ok=True)
            (att_dir / "assembly-of.jsonl").write_text(
                Signer(args.sign_key).sign_line(record) + "\n")
        regenerate_manifest(work, known=shas)
        staged.commit()
    finally:
        for r in readers.values():
            r.close()
        staged.abort()
    print(f"assembled {len(total)} MoE layers -> {args.out} "
          f"({record['verification']['mode']}, "
          f"{len(seg_records)} segments)", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssemblyError as e:
        print(f"fq_assemble: {e}", file=sys.stderr)
        sys.exit(2)
