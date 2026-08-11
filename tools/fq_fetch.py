#!/usr/bin/env python3
"""fq_fetch — fetch only the expert fragments your recipe actually needs.

The consumer half of Progressive Tensors (peer review, finding 3).  The old
quickstart said `hf download <repo>`, which pulls the entire segment
repository — 347 GB for the GLM-5.2 K3 base — even when the recipe touches
eight layers.  Segments are per-expert contiguous and `index-kK.json` gives
every expert's byte span, so the right operation is an HTTP Range read of
exactly those spans.  That is what this tool does:

    recipe (fq-policy/2)  +  one or more source repos
      -> per-expert byte ranges, coalesced, verified, resumable
      -> a local segment tree that fq_assemble consumes unchanged

Nothing else about the pipeline changes: the output directory holds ordinary
`layer-LLL.kK.safetensors` files (a *subset* of experts each) plus a local
`index-kK.json`, so `fq_assemble.py --segments <out>` works as if you had
downloaded everything.

Multi-source, per-expert.  `--source` is repeatable and ordered: the first
source that carries an expert at the required K wins, unless overridden by
content hash (`--prefer-sha`, "give me this exact fragment, whoever has it")
or by an explicit provider map (`--select map.json`, per layer or per
expert).  One output segment file may therefore mix fragments from several
publishers — each verified against *its own* publisher's signed attestation,
each recorded in `fq-fetch-report.json`.  That is the community-fragment
selection model, working.

Trust.  Every attestation used is signature-checked against a fingerprint
pinned out of band (`--trust-signer`, resolved through keys/FINGERPRINTS in
the git repo — see ../TRUST.md), and every fetched expert's bytes are hashed
against the attested digest before the file is finalized.  When the source
publishes a signed `fq-release/1` manifest, one signature covers the index
and attestation files too, fq_fetch checks their digests against it, and it
refuses when the release and the attestation disagree about a fragment.

The plan itself is authenticated too (peer review, finding P1-4c).  Tensor
names, dtypes, shapes and offsets come from the remote segment's
safetensors header, and a ranged read of a header is just bytes the server
chose to send: hashing the payload afterwards proves the *bytes* are the
attested ones, not that they are the tensors the header claims.  A
publisher who rewrote a header could relabel gate_proj as up_proj, or F16
as I16, and the reconstructed file would still pass every digest check —
and then be signed as trustworthy.  So before anything plans from a header,
--header-trust decides how it is proven:

  auto      (default) use the signed `fragment.header_sha256` when the
            publisher provides one; otherwise verify the WHOLE fragment
            against the signed `fragment.sha256` and read the header out of
            those verified bytes.  Always authenticated, sometimes slow —
            the plan summary says which fragments need the full read and
            how many bytes that costs before any of it happens.
  attested  require the signed header digest; refuse rather than download.
  full      always verify the whole fragment.
  unsafe    plan from the unauthenticated ranged header (the old
            behaviour).  Prints a banner, and the emitted attestation says
            `header_authentication: NONE` so downstream can see it.

Whatever is used is recorded per parent fragment in the `derived-from`
attestation this tool emits, so the tree says exactly which authenticated
inputs its layout came from.

    # what would this cost?
    fq_fetch.py --policy recipe.json --out ./segments --dry-run \\
        --source malaiwah/GLM-5.2-EXL3-FQ-segments@<commit>

    # fetch it, pinning the signer
    fq_fetch.py --policy recipe.json --out ./segments \\
        --source malaiwah/GLM-5.2-EXL3-FQ-segments@<commit> \\
        --source willfalco/GLM-5.2-EXL3-TR3-3.36bpw-FQ@<commit> \\
        --trust-signer a58b7bb79ba58457

    # then, with the fingerprint fq_fetch prints for the subset it signed:
    fq_assemble.py --segments ./segments --source <checkpoint> \\
        --policy recipe.json --out ./my-checkpoint \\
        --trust-signer <your local fq_fetch fingerprint>

Why a second key.  A subset file is a NEW file — fewer experts, different
offsets, a different digest — so no publisher signature can cover it, even
though every expert byte in it was hashed against the publisher's signed
digest as it arrived.  fq_fetch therefore signs what it actually produced
with a local key (`--sign-key`, created on demand), as a `derived-from`
attestation naming the publisher fragments as parents and pinning them by
digest.  The publishers' own lines are kept under attestations/<source>/ so
the upstream hops stay checkable offline.  `--no-attest` skips this, and
then assembly needs `--insecure`.

Interrupt it and run it again: completed experts are recorded in
`state.json` and skipped, partial segment files are resumed in place.

HTTP: URLs and authentication come from `huggingface_hub` when it is
installed (so `hf auth login`, HF_ENDPOINT and private repos work); without
it the tool falls back to the public `/resolve/` URL shape and the HF_TOKEN
environment variable.  The ranged GETs themselves are plain urllib, so a
consumer needs no hub client at all.
"""
from __future__ import annotations

import argparse
import binascii
import base64
import hashlib
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fq_trust  # noqa: E402
from fq_repack import expert_key  # noqa: E402
from fq_trust import TrustError  # noqa: E402

SEGMENT_SCHEMA = "fq-segment/1"
MANIFEST_SCHEMA = "fq-manifest/1"
POLICY_SCHEMA = "fq-policy/2"
SELECT_SCHEMA = "fq-select/1"
FETCH_BINDING_SCHEMA = "fq-fetch-binding/1"
REPORT_SCHEMA = "fq-fetch-report/1"
USER_AGENT = "fq_fetch/0.1 (+https://github.com/malaiwah/progressive-tensors)"
DEFAULT_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

# --header-trust rungs, strongest guarantee per byte first.
HEADER_ATTESTED = "attested"    # require a signed fragment.header_sha256
HEADER_AUTO = "auto"            # signed header digest, else whole fragment
HEADER_FULL = "full"            # always verify the whole fragment
HEADER_UNSAFE = "unsafe"        # plan from an unauthenticated header
HEADER_TRUST_MODES = (HEADER_AUTO, HEADER_ATTESTED, HEADER_FULL, HEADER_UNSAFE)
# how the header of a segment was proven, as recorded in the attestation
AUTH_HEADER_DIGEST = "attested-header-digest"
AUTH_FULL_FRAGMENT = "full-fragment"
AUTH_NONE = "NONE (--header-trust unsafe)"
FULL_VERIFY_CHUNK = 1 << 26  # 64 MB per ranged GET when hashing a fragment


# ---------------------------------------------------------------- transport

class _AuthStripRedirect(urllib.request.HTTPRedirectHandler):
    """Drop Authorization when a redirect leaves the original host (HF hands
    out presigned CDN URLs that reject a foreign auth header)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            old = urllib.parse.urlsplit(req.full_url).netloc
            if urllib.parse.urlsplit(newurl).netloc != old:
                new.headers.pop("Authorization", None)
        return new


_OPENER = urllib.request.build_opener(_AuthStripRedirect)


def hub_headers(url: str) -> dict:
    """Auth/UA headers, via huggingface_hub when available."""
    headers = {"User-Agent": USER_AGENT}
    try:  # pragma: no cover - depends on the environment
        from huggingface_hub.utils import build_hf_headers

        headers.update({k: v for k, v in build_hf_headers().items() if v})
        headers["User-Agent"] = USER_AGENT
    except Exception:  # noqa: BLE001 - hub absent or unconfigured
        tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if tok:
            headers["Authorization"] = f"Bearer {tok}"
    if not url.startswith(DEFAULT_ENDPOINT):
        headers.pop("Authorization", None)
    return headers


def hub_url(repo: str, revision: str, filename: str) -> str:
    """Resolve a file URL, via huggingface_hub when available."""
    try:  # pragma: no cover - depends on the environment
        from huggingface_hub import hf_hub_url

        return hf_hub_url(repo_id=repo, filename=filename, revision=revision)
    except Exception:  # noqa: BLE001 - hub absent
        rev = urllib.parse.quote(revision, safe="")
        return f"{DEFAULT_ENDPOINT}/{repo}/resolve/{rev}/{filename}"


def http_get_range(url: str, start: int, end: int, timeout: float = 600.0):
    """GET bytes [start, end) of url -> (data, meta).  Module-level so tests
    can monkeypatch every byte of remote IO."""
    req = urllib.request.Request(url, headers={
        **hub_headers(url), "Range": f"bytes={start}-{end - 1}"})
    with _OPENER.open(req, timeout=timeout) as r:
        data = r.read()
        meta = {"commit": r.headers.get("X-Repo-Commit")}
        cr = r.headers.get("Content-Range", "")
        if "/" in cr and cr.rsplit("/", 1)[1].isdigit():
            meta["total"] = int(cr.rsplit("/", 1)[1])
    if len(data) != end - start:
        raise IOError(f"range {start}-{end} of {url}: got {len(data)} bytes")
    return data, meta


def http_get_full(url: str, timeout: float = 600.0):
    """GET a whole (small) file -> (data, meta)."""
    req = urllib.request.Request(url, headers=hub_headers(url))
    with _OPENER.open(req, timeout=timeout) as r:
        return r.read(), {"commit": r.headers.get("X-Repo-Commit")}


class Transport:
    """Paced, retrying reader; counts requests and bytes (fetched vs used)."""

    RETRYABLE = {408, 425, 429, 500, 502, 503, 504}

    def __init__(self, pace: float = 0.1, retries: int = 6):
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
            except Exception as e:  # noqa: BLE001 - classified here
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
        data, meta = self._retrying(lambda: http_get_range(url, start, end))
        self.requests += 1
        self.bytes += len(data)
        return data, meta

    def get_full(self, url: str):
        data, meta = self._retrying(lambda: http_get_full(url))
        self.requests += 1
        self.bytes += len(data)
        return data, meta

    def stats(self) -> dict:
        return {"requests": self.requests, "bytes": self.bytes}


# ------------------------------------------------------------------ sources

def slugify(repo: str, revision: str, subdir: str = "") -> str:
    slug = f"{repo.replace('/', '__')}@{revision[:12]}"
    return f"{slug}:{subdir.replace('/', '__')}" if subdir else slug


class Source:
    """One pinned artifact repo, reached only through ranged reads.

    Small documents (manifest, indexes, attestations, release) are cached on
    disk; segment headers are cached as JSON so a resumed run re-plans
    without touching the network.
    """

    def __init__(self, repo: str, revision: str, transport: Transport,
                 cache_root: Path, *, subdir: str = "", order: int = 0):
        self.repo = repo
        self.revision = revision or "main"
        self.pinned = bool(revision)
        self.subdir = subdir
        self.t = transport
        self.order = order
        self.slug = slugify(repo, self.revision, subdir)
        self.cache = cache_root / self.slug
        self.cache.mkdir(parents=True, exist_ok=True)
        self.resolved_commit: str | None = None
        self._small: dict[str, bytes | None] = {}
        self._headers: dict[str, tuple[dict, int]] = {}
        self._header_bytes: dict[str, bytes] = {}
        self._attestations: dict[str, dict] = {}
        self._header_auth: dict[str, dict] = {}
        self._verified_pieces: dict[tuple[str, int, int], Path] = {}
        self.release: dict | None = None
        self.manifest: dict = {}

    # -- naming ------------------------------------------------------------

    @classmethod
    def parse(cls, spec: str, transport: Transport, cache_root: Path,
              order: int = 0) -> "Source":
        repo, marker, revision_path = spec.partition("@")
        if not repo:
            raise SystemExit(f"--source {spec!r}: empty repo id")
        revision, separator, subdir = revision_path.partition(":")
        if separator:
            if not revision or not subdir:
                raise SystemExit(
                    f"--source {spec!r}: nested sources require "
                    "REPO@REV:relative/path")
            parts = subdir.split("/")
            if any(part in ("", ".", "..") for part in parts):
                raise SystemExit(
                    f"--source {spec!r}: source path must be a clean relative path")
        if not marker:
            revision = ""
        return cls(repo, revision, transport, cache_root, subdir=subdir, order=order)

    def __str__(self) -> str:
        source = f"{self.repo}@{self.revision}"
        return f"{source}:{self.subdir}" if self.subdir else source

    def url(self, name: str) -> str:
        path = f"{self.subdir}/{name}" if self.subdir else name
        return hub_url(self.repo, self.revision, path)

    @staticmethod
    def index_name(k: int) -> str:
        return f"index-k{k}.json"

    @staticmethod
    def segment_name(layer: int, k: int) -> str:
        return f"layer-{layer:03d}.k{k}.safetensors"

    @staticmethod
    def attestation_name(layer: int, k: int) -> str:
        return f"attestations/layer-{layer:03d}.k{k}.jsonl"

    # -- small documents ---------------------------------------------------

    def small_file(self, name: str) -> bytes | None:
        """Fetch (and cache) a whole small file.  None when absent."""
        if name in self._small:
            return self._small[name]
        p = self.cache / name.replace("/", "__")
        if p.exists():
            data = p.read_bytes()
        else:
            try:
                data, meta = self.t.get_full(self.url(name))
            except Exception as e:  # noqa: BLE001 - optional documents
                code = getattr(e, "code", None)
                if code not in (401, 403, 404) and not isinstance(e, urllib.error.HTTPError):
                    raise
                print(f"  {self}: {name} unavailable ({code or e})", flush=True)
                self._small[name] = None
                return None
            self.resolved_commit = self.resolved_commit or meta.get("commit")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        self._small[name] = data
        return data

    def small_json(self, name: str):
        data = self.small_file(name)
        return json.loads(data) if data else None

    def load_manifest(self) -> dict:
        self.manifest = self.small_json("fq-manifest.json") or {}
        return self.manifest

    def manifest_sha256(self) -> str | None:
        """Digest of the exact manifest bytes that define this source family."""
        raw = self._small.get("fq-manifest.json")
        return hashlib.sha256(raw).hexdigest() if raw is not None else None

    def load_release(self, verifier: fq_trust.Verifier) -> dict | None:
        """Fetch and verify fq-release.json when the source publishes one.

        Returns the verified payload, or None.  A present-but-unverifiable
        release manifest is fatal: it is the strongest claim the repo makes,
        and a broken one means the repo is not what it says it is.
        """
        raw = self.small_file("fq-release.json")
        if raw is None:
            return None
        import fq_release

        self.release = fq_release.verify_release(
            json.loads(raw), verifier, where=f"{self}:fq-release.json")
        print(f"  {self}: fq-release/1 verified — "
              f"{self.release['counts']['files']} files under one signature",
              flush=True)
        return self.release

    def release_entry(self, name: str) -> dict | None:
        return ((self.release or {}).get("files") or {}).get(name)

    def cross_check_fragment(self, name: str, frag: dict) -> bool:
        """The signed release and the signed attestation must agree.

        Both are signatures over the same publisher's claims about the same
        file; when they disagree, one of them is a rollback or a swap, and
        preferring either would be a guess.  Returns True when the release
        actually covered this fragment.
        """
        want = self.release_entry(name)
        if not want:
            if self.release:
                raise TrustError(
                    f"{self}: {name} is not listed in the signed release "
                    f"manifest — refusing to fetch an uncovered fragment")
            return False
        got = frag.get("sha256")
        if got and got != want["sha256"]:
            raise TrustError(
                f"{self}: the signed attestation says {name} is "
                f"{got[:16]}… but the signed release says "
                f"{want['sha256'][:16]}… — refusing while the publisher's "
                f"own signatures disagree")
        if frag.get("size") and want.get("size") and frag["size"] != want["size"]:
            raise TrustError(
                f"{self}: attested size {frag['size']} != release size "
                f"{want['size']} for {name}")
        return True

    def check_against_release(self, name: str, data: bytes) -> None:
        """Digest-check a small document against the signed release list."""
        if not self.release:
            return
        want = self.release["files"].get(name)
        if want is None:
            raise TrustError(
                f"{self}: {name} is not listed in the signed release manifest "
                f"— refusing to use an uncovered document")
        got = hashlib.sha256(data).hexdigest()
        if got != want["sha256"]:
            raise TrustError(
                f"{self}: {name} sha256 {got[:16]}… does not match the signed "
                f"release ({want['sha256'][:16]}…)")

    def index(self, k: int) -> dict | None:
        name = self.index_name(k)
        raw = self.small_file(name)
        if raw is None:
            return None
        self.check_against_release(name, raw)
        return json.loads(raw)

    def attestation(self, layer: int, k: int, verifier: fq_trust.Verifier,
                    segment_file: str) -> dict:
        """Verified, merged attestation payload for one segment file.

        The file is JSON Lines and every line is checked on its own: a line
        this consumer cannot trust (a third party's countersignature, an
        older key, a corrupted line) is skipped with its reason recorded,
        and only lines a trusted key signed for THIS fragment contribute.
        Raises TrustError when the file is missing or when nothing in it
        verifies — a fetch that cannot be checked does not happen.
        """
        cache_key = f"{layer}.{k}"
        if cache_key in self._attestations:
            return self._attestations[cache_key]
        name = self.attestation_name(layer, k)
        raw = self.small_file(name)
        if raw is None:
            raise TrustError(
                f"{self}: no {name} — cannot verify fetched bytes for layer "
                f"{layer} K{k}; refusing (use --insecure-skip-signatures only "
                f"for offline fixtures, and even then attestations carry the "
                f"per-expert digests)")
        self.check_against_release(name, raw)
        merged: dict = {"expert_sha256": {}, "fragment": None, "lines": 0,
                        "rejected_lines": []}
        for n, line in enumerate(raw.decode().splitlines(), 1):
            if not line.strip():
                continue
            try:
                payload = verifier.verify_envelope(
                    json.loads(line), where=f"{self}:{name}:{n}")
            except (TrustError, json.JSONDecodeError, TypeError) as e:
                merged["rejected_lines"].append(f"line {n}: {e}")
                continue
            frag = payload.get("fragment") or {}
            if frag.get("file") != segment_file:
                continue
            merged["fragment"] = frag
            merged["lines"] += 1
            for eid, digest in (payload.get("expert_sha256") or {}).items():
                merged["expert_sha256"][str(eid)] = digest
            merged.setdefault("predicate", payload.get("predicate"))
            merged.setdefault("materials", payload.get("materials"))
            merged.setdefault("layout", payload.get("layout"))
        if not merged["lines"]:
            detail = ("; ".join(merged["rejected_lines"])
                      or f"no line names {segment_file}")
            raise TrustError(
                f"{self}: {name} has no trusted attestation line for "
                f"{segment_file} ({detail})")
        self.cross_check_fragment(segment_file, merged["fragment"] or {})
        self._attestations[cache_key] = merged
        return merged

    # -- segment headers ---------------------------------------------------

    def segment_header(self, layer: int, k: int) -> tuple[dict, int]:
        """(header, body_offset) of a remote segment, by ranged read.

        UNAUTHENTICATED: these are bytes the server chose to send.  Good
        enough to draft a plan and price it; authenticate_header() must
        prove them before anything is fetched or written.
        """
        name = self.segment_name(layer, k)
        if name in self._headers:
            return self._headers[name]
        cache = self.cache / f"hdr-{name}.json"
        obj = None
        if cache.exists():
            try:
                candidate = json.loads(cache.read_text())
                raw = base64.b64decode(candidate["header_bytes"], validate=True)
                body_offset = candidate["body_offset"]
                if (not isinstance(body_offset, int) or len(raw) != body_offset
                        or len(raw) < 8
                        or 8 + struct.unpack("<Q", raw[:8])[0] != body_offset):
                    raise ValueError("header length")
                header = json.loads(raw[8:])
                if not isinstance(header, dict) or header != candidate["header"]:
                    raise ValueError("header representation")
                obj = candidate
                out = (header, body_offset)
                self._header_bytes[name] = raw
                if isinstance(obj.get("authentication"), dict):
                    self._header_auth[name] = obj["authentication"]
            except (binascii.Error, KeyError, TypeError, ValueError,
                    json.JSONDecodeError, struct.error):
                # A cache is an optimization, never an authority.  In
                # particular, old cache entries did not retain the exact
                # bytes their authentication record claims to prove.
                obj = None
        if obj is None:
            url = self.url(name)
            d, meta = self.t.get_range(url, 0, 8)
            hlen = struct.unpack("<Q", d)[0]
            if not 0 < hlen < (1 << 31):
                raise IOError(f"{self}:{name}: implausible header length {hlen}")
            hj, meta = self.t.get_range(url, 8, 8 + hlen)
            self.resolved_commit = self.resolved_commit or meta.get("commit")
            header = json.loads(hj)
            out = (header, 8 + hlen)
            self._write_header_cache(name, header, out[1], None,
                                     bytes(d) + bytes(hj))
        self._headers[name] = out
        return out

    def _write_header_cache(self, name: str, header: dict, body_offset: int,
                            auth: dict | None, raw: bytes | None = None) -> None:
        raw = raw if raw is not None else self._header_bytes.get(name)
        if raw is None or len(raw) != body_offset:
            raise ValueError(f"{self}:{name}: refusing incomplete header cache")
        cache = self.cache / f"hdr-{name}.json"
        tmp = cache.with_name(f"{cache.name}.tmp-{os.getpid()}-{time.time_ns()}")
        tmp.write_text(json.dumps({
            "header": header,
            "header_bytes": base64.b64encode(raw).decode(),
            "body_offset": body_offset,
            "authentication": auth,
        }, sort_keys=True))
        os.replace(tmp, cache)
        self._headers[name] = (header, body_offset)
        self._header_bytes[name] = raw
        if auth is not None:
            self._header_auth[name] = auth

    def _cached_authentication_fits(self, name: str, prov: dict, mode: str,
                                    frag: dict, covered: bool) -> bool:
        """A persisted proof is useful only when it binds these exact bytes."""
        # A whole-file hash does not authenticate a separately cached prefix:
        # an attacker can replace both prefix and its self-reported digest.
        # Re-hash the pinned source on every process unless the publisher
        # supplied a signed header digest.
        if prov.get("method") == AUTH_FULL_FRAGMENT:
            return False
        try:
            header, body_offset = self._headers[name]
            raw = self._header_bytes[name]
            raw_header = json.loads(raw[8:])
            raw_digest = hashlib.sha256(raw).hexdigest()
            valid = (
                bool(prov.get("authenticated"))
                and len(raw) == body_offset
                and 8 + struct.unpack("<Q", raw[:8])[0] == body_offset
                and raw_header == header
                and prov.get("body_offset") == body_offset
                and prov.get("header_sha256") == raw_digest
                and bool(prov.get("release_manifest")) == covered
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError,
                struct.error):
            return False
        return valid and self._auth_fits(prov, mode, frag)

    def authenticate_header(self, layer: int, k: int, att: dict,
                            mode: str, *, retain=None) -> dict:
        """Prove a segment's safetensors header before planning from it.

        The header carries tensor names, dtypes, shapes and offsets — the
        meaning of the bytes.  Hashing payload spans afterwards cannot catch
        a rewritten header, so the header itself has to trace back to a
        signature.  Returns a provenance record describing how (see
        AUTH_* constants); raises TrustError when it cannot be proven under
        the requested mode.
        """
        name = self.segment_name(layer, k)
        frag = att.get("fragment") or {}
        covered = bool(self.release_entry(name))
        if mode == HEADER_UNSAFE:
            header, body_offset = self.segment_header(layer, k)
            prov = {"method": AUTH_NONE, "authenticated": False,
                    "body_offset": body_offset, "release_manifest": covered,
                    "note": "the plan came from an unverified ranged read"}
            self._header_auth[name] = prov
            return prov
        self.segment_header(layer, k)
        cached = self._header_auth.get(name)
        if (cached and cached.get("authenticated")
                and self._cached_authentication_fits(
                    name, cached, mode, frag, covered)):
            return cached
        if frag.get("header_sha256") and mode != HEADER_FULL:
            prov = self._auth_by_header_digest(name, frag)
        elif mode == HEADER_ATTESTED:
            raise TrustError(
                f"{self}: {name} has no signed fragment.header_sha256, so its "
                f"header cannot be authenticated cheaply, and --header-trust "
                f"attested refuses to guess. Ask the publisher to attest the "
                f"header digest, or re-run with --header-trust full to verify "
                f"the whole fragment ({frag.get('size') or '?'} bytes) — "
                f"--header-trust unsafe plans from the unverified header and "
                f"says so in the output tree.")
        else:
            prov = self._auth_by_full_fragment(name, frag, retain=retain)
        # This is deliberately before the only proof-bearing cache write:
        # a later process must not observe a half-recorded proof that lacks
        # the release coverage used in derived provenance.
        prov["release_manifest"] = covered
        header, body_offset = self._headers[name]
        self._write_header_cache(name, header, body_offset, prov)
        return prov

    @staticmethod
    def _auth_fits(prov: dict, mode: str, frag: dict) -> bool:
        """Is a cached authentication strong enough for this run?"""
        if mode == HEADER_ATTESTED:
            return prov.get("method") == AUTH_HEADER_DIGEST
        if mode == HEADER_FULL:
            return (prov.get("method") == AUTH_FULL_FRAGMENT
                    and prov.get("fragment_sha256") == frag.get("sha256"))
        if prov.get("method") == AUTH_FULL_FRAGMENT:
            return prov.get("fragment_sha256") == frag.get("sha256")
        return (prov.get("method") == AUTH_HEADER_DIGEST
                and prov.get("header_sha256") == frag.get("header_sha256"))

    def _auth_by_header_digest(self, name: str, frag: dict) -> dict:
        """Cheap path: the publisher signed a digest of the header bytes."""
        url = self.url(name)
        d, meta = self.t.get_range(url, 0, 8)
        hlen = struct.unpack("<Q", d)[0]
        if not 0 < hlen < (1 << 31):
            raise IOError(f"{self}:{name}: implausible header length {hlen}")
        hj, meta = self.t.get_range(url, 8, 8 + hlen)
        self.resolved_commit = self.resolved_commit or meta.get("commit")
        raw = bytes(d) + bytes(hj)
        got = hashlib.sha256(raw).hexdigest()
        want = frag["header_sha256"]
        if got != want:
            raise TrustError(
                f"{self}: {name} header hashes to {got[:16]}… but the signed "
                f"attestation says {want[:16]}… — the header this plan would "
                f"use is not the header the publisher signed")
        body_offset = 8 + hlen
        if frag.get("body_offset") not in (None, body_offset):
            raise TrustError(
                f"{self}: {name} body starts at {body_offset}, the signed "
                f"attestation says {frag['body_offset']}")
        header = json.loads(hj)
        self._headers[name] = (header, body_offset)
        self._header_bytes[name] = raw
        return {"method": AUTH_HEADER_DIGEST, "authenticated": True,
                "header_sha256": got, "body_offset": body_offset,
                "bytes_read": body_offset}

    def _auth_by_full_fragment(self, name: str, frag: dict, *, retain=None) -> dict:
        """Verify a whole fragment, retaining selected verified bytes on disk."""
        size, want = frag.get("size"), frag.get("sha256")
        if not want:
            raise TrustError(
                f"{self}: {name} has no attested fragment sha256; nothing "
                f"authenticates this fragment at all")
        if not isinstance(size, int) or size <= 0:
            raise TrustError(
                f"{self}: {name} has neither a signed header digest nor a "
                f"signed size, so the fragment cannot be verified in full. "
                f"Ask the publisher for fragment.header_sha256.")
        retained = {}
        try:
            if retain:
                spool = self.cache / ".verified-pieces"
                spool.mkdir(exist_ok=True)
                for piece in retain:
                    path = spool / (
                        f"{name}.{piece.remote_start}-{piece.remote_end}."
                        f"{os.getpid()}-{time.time_ns()}")
                    path.write_bytes(b"")
                    retained[piece] = path
            url = self.url(name)
            digest = hashlib.sha256()
            head = bytearray()
            need = 8
            off = 0
            while off < size:
                end = min(off + FULL_VERIFY_CHUNK, size)
                data, meta = self.t.get_range(url, off, end)
                if len(data) != end - off:
                    raise IOError(f"{self}:{name}: short read at {off}")
                self.resolved_commit = self.resolved_commit or meta.get("commit")
                digest.update(data)
                if off == 0:
                    need = 8 + struct.unpack("<Q", data[:8])[0]
                    if not 8 < need <= size:
                        raise IOError(
                            f"{self}:{name}: implausible header length {need - 8}")
                if len(head) < need:
                    head += data[:need - len(head)]
                for piece, path in retained.items():
                    start = max(off, piece.remote_start)
                    stop = min(end, piece.remote_end)
                    if start < stop:
                        with open(path, "ab") as f:
                            f.write(data[start - off: stop - off])
                off = end
            got = digest.hexdigest()
            if got != want:
                raise TrustError(
                    f"{self}: {name} hashes to {got[:16]}… over {size} bytes but "
                    f"the signed attestation says {want[:16]}… — refusing to plan "
                    f"from a fragment that is not the one that was signed")
            raw = bytes(head[:need])
            header = json.loads(raw[8:])
            self._headers[name] = (header, need)
            self._header_bytes[name] = raw
            for piece, path in retained.items():
                if path.stat().st_size == piece.size:
                    self._verified_pieces[(name, piece.remote_start,
                                           piece.remote_end)] = path
                else:
                    path.unlink(missing_ok=True)
            return {"method": AUTH_FULL_FRAGMENT, "authenticated": True,
                    "fragment_sha256": got, "body_offset": need,
                    "header_sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes_read": size}
        except BaseException:
            for path in retained.values():
                path.unlink(missing_ok=True)
            raise

    def take_verified_piece(self, name: str, piece) -> bytes | None:
        """Consume a current-run full-auth spool file, never a persisted cache."""
        path = self._verified_pieces.pop(
            (name, piece.remote_start, piece.remote_end), None)
        if path is None:
            return None
        try:
            data = path.read_bytes()
        finally:
            path.unlink(missing_ok=True)
        return data if len(data) == piece.size else None

    def discard_verified_pieces(self, name: str) -> None:
        """Remove unconsumed current-run full-auth spools for one fragment."""
        for key, path in list(self._verified_pieces.items()):
            if key[0] == name:
                self._verified_pieces.pop(key)
                path.unlink(missing_ok=True)


# ------------------------------------------------------------------- policy

def parse_int_set(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def load_policy(path: Path, layers: str | None) -> dict[int, dict[int, int]]:
    """fq-policy/2 -> {layer: {expert: k}}."""
    policy = json.loads(Path(path).read_text())
    schema = policy.get("schema")
    if schema != POLICY_SCHEMA:
        print(f"warning: policy schema is {schema!r}, expected {POLICY_SCHEMA!r}",
              file=sys.stderr)
    bpe = policy.get("bits_per_expert")
    if not isinstance(bpe, dict) or not bpe:
        raise SystemExit(f"{path}: no bits_per_expert map")
    wanted = parse_int_set(layers) if layers else None
    out: dict[int, dict[int, int]] = {}
    for layer_s, bits in bpe.items():
        layer = int(layer_s)
        if wanted is not None and layer not in wanted:
            continue
        if not isinstance(bits, list):
            raise SystemExit(f"{path}: bits_per_expert[{layer_s}] is not a list")
        out[layer] = {e: int(k) for e, k in enumerate(bits)}
    if not out:
        raise SystemExit(f"{path}: no layers selected")
    return out


def load_fetch_binding(path: Path) -> dict | None:
    """Load a recipe's complete, immutable provider/content binding."""
    binding = json.loads(Path(path).read_text()).get("fetch_binding")
    if binding is None:
        return None
    if not isinstance(binding, dict) or binding.get("schema") != FETCH_BINDING_SCHEMA:
        raise TrustError(
            f"{path}: fetch_binding must declare schema {FETCH_BINDING_SCHEMA!r}")
    providers = binding.get("providers")
    content = binding.get("content")
    if not isinstance(providers, dict) or providers.get("schema") != SELECT_SCHEMA:
        raise TrustError(
            f"{path}: fetch_binding.providers must be an {SELECT_SCHEMA} map")
    if not isinstance(content, dict):
        raise TrustError(f"{path}: fetch_binding.content must be an object")
    if not isinstance(content.get("manifests"), dict):
        raise TrustError(f"{path}: fetch_binding.content.manifests must be an object")
    if not isinstance(content.get("experts"), dict):
        raise TrustError(f"{path}: fetch_binding.content.experts must be an object")
    return binding


def binding_digest(binding: dict, layer: int, expert: int) -> str | None:
    return ((binding["content"]["experts"].get(str(layer)) or {})
            .get(str(expert)))


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def load_select(path: Path | None) -> dict:
    """fq-select/1 provider map: {"default": alias, "layers": {L: alias},
    "experts": {L: {E: alias}}}.  Aliases are repo ids or repo@rev."""
    if path is None:
        return {}
    sel = json.loads(Path(path).read_text())
    if sel.get("schema") not in (None, SELECT_SCHEMA):
        print(f"warning: select map schema is {sel.get('schema')!r}, "
              f"expected {SELECT_SCHEMA!r}", file=sys.stderr)
    return sel


def select_preference(sel: dict, layer: int, expert: int) -> str | None:
    experts = (sel.get("experts") or {}).get(str(layer)) or {}
    if str(expert) in experts:
        return experts[str(expert)]
    layers = sel.get("layers") or {}
    if str(layer) in layers:
        return layers[str(layer)]
    return sel.get("default")


def source_matches(source: Source, alias: str) -> bool:
    return alias in (source.repo, str(source), source.slug)


def validate_fetch_binding(binding: dict, policy: dict[int, dict[int, int]],
                           sources: list[Source]) -> None:
    """Prove every selected provider and manifest before payload planning."""
    providers = binding["providers"]
    manifests = binding["content"]["manifests"]
    source_by_name = {str(source): source for source in sources}
    selected: set[str] = set()
    for layer, experts in policy.items():
        for expert in experts:
            alias = select_preference(providers, layer, expert)
            digest = binding_digest(binding, layer, expert)
            if not isinstance(alias, str):
                raise TrustError(
                    f"fetch binding has no provider for layer {layer} expert {expert}")
            if not _is_sha256(digest):
                raise TrustError(
                    f"fetch binding has no valid digest for layer {layer} expert {expert}")
            source = source_by_name.get(alias)
            if source is None:
                raise TrustError(
                    f"fetch binding requires source {alias!r}, which was not supplied")
            if (not source.pinned or len(source.revision) != 40
                    or not all(c in "0123456789abcdefABCDEF"
                               for c in source.revision)):
                raise TrustError(
                    f"fetch binding source {alias!r} is not pinned to a commit")
            if source.resolved_commit and source.resolved_commit != source.revision:
                raise TrustError(
                    f"fetch binding source {alias!r} resolved to "
                    f"{source.resolved_commit}, not its pinned commit")
            selected.add(alias)
    for alias in selected:
        expected = manifests.get(alias)
        source = source_by_name[alias]
        actual = source.manifest_sha256()
        if not _is_sha256(expected):
            raise TrustError(
                f"fetch binding has no valid manifest digest for {alias!r}")
        if actual is None:
            raise TrustError(
                f"fetch binding source {alias!r} has no fq-manifest.json")
        if actual != expected:
            raise TrustError(
                f"fetch binding manifest for {alias!r} hashes to {actual[:16]}… "
                f"not {expected[:16]}…")


# ------------------------------------------------------------------ planning

class Piece:
    """One expert's bytes: where they are remotely, where they go locally."""

    __slots__ = ("expert", "source", "remote_start", "remote_end",
                 "local_off", "sha256", "names")

    def __init__(self, expert, source, remote_start, remote_end, local_off,
                 sha256, names):
        self.expert = expert
        self.source = source
        self.remote_start = remote_start
        self.remote_end = remote_end
        self.local_off = local_off
        self.sha256 = sha256
        self.names = names

    @property
    def size(self) -> int:
        return self.remote_end - self.remote_start


class FilePlan:
    """Everything needed to build one local layer-LLL.kK.safetensors."""

    def __init__(self, layer: int, k: int):
        self.layer = layer
        self.k = k
        self.name = Source.segment_name(layer, k)
        self.pieces: list[Piece] = []
        self.header: dict = {}
        self.body_size = 0
        self.body_offset = 0
        self.meta: dict = {}
        self.sources_used: dict[str, str] = {}
        # what the plan was drafted from, so it can be re-derived once the
        # remote headers have actually been authenticated
        self.chosen: list = []
        self.atts: dict[str, dict] = {}
        self.header_provenance: dict[str, dict] = {}

    def shape(self) -> tuple:
        """Everything about this plan that the remote header decides."""
        return (self.header, self.body_size,
                tuple((p.expert, str(p.source), p.remote_start, p.remote_end,
                       p.local_off, p.sha256, tuple(p.names))
                      for p in self.pieces))

    @property
    def bytes_needed(self) -> int:
        return sum(p.size for p in self.pieces)

    def digest(self) -> str:
        """Plan identity: changes whenever the expert set, their order, their
        provenance or their expected content changes, so a stale .part from a
        previous (different) recipe is never resumed into."""
        spec = json.dumps(
            [[p.expert, str(p.source), p.local_off, p.size, p.sha256]
             for p in self.pieces], sort_keys=True)
        return hashlib.sha256(spec.encode()).hexdigest()


def choose_source(candidates: list[tuple[Source, dict, dict]], layer: int,
                  expert: int, *, prefer_sha: set[str], select: dict,
                  required_select: bool = False):
    """Pick the provider for one expert.

    A recipe binding makes its provider selection mandatory; unlike an
    interactive preference, it never falls through to another available
    source.
    """
    if prefer_sha:
        for src, idx, att in candidates:
            if att["expert_sha256"].get(str(expert)) in prefer_sha:
                return src, idx, att
    alias = select_preference(select, layer, expert)
    if alias:
        for src, idx, att in candidates:
            matches = str(src) == alias if required_select else source_matches(src, alias)
            if matches:
                return src, idx, att
        if required_select:
            return None
    return candidates[0] if candidates else None


def plan_fetch(policy: dict[int, dict[int, int]], sources: list[Source],
               verifier: fq_trust.Verifier, *, prefer_sha: set[str],
               select: dict, binding: dict | None = None) -> tuple[list[FilePlan], list[str]]:
    """Resolve every (layer, expert, K) to a provider and a byte range."""
    plans: list[FilePlan] = []
    problems: list[str] = []
    indexes: dict[tuple[int, int], dict] = {}

    def index_for(src: Source, k: int):
        key = (id(src), k)
        if key not in indexes:
            indexes[key] = src.index(k) or {}
        return indexes[key]

    for layer in sorted(policy):
        by_k: dict[int, list[int]] = {}
        for expert, k in sorted(policy[layer].items()):
            by_k.setdefault(k, []).append(expert)
        for k in sorted(by_k):
            plan = FilePlan(layer, k)
            # Which sources carry this (layer, K) at all, with their verified
            # attestation — resolved once per file, then filtered per expert.
            cands: list[tuple[Source, dict, dict]] = []
            for src in sources:
                idx = index_for(src, k).get(str(layer))
                if not idx:
                    continue
                try:
                    att = src.attestation(layer, k, verifier, idx["file"])
                except TrustError as e:
                    problems.append(str(e))
                    continue
                cands.append((src, idx, att))
            chosen: list[tuple[int, Source, dict, str]] = []
            for expert in by_k[k]:
                have = [(s, i, a) for s, i, a in cands
                        if str(expert) in (i.get("experts") or {})]
                pick = choose_source(
                    have, layer, expert, prefer_sha=prefer_sha, select=select,
                    required_select=binding is not None)
                if pick is None:
                    problems.append(
                        f"layer {layer} K{k} expert {expert}: no source carries it")
                    continue
                src, idx, att = pick
                digest = att["expert_sha256"].get(str(expert))
                if not digest:
                    problems.append(
                        f"layer {layer} K{k} expert {expert}: {src} has the bytes "
                        f"but no attested digest — refusing to fetch unverifiable "
                        f"fragments")
                    continue
                expected = binding_digest(binding, layer, expert) if binding else None
                if expected is not None and digest != expected:
                    problems.append(
                        f"layer {layer} K{k} expert {expert}: {src} attests "
                        f"{digest[:16]}… but fetch binding requires {expected[:16]}…")
                    continue
                chosen.append((expert, src, idx, digest))
            if not chosen:
                continue
            _build_file_plan(plan, chosen, problems)
            if plan.pieces:
                plan.chosen = chosen
                plan.atts = {src.slug: att for src, _idx, att in cands}
                plans.append(plan)
    return plans, problems


def authenticate_plan(plan: FilePlan, mode: str) -> dict:
    """Prove every remote header this plan reads from, then re-derive it.

    Planning happens against an unauthenticated ranged header so the run can
    price itself before spending anything.  Nothing may be fetched or
    written from that draft: here each source's header is authenticated
    (signed header digest, or the whole fragment re-hashed against the
    signed file digest) and the plan is rebuilt from the proven bytes.  Any
    difference means the ranged read and the signed bytes disagree, which is
    exactly the substitution this check exists to catch.
    """
    provenance: dict[str, dict] = {}
    for src in sorted({p.source.slug: p.source for p in plan.pieces}.values(),
                      key=lambda s: s.slug):
        att = plan.atts.get(src.slug) or {}
        provenance[src.slug] = src.authenticate_header(
            plan.layer, plan.k, att, mode,
            retain=[p for p in plan.pieces if p.source is src])
    if mode == HEADER_UNSAFE:
        plan.header_provenance = provenance
        return provenance
    before = plan.shape()
    fresh = FilePlan(plan.layer, plan.k)
    problems: list[str] = []
    _build_file_plan(fresh, plan.chosen, problems)
    if problems:
        raise TrustError(
            f"{plan.name}: the authenticated header does not support this "
            f"plan ({'; '.join(problems)})")
    if fresh.shape() != before:
        raise TrustError(
            f"{plan.name}: the authenticated segment header differs from the "
            f"ranged header this plan was drafted from — tensor names, "
            f"dtypes, shapes or offsets were substituted between the two "
            f"reads; refusing")
    plan.header_provenance = provenance
    return provenance


def authentication_cost(plans: list[FilePlan], mode: str) -> dict:
    """What proving the plans' headers will cost, before it is spent."""
    methods: dict[str, int] = {}
    extra = 0
    for plan in plans:
        for src in {p.source.slug: p.source for p in plan.pieces}.values():
            frag = (plan.atts.get(src.slug) or {}).get("fragment") or {}
            name = Source.segment_name(plan.layer, plan.k)
            cached = src._cached_authentication_fits(
                name, src._header_auth.get(name) or {}, mode, frag,
                bool(src.release_entry(name)))
            if cached:
                methods["cached"] = methods.get("cached", 0) + 1
            elif mode == HEADER_UNSAFE:
                methods[AUTH_NONE] = methods.get(AUTH_NONE, 0) + 1
            elif frag.get("header_sha256") and mode != HEADER_FULL:
                methods[AUTH_HEADER_DIGEST] = methods.get(AUTH_HEADER_DIGEST, 0) + 1
            elif mode == HEADER_ATTESTED:
                methods["refused (no signed header digest)"] = methods.get(
                    "refused (no signed header digest)", 0) + 1
            else:
                methods[AUTH_FULL_FRAGMENT] = methods.get(AUTH_FULL_FRAGMENT, 0) + 1
                extra += int(frag.get("size") or 0)
    return {"mode": mode, "methods": methods, "extra_bytes": extra}


def _build_file_plan(plan: FilePlan, chosen, problems: list[str]) -> None:
    """Lay out the local segment file and attach one Piece per expert.

    Tensors are written in fq_repack canonical order (expert id, then
    projection/rank/component), which is what fq_assemble and every other
    tool expects, so a subset file is a well-formed fq-segment/1 with fewer
    experts — not a new format.
    """
    local_off = 0
    tensors: dict[str, dict] = {}
    pieces: list[Piece] = []
    meta_src = None
    sources_used: dict[str, str] = {}
    for expert, src, idx, digest in sorted(chosen, key=lambda c: c[0]):
        header, body_offset = src.segment_header(plan.layer, plan.k)
        meta_src = meta_src or (header.get("__metadata__") or {})
        names = sorted(
            (n for n in header
             if n != "__metadata__" and f".experts.{expert}." in n),
            key=expert_key)
        if not names:
            problems.append(
                f"layer {plan.layer} K{plan.k} expert {expert}: {src} segment "
                f"header has no tensors for that expert")
            continue
        lo, hi = idx["experts"][str(expert)]
        span = sum(header[n]["data_offsets"][1] - header[n]["data_offsets"][0]
                   for n in names)
        if span != hi - lo:
            problems.append(
                f"layer {plan.layer} K{plan.k} expert {expert}: {src} index span "
                f"{hi - lo} != header tensor span {span} — inconsistent source")
            continue
        first = header[names[0]]["data_offsets"][0]
        if first != lo:
            problems.append(
                f"layer {plan.layer} K{plan.k} expert {expert}: {src} index "
                f"start {lo} != first tensor offset {first}")
            continue
        start = local_off
        for n in names:
            a, b = header[n]["data_offsets"]
            tensors[n] = {"dtype": header[n]["dtype"], "shape": header[n]["shape"],
                          "data_offsets": [local_off, local_off + (b - a)]}
            local_off += b - a
        pieces.append(Piece(expert, src, body_offset + lo, body_offset + hi,
                            start, digest, names))
        sources_used[str(expert)] = str(src)
    if not pieces:
        return
    meta = {k: str(v) for k, v in (meta_src or {}).items()}
    meta.update({
        "fq_schema": SEGMENT_SCHEMA,
        "k": str(plan.k),
        "layer": str(plan.layer),
        "num_experts": str(len(pieces)),
        "fq_fetch": "subset",
        "fq_fetch_sources": ",".join(sorted({str(p.source) for p in pieces})),
    })
    plan.meta = meta
    plan.header = tensors
    plan.body_size = local_off
    plan.pieces = pieces
    plan.sources_used = sources_used


# ------------------------------------------------------------------ writing

class SegmentWriter:
    """Preallocated subset segment: header first, expert spans by pwrite."""

    def __init__(self, path: Path, meta: dict, tensors: dict, body_size: int):
        header = {"__metadata__": meta, **tensors}
        hj = json.dumps(header, separators=(",", ":")).encode()
        hj += b" " * ((8 - len(hj) % 8) % 8)
        self.path = path
        self.tmp = path.with_name(path.name + ".part")
        self.body_off = 8 + len(hj)
        self.body_size = body_size
        prefix = struct.pack("<Q", len(hj)) + hj
        if self.tmp.exists() and self.tmp.stat().st_size == self.body_off + body_size:
            with open(self.tmp, "r+b") as f:  # keep resumable payload bytes
                f.write(prefix)
        else:
            with open(self.tmp, "wb") as f:
                f.write(prefix)
                f.truncate(self.body_off + body_size)
        self.f = open(self.tmp, "r+b")

    def pwrite(self, off: int, data: bytes) -> None:
        os.pwrite(self.f.fileno(), data, self.body_off + off)

    def read(self, off: int, length: int) -> bytes:
        return os.pread(self.f.fileno(), length, self.body_off + off)

    def finalize(self) -> tuple[str, int]:
        self.f.close()
        h = hashlib.sha256()
        with open(self.tmp, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                h.update(block)
        os.replace(self.tmp, self.path)
        return h.hexdigest(), self.path.stat().st_size

    def abandon(self) -> None:
        try:
            self.f.close()
        except Exception:  # noqa: BLE001
            pass


def coalesce(pieces: list[Piece], max_chunk: int, max_gap: int) -> list[list]:
    """Group same-source pieces into ranged GETs.

    Adjacent (and near-adjacent, up to max_gap) spans become one request; the
    filler bytes across a gap are fetched and discarded, which is cheaper
    than a second round trip.  Chunks never exceed max_chunk.
    """
    chunks: list[list] = []
    for p in sorted(pieces, key=lambda x: x.remote_start):
        if chunks:
            cur = chunks[-1]
            gap = p.remote_start - cur[1]
            if 0 <= gap <= max_gap and (p.remote_end - cur[0]) <= max_chunk:
                cur[1] = max(cur[1], p.remote_end)
                cur[2].append(p)
                continue
        chunks.append([p.remote_start, p.remote_end, [p]])
    return chunks


# ------------------------------------------------------------------- output

def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.2f} TB"


def counterfactual(sources: list[Source], plans: list[FilePlan]) -> dict:
    """What the same recipe costs without ranged reads.

    whole_files: downloading every segment file the recipe touches.
    whole_repo:  downloading every segment listed in every index the sources
                 publish (the `hf download <repo>` the quickstart used to
                 recommend), summed over the sources actually consulted.
    """
    touched: dict[tuple[str, str], int] = {}
    for plan in plans:
        for p in plan.pieces:
            idx = p.source.index(plan.k) or {}
            entry = idx.get(str(plan.layer))
            if entry:
                touched[(p.source.slug, entry["file"])] = entry["size"]
    whole_repo = 0
    seen_repo = set()
    for src in sources:
        ks = src.manifest.get("k_variants") or []
        names = src.manifest.get("tensor_indexes") or {}
        idx_names = set(names.values()) | {src.index_name(int(k)) for k in ks}
        if not idx_names and src.manifest.get("tensor_index"):
            idx_names = {src.manifest["tensor_index"]}
        for name in sorted(idx_names):
            raw = src.small_file(name)
            if not raw:
                continue
            for entry in json.loads(raw).values():
                key = (src.slug, entry.get("file"))
                if key in seen_repo:
                    continue
                seen_repo.add(key)
                whole_repo += int(entry.get("size") or 0)
    return {"whole_files": sum(touched.values()), "whole_repo": whole_repo,
            "files_touched": len(touched)}


def write_outputs(out: Path, plans: list[FilePlan], entries: dict,
                  sources: list[Source], verifier: fq_trust.Verifier,
                  stats: dict, policy_path: Path,
                  local_signer: str | None = None) -> None:
    """Local index-kK.json, fq-manifest.json and the provenance report."""
    ks = sorted({p.k for p in plans})
    per_k: dict[int, dict] = {k: {} for k in ks}
    provenance: dict[str, dict] = {}
    for plan in plans:
        entry = entries.get((plan.layer, plan.k))
        if entry:
            per_k[plan.k][str(plan.layer)] = entry
        provenance.setdefault(str(plan.layer), {})[f"k{plan.k}"] = {
            str(p.expert): {"source": str(p.source), "sha256": p.sha256}
            for p in plan.pieces}
    index_names = {}
    for k, index in per_k.items():
        if not index:
            continue
        name = Source.index_name(k)
        (out / name).write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
        index_names[str(k)] = name

    base = sources[0].manifest if sources else {}
    layers = sorted({p.layer for p in plans})
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "kind": "fetched-subset",
        "base_model": base.get("base_model"),
        "revision": base.get("revision"),
        "predicate": base.get("predicate", "repack-of"),
        "layout": base.get("layout", "rank_sliced_tp4"),
        "hessian_id": base.get("hessian_id"),
        "k_variants": ks,
        "moe_layers": [min(layers), max(layers)] if layers else [],
        "num_experts": max((len(v.get("experts", {}))
                            for idx in per_k.values() for v in idx.values()),
                           default=0),
        "sources": [str(s) for s in sources],
        "tensor_indexes": index_names,
        # The key that signed THIS tree's attestations.  For a fetched subset
        # that is the local fq_fetch key (the subset files are new files);
        # the upstream key whose signatures were checked while fetching is
        # recorded separately, because conflating them is the mistake this
        # whole trust model exists to avoid.
        "signer_pubkey": local_signer,
        "upstream_signer": verifier.fingerprint,
        "upstream_trust_rung": verifier.rung,
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if len(index_names) == 1:
        manifest["tensor_index"] = next(iter(index_names.values()))
    (out / "fq-manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n")

    headers = {}
    for plan in plans:
        for slug, prov in (plan.header_provenance or {}).items():
            headers.setdefault(plan.name, {})[slug] = {
                "method": prov.get("method"),
                "authenticated": bool(prov.get("authenticated")),
                "header_sha256": prov.get("header_sha256"),
                "signed_release_manifest": bool(prov.get("release_manifest")),
            }
    report = {
        "schema": REPORT_SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policy": str(policy_path),
        "trust": {"rung": verifier.rung, "signer": verifier.fingerprint,
                  "key_id": verifier.key_id,
                  "signatures_verified": verifier.checked,
                  "local_signer": local_signer,
                  "plan_authenticated": all(
                      h["authenticated"]
                      for per_file in headers.values()
                      for h in per_file.values()) if headers else False},
        "header_authentication": headers,
        "sources": [{"repo": s.repo, "revision": s.revision,
                     "pinned": s.pinned, "resolved_commit": s.resolved_commit,
                     "release_manifest": bool(s.release)} for s in sources],
        "bytes": stats,
        "experts": provenance,
    }
    (out / "fq-fetch-report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n")


def copy_attestations(out: Path, plans: list[FilePlan]) -> None:
    """Keep the (verified) attestation lines that justify these bytes, one
    directory per source, so the fetched tree can be re-checked offline."""
    for plan in plans:
        for src in {p.source.slug: p.source for p in plan.pieces}.values():
            name = Source.attestation_name(plan.layer, plan.k)
            raw = src.small_file(name)
            if raw is None:
                continue
            dst = out / "attestations" / src.slug / Path(name).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(raw)


def local_attestation(plan: FilePlan, entry: dict, verifier: fq_trust.Verifier,
                      keyid: str) -> dict:
    """A `derived-from` payload for one locally materialized subset file.

    A subset segment is a NEW file: fewer experts, different offsets,
    therefore a different whole-file digest than anything the publisher
    signed.  The publisher's signature covers the experts (their per-expert
    digests were checked byte by byte as the ranges landed) but cannot cover
    this file — so fq_fetch signs what it actually produced, with the
    publisher's fragments named as parents and pinned by digest.

    The chain a consumer ends up with:

        publisher key (pinned at fetch time, out of band)
          -> attested per-expert digests
            -> these exact bytes, hashed on arrival
              -> this file, attested under YOUR key
                -> fq_assemble --trust-signer <your fingerprint>

    Your own key is the right signer for the last hop: nobody else can
    honestly attest a file only your machine assembled.  The publishers'
    original lines are kept under attestations/<source>/ so the earlier
    hops stay checkable offline.
    """
    parents = []
    for src in sorted({p.source.slug: p.source for p in plan.pieces}.values(),
                      key=lambda s: s.slug):
        experts = sorted(p.expert for p in plan.pieces if p.source is src)
        att = src.attestation(plan.layer, plan.k, verifier,
                              Source.segment_name(plan.layer, plan.k))
        frag = att.get("fragment") or {}
        prov = plan.header_provenance.get(src.slug) or {}
        parents.append({
            "role": "source_fragment",
            "repo": src.repo,
            "revision": src.revision,
            "file": frag.get("file"),
            "sha256": frag.get("sha256"),
            "size": frag.get("size"),
            "keyid": verifier.fingerprint,
            "experts": experts,
            # exactly which authenticated inputs this plan was computed from:
            # the layout (names, dtypes, shapes, offsets) came from the
            # parent's header, so say how that header was proven
            "header_authentication": prov.get("method"),
            "header_authenticated": bool(prov.get("authenticated")),
            "header_sha256": prov.get("header_sha256"),
            "signed_release_manifest": bool(prov.get("release_manifest")),
        })
    payload = {
        "schema": "fq-attestation/1",
        "predicate": "derived-from",
        "fragment": {"file": entry["file"], "sha256": entry["sha256"],
                     "size": entry["size"],
                     "header_sha256": entry.get("header_sha256"),
                     "body_offset": entry.get("body_offset")},
        "expert_sha256": {str(p.expert): p.sha256
                          for p in sorted(plan.pieces, key=lambda x: x.expert)},
        "layer": plan.layer,
        "k": plan.k,
        "derivation": {
            "rule": "range_subset_v1",
            "description": (
                "byte-range subset of the parent fragment(s): the experts this "
                "recipe routes to K{k} were range-read from the parents and "
                "written in canonical order into a smaller fq-segment/1 file. "
                "Expert bytes are verbatim — each span was hashed against the "
                "parent's signed expert_sha256 before this file was finalized "
                "— but the file digest is new because the file is new."
            ).format(k=plan.k),
        },
        "parents": parents,
        "verification": {
            "tool": "fq_fetch",
            "upstream_rung": verifier.rung,
            "upstream_signer": verifier.fingerprint,
            "expert_digests_checked": len(plan.pieces),
            # the byte plan (which tensor lives where, with what dtype and
            # shape) is only as trustworthy as the header it was read from
            "plan_inputs": {
                "expert_digests": "signed attestation (per-expert sha256)",
                "byte_layout": "segment header, authenticated per parent "
                               "(see parents[].header_authentication)",
                "index_spans": "cross-checked against the authenticated "
                               "header before use",
                "plan_authenticated": all(
                    (plan.header_provenance.get(s) or {}).get("authenticated")
                    for s in {p.source.slug for p in plan.pieces}),
            },
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for field in ("layout", "base_model"):
        value = plan.meta.get(field)
        if value:
            payload[field] = value
    return payload


def write_local_attestation(out: Path, plan: FilePlan, entry: dict,
                            signer, verifier: fq_trust.Verifier) -> None:
    """Sign and write attestations/<segment stem>.jsonl for a subset file."""
    payload = local_attestation(plan, entry, verifier, signer.pub_hex)
    dst = out / "attestations" / f"{Path(plan.name).stem}.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(signer.sign_line(payload) + "\n")


# --------------------------------------------------------------------- run

def load_state(out: Path) -> dict:
    p = out / "state.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            print("warning: unreadable state.json, starting fresh", file=sys.stderr)
    return {"schema": "fq-fetch-state/1", "files": {}}


def save_state(out: Path, state: dict) -> None:
    tmp = out / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, out / "state.json")

def _completed_entry(plan: FilePlan, target: Path) -> dict:
    """Re-establish every claim a done-state entry is allowed to make."""
    expected_header = {"__metadata__": plan.meta, **plan.header}
    expected_hj = json.dumps(expected_header, separators=(",", ":")).encode()
    expected_hj += b" " * ((8 - len(expected_hj) % 8) % 8)
    size = target.stat().st_size
    digest = hashlib.sha256()
    with open(target, "rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise TrustError(f"{plan.name}: completed file has no safetensors header")
        hlen = struct.unpack("<Q", prefix)[0]
        # State-controlled bytes must never select an unbounded read.  A
        # completed subset has exactly the canonical header SegmentWriter
        # would produce for this authenticated plan.
        if hlen != len(expected_hj):
            raise TrustError(
                f"{plan.name}: completed file header length does not match "
                "the authenticated plan")
        hj = f.read(hlen)
        if hj != expected_hj:
            raise TrustError(
                f"{plan.name}: completed file header no longer matches the "
                "authenticated plan")
        body_offset = 8 + hlen
        if size != body_offset + plan.body_size:
            raise TrustError(
                f"{plan.name}: completed file size no longer matches the plan")
        digest.update(prefix)
        digest.update(hj)
        for block in iter(lambda: f.read(1 << 22), b""):
            digest.update(block)
        for piece in plan.pieces:
            f.seek(body_offset + piece.local_off)
            got = hashlib.sha256(f.read(piece.size)).hexdigest()
            if got != piece.sha256:
                raise TrustError(
                    f"{plan.name}: completed expert {piece.expert} no longer "
                    "matches its signed digest")
    return {
        "file": plan.name,
        "sha256": digest.hexdigest(),
        "size": size,
        "body_offset": body_offset,
        "header_sha256": hashlib.sha256(prefix + hj).hexdigest(),
        "experts": {str(p.expert): [p.local_off, p.local_off + p.size]
                    for p in sorted(plan.pieces, key=lambda x: x.expert)},
        "sources": getattr(plan, "sources_used", {}),
    }


def fetch_plan(plan: FilePlan, out: Path, state: dict, transport: Transport,
               *, max_chunk: int, max_gap: int, verify_local: bool) -> dict:
    """Fetch one segment file's missing experts.  Returns its index entry."""
    key = plan.name
    st = state["files"].get(key) or {}
    target = out / plan.name
    digest = plan.digest()
    if (st.get("plan") == digest and st.get("status") == "done"
            and target.exists() and st.get("entry")):
        try:
            entry = _completed_entry(plan, target)
        except TrustError as e:
            print(f"  {plan.name}: {e} — refetching", flush=True)
            st = {"plan": digest, "done": [], "status": "partial"}
            target.unlink()
        else:
            st["entry"] = entry
            state["files"][key] = st
            save_state(out, state)
            print(f"  {plan.name}: complete ({len(plan.pieces)} experts) — skipped",
                  flush=True)
            return entry
    if st.get("plan") != digest:
        st = {"plan": digest, "done": [], "status": "partial"}
        stale = target.with_name(target.name + ".part")
        if stale.exists():
            stale.unlink()
        if target.exists():
            target.unlink()
    done = set(st.get("done") or [])

    writer = SegmentWriter(target, plan.meta, plan.header, plan.body_size)
    fetched = wasted = 0
    try:
        todo = [p for p in plan.pieces if p.expert not in done]
        if verify_local and done:
            # A resumed .part is bytes we did not just hash: re-check the
            # experts we are about to trust before adding to them.
            for p in plan.pieces:
                if p.expert in done:
                    got = hashlib.sha256(writer.read(p.local_off, p.size)).hexdigest()
                    if got != p.sha256:
                        print(f"  {plan.name}: resumed expert {p.expert} no longer "
                              f"matches its digest — refetching", flush=True)
                        done.discard(p.expert)
                        todo.append(p)
        by_source: dict[str, list[Piece]] = {}
        for p in todo:
            by_source.setdefault(p.source.slug, []).append(p)
        for pieces in by_source.values():
            src = pieces[0].source

            def accept(piece: Piece, blob: bytes) -> None:
                got = hashlib.sha256(blob).hexdigest()
                if got != piece.sha256:
                    raise TrustError(
                        f"{plan.name}: expert {piece.expert} from {piece.source} "
                        f"hashes to {got[:16]}… but the signed attestation "
                        f"says {piece.sha256[:16]}… — refusing these bytes")
                writer.pwrite(piece.local_off, blob)
                done.add(piece.expert)

            remaining = []
            for piece in pieces:
                blob = src.take_verified_piece(plan.name, piece)
                if blob is None:
                    remaining.append(piece)
                else:
                    accept(piece, blob)
            if len(remaining) != len(pieces):
                st["done"] = sorted(done)
                state["files"][key] = st
                save_state(out, state)
            url = src.url(plan.name)
            for start, end, group in coalesce(remaining, max_chunk, max_gap):
                data, _ = transport.get_range(url, start, end)
                fetched += len(data)
                wasted += len(data) - sum(p.size for p in group)
                for piece in group:
                    accept(piece, data[piece.remote_start - start:
                                       piece.remote_end - start])
                st["done"] = sorted(done)
                state["files"][key] = st
                save_state(out, state)
        sha, size = writer.finalize()
    except BaseException:
        writer.abandon()
        st["done"] = sorted(done)
        state["files"][key] = st
        save_state(out, state)
        raise
    with open(target, "rb") as f:  # the header digest OUR consumers can pin
        header_sha = hashlib.sha256(f.read(writer.body_off)).hexdigest()
    entry = {
        "file": plan.name,
        "sha256": sha,
        "size": size,
        "body_offset": writer.body_off,
        "header_sha256": header_sha,
        "experts": {str(p.expert): [p.local_off, p.local_off + p.size]
                    for p in sorted(plan.pieces, key=lambda x: x.expert)},
        "sources": getattr(plan, "sources_used", {}),
    }
    st.update({"status": "done", "entry": entry, "plan": digest,
               "done": sorted(done)})
    state["files"][key] = st
    save_state(out, state)
    print(f"  {plan.name}: {len(plan.pieces)} experts, {human(fetched)} fetched"
          + (f" ({human(wasted)} coalescing overhead)" if wasted else "")
          + f" -> {sha[:12]}…", flush=True)
    return entry


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, type=Path,
                   help="recipe: fq-policy/2 JSON with bits_per_expert")
    p.add_argument("--source", action="append", default=[],
                   metavar="REPO[@REV[:SUBDIR]]",
                   help="artifact repo or nested family, repeatable and ORDERED "
                   "(earlier wins ties). Pin a nested family as "
                   "REPO@40-hex-commit:relative/path; a bare repo follows main "
                   "and is a weaker pin.")
    p.add_argument("--out", required=True, type=Path,
                   help="local segment tree to build (fq_assemble --segments)")
    p.add_argument("--layers", default=None, help="subset, e.g. 3-10 or 3,5,7")
    p.add_argument("--dry-run", action="store_true",
                   help="plan and print byte counts, fetch nothing")
    p.add_argument("--prefer-sha", action="append", default=[], metavar="SHA256",
                   help="content-hash selection: prefer the source whose "
                        "attested expert digest is this, whoever publishes it. "
                        "Repeatable.")
    p.add_argument("--select", type=Path, default=None, metavar="MAP.json",
                   help="fq-select/1 provider map: per-expert or per-layer "
                        "source choice ({'experts': {'3': {'137': 'repo'}}})")
    p.add_argument("--pace", type=float, default=0.1,
                   help="minimum seconds between requests (default 0.1)")
    p.add_argument("--retries", type=int, default=6)
    p.add_argument("--chunk-mb", type=float, default=64.0,
                   help="maximum megabytes per ranged GET (default 64)")
    p.add_argument("--max-gap-mb", type=float, default=1.0,
                   help="merge spans separated by less than this, fetching "
                        "and discarding the filler (default 1 MB)")
    p.add_argument("--sign-key", type=Path,
                   default=Path.home() / ".fq_keys/fq_fetch.key",
                   help="ed25519 seed (32 bytes, created on demand) used to "
                        "attest the subset files this run materializes.  They "
                        "are new files, so no publisher signature can cover "
                        "them; pin the printed fingerprint when assembling.")
    p.add_argument("--no-attest", action="store_true",
                   help="do not sign the fetched subset — the tree will then "
                        "need fq_assemble --insecure")
    p.add_argument("--no-verify-resumed", action="store_true",
                   help="skip re-hashing partial .part expert bytes; completed "
                        "files are always revalidated before reuse")
    p.add_argument("--header-trust", choices=HEADER_TRUST_MODES,
                   default=HEADER_AUTO,
                   help="how the remote segment HEADER (tensor names, dtypes, "
                        "shapes, offsets — the meaning of the bytes) is "
                        "authenticated before the plan uses it.  auto: signed "
                        "fragment.header_sha256 when published, else verify "
                        "the whole fragment against the signed file digest.  "
                        "attested: require the signed header digest, never "
                        "download a whole fragment.  full: always verify the "
                        "whole fragment.  unsafe: plan from the unverified "
                        "ranged header (what fq_fetch did before) — recorded "
                        "as such in the emitted attestation.")
    p.add_argument("--json", type=Path, default=None,
                   help="write the plan/result summary here as well")
    fq_trust.add_trust_arguments(p)
    args = p.parse_args(argv)

    if not args.source:
        p.error("at least one --source is required")
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    cache_root = out / ".fq-fetch-cache"
    transport = Transport(pace=args.pace, retries=args.retries)
    sources = [Source.parse(s, transport, cache_root, order=i)
               for i, s in enumerate(args.source)]
    for s in sources:
        if not s.pinned:
            print(f"warning: {s.repo} is not pinned to a revision — following "
                  f"'main' means the bytes can change under you; pass "
                  f"{s.repo}@<commit>", file=sys.stderr)
        s.load_manifest()

    manifest0 = sources[0].manifest
    try:
        verifier = fq_trust.Verifier.from_args(args, manifest=manifest0)
        for s in sources:
            s.load_release(verifier)
        policy = load_policy(args.policy, args.layers)
        binding = load_fetch_binding(args.policy)
        if binding and (args.select or args.prefer_sha):
            raise TrustError(
                "a recipe fetch_binding is authoritative; do not pass "
                "--select or --prefer-sha")
        if binding:
            validate_fetch_binding(binding, policy, sources)
            select = binding["providers"]
            prefer_sha: set[str] = set()
        else:
            select = load_select(args.select)
            prefer_sha = set(args.prefer_sha)
        plans, problems = plan_fetch(policy, sources, verifier,
                                     prefer_sha=prefer_sha, select=select,
                                     binding=binding)
    except TrustError as e:
        print(f"TRUST FAILURE: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted while planning — nothing fetched", file=sys.stderr)
        return 130

    for msg in problems:
        print(f"  ! {msg}", file=sys.stderr)
    if binding and problems:
        print("bound recipe has unsatisfied providers or content digests; "
              "nothing fetched", file=sys.stderr)
        return 1
    if not plans:
        print("nothing to fetch: no source carries the experts this recipe asks "
              "for", file=sys.stderr)
        return 1

    needed = sum(pl.bytes_needed for pl in plans)
    experts = sum(len(pl.pieces) for pl in plans)
    cf = counterfactual(sources, plans)
    auth = authentication_cost(plans, args.header_trust)
    summary = {
        "experts": experts,
        "files": len(plans),
        "ranged_bytes": needed,
        "whole_segment_files_bytes": cf["whole_files"],
        "whole_repo_bytes": cf["whole_repo"],
        "segment_files_touched": cf["files_touched"],
        "trust_rung": verifier.rung,
        "signer": verifier.fingerprint,
        "header_authentication": auth,
    }
    print(f"plan: {experts} experts across {len(plans)} segment files")
    print(f"  ranged fetch:        {human(needed)}")
    print(f"  whole segment files: {human(cf['whole_files'])}"
          + (f"  ({cf['whole_files'] / max(needed, 1):.1f}x)" if needed else ""))
    print(f"  whole repo download: {human(cf['whole_repo'])}"
          + (f"  ({cf['whole_repo'] / max(needed, 1):.1f}x)" if needed else "")
          + "   <- what `hf download <repo>` costs")
    print("  header authentication: "
          + ", ".join(f"{n}x {m}" for m, n in sorted(auth["methods"].items()))
          + (f"  (+{human(auth['extra_bytes'])} read to prove the plan; ask "
             f"the publisher for fragment.header_sha256 to avoid it)"
             if auth["extra_bytes"] else ""))
    print(f"  {verifier.summary()}")
    if args.header_trust == HEADER_UNSAFE:
        print("!" * 72 + "\n"
              "!! --header-trust unsafe: tensor names, dtypes, shapes and\n"
              "!! offsets come from an UNAUTHENTICATED ranged read.  Expert\n"
              "!! bytes are still hashed against the signed attestation, but\n"
              "!! a publisher who rewrote a header can relabel them, and the\n"
              "!! subset this run signs would carry the relabelling.\n"
              + "!" * 72, file=sys.stderr, flush=True)
    if args.dry_run:
        if args.json:
            args.json.write_text(json.dumps(
                {"dry_run": True, **summary,
                 "plans": [{"file": pl.name, "experts": len(pl.pieces),
                            "bytes": pl.bytes_needed} for pl in plans]},
                indent=1) + "\n")
        print("dry run: nothing fetched (the plan above is drafted from "
              "unverified headers; a real run authenticates them first)")
        return 0

    signer = None
    if not args.no_attest:
        from fq_repack import Signer  # local key handling, same as fq_repack

        signer = Signer(args.sign_key)

    state = load_state(out)
    entries: dict[tuple[int, int], dict] = {}
    t0 = time.time()
    try:
        for plan in plans:
            # Prove the header before a single byte of this file is fetched
            # or written: the draft plan above is not evidence of anything.
            parents = {p.source.slug: p.source for p in plan.pieces}.values()
            try:
                authenticate_plan(plan, args.header_trust)
                entry = fetch_plan(
                    plan, out, state, transport,
                    max_chunk=max(1, int(args.chunk_mb * (1 << 20))),
                    max_gap=int(args.max_gap_mb * (1 << 20)),
                    verify_local=not args.no_verify_resumed)
                entries[(plan.layer, plan.k)] = entry
                if signer is not None:
                    write_local_attestation(out, plan, entry, signer, verifier)
            finally:
                for src in parents:
                    src.discard_verified_pieces(plan.name)
    except TrustError as e:
        print(f"TRUST FAILURE: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted — progress saved, re-run the same command to "
              "resume", file=sys.stderr)
        return 130

    copy_attestations(out, plans)
    stats = {**summary, "transport": transport.stats(),
             "elapsed_s": round(time.time() - t0, 1)}
    local_fp = signer.pub_hex if signer is not None else None
    write_outputs(out, plans, entries, sources, verifier, stats, args.policy,
                  local_signer=local_fp)
    if args.json:
        args.json.write_text(json.dumps({"dry_run": False, **stats}, indent=1) + "\n")
    print(f"fetched {experts} experts ({human(transport.bytes)} over "
          f"{transport.requests} range requests) -> {out}")
    if local_fp:
        print(f"subset attested as derived-from under your key {local_fp[:16]}… "
              f"({args.sign_key})")
        print(f"assemble with: fq_assemble.py --segments {out} --source "
              f"<checkpoint> --policy {args.policy} --out <dir> "
              f"--trust-signer {local_fp}")
    else:
        print(f"assemble with: fq_assemble.py --segments {out} --source "
              f"<checkpoint> --policy {args.policy} --out <dir> --insecure "
              f"(--no-attest was used, so nothing signs this subset)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
