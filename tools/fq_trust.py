#!/usr/bin/env python3
"""fq_trust — signer pinning and signature verification against a trust root
whose authority comes from git, not from the artifact repository.

The problem this module exists to solve (peer review, finding 5): every
Progressive Tensors artifact repo carries its own `fq-manifest.json` with a
`signer_pubkey` field, and every attestation carries a `keyid`.  Verifying a
signature against the key that shipped beside it proves only that one
uploader produced a self-consistent bundle — an attacker who can rewrite the
repository rewrites bytes, attestations and key together and every check
still passes.  A real trust root has to live somewhere the artifact host
cannot rewrite.

Ours is `keys/FINGERPRINTS` in github.com/malaiwah/progressive-tensors: a
plain-text list of authorized ed25519 public keys with status and dates,
changed only by reviewable commits.  See ../TRUST.md.

Usage from a tool:

    import fq_trust

    fq_trust.add_trust_arguments(parser)          # --trust-signer, ...
    ...
    verifier = fq_trust.Verifier.from_args(args, manifest=manifest)
    payload = verifier.verify_envelope(json.loads(line), where=att_path.name)
    print(verifier.summary())                     # one honest line for logs

`Verifier.verify_envelope` raises `TrustError` unless the signature actually
validates under a key the caller is entitled to trust.  It never "verifies"
by decoding: a payload that parses but whose signature is absent, truncated,
or wrong is an error, not a pass.

Trust ladder, strongest first — every tool prints which rung it used:

  pinned      --trust-signer <fingerprint>: the signature must validate under
              exactly that key.  A compromised artifact repo cannot pass.
  trust-root  no fingerprint given: the signature must validate under some
              key listed `active` in keys/FINGERPRINTS.  A compromised
              artifact repo cannot pass either, but the consumer has not said
              which of the project's keys they expect.
  unpinned    --allow-unpinned-signer: the signature must validate under the
              key named by the artifact itself.  This proves internal
              consistency only, and says nothing about who made the bytes.
  none        --insecure-skip-signatures: no cryptography at all.  For
              offline tests and synthetic fixtures.
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import re
import sys
from pathlib import Path

FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
SHORT_MIN = 16  # shortest accepted --trust-signer prefix, in hex chars

RUNG_PINNED = "pinned"
RUNG_TRUST_ROOT = "trust-root"
RUNG_UNPINNED = "unpinned"
RUNG_NONE = "none"


class TrustError(Exception):
    """A signature could not be trusted.  Callers should treat this as fatal."""


# --------------------------------------------------------------- trust root

def _candidate_roots() -> list[Path]:
    """Where keys/FINGERPRINTS may live, most authoritative first.

    In a clone the tools sit in tools/ next to keys/.  In an installed wheel
    the modules are top-level and the trust root ships alongside them as
    fq_data/keys/FINGERPRINTS — a snapshot of the git file at build time.
    The snapshot is a convenience, not the authority: `git pull` is how
    rotations and revocations reach you.
    """
    here = Path(__file__).resolve().parent
    return [
        here.parent / "keys" / "FINGERPRINTS",
        here / "fq_data" / "keys" / "FINGERPRINTS",
    ]


class TrustRoot:
    """Parsed keys/FINGERPRINTS."""

    def __init__(self, records: list[dict], path: Path | None):
        self.records = records
        self.path = path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "TrustRoot":
        """Load an explicit path, else $FQ_TRUST_ROOT, else the shipped file.

        Returns an empty root (path=None) when nothing is found; callers
        decide whether that is fatal for the rung they are on.
        """
        candidates: list[Path] = []
        if path is not None:
            candidates = [Path(path)]
        elif os.environ.get("FQ_TRUST_ROOT"):
            candidates = [Path(os.environ["FQ_TRUST_ROOT"])]
        else:
            candidates = _candidate_roots()
        for c in candidates:
            if c.is_dir():
                c = c / "FINGERPRINTS"
            if c.exists():
                return cls(cls._parse(c.read_text()), c)
        if path is not None or os.environ.get("FQ_TRUST_ROOT"):
            raise TrustError(f"trust root not found: {candidates[0]}")
        return cls([], None)

    @staticmethod
    def _parse(text: str) -> list[dict]:
        out = []
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3 or not FINGERPRINT_RE.match(parts[0]):
                continue
            out.append({
                "fingerprint": parts[0],
                "key_id": parts[1],
                "status": parts[2],
                "added": parts[3] if len(parts) > 3 else "",
                "role": parts[4] if len(parts) > 4 else "",
            })
        return out

    def status(self, fingerprint: str) -> str | None:
        for r in self.records:
            if r["fingerprint"] == fingerprint:
                return r["status"]
        return None

    def key_id(self, fingerprint: str) -> str | None:
        for r in self.records:
            if r["fingerprint"] == fingerprint:
                return r["key_id"]
        return None

    def active(self) -> list[str]:
        return [r["fingerprint"] for r in self.records if r["status"] == "active"]

    def resolve(self, spec: str) -> str:
        """Turn a --trust-signer value into a full 64-hex fingerprint.

        Accepts: a full fingerprint; an unambiguous hex prefix of at least 16
        characters; a key id from the trust root; or a path to a .pub file.
        A prefix that matches nothing is only accepted at full length — that
        way an out-of-band fingerprint still works with an old clone, while a
        typo'd short prefix is rejected instead of silently trusting nothing.
        """
        spec = spec.strip()
        p = Path(spec)
        if os.sep in spec and p.exists():
            spec = p.read_text().split()[0].strip().lower()
        low = spec.lower()
        if FINGERPRINT_RE.match(low):
            return low
        if re.match(r"^[0-9a-f]+$", low):
            if len(low) < SHORT_MIN:
                raise TrustError(
                    f"--trust-signer {spec!r}: a fingerprint prefix must be at "
                    f"least {SHORT_MIN} hex characters (or give all 64)")
            hits = [r["fingerprint"] for r in self.records
                    if r["fingerprint"].startswith(low)]
            if len(hits) == 1:
                return hits[0]
            if not hits:
                raise TrustError(
                    f"--trust-signer {spec!r}: no key with that prefix in "
                    f"{self.path or 'the trust root'} — check keys/FINGERPRINTS, "
                    f"or pass the full 64-hex fingerprint if you have it "
                    f"out of band")
            raise TrustError(f"--trust-signer {spec!r}: ambiguous prefix "
                             f"({len(hits)} keys match)")
        hits = [r["fingerprint"] for r in self.records if r["key_id"] == spec]
        if len(hits) == 1:
            return hits[0]
        raise TrustError(
            f"--trust-signer {spec!r}: not a fingerprint, a >={SHORT_MIN}-hex "
            f"prefix, or a key id in {self.path or 'the trust root'}")


# ------------------------------------------------------------- verification

def _b64(field: str, value, *, where: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise TrustError(f"{where}: envelope field {field!r} is missing or not a string")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as e:
        raise TrustError(f"{where}: envelope field {field!r} is not valid base64 ({e})")


def verify_signature(envelope: dict, fingerprint: str, *, where: str = "envelope") -> dict:
    """Verify one {payload, signature, keyid} envelope under `fingerprint`.

    Returns the decoded payload dict.  Raises TrustError on any failure —
    unknown key, wrong key, malformed base64, wrong signature length, bad
    signature, payload that is not a JSON object.  Decoding is never
    mistaken for verifying.
    """
    if not isinstance(envelope, dict):
        raise TrustError(f"{where}: expected a JSON object envelope")
    keyid = str(envelope.get("keyid", "")).lower()
    if not FINGERPRINT_RE.match(keyid):
        raise TrustError(f"{where}: envelope 'keyid' is not a 64-hex ed25519 key")
    if keyid != fingerprint:
        raise TrustError(
            f"{where}: signed by {keyid[:16]}… but the trusted signer is "
            f"{fingerprint[:16]}… — refusing (this is exactly the swap a "
            f"compromised artifact repo would perform)")
    raw = _b64("payload", envelope.get("payload"), where=where)
    sig = _b64("signature", envelope.get("signature"), where=where)
    if len(sig) != 64:
        raise TrustError(
            f"{where}: ed25519 signature must be 64 bytes, got {len(sig)} — "
            f"a placeholder or truncated signature is a verification failure, "
            f"not a pass")
    try:
        from nacl.exceptions import BadSignatureError
        from nacl.signing import VerifyKey
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise TrustError(
            f"{where}: pynacl is required to verify signatures ({e}); install "
            f"it, or re-run with --insecure-skip-signatures and accept that "
            f"nothing is proven") from e
    try:
        VerifyKey(bytes.fromhex(fingerprint)).verify(raw, sig)
    except BadSignatureError:
        raise TrustError(f"{where}: BAD SIGNATURE under {fingerprint[:16]}…")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise TrustError(f"{where}: signed payload is not JSON ({e})")
    if not isinstance(payload, dict):
        raise TrustError(f"{where}: signed payload is not a JSON object")
    return payload


def decode_payload(envelope: dict, *, where: str = "envelope") -> dict:
    """Decode an envelope's payload WITHOUT verifying anything.

    Only for --insecure-skip-signatures paths and for reporting what a bad
    envelope claimed.  Named so that no call site can mistake it for a check.
    """
    raw = _b64("payload", envelope.get("payload"), where=where)
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TrustError(f"{where}: signed payload is not a JSON object")
    return payload


class Verifier:
    """Fixed trust decision applied to every envelope of one run."""

    def __init__(self, *, fingerprint: str | None, rung: str,
                 trust_root: TrustRoot | None = None, key_id: str | None = None):
        self.fingerprint = fingerprint
        self.rung = rung
        self.trust_root = trust_root
        self.key_id = key_id
        self.checked = 0

    # -- construction ------------------------------------------------------

    @classmethod
    def from_args(cls, args, *, manifest: dict | None = None) -> "Verifier":
        """Build from parsed args (see add_trust_arguments) plus, for the
        unpinned rung, the artifact's own fq-manifest/1 dict."""
        return cls.resolve(
            trust_signer=getattr(args, "trust_signer", None),
            trust_root=getattr(args, "trust_root", None),
            allow_unpinned=getattr(args, "allow_unpinned_signer", False),
            skip=getattr(args, "insecure_skip_signatures", False),
            manifest=manifest,
        )

    @classmethod
    def resolve(cls, *, trust_signer: str | None = None,
                trust_root: str | Path | None = None,
                allow_unpinned: bool = False, skip: bool = False,
                manifest: dict | None = None) -> "Verifier":
        if skip:
            return cls(fingerprint=None, rung=RUNG_NONE)
        root = TrustRoot.load(trust_root)
        if trust_signer:
            fp = root.resolve(trust_signer)
            status = root.status(fp)
            if status == "revoked":
                raise TrustError(
                    f"--trust-signer {fp[:16]}… is REVOKED in {root.path} — "
                    f"refusing.  Revocation is retroactive: re-verify against "
                    f"an active key or re-derive the artifact from source.")
            return cls(fingerprint=fp, rung=RUNG_PINNED, trust_root=root,
                       key_id=root.key_id(fp))
        if allow_unpinned:
            fp = str((manifest or {}).get("signer_pubkey", "")).lower()
            if not FINGERPRINT_RE.match(fp):
                raise TrustError(
                    "--allow-unpinned-signer needs the artifact's own "
                    "fq-manifest.json signer_pubkey, which is missing or "
                    "malformed")
            return cls(fingerprint=fp, rung=RUNG_UNPINNED, trust_root=root,
                       key_id=root.key_id(fp))
        claimed = str((manifest or {}).get("signer_pubkey", "")).lower()
        if claimed:
            status = root.status(claimed)
            if status == "retired":
                print(f"warning: signer {claimed[:16]}… is RETIRED in "
                      f"{root.path} — signatures it made before retirement "
                      f"remain valid, so check that these artifacts predate "
                      f"it", file=sys.stderr)
            elif status != "active":
                raise TrustError(
                    f"the artifact claims signer {claimed[:16]}… which is "
                    f"{status or 'not listed'} in {root.path}.  Refusing.  If "
                    f"this key is legitimately new, update your clone (git "
                    f"pull) and look at the commit that added it.")
        active = root.active()
        if not active and not claimed:
            raise TrustError(
                "no trust root found (keys/FINGERPRINTS) and no "
                "--trust-signer given.  Pass --trust-signer <fingerprint> "
                "obtained out of band, point --trust-root at a checkout of "
                "github.com/malaiwah/progressive-tensors, or accept an "
                "unproven run with --allow-unpinned-signer.")
        fp = claimed or (active[0] if len(active) == 1 else None)
        if fp is None:
            raise TrustError(
                "the trust root lists several active keys and the artifact "
                "does not say which it used — pass --trust-signer")
        return cls(fingerprint=fp, rung=RUNG_TRUST_ROOT, trust_root=root,
                   key_id=root.key_id(fp))

    # -- use ---------------------------------------------------------------

    def verify_envelope(self, envelope: dict, *, where: str = "envelope") -> dict:
        if self.rung == RUNG_NONE:
            return decode_payload(envelope, where=where)
        payload = verify_signature(envelope, self.fingerprint, where=where)
        self.checked += 1
        return payload

    def verify_line(self, line: str, *, where: str = "envelope") -> dict:
        return self.verify_envelope(json.loads(line), where=where)

    def summary(self) -> str:
        if self.rung == RUNG_NONE:
            return ("signatures: NOT CHECKED (--insecure-skip-signatures) — "
                    "nothing about provenance is proven")
        who = f"{self.fingerprint[:16]}…"
        if self.key_id:
            who += f" ({self.key_id})"
        if self.rung == RUNG_PINNED:
            return (f"signatures: {self.checked} verified against pinned signer "
                    f"{who}")
        if self.rung == RUNG_TRUST_ROOT:
            return (f"signatures: {self.checked} verified against {who}, listed "
                    f"active in {self.trust_root.path}")
        return (f"signatures: {self.checked} verified against the artifact's own "
                f"key {who} — UNPINNED: self-consistent only, provenance "
                f"unproven")


def add_trust_arguments(parser) -> None:
    """Attach the standard trust flags to an argparse parser."""
    g = parser.add_argument_group("trust")
    g.add_argument(
        "--trust-signer", metavar="FINGERPRINT", default=None,
        help="pin the ed25519 signer: 64-hex fingerprint, an unambiguous "
             ">=16-hex prefix, a key id from keys/FINGERPRINTS, or a path to "
             "a .pub file.  Signatures must validate under exactly this key. "
             "Get the fingerprint from the git repo, never from the artifact "
             "download.")
    g.add_argument(
        "--trust-root", metavar="PATH", default=None,
        help="keys/FINGERPRINTS (or the keys/ dir) to resolve fingerprints "
             "and key ids against.  Default: the copy shipped with these "
             "tools, or $FQ_TRUST_ROOT.")
    g.add_argument(
        "--allow-unpinned-signer", action="store_true",
        help="verify signatures against the key the artifact itself names. "
             "Proves internal consistency only — a rewritten repo passes.")
    g.add_argument(
        "--insecure-skip-signatures", action="store_true",
        help="do not verify signatures at all (offline fixtures, tests).")


__all__ = [
    "TrustError", "TrustRoot", "Verifier", "add_trust_arguments",
    "verify_signature", "decode_payload",
    "RUNG_PINNED", "RUNG_TRUST_ROOT", "RUNG_UNPINNED", "RUNG_NONE",
]
