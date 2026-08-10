#!/usr/bin/env python3
"""fq_release — one signed document per release, listing every file's sha256.

Why this exists (peer review, finding 5).  Today a consumer who wants to know
that a segment tree is the one the project published has to check N
independent `fq-attestation/1` signatures — one per layer per K, 152 of them
for GLM-5.2 — and each of those signatures only ever says "this fragment came
from that source shard".  Nothing says *which set of fragments is the
release*.  Fragments can be added, dropped, or rolled back to an older signed
version and every individual signature still verifies.

`fq-release/1` fixes the granularity: ONE ed25519 signature over a document
that names the release and lists sha256 + size for every file in it,
including `index-k*.json`, `fq-manifest.json` and every
`attestations/*.jsonl`.  A consumer then:

    1. verifies one signature against a fingerprint pinned out of band
       (keys/FINGERPRINTS in the git repo — see ../TRUST.md),
    2. hashes the files it actually downloaded and compares,

and is done.  Because the attestation files are themselves covered by that
one signature, their per-expert digests become trusted data — a range-fetch
consumer (fq_fetch) can hash a single expert's bytes against the attestation
without checking any further signatures.  The release manifest turns "N
signatures, unknown completeness" into "1 signature, explicit set".

    # publisher, in the segment tree
    fq_release.py build --dir ~/fq-segments/GLM-5.2-EXL3-FQ \\
        --release "GLM-5.2-EXL3-FQ k3 base 0.1.0" \\
        --repo malaiwah/GLM-5.2-EXL3-FQ-segments

    # consumer, against a full or partial tree
    fq_release.py verify --dir ./segments \\
        --trust-signer a58b7bb79ba58457

Partial trees are first-class: `verify` reports how many listed files are
present and checks those, and only insists on completeness with --complete.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fq_trust  # noqa: E402
from fq_trust import TrustError  # noqa: E402

RELEASE_SCHEMA = "fq-release/1"
RELEASE_FILE = "fq-release.json"

# Local bookkeeping that is never part of a release.
EXCLUDE_NAMES = {RELEASE_FILE, "state.json", ".DS_Store"}
EXCLUDE_SUFFIXES = (".part", ".tmp", ".log")
EXCLUDE_DIRS = {".cache", "source-meta", "__pycache__", ".git"}


def sha256_file(path: Path, chunk: int = 1 << 22) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
            n += len(block)
    return h.hexdigest(), n


def iter_release_files(root: Path, include_meta: bool = False):
    """Every publishable file under root, as paths relative to root."""
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in EXCLUDE_DIRS for part in rel.parts[:-1]):
            if not (include_meta and rel.parts[0] == "source-meta"):
                continue
        if rel.name in EXCLUDE_NAMES or rel.name.endswith(EXCLUDE_SUFFIXES):
            continue
        yield rel


def build_payload(root: Path, *, release: str, repo: str | None,
                  revision: str | None, include_meta: bool = False,
                  progress: bool = False) -> dict:
    files: dict[str, dict] = {}
    total = 0
    for rel in iter_release_files(root, include_meta=include_meta):
        digest, size = sha256_file(root / rel)
        files[rel.as_posix()] = {"sha256": digest, "size": size}
        total += size
        if progress:
            print(f"  {rel.as_posix()}  {digest[:16]}…  {size}", flush=True)
    manifest_sha = files.get("fq-manifest.json", {}).get("sha256")
    fragments = sum(1 for n in files if n.endswith(".safetensors"))
    payload = {
        "schema": RELEASE_SCHEMA,
        "release": release,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {"files": len(files), "fragments": fragments, "bytes": total},
        "files": files,
    }
    if repo:
        payload["repo"] = repo
    if revision:
        payload["revision"] = revision
    if manifest_sha:
        payload["manifest_sha256"] = manifest_sha
    return payload


def load_release(path: Path) -> dict:
    """Read a release envelope from disk (no verification)."""
    return json.loads(Path(path).read_text())


def verify_release(envelope: dict, verifier: fq_trust.Verifier,
                   *, where: str = RELEASE_FILE) -> dict:
    """Verify the envelope and return the payload.  Raises TrustError."""
    payload = verifier.verify_envelope(envelope, where=where)
    if payload.get("schema") != RELEASE_SCHEMA:
        raise TrustError(f"{where}: signed payload is {payload.get('schema')!r}, "
                         f"expected {RELEASE_SCHEMA!r}")
    if not isinstance(payload.get("files"), dict) or not payload["files"]:
        raise TrustError(f"{where}: release manifest lists no files")
    return payload


def check_tree(payload: dict, root: Path, *, complete: bool = False,
               verbose: bool = False) -> dict:
    """Hash the files of `root` that the release lists.  Returns a report.

    Missing files are not an error by default: range-fetch consumers hold a
    deliberate subset.  Present-but-different, and (with --complete) missing,
    are errors.
    """
    present, missing, bad, extra = [], [], [], []
    for name, want in sorted(payload["files"].items()):
        p = root / name
        if not p.exists():
            missing.append(name)
            continue
        digest, size = sha256_file(p)
        if digest != want["sha256"] or size != want.get("size", size):
            bad.append({"file": name, "expected": want["sha256"],
                        "got": digest, "expected_size": want.get("size"),
                        "got_size": size})
        else:
            present.append(name)
            if verbose:
                print(f"  ok   {name}  {digest[:16]}…", flush=True)
    listed = set(payload["files"])
    for rel in iter_release_files(root):
        if rel.as_posix() not in listed:
            extra.append(rel.as_posix())
    ok = not bad and (not missing or not complete)
    return {"ok": ok, "present": present, "missing": missing, "bad": bad,
            "extra": extra}


# ------------------------------------------------------------------- CLI

def cmd_build(args) -> int:
    from fq_repack import Signer  # local import: signing needs the private key

    root = args.dir
    payload = build_payload(root, release=args.release, repo=args.repo,
                            revision=args.revision,
                            include_meta=args.include_source_meta,
                            progress=args.verbose)
    signer = Signer(args.sign_key)
    envelope = json.loads(signer.sign_line(payload))
    out = args.out or (root / RELEASE_FILE)
    out.write_text(json.dumps(envelope, indent=1, sort_keys=True) + "\n")
    c = payload["counts"]
    print(f"{RELEASE_SCHEMA}: {c['files']} files ({c['fragments']} fragments, "
          f"{c['bytes'] / 1e9:.2f} GB) signed by {signer.pub_hex[:16]}… -> {out}")
    print("publish this file with the artifacts; publish the FINGERPRINT in "
          "git (keys/FINGERPRINTS), not beside the artifacts")
    return 0


def cmd_verify(args) -> int:
    root = args.dir
    rel_path = args.release or (root / RELEASE_FILE)
    if not rel_path.exists():
        print(f"no release manifest at {rel_path}", file=sys.stderr)
        return 2
    envelope = load_release(rel_path)
    manifest_path = root / "fq-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    try:
        verifier = fq_trust.Verifier.from_args(args, manifest=manifest)
        payload = verify_release(envelope, verifier, where=rel_path.name)
    except TrustError as e:
        print(f"TRUST FAILURE: {e}", file=sys.stderr)
        return 1
    print(f"release: {payload.get('release')!r} "
          f"({payload['counts']['files']} files, created {payload.get('created_utc')})")
    print(verifier.summary())
    if payload.get("manifest_sha256") and manifest_path.exists():
        got, _ = sha256_file(manifest_path)
        state = "matches" if got == payload["manifest_sha256"] else "DIFFERS FROM"
        print(f"fq-manifest.json {state} the signed release")
    if args.no_check_files:
        return 0
    report = check_tree(payload, root, complete=args.complete, verbose=args.verbose)
    print(f"files: {len(report['present'])} verified, {len(report['missing'])} absent, "
          f"{len(report['bad'])} MISMATCHED, {len(report['extra'])} unlisted")
    for b in report["bad"]:
        print(f"  MISMATCH {b['file']}: expected {b['expected'][:16]}… "
              f"got {b['got'][:16]}…", file=sys.stderr)
    for x in report["extra"][:20]:
        print(f"  unlisted (not covered by the signature): {x}", file=sys.stderr)
    if args.json:
        args.json.write_text(json.dumps(
            {"release": payload.get("release"), "rung": verifier.rung,
             "signer": verifier.fingerprint, **report}, indent=1) + "\n")
    if not report["ok"]:
        print("release verification FAILED", file=sys.stderr)
        return 1
    if report["missing"]:
        print("note: absent files are not an error for a partial (range-fetched) "
              "tree; pass --complete to require the whole release")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="hash a segment tree and sign the file list")
    b.add_argument("--dir", required=True, type=Path, help="segment tree to release")
    b.add_argument("--release", required=True, help="human-readable release name")
    b.add_argument("--repo", default=None, help="artifact repo id, if published")
    b.add_argument("--revision", default=None, help="artifact repo revision, if known")
    b.add_argument("--sign-key", type=Path,
                   default=Path.home() / ".fq_keys/fq_signing.key",
                   help="ed25519 seed (32 bytes); created on demand")
    b.add_argument("--out", type=Path, default=None,
                   help=f"output envelope (default: <dir>/{RELEASE_FILE})")
    b.add_argument("--include-source-meta", action="store_true",
                   help="also cover source-meta/ (cached source headers)")
    b.add_argument("--verbose", action="store_true")
    b.set_defaults(func=cmd_build)

    v = sub.add_parser("verify", help="check one signature, then hash the files")
    v.add_argument("--dir", required=True, type=Path, help="local tree to check")
    v.add_argument("--release", type=Path, default=None,
                   help=f"release envelope (default: <dir>/{RELEASE_FILE})")
    v.add_argument("--complete", action="store_true",
                   help="fail when any listed file is absent locally")
    v.add_argument("--no-check-files", action="store_true",
                   help="verify the signature only, do not hash anything")
    v.add_argument("--json", type=Path, default=None, help="write a JSON report")
    v.add_argument("--verbose", action="store_true")
    fq_trust.add_trust_arguments(v)
    v.set_defaults(func=cmd_verify)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
