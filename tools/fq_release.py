#!/usr/bin/env python3
"""fq_release — one signed document per release, listing every file's sha256.

Why this exists (peer review, finding 5).  Today a consumer who wants to know
that a segment tree is the one the project published has to check N
independent `fq-attestation/1` signatures — one per layer per K, 152 of them
for GLM-5.2 — and each of those signatures only ever says "this fragment came
from that source shard".  Nothing says *which set of fragments is the
release*.  Fragments can be added, dropped, or rolled back to an older signed
version and every individual signature still verifies.

`fq-release/1` fixes the set granularity: ONE ed25519 signature over a
document that names the release and lists sha256 + size for every file in it,
including `index-k*.json`, `fq-manifest.json` and every
`attestations/*.jsonl`.  A consumer:

    1. verifies the release signature against a fingerprint pinned out of
       band (keys/FINGERPRINTS in the git repo — see ../TRUST.md),
    2. hashes each listed file it consumes and compares it to the release,
    3. applies that consumer's normal signer policy to every inner
       attestation line it consumes.

The release signature establishes the immutable release set and the bytes of
its listed documents.  It deliberately does **not** delegate authority over
inner attestation claims: an aggregation consumer must explicitly implement
and expose a separate delegation policy.  `fq_fetch` uses the conservative
model above, so its configured signer establishes expert and fragment digest
claims while the release signer establishes document integrity and coverage.

    # publisher, in the segment tree
    fq_release.py build --dir ~/fq-segments/GLM-5.2-EXL3-FQ \\
        --release "GLM-5.2-EXL3-FQ k3 base 0.1.0" \\
        --repo malaiwah/GLM-5.2-EXL3-FQ-segments

    # publisher, atomically, against a live repo
    fq_release.py publish --dir ~/fq-segments/GLM-5.2-EXL3-FQ \\
        --release "GLM-5.2-EXL3-FQ 0.1.0" \\
        --repo malaiwah/GLM-5.2-EXL3-FQ-segments

    # consumer, against a full or partial tree
    fq_release.py verify --dir ./segments \\
        --trust-signer a58b7bb79ba58457

Partial trees are first-class: `verify` reports how many listed files are
present and checks those, and only insists on completeness with --complete.
`--complete` is the strict rung: absent files AND unlisted files are both
failures, so "nothing was added and nothing was dropped" is a check you can
put in CI rather than a warning you have to read.

`publish` exists because `build` alone cannot make a release coherent on a
*remote* repo (finding P1-3).  A publisher that uploads file-by-file — the
default of every hub client, including our own campaign supervisor — leaves
the repository in a long succession of states that no signature describes,
and a second writer can interleave commits into the middle of one.  `publish`
therefore: reads the remote HEAD, builds and signs the release from the LOCAL
tree, and pushes every changed file plus `fq-release.json` as ONE
`create_commit` with `parent_commit=<the HEAD it read>`.  A concurrent writer
makes that push be *rejected* instead of silently interleaved; the retry loop
re-reads, rebuilds and re-attempts a bounded number of times.
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
CACHE_SCHEMA = "fq-release-cache/1"

# Local bookkeeping that is never part of a release.
EXCLUDE_NAMES = {RELEASE_FILE, "state.json", ".DS_Store"}
EXCLUDE_SUFFIXES = (".part", ".tmp", ".log")
EXCLUDE_DIRS = {".cache", ".huggingface", "source-meta", "__pycache__", ".git"}

# Retry budget for `publish` when a concurrent writer moves the remote HEAD.
DEFAULT_MAX_ATTEMPTS = 4


def sha256_file(path: Path, chunk: int = 1 << 22) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
            n += len(block)
    return h.hexdigest(), n


def git_blob_sha1(path: Path) -> str:
    """git's own object name for a file's content: sha1("blob <len>\\0" + bytes).

    Used only to answer "does the remote already hold exactly these bytes?"
    for files the hub stores in git rather than LFS, so that `publish` can
    skip re-uploading them.  It is a transfer optimisation, never a security
    decision: what a consumer verifies is the sha256 in the signed release.
    """
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324


class DigestCache:
    """Remember (size, mtime_ns, ctime_ns) -> sha256 across rebuilds.

    `publish` may have to rebuild the whole release several times when a
    concurrent writer wins the race, and rebuilding means re-hashing every
    byte of the tree — hundreds of gigabytes for a real segment repo.  The
    cache turns each retry into a stat() per file.

    Opt-in (`--cache PATH`) and deliberately so: it trusts the filesystem's
    metadata to tell it when content changed.  Do not point it at a tree
    something else rewrites in place while preserving timestamps.
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self.entries: dict[str, dict] = {}
        self.hits = self.misses = 0
        if self.path and self.path.exists():
            try:
                doc = json.loads(self.path.read_text())
                if doc.get("schema") == CACHE_SCHEMA:
                    self.entries = doc.get("entries") or {}
            except (OSError, ValueError):
                self.entries = {}          # a corrupt cache is just a cold one

    def digest(self, root: Path, rel) -> tuple[str, int]:
        key = rel.as_posix() if isinstance(rel, Path) else str(rel)
        p = root / key
        st = p.stat()
        e = self.entries.get(key)
        if (e and e.get("size") == st.st_size
                and e.get("mtime_ns") == st.st_mtime_ns
                and e.get("ctime_ns") == st.st_ctime_ns):
            self.hits += 1
            return e["sha256"], e["size"]
        digest, size = sha256_file(p)
        self.entries[key] = {"size": size, "mtime_ns": st.st_mtime_ns,
                             "ctime_ns": st.st_ctime_ns, "sha256": digest}
        self.misses += 1
        return digest, size

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"schema": CACHE_SCHEMA, "entries": self.entries}))
        tmp.replace(self.path)


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
                  progress: bool = False, cache: DigestCache | None = None,
                  extra: dict | None = None) -> dict:
    files: dict[str, dict] = {}
    total = 0
    for rel in iter_release_files(root, include_meta=include_meta):
        digest, size = (cache.digest(root, rel) if cache
                        else sha256_file(root / rel))
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
    if extra:
        payload.update(extra)
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
    deliberate subset.  Present-but-different is always an error.

    With `complete`, two further things are errors, and they are the whole
    point of the strict rung: a listed file that is **absent** (something was
    dropped or never arrived) and a present file that is **unlisted** (something
    was added that the signature does not cover).  Reporting an unlisted file
    as a warning made "no silent additions" unenforceable — a caller checking
    only the exit status could not tell the difference between a clean tree and
    a tree with an extra, unsigned `.safetensors` in it.
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
    ok = not bad and not (complete and (missing or extra))
    return {"ok": ok, "present": present, "missing": missing, "bad": bad,
            "extra": extra}


# ------------------------------------------------------------------- CLI

def cmd_build(args) -> int:
    root = args.dir
    cache = DigestCache(args.cache)
    payload = build_payload(root, release=args.release, repo=args.repo,
                            revision=args.revision,
                            include_meta=args.include_source_meta,
                            progress=args.verbose, cache=cache)
    cache.save()
    _, fingerprint, blob = sign_release(payload, args.sign_key)
    out = args.out or (root / RELEASE_FILE)
    out.write_bytes(blob)
    c = payload["counts"]
    print(f"{RELEASE_SCHEMA}: {c['files']} files ({c['fragments']} fragments, "
          f"{c['bytes'] / 1e9:.2f} GB) signed by {fingerprint[:16]}… -> {out}")
    print("publish this file with the artifacts; publish the FINGERPRINT in "
          "git (keys/FINGERPRINTS), not beside the artifacts")
    return 0


def sign_release(payload: dict, key_path: Path) -> tuple[dict, str, bytes]:
    """Sign a payload; return (envelope, signer fingerprint, serialized bytes)."""
    from fq_repack import Signer      # local import: signing needs the key

    signer = Signer(key_path)
    envelope = json.loads(signer.sign_line(payload))
    blob = (json.dumps(envelope, indent=1, sort_keys=True) + "\n").encode()
    return envelope, signer.pub_hex, blob


# ------------------------------------------------------- publish (atomic)

def read_remote_state(api, repo: str, *, repo_type: str, revision: str):
    """(head commit sha, {path: {size, lfs_sha256, blob_id}}) for the branch."""
    info = api.repo_info(repo, repo_type=repo_type, revision=revision,
                         files_metadata=True)
    remote = {}
    for s in info.siblings:
        lfs = s.lfs
        sha = None
        if lfs:
            sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        remote[s.rfilename] = {"size": s.size, "lfs_sha256": sha,
                               "blob_id": s.blob_id}
    return info.sha, remote


def remote_is_current(entry: dict, root: Path, rel: str, sha256: str) -> bool:
    """True when the remote already stores exactly the local bytes.

    LFS objects carry their sha256 in the pointer, so the comparison is
    direct.  Small files live in git, so we compare git's own object name.
    Either way a wrong answer only costs a redundant upload: the signed
    release always carries the locally computed sha256.
    """
    if entry.get("lfs_sha256"):
        return entry["lfs_sha256"] == sha256
    if entry.get("blob_id") and (entry.get("size") or 0) <= (16 << 20):
        return entry["blob_id"] == git_blob_sha1(root / rel)
    return False


def plan_commit(root: Path, payload: dict, remote: dict, *, prune: bool = False,
                include_meta: bool = False) -> tuple[list, list, list]:
    """(paths to upload, remote paths to delete, remote paths not in release).

    `orphans` are release-eligible files the remote holds and the local tree
    does not.  Left alone they sit in the published repo outside the release
    signature — exactly the "silent addition" the release manifest exists to
    rule out — so `publish` refuses on them unless told what to do.
    """
    upload, delete = [], []
    for rel, meta in sorted(payload["files"].items()):
        entry = remote.get(rel)
        if entry is None or not remote_is_current(entry, root, rel, meta["sha256"]):
            upload.append(rel)
    listed = set(payload["files"])
    orphans = sorted(p for p in remote
                     if p not in listed and p != RELEASE_FILE
                     and not is_excluded(p, include_meta=include_meta))
    if prune:
        delete = list(orphans)
        orphans = []
    return upload, delete, orphans


def is_excluded(rel: str, *, include_meta: bool = False) -> bool:
    """Would iter_release_files() have skipped this relative path?"""
    parts = rel.split("/")
    if any(p in EXCLUDE_DIRS for p in parts[:-1]):
        if not (include_meta and parts[0] == "source-meta"):
            return True
    name = parts[-1]
    return name in EXCLUDE_NAMES or name.endswith(EXCLUDE_SUFFIXES)


def _is_stale_parent(exc) -> bool:
    """Did the hub reject the push because someone else committed first?"""
    resp = getattr(exc, "response", None)
    status = getattr(resp, "status_code", None)
    if status in (409, 412):
        return True
    text = str(exc).lower()
    return ("parent_commit" in text or "precondition" in text
            or "a commit has happened since" in text)


def cmd_publish(args) -> int:
    try:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    except ImportError:
        print("publish needs huggingface_hub: pip install 'progressive-tensors[hub]'",
              file=sys.stderr)
        return 2
    try:
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:                                    # hub < 0.25
        from huggingface_hub.utils import HfHubHTTPError

    root = args.dir
    if not root.is_dir():
        print(f"{root}: not a directory", file=sys.stderr)
        return 2
    cache = DigestCache(args.cache)
    api = HfApi(token=args.token)

    for attempt in range(1, args.max_attempts + 1):
        parent, remote = read_remote_state(api, args.repo,
                                           repo_type=args.repo_type,
                                           revision=args.branch)
        print(f"attempt {attempt}/{args.max_attempts}: remote {args.repo}@{args.branch} "
              f"is at {parent} ({len(remote)} files)", flush=True)

        payload = build_payload(
            root, release=args.release, repo=args.repo, revision=None,
            include_meta=args.include_source_meta, progress=args.verbose,
            cache=cache, extra={"parent_revision": parent})
        cache.save()
        c = payload["counts"]
        print(f"  local release: {c['files']} files, {c['fragments']} fragments, "
              f"{c['bytes'] / 1e9:.2f} GB "
              f"(hashed {cache.misses}, cached {cache.hits})", flush=True)

        upload, delete, orphans = plan_commit(
            root, payload, remote, prune=args.prune,
            include_meta=args.include_source_meta)
        if orphans:
            stream = sys.stdout if args.allow_remote_extra else sys.stderr
            verb = ("publishing anyway (--allow-remote-extra): they stay in the "
                    "repo UNSIGNED" if args.allow_remote_extra else
                    "REFUSING TO PUBLISH")
            print(f"{verb}: {len(orphans)} files exist on the remote but not in "
                  f"the local release, so the signature would not cover them.",
                  file=stream)
            for o in orphans[:20]:
                print(f"  remote-only: {o}", file=stream)
            if len(orphans) > 20:
                print(f"  … and {len(orphans) - 20} more", file=stream)
            if not args.allow_remote_extra:
                print("  pass --prune to delete them in the same commit, or "
                      "--allow-remote-extra to publish anyway", file=stream)
                return 1

        envelope, fingerprint, blob = sign_release(payload, args.sign_key)
        if args.out:
            args.out.write_bytes(blob)

        print(f"  commit plan: {len(upload)} uploads, {len(delete)} deletions, "
              f"+ {RELEASE_FILE} signed by {fingerprint[:16]}…", flush=True)
        for u in upload[:20]:
            print(f"    upload {u}")
        if len(upload) > 20:
            print(f"    … and {len(upload) - 20} more")
        for d in delete[:20]:
            print(f"    delete {d}")
        if args.dry_run:
            print("--dry-run: nothing pushed")
            return 0

        ops = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(root / rel))
               for rel in upload]
        ops += [CommitOperationDelete(path_in_repo=d) for d in delete]
        ops.append(CommitOperationAdd(path_in_repo=RELEASE_FILE,
                                      path_or_fileobj=blob))
        message = args.message or f"release: {args.release}"
        try:
            commit = api.create_commit(
                repo_id=args.repo, repo_type=args.repo_type, revision=args.branch,
                operations=ops, commit_message=message,
                commit_description=(
                    f"{c['files']} files, {c['bytes']} bytes, one "
                    f"{RELEASE_SCHEMA} signature by {fingerprint[:16]}…\n"
                    f"built against parent {parent}"),
                parent_commit=parent)
        except HfHubHTTPError as e:
            if not _is_stale_parent(e):
                print(f"push failed: {e}", file=sys.stderr)
                return 1
            print(f"  rejected: the remote moved under us (a concurrent writer "
                  f"committed after {parent}). Re-reading and rebuilding.",
                  file=sys.stderr)
            if attempt == args.max_attempts:
                print(f"giving up after {args.max_attempts} attempts — the repo "
                      f"is being written to faster than a release can be built. "
                      f"Quiesce the other publisher and retry.", file=sys.stderr)
                return 1
            continue

        sha = getattr(commit, "oid", None) or getattr(commit, "commit_id", None)
        print(f"PUBLISHED {args.repo}@{sha}")
        print(f"  {c['files']} files, {c['bytes']} bytes ({c['bytes'] / 1e9:.2f} GB), "
              f"{c['fragments']} fragments")
        print(f"  one {RELEASE_SCHEMA} signature by {fingerprint}")
        print(f"  consumers: fq_release.py verify --dir ./segments --complete "
              f"--trust-signer {fingerprint[:16]}")
        return 0
    return 1


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
        label = "UNLISTED" if args.complete else "unlisted"
        print(f"  {label} (not covered by the signature): {x}", file=sys.stderr)
    if len(report["extra"]) > 20:
        print(f"  … and {len(report['extra']) - 20} more unlisted files",
              file=sys.stderr)
    if args.complete and report["extra"]:
        print("--complete: unlisted files are a FAILURE — the release "
              "signature does not cover them, so this tree is not the "
              "published release", file=sys.stderr)
    if args.complete and report["missing"]:
        print(f"--complete: {len(report['missing'])} listed files are absent",
              file=sys.stderr)
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
    b.add_argument("--cache", type=Path, default=None,
                   help="digest cache (size+mtime+ctime keyed); speeds up "
                        "rebuilds, do not use on a tree rewritten in place")
    b.add_argument("--verbose", action="store_true")
    b.set_defaults(func=cmd_build)

    pub = sub.add_parser(
        "publish",
        help="build+sign from a local tree and push the whole set in ONE commit")
    pub.add_argument("--dir", required=True, type=Path,
                     help="LOCAL tree that IS the release (the source of truth)")
    pub.add_argument("--repo", required=True, help="artifact repo id to publish to")
    pub.add_argument("--release", required=True, help="human-readable release name")
    pub.add_argument("--repo-type", default="model",
                     choices=["model", "dataset", "space"])
    pub.add_argument("--branch", default="main", help="branch to commit on")
    pub.add_argument("--message", default=None, help="commit message")
    pub.add_argument("--sign-key", type=Path,
                     default=Path.home() / ".fq_keys/fq_signing.key",
                     help="ed25519 seed (32 bytes); created on demand")
    pub.add_argument("--token", default=None,
                     help="hub token (default: the usual environment/cache)")
    pub.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                     help="rebuild+retry budget when a concurrent writer wins "
                          f"the race (default {DEFAULT_MAX_ATTEMPTS})")
    pub.add_argument("--prune", action="store_true",
                     help="delete remote files the local release does not list, "
                          "in the same commit")
    pub.add_argument("--allow-remote-extra", action="store_true",
                     help="publish even though the remote holds files the "
                          "release does not cover (they stay unsigned)")
    pub.add_argument("--include-source-meta", action="store_true",
                     help="also cover source-meta/ (cached source headers)")
    pub.add_argument("--cache", type=Path, default=None,
                     help="digest cache, so a rebuild after a rejected push "
                          "costs a stat() per file instead of a re-read")
    pub.add_argument("--out", type=Path, default=None,
                     help=f"also write the signed {RELEASE_FILE} here")
    pub.add_argument("--dry-run", action="store_true",
                     help="print the commit plan and stop")
    pub.add_argument("--verbose", action="store_true")
    pub.set_defaults(func=cmd_publish)

    v = sub.add_parser("verify", help="check one signature, then hash the files")
    v.add_argument("--dir", required=True, type=Path, help="local tree to check")
    v.add_argument("--release", type=Path, default=None,
                   help=f"release envelope (default: <dir>/{RELEASE_FILE})")
    v.add_argument("--complete", action="store_true",
                   help="strict: fail if any listed file is absent OR any "
                        "present file is unlisted (no silent additions)")
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
