# Changelog

Notable changes, newest first. Versions follow [PEP 440](https://peps.python.org/pep-0440/)
(`0.1.0a0` is the release named `v0.1.0-alpha`). While the project is in
alpha, **minor versions may break interfaces**; the schema version strings
inside documents (`fq-segment/1`, `fq-policy/2`, …) are the compatibility
contract that does not, and they are versioned separately.

## v0.1.0-alpha — unreleased

First tagged state. A working research system: honest about what is
measured, what is implemented, and what is not.

### Added

- **`fq_fetch`** — consumer-side range fetch. A recipe (`fq-policy/2`) plus
  one or more `--source repo[@rev]` becomes coalesced HTTP Range reads of
  exactly the expert byte spans the recipe names, written into a local
  segment tree that `fq_assemble` consumes unchanged. Multi-source and
  ordered; content-hash selection (`--prefer-sha`) and an `fq-select/1`
  per-expert provider map; per-expert verification against the source's own
  signed attestation before anything is finalized; resumable with plan
  digests so a changed recipe never resumes into a stale partial;
  `--dry-run` reports ranged bytes vs whole segment files vs whole repo.
- **Trust root** — `keys/FINGERPRINTS` publishes the authorized ed25519
  signer fingerprints *in this git repository*, so a key's authority comes
  from reviewable commit history rather than from the artifact download.
  `tools/fq_trust.py` implements `--trust-signer` pinning (full fingerprint,
  ≥16-hex prefix, key id, or `.pub` path) with four explicitly named rungs
  — pinned, trust-root, unpinned, none — and every tool prints which one it
  used. `keys/check_fingerprints.py` guards the trust root's shape in CI.
- **`fq_release`** — `fq-release/1`: one ed25519 signature over a document
  listing sha256 and size for every file of a release, including the
  indexes and the attestation files. A consumer verifies one signature and
  then hashes, instead of trusting N independent attestation lines with no
  statement of completeness. Partial (range-fetched) trees are first-class.
- **[TRUST.md](TRUST.md)** — the trust model: what each rung proves, what it
  does *not*, and what an attacker with full control of the artifact
  repository can and cannot do under fingerprint pinning.
- **JSON Schemas** in [`schemas/`](schemas/) for `fq-segment/1` (segment
  metadata and index), `fq-attestation/1`, `fq-manifest/1`, `fq-policy/2`
  and `fq-release/1` — derived from real emitted artifacts and re-validated
  against freshly emitted documents on every CI run.
- **Packaging** — `pyproject.toml` with console entry points (`fq-repack`,
  `fq-assemble`, `fq-fetch`, `fq-prime`, `fq-verify`, `fq-release`,
  `fq-eps`), a hashed universal dev lock (`requirements-dev.txt`), and
  GitHub Actions CI running the suite on ubuntu-latest and macos-latest for
  Python 3.11 / 3.12 / 3.13, plus wheel-build and trust-root jobs.
- **[docs/PRIOR-ART.md](docs/PRIOR-ART.md)** — commissioned independent
  prior-art review, and the single narrow claim this project makes.

### Changed

- **Signature verification is verification.** A payload that merely decodes
  is no longer treated as checked: wrong key id, malformed base64, a
  signature that is not 64 bytes, or a bad signature are all hard failures.
  (`AA==` as a signature is now a regression test.)
- **README honesty pass** — an experimental-alpha banner; the artifact repo
  described as prepared and private rather than published; `fq-segment/1`
  described as schema v1, versioned and frozen once CI and verification
  ship, rather than a "stable API"; the 79-vs-76 shard count reconciled (79
  quantized layer shards, 76 rebuilt from segments, 3 dense pass-through);
  the deduplication claim scoped to a segment family, since two independent
  K4 encodes of one expert differ (measured cosine 0.9934); `encode-of`
  qualified with its determinism scope, with `equivalence-of` named as the
  honest cross-stack rung; the attestation snippet made line-iterating and
  pinned to the git-published fingerprint; the quickstart now states the
  ~295 GB source-checkpoint requirement and fetches ranges instead of
  recommending a 347 GB `hf download`.
- **`assets/make_charts.py`** takes `--analysis`, `--work-root`,
  `--fq-eps-dir` and `--out` instead of hardcoded absolute paths into
  private trees. Defaults reproduce the published SVGs byte-for-byte on the
  build machine (modulo matplotlib's random element ids).

### Fixed

- Tracked `__pycache__/*.pyc` build artifacts removed from the index; the
  ignore list widened to cover packaging, venv and test-cache paths.

### Known gaps

Stated rather than shipped:

- The GLM-5.2 artifact repository is private; publication is gated on
  verification enforcement landing.
- One `--trust-signer` applies to a whole `fq_fetch` run: per-source
  pinning for multi-provider fetches is designed but not implemented.
- `--trust-signer` is not yet wired into `fq_assemble` and `fq_prime`.
- Attestation envelopes are ad-hoc `{payload, signature, keyid}`; migration
  to DSSE/in-toto and OpenSSF Model Signing compatibility is the intended
  direction, and transparency-log inclusion proofs are what would close the
  replay-an-old-release gap that pinning alone cannot.
- Countersignatures (a second party attesting they re-derived a fragment)
  are anticipated by the JSON Lines attestation format but not implemented.
