# Changelog

Notable changes, newest first. Versions follow [PEP 440](https://peps.python.org/pep-0440/)
(`0.1.0a0` is the release named `v0.1.0-alpha`). While the project is in
alpha, **minor versions may break interfaces**; the schema version strings
inside documents (`fq-segment/1`, `fq-policy/2`, …) are the compatibility
contract that does not, and they are versioned separately.

## v0.1.0-alpha — tag-ready (2026-08-10), not yet tagged

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
  Subset files are re-attested locally as `derived-from` (parents pinned by
  digest, signed with your own `--sign-key`) because a subset is a new file
  no publisher signature can cover — so a fetched tree assembles under a
  pinned signer rather than needing `fq_assemble --insecure`.
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
  `verify --complete` is the strict rung: a listed file that is absent and a
  present file the signature does not cover are both non-zero exits, so "no
  silent additions" is something CI can gate on.
- **`fq_release publish`** — release publication that is atomic and
  remote-state-safe. Uploading a release file-by-file, which is what every
  hub client does by default, walks the *published* repository through
  hundreds of states no signature describes and lets a second writer
  interleave commits into the middle of it. `publish` reads the remote HEAD,
  builds and signs the release from the local tree, and pushes every changed
  file plus `fq-release.json` in ONE `create_commit` with
  `parent_commit=<that HEAD>`: a concurrent writer causes a rejected push,
  and the tool re-reads, rebuilds and retries within a bounded budget. Only
  bytes the remote does not already hold are uploaded (LFS pointers compare
  by sha256, git blobs by object name), an opt-in digest cache keeps a
  rebuild after a lost race to a stat() per file, and publishing refuses
  outright when the remote holds release-eligible files the local release
  does not cover — `--prune` deletes them in the same commit.
- **[TRUST.md](TRUST.md)** — the trust model: what each rung proves, what it
  does *not*, and what an attacker with full control of the artifact
  repository can and cannot do under fingerprint pinning.
- **JSON Schemas** in [`schemas/`](schemas/) for `fq-segment/1` (segment
  metadata and index), `fq-attestation/1`, `fq-manifest/1`, `fq-policy/2`,
  `fq-cartridge/1`, and `fq-release/1` — derived from real emitted artifacts
  and re-validated against freshly emitted documents on every CI run.
- **Packaging** — `pyproject.toml` with console entry points (`fq-repack`,
  `fq-assemble`, `fq-fetch`, `fq-prime`, `fq-verify`, `fq-release`,
  `fq-eps`), a hashed universal dev lock (`requirements-dev.txt`), and
  GitHub Actions CI running the suite on ubuntu-latest and macos-latest for
  Python 3.11 / 3.12 / 3.13, plus wheel-build and trust-root jobs.
- **MSRT cartridge tools** — `fq-assemble-lora` creates a complete EXL3 base
  checkpoint plus validated, sharded full-rank residual cartridges;
  `fq-combine-cartridges` validates and combines separately encoded stages;
  `fq-measure-mse-fruit` compares the actual SIQ checkpoint and MSRT variants
  in original weight space. These custom cartridges are explicitly not
  standard PEFT/LoRA adapters and require an EXL3 MSRT-aware runtime.
- **[docs/PRIOR-ART.md](docs/PRIOR-ART.md)** — commissioned independent
  prior-art review, and the single narrow claim this project makes.

### Changed

- **Signature verification is verification.** A payload that merely decodes
  is no longer treated as checked: wrong key id, malformed base64, a
  signature that is not 64 bytes, or a bad signature are all hard failures.
  (`AA==` as a signature is now a regression test.)
- **README honesty pass** — maturity stated per component instead of one
  blanket label (segments/assembly/verification: heavily verified, with the
  evidence; runtime loader and live reallocation: experimental; artifact
  repo: not public yet; schemas: v1, not frozen), with the TLDR and
  quickstart kept above the fold and depth pushed into `docs/`; the four
  tiers named, including that K4 comes from *primed community fragments*
  rather than our own encodes; `fq-segment/1`
  described as schema v1, versioned and frozen once CI and verification
  ship, rather than a "stable API"; the shard-count contradiction reconciled
  against the measurement (all 76 MoE shards rebuilt from segments — 278.5 GB
  of expert bytes — with dense layers and embed/head passing through);
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

- **A base install shipped a broken command.** The wheel declares `fq-eps`,
  but `fq_eps` imported NumPy at module scope while NumPy lives in the
  `[numeric]` extra, so `pip install progressive-tensors` produced a command
  that died with `ModuleNotFoundError` the first time it ran. NumPy is now
  imported on demand inside the functions that need it; a base install gets
  working `--help` and a diagnostic naming the extra (exit 2). CI was hiding
  this by installing the wheel and NumPy in one step — it now builds a bare
  wheel, asserts NumPy is absent, and runs every console entry point.
- **GitHub Actions were pinned to mutable major tags** (`actions/checkout@v4`,
  `astral-sh/setup-uv@v5`, `actions/upload-artifact@v4`). A tag is a pointer
  its owner can move, and this workflow checks out the repository and runs
  beside release artifacts on `v*` tags. All three are pinned to commit
  SHAs, with the release recorded in a comment; `permissions: contents: read`
  added.
- **The published artifact repository had MIT metadata and no licence
  text.** `LICENSE` (scoped to our contribution) and `NOTICE` (the full
  attribution chain — GLM-5.2/Zhipu AI, the brandonmusic and willfalco
  source quants, exllamav3, safetensors, each pinned by revision) are now
  published with the segments, and the model card links both.
- **The model card told arriving users to `hf download` the whole 481 GB
  repository.** Replaced with a commit-pinned release tag plus per-recipe
  `--include` patterns and the measured disk cost of each (all-K3 279 GB,
  fast-load K2 269 GB, hot-K5 298 GB, primed-K4 294 GB, the K2 tier alone
  74 GB), and a correction of the K2 coverage claim, which was a fixed layer
  range in a card while the campaign kept extending it — coverage now points
  at `fq-manifest.json` `per_k`, which is rebuilt from the published
  inventory.
- Tracked `__pycache__/*.pyc` build artifacts removed from the index; the
  ignore list widened to cover packaging, venv and test-cache paths.

### Known gaps

Stated rather than shipped:

- `fq-release/1` describes **one commit**, and the GLM-5.2 campaign
  supervisor publishes incrementally, so `main` on that repository is
  normally ahead of the last release manifest. `verify --complete` against
  `main` will report the newer segments as unlisted; pin `--revision` to a
  release commit or its tag for completeness to mean anything. Freshness is
  still unaddressed — a replayed older release verifies perfectly.
- One `--trust-signer` applies to a whole `fq_fetch` run: per-source
  pinning for multi-provider fetches is designed but not implemented.
- Attestation envelopes are ad-hoc `{payload, signature, keyid}`; migration
  to DSSE/in-toto and OpenSSF Model Signing compatibility is the intended
  direction, and transparency-log inclusion proofs are what would close the
  replay-an-old-release gap that pinning alone cannot.
- Countersignatures (a second party attesting they re-derived a fragment)
  are anticipated by the JSON Lines attestation format but not implemented.
