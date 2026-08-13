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
  `fq-cartridge/2`, `fq-cartridge-adapter/3`, `fq-cartridge-assembly/2`, and
  `fq-release/1` — derived
  from real emitted artifacts and re-validated against freshly emitted
  documents on every CI run.
- **Packaging** — `pyproject.toml` with console entry points (`fq-repack`,
  `fq-assemble`, `fq-fetch`, `fq-prime`, `fq-verify`, `fq-release`,
  `fq-eps`), a hashed universal dev lock (`requirements-dev.txt`), and
  GitHub Actions CI running the suite on ubuntu-latest and macos-latest for
  Python 3.11 / 3.12 / 3.13, plus wheel-build and trust-root jobs.
- **MSRT cartridge campaign tools** — `fq-assemble-lora` encodes a whole
  `fq-cartridge/2` graph in one pass over the weights: every declared base
  tier becomes a complete EXL3 checkpoint, and every stage is a rescaled
  trellis residual against the reconstruction of the `parent` it names, so
  seven loadable products spanning 26 bits per weight cost seven quantization
  passes emitting 12 (measured on real GLM-5.2 experts: 1.58x less trellis
  kernel time and 2.17x fewer bytes than encoding each product separately; the
  nine-product graph is 1.83x and 2.5x). Subcommands
  `plan` / `skeleton` / `encode` / `finalize`; reads standard indexed Hugging
  Face shards or per-layer shards without ever loading a whole shard; work is
  addressed as (layer, 32-expert block), owned by an `flock` the kernel releases
  however the owner dies, and committed as one atomic unit, so an interrupted or
  preempted run resumes at block granularity and `--devices` runs one worker per
  GPU over disjoint blocks. One campaign directory takes one launcher and one
  signing key, both enforced rather than advised, and publication cannot overlap
  encoding. `fq-combine-cartridges` turns one published
  assembly plan into a self-contained `fq-cartridge-adapter/3` cartridge under
  a pinned signer, narrowing a full-expert stage to the experts a consumer
  actually wants — decided from the signed plan before any payload is read, and
  checked against the base checkpoint it will be loaded onto.
  `fq-measure-mse-fruit` compares the actual SIQ checkpoint against every
  graph node through the production encoder itself. `fq-promote-campaign`
  publishes a staged campaign to `main` in a single commit of server-side
  copies, after verifying the branch carries every file the finalized campaign
  holds. These custom cartridges are explicitly not standard PEFT/LoRA adapters
  and require an EXL3 MSRT-aware runtime.
- **Source-byte provenance is one transaction** — skeleton repacks and raw
  encoder reads copy and SHA-256 one `O_NOFOLLOW` regular-file fd into private
  `0600` staging, validate the inode before/after, deserialize only staged
  bytes, and attest that observed digest directly. Hub/manifest digests and the
  resume cache are strict expectations rather than substitutes for
  observation; symlinks, FIFOs, nonregular files, source drift and stale cache
  entries are refused.
- **Versioned MSRT runtime binding** — the pre-merge closed
  `fq-cartridge-assembly/2` and `fq-cartridge-adapter/3` contracts bind an
  ordered residual chain to exact per-layer logical base identities plus a
  TP-layout-invariant family root. Adapter/3 fixes base-owned rotations,
  packed int16 trellis plus scalar float32 scale semantics, and an explicit
  full-vs-rank-sharded layout/rank/axis map (unambiguous even at world size
  one). Every shard carries size and SHA-256; producer/runtime share config,
  shard, and total-size limits. `producer_verified_signer` records combiner
  provenance only, not runtime authentication. The paired vLLM loader rejects
  unversioned, incomplete, tampered, wrong-base, or wrong-TP cartridges before
  tensor deserialization.
- **Two recipes for GLM-5.2, priced against each other.**
  `recipes/glm52-k2k3-dag.json` ships nine products including two that sell a
  +1-bit upgrade to an installed 3 or 4 bpw tier;
  `recipes/glm52-k2k3-lean.json` drops those two. Measured on 168 comparisons
  across real GLM-5.2 layers and on all 88 blocks of a proxy rehearsal, the
  narrow-step path they serve is 9.0% (K2 family) and 6.8% (K3) worse than
  fetching the wider residual at the same bitrate, while costing twice the
  kernel time — so the lean graph is the recommended rental: **132.6 GPU-hours
  against 186.9**, both measured back to back on one RTX 5090.
- **[tools/msrt_campaign.sh](tools/msrt_campaign.sh)** — the campaign as one
  checked driver: window staging with last-use source retention, skeleton,
  encode, retirement, and a `PHASE=finalize` pass for a CPU machine. Refuses a
  `WINDOWS` list that does not cover the recipe exactly once, refuses to start
  without an explicit `DEVICES`, and refuses to let the GPU fleet be released
  until every prerequisite for finishing the campaign is on one persistent
  filesystem.
- **Signed provenance for every encoded fragment.** Each shard ships a
  `fq-attestation/1` line beside it: `encode-of` for expert shards, naming the
  sha256 of each expert's contiguous byte range, the encoder bundle (Python
  modules *and* the compiled extension), the determinism scope, the effective
  quant arguments, and the exact parent shard digest the residual corrects;
  `repack-of` for skeleton shards, naming per-tensor digests and the source
  file the bytes were copied from. Shard payloads carry no timestamp, so
  re-encoding inside the declared scope reproduces them byte for byte.
  `finalize` re-hashes all of it before publishing anything and refuses a
  campaign that spans two signers or two encoder builds, whose stages do not
  name the parents this campaign published, or that holds shards the recipe
  does not describe. It is also resumable: a preemption during that
  multi-terabyte pass costs the re-read, not the campaign.
- **[docs/MSRT-CAMPAIGN.md](docs/MSRT-CAMPAIGN.md)** — the GLM-5.2 campaign
  runbook: per-K trellis cost, both graphs measured as full 32-expert blocks on
  real GLM-5.2 weights, the resulting 133 GPU-hour / 1.147 TB projection with
  its measured/derived/unmeasured labels, exact window geometry from the pinned
  index, fleet sizing, the gates to run before renting, and the resume, publish
  and verification procedure — rehearsed end to end on a proxy.
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
