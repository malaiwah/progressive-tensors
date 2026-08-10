# Trust model

**Short version.** Get the signer fingerprint from *this git repository*
([`keys/FINGERPRINTS`](keys/FINGERPRINTS)), never from the artifact
download. Pass it to every tool as `--trust-signer`. Then the worst a
compromised artifact repository can do is refuse to serve you bytes — it
cannot make you accept the wrong ones.

```bash
# once, from a source the artifact host does not control
git clone https://github.com/malaiwah/progressive-tensors && cd progressive-tensors
cat keys/FINGERPRINTS
#  a58b7bb79ba5845716aa6fee7d54e714ef243c2875f23a617e1ef3247c565525  malaiwah-fq-1  active …

# thereafter, on every fetch and every verification
fq_fetch.py --policy recipe.json --out ./segments \
    --source malaiwah/GLM-5.2-EXL3-FQ-segments@<commit> \
    --trust-signer a58b7bb79ba58457            # 16-hex short form is fine
```

This document says what that buys, and — more importantly — what it does
not.

---

## 1. The problem this replaces

Until now the README told you to read `signer_pubkey` out of
`fq-manifest.json`, which arrives in the same unauthenticated download as
the fragments and the attestations. That is a self-consistency check
wearing a signature's clothes. An attacker who can write to the artifact
repository (stolen token, compromised account, malicious mirror, hostile
host) replaces:

1. the segment bytes,
2. the attestation lines that pin their digests,
3. the public key in the manifest,

signs (2) with their own key, and every verification passes. The
cryptography was never the weak part; the *key distribution* was.

A trust root has to live somewhere the artifact host cannot rewrite. Ours
is this git repository: a different system, a different account, a public
append-mostly history that other people have already cloned. Forging it
means forging a commit in a repo whose history is mirrored on every
clone — loud, and after the fact, provable.

## 2. The rungs, and what each one actually proves

Every tool prints which rung it used. They are not interchangeable.

| Rung | How you get it | What it proves | What it does **not** prove |
|---|---|---|---|
| **pinned** | `--trust-signer <fingerprint>` from git | These bytes were signed by *that specific key*. A repo compromise cannot pass. | That the key's owner is honest, competent, or that the *upstream quant* is any good. |
| **trust-root** | no flag: any key listed `active` in `keys/FINGERPRINTS` | Signed by *some* key this project authorized, as of your clone. | Which key you expected. If the project ever has several signers, you have not said which. |
| **unpinned** | `--allow-unpinned-signer` | The bundle is internally consistent. | Anything about provenance. A rewritten repo passes trivially. This rung exists so you can say "I know, I am just looking". |
| **none** | `--insecure-skip-signatures` | Nothing. | Nothing. For offline fixtures and tests. |

And below the signature layer, three *content* proofs that are separate
from who signed them:

| Proof | Mechanism | Establishes | Does not establish |
|---|---|---|---|
| **fragment digest** | `expert_sha256` in an attestation vs the bytes you fetched | The bytes you received are the bytes the signer committed to. | That those bytes came from the upstream quant they claim. |
| **materials pin** | `materials.repo` + `revision` + `file_sha256` | Which upstream file the signer says these bytes were copied from — checkable by anyone with a ranged read of that repo. | That the signer actually did the copy, until someone re-does the ranged read and compares (this is what `fq_verify --identity` automates). |
| **release completeness** | one `fq-release/1` signature over every file's digest | *Which set of files is the release*, so nothing can be added, dropped, or rolled back silently — **at the commit the release names**. | That the release is a good release, or that the branch head is still that commit. See §3.1. |

Three things nothing here establishes, ever, and no amount of signing will:

- **Attestation ≠ endorsement.** A signature says who assembled the bytes.
  It says nothing about whether the quantization is good, whether the base
  model is safe, or whether the calibration was honest.
- **Provenance chains terminate in reputation.** `repack-of` proves these
  bytes equal *that* upstream quant's bytes. Whether *that* quant is
  trustworthy is outside the system. We make the terminus explicit and
  checkable rather than hidden.
- **`encode-of` is not bit-reproducibility across stacks.** Re-running a
  quantizer reproduces bytes only inside a declared determinism scope
  (same GPU arch, same library versions, same batch shapes). Measured
  counter-examples exist: 1-ulp differences in CUDA `pow` for rotary
  `inv_freq`, row-order instability in sdpa/grouped_mm across batch
  shapes. Outside that scope the honest predicate is `equivalence-of`
  (measured numeric similarity), not identity — and attestations say which
  they mean.

## 3. What a repo compromise can and cannot do, under pinning

Assume the attacker fully controls the Hugging Face repository and the CDN
in front of it. You have a clone of this git repo from before the
compromise, and you pass `--trust-signer`.

**They cannot:**

- serve fragments you will accept — every fetched expert span is hashed
  against a digest inside a payload signed by the pinned key
  (`fq_fetch`, before any file is finalized);
- forge an attestation — a wrong `keyid` is refused before any signature
  arithmetic, and a wrong signature under the right `keyid` fails
  verification;
- pass off a placeholder — a signature that is not exactly 64 bytes is a
  failure, not a pass (`AA==` is a regression test, because it once
  wasn't);
- add, drop, or roll back files inside a release without detection, when
  the release publishes an `fq-release/1` manifest: the signed file list
  is exhaustive, absent files are reported, unlisted files are reported as
  uncovered;
- substitute a *different* signed-but-real fragment for the one your
  recipe asked for — the digest is per expert, per layer, per K.

**They can still:**

- **deny service** — serve nothing, serve corrupt bytes, be slow. Integrity
  is not availability;
- **replay an older signed release** — everything verifies, it is simply
  stale. Mitigation: pin `@<commit>`, and compare the release name and
  `created_utc` against what you expect. There is no online freshness
  service here, deliberately;
- **learn what you fetch** — ranged reads reveal which experts your recipe
  values. Privacy is not part of this model;
- **win completely if the signing key leaked.** Pinning binds bytes to a
  key; it cannot tell you the key is still in the right hands. That is what
  revocation is for, and revocation is a `git pull` away, which means your
  window of exposure is the staleness of your clone;
- **win if you never pin.** Every guarantee above is conditional on the
  fingerprint coming from outside the artifact repo. The default
  (trust-root rung) already refuses unknown keys, but it uses whatever your
  clone says today.

A subtle one worth stating: **pinning does not make an unpinned revision
safe.** `--source repo` without `@<commit>` follows `main`, so the
publisher (or whoever holds their token) can change the bytes under you
between runs; the signature still verifies because they re-sign. Pin the
revision *and* the signer. `fq_fetch` warns when you do not.

### 3.1 The release manifest describes one commit, not a branch

`fq-release/1` says "these 353 files, these digests, are the release". That
statement is scoped to **one commit**, and two things follow that are easy
to get wrong.

**Publication has to be atomic or the statement is not true of anything.**
A publisher that uploads file-by-file — the default of every hub client —
walks the repository through hundreds of intermediate states, none of which
any signature describes, and lets a second writer land commits in the
middle. `fq_release.py publish` therefore reads the remote HEAD, builds and
signs from the *local* tree, and pushes the whole coherent set in one
`create_commit` with `parent_commit=<that HEAD>`. A concurrent writer makes
the push fail, not interleave; the tool re-reads, rebuilds and retries
within a bounded budget. A rejected push is the mechanism working.

**A branch head is not a release.** Our own GLM-5.2 campaign supervisor
publishes *incrementally*: it uploads each encode window as it finishes, and
a release manifest is built afterwards. So on that repository `main` is
usually **ahead of** the last `fq-release.json`, and the extra segments are
real, individually attested, and *not covered by the release signature*.
`fq_release.py verify --complete` against `main` will correctly report them
as unlisted and exit non-zero. That is not a failure of the artifacts; it is
the completeness check telling you that "the branch" and "the release" are
different things. Pin `--revision` to a release commit (or its tag) if you
want `--complete` to mean something.

One consequence worth naming before it alarms someone: an incremental
publish also **rewrites `fq-manifest.json`** (deliberately — it is rebuilt
from the live inventory so it always describes what is actually published).
Verified against an older release, that file therefore reports as
`MISMATCHED`, not merely unlisted. It is a newer manifest, not a tampered
one; the signature cannot distinguish those two and must not pretend to.
This is what "the release is a commit, not a branch" costs in practice.

What `fq-release/1` still does not give you, in either mode, is
**freshness**: a replayed older release verifies perfectly and is simply
stale. Compare `release`, `created_utc` and `parent_revision` inside the
document against what you expected. Transparency-log inclusion proofs are
the real fix and are not implemented (§6).

## 4. The chain, end to end

```
git history  ──►  keys/FINGERPRINTS  ──►  --trust-signer <fingerprint>
(out of band)     (reviewed commits)      (what you pin)
                                              │
                                    ed25519 verification
                                              │
                     ┌────────────────────────┴───────────────────────┐
                     ▼                                                ▼
            fq-release/1 (one signature)                 fq-attestation/1 (per fragment)
            lists sha256 of EVERY file       ─binds─►    expert_sha256: per-expert digests
                     │                                                │
                     ▼                                                ▼
            hash the files you hold                        hash the expert span you fetched
                     │                                                │
                     └────────────────────►  bytes you can use  ◄─────┘
                                                     │
                                                     ▼
                                      fq_assemble → checkpoint, whose
                                      shard digests you can compare to
                                      the upstream quant (fq_verify)
```

One hop the diagram compresses: **a range-fetched subset is re-attested
locally.** The experts inside it are verbatim and were hashed against the
publisher's signed digests as they arrived, but the *file* is new — fewer
experts, different offsets, a digest no publisher ever signed. So `fq_fetch`
signs what it actually produced with a local key (`--sign-key`), as a
`derived-from` attestation naming the publisher fragments as parents and
pinning them by digest; `fq_assemble --trust-signer <your fingerprint>` then
verifies that. The publishers' own lines are kept under
`attestations/<source>/` so the upstream hops stay checkable offline. Your
key is the honest signer for that last hop: nobody else can truthfully
attest a file only your machine assembled.

Two paths into the same place. The release manifest is the cheap one: one
signature, then hashing. The per-fragment attestations are the granular
one: they carry the per-expert digests a ranged read needs, and they
survive being fetched individually. When both are present, the release
manifest also covers the attestation *files*, so their contents inherit
that single signature instead of needing N verifications.

## 5. Practical guidance

**Pin the fingerprint once, out of band.** A clone, a copy in your
deployment config, a line in your Ansible vars — anywhere that is not the
artifact download. 16 hex characters is enough to type; the tools accept a
prefix of ≥16 and refuse shorter ones rather than matching loosely.

**Pin the revision too.** `--source repo@<commit>`.

**Re-check what you already have.** `fq_release.py verify --dir ./segments
--trust-signer <fp>` re-hashes a tree you downloaded weeks ago. It is
cheap and it is the only way to notice bit-rot or a swapped file. Add
`--complete` when you pulled the whole release: it then exits non-zero both
on a listed file that is absent and on a local file the signature does not
cover, so "nothing was dropped and nothing was added" is something a script
can gate on rather than a line of output somebody has to read.

**When a key rotates**, the old line in `keys/FINGERPRINTS` becomes
`retired` (signatures made before retirement stay valid — tools warn) and
a new `active` line appears. When a key is **revoked**, treat every
signature by it as unproven regardless of date: a leaked key can backdate.
`git log -p keys/FINGERPRINTS` is the complete history of who was ever
allowed to sign, and it is the same history everyone else can read.

**If you are publishing fragments yourself**, sign with your own key and
publish your own fingerprint somewhere your artifact host cannot rewrite.
`fq_fetch` verifies each expert against *the attestation of the source it
came from*, so a multi-provider fetch is a multi-trust-root operation by
construction; per-source pinning (rather than one `--trust-signer` for the
whole run) is the obvious next step and is **not implemented yet** — today
one pinned fingerprint applies to every source in a run, which means
mixing providers means trusting one key for all of them, or dropping to
the trust-root rung.

## 6. Where this is going

The cryptographic plumbing here is deliberately boring, and it should not
stay ours. The direction of travel is:

- **DSSE / in-toto envelopes** instead of our ad-hoc
  `{payload, signature, keyid}` envelope, so attestations are consumable by
  existing supply-chain tooling and the statement/predicate split is the
  standard one;
- **OpenSSF Model Signing compatibility** (sigstore/model-transparency), so
  signing identity can be an OIDC identity with a transparency log rather
  than a bare ed25519 key we ask you to pin by hand;
- **transparency-log inclusion proofs**, which is the real answer to the
  replay-an-old-release gap that pinning alone cannot close.

Our contribution is not the signature format — it is the quant-segment
schema and the linker that turns attested fragments into a bootable
checkpoint. The signature layer should be the ecosystem's. Until that
migration lands, `keys/FINGERPRINTS` + `--trust-signer` is a small, honest
trust root that does the one thing the previous design did not: put the key
somewhere the artifacts cannot reach.

## 7. Status

**Implemented and tested** (`tests/test_fq_trust.py`,
`tests/test_fq_release.py`, `tests/test_fq_fetch.py`):

- `keys/FINGERPRINTS` as the trust root, with a CI guard on its shape;
- `--trust-signer` / `--trust-root` / `--allow-unpinned-signer` /
  `--insecure-skip-signatures` in `fq_fetch` and `fq_release`, with the
  rung printed; `fq_assemble` pins with `--trust-signer` / `--trust-file`
  (it reads this repo's `keys/FINGERPRINTS` format directly) and refuses to
  assemble anything it cannot verify;
- real ed25519 verification with the failure modes enumerated above;
- `fq-release/1` build and verify, including partial trees, and
  `--complete` as a strict rung where a listed-but-absent file *and* an
  unlisted-but-present file are both non-zero exits;
- `fq-release/1` **atomic publication** — one `create_commit` pinned to the
  parent HEAD, a bounded rebuild-and-retry when a concurrent writer wins the
  race, and a refusal to publish while the remote holds release-eligible
  files the release does not cover (§3.1);
- signature verification in `fq_prime spot-check`: every attestation line is
  verified against the pinned key before its `expert_sha256` is used, and a
  missing, unreadable or silent attestation is a failure rather than a pass;
- per-expert digest checking on every fetched byte range;
- local `derived-from` re-attestation of fetched subsets, so a
  range-fetched tree assembles under a pinned signer instead of needing
  `fq_assemble --insecure`.

**Not yet:**

- per-source pinning in a multi-provider fetch (see §5);
- **release freshness.** Nothing here detects a replayed older release;
  transparency-log inclusion proofs are the answer and are not built.
  Until then, compare `created_utc` / `parent_revision` yourself (§3.1);
- DSSE/in-toto envelopes and transparency-log inclusion proofs (§6);
- countersignatures — a second party attesting "I re-derived this fragment
  and got the same bytes". The attestation files are JSON Lines precisely
  so that additional signed lines can be appended later; nothing consumes
  them yet.
