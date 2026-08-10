# Progressive Tensors

**Mixed-K quants as a service — assemble your own EXL3 checkpoint from
shared, attested, per-expert segments. Pure safetensors. No new format.**

> ### ⚠️ Experimental alpha — read this before you plan anything around it
>
> **Status: `v0.1.0-alpha`.** This is a working research system, not a
> product. Concretely:
>
> - **The artifact repository is still private.** The GLM-5.2 segment tree
>   exists and is verified locally, but
>   [`malaiwah/GLM-5.2-EXL3-FQ-segments`](https://huggingface.co/malaiwah/GLM-5.2-EXL3-FQ-segments)
>   returns 401 today. Publication is deliberately gated on the verification
>   work landing — we are not shipping 347 GB that consumers cannot check.
>   The commands below are the real ones; they will work when it opens.
> - **The schemas are v1, versioned but not frozen.** They freeze when CI
>   and verification enforcement have shipped and been exercised against
>   published artifacts. Until then expect additive fields (see
>   [`schemas/`](schemas/)).
> - **Interfaces may move.** Flags, output layouts and predicates are still
>   being shaped by what verification turns out to need.
>
> What *is* solid: the byte-level properties, and the measurements. Those
> are stated below with what was actually measured, and the tests that hold
> them are in [`tests/`](tests/).

The community publishes many quantizations of the same MoE model — a flat
3.0 bpw here, a mixed 3.25 bpw there, a 3.42 bpw flagship — each a
multi-hundred-GB download, each an all-or-nothing choice. Progressive
Tensors decomposes them into their real unit of value: **the per-expert
encoded fragment**. One shared K3 base everyone downloads once; per-expert
K4/K5 enhancement fragments fetched (or encoded) as needed; any mix
reassembled into a bootable checkpoint with the recipe *you* choose.

Think progressive JPEG, for quants.

- **Every file is plain safetensors** — readable by any safetensors tool.
  "Progressive" is the fetch/assembly policy *across* files, not a
  container format.
- **Every fragment is content-addressed and signed** — an `fq-attestation/1`
  line pins the fragment's sha256, its source repo at an exact commit, and
  per-expert digests, so any fragment is independently spot-checkable with
  one ranged read. The signing key's authority comes from
  [`keys/FINGERPRINTS`](keys/FINGERPRINTS) **in this git repo**, not from
  the artifact download — see [TRUST.md](TRUST.md).
- **Reassembly is byte-faithful** — assembling the all-K3 recipe from the
  brandonmusic-derived segments reproduces the original checkpoint shards
  **sha256-identical**: re-verified across **all 76 MoE shards** of GLM-5.2
  (layers 3–78; 278.5 GB of expert bytes came from segments, the dense
  layers and embed/head pass through from the source). The full per-quant
  reconstruction proof — byte-identity rungs and measured numeric similarity
  across three community quants, including evidence that two *independent*
  producers' K4 fragments are equal in measured quality — is in
  [docs/RECONSTRUCTION.md](docs/RECONSTRUCTION.md).
- **Fetch only what your recipe uses** — segments are per-expert
  contiguous, so `fq_fetch` HTTP-Range-reads exactly the expert spans a
  recipe names, from one or several publishers, instead of downloading the
  repository.
- **Sharing is deduplicating *within a segment family*** — one published
  set of fragments is one set of bytes, however many recipes reference it,
  and Hugging Face Xet dedupes at chunk level on top. This is a property of
  reusing the *same* fragments, not of the model: two independent K4
  encodes of the same expert are different bytes (measured cosine 0.9934 —
  numerically equivalent, byte-wise unrelated), so they do not dedupe
  against each other and never will.
- **No special loader required** — the assembled output is a normal
  checkpoint for the target runtime (Gilded Gnosis vLLM / EXL3 stack).
  A runtime *progressive loader* and live bit-width reallocation are being
  built on top (see Research), but everything in this repo works today
  with plain files.


## The numbers behind it

<picture><source media="(prefers-color-scheme: dark)" srcset="assets/eps-ladder-dark.svg"><img alt="Per-expert encode error vs bit-width: a clean geometric ladder, ~3.8x lower error per +1 bit (K2 0.0903, K3 0.0231, K4 0.0060, K5 0.0016)" src="assets/eps-ladder-light.svg"></picture>

<picture><source media="(prefers-color-scheme: dark)" srcset="assets/benefit-concentration-dark.svg"><img alt="Cumulative share of K3-to-K4 upgrade benefit vs experts ranked by benefit: strongly concave - the top 16 of 256 experts carry about a third of the total benefit" src="assets/benefit-concentration-light.svg"></picture>

<picture><source media="(prefers-color-scheme: dark)" srcset="assets/k4-allocation-dark.svg"><img alt="K4 experts allocated per layer at a fixed global budget: far from uniform, ranging 42 to 152 across layers" src="assets/k4-allocation-light.svg"></picture>

*All three measured on the GLM-5.2-architecture proxy: one sealed 1.05M-token
capture, four hessian-identical encodes (K2/K3/K4/K5) with the sha-pinned
production encoder. Full campaign data and reports in the research branch.*

## Quickstart: verify + reassemble

Segments for **GLM-5.2** (from
[`brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw`](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw)
at pinned commit `9297b9f1`) live at
[`malaiwah/GLM-5.2-EXL3-FQ-segments`](https://huggingface.co/malaiwah/GLM-5.2-EXL3-FQ-segments).

**What you need on disk.** Assembly writes a complete checkpoint, so
besides the fragments you need the **source checkpoint snapshot** (~295 GB
for GLM-5.2): dense layers, attention, embeddings, `config.json` and the
safetensors header order all come from it — Progressive Tensors only
substitutes the routed-expert bytes. Plan for the source checkpoint plus
your fetched segments plus the assembled output.

```bash
git clone https://github.com/malaiwah/progressive-tensors
cd progressive-tensors
uv venv && uv pip install -r requirements-dev.txt
uv run pytest tests/            # the tools' own test suite

# your recipe: which K per expert, per layer (fq-policy/2 JSON)
python - <<'EOF'
import json
json.dump({"schema": "fq-policy/2",
           "bits_per_expert": {str(l): [3]*256 for l in range(3, 79)}},
          open("recipe-all-k3.json", "w"))
EOF

# what would this cost?  (ranged bytes vs whole files vs whole repo)
uv run tools/fq_fetch.py --policy recipe-all-k3.json --out ./segments \
  --source malaiwah/GLM-5.2-EXL3-FQ-segments@<commit> \
  --trust-signer a58b7bb79ba58457 --dry-run

# fetch ONLY the expert byte ranges the recipe names, verified as they land
uv run tools/fq_fetch.py --policy recipe-all-k3.json --out ./segments \
  --source malaiwah/GLM-5.2-EXL3-FQ-segments@<commit> \
  --trust-signer a58b7bb79ba58457

# assemble a bootable checkpoint (dense tensors come from the source repo).
# Assembly verifies every fragment it consumes — signature under a pinned
# signer, signed fragment and per-expert digests recomputed from the bytes on
# disk — and fails closed if you do not pin one.  fq_fetch prints the exact
# --trust-signer line for the tree it just wrote.
uv run tools/fq_assemble.py \
  --segments ./segments --source <source-checkpoint-dir> \
  --policy recipe-all-k3.json --out ./my-checkpoint \
  --trust-signer <fingerprint>   # or --trust-file keys/FINGERPRINTS

sha256sum ./my-checkpoint/model-layer-030.safetensors
#  -> identical to the original shard. That's the point.
```

**Two fingerprints, one chain.** For the *fetch* step, `--trust-signer` is
the publisher's fingerprint from [`keys/FINGERPRINTS`](keys/FINGERPRINTS) in
*this* repository — the point being that it does not come from the download
you are checking. For the *assemble* step it is **your own**: a fetched
subset is a new file (fewer experts, new offsets, new digest), so no
publisher signature can cover it, and `fq_fetch` signs what it materialized
as `derived-from`, naming the publisher fragments as parents and pinning
them by digest. It prints that fingerprint and the exact assemble command.
Assembling a tree you downloaded whole instead pins the publisher directly.

An all-K3
recipe over all 76 MoE layers is the whole base tier, so it fetches
essentially everything; the savings appear as soon as the recipe is
narrower than the repo (a layer window, or a K4 hot set on top of a K3
base). `--dry-run` always prints the three numbers so you can see which
case you are in before spending bandwidth.

### Disk-saving assembly (`--reflink`)

If segments and output live on a reflink-capable filesystem (XFS with
`reflink=1`, btrfs), add `--reflink` to `fq_assemble.py`: expert-tensor
bytes are then written with `copy_file_range`, letting the kernel share
extents between the segment files and the assembled shards instead of
storing the same bytes twice — *when alignment permits*.

**Measured on XFS (Ceph RBD, kernel 6.8), assembling real GLM-5.2 shards
twice (plain vs `--reflink`):** byte-identity always held (both outputs
sha256-identical to each other and to the original shards), every expert
region went through `copy_file_range` without a single fallback, and the
reflink run was faster (6.3 s → 3.8 s for three shards) — but **zero extents
ended up shared** (`filefrag -v`: no `shared` flags; identical block usage).
Cause, measured: XFS only shares blocks when source and destination offsets
agree mod 4096, and **0.00 % of expert-tensor bytes are 4K-congruent**
between segment and shard offsets — the canonical per-expert reordering plus
differing safetensors header lengths shift every offset. A positive control
(same syscall, aligned offsets, same filesystem) shared extents immediately,
so the limit is alignment, not the kernel or filesystem.

Caveats that remain true everywhere:

- **Sharing is the kernel's call.** `copy_file_range` silently performs a
  plain (server-side) copy when it cannot remap blocks.
- **Output bytes are always identical** to a non-`--reflink` run — the mode
  changes how bytes move, never what they are. Every region falls back
  automatically to the ordinary copy when `copy_file_range` is unavailable or
  fails (cross-filesystem `EXDEV`, `EOPNOTSUPP`, ...), so it is always safe
  to pass.
- **Local disk space only.** No effect on HF/remote transfer or storage (Xet
  chunk-dedupe covers that side).

Net: treat `--reflink` today as "safe, sometimes faster", not as a guaranteed
space saver. A future `fq-segment/2` could pad per-tensor segment offsets into
4K congruence with the source layout (≤ 4 KB per tensor, sub-1 % overhead) if
extent sharing is worth buying.

## Fetch only what the recipe needs (`fq_fetch`)

`fq_fetch` is the consumer half. Given a recipe and one or more source
repos, it reads `index-kK.json`, turns the recipe into per-expert byte
spans, coalesces them, and HTTP-Range-fetches exactly those — into a local
segment tree that `fq_assemble` consumes unchanged.

```bash
# expert-level fragments from several publishers, in priority order
uv run tools/fq_fetch.py --policy hot-k4.json --out ./segments \
  --source malaiwah/GLM-5.2-EXL3-FQ-segments@<commit> \
  --source willfalco/GLM-5.2-EXL3-TR3-3.36bpw-FQ@<commit> \
  --trust-signer a58b7bb79ba58457

# "I want this exact fragment, whoever publishes it"
  --prefer-sha 3c5b0b901ccec3f69836078d5e18b32f54c35887d010b0626d0398cbf05dacea

# or an explicit per-expert / per-layer provider map (fq-select/1)
  --select providers.json
```

- **Ordered sources.** The first source carrying an expert at the required
  K wins; content hash and the provider map override that.
- **Every expert is verified as it lands**, against the signed attestation
  *of the source it came from*, before the file is finalized. One output
  file may legitimately mix publishers;
  `fq-fetch-report.json` records who provided each expert.
- **Resumable.** Interrupt it, re-run the same command. Per-expert progress
  is recorded, partial files resume in place, resumed bytes are re-hashed
  before being trusted, and a changed recipe discards the stale partial
  rather than resuming into it.
- **`--dry-run`** prints ranged bytes vs whole-segment-files vs whole-repo.

### Verify provenance without downloading anything big

Each `attestations/layer-LLL.kK.jsonl` file is **JSON Lines** — one signed
line per fragment covered, and more lines later as countersignatures land —
so iterate lines, never `json.loads` the whole file:

```python
import base64, hashlib, json
from nacl.signing import VerifyKey

# The fingerprint comes from keys/FINGERPRINTS in the GIT repo, NOT from
# the download being checked.  That is the entire point (see TRUST.md).
TRUSTED = "a58b7bb79ba5845716aa6fee7d54e714ef243c2875f23a617e1ef3247c565525"
verify_key = VerifyKey(bytes.fromhex(TRUSTED))

digests = {}
for line in open("segments/attestations/layer-030.k3.jsonl"):
    if not line.strip():
        continue
    env = json.loads(line)
    assert env["keyid"] == TRUSTED, f"signed by an unexpected key: {env['keyid']}"
    raw = base64.b64decode(env["payload"])
    verify_key.verify(raw, base64.b64decode(env["signature"]))  # raises if bad
    payload = json.loads(raw)
    if payload["fragment"]["file"] == "layer-030.k3.safetensors":
        digests.update(payload["expert_sha256"])

# spot-check ONE expert with one ranged read against the segment file:
idx = json.load(open("segments/index-k3.json"))["30"]
lo, hi = idx["experts"]["137"]
blob = open(f"segments/{idx['file']}", "rb").read()[
    idx["body_offset"] + lo : idx["body_offset"] + hi]
assert hashlib.sha256(blob).hexdigest() == digests["137"]
```

The `materials` block pins the source repo + commit + file sha256, so the
same spot-check can be run against the *source* quant with an HTTP range
request — the trust chain is explicit and third-party-verifiable, which is
strictly stronger than "download 300 GB from a named uploader and hope."

## Prime segments from a community quant (`fq_prime`)

Any published EXL3 mixed quant on Hugging Face can be decomposed into
segments **without downloading its shards**. `fq_prime` fetches each layer
shard's safetensors header with a ranged GET, identifies the per-expert
fragments and their K from the trellis geometry, then range-reads only the
expert bytes it needs (coalesced, paced, resumable):

```bash
# per-expert-layout source (3.36 bpw): pull just the K4 experts
uv run tools/fq_prime.py prime \
  --repo willfalco/GLM-5.2-EXL3-TR3-3.36bpw \
  --revision 8d9aa923a17502675ca23737349b67f2e66bb69d \
  --layers 3-10 --k 4 --out ~/fq-primed/segments-336

# shared-H-layout source (3.42 bpw): emits BOTH families — the verbatim
# shared-h segments (repack-of) and, with --expand, the per-expert expanded
# view (derived-from: shared rows replicated into each expert, an exact,
# byte-preserving expansion)
uv run tools/fq_prime.py prime \
  --repo willfalco/GLM-5.2-EXL3-TR3-3.42bpw \
  --revision ae68c65947efa90bea37308e15421872f124c46d \
  --layers 3-10 --k 3,4 --expand --out ~/fq-primed/segments-342
```

**Measured over layers 3–10 of both quants: 46.6 GB transferred against a
72.2 GB full-download counterfactual.** The two runs are the honest bracket
around what ranged reads buy: the 3.36 bpw source, where only the K4 experts
were wanted, moved 13.7 of 36.0 GB (**62 % saved**, 464 range requests); the
3.42 bpw source, where every expert was wanted, moved 32.8 of 36.1 GB (**9 %
saved**, 254 requests — the residue is headers and non-expert tensors).
**The saving is selectivity, not compression.** Ask for everything and you
transfer everything; the win scales with how much of the repo your recipe
does *not* need.

## Verify a reconstruction (`fq_verify`)

`fq_verify` turns "trust me" into a checkable claim, at two rungs:

```bash
# byte-identity, at the strongest granularity that applies (auto-detected):
#  * local snapshot available -> stream-reassemble every shard from segments
#    and compare sha256 against the source checkpoint
uv run tools/fq_verify.py --identity --segments ./segments \
  --source <source-checkpoint-dir> --json id.json --md id.md

#  * primed-from-remote family -> re-fetch sampled experts with FRESH ranged
#    reads of the pinned source and byte-compare (plus a full local hash of
#    every expert span against the signed attestations)
uv run tools/fq_verify.py --identity --segments ~/fq-primed/segments-336

#  * expanded (derived-from) family -> re-derive every expert from its parent
#    shared-h segment + profile and byte-compare in full, checking the
#    parent sha256 pins
uv run tools/fq_verify.py --identity \
  --segments ~/fq-primed/segments-342/expanded \
  --parent ~/fq-primed/segments-342/shared-h

# numeric similarity (GPU, exllamav3 reference dequant): cosine / relative
# Frobenius error / max|diff| between families and against BF16 ground truth
CUDA_VISIBLE_DEVICES=0 python tools/fq_verify.py --similarity \
  --family bm-k3=./segments,k=3 \
  --family 342-k4=~/fq-primed/segments-342/expanded,k=4 \
  --bf16 <bf16-snapshot-dir> --layers 3-10 --experts 24 --seed 42
```

Byte-identity is claimed only where it actually holds — whole shards for a
fully repacked family, per-fragment for a primed layer window, full
re-derivation for expanded views — and everything softer is reported as
measured similarity, not identity. The complete proof table for the three
GLM-5.2 community quants is in
[docs/RECONSTRUCTION.md](docs/RECONSTRUCTION.md).

## Trust: where the key comes from

Signature checking is only as good as the key you check against. Reading
`signer_pubkey` out of the artifact repo's own manifest proves nothing — a
compromised repo rewrites the bytes, the attestations and the key in one
push. So the authorized fingerprints live **in this git repository**:

```bash
cat keys/FINGERPRINTS      # authority derives from the commit history
git log -p keys/FINGERPRINTS   # every key ever authorized, and when
```

Pass one to any tool as `--trust-signer <fingerprint>` (full 64-hex, or an
unambiguous ≥16-hex prefix). Under pinning, a compromised artifact
repository can deny you service — it cannot make you accept the wrong
bytes.

A release additionally publishes **`fq-release/1`**: one signature over a
document listing sha256 and size for *every* file, including the indexes
and the attestation files. Verify one signature, then hash:

```bash
uv run tools/fq_release.py verify --dir ./segments \
  --trust-signer a58b7bb79ba58457
```

That closes the gap N per-fragment signatures leave open — they each prove
a fragment's origin, but nothing says *which set of fragments is the
release*, so files could be added, dropped or rolled back silently.

[**TRUST.md**](TRUST.md) is the full model: the four signature rungs and
three content proofs, what each one does **not** establish, and exactly
what an attacker with full control of the artifact repo can and cannot do
under pinning.

## Contribute segments (encode-and-share)

Fragments the community hasn't produced yet (K2 fast-load base, K5
hot-expert tier, K4 for less-hot experts) can be encoded by anyone with
the model's BF16 weights and a captured calibration statistic, then
published with `encode-of` provenance: the attestation pins the base model,
the calibration capture, the encoder revision and the effective quant
arguments, so someone else can re-run it.

**With an honest qualifier about determinism.** Re-encoding reproduces the
same bytes only *within a declared stack scope* — same GPU architecture,
same library versions, same batch shapes. We have measured the boundary:
1-ulp differences in CUDA `pow` for rotary `inv_freq`, and row-order
instability in sdpa/grouped_mm across batch shapes, are enough to change
the output. So `encode-of` attestations carry an explicit
`determinism_scope` block, and the honest claim *across* stacks is
`equivalence-of` — measured numeric equivalence — not byte identity. It is
the reproducible-builds model applied where reproducible builds actually
hold, and named differently where they do not.

The encoder driver and capture tooling live in the research branch below
and are being promoted into this repo as they stabilize.

## Status & roadmap

| Piece | Status |
|---|---|
| Schemas `fq-segment/1`, `fq-attestation/1`, `fq-manifest/1`, `fq-policy/2`, `fq-release/1` | **schema v1 — versioned, not yet frozen**; freezes once CI + verification enforcement have shipped ([`schemas/`](schemas/)) |
| `fq_repack` (checkpoint → segments) / `fq_assemble` (segments+recipe → checkpoint) | working, tested |
| `fq_fetch` (recipe + sources → range-fetched segment tree) | working, tested; multi-source, content-hash selection, resumable |
| Trust root (`keys/FINGERPRINTS`, `--trust-signer`, `fq-release/1`) | working, tested; per-source pinning and DSSE/in-toto envelopes still to come ([TRUST.md](TRUST.md)) |
| GLM-5.2 K3 base segments on HF | **prepared, publication gated on verification enforcement** — repo still private (401); reassembly **sha256-verified 76/76 MoE shards** |
| K4 hot-set priming from community mixed quants (3.42/3.36 bpw) | layers 3–10 primed + verified (fragment byte-identity vs fresh source reads — [docs/RECONSTRUCTION.md](docs/RECONSTRUCTION.md)) |
| `fq_verify` (byte-identity + numeric similarity proofs) | working, tested |
| Mixed-size (true mixed-K) assembly + loader metadata | done — mixed-K checkpoint assembled and booted |
| K2/K5 tiers (fresh encodes) | window 1 (real GLM-5.2, layers 3–10) encoded and uploaded, `encode-of` attestations |
| Runtime progressive loader + live bit-width reallocation (vLLM/GG) | loader boots from segments; live reallocation demonstrated at 0.4 s |
| Packaging, CI (ubuntu + macOS, py3.11–3.13), JSON Schemas | landed this release |

## Prior art and positioning

Every ingredient here has prior art, and a commissioned independent review
says so in detail: expert-wise mixed precision (MC-MoE, MxMoE, GEMQ),
progressive/selectable precision (BitStack, Matryoshka Quantization, QStore,
Any-Precision LLM), runtime promotion and demotion of expert precision
(HOBBIT, DynaExq), tensor-aware storage (Git-Theta, HF Xet), signed model
parts (OCI, in-toto, Sigstore/OpenSSF Model Signing). **The contribution is
the integration, not any ingredient.** Runtime promotion in particular is
*not* novel — our angle is being a provenance-aware *source* of the
alternate representations such systems need. The one claim we do make:

> We are not aware of an earlier open system that lets consumers assemble a
> loader-native mixed-K EXL3 safetensors checkpoint from independently
> published, digest-pinned and attested quantization segments.

Full review, with links and a per-layer comparison:
[docs/PRIOR-ART.md](docs/PRIOR-ART.md). The axes we think this should be
judged on are the reviewer's: storage and transfer across *many* recipes
versus separate checkpoints **and** versus Xet dedupe alone; reconstruction
plus verification cost; byte identity; reuse across independently produced
tiers; and multi-provider interoperability.

## Research

Design docs, verification reports, and the runtime work live in
[`malaiwah/vllm-voipmonitor` branch `claude/gg-overview-exploration-jchgd3`](https://github.com/malaiwah/vllm-voipmonitor/tree/claude/gg-overview-exploration-jchgd3/research/fungible-quant)
(`research/fungible-quant/`). Built and verified on 8× RTX PRO 6000
(SM120) against the Gilded Gnosis vLLM + b12x stack.

MIT licensed. Attestation ≠ endorsement: provenance chains terminate at
the source quant's reputation — they make that trust explicit and
checkable, not unnecessary.
