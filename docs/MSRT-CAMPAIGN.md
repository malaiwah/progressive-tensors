# GLM-5.2 MSRT campaign runbook

Numbers below are measured on real GLM-5.2 weights with this encoder, on an
RTX 5090 (Blackwell, 170 SMs, 1.79 TB/s), because that is the local card. The
rental target is an RTX PRO 6000 Blackwell (188 SMs, but 1.60 TB/s on the
Server Edition), so per-GPU throughput is **not** assumed equal: §2 states a
range and §4.0 makes a one-block probe on the rented card a gate.

Every claim is labelled **measured**, **derived** or **unmeasured**.

## 1. What the campaign produces

One `fq-cartridge/2` recipe (`recipes/glm52-k2k3-dag.json`) declares a
quantization graph, not a list of checkpoints:

```
W ─┬─ k2 (2 bpw base) ─┬─ k2r1  (+1) ─── k2r1r1 (+1)
   │                   └─ k2r2  (+2) ─── k2r2r1 (+1)
   └─ k3 (3 bpw base) ─┬─ k3r1  (+1) ─── k3r1r1 (+1)
                       └─ k3r2  (+2)
```

Nine product contracts for an MSRT-aware runtime (§6), over all 256 routed
experts of all 76 MoE layers (3–78, including the MTP layer):

| Assembly | bpw | Base + chain |
|---|---:|---|
| `k2` | 2.0 | k2 |
| `k2-k3like` | 3.0 | k2 + k2r1 |
| `k2-k4like-stepped` | 4.0 | k2 + k2r1 + k2r1r1 |
| `k2-k4like-direct` | 4.0 | k2 + k2r2 |
| `k2-k5like` | 5.0 | k2 + k2r2 + k2r2r1 |
| `k3` | 3.0 | k3 |
| `k3-k4like` | 4.0 | k3 + k3r1 |
| `k3-k5like-stepped` | 5.0 | k3 + k3r1 + k3r1r1 |
| `k3-k5like-direct` | 5.0 | k3 + k3r2 |

Sharing parent reconstructions is the point: nine products spanning 35 *nominal
trellis* bits per weight are encoded in **9 passes emitting 14 nominal trellis
bits per weight**. Nominal means trellis payload only; §3 has the actual bytes,
which come to 14.09 bpw for the routed artifacts (the `suh`/`svh` vectors and
scales) and 14.51 bpw with the skeleton amortised in.

Every emitted fragment carries a signed `fq-attestation/1` line beside it:
`encode-of` for expert shards, with the sha256 of each expert's contiguous byte
range and the exact parent shard digest the residual corrects; `repack-of` for
skeleton shards, with per-tensor digests and the source file they were copied
from. `finalize` refuses to publish anything that does not hash to its recorded
digest, name its own bytes, agree on one signer and one encoder build, and name
the parent fragment this campaign actually published.

## 2. Cost

**Measured** per-matrix trellis cost, real GLM-5.2 expert (layer 40, expert
128), `tile_batch=128`, median of 2, foreign GPU utilisation 0%:

| K | s / 6144x2048 matrix | trellis states |
|---|---:|---:|
| K1 | 1.618 | 32768 |
| K2 | 0.806 | 16384 |
| K3 | 0.674 | 8192 |

Lower K is **more** expensive: the DP table is `65536 >> K` wide. The graph uses
five K1 stages, so **K1 is 72% of campaign GPU time**.

**Measured** end-to-end, current code, `tile_batch=128`, clean GPU at start:
one real 32-expert block (layer 40, experts 128–159, all 9 nodes) committed in
**1188.4 s = 12.379 s/matrix**, of which 12.333 s/matrix is the matrix loop.
Everything the block needs to be *done* — the nine grouped safetensors writes,
per-expert span hashing, nine ed25519 signatures and the digest sidecars — costs
**0.046 s/matrix, 0.37%**. Two earlier blocks of the same work measured 12.06 s
(older unbounded tiling) and 12.30 s/matrix, both with a foreign job resident
for part of the run, so the three runs cluster at 12.06–12.38 s/matrix.

| Quantity | Value |
|---|---:|
| Expert-projection matrices | 58,368 (76 layers x 256 experts x 3) |
| Routed weights | 734.44 G |
| Passes per matrix | 9 |
| Committed rate, measured | **12.379 s/matrix** |
| **Whole campaign, one RTX 5090** | **200.7 GPU-hours** |
| Same nine products, kernel time only | 19 passes, 20.47 s/matrix |
| **DAG saving, trellis kernels** | **1.83x** (measured per-K mix) |
| DAG saving, end to end | ~1.80x (inferred: independent encoding's own write and metadata cost is unmeasured) |
| Bytes saved | 2.5x (14 vs 35 nominal trellis bpw) |

The isolated kernel sum for one expert is 34.08 s (11.36 s/matrix), so the real
loop costs ~8% more than the kernels alone: regularization, nine inverse
transforms, eighteen MSE reductions and the device-to-host copies serialize with
the trellis search. That gap is why the campaign figure comes from a block, not
from the sweep.

Fleet projection at the live JarvisLabs spot price of $0.99/GPU-hour. A roofline
weighting of the measured per-K mix (K1 is 72.4% of kernel time and
memory-bound at 1.79 -> 1.60 TB/s; K2/K3 lean on SM count, 170 -> 188) puts the
parity-card centre at `0.724 x 1.122 + 0.276 x 0.904 = 1.062`:

| Assumption | GPU-h | 8-GPU compute wall | Compute |
|---|---:|---:|---:|
| RTX PRO 6000 10% faster | 181 | 22.6 h | $179 |
| **roofline centre (+6.2%)** | **213** | **26.6 h** | **$211** |
| RTX PRO 6000 25% slower | 251 | 31.4 h | $248 |

**Compute wall is not elapsed wall.** Nothing overlaps the first source window
or the final output drain, and finalize (§4.6) needs 0.4–1.9 h with GPUs paused.
Realistic elapsed time at the roofline centre and 350 Mbps is **28–31 h**.

Budget: **$240 planning centre, $300 authorization cap** — engineering
contingencies, not statistical bounds. Compute $179–$248; 2 TB for two days
~$13; gates, CPU-side finalize/upload tails and one all-GPU in-flight-block redo
$3–$10; the cap additionally covers one full 8-layer-window redo (~$21).
A routine spot preemption with persistent storage loses at most the in-flight
block per GPU: 2.64 GPU-h / $2.61 for all eight, not a window.

Resolve the card range with the §4.0 probe. One block gives the committed rate;
one complete layer across all eight GPUs (2.64 GPU-h, ~$2.61) gives the fleet
rate under real storage and multi-GPU contention. The campaign figure is
**extrapolated linearly from that gate**, not proven by it: one block samples
kernel work on identical geometry, not shard locality, eight-process I/O, or
sustained clocks over a day.

## 3. Storage

| Artifact | Size |
|---|---:|
| k2 base trellis | 184.6 GB |
| k3 base trellis | 276.4 GB |
| K1 stages (k2r1, k2r1r1, k2r2r1, k3r1, k3r1r1) | 92.8 GB each |
| K2 stages (k2r2, k3r2) | 184.6 GB each |
| suh/svh metadata, all 9 nodes | 8.6 GB |
| skeleton (everything that is not a routed expert) | 37.8 GB |
| **campaign on disk** | **1.332 TB** |
| upload (the second base re-ships the skeleton) | 1.370 TB |
| source download | 1.507 TB |
| **total WAN** | **2.876 TB** |

Per layer, all nine nodes emit 17.0 GB in 72 block files (9 nodes x 8 blocks of
32 experts). The campaign is 5,472 block files plus 282 skeleton shards, each
with a digest sidecar and a signed attestation beside it.

Actual routed bitrate is **14.09 bpw** (nominal 14 plus 8.6 GB of `suh`/`svh`
and scales), or **14.51 bpw** with the skeleton amortised over routed weights.
The upload figure is *logical repository bytes*: the second base re-ships the
same skeleton content, which Hugging Face's Xet content-addressed storage should
deduplicate, so physical WAN is likely ~2.838 TB. 2.876 TB is the conservative
number.

**Rent 2 TB of persistent storage.** Campaign payload 1.332 TB plus one 150 GB
source window is already 1.482 TB, and `finalize` re-reads and re-hashes every
fragment before publishing a manifest, so the whole campaign has to be local at
that moment. 1.6 TB leaves ~118 GB for partial files, HF/Xet caches, logs and
retries; the extra 0.4 TB costs ~$2.67 for two days. Rolling publication deletes
*source* shards, never campaign output. A 1 TB volume cannot run this plan.

Aggregate throughput of **280 Mbps** is sufficient to hide transfer under
overlap; require **≥350 Mbps measured** before committing the fleet. Below that,
stage bytes on a regional filesystem with a CPU VM instead of paying GPUs to
wait.

## 4. Rental procedure

Set once:

```bash
REPO_ID=zai-org/GLM-5.2
REV=<40-hex-commit>                # the revision every attestation pins
SRC=~/.cache/huggingface/hub/models--zai-org--GLM-5.2/snapshots/$REV
CAMPAIGN=/data/glm52-msrt          # on the 2 TB volume
KEY=~/.fq_keys/glm52-campaign.key  # NEVER inside $CAMPAIGN or $SRC
RECIPE=recipes/glm52-k2k3-dag.json
ENC=/opt/exllamav3-python/exllamav3
```

Every download must pass `--revision "$REV"`. Without it `hf download` populates
the `main` snapshot instead of `$SRC`, which either leaves the pinned source
incomplete or silently stages a different revision.

`--base-model` and `--base-revision` are inferred from a Hugging Face snapshot
path like the one above. Any other layout must state them explicitly; the
revision must be an immutable 40-hex commit, because that is what every
attestation pins.

### 4.0 Gates before the fleet

1. `pytest tests/ -q` — full suite green.
2. On the rented card:
   `FQ_ENCODER_SOURCE=$ENC FQ_PARITY_SOURCE=<small per-layer BF16 checkpoint>
   pytest tests/test_msrt_decode_parity.py -q`. Thirteen tests prove, with the
   runtime's own `ext.reconstruct`, that a *published* campaign decodes to the
   MSE it attested — encode, finalize, combine under a pinned key, then decode
   the combined adapter against the source weights.
3. `fq-assemble-lora plan --source $SRC --recipe $RECIPE` — confirms layout,
   block count and which source shards are absent.
4. One block on one GPU (below). Compare the **committed** rate — the encoder
   prints both `s/matrix` committed and the quantizing part of it — with §2.
   Record the SKU, power limit and sustained clocks while it runs.
5. One complete layer across all eight GPUs, timed with external wall clock:
   2.44 GPU-h, ~$2.41. This is the only sample that includes eight-process I/O
   on one volume and multi-GPU contention, and it is what the campaign figure is
   extrapolated from.
6. Measure throughput to the hub (`hf download` one 5 GB shard): need ≥350 Mbps.

```bash
fq-assemble-lora encode --source $SRC --recipe $RECIPE --out $CAMPAIGN \
  --encoder-source $ENC --sign-key $KEY --block-size 32 \
  --layers 40 --shard-index 4 --shard-count 8 --device cuda:0
```

### 4.1 Stage source bytes per window

`plan` reports the exact shard files each layer needs:

```bash
fq-assemble-lora plan --source $SRC --recipe $RECIPE \
  --layers 3-10 --block-size 32 --out-plan window.json
python - <<'PY'
import json
plan = json.load(open("window.json"))
shards = sorted({s for L in plan["layers"] for s in L["shards"]})
print(" ".join(f"--include {s}" for s in shards))
PY
```

Feed those to
`hf download "$REPO_ID" --revision "$REV" --include ...`. A GLM MoE layer spans
~4 shards (~21.4 GB); an 8-layer window is ~150 GB. Stage the first window
*before* starting the GPUs.

Optional but recommended: fetch the hub's own LFS digests once and pass them as
`--source-digests`, so the skeleton pass never re-hashes source payloads to
attest what it copied:

```bash
curl -s "https://huggingface.co/api/models/$REPO_ID/tree/$REV?recursive=1" \
  > source-tree.json     # lfs.oid is the sha256 of each file
```

### 4.2 Skeleton (once per window)

```bash
fq-assemble-lora skeleton --source $SRC --recipe $RECIPE --out $CAMPAIGN \
  --sign-key $KEY --block-size 32 --source-digests source-tree.json
```

Copies non-expert tensors out of whatever shards are present and reports how
many were absent; re-run after every window, completed shards are skipped. It
also creates the signing key, so run it before any parallel encode.

**Measured shape:** 85 of GLM-5.2's 282 shards hold non-expert tensors, 1,217
tensors totalling 37.8 GB. Without `--source-digests`, those 85 shards
(~454 GB) are hashed once each to pin `repack-of` materials; digests are cached
in `$CAMPAIGN/source-digests.json` so windows never repeat the work.

### 4.3 Encode

```bash
fq-assemble-lora encode --source $SRC --recipe $RECIPE --out $CAMPAIGN \
  --encoder-source $ENC --sign-key $KEY --block-size 32 \
  --layers 3-10 --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7
```

`--devices` runs one single-device worker per GPU over disjoint
`(layer, block)` work, logging to `$CAMPAIGN/logs/`. Workers have disjoint work
and output paths — no NCCL, no shared writes — but they do share the source
shards, the page cache and the output volume's I/O bandwidth. Each block is
claimed with an `O_EXCL` lock, so two launchers pointed at one campaign cannot
burn GPU hours on the same block; the launcher clears stale claims once, before
it forks.

**Run one launcher at a time.** The work split is disjoint within a launcher,
not across launchers.

### 4.4 Resume after preemption

Re-run the identical command. A block counts as done only when every node of it
has a committed digest, and a block is re-encoded as a whole: a residual is only
valid against the parent bytes published beside it, so the tool never rewrites
one stage against a parent it merely recomputed in memory. Commit markers are
retracted before a payload is rewritten, so an interrupted rewrite can never
leave a digest that agrees with stale bytes.

Nothing is deleted. A recipe, block-size or source-revision change is refused
outright — with or without `--force` — because converting a campaign in place
would leave the previous layout's shards to be published alongside the new ones.
`--force` re-encodes blocks of *this* campaign.

### 4.5 Publish per window

Every block file is final and self-describing on arrival, with its digest and
signed attestation beside it, so a window can be staged to the repository as
soon as it finishes — but a **base tier is not loadable until `finalize` links
the skeleton into it**, so per-window uploads are staging only (use a private
branch or a staging prefix), and the authoritative upload happens after §4.6.

```bash
# per window: stage what is final, then delete that window's SOURCE shards
for path in "$CAMPAIGN"/stages/*; do
  hf upload <repo> "$path" "stages/$(basename "$path")"   # + digests, attestations
done

# after finalize: publish the complete base trees, including the skeleton
# hardlinks, metadata, digests and attestations that make them loadable
for label in k2 k3; do
  hf upload <repo> "$CAMPAIGN/base/$label" "base/$label"
done
hf upload <repo> "$CAMPAIGN/assemblies" assemblies      # signed plans
hf upload <repo> "$CAMPAIGN/campaign_summary.json" campaign_summary.json
```

Xet deduplication means the second base's identical skeleton content should cost
almost no extra WAN. Upload from a CPU process while the GPUs work, and pause the
fleet before the final drain.

### 4.6 Finalize once, at the end

```bash
fq-assemble-lora finalize --source $SRC --recipe $RECIPE --out $CAMPAIGN \
  --sign-key $KEY --block-size 32
```

Needs only `config.json` and the source index — deleted source payloads do not
have to be restaged — but it **re-reads and re-hashes all 1.332 TB of campaign
output**: a digest computed by the process that wrote a file proves nothing
about the file that survived. Budget 0.4–1.9 h depending on volume throughput
and **pause the GPU fleet first**; this pass needs no GPU.

It fails, naming what is wrong, unless every expected block and skeleton shard
is present, hashes to its recorded digest, carries a signed line naming those
exact bytes and span digests, agrees on one signer and one encoder build, and
(for stages) names the parent fragment this campaign published. It also refuses
shards the recipe does not describe. On success it writes, per base:
`config.json` with `hybrid_tr3_tail` and `quantization_config`,
`tier_bitmap.json`, `model.safetensors.index.json`, `MANIFEST.sha256` (built
from the verified digests), and hardlinks the skeleton payload in. Then one
signed `assemblies/<label>/assembly.jsonl` per product, plus
`campaign_summary.json`.

## 5. Consumer side

```bash
KEYID=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["provenance"]["signer_pubkey"])' \
  campaign_summary.json)

fq-combine-cartridges --root $CAMPAIGN --assembly k2-k4like-direct \
  --out ./k4like --trust-key $KEYID --base $CAMPAIGN/base/k2
# hottest 96 experts only: 96 experts of residual, not 256
fq-combine-cartridges --root $CAMPAIGN --assembly k2-k4like-direct \
  --out ./k4like-hot96 --experts 0-95 --trust-key $KEYID
```

`--base` is optional but worth passing whenever the base checkpoint is local: it
compares the base's `MANIFEST.sha256` with the digest the plan pins and checks
that the chain's first stage names the base block bytes this checkpoint
publishes. A cartridge applied to a different base corrects other weights.

`--trust-key` is required (or an explicit `--insecure-unsigned`). With it, the
combiner verifies the signed plan, requires the campaign identity in the plan to
match every fragment's attestation, checks each chain edge against the parent
digest, re-hashes each selected shard and its per-expert spans, and validates
tensor rank, dtypes, trellis/vector geometry and a finite positive rescale
factor. Selection is decided from the signed plan *before* any payload is
touched, so a narrowing consumer only needs the shards it selected. A requested
expert or layer that the product does not carry is an error, not a silent
downgrade. The emitted `fq-cartridge-adapter/2` records the base manifest it is
bound to, the campaign identity, the verified signer, and per-stage per-layer
coverage.

## 6. Runtime constraint (still the blocker for serving)

These cartridges are full-rank additive trellis weights. Loading them needs an
EXL3 MSRT-aware runtime; standard `add_lora` cannot. The reference
implementation is
[local-inference-lab/vllm#299](https://github.com/local-inference-lab/vllm/pull/299),
still draft, currently TP=1 with one model-wide slot and dense FP16 shadow
weights — which cannot hold GLM-5.2's 734 G routed weights on any single node.
Encoding and publishing the artifacts does not depend on that PR; serving them
at scale does.

## 7. Trust-path scope

The campaign's attestations are `fq-attestation/1` documents and are verified by
`fq_combine_cartridges` against a pinned key. They are **not** wired into
`fq_fetch`'s discovery path: that consumer needs an `fq-manifest/1` plus
`index-kK.json` segment layout, which this campaign does not emit. Treat the
cartridge family as its own trust path with its own consumer, not as a drop-in
for the segment family.
