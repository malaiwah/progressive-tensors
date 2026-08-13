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

### 1.1 Which recipe to rent for

Two of the nine products exist only to sell a **+1-bit upgrade** to a consumer
who already installed a 3 bpw or 4 bpw product: `k2-k4like-stepped` upgrades
`k2-k3like` for one more bit, and `k3-k5like-stepped` upgrades `k3-k4like`. They
are the two nodes with a K1 parent that is itself K1-corrected, and they cost
**two K1 passes**, the most expensive kind (§2).

**Measured**: at the same nominal bitrate the narrow-step path is always worse
than fetching the wider residual, on all 88 blocks of the Fruit rehearsal
(§3.5) and on real GLM-5.2 experts:

| Comparison | Fruit, 88 blocks / 2,816 experts per stage | GLM-5.2, 84 matrices |
|---|---:|---:|
| `k2+k2r1+k2r1r1` vs `k2+k2r2` (4 bpw) | **1.0901x** worse, sigma 0.00017, 88/88 | **1.0902x** worse, sigma 0.00022, 84/84 |
| `k3+k3r1+k3r1r1` vs `k3+k3r2` (5 bpw) | **1.0679x** worse, sigma 0.00053, 88/88 | **1.0684x** worse, sigma 0.00044, 84/84 |

The GLM-5.2 column sweeps **28 (layer, expert) pairs x 3 projections = 84
matrices per family**, requested as layers 3, 10, 19, 30, 40, 50, 60, 70 x
experts 0, 1, 9, 73, 100, 128 and reduced to the pairs whose source shards are
staged locally — four experts each in layers 10, 30, 40, 50, 60, 70 and two each
in layers 3 and 19. **168 of 168 comparisons favour the wider residual**, and the
three projections agree to four decimal places (mean 1.07927 gate, 1.07937 up,
1.07935 down, over both families). The proxy and the real weights agree to three
decimals. This is a property of greedy residual trellis coding, not of a model or
a layer. The upgrade path is the only thing the extra passes buy, and it buys it
at 9.0%/6.8% higher error. For reference, the wider residual alone improves on
its base by a mean of **14.2x**.

`recipes/glm52-k2k3-lean.json` is the same menu without those two products:
**7 passes, 12 nominal bpw, 7 products, every one of them the best available at
its bitrate.**

Both graphs were then encoded on the same clean RTX 5090, one real 32-expert
block each, back to back (§2):

| | `glm52-k2k3-dag` | `glm52-k2k3-lean` |
|---|---:|---:|
| Passes per matrix | 9 | **7** |
| Products | 9 | 7 |
| Nominal trellis bpw | 14 | **12** |
| **Committed rate, measured** | 11.5276 s/matrix | **8.1775 s/matrix** |
| Campaign, one RTX 5090 | 186.9 GPU-h | **132.6 GPU-h** |
| Compute at $0.99, roofline centre | $197 | **$139** |
| 8-GPU compute wall, roofline centre | 24.8 h | **17.6 h** |
| Campaign on disk | 1.332 TB | **1.146 TB** |
| Logical WAN | 2.876 TB | **2.691 TB** |
| Throughput needed to hide I/O | 273 Mbps | 360 Mbps |

**Recommendation: rent for the lean recipe.** It saves **54.3 GPU-hours**
(measured: the difference between two blocks on this card, scaled to 58,368
matrices), which is **[DERIVED] $53.77** at $0.99/GPU-h on a card as fast as this
one and **$57–$60** at the projected parity-card range, plus 6.8 h of 8-GPU
compute wall and 183 GB of storage. Every product it ships is better than the one
it drops. Choose the full graph only if a
+1-bit upgrade for an already installed 3 bpw or 4 bpw base is worth $58 and 9%
more error on the two upgraded tiers. The rest of this runbook prices **both**:
`$RECIPE` selects which.

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

Lower K is **more** expensive: the DP table is `65536 >> K` wide. `dag` runs
five K1 stages and `lean` three, so K1 is **72%** of `dag` kernel time and
**61%** of `lean`. Any change to the graph should be priced with this table
first; §2 shows the prediction landing within 0.2% of the measurement.

**Measured** end to end, one real 32-expert block (layer 40, experts 128–159,
96 matrices), `tile_batch=128`, clean GPU, both recipes run back to back on the
same card with the same build — the only difference is the graph:

| Recipe | Passes | Block | Committed | Matrix loop | Commit overhead |
|---|---:|---:|---:|---:|---:|
| `glm52-k2k3-dag` | 9 | 1106.6 s | **11.5276 s/matrix** | 11.4828 | 0.39% |
| `glm52-k2k3-lean` | 7 | 785.0 s | **8.1775 s/matrix** | 8.1408 | 0.45% |

The measured lean/full ratio is **0.7093**; the per-K kernel mix predicts
`7.946 / 11.182 = 0.7108`. Two independent measurements agreeing to 0.2% is why
the per-K table can be trusted to price a graph change before running it.

Everything a block needs to be *done* — the grouped safetensors writes,
per-expert span hashing, one ed25519 signature per node and the digest sidecars
— costs **0.04 s/matrix, under half a percent**. Three earlier blocks of the
nine-node graph measured 12.06, 12.30 and 12.38 s/matrix with a foreign job
resident for part of each run; the 11.53 s above is the clean-card number and
supersedes them.

Subtracting the two graphs isolates the in-loop cost of one K1 pass:
`(11.4828 - 8.1408) / 2 = 1.671 s/matrix`, 3.3% above the 1.618 s measured in
isolation. The remaining nodes then account for 3.13 s/matrix against an
isolated 2.96 s. The loop therefore runs within ~5% of the sum of its kernels.

| Quantity | `dag` | `lean` |
|---|---:|---:|
| Expert-projection matrices | 58,368 | 58,368 |
| Routed weights | 734.44 G | 734.44 G |
| Passes per matrix | 9 | **7** |
| Committed rate, measured | 11.5276 s | **8.1775 s** |
| **Campaign, one RTX 5090** | **186.9 GPU-h** | **132.6 GPU-h** |
| Same products encoded separately, kernel only | 20.47 s/matrix | 15.44 s/matrix |
| **Graph saving, trellis kernels** | **1.83x** | **1.94x** |
| Nominal trellis bytes saved | 2.5x (14 vs 35 bpw) | 2.9x (12 vs 35 bpw) |

Fleet projection at the live JarvisLabs spot price of $0.99/GPU-hour.
**[DERIVED]** A roofline weighting of the measured per-K mix (K1 is memory-bound,
1.79 -> 1.60 TB/s, a 1.122 time factor; K2/K3 lean on SM count, 170 -> 188 SMs, a
0.904 time factor) is **recipe-specific**, because the two graphs do not have the
same K1 share:

| Graph | K1 share of kernel time | roofline time factor |
|---|---:|---:|
| `dag` | 5 x 1.618 / 11.182 = 0.7235 | **1.0617** |
| `lean` | 3 x 1.618 / 7.946 = 0.6109 | **1.0372** |

| Assumption | `dag` GPU-h | `dag` compute | `lean` GPU-h | `lean` compute |
|---|---:|---:|---:|---:|
| duration x0.90 (faster card) | 168 | $167 | 119 | $118 |
| **roofline centre** | **198** | **$196** | **138** | **$136** |
| duration x1.25 (slower card) | 234 | $231 | 166 | $164 |

The outer rows are *duration* multipliers, not throughput claims: x0.90 is a 10%
shorter run, which is 11% more throughput. 8-GPU compute wall at the roofline
centre: **24.8 h** (`dag`), **17.2 h** (`lean`).

**Compute wall is not elapsed wall.** Nothing overlaps the first source window
or the final output drain, and finalize pauses the GPUs (§4.6). **[MEASURED]**
finalize re-read and re-hashed a 9.32 GB campaign (8.68 GiB) in 10.263 s = **908
MB/s**, warm cache, one core. **[UNMEASURED]** extrapolating that 123x to a cold
1.146 TB tree: the pass is then bounded by `min(hash rate, volume read rate)` and
by filesystem metadata on 4,256 block files plus their sidecars, so budget **21
min at 1 GB/s, 38 min at 500 MB/s, and verify it on the rehearsal**.
**[DERIVED]** elapsed time at the roofline centre and 350 Mbps is **27–30 h**
(`dag`) or **20–22 h** (`lean`).

**[DERIVED]** Because the lean campaign shrinks compute by 29% but bytes by only
6%, it is the more I/O-bound of the two. Against the *rental* compute wall above,
hiding all logical WAN under compute needs **348 Mbps** (`lean`) versus **258
Mbps** (`dag`) — campaign averages only. The first window (203.9 GB) and the final
base drain (536.6 GB) overlap nothing by construction (§4.5), so they are elapsed
time whatever the link does; §2.1 prices them as idle-GPU tails.

Budget for the recommended `lean` graph: **$160 planning centre, $220
authorization cap** — engineering contingencies, not statistical bounds.

| Item | Centre | Worst |
|---|---:|---:|
| Compute (roofline centre; duration x1.25 card) | $136 | $164 |
| Eight GPUs idle through the first stage and finalize (§2.1) | $12 | $12 |
| 1.6 TB persistent storage (elapsed; two full days) | $5 | $11 |
| Gates: one block $0.22, one 8-GPU layer $1.73 | $2 | $2 |
| CPU VM for finalize and the 3.4 h final drain | $2 | $3 |
| Sub-total | **$157** | **$192** |
| Contingency: one 8-layer-window redo (13.96 GPU-h) | — | $14 |
| **Planning centre / authorization cap** | **$160** | **$220** |

The cap is the worst column plus one window redo, plus $14 of slack; at $200 the
same arithmetic came to $206 and had none. **[DERIVED, conditional]** a routine
spot preemption with persistent storage loses at most the in-flight block per GPU
— 1.74 GPU-h / $1.73 for all eight, not a window — which follows from the block
commit protocol (§4.4) but assumes the provider's volume survives the preemption,
that `$KEY` and `$REPO` survive on it (§4.5 checklist), and that `hf download`
resumes. None of those three is measured here. Forgetting the §4.5 release gate
costs $27 and is the single largest avoidable line item. For the `dag` graph add
$60 of compute: **$220 centre, $290 cap**.

### 2.1 Fleet size

Compute hours are fixed; what a larger fleet buys is wall time, and what it costs
is idle GPU during the serial tails. With `lean` at the roofline centre
(140.8 GPU-h), a 1.133 h first-window stage and a 0.351 h finalize during which
GPUs do nothing, total cost as a function of fleet size `n` is
`140.8 x 0.99 + n x 0.99 x 1.484 + 0.2223 x elapsed`:

| GPUs | Compute wall | Elapsed | Total |
|---:|---:|---:|---:|
| 1 | 140.8 h | 142.3 h | $168.50 |
| 2 | 70.4 h | 71.9 h | $154.68 |
| 4 | 35.2 h | 36.7 h | **$149.98** |
| 8 | 17.6 h | 19.1 h | $152.03 |

**Eight GPUs costs $2.05 more than the four-GPU optimum and finishes 17.6 h
sooner**, so take eight — but only with the release gate in §4.5. Leaving eight
GPUs attached through finalize and the 3.4 h final upload adds $27, which is more
than ten times the entire benefit of the fleet-size choice.

Resolve the card range with the §4.0 probe. One block gives the committed rate;
one complete layer across all eight GPUs gives the fleet rate under real storage
and multi-GPU contention. The campaign figure is **extrapolated linearly from
that gate**, not proven by it: one block samples kernel work on identical
geometry, not shard locality, eight-process I/O, or sustained clocks over a day.

## 3. Storage

| Artifact | `dag` | `lean` |
|---|---:|---:|
| k2 base trellis | 184.6 GB | 184.6 GB |
| k3 base trellis | 276.4 GB | 276.4 GB |
| K1 stages, 92.8 GB each | 5 = 464.0 GB | 3 = 278.4 GB |
| K2 stages, 184.6 GB each | 2 = 369.2 GB | 2 = 369.2 GB |
| suh/svh metadata | 8.6 GB (9 nodes) | 6.7 GB (7 nodes) |
| skeleton (everything not a routed expert) | 37.8 GB | 37.8 GB |
| **campaign on disk** | **1.330 TB** | **1.147 TB** |
| upload: stages, per window | 833.2 GB | 647.6 GB |
| upload: both bases incl. their skeleton copy, after finalize | 536.6 GB | 536.6 GB |
| source download, unique bytes | 1.5067 TB | 1.5067 TB |
| **total WAN** | **2.877 TB** | **2.691 TB** |
| same, if windows are re-downloaded naively | 2.984 TB | 2.798 TB |

**Measured from the pinned index and the hub's own tree listing** (revision
`b4734de4`, 282 shards, 1.506667 TB): the ten 8-layer windows are **198.6, 166.2,
160.7, 166.2, 171.6, 166.2, 160.8, 166.2, 166.2, 85.8 GB**, not a uniform 150 GB,
and **20 shards straddle a window boundary**. Deleting a window wholesale and
re-fetching those 20 costs **+107.2 GB of WAN**; keeping a shard until its last
consuming window costs **+5.3 GB of peak disk**. Keep them.

Peak local bytes, window by window (campaign output so far plus resident source,
with last-use retention):

| | at first stage | peak | when |
|---|---:|---:|---|
| `lean` | 241.7 GB | **1.266 TB** | after encoding layers 67–74 |
| `dag` | 241.7 GB | **1.439 TB** | after encoding layers 67–74 |

**A 1.6 TB volume runs either graph** — 334 GB of headroom for `lean`, 161 GB for
`dag` — provided the source is staged as real files (§4.1) and windows are
deleted as they retire. `finalize` needs the whole campaign local, which is the
peak above, not an extra copy.

Per layer, `dag` emits 17.0 GB in 72 block files (9 nodes x 8 blocks of 32
experts) and `lean` 14.6 GB in 56; the campaigns are 5,472 and 4,256 block files
plus 282 skeleton shards, each with a digest sidecar and a signed attestation
beside it.

Actual routed bitrate is **14.09 bpw** for `dag` (nominal 14 plus `suh`/`svh` and
scales) and **12.07 bpw** for `lean`, or 14.51 and 12.48 bpw with the skeleton
amortised over routed weights. The upload figure is *logical repository bytes*:
the second base re-ships the same skeleton content, which Hugging Face's Xet
content-addressed storage should deduplicate, so physical WAN is likely ~2.838 TB
(`dag`) or ~2.653 TB (`lean`). The larger numbers are the conservative ones.

**Storage: 1.6 TB runs `lean`; rent 2 TB for `dag`.** `finalize` re-reads and
re-hashes every fragment, so the whole campaign must be local at that moment.
`lean` needs 1.146 TB plus one 150 GB source window = 1.296 TB, leaving ~304 GB
on a 1.6 TB volume for partial files, HF/Xet caches, logs and retries. `dag`
needs 1.482 TB, leaving only ~118 GB, which is too tight — the extra 0.4 TB costs
~$2.67 for two days. Rolling publication deletes *source* shards, never campaign
output. A 1 TB volume cannot run either plan.

Throughput sufficient to hide transfer under compute is **273 Mbps** (`dag`) or
**360 Mbps** (`lean`, whose compute window is 29% shorter for 94% of the bytes);
require **≥400 Mbps measured** before committing the fleet to `lean`. Below that,
stage bytes on a regional filesystem with a CPU VM instead of paying GPUs to
wait.

## 3.5 Rehearsal: this procedure, run whole

**Measured.** The complete procedure below — `plan`, `skeleton`, `encode`,
resume check, `finalize`, two `fq-combine-cartridges` products and a pinned-key
rejection — was run end to end on the GLM-5.2-SIQ-Fruit proxy (11 MoE layers,
256 experts, 512-wide) on one RTX 5090, using the same nine-node recipe shape as
the GLM graph:

| Step | Result |
|---|---|
| `plan` | 88 blocks, 8,448 matrices, 9 passes/matrix, 14.0 bpw emitted |
| `skeleton` | 16 shards |
| `encode` | 88/88 blocks, 67m19s, **0.4778 s/matrix committed** of which 0.4752 quantizing (**0.54% commit overhead**), 1.12 GPU-h |
| resume | re-run skipped all 88 blocks: `nothing to do` |
| `finalize` | re-hashed and verified 792 expert fragments + 16 skeleton shards; published 9 assemblies at 2.0/3.0/4.0/4.0/5.0 and 3.0/4.0/5.0/5.0 bpw |
| combine `k2-k5like` | 88 shards, 67,584 tensors, 256 experts x 11 layers |
| combine `k2-k4like-direct --experts 0-95` | **33 shards**, 12,672 tensors, 96 experts |
| wrong `--trust-key` | refused |

**Measured** decode of those published products through the runtime's own
`ext.reconstruct`, worst of 3 experts x 3 projections in layer 3 block 0, against
the BF16 source:

| Product | MSE | vs its K2 base |
|---|---:|---:|
| `base/k2` alone | 1.517e-04 | 1x |
| `k4like-hot96` (K2 + K2 residual) | 1.044e-05 | **14.5x** |
| `k5like` (K2 + K2 + K1 residuals) | 2.877e-06 | **52.7x** |

Both products measure 1.40x the mean MSE their deepest stage attested, which is
the expected worst-of-nine to block-mean spread, not a decode error. The
narrowed product carries block 2 (experts 64–95) and not block 3 (96–127): a
consumer wanting the hot 96 downloads 33 of 88 shards.

The commit overhead here (0.54%) corroborates the 0.37% measured on a real
GLM-5.2 block in §2 — writing, hashing and signing every fragment is not what
this campaign costs.

Two operational failures were found by running it, not by reading it: two
launchers with two keys produced a campaign `finalize` correctly refused (§4.3),
and a stage shard written without its Hadamard sign vectors was caught by the
combiner's component check rather than by the encoder.

### 3.6 Rehearsal of the *procedure*, through the driver

**Measured.** `tools/msrt_campaign.sh` was then run end to end on the same proxy
with the **lean** recipe and two windows (`3-8`, `9-13`), against a `--local-dir`
source staged by a stand-in hub client:

| Step | Result |
|---|---|
| window `3-8` | staged 11 files, skeleton wrote 11, **48 blocks at 0.3399 s/matrix** |
| retirement | deleted layers 003–008; kept the skeleton-only shards and everything window 2 still needed |
| window `9-13` | staged only the 5 new files, skeleton wrote 5 (11 already complete, 0 absent), **40 blocks at 0.3406 s/matrix** |
| release gate | printed before finalize, which used no GPU |
| `finalize` | both bases published, **7 assemblies** at 2.0/3.0/4.0/5.0 and 3.0/4.0/5.0 bpw |
| products | `k2-k5like` 88 shards / 67,584 tensors / 256 experts; `k2-k4like-direct --experts 0-95` **33 of 88 shards** / 12,672 tensors |
| decode | 2.877e-06 (`k5like`) and 1.044e-05 (`hot96`), **bit-identical to the hand-run campaign** despite a different signing key, different windows and a different driver |

The lean rate on this proxy is 0.3399 s/matrix against 0.4778 for `dag`, a ratio of
**0.7115** — a third independent measurement of the same 0.709/0.711 ratio (§2).
The identical decode MSE from two independently driven campaigns is the
determinism claim (§4.4) holding across whole campaigns, not just shards.

## 4. Rental procedure

Set once:

```bash
export REPO_ID=zai-org/GLM-5.2
export REV=b4734de4facf877f85769a911abafc5283eab3d9   # pin; every attestation names it
[[ $REV =~ ^[0-9a-f]{40}$ ]] || { echo "REV must be a 40-hex commit"; return 1; }
export SRC=/data/glm52-src          # real files on the volume, NOT the HF cache
export CAMPAIGN=/data/glm52-msrt    # same 1.6 TB volume
export KEY=/data/keys/glm52-campaign.key   # on the volume, NEVER inside $CAMPAIGN or $SRC
export REPO=/data/progressive-tensors      # the checkout, ON the volume
export RECIPE=$REPO/recipes/glm52-k2k3-lean.json   # §1.1: the recommended menu
export ENC=/opt/exllamav3-python/exllamav3
export HF_XET_CACHE=/data/xet-cache        # if Xet is used at all, keep it here
export HF_HUB_DISABLE_XET=1                # source staging: see below

# metadata first: every later command resolves layers through the index
hf download "$REPO_ID" --revision "$REV" --local-dir "$SRC" \
  --include config.json --include model.safetensors.index.json \
  --include 'tokenizer*' --include '*.py' --include '*.md'
```

Everything is `export`ed because the staging and cleanup snippets below run
Python from a heredoc and read these names from the environment.

**Disable Xet for source staging, keep it for uploads.** Every source shard is
downloaded exactly once and then deleted, so the Xet chunk cache can only grow
against the same 1.6 TB volume the campaign needs — its occupancy is not in the
peak timeline in §3 and it is not bounded by anything in this procedure.
`HF_HUB_DISABLE_XET=1` removes that variable from the download side. Unset it for
the *uploads* in §4.5, where chunk deduplication is what makes the second base's
identical skeleton copy nearly free.

**Stage with `--local-dir`, never into the HF cache.** A cache download writes
payload into `<cache>/blobs/<sha>` and puts a *symlink* in `snapshots/<rev>/`;
deleting the snapshot entry frees 76 bytes, not 5 GB, so rolling windows would
accumulate the entire 1.5067 TB source and blow through the volume. `--local-dir`
writes real files that `rm` actually reclaims.

Every download must pass `--revision "$REV"`. Without it `hf download` resolves
`main`, which either stages a different revision or leaves the pinned one
incomplete.

Because `$SRC` is not a Hugging Face snapshot path, identity cannot be inferred
from it: **every `skeleton`, `encode` and `finalize` invocation must pass
`--base-model "$REPO_ID" --base-revision "$REV"`**, and the revision must be an
immutable 40-hex commit, because that is what every attestation pins. Define the
argument arrays once and reuse them so no invocation can drift:

```bash
IDENTITY=(--base-model "$REPO_ID" --base-revision "$REV")
COMMON=(--source "$SRC" --recipe "$RECIPE" --out "$CAMPAIGN" --block-size 32)
```

`$KEY` and `$REPO` live on the **volume**, not in the GPU instance's home
directory: §4.5 requires terminating the GPU instances and reattaching the volume
to a CPU VM, and `finalize` needs both the key and the tool after that handoff.

**`tools/msrt_campaign.sh` is this whole procedure as one checked driver** —
staging, skeleton, encode, last-use retirement, finalize, in order, with the
identity flags and the window arithmetic already right. Run `DRY_RUN=1` first to
print the exact command sequence. The sections below explain what each step does
and what it costs; the driver is what you should actually run.

### 4.0 Gates before the fleet

0. Rehearse the whole procedure on a small MoE checkpoint first (§3.5): it costs
   about one GPU-hour and it is the only step that exercises finalize, the
   products and the trust path at campaign scale.
1. `(cd "$REPO" && pytest tests/ -q)` — full suite green.
2. On the rented card:
   `FQ_ENCODER_SOURCE=$ENC FQ_PARITY_SOURCE=$PROXY \
   pytest "$REPO/tests/test_msrt_decode_parity.py" -q`, where `$PROXY` is a small
   per-layer BF16 MoE checkpoint. Thirteen tests prove, with the
   runtime's own `ext.reconstruct`, that a *published* campaign decodes to the
   MSE it attested — encode, finalize, combine under a pinned key, then decode
   the combined adapter against the source weights.
3. `fq-assemble-lora plan --source $SRC --recipe $RECIPE` — confirms layout,
   block count, which source shards are absent, and the `skeleton_only_shards`
   that no layer window would stage. This needs the metadata download from the
   "Set once" block above; on a fresh rental `$SRC` does not exist yet and the
   command correctly refuses a directory with no `config.json`.
4. One block on one GPU, then one complete layer on all eight, in a **throwaway
   campaign directory** (below). The single block gives the committed rate — the
   encoder prints both `s/matrix` committed and the quantizing part — to compare
   with §2. Record the SKU, power limit and sustained clocks while it runs. The
   eight-GPU layer costs 1.74 GPU-h / ~$1.73 for `lean` (2.46 GPU-h / $2.43 for
   `dag`) and is the only sample that includes eight-process I/O on one volume
   and multi-GPU contention; it is what the campaign figure is extrapolated from.
5. Measure throughput to the hub (`hf download` one 5 GB shard): need
   ≥400 Mbps for `lean`, ≥350 Mbps for `dag` (§3).
6. Run finalize on the rehearsal campaign twice. It must succeed both times:
   finalize re-reads the whole campaign, so on the real one it is a
   minutes-to-hours pass that has to be resumable after a preemption.

```bash
# a gate campaign, thrown away afterwards: never time a block into $CAMPAIGN,
# because a completed block would be skipped by the timed full-layer run
export GATE=/data/glm52-gate
GATE_COMMON=(--source "$SRC" --recipe "$RECIPE" --out "$GATE" --block-size 32)

# stage layer 40's payload (the metadata is already there from "Set once")
fq-assemble-lora plan "${GATE_COMMON[@]}" --layers 40 --out-plan gate.json
hf download "$REPO_ID" --revision "$REV" --local-dir "$SRC" \
  $(python3 -c 'import json;print(" ".join(f"--include {s}" for L in json.load(open("gate.json"))["layers"] for s in L["shards"]))')

# skeleton mints and binds the signing key; encode with --shard-count > 1 will
# not create one, so this must come first even for a gate
fq-assemble-lora skeleton "${GATE_COMMON[@]}" "${IDENTITY[@]}" --sign-key "$KEY"

# gate 4a: one block, one GPU
fq-assemble-lora encode "${GATE_COMMON[@]}" "${IDENTITY[@]}" \
  --encoder-source "$ENC" --sign-key "$KEY" \
  --layers 40 --shard-index 4 --shard-count 8 --device cuda:0

# gate 4b: the same layer across all eight GPUs, timed externally. --force
# re-encodes the block 4a already finished, so the wall clock covers a full layer
time fq-assemble-lora encode "${GATE_COMMON[@]}" "${IDENTITY[@]}" \
  --encoder-source "$ENC" --sign-key "$KEY" --layers 40 --force \
  --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7

rm -rf "$GATE"          # the real campaign must start from an empty --out
```

### 4.1 Stage source bytes per window

`plan` reports the exact shard files each layer needs:

```bash
fq-assemble-lora plan "${COMMON[@]}" --layers 3-10 \
  --out-plan "$CAMPAIGN/plans/3-10.json"
WINDOW_PLAN=$CAMPAIGN/plans/3-10.json python3 - <<'PY'
import json, os
plan = json.load(open(os.environ["WINDOW_PLAN"]))
shards = sorted({s for L in plan["layers"] for s in L["shards"]})
print(" ".join(f"--include {s}" for s in shards))
PY
```

Stage exactly those, plus the shards that belong to no layer at all:

```bash
hf download "$REPO_ID" --revision "$REV" --local-dir "$SRC" \
  $(WINDOW_PLAN=$CAMPAIGN/plans/3-10.json python3 - <<'PY'
import json, os
plan = json.load(open(os.environ["WINDOW_PLAN"]))
want = {s for L in plan["layers"] for s in L["shards"]}
want |= set(plan["skeleton_only_shards"])      # embeddings, lm_head, dense layers
for shard in sorted(want):
    print(f"--include {shard}", end=" ")
PY
)
```

`skeleton_only_shards` is why that second line exists: for GLM-5.2 one shard
(`model-00001-of-00282`, 5.3 GB) holds only non-expert tensors, so it appears in
**no** layer's shard list. Staging strictly per layer leaves it absent, `skeleton`
counts it as absent and returns 0, and `finalize` refuses to publish a base —
after the whole campaign. `plan` emits the field for every window; staging it once
is enough.

**Measured window sizes** (§3): 198.6, 166.2, 160.7, 166.2, 171.6, 166.2, 160.8,
166.2, 166.2, 85.8 GB for the ten 8-layer windows, first window plus the
skeleton-only shard = 203.9 GB. Stage the first window *before* starting the GPUs.

`$SRC` must keep `config.json` and `model.safetensors.index.json` for the whole
campaign: every later command resolves layers and experts through the index, and
`finalize` needs nothing else from the source — **verified** by running finalize
against a source directory holding only those two files. That holds for an
**indexed** checkpoint, which `zai-org/GLM-5.2` is; a source without an index
would have to open shards, so keep its payload until finalize.

Optional but recommended: fetch the hub's own LFS digests once and pass them as
`--source-digests`, so the skeleton pass never re-hashes source payloads to
attest what it copied:

```bash
curl -s "https://huggingface.co/api/models/$REPO_ID/tree/$REV?recursive=1" \
  > "$CAMPAIGN/source-tree.json"    # lfs.oid is the sha256 of each file
```

### 4.2 Skeleton (once per window)

```bash
fq-assemble-lora skeleton "${COMMON[@]}" "${IDENTITY[@]}" \
  --sign-key "$KEY" --source-digests "$CAMPAIGN/source-tree.json"
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
fq-assemble-lora encode "${COMMON[@]}" "${IDENTITY[@]}" \
  --encoder-source "$ENC" --sign-key "$KEY" --layers 3-10 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7
```

`--devices` runs one single-device worker per GPU over disjoint
`(layer, block)` work, logging to `$CAMPAIGN/logs/`. Workers have disjoint work
and output paths — no NCCL, no shared writes — but they do share the source
shards, the page cache and the output volume's I/O bandwidth.

Ownership of a block is an `flock`, not a file that exists: the kernel releases it
however the owner dies, so a crashed worker leaves nothing to clean up and no
launcher ever has to guess whether someone else's claim is stale. A launcher also
takes an exclusive `flock` on the campaign directory for its whole lifetime, which
`encode`, `skeleton` and `finalize` all respect, so two launchers cannot burn GPU
hours on the same blocks even if the first one's parent is killed and its workers
survive.

**One campaign directory takes one launcher, and one key.** Both are now
enforced rather than advised: the campaign `flock` refuses the second launcher,
and the sentinel binds `signer_pubkey` so a different `--sign-key` is refused
before any quantization. Observed in rehearsal, before either existed: two
launchers, each with its own key, split the 88 blocks between them and finished
the whole encode; `finalize` then refused the campaign — `fragments are signed by
3 different keys` — and 82 GPU-minutes had to be re-run. Keep the key file for the
life of the campaign, outside whatever the cleanup script deletes; a resumed run
must publish under the signer its earlier fragments already named.

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

`tools/msrt_campaign.sh` performs the staging, encoding and retirement loop,
including the last-use retention that keeps the 20 straddling shards (§3) — set
`UPLOAD_HOOK` to a script taking the window range if you want per-window uploads
inside it. Publication itself is deliberately manual, because it decides what the
world sees:

```bash
export DEST=malaiwah/GLM-5.2-MSRT           # the destination repo
export STAGING=refs/heads/staging           # never publish onto main directly

# per window, while the GPUs work on the next one: stages only. A base is not
# loadable until finalize links the skeleton into it, so bases wait.
for path in "$CAMPAIGN"/stages/*; do
  hf upload "$DEST" "$path" "stages/$(basename "$path")" \
    --revision "$STAGING" --commit-message "stages $(basename "$path")"
done

# ---- release the GPU fleet here (see below) ----

# after finalize, from the CPU VM: the complete base trees, with the skeleton
# hardlinks, metadata, digests and attestations that make them loadable
for label in k2 k3; do
  hf upload "$DEST" "$CAMPAIGN/base/$label" "base/$label" --revision "$STAGING"
done
hf upload "$DEST" "$CAMPAIGN/assemblies" assemblies --revision "$STAGING"
hf upload "$DEST" "$CAMPAIGN/campaign_summary.json" campaign_summary.json \
  --revision "$STAGING"

# promote once, when every fragment is up: one ref move, not eleven commits
hf repo branch merge "$DEST" "$STAGING" main   # or open a PR and merge it
```

Publishing onto a staging ref and promoting once is what keeps a consumer from
fetching a half-published family: until the ref moves, `main` has nothing new.
Xet deduplication means the second base's identical skeleton content should cost
almost no extra WAN. Upload from a CPU process, never from a GPU node.

**Release the GPUs before the final drain — this is a hard gate, not advice.**
Only the stage trees can be uploaded per window (647.6 GB of the 1.184 TB for
`lean`). Both bases, including the skeleton copy that makes them loadable, are
written by `finalize` and can only be uploaded after it: **536.6 GB, 3.4 h at 350
Mbps**. `finalize` itself needs no GPU (§4.6). Eight GPUs left running through
finalize and that drain cost **[DERIVED] $27** for nothing, which is 17% of the
whole campaign budget. The volume must therefore be **persistent and detachable**:
terminate the GPU instances, then attach the volume to a CPU VM (or keep the
cheapest single instance) for finalize and the drain. That handoff only works if
everything finalize needs already lives on the volume, which is why `$KEY`,
`$REPO` and `$SRC`'s metadata are all under `/data` in "Set once". Check before
releasing anything:

```bash
for path in "$KEY" "$REPO/tools/fq_assemble_lora.py" \
            "$SRC/model.safetensors.index.json" "$SRC/config.json"; do
  [[ -e $path ]] || { echo "NOT on the volume: $path"; exit 1; }
done
findmnt -no TARGET --target "$CAMPAIGN"    # must be the persistent volume
```

### 4.6 Finalize once, at the end

```bash
fq-assemble-lora finalize "${COMMON[@]}" "${IDENTITY[@]}" --sign-key "$KEY"
```

Needs only `config.json` and the source index — deleted source payloads do not
have to be restaged — but it **re-reads and re-hashes the whole campaign**
(1.146 TB for `lean`, 1.332 TB for `dag`): a digest computed by the process that
wrote a file proves nothing about the file that survived. **Measured 908 MB/s**
(8.72 GB in 10.26 s, warm cache), so budget `min(908 MB/s, volume read rate)`:
21 min on a 1 GB/s volume, 38 min at 500 MB/s. **Pause the GPU fleet first**;
this pass needs no GPU.

Finalize is **resumable**: it may be re-run any number of times on the same
campaign, so a preemption in the middle of that pass costs only the re-read.
(Publishing a base hardlinks the skeleton into it, and finalize accepts those
names on re-entry while still refusing shards from any other block layout.)

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
