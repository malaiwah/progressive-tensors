# Reconstruction proof: three community GLM-5.2 quants, taken apart and put back together

**Claim under proof:** a published quant, decomposed into Progressive Tensors
segments, can be reassembled — and the reassembly can be *checked*, not just
asserted. Byte-identity is claimed exactly where it holds; where a byte claim
is not applicable, the evidence is bounded numeric similarity of the
dequantized weights, measured against the BF16 original with the reference
decoder.

Every row below came out of `tools/fq_verify.py` against pinned public
sources, and every command is reproducible.

## Subjects

| Quant | Revision pin | Layout | Segments |
|---|---|---|---|
| [`brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw`](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw) | `9297b9f1` | `per_expert_v1`, flat K3 | full repack, layers 3–78 → [`malaiwah/GLM-5.2-EXL3-FQ-segments`](https://huggingface.co/malaiwah/GLM-5.2-EXL3-FQ-segments) |
| [`willfalco/GLM-5.2-EXL3-TR3-3.36bpw`](https://huggingface.co/willfalco/GLM-5.2-EXL3-TR3-3.36bpw) | `8d9aa923` | `per_expert_v1`, mixed K3/K4 | K4 experts primed by ranged read, layers 3–10 |
| [`willfalco/GLM-5.2-EXL3-TR3-3.42bpw`](https://huggingface.co/willfalco/GLM-5.2-EXL3-TR3-3.42bpw) | `ae68c659` | `shared_h_v1`, mixed K3/K4 | all experts + shared profiles primed, layers 3–10, plus the expanded per-expert view |
| [`zai-org/GLM-5.2`](https://huggingface.co/zai-org/GLM-5.2) | `b4734de4` | BF16 base | ground truth (similarity only) |

The layer windows are honest scope statements, not rounding: the willfalco
families were primed for layers 3–10, so their byte-identity is proven per
fragment inside that window. Whole-shard identity is applicable — and
claimed — only for the fully repacked brandonmusic family.

## Reconstruction table

| # | Reconstruction | Proof granularity | Result |
|---|---|---|---|
| 1 | brandonmusic 3.0 bpw — **all 76 MoE shards** (layers 3–78) reassembled from K3 segments + the source's non-expert bytes | whole-shard sha256 vs the source `MANIFEST.sha256` | **PASS 76/76 sha256-identical** — 278.5 GB of expert bytes came from segments, 31.7 GB header/attention/dense pass-through from source |
| 2 | the same segments vs their **signed attestations** (sampled layers 6, 17, 38) | per-expert span sha256 + ed25519 signature | **PASS** — 768 expert spans re-hashed, 0 mismatches, signatures verified |
| 3 | willfalco 3.36 bpw K4 fragments, layers 3–10 | fragment bytes vs **freshly range-read** source bytes at the pin | **PASS** — 722/722 expert spans match their attestations; 24/24 sampled experts byte-identical to a fresh re-fetch (469 MB re-read) |
| 4 | willfalco 3.42 bpw shared-h fragments + per-layer shared profiles, layers 3–10 | fragment bytes vs fresh ranged reads; profiles compared in full | **PASS** — 2048/2048 expert spans match their attestations; 48/48 sampled experts and 8/8 shared profiles byte-identical to a fresh re-fetch (804 MB re-read) |
| 5 | willfalco 3.42 bpw **expanded** per-expert view (`derived-from`) | every expert re-derived from parent segment + profile, every tensor byte-compared, parent sha256 pins re-hashed | **PASS 2048/2048 experts** (16 segments) — 32.8 GB verbatim + 0.30 GB replicated H-rows, 0 tensor mismatches, all parent pins intact |
| 6 | shared-h decode vs expanded decode of the same expert | bitwise equality of the dequantized fp16 weights | **EQUAL** on all 72 sampled (expert, projection) pairs |

Rows 1–5 are byte proofs. Row 6 and the table below are the numeric rung:
fragments from *different encodes* cannot be byte-identical, so the claim
there is bounded similarity instead of identity.

## Two K4 encodes from one uploader, plus cross-uploader comparisons

The evidence compares three published encodes: brandonmusic's flat K3,
willfalco's 3.36 bpw (`per_expert_v1`), and willfalco's 3.42 bpw
(`shared_h_v1`, a different layout and K partition). The two K4 rows are
different encodes/layouts from the **same uploader**. This measurement does
not establish that they were independently produced; the cross-uploader
comparisons are brandonmusic K3 versus each willfalco K4 encode.

All three were decomposed by the same segment tooling and decoded through the
same reference path (`LinearEXL3.get_weight_tensor`: `ext.reconstruct` +
Hadamard-128 + `diag(suh)/diag(svh)`), then compared in fp64 against the BF16
original.

24 experts spread over layers 3–10 (seed 42), all three projections,
4 rank slices each → n = 72 per pair.

| Pair | cos mean / min | relF mean / max | max abs | reading |
|---|---|---|---|---|
| 3.42 expanded ≡ 3.42 shared-h | **1.00000 / 1.00000** (bitwise EQUAL) | 0.0000 / 0.0000 | 0.0000 | exact expansion, proven at the weights |
| willfalco 3.42 K4 vs BF16 | 0.99684 / 0.99625 | 0.0793 / 0.0866 | 0.0129 | quality of this K4 encode |
| willfalco 3.36 K4 vs BF16 | 0.99684 / 0.99626 | 0.0792 / 0.0864 | 0.0142 | quality of this K4 encode |
| willfalco 3.42 K4 vs willfalco 3.36 K4 | 0.99370 / 0.99255 | 0.1120 / 0.1221 | 0.0196 | the two K4 encodes against each other |
| brandonmusic K3 vs BF16 | 0.98761 / 0.98566 | 0.1569 / 0.1692 | 0.0284 | the K3 tier, for scale |
| brandonmusic K3 vs willfalco 3.42 K4 | 0.98452 / 0.98229 | 0.1754 / 0.1881 | 0.0340 | cross-uploader, cross-bitrate |
| brandonmusic K3 vs willfalco 3.36 K4 | 0.98466 / 0.98245 | 0.1746 / 0.1873 | 0.0305 | cross-uploader, cross-bitrate |

Three things fall out of those numbers:

1. **The two willfalco K4 encodes have nearly the same measured quality.**
   Both sit at relative Frobenius error ≈ 0.079 from the BF16 original — a
   spread of 0.0001 in this sample. That supports comparability of these two
   published encodes; it does not establish producer independence.
2. **They are different bytes.** The two K4 encodes differ from each other
   (relF 0.112), while each is measured against the same ground truth (0.079).
   This is consistent with distinct encodes, but provenance claims stop at the
   signed source records and uploader identity.
3. **The bitrate ordering is monotone in this sample.** K3 sits at relF 0.157,
   K4 at 0.079 — a 1.98× step, matching the campaign's measured per-K error
   ladder. A mixed recipe can upgrade a K3 base with a compatible K4 fragment,
   subject to the source's explicit provenance and layout constraints.

**What is not claimed:** K4 fragments from different encodes are not
bit-identical, and this table does not prove their producers are independent.
The bounded claim is the reported similarity to the BF16 original and between
the sampled encodes.

## What each proof rung establishes

- **`repack-of`** (rows 1–4): the segment bytes *are* the published quant's
  bytes, re-addressed per expert. The strongest claim available, and
  falsifiable by anyone with one ranged read.
- **`derived-from`** (row 5): the expanded view adds only replicated shared
  rows; every byte matches either the parent segment or the parent profile,
  and both parents are pinned by sha256. Exact — but a derivation, labeled as
  one, never as `repack-of`.
- **`equivalence-of` evidence** (row 6 + the producer table): bitwise equality
  for the expansion, bounded similarity across encodes. Numeric evidence, not
  a byte claim; the table reports measured bounds and nothing more.

## Reproduce

```bash
# rows 1-2 — needs the source snapshot on disk
python tools/fq_verify.py --identity --segments <k3-segments> \
  --source <brandonmusic-snapshot> --attest 3 --seed 42

# rows 3-4 — network only: fresh ranged reads of the pinned revisions
python tools/fq_verify.py --identity --segments <segments-336> --sample 3 --seed 42
python tools/fq_verify.py --identity --segments <segments-342>/shared-h --sample 3 --seed 42

# row 5 — local, full coverage
python tools/fq_verify.py --identity --segments <segments-342>/expanded \
  --parent <segments-342>/shared-h

# row 6 + the producer table — GPU with the exllamav3 extension (~1 min, <1 GB VRAM)
python tools/fq_verify.py --similarity \
  --family bm-k3=<k3-segments>,k=3 \
  --family 342-k4=<segments-342>/expanded,k=4 \
  --family 342-sh=<segments-342>/shared-h,k=4 \
  --family 336-k4=<segments-336>,k=4 \
  --bf16 <zai-org-GLM-5.2-snapshot> --layers 3-10 --experts 24 --seed 42
```

`fq_verify --identity` picks its check automatically: whole-shard sha256 when
a local `--source` snapshot is given, fresh-ranged-read fragment comparison
for a family primed from a remote repo, and full re-derivation for a
`derived-from` family. Machine-readable JSON reports for every run behind this
page are pinned at
[`research/fungible-quant/runs/0c-campaign/verify/` revision `69fbef710e558e9cf8e2ad634eccc774f9a806fb`](https://github.com/malaiwah/vllm-voipmonitor/tree/69fbef710e558e9cf8e2ad634eccc774f9a806fb/research/fungible-quant/runs/0c-campaign/verify/).
