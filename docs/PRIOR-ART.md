# Prior art and where this project actually sits

Commissioned independent prior-art review, 2026-08-10. Summarized here in
full because a project like this should be honest about what it did *not*
invent. **Every individual ingredient below has prior art. The
contribution is the integration, and only the integration.**

## What we do not claim

We are **not** claiming any of these, all of which predate us:

- mixed-precision LLM quantization
- per-expert bit-width allocation for MoE
- progressive / selectable-precision checkpoints
- runtime promotion and demotion of expert precision
- signed model fragments, content addressing, or ranged tensor reads
- byte-level safetensors extraction and reconstruction

## The closest prior art, by layer

| Area | Existing work | What is different here |
|---|---|---|
| **Expert-wise mixed precision** | [MC-MoE](https://arxiv.org/abs/2410.06270), [MxMoE](https://arxiv.org/abs/2505.05799), [MoPEQ](https://arxiv.org/abs/2509.02512), [GEMQ](https://arxiv.org/abs/2605.23078) — per-expert bit widths from sensitivity, routing frequency, hardware cost, global optimization | We *reuse already-encoded expert alternatives from multiple producers* rather than quantizing one checkpoint per an optimizer's output |
| **Progressive / selectable precision** | [Any-Precision LLM](https://arxiv.org/abs/2402.10517), [BitStack](https://arxiv.org/abs/2410.23918), [Matryoshka Quantization](https://arxiv.org/abs/2502.06786), [QStore](https://arxiv.org/abs/2505.04081) — nested bitplanes, residual blocks, joint low/high storage | Our tiers are *native, independently valid encodings* that substitute for each other, not slices or residuals of one specially-constructed representation — and the output needs no custom runtime format |
| **Runtime precision change** | [HOBBIT](https://arxiv.org/abs/2411.01433) (low-precision experts on cache miss), [DynaExq](https://arxiv.org/abs/2511.15015) (hotness-aware promotion/demotion) | Runtime promotion is **not novel**. What we can add is a standardized, provenance-aware *source* for the alternate representations these systems need |
| **Tensor-aware storage** | [Git-Theta](https://proceedings.mlr.press/v202/kandpal23b.html) (parameter groups as reusable blobs), [HF Xet](https://huggingface.co/docs/hub/xet/deduplication) (content-defined chunking, CAS, ranged reconstruction) | A *semantic* coordinate system for interchangeable quantization variants, rather than versions of one model or opaque byte chunks |
| **Signed model parts** | [OCI image-spec](https://github.com/opencontainers/image-spec/blob/main/manifest.md), [KitOps ModelKits](https://kitops.org/docs/modelkit/spec/), [in-toto](https://github.com/in-toto/specification/blob/master/in-toto-spec.md), [OpenSSF Model Signing](https://github.com/sigstore/model-transparency) | Quantization-specific *compatibility and assembly* semantics. Raw Ed25519 signing is not novel — and we are migrating to DSSE/in-toto envelopes precisely so the plumbing is theirs, not ours |
| **Quantization attestations** | 2026 IETF individual draft [draft-sharif-ai-model-lifecycle-attestation](https://datatracker.ietf.org/doc/draft-sharif-ai-model-lifecycle-attestation/00/) — attestations binding source models, quantized outputs, tools, per-layer measurements | Per-*expert* alternatives, community resolution, recipe selection, and proof-carrying final assembly |
| **Extraction / reassembly** | [safetensors metadata + byte offsets](https://github.com/huggingface/safetensors/blob/main/docs/source/metadata_parsing.mdx), HF sharding, GGUF split/merge, [Tensorizer](https://github.com/coreweave/tensorizer) | Selecting among *compatible alternatives from independent sources* instead of reconstructing one predetermined file |

Reviewers will most likely raise **Git-Theta** (closest systems ancestor),
**BitStack** (closest collision with "progressive"), and
**MC-MoE / GEMQ** (closest collision with expert-wise recipes).

## Honest novelty assessment

| Layer | Novelty |
|---|---|
| EXL3 quantization itself | **Low** — it is our substrate, not our work |
| Per-expert mixed-K allocation | **Low** — anticipated by several MoE papers |
| One model family, several memory budgets | **Low–moderate** — Any-Precision, BitStack, MatQuant, QStore are ancestors |
| Runtime expert precision promotion/demotion | **Low** — HOBBIT, DynaExq |
| Content addressing, range reads, signatures | **Low** individually |
| Byte-level extraction/reconstruction mechanics | **Moderate** engineering, established file mechanics |
| Independently publishable, compatible, attested **native** quant segments | **Moderate–high** |
| Recipe-driven linking into stock safetensors **without requantization** | **Moderate–high** in combination with community fragments + provenance |

Algorithmic novelty: **low**. Protocol/systems novelty: **moderate–high**.

## The one sentence we are comfortable defending

> We are not aware of an earlier open system that lets consumers assemble
> a loader-native mixed-K EXL3 safetensors checkpoint from independently
> published, digest-pinned and attested quantization segments.

More precisely, the contribution is the *combination*:

> An immutable fragment identified by source model/revision, semantic
> tensor-or-expert identity, encoding family and precision, tensor
> inventory, byte-range digests and producer provenance — plus a
> deterministic linker that validates compatible fragments selected by a
> policy and emits a conventional checkpoint whose output digest derives
> from the complete assembly evidence.

No individual ingredient of that sentence is the invention.

## What this review changed in the project

1. **DSSE / in-toto envelope adoption** moved from "designed" to
   committed direction — the point is that signature plumbing should be
   the ecosystem's, so the schema and the linker are what we contribute.
2. **Positioning language** — "first"-style claims removed everywhere;
   the sentence above is the ceiling of what we assert.
3. **Evaluation axes** for the work now target *supply-chain economics*
   rather than quantizer quality: storage/transfer across many recipes
   vs separate checkpoints and vs Xet alone; reconstruction and
   verification cost; byte identity and loader compatibility; reuse
   across independently produced tiers; allocation quality vs MC-MoE /
   GEMQ; multi-budget serving vs BitStack / QStore; malicious or
   incompatible provider handling; and **at least two independent
   producers** emitting interoperable fragments.
4. **Two-producer interoperability** became an explicit goal rather than
   an implication — a single-producer ecosystem does not demonstrate the
   claim.

## Scope of the review

Bounded technical and patent search, not a legal novelty, patentability,
or freedom-to-operate opinion. Relevant patent families exist around
signed ML models, model-layer patches, decomposed fragments and differing
bit widths (e.g. US11574245B2, US11531932B2, US11604961B2, US12015526B2);
an examiner could combine those with OCI/in-toto and expert-wise
quantization art. This repository is already public, which starts
disclosure clocks — see [USPTO guidance](https://www.uspto.gov/patents/basics/international-protection/filing-patents-abroad)
and [EPC Article 54](https://www.epo.org/en/legal/epc/2020/a54.html).
Anyone with filing interests should consult counsel rather than rely on
this document.
