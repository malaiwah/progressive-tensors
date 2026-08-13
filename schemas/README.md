# `schemas/` — what the documents actually contain

Every JSON document Progressive Tensors emits has a schema here, **derived
from real emitted files, not from intentions**: the shapes below were read
back out of `~/fq-segments/GLM-5.2-EXL3-FQ` (76 GLM-5.2 layer segments),
`~/fq-0c/fruit-segments` (a four-K local encode), `~/fq-primed/segments-33{6,42}`
(two primed community quants, including the shared-H and expanded families)
and the recipes in `~/fq-0c/*.json`, then tightened until they still accepted
all of them. `tests/test_schemas.py` re-validates freshly emitted documents on
every CI run, so a tool that changes its output without updating its schema
fails the build.

| Schema | Document | Emitted by |
|---|---|---|
| [`fq-segment-1`](fq-segment-1.schema.json) | the `__metadata__` block inside `layer-LLL.kK.safetensors` | `fq_repack`, `fq_prime`, `fq_fetch` |
| [`fq-segment-index-1`](fq-segment-index-1.schema.json) | `index-kK.json` — per-layer, per-expert `[lo, hi)` byte spans | `fq_repack`, `fq_prime`, `fq_fetch` |
| [`fq-attestation-1`](fq-attestation-1.schema.json) | one line of `attestations/layer-LLL.kK.jsonl` | `fq_repack`, `fq_prime` |
| [`fq-manifest-1`](fq-manifest-1.schema.json) | `fq-manifest.json` (family, fetched-subset, or multi-source root) | `fq_repack`, `fq_prime`, `fq_fetch` |
| [`fq-policy-2`](fq-policy-2.schema.json) | the recipe: one K per expert per layer | you, or a solver |
| [`fq-release-1`](fq-release-1.schema.json) | `fq-release.json` — one signature over every file of a release | `fq_release` |
| [`fq-cartridge-assembly-2`](fq-cartridge-assembly-2.schema.json) | signed campaign plan binding residual shards to exact base bytes | `fq_assemble_lora` |
| [`fq-cartridge-adapter-3`](fq-cartridge-adapter-3.schema.json) | self-contained, digest-pinned runtime cartridge manifest | `fq_combine_cartridges` |
| [`fq-msrt-base-compatibility-2`](fq-msrt-base-compatibility-2.schema.json) | canonical root over per-layer TP-invariant base identities | `fq_assemble_lora`, verified by `fq_combine_cartridges --base` |
| [`fq-msrt-base-layer-compatibility-1`](fq-msrt-base-layer-compatibility-1.schema.json) | canonical logical tensor payload hashed as one layer's TP-invariant base identity | `fq_assemble_lora`, verified by `fq_combine_cartridges --base` |

One document deliberately has no schema yet: the `assembly-of` record
`fq_assemble` writes (`fq-assembly.json`, declaring `fq-attestation/2`).
It pins the recipe, every consumed fragment and every produced shard, and
its shape is still settling alongside the verification work — a schema for
it lands when it stops moving, rather than freezing a guess.

## Stability

Every version string (`fq-segment/1`, `fq-policy/2`,
`fq-cartridge-adapter/3`, …) names a distinct contract. Required fields never
change meaning within that version. A rename, retype, changed numerical
meaning, or newly required field is a new version.

The schema itself declares whether a particular contract is additive or
closed. Existing discovery/provenance formats retain extension points where
their schema permits `additionalProperties`. In contrast,
`fq-cartridge-assembly/2` and `fq-cartridge-adapter/3` are **closed executable
contracts**: producers and consumers reject unknown fields at every defined
object boundary. Adding even an optional field to either contract requires a
new schema version. Adapter/3 and assembly/2 are being coordinated pre-merge
with their first runtime consumer and have not been published as stable
artifacts; once merged, their exact closed shapes are immutable.

Concretely, the compatibility rules we hold ourselves to now:

- **Additive only where the schema says so.** New optional fields may appear in
  open extension points (they do:
  `fq_prime` adds provenance keys, `fq_fetch` adds subset markers). Consumers
  ignore unknown keys only in those schema-declared open objects.
- **Closed means exact.** Consumers of the cartridge adapter/assembly profiles
  reject missing and unknown fields. Evolution is a new `/N`, never a silent
  in-version addition.
- **Required fields never change meaning.** Renaming, retyping or
  repurposing a required field is a version bump, full stop.
- **Predicates are enumerated on purpose.** `repack-of`, `derived-from`,
  `encode-of`, `equivalence-of` mean specific things (see
  [`../TRUST.md`](../TRUST.md)); adding a fifth is a schema change reviewed
  as such, because each one is a different claim about how the bytes came to
  exist.

## Per-K sparse coverage

The v1 `per_k[K].layers` field is a legacy inclusive `[minimum, maximum]`
convenience. It cannot establish coverage for a sparse K: when
`per_k[K].layer_coverage` is absent, consumers must read the signed
`index-kK.json` keys.

New producers write `per_k[K].layer_coverage` as
`{"schema": "fq-layer-coverage/1", "layers": [...]}`. Its `layers` is the
**exact sorted membership list** and is the coverage authority: a layer not
listed is unavailable for that K, including values between two listed layers.

## Validating

```bash
uv run --with jsonschema python - <<'EOF'
import json, sys
from jsonschema import Draft202012Validator
schema = json.load(open("schemas/fq-manifest-1.schema.json"))
doc = json.load(open("segments/fq-manifest.json"))
Draft202012Validator(schema).validate(doc)
print("ok")
EOF
```

For the two signed documents (`fq-attestation-1`, `fq-release-1`) the root
schema validates the **envelope**; the signed payload is
`$defs/payload`, validated after base64-decoding it. Validate in that order,
and — this matters — **validate the payload you decoded from a signature you
already verified**. Schema conformance is not authenticity: a well-formed
document with a bad signature is a well-formed lie.
