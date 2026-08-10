# Progressive Tensors

**Mixed-K quants as a service — assemble your own EXL3 checkpoint from
shared, attested, per-expert segments. Pure safetensors. No new format.**

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
  one ranged read.
- **Reassembly is byte-faithful** — assembling the all-K3 recipe from the
  brandonmusic-derived segments reproduces the original checkpoint shards
  **sha256-identical** (verified across all 79 shards of GLM-5.2).
- **Sharing is deduplicating** — identical fragments are identical bytes
  (and Hugging Face Xet dedupes at chunk level), so N overlapping quants
  cost far less than N full downloads.
- **No special loader required** — the assembled output is a normal
  checkpoint for the target runtime (Gilded Gnosis vLLM / EXL3 stack).
  A runtime *progressive loader* and live bit-width reallocation are being
  built on top (see Research), but everything in this repo works today
  with plain files.

## Quickstart: verify + reassemble

Segments for **GLM-5.2** (from
[`brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw`](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw)
at pinned commit `9297b9f1`) live at
[`malaiwah/GLM-5.2-EXL3-FQ-segments`](https://huggingface.co/malaiwah/GLM-5.2-EXL3-FQ-segments).

```bash
git clone https://github.com/malaiwah/progressive-tensors
cd progressive-tensors
uv venv && uv pip install pynacl pytest numpy huggingface_hub
uv run pytest tests/            # the tools' own test suite

# fetch segments (or range-read just the experts you need via index-k3.json)
hf download malaiwah/GLM-5.2-EXL3-FQ-segments --local-dir ./segments

# your recipe: which K per expert, per layer (fq-policy/2 JSON)
python - <<'EOF'
import json
json.dump({"schema": "fq-policy/2",
           "bits_per_expert": {str(l): [3]*256 for l in range(3, 79)}},
          open("recipe-all-k3.json", "w"))
EOF

# assemble a bootable checkpoint (dense tensors come from the source repo)
uv run tools/fq_assemble.py \
  --segments ./segments --source <source-checkpoint-dir> \
  --policy recipe-all-k3.json --out ./my-checkpoint

sha256sum ./my-checkpoint/model-layer-030.safetensors
#  -> identical to the original shard. That's the point.
```

### Disk-saving assembly (`--reflink`)

If segments and output live on a reflink-capable filesystem (XFS with
`reflink=1`, btrfs), add `--reflink` to `fq_assemble.py`: expert-tensor
bytes are then written with `copy_file_range`, letting the kernel share
extents between the segment files and the assembled shards instead of
storing the same bytes twice. Honest caveats:

- **Sharing is the kernel's call.** `copy_file_range` may silently perform
  a plain copy; extent sharing only happens when the filesystem chooses to
  reflink.
- **Savings depend on 4K-block alignment** of identical regions at *both*
  the source and destination offsets. MB-scale fragments usually share
  most interior blocks, but alignment is not guaranteed.
- **Output bytes are always identical** to a non-`--reflink` run — the
  mode changes how bytes move, never what they are. Every region falls
  back automatically to the ordinary copy when `copy_file_range` is
  unavailable or fails (cross-filesystem `EXDEV`, `EOPNOTSUPP`, ...), so
  it is always safe to pass.
- **Local disk space only.** It has no effect on HF/remote transfer or
  storage (Xet chunk-dedupe covers that side).

### Verify provenance without downloading anything big

Each `attestations/layer-LLL.kK.jsonl` line is an ed25519-signed payload:

```python
import base64, json, hashlib
from nacl.signing import VerifyKey

line = json.loads(open("segments/attestations/layer-030.k3.jsonl").read())
payload = json.loads(base64.b64decode(line["payload"]))
manifest = json.load(open("segments/fq-manifest.json"))
VerifyKey(bytes.fromhex(manifest["signer_pubkey"])).verify(
    base64.b64decode(line["payload"]), base64.b64decode(line["signature"]))

# spot-check ONE expert with one ranged read against the segment file:
idx = json.load(open("segments/index-k3.json"))["30"]
lo, hi = idx["experts"]["137"]
blob = open(f"segments/{idx['file']}", "rb").read()[
    idx["body_offset"] + lo : idx["body_offset"] + hi]
assert hashlib.sha256(blob).hexdigest() == payload["expert_sha256"]["137"]
```

The `materials` block pins the source repo + commit + file sha256, so the
same spot-check can be run against the *source* quant with an HTTP range
request — the trust chain is explicit and third-party-verifiable, which is
strictly stronger than "download 300 GB from a named uploader and hope."

## Contribute segments (encode-and-share)

Fragments the community hasn't produced yet (K2 fast-load base, K5
hot-expert tier, K4 for less-hot experts) can be encoded by anyone with
the model's BF16 weights and a captured calibration statistic, then
published with `encode-of` provenance (deterministic encoder + pinned
inputs ⇒ independently re-encodable and countersignable — the
reproducible-builds model for quants). The encoder driver and capture
tooling live in the research branch below and are being promoted into
this repo as they stabilize.

## Status & roadmap

| Piece | Status |
|---|---|
| Segment schema `fq-segment/1`, attestations, manifest | stable API |
| `fq_repack` (checkpoint → segments) / `fq_assemble` (segments+recipe → checkpoint) | working, tested |
| GLM-5.2 K3 base segments on HF | published (brandonmusic lineage) |
| K4 hot-set priming from community mixed quants (3.42/3.36 bpw) | in progress |
| Mixed-size (true mixed-K) assembly + loader metadata | in progress |
| K2/K5 tiers (novel encodes) | in progress |
| Runtime progressive loader + live bit-width reallocation (vLLM/GG) | in development |

## Research

Design docs, verification reports, and the runtime work live in
[`malaiwah/vllm-voipmonitor` branch `claude/gg-overview-exploration-jchgd3`](https://github.com/malaiwah/vllm-voipmonitor/tree/claude/gg-overview-exploration-jchgd3/research/fungible-quant)
(`research/fungible-quant/`). Built and verified on 8× RTX PRO 6000
(SM120) against the Gilded Gnosis vLLM + b12x stack.

MIT licensed. Attestation ≠ endorsement: provenance chains terminate at
the source quant's reputation — they make that trust explicit and
checkable, not unnecessary.
