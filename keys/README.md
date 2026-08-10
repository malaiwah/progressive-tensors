# `keys/` — the out-of-band trust root

Signatures on Progressive Tensors artifacts (`fq-attestation/1` lines,
`fq-release/1` manifests) are ed25519. **The public keys that may make those
signatures are published here, in this git repository, and nowhere else that
counts.**

That separation is the whole point. The artifact repository on Hugging Face
carries the fragments, the attestations, *and* a copy of the signer public
key in `fq-manifest.json`. Reading the key from there proves nothing: whoever
can replace the fragments can replace the attestations and the key in the
same push. Reading the fingerprint from *this* repo, at a commit you have
seen before, means an attacker must additionally forge a commit in a
different system, under a different account, with a public history.

## Files

| File | What it is |
|---|---|
| `FINGERPRINTS` | The authoritative list. One line per key: fingerprint, key id, status, date added, role. Machine-readable, hand-reviewable. |
| `<key-id>.ed25519.pub` | The raw public key, lowercase hex, one line. Same bytes as the fingerprint. |
| `check_fingerprints.py` | CI guard: every `.pub` is well-formed hex, matches a `FINGERPRINTS` record, and no fingerprint or key id is duplicated. |

## Fingerprint format

The fingerprint **is** the ed25519 public key, as 64 lowercase hex
characters. An ed25519 public key is 32 bytes — smaller than the SHA-256
digest that would "fingerprint" it — so hashing it would add a step and
remove information. This is the same string the tools already emit as
`keyid` in every attestation envelope and as `signer_pubkey` in
`fq-manifest.json`.

Tools accept either the full 64-hex fingerprint or an unambiguous prefix of
at least 16 hex characters:

```bash
--trust-signer a58b7bb79ba5845716aa6fee7d54e714ef243c2875f23a617e1ef3247c565525
--trust-signer a58b7bb79ba58457      # 16-hex short form, same key
```

A prefix shorter than 16 hex characters is rejected rather than matched
loosely.

## Verifying a fingerprint you were handed

```bash
git clone https://github.com/malaiwah/progressive-tensors
cd progressive-tensors
git log -p keys/FINGERPRINTS          # when did this key appear, in which commit
grep -i "$FINGERPRINT" keys/FINGERPRINTS
```

If the fingerprint is not in `FINGERPRINTS`, or the commit that introduced it
is newer than the artifacts you are checking and you cannot explain why, stop.

## Adding, rotating, revoking

All three are ordinary commits to `FINGERPRINTS`, reviewed like code:

- **Add** — new line, `status active`, with the date and role.
- **Rotate** — new line for the new key; the old line becomes `retired` with
  a retirement date. Signatures made before that date remain valid; the key
  makes no new ones.
- **Revoke** — the line becomes `revoked`. Revocation is retroactive: every
  signature by that key is treated as unproven, whatever its date, because a
  leaked key can backdate. Re-verify affected artifacts against a key that is
  still `active`, or re-derive them from source.

There is deliberately no online revocation service. `git pull` is the
revocation channel; the freshness of your clone is the freshness of your
revocation data, and that trade is stated rather than hidden.

## What this does not do

Publishing a fingerprint here says **"this key is the one this project
signs with."** It does not say the signed bytes are good, that the source
quant is any good, or that the signer is trustworthy about anything else.
See [`../TRUST.md`](../TRUST.md) for what each rung of the chain does and
does not establish.
