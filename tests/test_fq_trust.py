"""Tests for fq_trust: the trust root, fingerprint pinning, and the rule that
decoding a payload is never verifying a signature.

The peer review's finding 5 in test form: a repo that swaps the key, the
signature and the bytes together must fail, and a placeholder signature must
never be mistaken for a valid one.
"""
import base64
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import fq_trust  # noqa: E402
from fq_trust import TrustError  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent


def make_key(tmp_path, name="k"):
    from nacl.signing import SigningKey

    key = SigningKey.generate()
    return key, key.verify_key.encode().hex()


def envelope(key, payload: dict, keyid: str = None) -> dict:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {"payload": base64.b64encode(raw).decode(),
            "signature": base64.b64encode(key.sign(raw).signature).decode(),
            "keyid": keyid or key.verify_key.encode().hex()}


def write_root(tmp_path, *records) -> Path:
    lines = ["# test trust root"]
    for fp, key_id, status in records:
        lines.append(f"{fp}  {key_id}  {status}  2026-08-10  segments")
        (tmp_path / f"{key_id}.ed25519.pub").write_text(fp + "\n")
    (tmp_path / "FINGERPRINTS").write_text("\n".join(lines) + "\n")
    return tmp_path / "FINGERPRINTS"


# ------------------------------------------------------------ the shipped root

def test_repo_trust_root_is_present_and_parsable():
    """The published trust root is the whole point; it must exist in git."""
    root = fq_trust.TrustRoot.load(REPO_ROOT / "keys" / "FINGERPRINTS")
    assert root.active(), "keys/FINGERPRINTS lists no active signer"
    for fp in root.active():
        assert len(fp) == 64 and root.key_id(fp)
        pub = REPO_ROOT / "keys" / f"{root.key_id(fp)}.ed25519.pub"
        assert pub.exists() and pub.read_text().strip() == fp


def test_default_root_is_found_without_arguments(monkeypatch):
    monkeypatch.delenv("FQ_TRUST_ROOT", raising=False)
    assert fq_trust.TrustRoot.load().active()


def test_env_var_overrides_and_missing_explicit_root_is_fatal(tmp_path, monkeypatch):
    _, fp = make_key(tmp_path)
    path = write_root(tmp_path, (fp, "env-key", "active"))
    monkeypatch.setenv("FQ_TRUST_ROOT", str(path))
    assert fq_trust.TrustRoot.load().active() == [fp]
    with pytest.raises(TrustError, match="trust root not found"):
        fq_trust.TrustRoot.load(tmp_path / "nope" / "FINGERPRINTS")


# ------------------------------------------------------------- fingerprints

def test_fingerprint_resolution_forms(tmp_path):
    _, fp = make_key(tmp_path)
    root = fq_trust.TrustRoot.load(write_root(tmp_path, (fp, "k1", "active")))
    assert root.resolve(fp) == fp
    assert root.resolve(fp.upper()) == fp
    assert root.resolve(fp[:16]) == fp
    assert root.resolve("k1") == fp
    assert root.resolve(str(tmp_path / "k1.ed25519.pub")) == fp


def test_short_or_unknown_prefixes_are_refused(tmp_path):
    _, fp = make_key(tmp_path)
    root = fq_trust.TrustRoot.load(write_root(tmp_path, (fp, "k1", "active")))
    with pytest.raises(TrustError, match="at least 16 hex"):
        root.resolve(fp[:8])
    with pytest.raises(TrustError, match="no key with that prefix"):
        root.resolve("dead" * 4)
    with pytest.raises(TrustError, match="not a fingerprint"):
        root.resolve("some-other-name")


def test_full_fingerprint_works_against_an_older_clone(tmp_path):
    """A fingerprint handed over out of band must still pin, even if the
    local checkout has never heard of the key."""
    _, known = make_key(tmp_path)
    _, unknown = make_key(tmp_path, "u")
    root = fq_trust.TrustRoot.load(write_root(tmp_path, (known, "k1", "active")))
    assert root.resolve(unknown) == unknown
    assert root.status(unknown) is None


# ---------------------------------------------------------------- signatures

def test_valid_signature_verifies(tmp_path):
    key, fp = make_key(tmp_path)
    payload = fq_trust.verify_signature(envelope(key, {"a": 1}), fp)
    assert payload == {"a": 1}


def test_placeholder_signature_is_rejected(tmp_path):
    key, fp = make_key(tmp_path)
    env = envelope(key, {"a": 1})
    env["signature"] = "AA=="
    with pytest.raises(TrustError, match="64 bytes"):
        fq_trust.verify_signature(env, fp)


def test_tampered_payload_is_rejected(tmp_path):
    key, fp = make_key(tmp_path)
    env = envelope(key, {"a": 1})
    env["payload"] = base64.b64encode(b'{"a":2}').decode()
    with pytest.raises(TrustError, match="BAD SIGNATURE"):
        fq_trust.verify_signature(env, fp)


def test_key_swap_is_rejected_even_though_it_is_self_consistent(tmp_path):
    """The repo-compromise case: attacker re-signs everything with their own
    key and updates the manifest.  Pinning is what catches it."""
    attacker, attacker_fp = make_key(tmp_path, "evil")
    _, real_fp = make_key(tmp_path, "real")
    env = envelope(attacker, {"a": 1})
    # self-consistent under the attacker's key...
    assert fq_trust.verify_signature(env, attacker_fp) == {"a": 1}
    # ...and worthless against the pinned one
    with pytest.raises(TrustError, match="refusing"):
        fq_trust.verify_signature(env, real_fp)


def test_malformed_envelopes_are_rejected(tmp_path):
    key, fp = make_key(tmp_path)
    for broken, match in [
        ({}, "keyid"),
        ({"payload": "x", "signature": "y", "keyid": "zz"}, "keyid"),
        ({"payload": "!!!", "signature": "AA==", "keyid": fp}, "base64"),
        ({"payload": base64.b64encode(b"{}").decode(), "keyid": fp}, "signature"),
    ]:
        with pytest.raises(TrustError, match=match):
            fq_trust.verify_signature(broken, fp)


def test_decode_payload_is_named_so_it_cannot_be_mistaken(tmp_path):
    key, fp = make_key(tmp_path)
    env = envelope(key, {"a": 1})
    env["signature"] = "AA=="
    assert fq_trust.decode_payload(env) == {"a": 1}  # decoding still works
    with pytest.raises(TrustError):                  # verifying does not
        fq_trust.verify_signature(env, fp)


# --------------------------------------------------------------------- rungs

class Args:
    def __init__(self, **kw):
        self.trust_signer = kw.get("trust_signer")
        self.trust_root = kw.get("trust_root")
        self.allow_unpinned_signer = kw.get("allow_unpinned_signer", False)
        self.insecure_skip_signatures = kw.get("insecure_skip_signatures", False)


def test_rung_pinned(tmp_path):
    _, fp = make_key(tmp_path)
    root = write_root(tmp_path, (fp, "k1", "active"))
    v = fq_trust.Verifier.from_args(Args(trust_signer=fp[:16], trust_root=root))
    assert v.rung == fq_trust.RUNG_PINNED and v.fingerprint == fp
    assert "pinned signer" in v.summary()


def test_rung_trust_root_accepts_only_listed_active_keys(tmp_path):
    _, good = make_key(tmp_path)
    _, other = make_key(tmp_path, "o")
    root = write_root(tmp_path, (good, "k1", "active"))
    v = fq_trust.Verifier.from_args(Args(trust_root=root),
                                    manifest={"signer_pubkey": good})
    assert v.rung == fq_trust.RUNG_TRUST_ROOT
    with pytest.raises(TrustError, match="not listed"):
        fq_trust.Verifier.from_args(Args(trust_root=root),
                                    manifest={"signer_pubkey": other})


def test_revoked_keys_are_refused_retroactively(tmp_path):
    _, fp = make_key(tmp_path)
    root = write_root(tmp_path, (fp, "k1", "revoked"))
    with pytest.raises(TrustError, match="REVOKED"):
        fq_trust.Verifier.from_args(Args(trust_signer=fp, trust_root=root))
    with pytest.raises(TrustError, match="revoked"):
        fq_trust.Verifier.from_args(Args(trust_root=root),
                                    manifest={"signer_pubkey": fp})


def test_rung_unpinned_says_so_out_loud(tmp_path):
    key, fp = make_key(tmp_path)
    root = write_root(tmp_path, (fp, "k1", "active"))
    v = fq_trust.Verifier.from_args(
        Args(allow_unpinned_signer=True, trust_root=root),
        manifest={"signer_pubkey": fp})
    v.verify_envelope(envelope(key, {"a": 1}))
    assert v.rung == fq_trust.RUNG_UNPINNED
    assert "UNPINNED" in v.summary() and "unproven" in v.summary()


def test_rung_none_proves_nothing_and_admits_it(tmp_path):
    key, _ = make_key(tmp_path)
    v = fq_trust.Verifier.from_args(Args(insecure_skip_signatures=True))
    env = envelope(key, {"a": 1})
    env["signature"] = "AA=="
    assert v.verify_envelope(env) == {"a": 1}
    assert "NOT CHECKED" in v.summary()


def test_no_root_and_no_pin_is_an_error_not_a_default_yes(tmp_path):
    _, fp = make_key(tmp_path)
    empty = tmp_path / "FINGERPRINTS"
    empty.write_text("# nothing active here\n")
    # a key the trust root has never heard of is refused, not defaulted to
    with pytest.raises(TrustError, match="not listed"):
        fq_trust.Verifier.from_args(Args(trust_root=empty),
                                    manifest={"signer_pubkey": fp})
    # and with nothing to go on at all, the tool says what to do instead
    with pytest.raises(TrustError, match="--trust-signer"):
        fq_trust.Verifier.from_args(Args(trust_root=empty), manifest={})


def test_retired_keys_still_verify_but_warn(tmp_path, capsys):
    key, fp = make_key(tmp_path)
    root = write_root(tmp_path, (fp, "k1", "retired"))
    v = fq_trust.Verifier.from_args(Args(trust_root=root),
                                    manifest={"signer_pubkey": fp})
    assert v.verify_envelope(envelope(key, {"a": 1})) == {"a": 1}
    assert "RETIRED" in capsys.readouterr().err


def test_check_fingerprints_script_passes_on_the_repo():
    import subprocess

    r = subprocess.run([sys.executable, str(REPO_ROOT / "keys" / "check_fingerprints.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "trust root OK" in r.stdout
