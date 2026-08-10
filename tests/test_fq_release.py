"""Tests for fq_release: one signature over a whole release.

What the release manifest has to buy, and therefore what is tested here:
a single verification covers the complete file set, catches substitution of
any covered file (including the attestation files whose digests everything
else leans on), tolerates a deliberately partial tree, and refuses when the
signer is not the pinned one.
"""
import base64
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import fq_release  # noqa: E402
import fq_repack  # noqa: E402
import fq_trust  # noqa: E402
from test_fq_repack import LAYERS, write_shard  # noqa: E402


@pytest.fixture()
def tree(tmp_path):
    """A small signed segment tree plus its trust root."""
    snap = tmp_path / "snap"
    snap.mkdir()
    for i, layer in enumerate(LAYERS):
        write_shard(snap / f"model-layer-{layer:03d}.safetensors", layer, scramble=bool(i))
    out = tmp_path / "segments"
    key = tmp_path / "sign.key"
    assert fq_repack.main([
        "--snapshot", str(snap), "--source-repo", "test/src", "--revision", "cafe",
        "--base-model", "test/base", "--out", str(out), "--sign-key", str(key)]) == 0
    pub = json.loads((out / "fq-manifest.json").read_text())["signer_pubkey"]
    root = tmp_path / "FINGERPRINTS"
    root.write_text(f"{pub}  test-signer  active  2026-08-10  segments\n")
    return out, key, pub, root


def build(tree, **kw):
    out, key, pub, root = tree
    argv = ["build", "--dir", str(out), "--release", "test 0.1.0",
            "--sign-key", str(key)]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    assert fq_release.main(argv) == 0
    return out / "fq-release.json"


def verify(tree, target=None, expect=0, **flags):
    out, key, pub, root = tree
    argv = ["verify", "--dir", str(target or out), "--trust-signer", pub,
            "--trust-root", str(root)]
    if (target or out) != out:
        argv += ["--release", str(out / "fq-release.json")]
    for k, v in flags.items():
        argv += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
    assert fq_release.main(argv) == expect
    return expect


def test_build_covers_every_published_file(tree):
    out = tree[0]
    rel = json.loads(build(tree).read_text())
    payload = json.loads(base64.b64decode(rel["payload"]))
    assert payload["schema"] == "fq-release/1"
    names = set(payload["files"])
    for layer in LAYERS:
        assert f"layer-{layer:03d}.k3.safetensors" in names
        assert f"attestations/layer-{layer:03d}.k3.jsonl" in names
    assert {"index-k3.json", "fq-manifest.json"} <= names
    assert "state.json" not in names  # local bookkeeping is not a release
    assert payload["manifest_sha256"] == hashlib.sha256(
        (out / "fq-manifest.json").read_bytes()).hexdigest()
    for name, meta in payload["files"].items():
        assert (out / name).stat().st_size == meta["size"]


def test_verify_round_trip(tree, capsys):
    build(tree)
    verify(tree)
    seen = capsys.readouterr().out
    assert "pinned signer" in seen and "0 MISMATCHED" in seen


def test_one_signature_catches_a_swapped_fragment(tree, capsys):
    out = tree[0]
    build(tree)
    seg = out / f"layer-{LAYERS[0]:03d}.k3.safetensors"
    raw = bytearray(seg.read_bytes())
    raw[-1] ^= 0xFF
    seg.write_bytes(bytes(raw))
    verify(tree, expect=1)
    assert "MISMATCH" in capsys.readouterr().err


def test_one_signature_also_covers_the_attestation_files(tree, capsys):
    """The chain that makes per-expert digests trustworthy without checking
    every attestation signature separately."""
    out = tree[0]
    build(tree)
    att = out / "attestations" / f"layer-{LAYERS[0]:03d}.k3.jsonl"
    line = json.loads(att.read_text())
    payload = json.loads(base64.b64decode(line["payload"]))
    first = sorted(payload["expert_sha256"])[0]
    payload["expert_sha256"][first] = "0" * 64      # lie about an expert digest
    line["payload"] = base64.b64encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).decode()
    att.write_text(json.dumps(line) + "\n")
    verify(tree, expect=1)
    assert "MISMATCH" in capsys.readouterr().err


def test_added_file_is_reported_as_uncovered(tree, capsys):
    out = tree[0]
    build(tree)
    (out / "layer-099.k3.safetensors").write_bytes(b"surprise")
    verify(tree)  # not fatal: it is simply not part of the release
    assert "unlisted" in capsys.readouterr().err


def test_partial_tree_is_fine_unless_completeness_is_demanded(tree, tmp_path, capsys):
    out = tree[0]
    build(tree)
    partial = tmp_path / "partial"
    (partial / "attestations").mkdir(parents=True)
    for name in ("fq-manifest.json", "index-k3.json",
                 f"layer-{LAYERS[0]:03d}.k3.safetensors"):
        (partial / name).write_bytes((out / name).read_bytes())
    verify(tree, target=partial)
    assert "absent" in capsys.readouterr().out
    verify(tree, target=partial, expect=1, complete=True)


def test_wrong_signer_is_refused(tree, tmp_path, capsys):
    out, key, pub, root = tree
    build(tree)
    other = "ab" * 32
    assert fq_release.main([
        "verify", "--dir", str(out), "--trust-signer", other,
        "--trust-root", str(root)]) == 1
    assert "TRUST FAILURE" in capsys.readouterr().err


def test_resigned_release_by_another_key_does_not_pass_pinning(tree, tmp_path, capsys):
    """Repo compromise: attacker rewrites the bytes AND re-signs the release
    with their own key.  Everything is internally consistent; pinning is the
    only thing that says no."""
    out, key, pub, root = tree
    build(tree)
    seg = out / f"layer-{LAYERS[0]:03d}.k3.safetensors"
    seg.write_bytes(seg.read_bytes() + b"evil")
    evil_key = tmp_path / "evil.key"
    assert fq_release.main([
        "build", "--dir", str(out), "--release", "test 0.1.0",
        "--sign-key", str(evil_key)]) == 0
    # self-consistent under the attacker's key
    evil_pub = json.loads(fq_repack.Signer(evil_key).pub_hex)  \
        if False else fq_repack.Signer(evil_key).pub_hex
    assert fq_release.main([
        "verify", "--dir", str(out), "--trust-signer", evil_pub,
        "--trust-root", str(root)]) == 0
    # and refused against the fingerprint published in git
    assert fq_release.main([
        "verify", "--dir", str(out), "--trust-signer", pub,
        "--trust-root", str(root)]) == 1
    assert "trusted signer" in capsys.readouterr().err


def test_json_report(tree, tmp_path):
    build(tree)
    out = tree[0]
    report = tmp_path / "r.json"
    verify(tree, json=report)
    data = json.loads(report.read_text())
    assert data["ok"] and data["rung"] == fq_trust.RUNG_PINNED
    assert not data["bad"] and data["present"]
