"""Tests for fq_verify: metric math on known-identical vs known-perturbed
tensors, streaming local identity (pass, corruption detection, and agreement
with fq_assemble's materialized output), remote fragment identity against a
monkeypatched HTTP source, full re-derivation of expanded families, and the
trust rules — the signer is pinned by the caller, and anything short of a
positive verification fails the run."""
import base64
import hashlib
import json
import sys
import urllib.error
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import fq_assemble  # noqa: E402
import fq_repack  # noqa: E402
import fq_trust  # noqa: E402
import fq_verify  # noqa: E402
from test_fq_prime import build_repo, prime, served  # noqa: E402,F401
import fq_fetch  # noqa: E402
from test_fq_fetch import REV, build_source, write_policy  # noqa: E402
from test_fq_repack import LAYERS, E, write_shard  # noqa: E402


def _supports_trust() -> bool:
    """True when this fq_assemble build takes --trust-signer (fail-closed
    signature checking).  The two repos that carry these tools can be at
    different sync points, so probe instead of assuming."""
    import inspect
    return "--trust-signer" in inspect.getsource(fq_assemble)


def signer_of(family: Path) -> str:
    """The fingerprint the family claims — which a consumer confirms out of
    band and then PINS.  Tests pin it explicitly, exactly as a user must."""
    return json.loads((family / "fq-manifest.json").read_text())["signer_pubkey"]


def pin(family: Path) -> list[str]:
    return ["--trust-signer", signer_of(family)]


def att_lines(family: Path, stem: str) -> Path:
    return family / "attestations" / f"{stem}.jsonl"


def resign(path: Path, key: Path, mutate) -> None:
    """Rewrite an attestation line's payload and re-sign it with `key`, so a
    test can change what the signature actually covers."""
    envelope = json.loads(path.read_text().splitlines()[0])
    payload = json.loads(base64.b64decode(envelope["payload"]))
    mutate(payload)
    path.write_text(fq_repack.Signer(key).sign_line(payload) + "\n")


# ---------------------------------------------------------------- metrics

def test_metrics_identical():
    a = np.random.RandomState(0).randn(64, 48).astype(np.float16)
    m = fq_verify.compute_metrics(a, a.copy())
    assert m["bitwise_equal"] is True
    assert m["cosine"] == pytest.approx(1.0, abs=1e-12)
    assert m["rel_frob"] == 0.0
    assert m["max_abs"] == 0.0


def test_metrics_perturbed():
    rs = np.random.RandomState(1)
    a = rs.randn(128, 96)
    eps = 0.05  # 5% relative Frobenius noise
    noise = rs.randn(*a.shape)
    b = a + eps * np.linalg.norm(a) / np.linalg.norm(noise) * noise
    m = fq_verify.compute_metrics(b, a)
    assert m["bitwise_equal"] is False
    assert m["rel_frob"] == pytest.approx(eps, rel=0.02)
    # small-angle: 1 - cos ~= eps^2 / 2
    assert 1.0 - m["cosine"] == pytest.approx(eps ** 2 / 2, rel=0.15)
    assert m["max_abs"] > 0
    # gross perturbation drives cosine down hard
    m2 = fq_verify.compute_metrics(rs.randn(*a.shape), a)
    assert abs(m2["cosine"]) < 0.1 and m2["rel_frob"] > 1.0


def test_metrics_threshold_discrimination():
    """The similarity report must separate 'same weights, different quant'
    (high cosine) from 'different weights' (low cosine)."""
    rs = np.random.RandomState(2)
    w = rs.randn(256, 64)
    quant_a = w + 0.02 * rs.randn(*w.shape)  # two independent light quants
    quant_b = w + 0.02 * rs.randn(*w.shape)
    m = fq_verify.compute_metrics(quant_a, quant_b)
    assert m["cosine"] > 0.999
    assert m["rel_frob"] < 0.05
    other = rs.randn(*w.shape)
    m2 = fq_verify.compute_metrics(quant_a, other)
    assert m2["cosine"] < 0.1


# --------------------------------------------------------- identity: local

def _k3_workspace(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    for i, layer in enumerate(LAYERS):
        write_shard(snap / f"model-layer-{layer:03d}.safetensors", layer,
                    scramble=bool(i))
    lines = [
        f"{hashlib.sha256((snap / f'model-layer-{L:03d}.safetensors').read_bytes()).hexdigest()}"
        f"  model-layer-{L:03d}.safetensors" for L in LAYERS]
    (snap / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    out = tmp_path / "segments"
    assert fq_repack.main([
        "--snapshot", str(snap), "--source-repo", "test/src", "--revision",
        "r0", "--base-model", "test/base", "--out", str(out),
        "--sign-key", str(tmp_path / "k.key")]) == 0
    return snap, out


def test_identity_local_pass(tmp_path, capsys):
    snap, seg = _k3_workspace(tmp_path)
    rc = fq_verify.main([
        "--identity", "--segments", str(seg), "--source", str(snap),
        "--attest", "all", "--seed", "0", *pin(seg),
        "--json", str(tmp_path / "r.json"), "--md", str(tmp_path / "r.md")])
    assert rc == 0
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["check"] == "local"
    s = report["summary"]
    assert s["shards"] == len(LAYERS) and s["shards_matched"] == len(LAYERS)
    assert s["failures"] == 0
    assert all(r["source_sha_from"] == "MANIFEST.sha256" for r in report["layers"])
    assert all(a["signature"] == "verified" and a["fragment_sha_match"]
               and a["expert_sha_mismatches"] == 0
               for a in report["attestation_sample"])
    assert "PASS" in (tmp_path / "r.md").read_text()


def test_identity_local_stream_matches_fq_assemble(tmp_path):
    """The streamed sha must equal the sha of the shard fq_assemble writes."""
    snap, seg = _k3_workspace(tmp_path)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "schema": "fq-policy/2",
        "bits_per_expert": {str(L): [3] * E for L in LAYERS}}))
    out = tmp_path / "assembled"
    argv = ["--segments", str(seg), "--source", str(snap),
            "--policy", str(policy), "--out", str(out)]
    if "--trust-signer" in (fq_assemble.main.__doc__ or "") or _supports_trust():
        # fq_assemble verifies segments against a pinned signer (fail-closed):
        # pin the family's own key, published in its manifest
        argv += ["--trust-signer",
                 json.loads((seg / "fq-manifest.json").read_text())["signer_pubkey"]]
    assert fq_assemble.main(argv) == 0
    idx = json.loads((seg / "index-k3.json").read_text())
    for layer in LAYERS:
        reader = fq_assemble.SegmentReader(
            seg / idx[str(layer)]["file"])
        got = fq_verify.stream_assemble_sha(
            snap / f"model-layer-{layer:03d}.safetensors",
            {3: reader}, lambda e: 3)
        reader.close()
        want = hashlib.sha256(
            (out / f"model-layer-{layer:03d}.safetensors").read_bytes()).hexdigest()
        assert got["sha256"] == want
        assert got["expert_tensors"] > 0 and got["bytes_from_segments"] > 0


def test_identity_local_detects_corruption(tmp_path):
    snap, seg = _k3_workspace(tmp_path)
    idx = json.loads((seg / "index-k3.json").read_text())[str(LAYERS[0])]
    p = seg / idx["file"]
    raw = bytearray(p.read_bytes())
    raw[idx["body_offset"] + idx["experts"]["1"][0] + 3] ^= 0x55
    p.write_bytes(bytes(raw))
    rc = fq_verify.main([
        "--identity", "--segments", str(seg), "--source", str(snap),
        "--attest", "all", "--seed", "0", *pin(seg),
        "--json", str(tmp_path / "r.json")])
    assert rc == 1
    report = json.loads((tmp_path / "r.json").read_text())
    bad = [r for r in report["layers"] if not r["match"]]
    assert [r["layer"] for r in bad] == [LAYERS[0]]
    assert report["summary"]["failures"] >= 1


# -------------------------------------------------------- identity: remote

def test_identity_remote_pass(served, tmp_path):
    srv_tmp, calls = served
    build_repo(srv_tmp / "repo", "shared_h_v1")
    assert prime(srv_tmp, "segments-sh") == 0
    fam = srv_tmp / "primed" / "segments-sh" / "shared-h"
    rc = fq_verify.main([
        "--identity", "--segments", str(fam), "--sample", "all",
        "--pace", "0", "--seed", "1", *pin(fam),
        "--json", str(tmp_path / "r.json"), "--md", str(tmp_path / "r.md")])
    assert rc == 0
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["check"] == "remote"
    s = report["summary"]
    assert s["experts_refetched"] == s["experts_byte_equal"] > 0
    assert s["profiles_refetched"] == s["profiles_byte_equal"] == len(LAYERS)
    assert s["failures"] == 0
    assert all(r["signature"] == "verified" for r in report["local_integrity"])
    # fresh header + payload were actually re-fetched
    assert report["transport"]["requests"] > 0
    assert "byte-identical to freshly range-read" in (tmp_path / "r.md").read_text()


def test_identity_remote_detects_corruption(served, tmp_path):
    srv_tmp, calls = served
    build_repo(srv_tmp / "repo", "shared_h_v1")
    assert prime(srv_tmp, "segments-sh") == 0
    fam = srv_tmp / "primed" / "segments-sh" / "shared-h"
    idx = json.loads((fam / "index-k3.json").read_text())["3"]
    p = fam / idx["file"]
    raw = bytearray(p.read_bytes())
    raw[idx["body_offset"] + idx["experts"]["0"][0]] ^= 0xFF
    p.write_bytes(bytes(raw))
    rc = fq_verify.main([
        "--identity", "--segments", str(fam), "--sample", "all",
        "--pace", "0", "--seed", "1", *pin(fam),
        "--json", str(tmp_path / "r.json")])
    assert rc == 1
    report = json.loads((tmp_path / "r.json").read_text())
    # both the local integrity pass and the remote byte comparison catch it
    bad_local = [r for r in report["local_integrity"]
                 if r["expert_sha_mismatches"] or not r["index_sha_match"]]
    bad_remote = [r for r in report["remote_fragments"] if not r["byte_equal"]]
    assert bad_local and bad_remote


# ------------------------------------------------------- identity: derived

def test_identity_derived_pass(served, tmp_path):
    srv_tmp, calls = served
    build_repo(srv_tmp / "repo", "shared_h_v1")
    assert prime(srv_tmp, "segments-sh", ["--expand"]) == 0
    fam = srv_tmp / "primed" / "segments-sh" / "expanded"
    rc = fq_verify.main([
        "--identity", "--segments", str(fam), *pin(fam),
        "--parent", str(srv_tmp / "primed" / "segments-sh" / "shared-h"),
        "--json", str(tmp_path / "r.json"), "--md", str(tmp_path / "r.md")])
    assert rc == 0
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["check"] == "derived"
    s = report["summary"]
    assert s["failures"] == 0 and s["experts_checked"] > 0
    assert s["replicated_bytes"] > 0 and s["verbatim_bytes"] > 0
    for r in report["layers"]:
        assert r["tensor_mismatches"] == 0
        assert r["parent_segment_sha_match"] and r["parent_profile_sha_match"]
        # per_expert_v1 unit: 12 tensor slots, 6 of them H-side replicas
        assert r["replicated_tensors"] == r["experts"] * 6
    assert "re-derived and byte-compared in full" in (tmp_path / "r.md").read_text()


def test_identity_derived_detects_bad_replica(served, tmp_path):
    srv_tmp, calls = served
    build_repo(srv_tmp / "repo", "shared_h_v1")
    assert prime(srv_tmp, "segments-sh", ["--expand"]) == 0
    fam = srv_tmp / "primed" / "segments-sh" / "expanded"
    # corrupt one replicated shared row inside the expanded segment
    idx = json.loads((fam / "index-k3.json").read_text())["3"]
    seg = fam / idx["file"]
    hdr, body = fq_repack.read_header(seg)
    hdr.pop("__metadata__")
    name = next(n for n in hdr if n.endswith(".gate_proj.rank0.suh"))
    raw = bytearray(seg.read_bytes())
    raw[body + hdr[name]["data_offsets"][0]] ^= 0x01
    seg.write_bytes(bytes(raw))
    rc = fq_verify.main([
        "--identity", "--segments", str(fam), *pin(fam),
        "--parent", str(srv_tmp / "primed" / "segments-sh" / "shared-h"),
        "--json", str(tmp_path / "r.json")])
    assert rc == 1
    report = json.loads((tmp_path / "r.json").read_text())
    assert any(r["tensor_mismatches"] for r in report["layers"])


def test_identity_derived_detects_parent_pin_break(served, tmp_path):
    srv_tmp, calls = served
    build_repo(srv_tmp / "repo", "shared_h_v1")
    assert prime(srv_tmp, "segments-sh", ["--expand"]) == 0
    sh = srv_tmp / "primed" / "segments-sh" / "shared-h"
    fam = srv_tmp / "primed" / "segments-sh" / "expanded"
    # flip a byte in the parent profile: pins must break (and the replicas
    # no longer match the parent bytes)
    prof = next(iter(sh.glob("layer-*.shared.safetensors")))
    raw = bytearray(prof.read_bytes())
    raw[-1] ^= 0x80
    prof.write_bytes(bytes(raw))
    rc = fq_verify.main([
        "--identity", "--segments", str(fam), "--parent", str(sh), *pin(fam),
        "--json", str(tmp_path / "r.json")])
    assert rc == 1
    report = json.loads((tmp_path / "r.json").read_text())
    assert any(not r["parent_profile_sha_match"] for r in report["layers"])


# ------------------------------------------------------ trust rules (P1-4a)

def _local_run(tmp_path, seg, snap, *extra, json_name="r.json"):
    return fq_verify.main([
        "--identity", "--segments", str(seg), "--source", str(snap),
        "--attest", "all", "--seed", "0", *extra,
        "--json", str(tmp_path / json_name)])


def test_verify_refuses_without_a_pin_or_trust_root(tmp_path, capsys):
    """The signer must come from the caller, never from the artifact.  With
    no pin and a trust root that does not list the family's key, the run is
    a TRUST FAILURE — not a pass with 'unverified' next to it."""
    snap, seg = _k3_workspace(tmp_path)
    empty_root = tmp_path / "root"
    empty_root.mkdir()
    (empty_root / "FINGERPRINTS").write_text("# nobody is trusted here\n")
    rc = _local_run(tmp_path, seg, snap, "--trust-root", str(empty_root))
    assert rc == 1
    err = capsys.readouterr().err
    assert "TRUST FAILURE" in err
    assert not (tmp_path / "r.json").exists()


def test_verify_refuses_a_signer_the_trust_root_does_not_list(tmp_path, capsys):
    """A rewritten repo names its own key in fq-manifest.json.  Against a
    real trust root that key is simply not listed, and that is fatal."""
    snap, seg = _k3_workspace(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    (root / "FINGERPRINTS").write_text(
        f"{'ab' * 32}  someone-else  active  2026-08-10  segments\n")
    assert _local_run(tmp_path, seg, snap, "--trust-root", str(root)) == 1
    assert "TRUST FAILURE" in capsys.readouterr().err


def test_verify_rejects_a_signer_swap(tmp_path, capsys):
    """The finding itself: an attacker re-signs every attestation with their
    own key and updates the manifest to match.  Self-consistent — and
    refused, because the pin came from the user."""
    snap, seg = _k3_workspace(tmp_path)
    good = signer_of(seg)
    evil = fq_repack.Signer(tmp_path / "evil.key")
    for att in sorted((seg / "attestations").glob("*.jsonl")):
        payload = json.loads(base64.b64decode(
            json.loads(att.read_text().splitlines()[0])["payload"]))
        att.write_text(evil.sign_line(payload) + "\n")
    manifest = json.loads((seg / "fq-manifest.json").read_text())
    manifest["signer_pubkey"] = evil.pub_hex
    (seg / "fq-manifest.json").write_text(json.dumps(manifest, indent=1))
    assert signer_of(seg) != good
    # pinned to the key the user actually trusts -> refused
    assert _local_run(tmp_path, seg, snap, "--trust-signer", good) == 1
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["summary"]["failures"] == len(report["attestation_sample"])
    assert all("the trusted signer is" in a["signature"]
               for a in report["attestation_sample"])
    assert report["trust"]["signer"] == good
    capsys.readouterr()
    # ... and taking the signer from the artifact (the old behaviour) is not
    # something the CLI can even be asked to do without saying so out loud
    assert _local_run(tmp_path, seg, snap, "--allow-unpinned-signer",
                      json_name="unpinned.json") == 0
    report = json.loads((tmp_path / "unpinned.json").read_text())
    assert report["trust"]["rung"] == fq_trust.RUNG_UNPINNED


def test_verify_rejects_a_tampered_signature(tmp_path):
    snap, seg = _k3_workspace(tmp_path)
    att = att_lines(seg, f"layer-{LAYERS[0]:03d}.k3")
    envelope = json.loads(att.read_text().splitlines()[0])
    sig = bytearray(base64.b64decode(envelope["signature"]))
    sig[0] ^= 0xFF
    envelope["signature"] = base64.b64encode(bytes(sig)).decode()
    att.write_text(json.dumps(envelope) + "\n")
    assert _local_run(tmp_path, seg, snap, *pin(seg)) == 1
    report = json.loads((tmp_path / "r.json").read_text())
    bad = [a for a in report["attestation_sample"]
           if a["signature"].startswith("BAD")]
    assert bad and "BAD SIGNATURE" in bad[0]["signature"]


def test_verify_rejects_a_placeholder_signature(tmp_path):
    """'AA==' decodes fine and proves nothing."""
    snap, seg = _k3_workspace(tmp_path)
    att = att_lines(seg, f"layer-{LAYERS[0]:03d}.k3")
    envelope = json.loads(att.read_text().splitlines()[0])
    envelope["signature"] = base64.b64encode(b"\x00" * 64).decode()
    att.write_text(json.dumps(envelope) + "\n")
    assert _local_run(tmp_path, seg, snap, *pin(seg)) == 1
    report = json.loads((tmp_path / "r.json").read_text())
    assert any(a["signature"].startswith("BAD")
               for a in report["attestation_sample"])


def test_verify_rejects_a_missing_attestation(tmp_path):
    """Absent evidence is a failed check, not a skipped one."""
    snap, seg = _k3_workspace(tmp_path)
    att_lines(seg, f"layer-{LAYERS[0]:03d}.k3").unlink()
    assert _local_run(tmp_path, seg, snap, *pin(seg)) == 1
    report = json.loads((tmp_path / "r.json").read_text())
    assert any(a["signature"] == fq_verify.ATT_MISSING
               for a in report["attestation_sample"])
    assert report["summary"]["failures"] >= 1


def test_verify_rejects_an_empty_attestation_file(tmp_path):
    snap, seg = _k3_workspace(tmp_path)
    att_lines(seg, f"layer-{LAYERS[0]:03d}.k3").write_text("")
    assert _local_run(tmp_path, seg, snap, *pin(seg)) == 1


def test_verify_rejects_a_missing_expert_digest(tmp_path):
    """A validly signed line that simply omits an expert proves nothing
    about that expert's bytes."""
    snap, seg = _k3_workspace(tmp_path)
    att = att_lines(seg, f"layer-{LAYERS[0]:03d}.k3")
    resign(att, tmp_path / "k.key",
           lambda p: p["expert_sha256"].pop(sorted(p["expert_sha256"])[0]))
    assert _local_run(tmp_path, seg, snap, *pin(seg)) == 1
    report = json.loads((tmp_path / "r.json").read_text())
    assert any(a["expert_sha_mismatches"] >= 1
               for a in report["attestation_sample"]
               if "expert_sha_mismatches" in a)


def test_verify_iterates_every_jsonl_line(tmp_path):
    """The file is JSON Lines: a leading line signed by someone else (say a
    countersignature) must not hide the line that does verify."""
    snap, seg = _k3_workspace(tmp_path)
    stranger = fq_repack.Signer(tmp_path / "stranger.key")
    for att in sorted((seg / "attestations").glob("*.jsonl")):
        good = att.read_text().splitlines()[0]
        payload = json.loads(base64.b64decode(json.loads(good)["payload"]))
        att.write_text(stranger.sign_line(payload) + "\n" + good + "\n")
    assert _local_run(tmp_path, seg, snap, *pin(seg)) == 0
    report = json.loads((tmp_path / "r.json").read_text())
    assert all(a["signature"] == "verified"
               for a in report["attestation_sample"])
    assert report["summary"]["failures"] == 0


def test_verify_insecure_skip_can_never_exit_zero(tmp_path, capsys):
    """The escape hatch prints the byte report and still fails: a run that
    checked no signature has not verified anything."""
    snap, seg = _k3_workspace(tmp_path)
    rc = _local_run(tmp_path, seg, snap, "--insecure-skip-signatures")
    assert rc == 1
    err = capsys.readouterr().err
    assert "proves nothing" in err
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["trust"]["rung"] == fq_trust.RUNG_NONE
    assert all(a["signature"] == fq_verify.ATT_NOT_CHECKED
               for a in report["attestation_sample"])
    # the byte-level work still happened and still matched
    assert report["summary"]["shards_matched"] == len(LAYERS)


def test_verify_remote_requires_a_pin(served, tmp_path, capsys):
    srv_tmp, calls = served
    build_repo(srv_tmp / "repo", "shared_h_v1")
    assert prime(srv_tmp, "segments-sh") == 0
    fam = srv_tmp / "primed" / "segments-sh" / "shared-h"
    root = tmp_path / "root"
    root.mkdir()
    (root / "FINGERPRINTS").write_text("# empty\n")
    rc = fq_verify.main([
        "--identity", "--segments", str(fam), "--sample", "1",
        "--pace", "0", "--seed", "1", "--trust-root", str(root)])
    assert rc == 1
    assert "TRUST FAILURE" in capsys.readouterr().err


def test_verify_remote_rejects_a_missing_attestation(served, tmp_path):
    srv_tmp, calls = served
    build_repo(srv_tmp / "repo", "shared_h_v1")
    assert prime(srv_tmp, "segments-sh") == 0
    fam = srv_tmp / "primed" / "segments-sh" / "shared-h"
    next(iter((fam / "attestations").glob("layer-*.k*.jsonl"))).unlink()
    rc = fq_verify.main([
        "--identity", "--segments", str(fam), "--sample", "all",
        "--pace", "0", "--seed", "1", *pin(fam),
        "--json", str(tmp_path / "r.json")])
    assert rc == 1
    report = json.loads((tmp_path / "r.json").read_text())
    assert any(r["signature"] == fq_verify.ATT_MISSING
               for r in report["local_integrity"])


def test_verify_derived_rejects_a_missing_attestation(served, tmp_path):
    srv_tmp, calls = served
    build_repo(srv_tmp / "repo", "shared_h_v1")
    assert prime(srv_tmp, "segments-sh", ["--expand"]) == 0
    sh = srv_tmp / "primed" / "segments-sh" / "shared-h"
    fam = srv_tmp / "primed" / "segments-sh" / "expanded"
    for att in (fam / "attestations").glob("*.jsonl"):
        att.unlink()
    rc = fq_verify.main([
        "--identity", "--segments", str(fam), "--parent", str(sh), *pin(fam),
        "--json", str(tmp_path / "r.json")])
    assert rc == 1
    report = json.loads((tmp_path / "r.json").read_text())
    assert all(r["signature"] == fq_verify.ATT_MISSING
               for r in report["layers"])


# ------------------------------------------------------------ auto-detect

def test_check_autodetect(served, tmp_path):
    srv_tmp, calls = served
    build_repo(srv_tmp / "repo", "shared_h_v1")
    assert prime(srv_tmp, "segments-sh", ["--expand"]) == 0
    base = srv_tmp / "primed" / "segments-sh"
    assert fq_verify.detect_check(base / "shared-h", None) == "remote"
    assert fq_verify.detect_check(base / "expanded", None) == "derived"
    assert fq_verify.detect_check(base / "shared-h", Path("/x")) == "local"



def test_identity_derived_binds_signed_fragment_and_metadata(served, tmp_path):
    srv_tmp, _ = served
    build_repo(srv_tmp / "repo", "shared_h_v1")
    assert prime(srv_tmp, "segments-sh", ["--expand"]) == 0
    fam = srv_tmp / "primed" / "segments-sh" / "expanded"
    idx = json.loads((fam / "index-k3.json").read_text())
    idx["3"]["sha256"] = "0" * 64
    (fam / "index-k3.json").write_text(json.dumps(idx))
    assert fq_verify.main([
        "--identity", "--segments", str(fam), *pin(fam),
        "--parent", str(srv_tmp / "primed" / "segments-sh" / "shared-h"),
        "--json", str(tmp_path / "derived.json")]) == 1
    row = json.loads((tmp_path / "derived.json").read_text())["layers"][0]
    assert not row["index_sha_match"]


def test_identity_remote_requires_shared_profile_attestation(served, tmp_path):
    srv_tmp, _ = served
    build_repo(srv_tmp / "repo", "shared_h_v1")
    assert prime(srv_tmp, "segments-sh") == 0
    fam = srv_tmp / "primed" / "segments-sh" / "shared-h"
    att_lines(fam, f"layer-{LAYERS[0]:03d}.shared").unlink()
    assert fq_verify.main([
        "--identity", "--segments", str(fam), "--sample", "all", "--pace", "0",
        *pin(fam), "--json", str(tmp_path / "remote.json")]) == 1
    profiles = json.loads((tmp_path / "remote.json").read_text())["shared_profiles"]
    assert any(row["signature"] == fq_verify.ATT_MISSING for row in profiles)


def test_identity_derived_rejects_parent_metadata_repoint(served, tmp_path):
    srv_tmp, _ = served
    build_repo(srv_tmp / "repo", "shared_h_v1")
    assert prime(srv_tmp, "segments-sh", ["--expand"]) == 0
    fam = srv_tmp / "primed" / "segments-sh" / "expanded"
    idx = json.loads((fam / "index-k3.json").read_text())
    seg = fam / idx["3"]["file"]
    raw = seg.read_bytes()
    old = b"layer-003.k3.safetensors"
    new = b"layer-004.k3.safetensors"
    assert old in raw and len(old) == len(new)
    seg.write_bytes(raw.replace(old, new, 1))
    assert fq_verify.main([
        "--identity", "--segments", str(fam), *pin(fam),
        "--parent", str(srv_tmp / "primed" / "segments-sh" / "shared-h"),
        "--json", str(tmp_path / "repoint.json")]) == 1
    assert not json.loads((tmp_path / "repoint.json").read_text())[
        "layers"][0]["metadata_matches_attestation"]


def _fetched_workspace(tmp_path, monkeypatch, *, unsafe=False):
    repo, _, publisher = build_source(tmp_path, "pub", ks=(3,))

    def local_path(url):
        marker = f"/test/pub/resolve/{REV}/"
        assert marker in url
        return repo / url.split(marker, 1)[1]

    def get_range(url, start, end, timeout=600.0):
        data = local_path(url).read_bytes()[start:end]
        assert len(data) == end - start
        return data, {"commit": REV, "total": local_path(url).stat().st_size}

    def get_full(url, timeout=600.0):
        path = local_path(url)
        if not path.exists():
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)
        return path.read_bytes(), {"commit": REV}
    monkeypatch.setattr(fq_fetch, "http_get_range", get_range)
    monkeypatch.setattr(fq_fetch, "http_get_full", get_full)
    out = tmp_path / "fetched"
    policy = write_policy(tmp_path / "recipe.json", {layer: [3] * E for layer in LAYERS})
    argv = ["--policy", str(policy), "--out", str(out), "--source",
            f"test/pub@{REV}", "--trust-signer", publisher,
            "--sign-key", str(tmp_path / "local.key")]
    if unsafe:
        argv += ["--header-trust", "unsafe"]
    assert fq_fetch.main(argv) == 0
    return out, publisher


def test_identity_fetched_chain_and_unsafe_policy(tmp_path, monkeypatch):
    fam, publisher = _fetched_workspace(tmp_path, monkeypatch)
    local = signer_of(fam)
    argv = ["--identity", "--check", "fetched", "--segments", str(fam),
            "--trust-signer", local, "--upstream-trust-signer", publisher,
            "--json", str(tmp_path / "fetched.json")]
    assert fq_verify.main(argv) == 0
    assert fq_verify.detect_check(fam, None) == "fetched"
    report = json.loads((tmp_path / "fetched.json").read_text())
    assert report["summary"]["segments_verified"] > 0

    copied = next((fam / "attestations").glob("test__pub@*/layer-*.k*.jsonl"))
    saved = copied.read_text()
    copied.unlink()
    assert fq_verify.main(argv) == 1
    copied.write_text(saved)
    wrong_pin = "de" * 32
    assert fq_verify.main([
        "--identity", "--check", "fetched", "--segments", str(fam),
        "--trust-signer", local, "--upstream-trust-signer", wrong_pin]) == 1
    unsafe_fam, unsafe_publisher = _fetched_workspace(
        tmp_path / "unsafe", monkeypatch, unsafe=True)
    unsafe_argv = ["--identity", "--check", "fetched", "--segments", str(unsafe_fam),
                  "--trust-signer", signer_of(unsafe_fam),
                  "--upstream-trust-signer", unsafe_publisher]
    assert fq_verify.main(unsafe_argv) == 1
    assert fq_verify.main(unsafe_argv + ["--accept-unsafe-header-plan"]) == 0


def test_identity_fetched_binds_parent_coverage_header_and_jsonl(tmp_path, monkeypatch):
    fam, publisher = _fetched_workspace(tmp_path, monkeypatch)
    local = signer_of(fam)
    argv = ["--identity", "--check", "fetched", "--segments", str(fam),
            "--trust-signer", local, "--upstream-trust-signer", publisher]
    local_att = next((fam / "attestations").glob("layer-*.k*.jsonl"))
    saved_local = local_att.read_text()
    resign(local_att, tmp_path / "local.key",
           lambda p: p["parents"][0].update({"experts": []}))
    assert fq_verify.main(argv) == 1
    local_att.write_text(saved_local)
    resign(local_att, tmp_path / "local.key",
           lambda p: p["parents"][0].update({"header_sha256": "0" * 64}))
    assert fq_verify.main(argv) == 1
    local_att.write_text(saved_local)

    copied = next((fam / "attestations").glob("test__pub@*/layer-*.k*.jsonl"))
    upstream = json.loads(base64.b64decode(
        json.loads(copied.read_text().splitlines()[0])["payload"]))
    digests = list(upstream["expert_sha256"].items())
    upstream["expert_sha256"] = dict(digests[:len(digests) // 2])
    other = dict(upstream)
    other["expert_sha256"] = dict(digests[len(digests) // 2:])
    publisher_key = fq_repack.Signer(tmp_path / "pub.key")
    copied.write_text(publisher_key.sign_line(upstream) + "\n" +
                      publisher_key.sign_line(other) + "\n")
    assert fq_verify.main(argv) == 0