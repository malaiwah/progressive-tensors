"""Tests for fq_verify: metric math on known-identical vs known-perturbed
tensors, streaming local identity (pass, corruption detection, and agreement
with fq_assemble's materialized output), remote fragment identity against a
monkeypatched HTTP source, and full re-derivation of expanded families."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import fq_assemble  # noqa: E402
import fq_repack  # noqa: E402
import fq_verify  # noqa: E402
from test_fq_prime import build_repo, prime, served  # noqa: E402,F401
from test_fq_repack import LAYERS, E, write_shard  # noqa: E402


def _supports_trust() -> bool:
    """True when this fq_assemble build takes --trust-signer (fail-closed
    signature checking).  The two repos that carry these tools can be at
    different sync points, so probe instead of assuming."""
    import inspect
    return "--trust-signer" in inspect.getsource(fq_assemble)


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
        "--attest", "all", "--seed", "0",
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
        "--attest", "all", "--seed", "0", "--json", str(tmp_path / "r.json")])
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
        "--pace", "0", "--seed", "1", "--json", str(tmp_path / "r.json"),
        "--md", str(tmp_path / "r.md")])
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
        "--pace", "0", "--seed", "1", "--json", str(tmp_path / "r.json")])
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
        "--identity", "--segments", str(fam),
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
        "--identity", "--segments", str(fam),
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
        "--identity", "--segments", str(fam), "--parent", str(sh),
        "--json", str(tmp_path / "r.json")])
    assert rc == 1
    report = json.loads((tmp_path / "r.json").read_text())
    assert any(not r["parent_profile_sha_match"] for r in report["layers"])


# ------------------------------------------------------------ auto-detect

def test_check_autodetect(served, tmp_path):
    srv_tmp, calls = served
    build_repo(srv_tmp / "repo", "shared_h_v1")
    assert prime(srv_tmp, "segments-sh", ["--expand"]) == 0
    base = srv_tmp / "primed" / "segments-sh"
    assert fq_verify.detect_check(base / "shared-h", None) == "remote"
    assert fq_verify.detect_check(base / "expanded", None) == "derived"
    assert fq_verify.detect_check(base / "shared-h", Path("/x")) == "local"
