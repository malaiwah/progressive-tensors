"""Tests for fq_assemble: byte-identity round trip through repack + assemble,
mandatory segment verification, and output integrity metadata."""
import base64
import errno
import hashlib
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import fq_repack  # noqa: E402
import fq_assemble  # noqa: E402
import fq_trust  # noqa: E402
from test_fq_repack import LAYERS, E, write_shard, tensor_bytes  # noqa: E402


def _build_k3_workspace(root: Path) -> tuple[Path, Path]:
    """Synthetic K3 snapshot + repacked segments under root: (snap, segments)."""
    snap = root / "snap"
    snap.mkdir()
    for i, layer in enumerate(LAYERS):
        write_shard(snap / f"model-layer-{layer:03d}.safetensors", layer, scramble=bool(i))
    (snap / "config.json").write_text("{}")
    lines = [
        f"{hashlib.sha256((snap / f'model-layer-{l:03d}.safetensors').read_bytes()).hexdigest()}"
        f"  model-layer-{l:03d}.safetensors"
        for l in LAYERS
    ]
    (snap / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    out = root / "segments"
    fq_repack.main([
        "--snapshot", str(snap),
        "--source-repo", "test/src", "--revision", "r0",
        "--base-model", "test/base", "--out", str(out),
        "--sign-key", str(root / "k.key"),
    ])
    return snap, out


def signer_of(segments: Path) -> str:
    return json.loads((segments / "fq-manifest.json").read_text())["signer_pubkey"]


def write_trust_root(path: Path, *records: tuple[str, str, str]) -> Path:
    path.write_text("".join(
        f"{fingerprint}  {key_id}  {status}  2026-08-10  test\n"
        for fingerprint, key_id, status in records))
    return path


def assemble(segments, snap, policy, out, *extra):
    """fq_assemble.main with the family's own signer pinned, unless the test
    supplies its own trust flags."""
    argv = ["--segments", str(segments), "--source", str(snap),
            "--policy", str(policy), "--out", str(out), *extra]
    if not {"--trust-signer", "--trust-file", "--insecure"} & set(extra):
        argv += ["--trust-signer", signer_of(Path(segments))]
    return fq_assemble.main(argv)


def policy_file(root: Path, bits: dict, name="policy.json") -> Path:
    p = root / name
    p.write_text(json.dumps({"schema": "fq-policy/2", "bits_per_expert": bits}))
    return p


@pytest.fixture()
def repacked(tmp_path):
    snap, out = _build_k3_workspace(tmp_path)
    return snap, out, tmp_path


def test_all_k3_assembly_is_byte_identical(repacked):
    snap, segments, tmp = repacked
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    out = tmp / "assembled"
    assemble(segments, snap, ppath, out)
    for layer in LAYERS:
        a = (snap / f"model-layer-{layer:03d}.safetensors").read_bytes()
        b = (out / f"model-layer-{layer:03d}.safetensors").read_bytes()
        assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest(), layer
    assert (out / "config.json").exists(), "non-shard files copied through"


@pytest.fixture()
def repacked_multi_k(tmp_path):
    """K3 source snapshot + K3 AND K4 segments (larger K4 trellis tensors)."""
    snaps = {}
    for k in (3, 4):
        snap = tmp_path / f"snap-k{k}"
        snap.mkdir()
        for i, layer in enumerate(LAYERS):
            write_shard(snap / f"model-layer-{layer:03d}.safetensors", layer,
                        scramble=bool(i), k=k)
        snaps[k] = snap
    snap3 = snaps[3]
    (snap3 / "config.json").write_text(json.dumps({
        "hybrid_tr3_tail": {
            "format": "exl3-trellis", "bits": 3.0, "codebook": "mcg",
            "experts_per_layer": E, "moe_layers": [min(LAYERS), max(LAYERS)],
            "tensor_schema": "model.layers.{L}.mlp.experts.{E}.{proj}"
                             ".rank{r}.{trellis|suh|svh|mcg}",
            "tp": 4,
        },
        # the source ships a uniform quantization_config: a mixed assembly
        # that leaves it at 3.0 is exactly finding 4's contradiction
        "quantization_config": {"quant_method": "exl3", "bits": 3.0,
                                "codebook": "mcg", "version": "rank-sliced"},
    }))
    (snap3 / "tier_bitmap.json").write_text(json.dumps(
        {str(layer): {"keep_nvfp4": []} for layer in LAYERS}))
    (snap3 / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": 0}, "weight_map": {}}))
    (snap3 / "MANIFEST.sha256").write_text("\n".join(
        f"{hashlib.sha256((snap3 / f'model-layer-{l:03d}.safetensors').read_bytes()).hexdigest()}"
        f"  model-layer-{l:03d}.safetensors"
        for l in LAYERS) + "\n")
    segments = tmp_path / "segments"
    for k in (3, 4):
        fq_repack.main([
            "--snapshot", str(snaps[k]),
            "--source-repo", f"test/src-k{k}", "--revision", "r0",
            "--base-model", "test/base", "--out", str(segments),
            "--k", str(k),
            "--sign-key", str(tmp_path / "k.key"),
        ])
    return snap3, segments, tmp_path


def test_mixed_policy_reindex(repacked_multi_k):
    snap, segments, tmp = repacked_multi_k
    mixed_layer, uniform_layer = LAYERS[0], LAYERS[1]
    mixed_bits = [3, 4, 4, 3]
    assert len(mixed_bits) == E
    ppath = policy_file(tmp, {str(mixed_layer): mixed_bits,
                              str(uniform_layer): [3] * E}, "policy-mixed.json")
    out = tmp / "assembled-mixed"
    assert assemble(segments, snap, ppath, out) == 0

    # (c) the uniform-K3 layer still goes through the byte-identity path
    a = (snap / f"model-layer-{uniform_layer:03d}.safetensors").read_bytes()
    b = (out / f"model-layer-{uniform_layer:03d}.safetensors").read_bytes()
    assert a == b

    src_shard = snap / f"model-layer-{mixed_layer:03d}.safetensors"
    out_shard = out / f"model-layer-{mixed_layer:03d}.safetensors"
    src_hdr, src_body = fq_repack.read_header(src_shard)
    out_hdr, out_body = fq_repack.read_header(out_shard)
    src_meta = src_hdr.pop("__metadata__", None)
    assert out_hdr.pop("__metadata__", None) == src_meta
    src_raw = src_shard.read_bytes()
    out_raw = out_shard.read_bytes()

    # tensor ORDER preserved; offsets contiguous from 0
    src_order = sorted(src_hdr, key=lambda n: src_hdr[n]["data_offsets"][0])
    out_order = sorted(out_hdr, key=lambda n: out_hdr[n]["data_offsets"][0])
    assert src_order == out_order
    expect_off = 0
    for name in out_order:
        a0, b0 = out_hdr[name]["data_offsets"]
        assert a0 == expect_off
        expect_off = b0
    assert out_body + expect_off == len(out_raw)

    seg_hdrs = {}
    for k in (3, 4):
        seg = segments / f"layer-{mixed_layer:03d}.k{k}.safetensors"
        h, body = fq_repack.read_header(seg)
        h.pop("__metadata__", None)
        seg_hdrs[k] = (h, body, seg.read_bytes())

    for name, t in out_hdr.items():
        a0, b0 = t["data_offsets"]
        data = out_raw[out_body + a0: out_body + b0]
        m = fq_repack.EXPERT_RE.match(name)
        if m is None:
            # (a) non-expert tensors byte-identical to the source shard
            sa, sb = src_hdr[name]["data_offsets"]
            assert data == src_raw[src_body + sa: src_body + sb], name
            assert t["dtype"] == src_hdr[name]["dtype"]
            assert t["shape"] == src_hdr[name]["shape"]
        else:
            # (b) expert tensors byte-identical to the policy-K segment,
            # with the segment's dtype/shape in the rebuilt header
            _, e, proj, rank, comp = m.groups()
            k = mixed_bits[int(e)]
            sh, sbody, sraw = seg_hdrs[k]
            ga, gb = sh[name]["data_offsets"]
            assert data == sraw[sbody + ga: sbody + gb], name
            assert data == tensor_bytes(mixed_layer, int(e), proj,
                                        int(rank), comp, k), name
            assert t["dtype"] == sh[name]["dtype"]
            assert t["shape"] == sh[name]["shape"]


def test_mixed_metadata_and_index(repacked_multi_k):
    snap, segments, tmp = repacked_multi_k
    bits = {str(LAYERS[0]): [3, 4, 4, 3], str(LAYERS[1]): [4, 3, 3, 4]}
    ppath = policy_file(tmp, bits, "policy-meta.json")
    out = tmp / "assembled-meta"
    assert assemble(segments, snap, ppath, out) == 0

    cfg = json.loads((out / "config.json").read_text())
    tail = cfg["hybrid_tr3_tail"]
    assert tail["bits"] == "mixed"
    assert tail["k_values"] == [3, 4]
    assert cfg["quantization_config"]["quant_method"] == "exl3"
    assert cfg["quantization_config"]["bits"] == "mixed"
    fname, field = tail["bits_per_expert"].rsplit(":", 1)
    bitmap = json.loads((out / fname).read_text())
    for layer in LAYERS:
        assert bitmap[str(layer)][field] == bits[str(layer)]
        assert bitmap[str(layer)]["keep_nvfp4"] == []  # base fields kept

    index = json.loads((out / "model.safetensors.index.json").read_text())
    total = 0
    for shard in sorted(out.glob("model-*.safetensors")):
        hdr, _ = fq_repack.read_header(shard)
        hdr.pop("__metadata__", None)
        for name, t in hdr.items():
            assert index["weight_map"][name] == shard.name
            total += t["data_offsets"][1] - t["data_offsets"][0]
    assert index["metadata"]["total_size"] == total

    manifest = dict(
        line.split()[::-1] for line in
        (out / "MANIFEST.sha256").read_text().splitlines())
    for layer in LAYERS:
        name = f"model-layer-{layer:03d}.safetensors"
        assert manifest[name] == hashlib.sha256(
            (out / name).read_bytes()).hexdigest()


def test_k3_uniform_round_trip_still_byte_identical(repacked_multi_k):
    """K3-uniform policy from a multi-K segment dir: still byte-identical."""
    snap, segments, tmp = repacked_multi_k
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS}, "policy-k3.json")
    out = tmp / "assembled-k3"
    assert assemble(segments, snap, ppath, out) == 0
    for layer in LAYERS:
        a = (snap / f"model-layer-{layer:03d}.safetensors").read_bytes()
        b = (out / f"model-layer-{layer:03d}.safetensors").read_bytes()
        assert a == b, layer
    # uniform policy leaves the source config/tier metadata untouched
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["hybrid_tr3_tail"]["bits"] == 3.0
    assert "k_values" not in cfg["hybrid_tr3_tail"]


def test_missing_segment_fails_loudly(repacked):
    snap, segments, tmp = repacked
    ppath = policy_file(tmp, {str(LAYERS[0]): [4] * E}, "p4.json")
    with pytest.raises(FileNotFoundError, match="k4"):
        assemble(segments, snap, ppath, tmp / "x")


# ------------------------------------ FINDING 1: verification is mandatory

def att_path(segments: Path, layer: int, k: int) -> Path:
    return segments / "attestations" / f"layer-{layer:03d}.k{k}.jsonl"


def read_payload(path: Path) -> dict:
    return json.loads(base64.b64decode(json.loads(path.read_text())["payload"]))


def resign(path: Path, key: Path, mutate) -> dict:
    payload = read_payload(path)
    mutate(payload)
    path.write_text(fq_repack.Signer(key).sign_line(payload) + "\n")
    return payload


def reattest(segments: Path, layer: int, k: int, key: Path) -> None:
    """Re-sign an attestation so it describes the segment's CURRENT bytes."""
    seg = segments / f"layer-{layer:03d}.k{k}.safetensors"
    hdr, _meta, body_off, _size = fq_assemble.validate_safetensors_file(seg)
    spans = {str(e): s for e, s in
             fq_assemble.expert_spans(hdr, body_off).items()}
    file_sha, per_expert = fq_assemble.digest_file_and_spans(seg, spans)

    def mutate(p):
        p["fragment"]["sha256"] = file_sha
        p["fragment"]["size"] = seg.stat().st_size
        p["expert_sha256"] = per_expert

    resign(att_path(segments, layer, k), key, mutate)


def flip_expert_byte(segments: Path, layer: int, k: int, expert: int = 0) -> None:
    """Corrupt one byte inside an expert payload, keeping the file size."""
    seg = segments / f"layer-{layer:03d}.k{k}.safetensors"
    hdr, _meta, body_off, _size = fq_assemble.validate_safetensors_file(seg)
    lo, _hi = fq_assemble.expert_spans(hdr, body_off)[expert]
    raw = bytearray(seg.read_bytes())
    raw[lo] ^= 0xFF
    seg.write_bytes(bytes(raw))


def test_unpinned_assembly_fails_closed(repacked):
    """No --trust-signer and no --insecure: refuse, and say how to pin."""
    snap, segments, tmp = repacked
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    with pytest.raises(fq_assemble.VerificationError, match="--trust-signer"):
        fq_assemble.main(["--segments", str(segments), "--source", str(snap),
                          "--policy", str(ppath), "--out", str(tmp / "nope")])
    assert not (tmp / "nope").exists(), "nothing may be written before trust"


def test_pinned_signer_happy_path_reports_verification(repacked, capsys):
    snap, segments, tmp = repacked
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    out = tmp / "asm"
    assert assemble(segments, snap, ppath, out) == 0
    assert "verified" in capsys.readouterr().out
    record = json.loads((out / "fq-assembly.json").read_text())
    assert record["verification"]["mode"] == "verified"
    assert record["verification"]["trusted_signers"] == [signer_of(segments)]
    assert record["verification"]["segments_verified"] == len(LAYERS)
    for frag in record["materials"]["segments"]["fragments"]:
        assert frag["verified"] and frag["predicate"] == "repack-of"
        assert frag["keyid"] == signer_of(segments)


def test_trust_file_pins_signer(repacked):
    snap, segments, tmp = repacked
    tf = tmp / "trusted.txt"
    tf.write_text(f"# operator-confirmed fingerprints\n{signer_of(segments)}\n")
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    assert assemble(segments, snap, ppath, tmp / "asm-tf",
                    "--trust-file", str(tf)) == 0


def test_trust_file_reads_the_project_trust_root_format(repacked):
    """`<fingerprint> <key-id> <status> ...` records, revoked ones skipped."""
    snap, segments, tmp = repacked
    pub = signer_of(segments)
    root = tmp / "FINGERPRINTS"
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    root.write_text("# trust root\n"
                    f"{pub}  malaiwah-fq-1  active  2026-08-10  builder\n")
    assert assemble(segments, snap, ppath, tmp / "asm-root",
                    "--trust-file", str(root)) == 0
    root.write_text(f"{pub}  malaiwah-fq-1  revoked  2026-08-10  builder\n")
    with pytest.raises(fq_assemble.VerificationError, match="refusing to assemble"):
        assemble(segments, snap, ppath, tmp / "asm-revoked",
                 "--trust-file", str(root))  # revoked pins nobody: fail closed


@pytest.mark.parametrize("kind", ("full", "prefix", "key-id", "pub-path"))
def test_trust_signer_tokens_match_shared_resolver(repacked, kind):
    """Assembler pins resolve exactly as fq_trust does, then verify normally."""
    snap, segments, tmp = repacked
    pub = signer_of(segments)
    root = write_trust_root(
        tmp / "FINGERPRINTS", (pub, "assemble-test-signer", "active"))
    pub_path = tmp / "assemble-test-signer.pub"
    pub_path.write_text(pub + "\n")
    token = {
        "full": pub,
        "prefix": pub[:16],
        "key-id": "assemble-test-signer",
        "pub-path": str(pub_path),
    }[kind]
    expected = fq_trust.Verifier.resolve(
        trust_signer=token, trust_root=root).fingerprint
    assert expected == pub

    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    assert assemble(segments, snap, ppath, tmp / f"asm-{kind}",
                    "--trust-root", str(root),
                    "--trust-signer", token) == 0


@pytest.mark.parametrize("kind", ("ed25519-prefix", "bare-pub-path"))
def test_legacy_trust_signer_tokens_remain_supported(repacked, monkeypatch, kind):
    """Legacy assembler spellings retain the same explicit-key verification."""
    snap, segments, tmp = repacked
    pub = signer_of(segments)
    root = write_trust_root(
        tmp / "FINGERPRINTS", (pub, "assemble-test-signer", "active"))
    pub_path = tmp / "assemble-test-signer.pub"
    pub_path.write_text(pub + "\n")
    if kind == "ed25519-prefix":
        token = f"ed25519:{pub}"
    else:
        token = pub_path.name
        monkeypatch.chdir(tmp)

    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    assert assemble(segments, snap, ppath, tmp / f"asm-{kind}",
                    "--trust-root", str(root),
                    "--trust-signer", token) == 0


def test_bare_pub_filename_cannot_override_trust_root_key_id(repacked, monkeypatch):
    """A local same-named .pub file cannot replace a publisher's key-id pin."""
    snap, segments, tmp = repacked
    pub = signer_of(segments)
    token = "publisher.pub"
    root = write_trust_root(tmp / "FINGERPRINTS", (pub, token, "active"))
    (tmp / token).write_text(
        fq_repack.Signer(tmp / "attacker.key").pub_hex + "\n")
    monkeypatch.chdir(tmp)

    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    out = tmp / "asm-key-id-precedence"
    assert assemble(segments, snap, ppath, out, "--trust-root", str(root),
                    "--trust-signer", token) == 0
    record = json.loads((out / "fq-assembly.json").read_text())
    assert record["verification"]["trusted_signers"] == [pub]


def test_trust_signer_rejections_match_shared_resolver(repacked):
    """Short, ambiguous, unknown, and revoked pins fail with fq_trust's text."""
    snap, segments, tmp = repacked
    pub = signer_of(segments)
    other_tail = "0" * 48 if pub[16:] != "0" * 48 else "1" * 48
    root = write_trust_root(
        tmp / "FINGERPRINTS",
        (pub, "assemble-test-signer", "active"),
        (pub[:16] + other_tail, "ambiguous-signer", "active"))
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})

    for name, token in (("short", pub[:15]), ("ambiguous", pub[:16]),
                        ("unknown", "missing-signer")):
        with pytest.raises(fq_trust.TrustError) as expected:
            fq_trust.Verifier.resolve(trust_signer=token, trust_root=root)
        with pytest.raises(fq_assemble.VerificationError) as actual:
            assemble(segments, snap, ppath, tmp / f"asm-{name}",
                     "--trust-root", str(root), "--trust-signer", token)
        assert str(actual.value) == str(expected.value)

    revoked = write_trust_root(
        tmp / "REVOKED", (pub, "assemble-test-signer", "revoked"))
    with pytest.raises(fq_trust.TrustError) as expected:
        fq_trust.Verifier.resolve(trust_signer=pub, trust_root=revoked)
    with pytest.raises(fq_assemble.VerificationError) as actual:
        assemble(segments, snap, ppath, tmp / "asm-revoked-explicit",
                 "--trust-root", str(revoked), "--trust-signer", pub)
    assert str(actual.value) == str(expected.value)


def test_short_publisher_pin_does_not_trust_local_subset(repacked):
    """Resolving a publisher prefix cannot bless a subset signed by local key."""
    snap, segments, tmp = repacked
    publisher = fq_repack.Signer(tmp / "publisher.key").pub_hex
    root = write_trust_root(
        tmp / "FINGERPRINTS", (publisher, "publisher-signer", "active"))
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    with pytest.raises(fq_assemble.VerificationError,
                       match="no attestation line signed"):
        assemble(segments, snap, ppath, tmp / "asm-local-subset",
                 "--trust-root", str(root),
                 "--trust-signer", publisher[:16])


def test_corrupted_segment_byte_fails(repacked):
    """One flipped payload byte must stop the assembly (finding 1's core)."""
    snap, segments, tmp = repacked
    flip_expert_byte(segments, LAYERS[0], 3)
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    with pytest.raises(fq_assemble.VerificationError, match="changed after it was attested"):
        assemble(segments, snap, ppath, tmp / "asm-corrupt")


def test_tampered_signature_fails(repacked):
    snap, segments, tmp = repacked
    path = att_path(segments, LAYERS[0], 3)
    line = json.loads(path.read_text())
    sig = bytearray(base64.b64decode(line["signature"]))
    sig[0] ^= 0xFF
    line["signature"] = base64.b64encode(bytes(sig)).decode()
    path.write_text(json.dumps(line) + "\n")
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    with pytest.raises(fq_assemble.VerificationError, match="does not verify"):
        assemble(segments, snap, ppath, tmp / "asm-badsig")


def test_wrong_signer_fails(repacked):
    """A perfectly valid attestation from a key we did not pin is not trust."""
    snap, segments, tmp = repacked
    other = fq_repack.Signer(tmp / "other.key").pub_hex
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    with pytest.raises(fq_assemble.VerificationError, match="no attestation line signed"):
        assemble(segments, snap, ppath, tmp / "asm-other",
                 "--trust-signer", other)


def test_attacker_resigning_with_own_key_fails(repacked):
    """Re-sign the doctored segment with an attacker key: still refused."""
    snap, segments, tmp = repacked
    flip_expert_byte(segments, LAYERS[0], 3)
    seg = segments / f"layer-{LAYERS[0]:03d}.k3.safetensors"
    hdr, _m, body_off, _s = fq_assemble.validate_safetensors_file(seg)
    spans = {str(e): s for e, s in fq_assemble.expert_spans(hdr, body_off).items()}
    file_sha, per_expert = fq_assemble.digest_file_and_spans(seg, spans)
    evil = fq_repack.Signer(tmp / "evil.key")
    payload = read_payload(att_path(segments, LAYERS[0], 3))
    payload["fragment"]["sha256"] = file_sha
    payload["expert_sha256"] = per_expert
    att_path(segments, LAYERS[0], 3).write_text(evil.sign_line(payload) + "\n")
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    with pytest.raises(fq_assemble.VerificationError, match="no attestation line signed"):
        assemble(segments, snap, ppath, tmp / "asm-evil")


def test_disallowed_predicate_fails(repacked):
    snap, segments, tmp = repacked
    resign(att_path(segments, LAYERS[0], 3), tmp / "k.key",
           lambda p: p.update(predicate="laundered-of"))
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    with pytest.raises(fq_assemble.VerificationError, match="not allowed"):
        assemble(segments, snap, ppath, tmp / "asm-pred")
    # ... and an explicitly widened allow-list accepts it
    assert assemble(segments, snap, ppath, tmp / "asm-pred-ok",
                    "--allow-predicate", "laundered-of",
                    "--allow-predicate", "repack-of") == 0


def test_expert_digest_mismatch_fails(repacked):
    """Whole-file digest repaired, per-expert digests stale: still refused."""
    snap, segments, tmp = repacked
    flip_expert_byte(segments, LAYERS[0], 3)
    seg = segments / f"layer-{LAYERS[0]:03d}.k3.safetensors"
    resign(att_path(segments, LAYERS[0], 3), tmp / "k.key",
           lambda p: p["fragment"].update(sha256=fq_repack.sha256_file(seg)))
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    with pytest.raises(fq_assemble.VerificationError, match="digest"):
        assemble(segments, snap, ppath, tmp / "asm-expert")


def test_missing_attestation_fails(repacked):
    snap, segments, tmp = repacked
    att_path(segments, LAYERS[0], 3).unlink()
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    with pytest.raises(fq_assemble.VerificationError, match="no attestation"):
        assemble(segments, snap, ppath, tmp / "asm-noatt")


def test_layout_mismatch_fails(repacked):
    snap, segments, tmp = repacked
    man = segments / "fq-manifest.json"
    m = json.loads(man.read_text())
    m["layout"] = "shared_h_v1"
    man.write_text(json.dumps(m))
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    with pytest.raises(fq_assemble.VerificationError, match="layout"):
        assemble(segments, snap, ppath, tmp / "asm-layout")


def test_base_model_mismatch_fails(repacked):
    """The recipe names the model it is for; segments from another one lose."""
    snap, segments, tmp = repacked
    p = tmp / "policy-bm.json"
    p.write_text(json.dumps({"schema": "fq-policy/2", "base_model": "other/model",
                             "bits_per_expert": {str(l): [3] * E for l in LAYERS}}))
    with pytest.raises(fq_assemble.VerificationError, match="base_model"):
        assemble(segments, snap, p, tmp / "asm-bm")


def test_segment_declaring_another_layer_fails(repacked):
    """Swap two segments' bytes and re-attest them honestly: the digests all
    check out, but each file now holds a foreign layer's weights."""
    snap, segments, tmp = repacked
    a = segments / f"layer-{LAYERS[0]:03d}.k3.safetensors"
    b = segments / f"layer-{LAYERS[1]:03d}.k3.safetensors"
    a_raw, b_raw = a.read_bytes(), b.read_bytes()
    a.write_bytes(b_raw)
    b.write_bytes(a_raw)
    for layer in LAYERS:
        reattest(segments, layer, 3, tmp / "k.key")
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    with pytest.raises(fq_assemble.VerificationError, match="declares layer"):
        assemble(segments, snap, ppath, tmp / "asm-swap")


def test_insecure_warns_loudly_and_records_it(repacked, capsys):
    snap, segments, tmp = repacked
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    out = tmp / "asm-insecure"
    assert assemble(segments, snap, ppath, out, "--insecure") == 0
    err = capsys.readouterr().err
    assert "--insecure" in err and "NOT being verified" in err
    record = json.loads((out / "fq-assembly.json").read_text())
    assert record["verification"]["mode"] == "INSECURE (unverified)"
    assert record["verification"]["segments_verified"] == 0


def test_bad_trust_fingerprint_rejected(repacked):
    snap, segments, tmp = repacked
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    with pytest.raises(fq_assemble.VerificationError, match="not a fingerprint"):
        assemble(segments, snap, ppath, tmp / "asm-hex", "--trust-signer", "zz")
    with pytest.raises(fq_assemble.VerificationError, match="no key with that prefix"):
        assemble(segments, snap, ppath, tmp / "asm-len", "--trust-signer", "ab" * 16)


# ------------------------------- FINDING 1: strict safetensors bounds checks

def rewrite_header(path: Path, mutate) -> None:
    hdr, body_off = fq_repack.read_header(path)
    body = path.read_bytes()[body_off:]
    mutate(hdr)
    hj = json.dumps(hdr, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(hj)) + hj + body)


def some_tensor(hdr: dict) -> str:
    return next(n for n in hdr if n != "__metadata__")


def test_out_of_bounds_offsets_rejected_before_any_read(repacked):
    snap, segments, tmp = repacked
    seg = segments / f"layer-{LAYERS[0]:03d}.k3.safetensors"

    def blow_up(hdr):
        name = some_tensor(hdr)
        a, b = hdr[name]["data_offsets"]
        hdr[name]["data_offsets"] = [a, b + (1 << 30)]
        hdr[name]["shape"] = [(b - a + (1 << 30)) // 2]

    rewrite_header(seg, blow_up)
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    with pytest.raises(fq_assemble.SegmentIntegrityError, match="outside the"):
        assemble(segments, snap, ppath, tmp / "asm-oob")


def test_overlapping_tensors_rejected(repacked):
    snap, segments, tmp = repacked
    seg = segments / f"layer-{LAYERS[0]:03d}.k3.safetensors"

    def overlap(hdr):
        names = sorted((n for n in hdr if n != "__metadata__"),
                       key=lambda n: hdr[n]["data_offsets"][0])
        a, b = hdr[names[1]]["data_offsets"]
        hdr[names[1]]["data_offsets"] = [a - 2, b - 2]

    rewrite_header(seg, overlap)
    with pytest.raises(fq_assemble.SegmentIntegrityError, match="overlaps"):
        fq_assemble.validate_safetensors_file(seg)


def test_dtype_shape_byte_length_must_agree(repacked):
    seg = repacked[1] / f"layer-{LAYERS[0]:03d}.k3.safetensors"
    rewrite_header(seg, lambda h: h[some_tensor(h)].update(shape=[99]))
    with pytest.raises(fq_assemble.SegmentIntegrityError, match="dtype/shape imply"):
        fq_assemble.validate_safetensors_file(seg)


def test_unknown_dtype_rejected(repacked):
    seg = repacked[1] / f"layer-{LAYERS[0]:03d}.k3.safetensors"
    rewrite_header(seg, lambda h: h[some_tensor(h)].update(dtype="F13"))
    with pytest.raises(fq_assemble.SegmentIntegrityError, match="unknown dtype"):
        fq_assemble.validate_safetensors_file(seg)


def test_absurd_header_length_rejected(tmp_path):
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(struct.pack("<Q", 1 << 40) + b"{}")
    with pytest.raises(fq_assemble.SegmentIntegrityError, match="does not fit"):
        fq_assemble.validate_safetensors_file(bad)
    empty = tmp_path / "empty.safetensors"
    empty.write_bytes(b"\x00\x00\x00")
    with pytest.raises(fq_assemble.SegmentIntegrityError, match="too small"):
        fq_assemble.validate_safetensors_file(empty)


def test_non_contiguous_expert_rejected(repacked):
    """Per-expert contiguity is the format's promise; without it the signed
    per-expert digests cannot be checked at all."""
    snap, segments, tmp = repacked
    seg = segments / f"layer-{LAYERS[0]:03d}.k3.safetensors"

    def shift(hdr):
        names = sorted((n for n in hdr if n != "__metadata__"),
                       key=lambda n: hdr[n]["data_offsets"][0])
        last = names[-1]
        a, b = hdr[last]["data_offsets"]
        hdr[last]["data_offsets"] = [a + 8, b + 8]  # leave an 8-byte hole

    rewrite_header(seg, shift)
    (seg).write_bytes(seg.read_bytes() + b"\x00" * 8)
    with pytest.raises(fq_assemble.SegmentIntegrityError, match="not contiguous"):
        hdr, _m, body_off, _s = fq_assemble.validate_safetensors_file(seg)
        fq_assemble.expert_spans(hdr, body_off)


# ------------------------------ FINDING 4: output integrity metadata is true

def test_manifest_detects_same_size_payload_change(repacked):
    """A same-size payload change used to keep a stale MANIFEST.sha256."""
    snap, segments, tmp = repacked
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    out = tmp / "asm-restamp"
    assemble(segments, snap, ppath, out)
    first = dict(line.split()[::-1] for line in
                 (out / "MANIFEST.sha256").read_text().splitlines())

    flip_expert_byte(segments, LAYERS[0], 3)
    reattest(segments, LAYERS[0], 3, tmp / "k.key")
    assemble(segments, snap, ppath, out, "--force")

    second = dict(line.split()[::-1] for line in
                  (out / "MANIFEST.sha256").read_text().splitlines())
    changed = f"model-layer-{LAYERS[0]:03d}.safetensors"
    assert second[changed] != first[changed], "same-size change went unnoticed"
    for name, digest in second.items():
        assert digest == hashlib.sha256((out / name).read_bytes()).hexdigest(), name
    assert second[changed] == hashlib.sha256((out / changed).read_bytes()).hexdigest()


def test_uniform_k4_config_cannot_contradict_the_tier_bitmap(repacked_multi_k):
    """setdefault left quantization_config.bits at the source's 3.0."""
    snap, segments, tmp = repacked_multi_k
    ppath = policy_file(tmp, {str(l): [4] * E for l in LAYERS}, "policy-k4.json")
    out = tmp / "asm-k4"
    assert assemble(segments, snap, ppath, out) == 0
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["hybrid_tr3_tail"]["bits"] == 4.0
    assert cfg["quantization_config"]["bits"] == 4.0
    bitmap = json.loads((out / "tier_bitmap.json").read_text())
    for layer in LAYERS:
        assert bitmap[str(layer)]["bits_per_expert"] == [4] * E


def test_mixed_config_and_tier_bitmap_agree(repacked_multi_k):
    snap, segments, tmp = repacked_multi_k
    bits = {str(LAYERS[0]): [3, 4, 4, 3], str(LAYERS[1]): [4, 4, 3, 3]}
    ppath = policy_file(tmp, bits, "policy-mix2.json")
    out = tmp / "asm-mix2"
    assert assemble(segments, snap, ppath, out) == 0
    cfg = json.loads((out / "config.json").read_text())
    tail = cfg["hybrid_tr3_tail"]
    assert tail["bits"] == "mixed" and cfg["quantization_config"]["bits"] == "mixed"
    fname, field = tail["bits_per_expert"].rsplit(":", 1)
    bitmap = json.loads((out / fname).read_text())
    for layer_s, want in bits.items():
        assert bitmap[layer_s][field] == want
    # every K named by the config is a K the bitmap actually uses
    assert sorted({b for v in bits.values() for b in v}) == tail["k_values"]


def test_assembly_of_record_validates(repacked):
    snap, segments, tmp = repacked
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    out = tmp / "asm-record"
    assemble(segments, snap, ppath, out, "--sign-key", str(tmp / "asm.key"))
    record = json.loads((out / "fq-assembly.json").read_text())
    assert record["schema"] == "fq-attestation/2"
    assert record["predicate"] == "assembly-of"
    assert record["tool"] == {"name": "fq_assemble", "version": fq_assemble.TOOL_VERSION}
    assert record["created_utc"].endswith("Z")
    assert record["recipe"]["sha256"] == fq_repack.sha256_file(ppath)
    assert record["recipe"]["k_values"] == [3]
    assert record["recipe"]["layers"] == LAYERS
    frags = record["materials"]["segments"]["fragments"]
    assert [f["layer"] for f in frags] == LAYERS
    for frag in frags:
        assert frag["sha256"] == fq_repack.sha256_file(segments / frag["file"])
        assert frag["experts"] == E
    products = {p["file"]: p["sha256"] for p in record["products"]}
    assert set(products) == {f"model-layer-{l:03d}.safetensors" for l in LAYERS}
    for name, digest in products.items():
        assert digest == fq_repack.sha256_file(out / name)
    # the signed line carries the same record
    from nacl.signing import VerifyKey
    line = json.loads((out / "attestations" / "assembly-of.jsonl").read_text())
    raw = base64.b64decode(line["payload"])
    VerifyKey(bytes.fromhex(line["keyid"])).verify(raw, base64.b64decode(line["signature"]))
    assert json.loads(raw) == record
    # ... and MANIFEST.sha256 covers the record and the signature
    manifest = dict(line.split()[::-1] for line in
                    (out / "MANIFEST.sha256").read_text().splitlines())
    assert manifest["fq-assembly.json"] == fq_repack.sha256_file(out / "fq-assembly.json")
    assert "attestations/assembly-of.jsonl" in manifest


def test_out_dir_must_be_empty_or_force(repacked):
    snap, segments, tmp = repacked
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    out = tmp / "asm-dirty"
    out.mkdir()
    stale = out / "model-layer-099.safetensors"
    stale.write_bytes(b"leftovers from another recipe")
    with pytest.raises(fq_assemble.AssemblyError, match="not empty"):
        assemble(segments, snap, ppath, out)
    assert stale.exists(), "a refused run must not delete anything"
    # --force alone is not enough: this dir was not made by fq_assemble
    with pytest.raises(fq_assemble.AssemblyError, match="no .fq-assembly"):
        assemble(segments, snap, ppath, out, "--force")
    assert stale.exists(), "a refused --force must not delete anything"
    # marked as a previous assembly, --force replaces it
    fq_assemble.write_sentinel(out)
    assert assemble(segments, snap, ppath, out, "--force") == 0
    assert not stale.exists(), "--force must replace, not merge"
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert not any(v == stale.name for v in index["weight_map"].values())


# --------------------------------------------------- destructive --out (P1-5)

def _dirty(path: Path) -> Path:
    """A non-empty directory standing in for the user's data."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "precious.safetensors").write_bytes(b"the user's model")
    return path


def test_assembly_writes_the_sentinel_it_later_requires(repacked):
    """Every output carries the marker that authorizes its own replacement."""
    snap, segments, tmp = repacked
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    out = tmp / "asm-sentinel"
    assert assemble(segments, snap, ppath, out) == 0
    marker = json.loads((out / fq_assemble.SENTINEL).read_text())
    assert marker["schema"] == fq_assemble.SENTINEL_SCHEMA
    # ... and it is covered by the checkpoint's own MANIFEST.sha256
    manifest = dict(line.split()[::-1] for line in
                    (out / "MANIFEST.sha256").read_text().splitlines())
    assert fq_assemble.SENTINEL in manifest
    # a second --force run over our own output is allowed
    assert assemble(segments, snap, ppath, out, "--force") == 0


def test_out_refuses_the_filesystem_root(repacked):
    snap, segments, tmp = repacked
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    with pytest.raises(fq_assemble.AssemblyError, match="filesystem root"):
        assemble(segments, snap, ppath, Path("/"), "--force")


def test_out_refuses_the_home_directory(repacked, monkeypatch):
    snap, segments, tmp = repacked
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    home = _dirty(tmp / "home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    with pytest.raises(fq_assemble.AssemblyError, match="home directory"):
        assemble(segments, snap, ppath, home, "--force")
    assert (home / "precious.safetensors").exists()


def test_out_refuses_a_parent_of_the_home_directory(repacked, monkeypatch):
    snap, segments, tmp = repacked
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    home = _dirty(tmp / "houses" / "home")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    with pytest.raises(fq_assemble.AssemblyError, match="contains your home"):
        assemble(segments, snap, ppath, tmp / "houses", "--force")
    assert (home / "precious.safetensors").exists()


def test_out_refuses_this_checkout(repacked):
    """--out <the tools repo> would assemble over fq_assemble itself."""
    snap, segments, tmp = repacked
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    checkout = Path(fq_assemble.__file__).resolve().parent
    with pytest.raises(fq_assemble.AssemblyError, match="checkout"):
        assemble(segments, snap, ppath, checkout, "--force")
    assert Path(fq_assemble.__file__).exists()


def test_out_refuses_a_git_working_tree(repacked):
    snap, segments, tmp = repacked
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    repo = _dirty(tmp / "someones-repo")
    (repo / ".git").mkdir()
    with pytest.raises(fq_assemble.AssemblyError, match="git checkout"):
        assemble(segments, snap, ppath, repo, "--force")
    assert (repo / "precious.safetensors").exists()


def test_out_refuses_a_symlink(repacked):
    """Purging through a symlink empties whatever it points at."""
    snap, segments, tmp = repacked
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    target = _dirty(tmp / "real-model")
    link = tmp / "link-to-model"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(fq_assemble.AssemblyError, match="symlink"):
        assemble(segments, snap, ppath, link, "--force")
    assert (target / "precious.safetensors").exists()


def test_out_refuses_the_source_checkpoint(repacked):
    """The review's headline case: --out <the source> with --force."""
    snap, segments, tmp = repacked
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    with pytest.raises(fq_assemble.AssemblyError, match="overlaps --source"):
        assemble(segments, snap, ppath, snap, "--force")
    assert sorted(snap.glob("model-layer-*.safetensors")), "source untouched"


def test_out_refuses_a_parent_of_the_source(repacked):
    snap, segments, tmp = repacked
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    with pytest.raises(fq_assemble.AssemblyError, match="overlaps --source"):
        assemble(segments, snap, ppath, snap.parent, "--force")
    assert sorted(snap.glob("model-layer-*.safetensors"))


def test_out_refuses_a_dir_inside_the_segments(repacked):
    snap, segments, tmp = repacked
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    with pytest.raises(fq_assemble.AssemblyError, match="overlaps --segments"):
        assemble(segments, snap, ppath, segments / "nested", "--force")
    assert sorted(segments.glob("*.safetensors"))


def test_out_refuses_to_swallow_the_policy_file(repacked):
    snap, segments, tmp = repacked
    recipes = tmp / "recipes"
    recipes.mkdir()
    ppath = policy_file(recipes, {str(l): [3] * E for l in LAYERS})
    with pytest.raises(fq_assemble.AssemblyError, match="policy file"):
        assemble(segments, snap, ppath, recipes, "--force")
    assert ppath.exists()


def test_out_refuses_the_upstream_cache(repacked, monkeypatch):
    snap, segments, tmp = repacked
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    cache = _dirty(tmp / "hf-cache")
    monkeypatch.setenv("HF_HOME", str(cache))
    with pytest.raises(fq_assemble.AssemblyError, match="model cache"):
        assemble(segments, snap, ppath, cache / "hub", "--force")
    assert (cache / "precious.safetensors").exists()


def test_out_next_to_the_policy_is_fine(repacked):
    """The refusals must not ban the ordinary layout (recipe beside output)."""
    snap, segments, tmp = repacked
    recipes = tmp / "work"
    recipes.mkdir()
    ppath = policy_file(recipes, {str(l): [3] * E for l in LAYERS})
    assert assemble(segments, snap, ppath, recipes / "out") == 0
    assert ppath.exists()


def test_failed_assembly_leaves_the_previous_output_intact(repacked, monkeypatch):
    """Staging + rename: an interrupted run can never half-purge --out."""
    snap, segments, tmp = repacked
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    out = tmp / "asm-staged"
    assert assemble(segments, snap, ppath, out) == 0
    before = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    assert before

    boom = [0]

    def explode(*a, **kw):
        boom[0] += 1
        raise KeyboardInterrupt("simulated interrupt mid-assembly")

    monkeypatch.setattr(fq_assemble, "assemble_layer", explode)
    with pytest.raises(KeyboardInterrupt):
        assemble(segments, snap, ppath, out, "--force")
    assert boom[0] == 1
    after = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    assert after == before, "the previous checkpoint must survive a failed run"
    leftovers = [p.name for p in tmp.iterdir() if ".fq-staging" in p.name
                 or ".fq-trash" in p.name]
    assert not leftovers, f"staging dirs must be cleaned up: {leftovers}"


def test_failed_first_assembly_creates_no_output_dir(repacked, monkeypatch):
    snap, segments, tmp = repacked
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    out = tmp / "asm-never"

    def explode(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(fq_assemble, "assemble_layer", explode)
    with pytest.raises(OSError, match="disk full"):
        assemble(segments, snap, ppath, out)
    assert not out.exists(), "a failed run must not leave a partial --out"
    assert not list(tmp.glob(".*fq-staging*"))


def test_index_and_manifest_always_regenerated(repacked):
    """Even a byte-identical uniform assembly restamps its own metadata."""
    snap, segments, tmp = repacked
    ppath = policy_file(tmp, {str(l): [3] * E for l in LAYERS})
    out = tmp / "asm-always"
    assemble(segments, snap, ppath, out)
    assert (out / "model.safetensors.index.json").exists()
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert index["weight_map"], "index rebuilt from the shards actually written"
    manifest = dict(line.split()[::-1] for line in
                    (out / "MANIFEST.sha256").read_text().splitlines())
    for f in out.rglob("*"):
        if f.is_file() and f.name != "MANIFEST.sha256":
            rel = f.relative_to(out).as_posix()
            assert manifest[rel] == fq_repack.sha256_file(f), rel


# ---------------------------------------------------------------- reflink

def _k3_policy_file(root: Path) -> Path:
    return policy_file(root, {str(l): [3] * E for l in LAYERS},
                       "policy-k3-reflink.json")


def _assemble(segments, snap, ppath, out, *extra):
    assert assemble(segments, snap, ppath, out, *extra) == 0


def test_reflink_output_byte_identical(repacked):
    """--reflink output == plain output == source shards (the invariant).

    pytest's tmp_path typically lives on tmpfs/overlayfs where extent
    sharing cannot happen: copy_file_range either plain-copies in kernel or
    the assembler falls back per region.  Either way byte identity must
    hold — that IS the mode's contract (it changes how bytes move, never
    what they are), so this test asserts nothing about extent sharing."""
    snap, segments, tmp = repacked
    ppath = _k3_policy_file(tmp)
    plain, refl = tmp / "asm-plain", tmp / "asm-reflink"
    _assemble(segments, snap, ppath, plain)
    _assemble(segments, snap, ppath, refl, "--reflink")
    for layer in LAYERS:
        name = f"model-layer-{layer:03d}.safetensors"
        src = (snap / name).read_bytes()
        assert (plain / name).read_bytes() == src, name
        assert (refl / name).read_bytes() == src, name


def test_reflink_mixed_policy_byte_identical(repacked_multi_k):
    """--reflink through the mixed-K reindex path: identical to plain."""
    snap, segments, tmp = repacked_multi_k
    bits = {str(LAYERS[0]): [3, 4, 4, 3], str(LAYERS[1]): [3] * E}
    ppath = policy_file(tmp, bits, "policy-mixed-reflink.json")
    plain, refl = tmp / "asm-mixed-plain", tmp / "asm-mixed-reflink"
    _assemble(segments, snap, ppath, plain)
    _assemble(segments, snap, ppath, refl, "--reflink")
    for layer in LAYERS:
        name = f"model-layer-{layer:03d}.safetensors"
        assert (refl / name).read_bytes() == (plain / name).read_bytes(), name


def test_reflink_falls_back_when_copy_file_range_fails(repacked, monkeypatch):
    """copy_file_range raising (EXDEV: cross-filesystem) per call must leave
    a byte-identical output via the ordinary per-region fallback.

    Patched at fq_assemble's own seam, not on os: macOS has no
    os.copy_file_range to patch, and the test is about the assembler's
    fallback, not about the platform."""
    snap, segments, tmp = repacked
    calls = []

    def exdev(*a, **kw):
        calls.append(a)
        raise OSError(errno.EXDEV, "cross-device link (simulated)")

    monkeypatch.setattr(fq_assemble, "_COPY_FILE_RANGE", exdev)
    ppath = _k3_policy_file(tmp)
    out = tmp / "asm-exdev"
    _assemble(segments, snap, ppath, out, "--reflink")
    assert calls, "reflink path never attempted copy_file_range"
    for layer in LAYERS:
        name = f"model-layer-{layer:03d}.safetensors"
        assert (out / name).read_bytes() == (snap / name).read_bytes(), name


def test_reflink_fallback_after_partial_copy(repacked, monkeypatch):
    """A copy_file_range that reports partial progress — even progress that
    wrote WRONG bytes — then fails mid-region must not corrupt the output:
    the assembler rewinds and rewrites the whole region."""
    snap, segments, tmp = repacked
    state = {"armed": True}

    def flaky(src_fd, dst_fd, count, offset_src=0, offset_dst=0):
        if state["armed"]:
            state["armed"] = False
            n = min(count, 7)
            os.pwrite(dst_fd, b"\xff" * n, offset_dst)  # garbage progress
            return n
        raise OSError(errno.EIO, "flaky (simulated)")

    monkeypatch.setattr(fq_assemble, "_COPY_FILE_RANGE", flaky)
    ppath = _k3_policy_file(tmp)
    out = tmp / "asm-flaky"
    _assemble(segments, snap, ppath, out, "--reflink")
    for layer in LAYERS:
        name = f"model-layer-{layer:03d}.safetensors"
        assert (out / name).read_bytes() == (snap / name).read_bytes(), name


def test_reflink_when_copy_file_range_absent(repacked, monkeypatch):
    """No copy_file_range at all (macOS, pre-3.8, exotic platforms):
    --reflink degrades to the ordinary copy path, byte-identically."""
    snap, segments, tmp = repacked
    monkeypatch.setattr(fq_assemble, "_COPY_FILE_RANGE", None)
    ppath = _k3_policy_file(tmp)
    out = tmp / "asm-nocfr"
    _assemble(segments, snap, ppath, out, "--reflink")
    for layer in LAYERS:
        name = f"model-layer-{layer:03d}.safetensors"
        assert (out / name).read_bytes() == (snap / name).read_bytes(), name


@pytest.mark.skipif(fq_assemble._COPY_FILE_RANGE is None,
                    reason="os.copy_file_range unavailable on this platform")
def test_reflink_on_local_filesystem(monkeypatch):
    """Integration: source, segments, and output all on the filesystem that
    hosts this repo (XFS with reflink=1 on the dev box), asserting byte
    identity AND that copy_file_range was actually used (no fallback).

    Deliberately does NOT assert actual extent sharing: whether the kernel
    reflinks or plain-copies is filesystem policy (caveat 1 of the mode)
    and there is no portable userspace way to verify it — FIEMAP/filefrag
    output is neither universally available nor stable across filesystems.
    Skips where the local filesystem refuses copy_file_range."""
    here = Path(__file__).resolve().parent
    try:
        tdctx = tempfile.TemporaryDirectory(dir=here, prefix=".reflink-it-")
    except OSError as e:
        pytest.skip(f"cannot create temp dir beside the tests: {e}")
    with tdctx as td:
        root = Path(td)
        # probe: does this filesystem accept copy_file_range at all?
        probe_src, probe_dst = root / "p.src", root / "p.dst"
        probe_src.write_bytes(b"x" * 4096)
        with open(probe_src, "rb") as s, open(probe_dst, "wb") as d:
            try:
                fq_assemble._COPY_FILE_RANGE(s.fileno(), d.fileno(), 4096, 0, 0)
            except OSError as e:
                pytest.skip(f"filesystem refuses copy_file_range: {e}")
        snap, segments = _build_k3_workspace(root)
        real = fq_assemble._COPY_FILE_RANGE
        used = []

        def recording(*a, **kw):
            n = real(*a, **kw)
            used.append(n)
            return n

        monkeypatch.setattr(fq_assemble, "_COPY_FILE_RANGE", recording)
        ppath = _k3_policy_file(root)
        out = root / "asm-local"
        _assemble(segments, snap, ppath, out, "--reflink")
        assert used, "copy_file_range was never used on a supporting fs"
        for layer in LAYERS:
            name = f"model-layer-{layer:03d}.safetensors"
            assert (out / name).read_bytes() == (snap / name).read_bytes(), name
