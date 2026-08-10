"""Tests for fq_assemble: byte-identity round trip through repack + assemble."""
import errno
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import fq_repack  # noqa: E402
import fq_assemble  # noqa: E402
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


@pytest.fixture()
def repacked(tmp_path):
    snap, out = _build_k3_workspace(tmp_path)
    return snap, out, tmp_path


def test_all_k3_assembly_is_byte_identical(repacked):
    snap, segments, tmp = repacked
    policy = {
        "schema": "fq-policy/2",
        "bits_per_expert": {str(l): [3] * E for l in LAYERS},
    }
    ppath = tmp / "policy.json"
    ppath.write_text(json.dumps(policy))
    out = tmp / "assembled"
    fq_assemble.main([
        "--segments", str(segments), "--source", str(snap),
        "--policy", str(ppath), "--out", str(out),
    ])
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
        }}))
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
    policy = {
        "schema": "fq-policy/2",
        "bits_per_expert": {str(mixed_layer): mixed_bits,
                            str(uniform_layer): [3] * E},
    }
    ppath = tmp / "policy-mixed.json"
    ppath.write_text(json.dumps(policy))
    out = tmp / "assembled-mixed"
    assert fq_assemble.main([
        "--segments", str(segments), "--source", str(snap),
        "--policy", str(ppath), "--out", str(out),
    ]) == 0

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
    policy = {"schema": "fq-policy/2", "bits_per_expert": bits}
    ppath = tmp / "policy-meta.json"
    ppath.write_text(json.dumps(policy))
    out = tmp / "assembled-meta"
    assert fq_assemble.main([
        "--segments", str(segments), "--source", str(snap),
        "--policy", str(ppath), "--out", str(out),
    ]) == 0

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
    policy = {"schema": "fq-policy/2",
              "bits_per_expert": {str(l): [3] * E for l in LAYERS}}
    ppath = tmp / "policy-k3.json"
    ppath.write_text(json.dumps(policy))
    out = tmp / "assembled-k3"
    assert fq_assemble.main([
        "--segments", str(segments), "--source", str(snap),
        "--policy", str(ppath), "--out", str(out),
    ]) == 0
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
    policy = {"schema": "fq-policy/2",
              "bits_per_expert": {str(LAYERS[0]): [4] * E}}
    ppath = tmp / "p4.json"
    ppath.write_text(json.dumps(policy))
    with pytest.raises(FileNotFoundError, match="k4"):
        fq_assemble.main([
            "--segments", str(segments), "--source", str(snap),
            "--policy", str(ppath), "--out", str(tmp / "x"),
        ])


# ---------------------------------------------------------------- reflink

def _k3_policy_file(root: Path) -> Path:
    ppath = root / "policy-k3-reflink.json"
    ppath.write_text(json.dumps({
        "schema": "fq-policy/2",
        "bits_per_expert": {str(l): [3] * E for l in LAYERS},
    }))
    return ppath


def _assemble(segments, snap, ppath, out, *extra):
    assert fq_assemble.main([
        "--segments", str(segments), "--source", str(snap),
        "--policy", str(ppath), "--out", str(out), *extra,
    ]) == 0


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
    ppath = tmp / "policy-mixed-reflink.json"
    ppath.write_text(json.dumps({"schema": "fq-policy/2",
                                 "bits_per_expert": bits}))
    plain, refl = tmp / "asm-mixed-plain", tmp / "asm-mixed-reflink"
    _assemble(segments, snap, ppath, plain)
    _assemble(segments, snap, ppath, refl, "--reflink")
    for layer in LAYERS:
        name = f"model-layer-{layer:03d}.safetensors"
        assert (refl / name).read_bytes() == (plain / name).read_bytes(), name


def test_reflink_falls_back_when_copy_file_range_fails(repacked, monkeypatch):
    """copy_file_range raising (EXDEV: cross-filesystem) per call must leave
    a byte-identical output via the ordinary per-region fallback."""
    snap, segments, tmp = repacked
    calls = []

    def exdev(*a, **kw):
        calls.append(a)
        raise OSError(errno.EXDEV, "cross-device link (simulated)")

    monkeypatch.setattr(os, "copy_file_range", exdev)
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

    monkeypatch.setattr(os, "copy_file_range", flaky)
    ppath = _k3_policy_file(tmp)
    out = tmp / "asm-flaky"
    _assemble(segments, snap, ppath, out, "--reflink")
    for layer in LAYERS:
        name = f"model-layer-{layer:03d}.safetensors"
        assert (out / name).read_bytes() == (snap / name).read_bytes(), name


def test_reflink_when_copy_file_range_absent(repacked, monkeypatch):
    """No os.copy_file_range at all (pre-3.8 / exotic platform): --reflink
    degrades to the ordinary copy path, byte-identically."""
    snap, segments, tmp = repacked
    monkeypatch.delattr(os, "copy_file_range")
    ppath = _k3_policy_file(tmp)
    out = tmp / "asm-nocfr"
    _assemble(segments, snap, ppath, out, "--reflink")
    for layer in LAYERS:
        name = f"model-layer-{layer:03d}.safetensors"
        assert (out / name).read_bytes() == (snap / name).read_bytes(), name


def test_reflink_on_local_filesystem(monkeypatch):
    """Integration: source, segments, and output all on the filesystem that
    hosts this repo (XFS with reflink=1 on the dev box), asserting byte
    identity AND that copy_file_range was actually used (no fallback).

    Deliberately does NOT assert actual extent sharing: whether the kernel
    reflinks or plain-copies is filesystem policy (caveat 1 of the mode)
    and there is no portable userspace way to verify it — FIEMAP/filefrag
    output is neither universally available nor stable across filesystems.
    Skips where the local filesystem refuses copy_file_range."""
    if not hasattr(os, "copy_file_range"):
        pytest.skip("os.copy_file_range unavailable on this platform")
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
                os.copy_file_range(s.fileno(), d.fileno(), 4096, 0, 0)
            except OSError as e:
                pytest.skip(f"filesystem refuses copy_file_range: {e}")
        snap, segments = _build_k3_workspace(root)
        real = os.copy_file_range
        used = []

        def recording(*a, **kw):
            n = real(*a, **kw)
            used.append(n)
            return n

        monkeypatch.setattr(os, "copy_file_range", recording)
        ppath = _k3_policy_file(root)
        out = root / "asm-local"
        _assemble(segments, snap, ppath, out, "--reflink")
        assert used, "copy_file_range was never used on a supporting fs"
        for layer in LAYERS:
            name = f"model-layer-{layer:03d}.safetensors"
            assert (out / name).read_bytes() == (snap / name).read_bytes(), name
