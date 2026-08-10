"""Tests for fq_assemble: byte-identity round trip through repack + assemble."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import fq_repack  # noqa: E402
import fq_assemble  # noqa: E402
from test_fq_repack import LAYERS, E, write_shard, tensor_bytes  # noqa: E402


@pytest.fixture()
def repacked(tmp_path):
    snap = tmp_path / "snap"
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
    out = tmp_path / "segments"
    fq_repack.main([
        "--snapshot", str(snap),
        "--source-repo", "test/src", "--revision", "r0",
        "--base-model", "test/base", "--out", str(out),
        "--sign-key", str(tmp_path / "k.key"),
    ])
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
