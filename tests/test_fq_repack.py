"""Tests for fq_repack: byte-identity, contiguity, attestation, resume, range-read."""
import base64
import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import fq_repack  # noqa: E402

E, RANKS = 4, 2
LAYERS = [3, 4]
PROJS = ["gate_proj", "up_proj", "down_proj"]
COMPS = ["trellis", "suh", "svh", "mcg"]


def tensor_bytes(layer, e, proj, rank, comp, k=3) -> bytes:
    """Synthetic tensor payload; trellis size scales with K (16*K bytes, the
    same 4/3 K3->K4 ratio as real [in/16, out/16, 16*K] i16 tensors) and all
    expert tensor CONTENT depends on K (independent encodes per K)."""
    seed = f"{layer}.{e}.{proj}.{rank}.{comp}.k{k}".encode()
    h = hashlib.sha256(seed).digest()
    return (h * 3)[: 16 * k if comp == "trellis" else 16]


def write_shard(path: Path, layer: int, scramble: bool, k: int = 3) -> None:
    """Synthetic shard; scramble=True stores tensors in non-logical order."""
    entries = []
    for e in range(E):
        for proj in PROJS:
            for rank in range(RANKS):
                for comp in COMPS:
                    name = f"model.layers.{layer}.mlp.experts.{e}.{proj}.rank{rank}.{comp}"
                    entries.append((name, tensor_bytes(layer, e, proj, rank, comp, k)))
    entries.append((f"model.layers.{layer}.self_attn.o_proj.weight", b"\x01" * 32))
    if scramble:
        entries.sort(key=lambda kv: hashlib.md5(kv[0].encode()).hexdigest())
    hdr, off, blobs = {}, 0, []
    for name, data in entries:
        hdr[name] = {
            "dtype": "I16" if name.endswith(".trellis") else "F16",
            "shape": [len(data) // 2],
            "data_offsets": [off, off + len(data)],
        }
        blobs.append(data)
        off += len(data)
    hj = json.dumps({"__metadata__": {"format": "pt"}, **hdr}, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)


@pytest.fixture()
def workspace(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    for i, layer in enumerate(LAYERS):
        write_shard(snap / f"model-layer-{layer:03d}.safetensors", layer, scramble=bool(i))
    lines = [
        f"{hashlib.sha256((snap / f'model-layer-{l:03d}.safetensors').read_bytes()).hexdigest()}"
        f"  model-layer-{l:03d}.safetensors"
        for l in LAYERS
    ]
    (snap / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    return snap, tmp_path / "out", tmp_path / "sign.key"


def run(snap, out, key):
    return fq_repack.main(
        [
            "--snapshot", str(snap),
            "--source-repo", "test/source-repo",
            "--revision", "deadbeef",
            "--base-model", "test/base",
            "--out", str(out),
            "--sign-key", str(key),
        ]
    )


def test_round_trip_byte_identity(workspace):
    snap, out, key = workspace
    assert run(snap, out, key) == 0
    for layer in LAYERS:
        seg = out / f"layer-{layer:03d}.k3.safetensors"
        hdr, body = fq_repack.read_header(seg)
        meta = hdr.pop("__metadata__")
        assert meta["fq_schema"] == "fq-segment/1"
        assert meta["predicate"] == "repack-of"
        assert meta["layout"] == "rank_sliced_tp4"
        assert meta["source_revision"] == "deadbeef"
        assert meta["source_file_sha256"]
        raw = seg.read_bytes()[body:]
        assert len(hdr) == E * len(PROJS) * RANKS * len(COMPS)
        for name, t in hdr.items():
            m = fq_repack.EXPERT_RE.match(name)
            layer_s, e, proj, rank, comp = m.groups()
            expect = tensor_bytes(int(layer_s), int(e), proj, int(rank), comp)
            a, b = t["data_offsets"]
            assert raw[a:b] == expect, name


def test_per_expert_contiguity_and_index_range_read(workspace):
    snap, out, key = workspace
    run(snap, out, key)
    index = json.loads((out / "index-k3.json").read_text())
    for layer in LAYERS:
        entry = index[str(layer)]
        seg = out / entry["file"]
        raw = seg.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        hdr, body = fq_repack.read_header(seg)
        hdr.pop("__metadata__")
        assert body == entry["body_offset"]
        for e_str, (lo, hi) in entry["experts"].items():
            span = raw[body + lo : body + hi]
            names = sorted(
                (fq_repack.expert_key(n), n)
                for n in hdr
                if f".experts.{e_str}." in n
            )
            concat = b"".join(
                raw[body + hdr[n]["data_offsets"][0] : body + hdr[n]["data_offsets"][1]]
                for _, n in names
            )
            assert span == concat
            spans = sorted(hdr[n]["data_offsets"] for _, n in names)
            for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
                assert b1 == a2, "expert tensors must be contiguous"
            assert spans[0][0] == lo and spans[-1][1] == hi


def test_attestation_signature_and_digests(workspace):
    from nacl.signing import VerifyKey

    snap, out, key = workspace
    run(snap, out, key)
    manifest = json.loads((out / "fq-manifest.json").read_text())
    vk = VerifyKey(bytes.fromhex(manifest["signer_pubkey"]))
    for layer in LAYERS:
        line = json.loads(
            (out / "attestations" / f"layer-{layer:03d}.k3.jsonl").read_text()
        )
        raw = base64.b64decode(line["payload"])
        vk.verify(raw, base64.b64decode(line["signature"]))
        payload = json.loads(raw)
        assert payload["materials"]["file_sha256"] == hashlib.sha256(
            (snap / payload["materials"]["file"]).read_bytes()
        ).hexdigest()
        seg = out / payload["fragment"]["file"]
        assert hashlib.sha256(seg.read_bytes()).hexdigest() == payload["fragment"]["sha256"]
        index = json.loads((out / "index-k3.json").read_text())[str(layer)]
        raw_seg = seg.read_bytes()
        for e_str, (lo, hi) in index["experts"].items():
            got = hashlib.sha256(
                raw_seg[index["body_offset"] + lo : index["body_offset"] + hi]
            ).hexdigest()
            assert got == payload["expert_sha256"][e_str]


def test_resume_skips_done_layers(workspace, capsys):
    snap, out, key = workspace
    run(snap, out, key)
    mtimes = {p.name: p.stat().st_mtime_ns for p in out.glob("*.safetensors")}
    run(snap, out, key)
    assert {p.name: p.stat().st_mtime_ns for p in out.glob("*.safetensors")} == mtimes
    assert "skip (done" in capsys.readouterr().out


def test_manifest_fields(workspace):
    snap, out, key = workspace
    run(snap, out, key)
    m = json.loads((out / "fq-manifest.json").read_text())
    assert m["schema"] == "fq-manifest/1"
    assert m["moe_layers"] == [min(LAYERS), max(LAYERS)]
    assert m["num_experts"] == E
    assert m["k_variants"] == [3]
    assert m["sources"] == ["test/source-repo"]
    assert m["predicate"] == "repack-of"
