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

E, RANKS = 4, 4
LAYERS = [3, 4]
PROJS = ["gate_proj", "up_proj", "down_proj"]
COMPS = ["trellis", "suh", "svh", "mcg"]


def tensor_bytes(layer, e, proj, rank, comp, k=3) -> bytes:
    """Synthetic tensor payload with the rank-sliced topology invariants."""
    seed = f"{layer}.{e}.{proj}.{rank}.{comp}.k{k}".encode()
    h = hashlib.sha256(seed).digest()
    size = 32 * k if comp == "trellis" else 32
    return (h * (size // len(h) + 1))[:size]


def write_shard(path: Path, layer: int, scramble: bool, k: int = 3,
                ranks: int = RANKS) -> None:
    """Synthetic shard; scramble=True stores tensors in non-logical order."""
    entries = []
    for e in range(E):
        for proj in PROJS:
            for rank in range(ranks):
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
            "shape": [1, 1, 16 * k] if name.endswith(".trellis") else [16],
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
    (snap / "config.json").write_text(json.dumps({
        "hybrid_tr3_tail": {"tp": 4, "codebook": "mcg"}}))
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


def test_repack_refuses_to_stamp_tp4_on_tp2_source(tmp_path):
    snap = tmp_path / "tp2-source"
    snap.mkdir()
    write_shard(snap / "model-layer-003.safetensors", LAYERS[0],
                scramble=False, ranks=2)
    (snap / "config.json").write_text(json.dumps({
        "hybrid_tr3_tail": {"tp": 2, "codebook": "mcg"}}))
    out = tmp_path / "segments"
    with pytest.raises(fq_repack.SourceLayoutError, match="TP1/TP2 cannot be repacked"):
        fq_repack.main([
            "--snapshot", str(snap), "--source-repo", "test/source-repo",
            "--revision", "deadbeef", "--base-model", "test/base",
            "--out", str(out), "--sign-key", str(tmp_path / "sign.key")])
    assert not out.exists()


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


# ------------------------------------------- FINDING 2: multi-K state/manifest

def write_snapshot(root: Path, k: int) -> Path:
    """A snapshot whose expert payloads (and trellis sizes) are K-specific."""
    root.mkdir(parents=True, exist_ok=True)
    for i, layer in enumerate(LAYERS):
        write_shard(root / f"model-layer-{layer:03d}.safetensors", layer,
                    scramble=bool(i), k=k)
    (root / "MANIFEST.sha256").write_text("\n".join(
        f"{hashlib.sha256((root / f'model-layer-{L:03d}.safetensors').read_bytes()).hexdigest()}"
        f"  model-layer-{L:03d}.safetensors" for L in LAYERS) + "\n")
    return root


def run_k(snap, out, key, k, *extra, repo="test/source-repo", rev="deadbeef",
          base="test/base"):
    return fq_repack.main([
        "--snapshot", str(snap), "--source-repo", repo, "--revision", rev,
        "--base-model", base, "--out", str(out), "--k", str(k),
        "--sign-key", str(key), *extra])


@pytest.fixture()
def multi_k(tmp_path):
    return ({k: write_snapshot(tmp_path / f"snap-k{k}", k) for k in (3, 4)},
            tmp_path / "segs", tmp_path / "sign.key")


def _assert_index_sane(out: Path, k: int, layers=LAYERS):
    """index-kK.json must name only .kK files whose bytes hash as recorded."""
    idx = json.loads((out / f"index-k{k}.json").read_text())
    assert sorted(int(L) for L in idx) == sorted(layers), (k, sorted(idx))
    for layer_s, entry in idx.items():
        assert entry["file"] == f"layer-{int(layer_s):03d}.k{k}.safetensors"
        seg = out / entry["file"]
        assert hashlib.sha256(seg.read_bytes()).hexdigest() == entry["sha256"]
        assert seg.stat().st_size == entry["size"]
        hdr, _ = fq_repack.read_header(seg)
        assert hdr["__metadata__"]["k"] == str(k)


def test_k3_k4_k3_sequence_keeps_per_k_indexes_correct(multi_k):
    """The exact K3 -> K4 -> K3 sequence into ONE output dir.

    v1 keyed state by layer alone, so the K4 run saw every layer as "done"
    and copied the K3 index entries into index-k4.json (and the K3 rerun then
    republished whatever K happened to be last).  Each K's index must name
    only that K's segments, and the manifest must list both Ks."""
    snaps, out, key = multi_k
    assert run_k(snaps[3], out, key, 3) == 0
    assert run_k(snaps[4], out, key, 4, repo="test/source-k4") == 0
    assert run_k(snaps[3], out, key, 3) == 0

    for k in (3, 4):
        _assert_index_sane(out, k)
        for layer in LAYERS:
            assert (out / f"layer-{layer:03d}.k{k}.safetensors").exists()
    # the two Ks really hold different bytes (K4 trellis rows are 4/3 the size)
    assert (out / "layer-003.k3.safetensors").stat().st_size != \
        (out / "layer-003.k4.safetensors").stat().st_size

    m = json.loads((out / "fq-manifest.json").read_text())
    assert m["k_variants"] == [3, 4]
    assert set(m["per_k"]) == {"3", "4"}
    for k in (3, 4):
        ent = m["per_k"][str(k)]
        assert ent["index"] == f"index-k{k}.json"
        assert ent["layers"] == [min(LAYERS), max(LAYERS)]
        assert ent["segment_count"] == len(LAYERS)
        assert ent["num_experts"] == E
        assert "repack-of" in ent["provenance"]
    assert m["per_k"]["4"]["source_repo"] == "test/source-k4"
    assert m["per_k"]["3"]["source_repo"] == "test/source-repo"
    assert sorted(m["sources"]) == ["test/source-k4", "test/source-repo"]


def test_resume_does_not_cross_contaminate_ks(multi_k):
    """A K3 rerun must not touch K4 files, entries or index."""
    snaps, out, key = multi_k
    run_k(snaps[3], out, key, 3)
    run_k(snaps[4], out, key, 4)
    k4_before = {p.name: (p.stat().st_mtime_ns, p.read_bytes())
                 for p in out.glob("*.k4.safetensors")}
    idx4_before = (out / "index-k4.json").read_text()
    att4_before = {p.name: p.read_text()
                   for p in (out / "attestations").glob("*.k4.jsonl")}

    assert run_k(snaps[3], out, key, 3) == 0
    assert {p.name: (p.stat().st_mtime_ns, p.read_bytes())
            for p in out.glob("*.k4.safetensors")} == k4_before
    assert (out / "index-k4.json").read_text() == idx4_before
    assert {p.name: p.read_text()
            for p in (out / "attestations").glob("*.k4.jsonl")} == att4_before
    _assert_index_sane(out, 3)
    _assert_index_sane(out, 4)


def test_partial_layer_runs_accumulate_one_index(multi_k):
    """index-kK.json is rebuilt from the K's whole state, not just this run."""
    snaps, out, key = multi_k
    assert run_k(snaps[3], out, key, 3, "--layers", str(LAYERS[0])) == 0
    assert json.loads((out / "index-k3.json").read_text()).keys() == {str(LAYERS[0])}
    assert run_k(snaps[3], out, key, 3, "--layers", str(LAYERS[1])) == 0
    _assert_index_sane(out, 3)


def test_state_is_keyed_by_source_k_layer(multi_k):
    snaps, out, key = multi_k
    run_k(snaps[3], out, key, 3)
    run_k(snaps[4], out, key, 4)
    state = json.loads((out / "state.json").read_text())
    assert state["schema"] == "fq-repack-state/2"
    fp = fq_repack.source_fingerprint(
        "test/source-repo", "deadbeef", "test/base", "rank_sliced_tp4")
    assert set(state["sources"]) == {fp}
    assert set(state["sources"][fp]["k"]) == {"3", "4"}
    for k in ("3", "4"):
        layers = state["sources"][fp]["k"][k]["layers"]
        assert sorted(int(L) for L in layers) == LAYERS
        for layer_s, entry in layers.items():
            assert entry["index"]["file"].endswith(f".k{k}.safetensors")
            assert entry["source_file_sha256"]
            assert entry["signer_pubkey"] and entry["tool_version"]


def test_v1_state_is_migrated_by_segment_name(multi_k):
    """A legacy layer-keyed state must not be read as if it were this K's."""
    snaps, out, key = multi_k
    run_k(snaps[3], out, key, 3)
    v2 = json.loads((out / "state.json").read_text())
    fp = next(iter(v2["sources"]))
    legacy = {"layers": {L: v2["sources"][fp]["k"]["3"]["layers"][L]
                         for L in v2["sources"][fp]["k"]["3"]["layers"]}}
    (out / "state.json").write_text(json.dumps(legacy))
    assert run_k(snaps[4], out, key, 4) == 0  # K4 must NOT skip on K3 entries
    _assert_index_sane(out, 3)
    _assert_index_sane(out, 4)


def test_refuses_foreign_base_model_or_layout(multi_k):
    snaps, out, key = multi_k
    run_k(snaps[3], out, key, 3)
    with pytest.raises(fq_repack.ProvenanceError, match="base_model"):
        run_k(snaps[3], out, key, 4, base="other/base")


def test_refuses_same_k_from_another_source(multi_k):
    snaps, out, key = multi_k
    run_k(snaps[3], out, key, 3)
    with pytest.raises(fq_repack.ProvenanceError, match="overwrites that K"):
        run_k(snaps[3], out, key, 3, repo="someone/else", rev="0ther")
    # explicit opt-in re-repacks the K and drops the superseded source entry
    assert run_k(snaps[3], out, key, 3, "--allow-provenance-change",
                 repo="someone/else", rev="0ther") == 0
    _assert_index_sane(out, 3)
    state = json.loads((out / "state.json").read_text())
    owners = [fp for fp, s in state["sources"].items()
              if s["k"].get("3", {}).get("layers")]
    assert len(owners) == 1
    m = json.loads((out / "fq-manifest.json").read_text())
    assert m["per_k"]["3"]["source_repo"] == "someone/else"


# --------------------------------- FINDING 6: source binding is computed here

def test_lying_source_manifest_is_fatal(workspace):
    snap, out, key = workspace
    mf = snap / "MANIFEST.sha256"
    good = mf.read_text().splitlines()
    mf.write_text("\n".join(["0" * 64 + "  " + good[0].split()[1]] + good[1:]) + "\n")
    with pytest.raises(fq_repack.ProvenanceError, match="MANIFEST.sha256"):
        run(snap, out, key)


def test_source_sha_is_computed_without_a_manifest(workspace):
    """No MANIFEST.sha256: the attestation still pins the real shard bytes."""
    snap, out, key = workspace
    (snap / "MANIFEST.sha256").unlink()
    assert run(snap, out, key) == 0
    for layer in LAYERS:
        line = json.loads(
            (out / "attestations" / f"layer-{layer:03d}.k3.jsonl").read_text())
        payload = json.loads(base64.b64decode(line["payload"]))
        mats = payload["materials"]
        assert mats["file_sha256"] == hashlib.sha256(
            (snap / mats["file"]).read_bytes()).hexdigest()
        assert mats["file_sha256_source"] == "computed"
        hdr, _ = fq_repack.read_header(out / payload["fragment"]["file"])
        assert hdr["__metadata__"]["source_file_sha256"] == mats["file_sha256"]


def test_resume_redoes_a_layer_whose_source_changed(workspace, capsys):
    snap, out, key = workspace
    run(snap, out, key)
    shard = snap / f"model-layer-{LAYERS[0]:03d}.safetensors"
    raw = bytearray(shard.read_bytes())
    raw[-1] ^= 0xFF
    shard.write_bytes(bytes(raw))
    lines = [f"{hashlib.sha256((snap / f'model-layer-{L:03d}.safetensors').read_bytes()).hexdigest()}"
             f"  model-layer-{L:03d}.safetensors" for L in LAYERS]
    (snap / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    assert run(snap, out, key) == 0
    assert "source shard changed" in capsys.readouterr().out
    _assert_index_sane(out, 3)


def test_recheck_detects_a_tampered_segment_on_resume(workspace, capsys):
    snap, out, key = workspace
    run(snap, out, key)
    seg = out / f"layer-{LAYERS[0]:03d}.k3.safetensors"
    raw = bytearray(seg.read_bytes())
    raw[-1] ^= 0xFF
    seg.write_bytes(bytes(raw))
    assert run(snap, out, key) == 0
    assert "skip (done" in capsys.readouterr().out  # size unchanged: cheap path
    assert fq_repack.main([
        "--snapshot", str(snap), "--source-repo", "test/source-repo",
        "--revision", "deadbeef", "--base-model", "test/base", "--out", str(out),
        "--sign-key", str(key), "--recheck"]) == 0
    assert "--recheck" in capsys.readouterr().out
    _assert_index_sane(out, 3)


def test_attestation_carries_a_usable_header_digest(workspace):
    """A consumer must be able to authenticate the tensor layout it plans
    from with ONE ranged read: the signed fragment.header_sha256 has to
    match sha256(prefix||header) of the published segment, and
    body_offset must point exactly past it."""
    import base64
    snap, out, key = workspace
    run(snap, out, key)
    for layer in LAYERS:
        line = json.loads(
            (out / "attestations" / f"layer-{layer:03d}.k3.jsonl").read_text())
        frag = json.loads(base64.b64decode(line["payload"]))["fragment"]
        raw = (out / frag["file"]).read_bytes()
        body = frag["body_offset"]
        assert hashlib.sha256(raw[:body]).hexdigest() == frag["header_sha256"]
        n = struct.unpack("<Q", raw[:8])[0]
        assert body == 8 + n, "body_offset must sit exactly past the header"
