"""Tests for fq_prime: K identification, range coalescing, both-family
emission from a shared_h_v1 source, expansion bit-exactness, K-filtered
per-expert priming, attestation verification, resume/idempotence, and the
transport spot-check — all against a synthetic mini-repo served from local
files by monkeypatching the HTTP fetch functions."""
import base64
import hashlib
import json
import struct
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import fq_prime  # noqa: E402
import fq_repack  # noqa: E402
import fq_trust  # noqa: E402

LAYERS = [3, 4]
NEXP, RANKS = 6, 2
PROJS = ["gate_proj", "up_proj", "down_proj"]
IN_TILES, OUT_TILES = 2, 1
# per-layer K maps: layer 3 mixed, layer 4 mixed differently
K_MAP = {3: {0: 3, 1: 3, 2: 4, 3: 3, 4: 4, 5: 3},
         4: {0: 4, 1: 3, 2: 3, 3: 4, 4: 4, 5: 3}}
REPO, REV = "test/mini-quant", "cafebabe"
REV2 = "f00dface"  # a second commit of the same repo


def tbytes(layer, owner, proj, rank, comp, k=3) -> bytes:
    """Deterministic synthetic payload; trellis scales with K like the real
    [in/16, out/16, 16*K] i16 tensors."""
    seed = f"{layer}.{owner}.{proj}.{rank}.{comp}.k{k}".encode()
    h = hashlib.sha256(seed).digest()
    n = {"trellis": IN_TILES * OUT_TILES * 16 * k * 2, "mcg": 4,
         "suh": 32, "svh": 32}[comp]
    return (h * (n // len(h) + 1))[:n]


def expert_tensors(layer, e, k, layout):
    """(name, bytes, dtype, shape) for one expert under the given layout."""
    out = []
    for proj in PROJS:
        for rank in range(RANKS):
            comps = (["trellis", "svh", "mcg"] if proj != "down_proj"
                     else ["trellis", "suh", "mcg"])
            if layout == "per_expert_v1":
                comps = ["trellis", "suh", "svh", "mcg"]
            for comp in comps:
                name = f"model.layers.{layer}.mlp.experts.{e}.{proj}.rank{rank}.{comp}"
                data = tbytes(layer, e, proj, rank, comp, k)
                if comp == "trellis":
                    shape = ([IN_TILES, OUT_TILES, 16 * k] if proj != "down_proj"
                             else [OUT_TILES, IN_TILES, 16 * k])
                    dtype = "I16"
                elif comp == "mcg":
                    shape, dtype = [], "I32"
                else:
                    shape, dtype = [len(data) // 2], "F16"
                out.append((name, data, dtype, shape))
    return out


def shared_tensors(layer):
    out = []
    for proj in PROJS:
        comp = "suh" if proj != "down_proj" else "svh"
        for rank in range(RANKS):
            name = f"model.layers.{layer}.mlp.experts.shared_h.{proj}.rank{rank}.{comp}"
            data = tbytes(layer, "shared", proj, rank, comp)
            out.append((name, data, "F16", [len(data) // 2]))
    return out


def write_shard(path: Path, layer: int, layout: str, pad: int = 0) -> None:
    """Mimic the real shards: non-expert tensor first, experts contiguous in
    LEXICOGRAPHIC id order with ALPHABETICAL within-expert tensor order,
    shared block (if any) last.  pad>0 prepends an extra dense tensor, which
    shifts every expert's byte range — how a second commit of the same repo
    differs from the first."""
    entries = [(f"model.layers.{layer}.self_attn.o_proj.weight",
                b"\x07" * 64, "BF16", [32])]
    if pad:
        entries.insert(0, (f"model.layers.{layer}.self_attn.qkv_pad.weight",
                           b"\x09" * pad, "BF16", [pad // 2]))
    for e in sorted(range(NEXP), key=str):
        unit = expert_tensors(layer, e, K_MAP[layer][e], layout)
        unit.sort(key=lambda t: t[0])
        entries.extend(unit)
    if layout == "shared_h_v1":
        entries.extend(shared_tensors(layer))
    hdr, off, blobs = {}, 0, []
    for name, data, dtype, shape in entries:
        hdr[name] = {"dtype": dtype, "shape": shape,
                     "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    hj = json.dumps({"__metadata__": {"format": "pt"}, **hdr},
                    separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)


def build_repo(root: Path, layout: str, pad: int = 0) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for layer in LAYERS:
        write_shard(root / f"model-layer-{layer:03d}.safetensors", layer, layout, pad)
    lines = [
        f"{hashlib.sha256((root / f'model-layer-{L:03d}.safetensors').read_bytes()).hexdigest()}"
        f"  model-layer-{L:03d}.safetensors" for L in LAYERS]
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    (root / "config.json").write_text(json.dumps({
        "hybrid_tr3_tail": {"rotation_layout": layout, "producer_version": "test",
                            "exllamav3_version": "0.0.43", "mcg_multiplier": 3417055213}}))
    (root / "tier_bitmap.json").write_text(json.dumps({
        str(L): {"k": [K_MAP[L][e] for e in range(NEXP)]} for L in LAYERS}))
    return root


@pytest.fixture()
def served(tmp_path, monkeypatch):
    """Serve tmp repos over monkeypatched HTTP; records calls and ranges."""
    calls = {"n": 0, "ranges": []}

    def local_path(url: str) -> Path:
        assert url.startswith("https://huggingface.co/")
        repo_rev_file = url.split("https://huggingface.co/", 1)[1]
        parts = repo_rev_file.split("/")
        assert parts[2] == "resolve"
        root = "repo" if parts[3] == REV else f"repo@{parts[3]}"
        return tmp_path / root / "/".join(parts[4:])

    def fake_range(url, start, end, timeout=0):
        calls["n"] += 1
        calls["ranges"].append((url.rsplit("/", 1)[1], start, end))
        data = local_path(url).read_bytes()
        assert 0 <= start <= end <= len(data), (start, end, len(data))
        return data[start:end], len(data)

    def fake_full(url, timeout=0):
        p = local_path(url)
        if not p.exists():
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)
        calls["n"] += 1
        return p.read_bytes()

    monkeypatch.setattr(fq_prime, "http_get_range", fake_range)
    monkeypatch.setattr(fq_prime, "http_get_full", fake_full)
    return tmp_path, calls


def prime(tmp_path, out_name="segments-t", extra=(), rev=REV, key="sign.key"):
    return fq_prime.main([
        "prime", "--repo", REPO, "--revision", rev, "--layers", "3-4",
        "--out", str(tmp_path / "primed" / out_name),
        "--base-model", "test/base", "--sign-key", str(tmp_path / key),
        "--pace", "0", *extra])


# ------------------------------------------------------------ pure helpers

def test_k_identification_and_layout():
    root = Path("/nonexistent")  # not touched: build header in memory
    tensors = {}
    off = 0
    for e in range(NEXP):
        for name, data, dtype, shape in expert_tensors(3, e, K_MAP[3][e], "shared_h_v1"):
            tensors[name] = {"dtype": dtype, "shape": shape,
                             "data_offsets": [off, off + len(data)]}
            off += len(data)
    for name, data, dtype, shape in shared_tensors(3):
        tensors[name] = {"dtype": dtype, "shape": shape,
                         "data_offsets": [off, off + len(data)]}
        off += len(data)
    experts, shared, k_of, layout = fq_prime.analyze_layer(tensors, 3)
    assert layout == "shared_h_v1"
    assert k_of == K_MAP[3]
    assert len(shared) == 3 * RANKS
    assert all(len(v) == 3 * RANKS * 3 for v in experts.values())  # 3 comps/slot
    # per_expert layout detected when no shared rows (48-tensor signature)
    t2, off = {}, 0
    for e in range(NEXP):
        for name, data, dtype, shape in expert_tensors(3, e, K_MAP[3][e], "per_expert_v1"):
            t2[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [off, off + len(data)]}
            off += len(data)
    *_, layout2 = fq_prime.analyze_layer(t2, 3)
    assert layout2 == "per_expert_v1"
    # signature violation raises
    broken = dict(t2)
    broken.pop(f"model.layers.3.mlp.experts.0.gate_proj.rank0.mcg")
    with pytest.raises(ValueError, match="signature"):
        fq_prime.analyze_layer(broken, 3)


def test_coalescing():
    class W:  # noqa: D401 — dummy writer
        def pwrite(self, off, data):
            pass

    w = W()
    pieces = [(0, 10, w, 0), (10, 20, w, 10), (30, 40, w, 20), (40, 50, w, 30)]
    chunks = fq_prime.coalesce_pieces(pieces, max_chunk=1000)
    assert [(c[0][0], c[-1][1]) for c in chunks] == [(0, 20), (30, 50)]
    # max_chunk splits adjacent runs
    chunks = fq_prime.coalesce_pieces(pieces, max_chunk=15)
    assert [(c[0][0], c[-1][1]) for c in chunks] == [(0, 10), (10, 20), (30, 40), (40, 50)]
    # overlap rejected
    with pytest.raises(AssertionError, match="overlap"):
        fq_prime.coalesce_pieces([(0, 10, w, 0), (5, 15, w, 0)], 100)


# ------------------------------------------------- shared_h source priming

def test_both_families_and_expansion_bit_exact(served):
    tmp_path, calls = served
    build_repo(tmp_path / "repo", "shared_h_v1")
    assert prime(tmp_path, "segments-sh", ["--expand"]) == 0
    out = tmp_path / "primed" / "segments-sh"
    sh, ex = out / "shared-h", out / "expanded"

    for layer in LAYERS:
        ks_here = sorted(set(K_MAP[layer].values()))
        for k in ks_here:
            assert (sh / f"layer-{layer:03d}.k{k}.safetensors").exists()
            assert (ex / f"layer-{layer:03d}.k{k}.safetensors").exists()
        assert (sh / f"layer-{layer:03d}.shared.safetensors").exists()

    # shared-h expert segment: verbatim bytes, canonical order, 36-per-expert sig
    hdr, body = fq_repack.read_header(sh / "layer-003.k3.safetensors")
    meta = hdr.pop("__metadata__")
    assert meta["layout"] == "shared_h_v1"
    assert meta["predicate"] == "repack-of"
    assert meta["shared_profile_sha256"]
    raw = (sh / "layer-003.k3.safetensors").read_bytes()[body:]
    for name, t in hdr.items():
        L, e, proj, rank, comp = fq_repack.EXPERT_RE.match(name).groups()
        assert K_MAP[3][int(e)] == 3
        a, b = t["data_offsets"]
        assert raw[a:b] == tbytes(int(L), int(e), proj, int(rank), comp, 3), name

    # profile fragment: the layer's shared rows verbatim
    phdr, pbody = fq_repack.read_header(sh / "layer-003.shared.safetensors")
    pmeta = phdr.pop("__metadata__")
    assert pmeta["kind"] == "shared_profile"
    praw = (sh / "layer-003.shared.safetensors").read_bytes()[pbody:]
    assert len(phdr) == 3 * RANKS
    for name, t in phdr.items():
        _, proj, rank, comp = fq_prime.SHARED_RE.match(name).groups()
        a, b = t["data_offsets"]
        assert praw[a:b] == tbytes(3, "shared", proj, int(rank), comp)

    # EXPANSION BIT-EXACTNESS: expanded unit == hand-built per_expert_v1 unit
    for layer in LAYERS:
        for k in sorted(set(K_MAP[layer].values())):
            seg = ex / f"layer-{layer:03d}.k{k}.safetensors"
            ehdr, ebody = fq_repack.read_header(seg)
            emeta = ehdr.pop("__metadata__")
            assert emeta["layout"] == "rank_sliced_tp4"
            assert emeta["predicate"] == "derived-from"
            assert emeta["derived_rule"] == fq_prime.DERIVED_RULE
            eraw = seg.read_bytes()[ebody:]
            eindex = json.loads((ex / f"index-k{k}.json").read_text())[str(layer)]
            for e in [e for e, kk in K_MAP[layer].items() if kk == k]:
                # hand-built: per_expert_v1 names in canonical order; shared
                # rows fill the {gate,up}.suh / down.svh slots
                hand = b""
                names = []
                for nm, data, _, _ in expert_tensors(layer, e, k, "per_expert_v1"):
                    names.append(nm)
                for nm in sorted(names, key=fq_repack.expert_key):
                    _, _, proj, rank, comp = fq_repack.EXPERT_RE.match(nm).groups()
                    is_shared_slot = (comp == "suh" and proj != "down_proj") or (
                        comp == "svh" and proj == "down_proj")
                    if is_shared_slot:
                        hand += tbytes(layer, "shared", proj, int(rank), comp)
                    else:
                        hand += tbytes(layer, e, proj, int(rank), comp, k)
                lo, hi = eindex["experts"][str(e)]
                assert eraw[lo:hi] == hand, f"layer {layer} k{k} expert {e}"
                # name/shape signature is exactly per_expert_v1
                got = sorted(n for n in ehdr if f".experts.{e}." in n)
                assert got == sorted(names)

    # family manifests + root multi-source manifest
    shm = json.loads((sh / "fq-manifest.json").read_text())
    assert shm["layout"] == "shared_h_v1" and shm["predicate"] == "repack-of"
    assert shm["dependencies"]["shared_profile_per_layer"]["3"]["sha256"]
    exm = json.loads((ex / "fq-manifest.json").read_text())
    assert exm["layout"] == "rank_sliced_tp4" and exm["predicate"] == "derived-from"
    assert exm["parent_family"] == "segments-sh/shared-h"
    root = json.loads((out.parent / "fq-manifest.json").read_text())
    assert {f["path"] for f in root["families"]} == {
        "segments-sh/shared-h", "segments-sh/expanded"}


def test_attestations_verify(served):
    from nacl.signing import VerifyKey

    tmp_path, calls = served
    repo = build_repo(tmp_path / "repo", "shared_h_v1")
    assert prime(tmp_path, "segments-sh", ["--expand"]) == 0
    out = tmp_path / "primed" / "segments-sh"
    sh, ex = out / "shared-h", out / "expanded"
    vk = VerifyKey(bytes.fromhex(
        json.loads((sh / "fq-manifest.json").read_text())["signer_pubkey"]))

    def verified(path):
        line = json.loads(path.read_text())
        raw = base64.b64decode(line["payload"])
        vk.verify(raw, base64.b64decode(line["signature"]))
        return json.loads(raw)

    for layer in LAYERS:
        for k in sorted(set(K_MAP[layer].values())):
            pl = verified(sh / "attestations" / f"layer-{layer:03d}.k{k}.jsonl")
            assert pl["predicate"] == "repack-of"
            assert pl["materials"]["file_sha256"] == hashlib.sha256(
                (repo / pl["materials"]["file"]).read_bytes()).hexdigest()
            seg = sh / pl["fragment"]["file"]
            assert hashlib.sha256(seg.read_bytes()).hexdigest() == pl["fragment"]["sha256"]
            # per-expert digests match segment spans and expert_k rides along
            idx = json.loads((sh / f"index-k{k}.json").read_text())[str(layer)]
            raw = seg.read_bytes()
            for e_s, (lo, hi) in idx["experts"].items():
                got = hashlib.sha256(
                    raw[idx["body_offset"] + lo:idx["body_offset"] + hi]).hexdigest()
                assert got == pl["expert_sha256"][e_s]
                assert pl["expert_k"][e_s] == k
            assert pl["source_config"]["mcg_multiplier"] == 3417055213
            # derived-from attestation links back to both parents by sha
            dl = verified(ex / "attestations" / f"layer-{layer:03d}.k{k}.jsonl")
            assert dl["predicate"] == "derived-from"
            roles = {p["role"]: p["sha256"] for p in dl["parents"]}
            assert roles["expert_segment"] == pl["fragment"]["sha256"]
            prof = verified(sh / "attestations" / f"layer-{layer:03d}.shared.jsonl")
            assert roles["shared_profile"] == prof["fragment"]["sha256"]
            assert dl["derivation"]["rule"] == fq_prime.DERIVED_RULE


def test_resume_idempotent(served):
    tmp_path, calls = served
    build_repo(tmp_path / "repo", "shared_h_v1")
    assert prime(tmp_path, "segments-sh", ["--expand"]) == 0
    out = tmp_path / "primed" / "segments-sh"
    files = sorted(out.rglob("*.safetensors"))
    mtimes = {p: p.stat().st_mtime_ns for p in files}
    n_before = calls["n"]
    assert prime(tmp_path, "segments-sh", ["--expand"]) == 0
    assert calls["n"] == n_before, "resume must make zero HTTP calls"
    assert {p: p.stat().st_mtime_ns for p in files} == mtimes


# ----------------------------------------------- per-expert source, K filter

def test_k_filter_per_expert_source(served):
    tmp_path, calls = served
    build_repo(tmp_path / "repo", "per_expert_v1")
    assert prime(tmp_path, "segments-pe", ["--k", "4"]) == 0
    out = tmp_path / "primed" / "segments-pe"
    # single family directly under out; only k4 files
    assert sorted(p.name for p in out.glob("*.safetensors")) == [
        "layer-003.k4.safetensors", "layer-004.k4.safetensors"]
    hdr, body = fq_repack.read_header(out / "layer-003.k4.safetensors")
    meta = hdr.pop("__metadata__")
    assert meta["layout"] == "rank_sliced_tp4" and meta["k"] == "4"
    raw = (out / "layer-003.k4.safetensors").read_bytes()[body:]
    k4_experts = sorted(e for e, k in K_MAP[3].items() if k == 4)
    assert sorted({int(fq_repack.EXPERT_RE.match(n).group(2)) for n in hdr}) == k4_experts
    for name, t in hdr.items():
        L, e, proj, rank, comp = fq_repack.EXPERT_RE.match(name).groups()
        a, b = t["data_offsets"]
        assert raw[a:b] == tbytes(int(L), int(e), proj, int(rank), comp, 4)

    # no K3 expert payload bytes were fetched: every payload range must avoid
    # K3 expert spans of the source shard
    src_hdr, src_body = fq_repack.read_header(
        tmp_path / "repo" / "model-layer-003.safetensors")
    src_hdr.pop("__metadata__")
    k3_spans = []
    for n, t in src_hdr.items():
        m = fq_repack.EXPERT_RE.match(n)
        if m and K_MAP[3][int(m.group(2))] == 3:
            a, b = t["data_offsets"]
            k3_spans.append((src_body + a, src_body + b))
    for fname, a, b in calls["ranges"]:
        if fname != "model-layer-003.safetensors" or a < src_body:
            continue  # header reads
        for sa, sb in k3_spans:
            assert not (a < sb and sa < b), f"fetched K3 bytes {a}-{b}"


# --------------------------------------------------------------- spot-check

def _signer_fp(family: Path) -> str:
    return json.loads((family / "fq-manifest.json").read_text())["signer_pubkey"]


def test_spot_check_pass_and_detect_corruption(served):
    tmp_path, calls = served
    build_repo(tmp_path / "repo", "shared_h_v1")
    assert prime(tmp_path, "segments-sh") == 0
    sh = tmp_path / "primed" / "segments-sh" / "shared-h"
    total = 2 * NEXP  # every (layer, expert)
    pin = ["--trust-signer", _signer_fp(sh)]
    assert fq_prime.main(["spot-check", "--dir", str(sh), "--n", str(total),
                          "--seed", "7", "--pace", "0", *pin]) == 0
    # corrupt one byte inside expert payload of one segment
    seg = sh / "layer-003.k3.safetensors"
    idx = json.loads((sh / "index-k3.json").read_text())["3"]
    raw = bytearray(seg.read_bytes())
    lo = idx["body_offset"] + idx["experts"]["0"][0]
    raw[lo] ^= 0xFF
    seg.write_bytes(bytes(raw))
    assert fq_prime.main(["spot-check", "--dir", str(sh), "--n", str(total),
                          "--seed", "7", "--pace", "0", *pin]) == 1


def test_spot_check_rejects_placeholder_signatures(served):
    """Decoding is not verifying: a family whose attestation signatures are
    replaced with padding must FAIL, not pass (TRUST.md 7)."""
    tmp_path, calls = served
    sh = _primed_family(tmp_path)
    for att in (sh / "attestations").glob("*.jsonl"):
        out = []
        for line in att.read_text().splitlines():
            env = json.loads(line)
            env["signature"] = base64.b64encode(b"\x00" * 64).decode()
            out.append(json.dumps(env, separators=(",", ":")))
        att.write_text("\n".join(out) + "\n")
    with pytest.raises(fq_trust.TrustError):
        _spot_check(sh)


# ------------------------------------ spot-check: absent evidence (P1-4b)

def _primed_family(tmp_path) -> Path:
    build_repo(tmp_path / "repo", "shared_h_v1")
    assert prime(tmp_path, "segments-sh") == 0
    return tmp_path / "primed" / "segments-sh" / "shared-h"


def _spot_check(family: Path, *extra, n=2, seed=7):
    return fq_prime.main(["spot-check", "--dir", str(family), "--n", str(n),
                          "--seed", str(seed), "--pace", "0",
                          "--trust-signer", _signer_fp(family), *extra])


def _resign(path: Path, key: Path, mutate) -> None:
    """Re-sign an attestation line after changing what it claims."""
    payload = json.loads(base64.b64decode(
        json.loads(path.read_text().splitlines()[0])["payload"]))
    mutate(payload)
    path.write_text(fq_repack.Signer(key).sign_line(payload) + "\n")


def test_spot_check_fails_when_the_attestation_is_missing(served):
    """A sampled expert with no signed digest has not been spot-checked."""
    tmp_path, calls = served
    sh = _primed_family(tmp_path)
    for att in (sh / "attestations").glob("layer-*.k*.jsonl"):
        att.unlink()
    with pytest.raises(fq_trust.TrustError, match="missing"):
        _spot_check(sh, n=2 * NEXP)


def test_spot_check_fails_when_the_expert_digest_is_absent(served):
    """A validly signed line that simply omits the sampled expert."""
    tmp_path, calls = served
    sh = _primed_family(tmp_path)
    key = tmp_path / "sign.key"
    for att in sorted((sh / "attestations").glob("layer-*.k*.jsonl")):
        _resign(att, key, lambda p: p.__setitem__("expert_sha256", {}))
    with pytest.raises(fq_trust.TrustError, match="unattested"):
        _spot_check(sh, n=2 * NEXP)


def test_spot_check_fails_on_an_empty_attestation_file(served):
    tmp_path, calls = served
    sh = _primed_family(tmp_path)
    for att in (sh / "attestations").glob("layer-*.k*.jsonl"):
        att.write_text("")
    with pytest.raises(fq_trust.TrustError, match="no trusted attestation"):
        _spot_check(sh, n=2 * NEXP)


def test_spot_check_reads_every_jsonl_line(served):
    """JSON Lines, not one JSON document: a leading line signed by a
    stranger must neither crash the tool nor hide the trusted line."""
    tmp_path, calls = served
    sh = _primed_family(tmp_path)
    stranger = fq_repack.Signer(tmp_path / "stranger.key")
    for att in sorted((sh / "attestations").glob("layer-*.k*.jsonl")):
        good = att.read_text().splitlines()[0]
        payload = json.loads(base64.b64decode(json.loads(good)["payload"]))
        att.write_text(stranger.sign_line(payload) + "\n" + good + "\n")
    assert _spot_check(sh, n=2 * NEXP) == 0


def test_spot_check_detects_a_tampered_expert_digest(served):
    """A re-signed line that claims the wrong digest for a real expert."""
    tmp_path, calls = served
    sh = _primed_family(tmp_path)
    key = tmp_path / "sign.key"
    for att in sorted((sh / "attestations").glob("layer-*.k*.jsonl")):
        _resign(att, key, lambda p: p["expert_sha256"].update(
            {e: "0" * 64 for e in p["expert_sha256"]}))
    assert _spot_check(sh, n=2 * NEXP) == 1


def test_spot_check_requires_a_pin_or_a_trust_root(served, tmp_path):
    """No pin, and a trust root that lists nobody: the family's own
    signer_pubkey is not authority."""
    srv_tmp, calls = served
    sh = _primed_family(srv_tmp)
    root = tmp_path / "root"
    root.mkdir()
    (root / "FINGERPRINTS").write_text("# nobody\n")
    with pytest.raises(fq_trust.TrustError):
        fq_prime.main(["spot-check", "--dir", str(sh), "--n", "1",
                       "--seed", "7", "--pace", "0",
                       "--trust-root", str(root)])


# ------------------- FINDING 6: caches and resume are bound to the source

def test_shard_header_cache_is_scoped_to_repo_at_revision(served):
    """Two commits of one repo must not share a header cache entry.

    Caching by bare file name served the FIRST commit's header for the
    second — and every byte range computed from it then points at the wrong
    offsets in the new shard."""
    tmp_path, calls = served
    build_repo(tmp_path / "repo", "per_expert_v1")
    build_repo(tmp_path / f"repo@{REV2}", "per_expert_v1", pad=512)
    cache = tmp_path / "source-meta"
    t = fq_prime.Transport(pace=0)
    h1, b1, size1, _ = fq_prime.Source(
        REPO, REV, "test/base", t, cache).shard_header(3)
    h2, b2, size2, _ = fq_prime.Source(
        REPO, REV2, "test/base", t, cache).shard_header(3)
    assert (b1, size1) != (b2, size2), "the two commits really do differ"
    assert h2 != h1, "revision 2 was served revision 1's cached header"
    name = "model.layers.3.mlp.experts.0.gate_proj.rank0.trellis"
    assert h2[name]["data_offsets"] != h1[name]["data_offsets"]
    assert sorted(p.name for p in cache.iterdir()) == [
        f"test__mini-quant@{REV}", f"test__mini-quant@{REV2}"]


def test_cached_header_for_another_revision_is_ignored(served):
    """Even inside one cache dir, an entry that names another revision is
    refetched rather than trusted (caches get copied and renamed)."""
    tmp_path, calls = served
    build_repo(tmp_path / "repo", "per_expert_v1")
    cache = tmp_path / "source-meta"
    src = fq_prime.Source(REPO, REV, "test/base", fq_prime.Transport(pace=0),
                          cache)
    good, _b, _s, _u = src.shard_header(3)
    entry = src.cache / "hdr-model-layer-003.safetensors.json"
    obj = json.loads(entry.read_text())
    obj["revision"] = "0000000"
    obj["header"] = {"bogus": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}
    entry.write_text(json.dumps(obj))
    again, _b, _s, _u = src.shard_header(3)
    assert again == good


def test_state_records_the_full_provenance_tuple(served):
    tmp_path, calls = served
    build_repo(tmp_path / "repo", "per_expert_v1")
    assert prime(tmp_path, "segments-prov", ["--k", "4"]) == 0
    state = json.loads(
        (tmp_path / "primed" / "segments-prov" / "state.json").read_text())
    assert state["schema"] == "fq-prime-state/2"
    prov = state["provenance"]
    assert prov["repo"] == REPO and prov["revision"] == REV
    assert prov["base_model"] == "test/base"
    assert prov["layout"] == "per_expert_v1"
    assert prov["tool_version"] == fq_prime.TOOL_VERSION
    assert len(prov["signer_pubkey"]) == 64
    for st in state["layers"].values():
        assert st["source"] == {"repo": REPO, "revision": REV}


def test_resume_refuses_a_changed_revision(served):
    tmp_path, calls = served
    build_repo(tmp_path / "repo", "per_expert_v1")
    build_repo(tmp_path / f"repo@{REV2}", "per_expert_v1", pad=512)
    assert prime(tmp_path, "segments-rev", ["--k", "4"]) == 0
    with pytest.raises(fq_prime.ProvenanceError, match="different provenance"):
        prime(tmp_path, "segments-rev", ["--k", "4"], rev=REV2)


def test_resume_refuses_a_changed_signer(served):
    tmp_path, calls = served
    build_repo(tmp_path / "repo", "per_expert_v1")
    assert prime(tmp_path, "segments-key", ["--k", "4"]) == 0
    with pytest.raises(fq_prime.ProvenanceError, match="signer_pubkey"):
        prime(tmp_path, "segments-key", ["--k", "4"], key="other.key")
    # opting in keeps the already-primed layers and records the old producer
    assert prime(tmp_path, "segments-key",
                 ["--k", "4", "--allow-provenance-change"], key="other.key") == 0
    state = json.loads(
        (tmp_path / "primed" / "segments-key" / "state.json").read_text())
    assert state["provenance_history"][0]["layers_discarded"] is False
    assert sorted(int(L) for L in state["layers"]) == LAYERS


def test_allowed_source_change_reprimes_from_the_new_revision(served):
    """The end-to-end shape of the bug: a new commit, one output dir."""
    tmp_path, calls = served
    build_repo(tmp_path / "repo", "per_expert_v1")
    assert prime(tmp_path, "segments-move", ["--k", "4"]) == 0
    build_repo(tmp_path / f"repo@{REV2}", "per_expert_v1", pad=512)
    assert prime(tmp_path, "segments-move",
                 ["--k", "4", "--allow-provenance-change"], rev=REV2) == 0

    out = tmp_path / "primed" / "segments-move"
    state = json.loads((out / "state.json").read_text())
    assert state["provenance"]["revision"] == REV2
    assert state["provenance_history"][0]["layers_discarded"] is True
    for st in state["layers"].values():
        assert st["source"]["revision"] == REV2
    # bytes come from the NEW shard layout, not from the old cached header
    for layer in LAYERS:
        seg = out / f"layer-{layer:03d}.k4.safetensors"
        hdr, body = fq_repack.read_header(seg)
        assert hdr.pop("__metadata__")["source_revision"] == REV2
        raw = seg.read_bytes()[body:]
        for name, t in hdr.items():
            L, e, proj, rank, comp = fq_repack.EXPERT_RE.match(name).groups()
            a, b = t["data_offsets"]
            assert raw[a:b] == tbytes(int(L), int(e), proj, int(rank), comp, 4), name
    man = json.loads((out / "fq-manifest.json").read_text())
    assert man["per_k"]["4"]["source_revision"] == REV2
    assert man["per_k"]["4"]["segment_count"] == len(LAYERS)
    assert man["per_k"]["4"]["index"] == "index-k4.json"


def test_dry_run_writes_nothing(served):
    tmp_path, calls = served
    build_repo(tmp_path / "repo", "shared_h_v1")
    assert prime(tmp_path, "segments-dry", ["--dry-run"]) == 0
    out = tmp_path / "primed" / "segments-dry"
    assert list(out.rglob("*.safetensors")) == []
    assert not (out / "state.json").exists()
